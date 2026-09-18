"""Stage 1: complete-model offline chat and reference measurements."""

import argparse
import json
import sys
import time
import uuid
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent     # FedSEA 文件夹
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedsea.model_registry import ModelRegistry
from fedsea.project import load_runtime_config, read_json
from fedsea.runtime import ModelRuntime, make_messages, validate_config, validate_length


class Baseline(ModelRuntime):
    def generate(self, messages, artifact_dir=None):
        from transformers import StoppingCriteria, StoppingCriteriaList

        inputs = self.tokenize(messages)        # tokenize 后输入
        input_length = inputs["input_ids"].shape[-1]
        runner = self

        # 获得首字延迟
        class FirstTokenTimer(StoppingCriteria):
            first_token_seconds = None

            def __call__(self, input_ids, scores, **kwargs):
                if self.first_token_seconds is None:
                    runner.synchronize()
                    self.first_token_seconds = time.perf_counter() - start
                return False

        timer = FirstTokenTimer()
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            with self.torch.inference_mode():
                first_logits = self.model(**inputs, use_cache=False).logits[0, -1].float().cpu()
            np.save(artifact_dir / "first_token_logits.npy", first_logits.numpy(), allow_pickle=False)
            np.save(artifact_dir / "input_ids.npy", inputs["input_ids"].cpu().numpy(), allow_pickle=False)

        self.synchronize()
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()     # 开始计时
        with self.torch.inference_mode():
            output = self.model.generate(   # generate 生成
                **inputs, generation_config=self.generation_config,
                stopping_criteria=StoppingCriteriaList([timer]),
            )
        self.synchronize()
        seconds = time.perf_counter() - start   # 获得总耗时

        # generate() 返回“输入 + 新生成内容”，截去输入后才是本轮回答。
        generated = output[0, input_length:].cpu()
        text = self.tokenizer.decode(generated, skip_special_tokens=True)       # decode 后输出
        # 记录相关信息
        if artifact_dir is not None:
            np.save(artifact_dir / "generated_ids.npy", generated.numpy(), allow_pickle=False)
        metrics = {
            "input_tokens": input_length,
            "generated_tokens_including_eos": len(generated),
            "generation_seconds": seconds,
            "first_token_seconds": timer.first_token_seconds,
            "tokens_per_second_including_prefill": len(generated) / seconds,
            "peak_gpu_allocated_mib": (
                self.torch.cuda.max_memory_allocated(self.device) / 1024**2
                if self.device.type == "cuda" else None
            ),
            "peak_gpu_reserved_mib": (
                self.torch.cuda.max_memory_reserved(self.device) / 1024**2
                if self.device.type == "cuda" else None
            ),
            "generated_token_ids": generated.tolist(),
        }
        return text, metrics


# 创建保存结果文件夹
def new_run_dir():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = STAGE / "runs" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True)
    return path


def save_run(path, runner, results, warmup):
    record = {
        "stage": "1_model_base", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": False, "noise_std": 0.0, "network_inference": False, "warmup": warmup,
        "model": runner.spec, "model_profile": runner.profile.describe(),
        "config": runner.config, "environment": runner.environment, "results": results,
    }
    (path / "run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved prompt/output record: {path / 'run.json'}", flush=True)


# 输出记录
def print_metrics(metrics):
    print(
        f"[{metrics['generated_tokens_including_eos']} tokens, "
        f"{metrics['generation_seconds']:.2f}s, TTFT {metrics['first_token_seconds']:.3f}s, "
        f"{metrics['tokens_per_second_including_prefill']:.2f} tokens/s]"
    )


# 聊天设置
def chat(runner):
    print("Local chat. /reset clears history; /exit quits. Transcripts are not saved.")
    messages = [{"role": "system", "content": runner.config["system_prompt"]}]
    while True:
        try:
            prompt = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if prompt == "/exit":
            return
        if prompt == "/reset":
            messages = messages[:1]
            print("History cleared.")
            continue
        if not prompt:
            continue
        
        candidate = messages + [{"role": "user", "content": prompt}]
        try:
            response, metrics = runner.generate(candidate)  # 获得回复
        except ValueError as error:
            print(f"Input rejected: {error}")
            continue
        print(f"\nAssistant> {response}")
        print_metrics(metrics)
        messages = candidate + [{"role": "assistant", "content": response}]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prompt", help="Single prompt; saves prompt, response, and metrics.")
    modes.add_argument("--benchmark", action="store_true", help="Run the fixed smoke-test prompts.")
    modes.add_argument("--chat", action="store_true", help="Chat (default); no saved transcript.")
    modes.add_argument("--list-models", action="store_true", help="List profiles; no weights are loaded.")
    modes.add_argument("--describe", action="store_true", help="Inspect a profile without loading weights.")
    parser.add_argument("--model", help="Profile name from configs/model.json; default is inherited.")
    parser.add_argument("--config", type=Path, default=STAGE / "config.json")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--save-artifacts", action="store_true", help="Save token IDs and first-token logits for later comparison.")

    args = parser.parse_args(argv)
    registry = ModelRegistry()
    if args.list_models:
        registry.list_models()
        return
    stage_config = read_json(args.config)
    profile = registry.get(args.model or stage_config.get("model"))
    if args.describe:
        print(json.dumps(profile.describe(), indent=2, ensure_ascii=False))
        return
    if args.save_artifacts and not (args.benchmark or args.prompt is not None):
        parser.error("--save-artifacts requires --prompt or --benchmark.")
    config = load_runtime_config(stage_config)
    if args.device is not None:
        config["device"] = args.device
    if args.max_new_tokens is not None:
        config["max_new_tokens"] = args.max_new_tokens
    validate_config(config)
    if args.prompt is not None:
        make_messages(args.prompt, config["system_prompt"])
    runner = Baseline(config, profile.name)
    if args.prompt is None and not args.benchmark:
        chat(runner)
        return
    path = new_run_dir()
    if args.benchmark:
        print("Running an unmeasured warmup...", flush=True)
        runner.generate(make_messages("Reply with OK.", config["system_prompt"]))
        cases = read_json(STAGE / "prompts.json")
    else:
        cases = [{"id": "prompt", "prompt": args.prompt}]
    results = []
    for index, case in enumerate(cases):
        print(f"\n[{case['id']}] {case['prompt']}", flush=True)
        messages = make_messages(case["prompt"], config["system_prompt"])
        artifact_dir = path / f"case_{index:02d}" if args.save_artifacts else None
        text, metrics = runner.generate(messages, artifact_dir)
        print(text, flush=True)
        print_metrics(metrics)
        result = {"id": case["id"], "messages": messages, "response": text, "metrics": metrics}
        if "expected_substring" in case:
            result["smoke_check_passed"] = case["expected_substring"] in text
        results.append(result)
        save_run(path, runner, results, warmup=args.benchmark)
    if any(not row["response"].strip() for row in results):
        raise RuntimeError("At least one test generated an empty response.")
    if any(row.get("smoke_check_passed") is False for row in results):
        raise RuntimeError("At least one basic instruction/arithmetic smoke check failed.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
