"""Stage 5: A-output Gaussian noise over HTTPS, chat, and controlled evaluations."""

import sys

from settings import STAGE, parser, select, load_cases, project_path
from https_core.runtime import ClientRuntime
from fedsea.runtime import make_messages
from execution import run_trial
from noise import derive_seed
from reporting import metadata, new_run_dir, write_json


def answer(runner, messages, case_id):
    inputs = runner.tokenize(messages)
    seed = derive_seed(runner.stage["noise"]["seed"], case_id)
    return run_trial(runner, inputs, runner.stage["noise"], seed, trace=False)


def chat(runner):
    messages = [{"role": "system", "content": runner.config["system_prompt"]}]
    turn = 0
    print(f"Gaussian A-output noise: sigma={runner.stage['noise']['sigma']}, base seed={runner.stage['noise']['seed']}.")
    print("Fresh caches each turn; within-turn KV cache ON. No transcripts saved.")
    print("/reset clears conversation and resets the experimental seed sequence; /exit quits.")
    while True:
        try:
            prompt = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt == "/exit":
            break
        if prompt == "/reset":
            messages, turn = messages[:1], 0
            print("Conversation cleared.")
            continue
        if not prompt:
            continue
        candidate = messages + [{"role": "user", "content": prompt}]
        try:
            result = answer(runner, candidate, f"chat-turn-{turn}")
        except (ValueError, RuntimeError, OSError) as error:
            print(f"Request failed: {error}\nCaches cleared; conversation unchanged.")
            continue
        except KeyboardInterrupt:
            print("\nGeneration cancelled; caches cleared.")
            continue
        print(f"\nAssistant (HTTPS + noise)> {result['response']}")
        print(
            f"[new tokens={len(result['token_ids'])}, "
            f"effective noise RMS={result['noise']['effective_noise_rms']:.6f}, "
            f"generation={result['metrics']['generation_seconds']:.2f}s]"
        )
        messages = candidate + [{"role": "assistant", "content": result["response"]}]
        turn += 1


def single_prompt(runner, prompt):
    folder = new_run_dir(STAGE)
    messages = make_messages(prompt, runner.config["system_prompt"])
    record = metadata(runner, "prompt")
    record.update(messages=messages, full_model_loaded_for_reference=False, zero_noise_gate=None)
    write_json(folder / "run.json", record)
    try:
        result = answer(runner, messages, "custom_prompt")
        record.update({key: value for key, value in result.items() if key != "logits"})
        record["status"] = "completed"
        print(f"Assistant (HTTPS + noise)> {result['response']}")
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(folder / "run.json", record)
        print(f"Saved prompt/answer: {folder / 'run.json'}", flush=True)


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    selected = select(args)
    if selected is None:
        return
    stage, runtime, profile, part = selected
    cases = load_cases(project_path(stage["evaluation"]["prompts_file"])) if args.evaluate else None
    runner = ClientRuntime(
        stage, runtime, profile, part, boundary_label="A-output Gaussian noise configured per session",
    )
    if args.evaluate:
        from evaluation import evaluate
        evaluate(runner, cases)
    elif args.prompt is not None:
        single_prompt(runner, args.prompt)
    else:
        chat(runner)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; noisy sessions closed.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
