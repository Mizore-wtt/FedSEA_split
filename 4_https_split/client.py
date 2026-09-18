"""Stage 4 client: local A -> HTTPS B -> local C. Chat is not saved."""

import json
import sys
from datetime import datetime, timezone
import uuid

from https_core.settings import parser, select, STAGE
from https_core.runtime import ClientRuntime
from https_core.session import HttpsSession
from fedsea.runtime import make_messages


def answer(runner, session, messages):
    inputs = runner.tokenize(messages)
    output, metrics = session.generate(inputs, runner.generation_config, synchronize=runner.synchronize)
    response = runner.tokenizer.decode(
        output.sequences[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    del output
    return response, metrics


def chat(runner):
    session = HttpsSession(runner.model)
    history = [{"role": "system", "content": runner.config["system_prompt"]}]
    print("HTTPS split chat. Transcripts not saved. /reset clears A/B/C; /exit quits.")
    try:
        while True:
            try:
                prompt = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt == "/exit":
                break
            if prompt == "/reset":
                history = history[:1]
                session.reset()
                print("Conversation and A/B/C caches cleared.")
                continue
            if not prompt:
                continue
            candidate = history + [{"role": "user", "content": prompt}]
            try:
                response, metrics = answer(runner, session, candidate)
            except (ValueError, RuntimeError, OSError) as error:
                session.reset()
                print(f"Request failed: {error}\nCache cleared; conversation text remains local.")
                continue
            except KeyboardInterrupt:
                session.reset()
                print("\nGeneration cancelled; cache cleared.")
                continue
            print(f"\nAssistant (HTTPS)> {response}")
            print(f"[new={metrics['generated_tokens_including_eos']}, "
                  f"reused prefix={metrics['reused_prefix_tokens']}, cached={metrics['cached_tokens']}]")
            history = candidate + [{"role": "assistant", "content": response}]
    finally:
        session.close()


def main(argv=None):
    argparser = parser(__doc__)
    modes = argparser.add_mutually_exclusive_group()
    modes.add_argument("--chat", action="store_true")
    modes.add_argument("--prompt")
    modes.add_argument("--compare", action="store_true", help="Also loads a full reference model, saves verification.")
    args = argparser.parse_args(argv)
    selected = select(args)
    if selected is None:
        return
    runner = ClientRuntime(*selected)
    if args.compare:
        from verify import compare
        compare(runner)
    elif args.prompt is not None:
        session = HttpsSession(runner.model)
        try:
            messages = make_messages(args.prompt, runner.config["system_prompt"])
            response, metrics = answer(runner, session, messages)
            print(f"Assistant (HTTPS)> {response}")
            folder = STAGE / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
            folder.mkdir(parents=True)
            record = {
                "stage": "4_https_split", "mode": "prompt", "status": "completed",
                "identity": runner.identity, "messages": messages, "response": response,
                "metrics": metrics, "communication": runner.rpc.stats, "noise_std": 0.0,
            }
            (folder / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Saved prompt/response: {folder / 'run.json'}")
        finally:
            session.close()
    else:
        chat(runner)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Client interrupted; remote leftovers expire by TTL.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
