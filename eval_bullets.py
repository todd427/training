#!/usr/bin/env python3
import argparse, json, subprocess, shlex, tempfile, os, sys, pathlib, re

HERE = pathlib.Path(__file__).resolve().parent
SANITY = HERE / "sanity_test_fixed.py"

def run_cmd(cmd, env=None):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout, p.stderr

def extract_reply(stdout: str) -> str:
    # Grab everything after the marker
    if "=== Assistant ===" in stdout:
        return stdout.split("=== Assistant ===",1)[1].strip()
    return stdout.strip()

def count_bullets(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.strip().startswith(("-", "•", "*")))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="jsonl with fields: prompt, bullets (int), stop (optional)")
    ap.add_argument("--stop", default="END")
    ap.add_argument("--bullets", type=int, default=3)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    data = []
    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    env_base = os.environ.copy()
    env_base.setdefault("DO_SAMPLE","0")
    env_base.setdefault("TEMPERATURE","0.2")
    env_base.setdefault("TOP_P","0.9")
    env_base.setdefault("MAX_NEW","80")
    env_base.setdefault("REPETITION_PENALTY","1.05")
    env_base.setdefault("STOP_SENTINEL", args.stop)

    results = {"base": {"total":0,"pass":0, "details":[]},
               "lora": {"total":0,"pass":0, "details":[]},
               "config":{"stop":args.stop,"bullets":args.bullets, "load_in_4bit":args.load_in_4bit}}

    flag_4bit = " --load-in-4bit" if args.load_in_4bit else ""

    for row in data:
        prompt = row.get("prompt")
        want = int(row.get("bullets", args.bullets))
        stop = row.get("stop", args.stop)

        # Base
        cmd_base = f'python "{SANITY}" --no-lora{flag_4bit} --prompt {shlex.quote(prompt)} --stop {shlex.quote(stop)}'
        rc, out, err = run_cmd(cmd_base, env=env_base)
        reply = extract_reply(out)
        got = count_bullets(reply)
        ok = (got == want)
        results["base"]["total"] += 1
        results["base"]["pass"] += int(ok)
        results["base"]["details"].append({"ok":ok,"got":got,"want":want,"reply":reply})

        # LoRA
        cmd_lora = f'python "{SANITY}"{flag_4bit} --prompt {shlex.quote(prompt)} --stop {shlex.quote(stop)}'
        rc, out, err = run_cmd(cmd_lora, env=env_base)
        reply = extract_reply(out)
        got = count_bullets(reply)
        ok = (got == want)
        results["lora"]["total"] += 1
        results["lora"]["pass"] += int(ok)
        results["lora"]["details"].append({"ok":ok,"got":got,"want":want,"reply":reply})

    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
