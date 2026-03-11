"""
Verify all three inference modes produce identical outputs (greedy decoding).

Usage:
  python verify_export.py
"""

import os
import subprocess
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))

MODES = ["eager", "export_eager", "exported"]
PROMPT = "A"
NUM_TOKENS = 15
# greedy: temperature near 0, top_k=1
COMMON_ARGS = [
    "--prompt", PROMPT,
    "--num_tokens", str(NUM_TOKENS),
    "--temperature", "0.001",
    "--top_k", "1",
    "--device", "cpu",
]


def run_mode(mode):
    cmd = [sys.executable, os.path.join(_DIR, "inference.py"), "--mode", mode] + COMMON_ARGS
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_DIR)
    if result.returncode != 0:
        print(f"  [{mode}] FAILED:\n{result.stderr[-500:]}")
        return None
    # extract text between === lines
    lines = result.stdout.strip().split("\n")
    inside = False
    text_lines = []
    for line in lines:
        if line.startswith("===="):
            if inside:
                break
            inside = True
            continue
        if inside:
            text_lines.append(line)
    return "\n".join(text_lines)


def main():
    results = {}
    for mode in MODES:
        print(f"Running {mode}...")
        text = run_mode(mode)
        if text is None:
            return
        results[mode] = text
        print(f"  Output: {repr(text)}")

    print()
    ref = results[MODES[0]]
    all_match = True
    for mode in MODES[1:]:
        match = results[mode] == ref
        status = "MATCH" if match else "MISMATCH"
        print(f"  {MODES[0]} vs {mode}: {status}")
        if not match:
            all_match = False

    print()
    if all_match:
        print("VERIFICATION PASSED — all modes produce identical output")
    else:
        print("VERIFICATION FAILED — outputs differ")


if __name__ == "__main__":
    main()
