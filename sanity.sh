#!/usr/bin/env bash
set -euo pipefail
set -x

# Conservative decoding for instruction fidelity
export DO_SAMPLE=0
export TEMPERATURE=0.2
export TOP_P=0.9
export MAX_NEW=80
export REPETITION_PENALTY=1.05
# Optional: stop sentinel if your prompt includes it
# export STOP_SENTINEL="END"
# Optional: enforce exact bullets in post-process
export ENFORCE_BULLETS=1
export BULLETS=3

PROMPT=$'You must output EXACTLY three bullet points and nothing else.\nNo preamble. No follow-up. No extra text.\nFormat:\n- <point 1>\n- <point 2>\n- <point 3>\nEND\n\nSummarize this email in 3 bullets:\n\nDear Todd\nI hope you\'re doing well.  Lectures commence on Monday, September 15th 2025.'

export DO_SAMPLE=0 TEMPERATURE=0.2 TOP_P=0.9 MAX_NEW=80 REPETITION_PENALTY=1.05 STOP_SENTINEL=END
python sanity_test_fixed.py --no-lora --load-in-4bit --prompt "$PROMPT" --stop END
python sanity_test_fixed.py --load-in-4bit --prompt "$PROMPT" --stop END

# Base (no LoRA)
#python sanity_test_fixed.py --no-lora --load-in-4bit \
#  --prompt "$PROMPT" --stop END

# Trained (with LoRA)
#python sanity_test_fixed.py --load-in-4bit \
#  --prompt "$PROMPT" --stop END
