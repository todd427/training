export DO_SAMPLE=0 TEMPERATURE=0.2 TOP_P=0.9 MAX_NEW=80 REPETITION_PENALTY=1.05 STOP_SENTINEL=END
python eval_bullets.py --file eval_bullets_sample.jsonl --load-in-4bit > eval_bullets_report.json

