# gtrs_filter.py
import json

THRESH = 0.0

with open("gtrs_scores.json", "r") as f:
    gtrs = json.load(f)

selected = {k:v for k,v in gtrs.items() if v["score"] > THRESH}

with open("gtrs_filtered.json", "w") as f:
    json.dump(selected, f, indent=2)

print(f"Selected {len(selected)} high-confidence samples")
