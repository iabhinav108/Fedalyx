# scripts/check_partitions.py
import json
from pathlib import Path
from collections import Counter

p = Path("outputs/partitions/partitions.json")
data = json.loads(p.read_text())

print("Global teacher counts:")
for cls, files in data["global_teacher"].items():
    print(f"  {cls}: {len(files)}")

print("\nClient counts:")
for cid, clsmap in data["clients"].items():
    total = sum(len(v) for v in clsmap.values())
    print(f"  client_{cid}: {total}")
    for cls, files in clsmap.items():
        print(f"    {cls}: {len(files)}")
