import csv
from pathlib import Path
import subprocess
import sys
from collections import Counter

metadata_path = Path("metadata.csv")

with open(metadata_path, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

def save():
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

for row in rows:
    if row["label"].strip():
        continue

    file_path = Path("wav") / row["file_name"]

    print("\n-------------------------")
    print(f"File: {row['file_name']}")
    print(f"Word: {row['word']}")
    print(f"Target sound: /{row['target_sound']}/")

    subprocess.run(["afplay", str(file_path)])

    while True:
        label = input("Type c=clear, u=unclear, r=replay, q=quit: ").strip().lower()

        if label == "r":
            subprocess.run(["afplay", str(file_path)])
        elif label == "c":
            row["label"] = "clear"
            save()
            break
        elif label == "u":
            row["label"] = "unclear"
            save()
            break
        elif label == "q":
            save()
            print("Saved progress. You can run this again later.")
            sys.exit()
        else:
            print("Type c, u, r, or q.")

save()

counts = Counter(row["label"] for row in rows)
print("\nDone. metadata.csv is labeled.")
print(counts)
