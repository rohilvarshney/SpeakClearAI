from pathlib import Path
import csv
import cmudict

OUTPUT_PATH = Path("data/word_bank_expanded.csv")
OUTPUT_PATH.parent.mkdir(exist_ok=True)

TARGETS = {
    "s": "S",
    "z": "Z",
    "r": "R",
    "th": "TH",
}

MAX_WORDS_PER_SOUND = 100
MIN_LEN = 3
MAX_LEN = 10


def is_simple_word(word):
    return (
        word.isalpha()
        and MIN_LEN <= len(word) <= MAX_LEN
        and word == word.lower()
    )


def difficulty_for_word(word):
    length = len(word)

    if 3 <= length <= 5:
        return "easy"
    elif 6 <= length <= 7:
        return "medium"
    else:
        return "hard"


def main():
    entries = cmudict.entries()

    rows = []
    seen_by_sound = {sound: set() for sound in TARGETS}

    for word, pronunciation in entries:
        word = word.lower().strip()

        if not is_simple_word(word):
            continue

        for target_sound, phoneme in TARGETS.items():
            if len(seen_by_sound[target_sound]) >= MAX_WORDS_PER_SOUND:
                continue

            if phoneme in pronunciation and word not in seen_by_sound[target_sound]:
                seen_by_sound[target_sound].add(word)

                rows.append({
                    "word": word,
                    "target_sound": target_sound,
                    "source": "CMU Pronouncing Dictionary",
                    "difficulty": difficulty_for_word(word),
                    "notes": f"Matched CMU phoneme {phoneme}"
                })

    rows = sorted(rows, key=lambda x: (x["target_sound"], x["difficulty"], x["word"]))

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["word", "target_sound", "source", "difficulty", "notes"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {OUTPUT_PATH}")
    print(f"Total words: {len(rows)}")

    for sound in TARGETS:
        count = sum(1 for row in rows if row["target_sound"] == sound)
        print(f"{sound}: {count} words")


if __name__ == "__main__":
    main()
