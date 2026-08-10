from pathlib import Path
import csv
import cmudict
import string

OUTPUT_PATH = Path("data/alphabet_word_bank.csv")
OUTPUT_PATH.parent.mkdir(exist_ok=True)

WORDS_PER_LETTER = 100
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

    seen_words = set()
    rows_by_letter = {letter: [] for letter in string.ascii_lowercase}

    for word, pronunciation in entries:
        word = word.lower().strip()

        if not is_simple_word(word):
            continue

        if word in seen_words:
            continue

        first_letter = word[0]

        if first_letter not in rows_by_letter:
            continue

        if len(rows_by_letter[first_letter]) >= WORDS_PER_LETTER:
            continue

        seen_words.add(word)

        rows_by_letter[first_letter].append({
            "word": word,
            "letter": first_letter.upper(),
            "source": "CMU Pronouncing Dictionary",
            "difficulty": difficulty_for_word(word),
            "notes": f"Alphabet practice word starting with {first_letter.upper()}"
        })

    rows = []
    for letter in string.ascii_lowercase:
        rows.extend(rows_by_letter[letter])

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["word", "letter", "source", "difficulty", "notes"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {OUTPUT_PATH}")
    print(f"Total words: {len(rows)}")

    for letter in string.ascii_lowercase:
        print(f"{letter.upper()}: {len(rows_by_letter[letter])} words")


if __name__ == "__main__":
    main()
