from pathlib import Path
import csv
from tqdm import tqdm
from datasets import load_dataset, Audio

OUTPUT_CSV = Path("data/speechocean_targets.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

TARGET_PHONES = {
    "s": ["S"],
    "z": ["Z"],
    "r": ["R"],
    "th": ["TH"],  # voiceless th only
}

CLEAR_THRESHOLD = 1.8
UNCLEAR_THRESHOLD = 1.0


def label_from_score(score):
    if score >= CLEAR_THRESHOLD:
        return "clear"
    if score < UNCLEAR_THRESHOLD:
        return "unclear"
    return None


def clean_phone(phone):
    return "".join(ch for ch in str(phone) if not ch.isdigit()).upper()


def get_word_text(word_info):
    for key in ["text", "word", "alignedWord"]:
        if key in word_info and word_info[key] is not None:
            return str(word_info[key]).lower()
    return ""


def get_phone_list(word_info):
    for key in ["phones", "phonemes"]:
        if key in word_info and word_info[key] is not None:
            return word_info[key]
    return []


def get_phone_scores(word_info):
    for key in ["phones-accuracy", "phones_accuracy", "phone_scores", "accuracy"]:
        if key in word_info and word_info[key] is not None and isinstance(word_info[key], list):
            return word_info[key]
    return []


def process_split(split_name):
    print(f"Loading SpeechOcean762 split without decoding audio: {split_name}")
    ds = load_dataset("mispeech/speechocean762", split=split_name)
    ds = ds.cast_column("audio", Audio(decode=False))

    rows = []

    for idx in tqdm(range(len(ds)), desc=f"Processing {split_name}"):
        ex = ds[idx]

        speaker_id = str(ex.get("speaker", "unknown"))
        sentence = str(ex.get("text", ""))
        audio_info = ex.get("audio") or {}
        audio_path = audio_info.get("path", "")

        for word_info in ex.get("words", []):
            word_text = get_word_text(word_info)
            phones = get_phone_list(word_info)
            phone_scores = get_phone_scores(word_info)

            for phone_index, raw_phone in enumerate(phones):
                phone = clean_phone(raw_phone)

                if phone_index >= len(phone_scores):
                    continue

                try:
                    phone_score = float(phone_scores[phone_index])
                except Exception:
                    continue

                label = label_from_score(phone_score)
                if label is None:
                    continue

                for target_sound, target_phones in TARGET_PHONES.items():
                    if phone in target_phones:
                        rows.append({
                            "source": "speechocean762",
                            "split": split_name,
                            "original_index": idx,
                            "audio_path": audio_path,
                            "speaker_id": f"speechocean_{speaker_id}",
                            "gender": ex.get("gender", ""),
                            "age": ex.get("age", ""),
                            "sentence": sentence,
                            "word": word_text,
                            "target_sound": target_sound,
                            "phone": phone,
                            "phone_score": phone_score,
                            "label": label,
                        })

    return rows


def main():
    all_rows = []
    for split in ["train", "test"]:
        all_rows.extend(process_split(split))

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "split",
                "original_index",
                "audio_path",
                "speaker_id",
                "gender",
                "age",
                "sentence",
                "word",
                "target_sound",
                "phone",
                "phone_score",
                "label",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nCreated:", OUTPUT_CSV)
    print("Total target-phone examples:", len(all_rows))

    for sound in TARGET_PHONES:
        sound_rows = [r for r in all_rows if r["target_sound"] == sound]
        clear_count = sum(1 for r in sound_rows if r["label"] == "clear")
        unclear_count = sum(1 for r in sound_rows if r["label"] == "unclear")
        print(f"{sound}: {len(sound_rows)} total | clear={clear_count} | unclear={unclear_count}")


if __name__ == "__main__":
    main()
