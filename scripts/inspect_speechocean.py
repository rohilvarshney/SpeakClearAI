from datasets import load_dataset, Audio

print("Loading a small preview of SpeechOcean762 without decoding audio...")
ds = load_dataset("mispeech/speechocean762", split="train[:3]")
ds = ds.cast_column("audio", Audio(decode=False))

print("\nColumns:")
print(ds.column_names)

for i in range(len(ds)):
    ex = ds[i]

    print("\n" + "=" * 80)
    print(f"Example {i}")
    print("Text:", ex.get("text"))
    print("Speaker:", ex.get("speaker"))
    print("Gender:", ex.get("gender"))
    print("Age:", ex.get("age"))

    audio = ex.get("audio")
    if audio:
        print("Audio keys:", audio.keys())
        print("Audio path:", audio.get("path"))
        print("Audio bytes exists:", audio.get("bytes") is not None)

    print("\nFirst few word annotations:")
    for word in ex.get("words", [])[:5]:
        print(word)
