from datasets import load_dataset
import pandas as pd
import os

LANGUAGES = {
    "hau": "Hausa",
    "yor": "Yoruba",
    "ibo": "Igbo"
}

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


def load_afrisenti(lang_code):
 
    print(f"\nLoading {LANGUAGES[lang_code]} ({lang_code}) dataset...")

    dataset = load_dataset(
        "shmuhammad/AfriSenti-twitter-sentiment",
        lang_code,
        trust_remote_code=True
    )

    splits = {}
    for split_name in ["train", "validation", "test"]:
        df = pd.DataFrame(dataset[split_name])
        df["label_name"] = df["label"].map(LABEL_NAMES)
        splits[split_name] = df
        print(f"  {split_name}: {len(df)} samples")

    return splits


def load_all_languages():
    """
    Loads all three language subsets and returns them as a dictionary.
    Structure: { lang_code: { "train": df, "validation": df, "test": df } }
    """
    all_data = {}
    for lang_code in LANGUAGES:
        all_data[lang_code] = load_afrisenti(lang_code)
    return all_data


def explore_dataset(all_data):
    """
    Prints a summary of each language subset including class distributions.
    This helps us understand the data before we start training.
    """
    print("\n" + "="*60)
    print("DATASET EXPLORATION SUMMARY")
    print("="*60)

    for lang_code, splits in all_data.items():
        lang_name = LANGUAGES[lang_code]
        train_df = splits["train"]

        print(f"\n--- {lang_name} ({lang_code}) ---")
        print(f"Total training samples: {len(train_df)}")
        print(f"Total validation samples: {len(splits['validation'])}")
        print(f"Total test samples: {len(splits['test'])}")

        print(f"\nClass distribution (train):")
        dist = train_df["label_name"].value_counts()
        total = len(train_df)
        for label, count in dist.items():
            pct = (count / total) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")


def save_to_csv(all_data):
    """
    Saves all splits to CSV files inside the data/ folder.
    This means we do not need to re-download from HuggingFace every time.
    """
    os.makedirs("data", exist_ok=True)

    for lang_code, splits in all_data.items():
        for split_name, df in splits.items():
            filename = f"data/{lang_code}_{split_name}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved: {filename}")

    print("\nAll datasets saved to data/ folder.")


if __name__ == "__main__":
    all_data = load_all_languages()
    explore_dataset(all_data)
    save_to_csv(all_data)