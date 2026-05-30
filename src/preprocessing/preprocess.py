import pandas as pd
import re
import os

# Function to clean individual tweets by removing unwanted text patterns

def clean_tweet(text):
    """
    Cleans a single tweet for African language sentiment analysis.

    We are careful not to remove diacritics or special characters that
    are part of African languages like Yoruba tonal marks.
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # Remove the # symbol but keep the word (hashtags often carry sentiment)
    text = re.sub(r'#', '', text)

    # Remove RT (retweet indicator)
    text = re.sub(r'\bRT\b', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Applies tweet cleaning to an entire dataset split
def preprocess_split(df):
    """
    Applies cleaning to the tweet column of a DataFrame.
    Also removes any rows where the tweet is empty after cleaning.
    """
    df = df.copy()
    df["cleaned_text"] = df["tweet"].apply(clean_tweet)

    # Track how many rows we drop
    before = len(df)
    df = df[df["cleaned_text"].str.strip() != ""]
    after = len(df)

    if before != after:
        print(f"  Dropped {before - after} empty rows after cleaning")

    return df

# Processes all dataset splits for a specific language
def preprocess_language(lang_code):
    """
    Loads the saved CSVs for a language, applies preprocessing,
    and saves the cleaned versions to data/processed/.
    """
    os.makedirs("data/processed", exist_ok=True)

    splits = ["train", "validation", "test"]
    processed = {}

    for split in splits:
        input_path = f"data/{lang_code}_{split}.csv"

        if not os.path.exists(input_path):
            print(f"File not found: {input_path}. Run load_data.py first.")
            return None

        df = pd.read_csv(input_path)
        print(f"  Processing {split}: {len(df)} samples")

        df = preprocess_split(df)

        output_path = f"data/processed/{lang_code}_{split}_clean.csv"
        df.to_csv(output_path, index=False)

        processed[split] = df

    return processed

# Displays sample tweets before and after cleaning for verification
def show_cleaning_examples(lang_code, n=5):
    """
    Prints a few examples of original vs cleaned tweets so we can
    visually confirm the cleaning is working correctly.
    """
    # Load training data for inspection
    df = pd.read_csv(f"data/{lang_code}_train.csv")
    df["cleaned_text"] = df["tweet"].apply(clean_tweet)

    print(f"\nCleaning examples for {lang_code}:")
    print("-" * 60)
  # Display a few examples to verify preprocessing quality
    for _, row in df.head(n).iterrows():
        print(f"Original : {row['tweet']}")
        print(f"Cleaned  : {row['cleaned_text']}")
        print(f"Label    : {row['label_name']}")
        print()


if __name__ == "__main__":
    LANGUAGES = {
        "hau": "Hausa",
        "yor": "Yoruba",
        "ibo": "Igbo"
    }

    print("Starting preprocessing for all languages...\n")

    for lang_code, lang_name in LANGUAGES.items():
        print(f"\n{'='*50}")
        print(f"Processing {lang_name} ({lang_code})")
        print('='*50)
        preprocess_language(lang_code)
        show_cleaning_examples(lang_code, n=3)

    print("\nPreprocessing complete. Cleaned files saved to data/processed/")
