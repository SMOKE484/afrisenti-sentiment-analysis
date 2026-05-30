# AfriSenti Sentiment Analysis
### Natural Language Processing — Group 36
**University of Pretoria**

Vhulenda Mashamba | Phetho Nemavhola | Thabelo Mulaudzi
---

## Overview

This project fine-tunes three multilingual transformer models for sentiment classification on three West African languages from the AfriSenti Twitter dataset: Hausa (hau), Yoruba (yor), and Igbo (ibo). Each tweet is labelled as positive, negative, or neutral by native speakers.

The project addresses three research questions:

- **RQ1:** Which multilingual model (mBERT, AfriBERTa, AfroXLMR) performs best for Hausa, Yoruba, and Igbo?
- **RQ2:** How does data augmentation (back-translation and synonym replacement) affect model performance on these low-resource subsets?
- **RQ3:** What misclassification and bias patterns emerge from error analysis across the three languages?

---

## Repository Structure

```
afrisenti-sentiment-analysis/
│
├── data/
│   ├── hau_train.csv               # Raw Hausa splits
│   ├── hau_validation.csv
│   ├── hau_test.csv
│   ├── yor_train.csv               # Raw Yoruba splits
│   ├── yor_validation.csv
│   ├── yor_test.csv
│   ├── ibo_train.csv               # Raw Igbo splits
│   ├── ibo_validation.csv
│   ├── ibo_test.csv
│   └── processed/                  # Cleaned CSVs after preprocessing
│
├── src/
│   ├── preprocessing/
│   │   ├── load_data.py            # Loads AfriSenti data from HuggingFace
│   │   └── preprocess.py           # Cleans tweets (URLs, mentions, RT markers, etc.)
│   └── baseline/
│       └── baseline.py             # TF-IDF + Logistic Regression baseline
│
├── notebooks/
│   └── afrisentifinetuning.ipynb   # Fine-tuning, augmentation, and error analysis
│
├── results/
│   ├── models/                     # Saved baseline models (.pkl)
│   ├── plots/                      # Transformer confusion matrices and LIME images
│   ├── hau_baseline_confusion_matrix.png
│   ├── yor_baseline_confusion_matrix.png
│   ├── ibo_baseline_confusion_matrix.png
│   ├── finetuning_results.csv      # Test F1 and accuracy for all 9 experiments
│   └── augmentation_results.csv    # Augmentation experiment results
│
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset used is the [AfriSenti Twitter Sentiment dataset](https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment) available on HuggingFace.

| Language | Code | Train  | Validation | Test  |
|----------|------|--------|------------|-------|
| Hausa    | hau  | 14,172 | 2,677      | 5,303 |
| Yoruba   | yor  | 8,522  | 2,090      | 4,515 |
| Igbo     | ibo  | 10,192 | 1,841      | 3,682 |

Each tweet is labelled as **positive**, **negative**, or **neutral**.

---

## Setup

### Requirements

- Python 3.11
- CUDA-compatible GPU recommended for transformer fine-tuning (experiments were run on Kaggle with dual T4 GPUs)

Key library versions (full list in `requirements.txt`):

| Package | Version |
|---------|---------|
| torch | >=2.0.0 |
| transformers | >=4.35.0 |
| datasets | >=2.14.0,<4.0.0 |
| scikit-learn | >=1.3.0 |
| pandas | >=2.0.0 |
| numpy | >=1.24.0 |
| accelerate | >=0.24.0 |
| evaluate | >=0.4.0 |
| lime | >=0.2.0 |
| nltk | >=3.8.0 |

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Afrisent-Sentiment-Analysis/afrisenti-sentiment-analysis.git
cd afrisenti-sentiment-analysis
pip install -r requirements.txt
```

---

## Running the Experiments

### 1. Preprocessing

> **Note on data:** The raw CSV files are not included in the zip submission due to their size. Run the script below to download them directly from HuggingFace before running any other step.

```bash
python src/preprocessing/load_data.py
python src/preprocessing/preprocess.py
```

`load_data.py` downloads the AfriSenti dataset from [HuggingFace](https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment) and saves the raw splits to `data/`. `preprocess.py` then cleans the tweets (removes URLs, @mentions, RT markers, the hash symbol, and digits) and saves the output to `data/processed/`. African language diacritics and emojis are intentionally preserved.

### 2. Baseline Model

```bash
python src/baseline/baseline.py
```

Trains a TF-IDF (character n-grams, range 2-5) + balanced Logistic Regression classifier per language. Saved models are stored in `results/models/`.

**Baseline Results:**

| Language | Test F1 | Test Acc |
|----------|---------|----------|
| Hausa    | 0.7694  | 0.7692   |
| Yoruba   | 0.7338  | 0.7298   |
| Igbo     | 0.7934  | 0.7933   |

### 3. Transformer Fine-tuning

Open and run `notebooks/afrisentifinetuning.ipynb` on Kaggle (or any GPU environment).

Training configuration:
- Epochs: 5
- Batch size: 16
- Learning rate: 2e-5
- Warmup ratio: 0.1
- Weight decay: 0.01
- Mixed precision: fp16
- Early stopping patience: 2
- Primary metric: weighted F1

**Fine-tuning Results:**

| Model      | Language | Test F1 | vs Baseline |
|------------|----------|---------|-------------|
| mBERT      | Hausa    | 0.7337  | -0.0357     |
| mBERT      | Yoruba   | 0.6719  | -0.0619     |
| mBERT      | Igbo     | 0.7618  | -0.0316     |
| AfriBERTa  | Hausa    | 0.7953  | +0.0259     |
| AfriBERTa  | Yoruba   | 0.7274  | -0.0064     |
| AfriBERTa  | Igbo     | 0.7840  | -0.0094     |
| AfroXLMR   | Hausa    | 0.7782  | +0.0088     |
| AfroXLMR   | Yoruba   | 0.6977  | -0.0361     |
| AfroXLMR   | Igbo     | 0.7637  | -0.0297     |

**AfriBERTa is the best performing model overall.** mBERT underperforms the TF-IDF baseline across all three languages.

### 4. Data Augmentation

Open and run `notebooks/afrisentifinetuning.ipynb`.

Back-translation (Helsinki-NLP MarianMT) was applied to Hausa only, as Helsinki-NLP models for Yoruba and Igbo are not available on HuggingFace. Synonym replacement (NLTK WordNet) was applied to all three languages.

**Augmentation Results (AfriBERTa):**

| Language | Original F1 | Augmented F1 | Change  |
|----------|-------------|--------------|---------|
| Hausa    | 0.7953      | 0.7683       | -0.0270 |
| Yoruba   | 0.7274      | 0.7223       | -0.0051 |
| Igbo     | 0.7840      | 0.7872       | +0.0032 |

### 5. Error Analysis

Open and run `notebooks/afrisentifinetuning.ipynb`.

This generates confusion matrices and LIME explanations for AfriBERTa predictions. Output plots are saved to `results/plots/`.

---

## Models Used

| Model      | HuggingFace Identifier             |
|------------|------------------------------------|
| mBERT      | `bert-base-multilingual-cased`     |
| AfriBERTa  | `castorini/afriberta_large`        |
| AfroXLMR   | `Davlan/afro-xlmr-base`            |

---

## Key Findings

- AfriBERTa achieved the best overall performance, with its strongest result on Hausa (F1: 0.7953).
- mBERT consistently underperformed the TF-IDF baseline, suggesting it lacks sufficient African language representation in its pre-training data.
- Data augmentation did not improve performance. It either had no meaningful effect or slightly hurt results across all three languages.
- The neutral class was the most frequently misclassified across all languages, which is consistent with findings from the AfriSenti-SemEval shared task.
- LIME analysis on Hausa revealed that named entities (e.g. politician names), religious terms, and code-switched English words contribute to misclassification in inconsistent ways.

---

## References

- Muhammad, S. H., et al. (2023). AfriSenti: A Twitter sentiment analysis benchmark for African languages. In *Proceedings of EMNLP 2023*.
- Alabi, J. O., Adelani, D. I., Mosbach, M., and Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. In *Proceedings of COLING 2022*.
- Ogueji, K., Zhu, Y., and Lin, J. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resourced languages. In *Proceedings of the 1st Workshop on Multilingual Representation Learning*.
- Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019*.
- Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. In *Proceedings of ACL 2020*.
