# 🎬 Movie Review Sentiment Analysis | NLP & Machine Learning

### Classifying IMDb Movie Reviews Using Natural Language Processing and Supervised Learning

**Python • Pandas • Scikit-learn • NLTK • spaCy • TF-IDF • Logistic Regression • LightGBM**

## 📌 Project Overview

Understanding customer opinions is an important application of Natural Language Processing (NLP). Automated sentiment classification helps businesses analyze large volumes of written feedback and identify positive or negative opinions.

This project was developed for **Film Junky Union**, a fictional community for classic movie enthusiasts.

The goal is to build a machine learning system capable of classifying IMDb movie reviews as **positive or negative**, using text preprocessing, feature extraction, and supervised learning algorithms.

The project's target performance is an **F1-score of at least 0.85** on the test dataset.

## 🎯 Project Objectives

- Explore and preprocess IMDb movie review data.
- Examine the distribution of positive and negative reviews.
- Normalize text by removing punctuation and other unnecessary characters.
- Convert text into numerical features using TF-IDF.
- Compare a baseline classifier with machine learning models.
- Investigate the effect of lemmatization using spaCy.
- Evaluate models using classification metrics.
- Test trained classifiers on additional movie reviews.
- Identify a suitable approach for automated sentiment classification.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data processing and model development |
| Pandas | Dataset exploration and manipulation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Exploratory data visualization |
| NLTK | Stopword processing |
| spaCy | Text lemmatization |
| Scikit-learn | TF-IDF, classification, and evaluation |
| LightGBM | Gradient boosting classification |
| Transformers / PyTorch | BERT experimentation |
| Jupyter Notebook | Analysis and documentation |

## 📂 Dataset Description

The project uses `imdb_reviews.tsv`, containing IMDb movie reviews and associated metadata.

Relevant variables include:

| Column | Description |
|---|---|
| `review` | Original movie review text |
| `pos` | Sentiment label: 0 = negative, 1 = positive |
| `ds_part` | Training or test partition |
| `rating` | Movie rating |
| `tconst` | Movie identifier |
| `start_year` | Movie release year |

The dataset contains approximately 47,000 reviews, divided into training and test subsets.

### Sentiment Distribution

The classes are approximately balanced:

- **Negative reviews:** 50.10%
- **Positive reviews:** 49.90%

This distribution provides a useful starting point for binary classification.

## 🔎 Exploratory Data Analysis

The exploratory analysis examined:

- The distribution of positive and negative reviews.
- The number of movies and reviews across different years.
- The distribution of review counts per movie.
- Movie ratings in training and test datasets.
- Sentiment distribution over time.

These investigations helped establish the structure and characteristics of the dataset before model training.

## 🧹 Text Preprocessing

Text normalization was performed to make reviews more consistent for machine learning.

The preprocessing workflow included:

1. Converting text to lowercase.
2. Removing digits and punctuation while retaining apostrophes.
3. Replacing unnecessary characters with spaces.
4. Normalizing repeated whitespace.
5. Applying stopword filtering during TF-IDF vectorization.
6. Experimenting with lemmatization using spaCy.

### TF-IDF Feature Extraction

Term Frequency–Inverse Document Frequency (TF-IDF) was used to represent reviews numerically.

TF-IDF assigns weights to words based on their frequency within a document and their relative importance across the corpus.

The vectorizer was fitted on training data and then applied to the test data to avoid learning vocabulary statistics from the test set.

## 🤖 Machine Learning Models

### Model 0 — Dummy Classifier

A `DummyClassifier` using the most-frequent-class strategy was used as a baseline.

This model does not learn meaningful relationships between review text and sentiment.

It provides a reference for evaluating more advanced classifiers.

### Model 1 — NLTK + TF-IDF + Logistic Regression

This model combines:

- NLTK English stopwords
- TF-IDF text vectorization
- Logistic Regression

The model achieved an **F1-score of 0.88 on the test set**, exceeding the project's required minimum of 0.85.

### Model 3 — spaCy + TF-IDF + Logistic Regression

This approach introduces spaCy lemmatization before TF-IDF feature extraction.

The model achieved a test F1-score of **0.88**, matching Model 1.

The additional lemmatization step did not improve test performance in the recorded experiment.

### Model 4 — spaCy + TF-IDF + LightGBM

This model combines lemmatized text features with a gradient boosting classifier.

It achieved a test F1-score of **0.84**, below the project's required threshold.

### Model 9 — BERT

The notebook also includes an experimental BERT-based feature extraction workflow using Hugging Face Transformers and PyTorch.

However, the BERT experiment was not completed successfully due to memory and kernel limitations.

**BERT is therefore not included in the validated model comparison.**

## 📊 Model Comparison

| Model | Test Accuracy | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|
| Dummy Classifier | 0.50 | 0.67 | 0.50 |
| **TF-IDF + Logistic Regression** | **0.88** | **0.88** | **0.95** |
| spaCy + TF-IDF + Logistic Regression | 0.88 | 0.88 | 0.95 |
| spaCy + TF-IDF + LightGBM | 0.84 | 0.84 | 0.92 |

*Metrics are based on the results documented in the notebook.*

### Evaluation Metrics

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1-score
- Average Precision Score (APS)
- ROC-AUC
- Confusion Matrix

The main success criterion is test F1-score.

## 🏆 Best Model

**TF-IDF + Logistic Regression is the recommended approach for this project.**

It achieved:

- **Test Accuracy:** 0.88
- **Test F1-score:** 0.88
- **Test ROC-AUC:** 0.95

The model met the performance requirement while avoiding the additional computational complexity of spaCy lemmatization.

Although Model 3 achieved equivalent test performance, it required an additional text processing stage without a measurable improvement in the reported results.

## 📝 Testing on Custom Reviews

The notebook includes an additional section for testing trained models on example movie reviews.

The workflow:

1. Normalizes the review text.
2. Applies the appropriate preprocessing pipeline.
3. Transforms text using the fitted TF-IDF vectorizer.
4. Generates positive sentiment probabilities.
5. Compares predictions from different trained models.

This section illustrates how the classification workflow can be applied to previously unseen text.

## 💡 Business Applications

**Automated Review Classification**

Classify large volumes of movie reviews as positive or negative.

**Audience Sentiment Monitoring**

Identify changes in audience sentiment across collections of reviews.

**Content Discovery**

Explore how sentiment predictions could contribute to movie discovery and categorization systems.

**Feedback Analytics**

Support analysis of written customer opinions without manually reviewing every submission.

These are potential applications; the notebook does not evaluate a deployed production system.

## ⚠️ Limitations

- The dataset contains English-language IMDb reviews.
- Model performance may differ on reviews from other platforms or domains.
- TF-IDF does not directly capture contextual meaning in the same way as transformer-based representations.
- The BERT experiment was not completed because of computational limitations.
- The reported metrics reflect the notebook's existing evaluation and are not evidence of production performance.
- Additional validation would be needed to assess robustness, inference latency, and model behavior on different writing styles.

## 🚀 How to Run the Project

1. Clone or download the repository.
2. Install the required dependencies.
3. Download the NLTK stopwords corpus.
4. Install the spaCy English language model.
5. Place `imdb_reviews.tsv` inside a local `data/` directory.
6. Update the dataset path in the notebook.
7. Open the notebook in Jupyter Notebook or JupyterLab.
8. Execute the supported model sections sequentially.

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk spacy lightgbm tqdm notebook
python -m spacy download en_core_web_sm
```

### Download NLTK Stopwords

```python
import nltk
nltk.download("stopwords")
```

The optional BERT experiment requires additional packages:

```bash
pip install torch transformers
```

BERT may require substantial memory and processing time. The BERT section is experimental and may need further implementation work before it can run successfully.

## 🔮 Future Improvements

- Refactor preprocessing and modeling into reusable Scikit-learn pipelines.
- Add systematic hyperparameter optimization.
- Investigate misclassified reviews and model errors.
- Evaluate performance on additional external review datasets.
- Complete and benchmark the BERT experiment.
- Compare model
