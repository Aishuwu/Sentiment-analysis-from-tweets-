# Sentiment Analysis from Tweets using Support Vector Machines

## Introduction

This project implements a sentiment classifier that analyzes tweets and determines whether their sentiment is positive or negative. The classification is performed using a Support Vector Machine (SVM), with features extracted from preprocessed tweet text. The project explores various stages of preprocessing, feature extraction, cross-validation, error analysis, and performance optimization.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Advanced Pipeline](#advanced-pipeline)
- [Evaluation](#evaluation)
- [Results](#results)
- [Improvements](#improvements)
- [Dependencies](#dependencies)
- [License](#license)

## Features

- Tweet classification into **positive** or **negative** sentiment.
- Text preprocessing with tokenization, punctuation handling, and lemmatization.
- Feature extraction with binary and term frequency representations, including bigrams.
- Model training using `LinearSVC` with pipeline support.
- Performance evaluation via accuracy, precision, recall, and F1-score.
- Confusion matrix visualization and error logging.
- Extended cross-validation with dynamic folds and statistical averaging.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Aishuwu/sentiment-analysis-tweets.git
   cd sentiment-analysis-tweets
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Jupyter Notebooks in the following order:
   - `Sentiment_analysis_from_tweets.ipynb`
   - `Sentiment_analysis_from_tweets_advanced.ipynb`

2. Customize preprocessing and feature extraction strategies in the advanced notebook.

3. Use the final trained classifier for inference and evaluation.

## Dataset

- Dataset consists of tweets labeled with sentiment (`positive` or `negative`).
- Each sample includes a label and tweet text (tab-separated).
- Preprocessing includes:
  - Lowercasing
  - Punctuation separation
  - Tokenization
  - Stopword removal
  - Lemmatization

## Methodology

1. **Preprocessing**:
   - Clean text and tokenize using regex and NLTK tools.
   - Remove stopwords, lowercase tokens, and apply lemmatization.

2. **Feature Extraction**:
   - Create binary feature vectors from tokens.
   - Optionally use term frequency representation.
   - Include stylistic features such as words per sentence and bigrams.

3. **Model Training**:
   - Train a linear SVM classifier using scikit-learn’s `LinearSVC`.
   - Use pipeline integration for clean training and inference.

4. **Cross-validation**:
   - Implement 10-fold cross-validation.
   - Report accuracy, precision, recall, and F1-score per fold and average.

5. **Error Analysis**:
   - Generate confusion matrix heatmaps.
   - Identify and log false positives and false negatives to text files.

## Advanced Pipeline

- Extended preprocessing and feature tuning:
  - Dynamic choice between binary/TF weights.
  - Option to include n-grams and stylistic indicators.
  - Cross-validation with variable fold sizes and repetitions for robustness.

## Evaluation

### Metrics Used:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Basic Pipeline Results:
- Precision: 0.8517  
- Recall: 0.8530  
- F1-Score: 0.8521  
- Accuracy: 85.30%

### Advanced Pipeline Results:
- Precision: 0.8602  
- Recall: 0.8617  
- F1-Score: 0.8601  
- Accuracy: 86.17%

## Results

- The advanced pipeline consistently outperformed the basic version due to improved text normalization and richer feature sets.
- Confusion matrix visualizations helped interpret classifier performance across classes.
- Detailed logs of misclassifications provide insights into model behavior and room for further refinement.

## Improvements

Future enhancements may include:
- Hyperparameter tuning with grid search or Bayesian optimization.
- Incorporating deep learning models (e.g., LSTM, BERT) for semantic understanding.
- Leveraging word embeddings (e.g., Word2Vec, GloVe) for richer feature representations.
- Real-time deployment of the classifier via a REST API or web app.

## Dependencies

```
scikit-learn
nltk
numpy
matplotlib
seaborn
tqdm
```

Install them with:
```bash
pip install -r requirements.txt
```

## License

This project is licensed for academic and research purposes only.
