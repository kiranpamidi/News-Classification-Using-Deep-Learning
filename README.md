# News Classification Using Deep Learning

This project is an NLP-based system that classifies news articles as either **Fake** or **Real** using Deep Learning techniques. It leverages text preprocessing, TF-IDF/token embedding, and deep neural networks to detect misinformation.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Deep Learning Model](#deep-learning-model)
- [How to Run](#how-to-run)
- [Manual Testing](#manual-testing)
- [Results](#results)
- [License](#license)

---

## Overview

With the rise of misinformation, detecting fake news has become critical. This project builds a deep learning model that processes news content and classifies it as **fake** or **real**.

---

## Dataset

- **Source**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Files**:
  - `Fake.csv`: Fake news articles
  - `True.csv`: Real news articles

After combining both, each news article is labeled:
- `0` for Fake
- `1` for Real

---

## Technologies Used

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- TensorFlow / Keras
- Scikit-learn
- NLP (Text Cleaning, Tokenization, TF-IDF/Embeddings)

---

## Deep Learning Model

A sample architecture used in this project (optional based on your final setup):

```python
Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=500),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])
