# Topic-Modelling-and-Sentiment-Analysis-for-Malaysian-Food-Tourism

This project leverages Natural Language Processing (NLP) techniques and machine learning models to analyze Google reviews of restaurants across various cities in Malaysia. The aim is to extract meaningful insights about food tourism trends, customer sentiments, and prevalent topics in the restaurant industry.

🗂️ Dataset

Source: Google Reviews of restaurants in Malaysia

Size: ~200,000 reviews

Features: Textual reviews, ratings, restaurant metadata

🔧 Key Features

1. Data Preprocessing with NLP

Tokenization – Breaking down reviews into individual words or tokens

Stop Word Removal – Filtering out common words to reduce noise

Stemming – Reducing words to their base or root form

Part-of-Speech (POS) Tagging – Labeling words with their grammatical role for better context understanding

2. Topic Modelling

LDA (Latent Dirichlet Allocation) is used to discover hidden topics within restaurant reviews

Provides insights into popular themes across different cities and cuisines

3. Sentiment Analysis

Three different approaches are implemented to classify the sentiment of reviews:

Naive Bayes Classifier – Probabilistic model based on word frequencies

Logistic Regression – Linear model for binary/multiclass sentiment classification

VADER (Valence Aware Dictionary and Sentiment Reasoner) – Rule-based model optimized for social media text
