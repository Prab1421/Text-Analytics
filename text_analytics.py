import os
import nltk
import string
import fitz
import docx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag

# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger")

def read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(file_path)
        text = " ".join([page.get_text() for page in doc])
    elif ext == ".docx":
        doc = docx.Document(file_path)
        text = " ".join([para.text for para in doc.paragraphs])
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    else:
        raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT.")
    return text

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    pos_tags = pos_tag(tokens)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens, pos_tags, stemmed_tokens, lemmatized_tokens

def compute_tfidf(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return df

def plot_tfidf(tfidf_df):
    word_tfidf = tfidf_df.sum(axis=0).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=word_tfidf.head(10).index, y=word_tfidf.head(10).values)
    plt.title('Top 10 Words with the Highest TF-IDF Scores')
    plt.xlabel('Words')
    plt.ylabel('TF-IDF Score')
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    file_path = input("Enter the path to your text document: ")
    try:
        text = read_file(file_path)
        tokens, pos_tags, stemmed, lemmatized = preprocess_text(text)
        print("\nTokenized Words:", tokens)
        print("\nPOS Tags:", pos_tags)
        print("\nStemmed Words:", stemmed)
        print("\nLemmatized Words:", lemmatized)
        
        tfidf_df = compute_tfidf([text])
        print("\nTF-IDF Representation:\n", tfidf_df)
        
        plot_tfidf(tfidf_df)
        
    except Exception as e:
        print("Error:", e)