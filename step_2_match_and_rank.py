import pandas as pd
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)

# Load cleaned resumes
df = pd.read_csv('output/cleaned_resumes.csv')

# Load & clean job description
with open('data/job_description.txt', 'r', encoding='utf-8') as file:
    job_description = clean_text(file.read())

# Combine job description and resumes for vectorization
texts = [job_description] + df['clean_resume'].tolist()

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# Cosine similarity
job_vector = tfidf_matrix[0]
resume_vectors = tfidf_matrix[1:]

similarities = cosine_similarity(job_vector, resume_vectors).flatten()
df['similarity'] = similarities

# Sort and save
df_sorted = df.sort_values(by='similarity', ascending=False)
print(df_sorted[['Category', 'similarity']].head())
df_sorted.to_csv('output/top_matches.csv', index=False)
print("✅ Top matches saved to output/top_matches.csv")
