import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load cleaned resumes
df = pd.read_csv('output/cleaned_resumes.csv')

# Get job description input (you can automate this later)
job_description = """
Looking for a Data Scientist skilled in Python, Machine Learning, Data Analysis,
NLP, and model deployment. Experience with pandas, scikit-learn, and SQL required.
"""

# Combine job description with resumes
documents = df['clean_resume'].tolist()
documents.insert(0, job_description)  # Job description is the first document

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Compute cosine similarity of job_description vs all resumes
cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Add match scores to dataframe
df['match_percent'] = (cosine_similarities * 100).round(2)

# Sort by match
df_sorted = df.sort_values(by='match_percent', ascending=False)

# Save to CSV
df_sorted.to_csv('output/resume_matches.csv', index=False)

print("✅ Resume matching completed. Check 'output/resume_matches.csv' for results.")
