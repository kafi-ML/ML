import pandas as pd
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)

# Load dataset
df = pd.read_csv('data/UpdatedResumeDataSet.csv')

# Clean the resume column
df['clean_resume'] = df['Resume'].apply(clean_text)

# Show preview
print(df[['Category', 'clean_resume']].head())

# Optionally save this cleaned data

df[['Category', 'Resume', 'clean_resume']].to_csv('output/cleaned_resumes.csv', index=False)



print("✅ Cleaned resumes saved to output/cleaned_resumes.csv")
