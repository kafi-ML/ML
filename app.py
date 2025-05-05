import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smart ATS Resume Matcher", layout="wide")
st.title("📄 Smart ATS - Resume Matcher")

df = pd.read_csv('output/cleaned_resumes.csv')


# Upload cleaned resumes CSV
uploaded_file = st.file_uploader("Upload Cleaned Resumes CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Step 1: Paste Job Description")
    job_description = st.text_area("Paste the job description here:", height=200)

    if st.button("🔍 Match Resumes"):
        if not job_description.strip():
            st.warning("Please enter a job description first.")
        else:
            # Combine job description with resumes
            df = pd.read_csv('output/cleaned_resumes.csv')
            documents = df['clean_resume'].tolist()
            documents.insert(0, job_description)

   



            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(documents)

            # Compute cosine similarity
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            # Add scores
            df['match_percent'] = (cosine_similarities * 100).round(2)
            df_sorted = df.sort_values(by='match_percent', ascending=False)

            st.success("✅ Matching complete!")
            st.dataframe(df_sorted[['Category', 'match_percent']].reset_index(drop=True))

            csv = df_sorted.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results as CSV", csv, "resume_matches.csv", "text/csv")
