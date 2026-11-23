import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🔍 AI Job Matcher - Semantic Search")
st.write("Paste Job Description -> Find Best Matching Candidates using Embeddings / Semantic Search")

# Sample Candidates Database
candidates = [
    {"Name": "Dev", "Skills": "Python, Streamlit, Groq API, Llama, SQL, GitHub"},
    {"Name": "Rahul", "Skills": "Java, Spring Boot, MySQL, HTML, CSS"},
    {"Name": "Priya", "Skills": "Python, Machine Learning, Embeddings, Semantic Search, FastAPI"},
    {"Name": "Sai", "Skills": "Python, SQL, Data Analysis, Pandas, Excel"},
    {"Name": "Anitha", "Skills": "React, JavaScript, HTML, CSS, Node.js"},
    {"Name": "Kiran", "Skills": "Python, LLM, Prompt Engineering, RAG, Vector Database"},
]

df = pd.DataFrame(candidates)
st.subheader("📋 All Candidates")
st.dataframe(df, use_container_width=True)

st.subheader("💼 Paste Job Description")
job_desc = st.text_area("Example: Looking for Python developer with LLM, Embeddings, FastAPI experience", height=120)

if st.button("Find Best Match"):
    if job_desc == "":
        st.warning("Please paste Job Description first")
    else:
        all_texts = [job_desc] + df["Skills"].tolist()
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(all_texts)
        scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        df["Match Score %"] = (scores * 100).round(1)
        df_sorted = df.sort_values(by="Match Score %", ascending=False)
        
        st.subheader("🏆 Top Matches")
        st.dataframe(df_sorted, use_container_width=True)
        
        best = df_sorted.iloc[0]
        st.success(f"Best Match: {best['Name']} - {best['Match Score %']}% Match - Skills: {best['Skills']}")
        st.bar_chart(df_sorted.set_index("Name")["Match Score %"])