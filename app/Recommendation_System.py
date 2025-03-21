import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_text
from fuzzywuzzy import process

# Load datasets
business_df = pd.read_csv("../datasets/corrected_business_ideas.csv", encoding="ISO-8859-1")
learning_df = pd.read_csv("../datasets/full_learning_paths.csv")

# Skill synonym dictionary
skill_synonyms = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "photoshop": "graphic design",
    "tailoring": "sewing",
    "stiching": "sewing",
    "Finance": "Accounting",
    "Consulting": "Business Advisory",
    "E-Commerce": "Online Selling",
    "Leadership": "Management",
    "Coding": "Programming",
    "Marketing": "Digital Marketing",
    "SEO": "Search Engine Optimization",
    "Graphic Design": "UI/UX",
    "Data Analysis": "Data Science",
    "Writing": "Content Creation",
    "Social Media": "Influencer Marketing",
    "Photography": "Photo Editing",
    "Legal": "Law",
    "Teaching": "Education",
}

def normalize_skills(user_skills):
    return " ".join([skill_synonyms.get(skill, skill) for skill in user_skills.split()])

# Preprocess & normalize
business_df['Required Skills'] = business_df['Required Skills'].astype(str).apply(lambda x: normalize_skills(preprocess_text(x)))
learning_df['Skill'] = learning_df['Skill'].apply(lambda x: normalize_skills(preprocess_text(x)))

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
vectorizer.fit(pd.concat([business_df['Required Skills'], learning_df['Skill']]))

business_tfidf = vectorizer.transform(business_df['Required Skills'])
learning_tfidf = vectorizer.transform(learning_df['Skill'])

def recommend_business_ideas(user_skills, num_recommendations=3):
    user_vector = vectorizer.transform([preprocess_text(user_skills)])
    similarity_scores = cosine_similarity(user_vector, business_tfidf).flatten()
    top_indices = similarity_scores.argsort()[-num_recommendations:][::-1]
    return business_df.iloc[top_indices]

def recommend_learning_paths(user_skills, num_recommendations=3, platform_filter="All"):
    user_vector = vectorizer.transform([preprocess_text(user_skills)])
    similarity_scores = cosine_similarity(user_vector, learning_tfidf).flatten()
    top_indices = similarity_scores.argsort()[-num_recommendations:][::-1]
    recommendations = learning_df.iloc[top_indices]
    
    if platform_filter != "All":
        recommendations = recommendations[recommendations['Platform'].str.contains(platform_filter, case=False, na=False)]
    
    return recommendations

# Streamlit UI
st.title("🚀 Skill-Based Business Recommendation System")
st.write("Helping you find the best business opportunities & learning resources!")

user_skills = st.text_input("🎯 Enter your skills (comma-separated):", "")
num_recommendations = st.slider("🔢 Number of recommendations:", 1, 5, 3)
interest = st.radio("💡 What are you looking for?", ("Business Ideas", "Learning Paths & Resources"))

platform_filter = "All"
if interest == "Learning Paths & Resources":
    platform_filter = st.selectbox("🎓 Select platform:", ["All", "YouTube", "Udemy", "Coursera"])

if st.button("🔍 Get Recommendations"):
    if interest == "Business Ideas":
        business_recs = recommend_business_ideas(user_skills, num_recommendations)
        st.subheader("🚀 Best Business Ideas for You:")
        for _, row in business_recs.iterrows():
            st.markdown(f"✅ **{row['Business Idea']}**")
            st.text(f"📌 Domain: {row['Domain']}")
            st.text(f"💰 Investment: {row['Initial Investment']}")
            st.text(f"🔥 Profit Level: {row['Profit Potential']}")
            st.text(f"📈 Scalability: {row['Scalability']}")
            st.markdown("---")
    else:
        learning_recs = recommend_learning_paths(user_skills, num_recommendations, platform_filter)
        
        st.subheader("🎓 Recommended Learning Paths & Resources:")
        for _, row in learning_recs.iterrows():
            st.markdown(f"✅ **{row['Skill']} ({row['Level']})**")
            st.text(f"📌 Resource: {row['Resource']}")
            st.text(f"🎓 Platform: {row['Platform']}")
            st.text(f"💰 Type: {row['Type']}")
            st.markdown("---")

st.markdown("🌟 Thank you for using the Skill-Based Recommendation System! 🌟")
