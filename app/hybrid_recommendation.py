import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from preprocessing import preprocess_text  # ✅ Importing preprocessing function

# Load datasets
business_df = pd.read_csv("../datasets/business_ideas_meaningful.csv")
learning_df = pd.read_csv("../datasets/learning_paths_expanded.csv")
skill_demand_df = pd.read_csv("../datasets/skill_demand_expanded.csv")
market_trends_df = pd.read_csv("../datasets/market_trends_expanded.csv")
learning_resources_df = pd.read_csv("../datasets/learning_resources.csv")

# Apply preprocessing
business_df['Skill_Required'] = business_df['Skill_Required'].apply(preprocess_text)
learning_df['Skill_Category'] = learning_df['Skill_Category'].apply(preprocess_text)
skill_demand_df['Skill_Name'] = skill_demand_df['Skill_Name'].apply(preprocess_text)
market_trends_df['Industry'] = market_trends_df['Industry'].apply(preprocess_text)
learning_resources_df['Skill_Name'] = learning_resources_df['Skill_Name'].apply(preprocess_text)

# Combine all skills for a unified TF-IDF vectorizer
all_skills = pd.concat([
    business_df['Skill_Required'], 
    learning_df['Skill_Category'], 
    skill_demand_df['Skill_Name'],
    learning_resources_df['Skill_Name']
])

vectorizer = TfidfVectorizer()
vectorizer.fit(all_skills)

# Transform datasets using the trained vectorizer
business_tfidf = vectorizer.transform(business_df['Skill_Required'])
learning_tfidf = vectorizer.transform(learning_df['Skill_Category'])
learning_resources_tfidf = vectorizer.transform(learning_resources_df['Skill_Name'])

def recommend_business_ideas(user_skills):
    user_vector = vectorizer.transform([preprocess_text(user_skills)])
    similarity_scores = cosine_similarity(user_vector, business_tfidf).flatten()
    top_indices = similarity_scores.argsort()[-5:][::-1]
    return business_df.iloc[top_indices][['Business_Idea', 'Category', 'Initial_Investment']]

def recommend_learning_paths(user_skills):
    user_vector = vectorizer.transform([preprocess_text(user_skills)])
    similarity_scores = cosine_similarity(user_vector, learning_tfidf).flatten()
    top_indices = similarity_scores.argsort()[-5:][::-1]
    return learning_df.iloc[top_indices][['Skill_Category', 'Beginner_Level', 'Intermediate_Level', 'Advanced_Level', 'Duration', 'Earning_Potential_Post_Learning']]

def recommend_learning_courses(user_skills):
    user_vector = vectorizer.transform([preprocess_text(user_skills)])
    similarity_scores = cosine_similarity(user_vector, learning_resources_tfidf).flatten()
    top_indices = similarity_scores.argsort()[-5:][::-1]
    return learning_resources_df.iloc[top_indices][['Skill_Name', 'Resource_Type', 'Resource_Name', 'Platform', 'Cost', 'Skill_Level', 'Duration', 'Earning_Potential_After_Learning']]

def suggest_trends_and_tips():
    return market_trends_df.sample(3)[['Industry', 'Current_Trend', 'Future_Potential']]

def hybrid_recommendation():
    user_skills = input("Enter your skills (comma-separated): ").strip().lower()
    interest = input("Are you interested in starting a business? (yes/no): ").strip().lower()
    wants_business = interest == "yes"
    
    if wants_business:
        business_recs = recommend_business_ideas(user_skills)
        trends = suggest_trends_and_tips()
        print("\nBusiness Ideas:\n", business_recs.to_string(), "\n\nTrends:\n", trends.to_string())
    else:
        learning_recs = recommend_learning_paths(user_skills)
        courses_recs = recommend_learning_courses(user_skills)
        print("\nRecommended Learning Paths:\n", learning_recs.to_string())
        print("\nRecommended Learning Resources:\n", courses_recs.to_string())

# Run the interactive recommendation system
if __name__ == "__main__":
    hybrid_recommendation()
