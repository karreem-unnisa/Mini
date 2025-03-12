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

def recommend_business_ideas(user_skills, num_recommendations=1):
    user_vector = vectorizer.transform([preprocess_text(user_skills)])
    similarity_scores = cosine_similarity(user_vector, business_tfidf).flatten()
    top_indices = similarity_scores.argsort()[-num_recommendations:][::-1]
    return business_df.iloc[top_indices][['Business_Idea', 'Category', 'Initial_Investment']]

def recommend_learning_paths(user_skills, num_recommendations=1):
    user_vector = vectorizer.transform([preprocess_text(user_skills)])
    similarity_scores = cosine_similarity(user_vector, learning_tfidf).flatten()
    top_indices = similarity_scores.argsort()[-num_recommendations:][::-1]
    return learning_df.iloc[top_indices][['Skill_Category', 'Beginner_Level', 'Intermediate_Level', 'Advanced_Level', 'Duration', 'Earning_Potential_Post_Learning']]

def recommend_learning_courses(user_skills, num_recommendations=1):
    user_vector = vectorizer.transform([preprocess_text(user_skills)])
    similarity_scores = cosine_similarity(user_vector, learning_resources_tfidf).flatten()
    top_indices = similarity_scores.argsort()[-num_recommendations:][::-1]
    return learning_resources_df.iloc[top_indices][['Skill_Name', 'Resource_Type', 'Resource_Name', 'Platform', 'Cost', 'Skill_Level', 'Duration', 'Earning_Potential_After_Learning']]

def suggest_trends_and_tips():
    return market_trends_df.sample(3)[['Industry', 'Current_Trend', 'Future_Potential']]

def hybrid_recommendation():
    print("🌟 Welcome to the Skill-Based Recommendation System! Let's find the best opportunities for you! 🌟")
    user_skills = input("🎯 What skills do you have? (comma-separated): ").strip().lower()
   
    interest = input("💡 Are you looking to start a business? (yes/no): ").strip().lower()
    
    wants_business = interest == "yes"
    
    if wants_business:
        num_recommendations = int(input("🔢 How many recommendations do you want? (Enter a number): "))
        business_recs = recommend_business_ideas(user_skills, num_recommendations)
        trends = suggest_trends_and_tips()
        
        print("\n🚀 *Best Business Ideas for You:*\n")
        for _, row in business_recs.iterrows():
            print(f"🔹 {row['Business_Idea']} ({row['Category']})")
            print(f"   - Estimated Initial Investment: ${row['Initial_Investment']}")
            print("--------------------")
        print("\n📊 *Trends & Market Insights:*\n", trends.to_string())
    else:
        print('\n Try learning these skills to make yourself more confident and efficient in them by following these learning paths and resources! \n')
        num_recommendations = int(input("🔢 How many learning paths/resources do you want? (Enter a number): "))
        learning_recs = recommend_learning_paths(user_skills, num_recommendations)
        courses_recs = recommend_learning_courses(user_skills, num_recommendations)
        print('\n Try learning these skills to make yourself more confident and efficient in them by following these learning paths and resources! \n')
        print("\n🎓 *Recommended Learning Paths:*\n")
        for _, row in learning_recs.iterrows():
            print(f"🔹 {row['Skill_Category']}")
            print(f"   - Duration: {row['Duration']} months")
            print(f"   - Earning Potential: ${row['Earning_Potential_Post_Learning']}")
            print("   - Beginner Level: ", row['Beginner_Level'])
            print("   - Intermediate Level: ", row['Intermediate_Level'])
            print("   - Advanced Level: ", row['Advanced_Level'])
            print("--------------------")
        
        print("\n📚 *Recommended Learning Resources:*\n")
        for _, row in courses_recs.iterrows():
            print(f"📖 {row['Resource_Name']} ({row['Platform']})")
            print(f"   - Type: {row['Resource_Type']}")
            print(f"   - Cost: ${row['Cost']}")
            print(f"   - Skill Level: {row['Skill_Level']}")
            print(f"   - Duration: {row['Duration']} hours")
            print(f"   - Earning Potential After Learning: ${row['Earning_Potential_After_Learning']}")
            print("--------------------")

# Run the interactive recommendation system
if __name__ == "__main__":
    hybrid_recommendation()
print("\n🌟 Thank you for using the Skill-Based Recommendation System! 🌟")
