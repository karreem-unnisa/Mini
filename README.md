# Skill-Based Business Recommendation Model for Women

## Overview
This project is a **Hybrid Recommendation System** that helps women discover **business ideas** and **learning paths** based on their skills and interests. The model starts with **content-based filtering** and transitions to **collaborative filtering** as user data becomes available. It also considers **market trends**, **skill demand**, and **earning potential** while making recommendations.

## Features
- **Business Idea Recommendations**: Suggests business ideas based on user skills and interests.
- **Learning Path Recommendations**: If the user lacks confidence to start a business, suggests structured learning paths.
- **Trend Awareness**: Uses market trends and skill demand to provide relevant suggestions.
- **Confidence Level Analysis**: Determines user confidence based on input patterns.
- **No User ID Required**: Provides recommendations without requiring login credentials.
- **Text-Based UI-Friendly Output**: Generates readable suggestions in a structured format.

---
## Project Structure
```
mini-project/
│── Mini/ (Git Repository)
│   ├── datasets/  # Contains CSV files for training and recommendations
│   ├── app/       # Contains application logic and model execution scripts
│   ├── models/    # Trained ML models
│   ├── requirements.txt  # Dependencies for the project
```

### Datasets Used
1. **business_ideas_large.csv** - Expanded business ideas dataset
2. **learning_paths_large.csv** - Detailed learning paths dataset
3. **skill_demand_large.csv** - Skill demand dataset
4. **market_trends_large.csv** - Market trends dataset

---
## Installation & Setup

### Prerequisites
Ensure you have Python installed (preferably **Python 3.10+**).

### Step 1: Clone the Repository
```sh
git clone <repository_url>
cd Mini
```

### Step 2: Install Required Libraries
Run the following command inside the project directory:
```sh
pip install -r requirements.txt
```
This will install the necessary dependencies like:
- `pandas`
- `scikit-learn`
- `numpy`
- `nltk`
- `flask` (if deploying as a web service)

### Step 3: Run the Model
To execute the recommendation system:
```sh
python app/hybrid_recommendation.py
```

---
## How to Use
1. **Input Skills & Interests**: The system will first ask for your skills and whether you're interested in starting a business.
2. **Business or Learning Paths**: Based on your response, it will generate either **business recommendations** or **learning paths**.
3. **Trend-Based Suggestions**: Additional suggestions based on market trends and skill demand.
4. **Output**: The results are displayed in a structured, UI-friendly text format.

---
## Future Enhancements
- Integration of **community networks** for businesswomen.
- Implementation of **confidence assessment via NLP**.
- Expansion to include **multilingual support**.


