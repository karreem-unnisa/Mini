import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Function to ensure required NLTK data is available
def download_nltk_resources():
    nltk_packages = ['stopwords', 'wordnet']
    for package in nltk_packages:
        try:
            nltk.data.find(f'corpora/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

# Download resources only if not available
download_nltk_resources()

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Cleans and preprocesses text for better ML training."""
    if not isinstance(text, str) or text.strip() == "":
        return ""  # Handle missing values

    text = text.lower().strip()  # Lowercasing & trimming
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters & numbers
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]  # Tokenization, Stopword Removal & Lemmatization
    return ' '.join(words)  # Convert back to string
