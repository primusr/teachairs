# ==============================
# TeachAIRs: Utility Functions
# Helper functions for sentiment analysis
# ==============================

import re
import warnings
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import google.generativeai as genai
import streamlit as st
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
  font-family: "Poppins", sans-serif;

}

/* Sidebar */
[data-testid="stSidebar"] {
   font-family: "Poppins", sans-serif;

}

/* Buttons */
button {
   font-family: "Poppins", sans-serif;
  
}
</style>
""", unsafe_allow_html=True)

warnings.filterwarnings("ignore")

# ------------------------------
# NLTK Setup
# ------------------------------
@st.cache_resource
def load_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("vader_lexicon")
    nltk.download('punkt_tab')

load_nltk()

# ------------------------------
# Gemini API Configuration
# ------------------------------
@st.cache_resource
def configure_gemini(key):
    if not key:
        return None
    try:
        genai.configure(api_key=key)
        return genai.GenerativeModel("models/gemini-3.1-pro-preview")
    except:
        return None

# ------------------------------
# Initialize VADER Models
# ------------------------------
vader_standard = SentimentIntensityAnalyzer()
vader_augmented = SentimentIntensityAnalyzer()

# Function to update augmented VADER with custom lexicon
def update_vader_lexicon(lexicon_file):
    """Update vader_augmented with custom Filipino lexicon"""
    try:
        lex_df = pd.read_csv(lexicon_file)
        custom_dict = dict(zip(lex_df.iloc[:, 0], lex_df.iloc[:, 1]))
        vader_augmented.lexicon.update(custom_dict)
        return True, "✅ Filipino Lexicon Applied to Augmented VADER"
    except Exception as e:
        return False, f"Error loading lexicon: {e}"

# ------------------------------
# Stopwords Definition
# ------------------------------
FILIPINO_STOPWORDS = {
    "ang","ng","sa","si","ni","mga","ito","iyan","iyon","ako","ikaw","siya", 
    "kami","tayo","kayo","sila","natin","amin","nila","mo","ko","ka","pa", 
    "din","rin","lang","naman","po","opo","ata","kasi","pero","dahil", 
    "kung","kapag","habang","mula","para","gaya","tulad","ganito","ganyan","ganoon","dito","diyan","doon"
}

DOMAIN_STOPWORDS = {
    "teacher","professor","sir","maam","mam","ma'am", 
    "subject","course","class","lesson","topic","discussion", 
    "activity","activities","student","students","school", 
    "semester","learning","teach","teaching",":)"
}

FILLER_WORDS = {
    "good","nice","great","really","very","much","many","lot","lots", 
    "quite","something","anything","everything","nothing", 
    "ok","okay","yes","no","maybe","also","still","even","well","yet"
}

LDA_STOPWORDS = {
    "a", "about", "all", "am", "an", "and", "any", "are", "as", "at", "be", 
    "because", "been", "but", "by", "can", "do", "does", "for", "from", "had", 
    "has", "have", "he", "her", "here", "hers", "him", "his", "how", "i", "if", 
    "in", "into", "is", "it", "its", "just", "me", "more", "most", "my", "no", 
    "not", "of", "on", "only", "or", "other", "our", "out", "over", "own", 
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", 
    "them", "then", "there", "these", "they", "this", "those", "through", "to", 
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", 
    "while", "who", "whom", "why", "will", "with", "you", "your", "mam", "maam", 
    "sir", "po", "lang", "naman", "wala", "nya", "sana", "da", "en", "mag", 
    "pala", "kasi", "wag", "tsaka", "di", "pang", "pag", "thankyou", "ako", 
    "naman", "kita", "ur", "jan", "kay", "niyo", "rin", "paki", "ta", "ata", "kayo"
}

# Combine all stopwords
STOP_WORDS = set(stopwords.words("english")).union(
    FILIPINO_STOPWORDS,
    DOMAIN_STOPWORDS,
    FILLER_WORDS,
    LDA_STOPWORDS
)

lemmatizer = WordNetLemmatizer()

# ------------------------------
# Text Preprocessing Function
# ------------------------------
def preprocess(text):
    """Preprocess text: lowercase, remove URLs, remove non-letters, tokenize, remove stopwords, lemmatize"""
    text = str(text).lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove non letters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # tokenize
    tokens = nltk.word_tokenize(text)

    # remove stopwords + lemmatize
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in STOP_WORDS and len(w) > 2
    ]

    return " ".join(tokens)

# ------------------------------
# Sentiment Scoring Functions
# ------------------------------
def get_standard_vader(text):
    """Get VADER standard sentiment score"""
    return vader_standard.polarity_scores(text)["compound"]

def get_augmented_vader(text):
    """Get VADER augmented sentiment score (with Filipino lexicon)"""
    return vader_augmented.polarity_scores(text)["compound"]

def label_from_score(score):
    """Convert sentiment score to label"""
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

def classify_sentiment(score):
    """Classify sentiment (alias for label_from_score)"""
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

def sentiment_color(score):
    """Get color for sentiment visualization"""
    if score > 0:
        return "green"
    elif score < 0:
        return "red"
    else:
        return "yellow"

# Filipino keyword sentiment analysis
FILIPINO_POSITIVE = ["maganda", "mabuti", "mahusay", "salamat", "okay"]
FILIPINO_NEGATIVE = ["pangit", "masama", "mahina", "problema", "hindi"]

def filipino_keyword_sentiment(text):
    """Analyze sentiment using Filipino keywords"""
    text = str(text).lower()
    pos = sum(word in text for word in FILIPINO_POSITIVE)
    neg = sum(word in text for word in FILIPINO_NEGATIVE)

    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    else:
        return "Neutral"

# Mapping for Filipino sentiment scores
SENTIMENT_SCORE_MAP = {"Positive": 1, "Neutral": 0, "Negative": -1}
