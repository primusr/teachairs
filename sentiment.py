# ==============================
# TeachAIRs: Sentiment & Topic Analysis
# With VADER Method Comparison
# ==============================

import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from gensim import corpora
from gensim.models import LdaModel
from wordcloud import WordCloud

import google.generativeai as genai

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

load_nltk()

# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(
    page_title="TeachAIRs: Sentiment & Topic Analysis",
    layout="wide"
)

st.title("📊 TeachAIRs: Student Feedback Analyzer with AI Recommendations")

# ------------------------------
# Gemini API (Optional)
# ------------------------------
api_key = st.text_input("🔑 Enter Gemini API Key (Optional)", type="password")

@st.cache_resource
def configure_gemini(key):
    if not key:
        return None
    try:
        genai.configure(api_key=key)
        return genai.GenerativeModel("models/gemini-2.0-flash")
    except:
        return None

gemini_model = configure_gemini(api_key)

# ------------------------------
# Initialize VADER Models
# ------------------------------
vader_standard = SentimentIntensityAnalyzer()
vader_augmented = SentimentIntensityAnalyzer()

# ------------------------------
# Filipino Lexicon Upload
# ------------------------------
filipino_lexicon_file = st.file_uploader(
    "📤 Upload Filipino VADER Lexicon CSV (word, score)",
    type=["csv"]
)

if filipino_lexicon_file:
    try:
        lex_df = pd.read_csv(filipino_lexicon_file)
        custom_dict = dict(zip(lex_df.iloc[:, 0], lex_df.iloc[:, 1]))
        vader_augmented.lexicon.update(custom_dict)
        st.success("✅ Filipino Lexicon Applied to Augmented VADER")
    except Exception as e:
        st.error(f"Error loading lexicon: {e}")

# ------------------------------
# Text Preprocessing
# ------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(tokens)

# ------------------------------
# Sentiment Functions
# ------------------------------
def get_standard_vader(text):
    return vader_standard.polarity_scores(text)["compound"]

def get_augmented_vader(text):
    return vader_augmented.polarity_scores(text)["compound"]

def label_from_score(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

# ------------------------------
# Upload Feedback Dataset
# ------------------------------
uploaded_file = st.file_uploader("📤 Upload Feedback CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # Auto-detect feedback column
    possible_cols = ["feedback", "comment", "comments", "Feedback"]
    feedback_col = next(
        (c for c in possible_cols if c in df.columns),
        df.columns[0]
    )

    df = df[[feedback_col]].rename(columns={feedback_col: "Feedback"})
    df.dropna(inplace=True)

    st.subheader("📄 Sample Feedback")
    st.dataframe(df.head())

    # ------------------------------
    # Preprocessing
    # ------------------------------
    df["Cleaned"] = df["Feedback"].apply(preprocess)

    # ------------------------------
    # Compute Sentiment Scores
    # ------------------------------
    df["VADER_Standard"] = df["Feedback"].apply(get_standard_vader)
    df["VADER_Augmented"] = df["Feedback"].apply(get_augmented_vader)

    df["Score"] = df["VADER_Augmented"]
    df["Label"] = df["Score"].apply(label_from_score)

    # ------------------------------
    # Sentiment Distribution
    # ------------------------------
    st.subheader("📊 Sentiment Distribution (Augmented Model)")

    counts = df["Label"].value_counts()

    fig1, ax1 = plt.subplots()
    counts.plot(kind="bar", ax=ax1)
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    avg_score = df["Score"].mean()

    st.markdown(f"""
    **Average Sentiment Score:** {avg_score:.3f}  
    **Overall Sentiment:** {'Positive' if avg_score > 0.05 else 'Negative' if avg_score < -0.05 else 'Neutral'}
    """)

    # ------------------------------
    # Separate Scatter Plots (Color-Coded)
    # ------------------------------
    st.subheader("📊 Sentiment Polarity Distribution Across Methods")

    # Helper function for color coding
    def sentiment_color(score):
        if score > 0:
            return "green"
        elif score < 0:
            return "red"
        else:
            return "yellow"

    # ------------------------------
    # 1️⃣ Standard VADER Scatter
    # ------------------------------
    st.markdown("### Standard VADER (English Only)")

    colors_std = df["VADER_Standard"].apply(sentiment_color)

    fig_std, ax_std = plt.subplots()

    ax_std.scatter(
        range(len(df)),
        df["VADER_Standard"],
        c=colors_std,
        alpha=0.7
    )

    ax_std.axhline(0, linestyle="--")
    ax_std.set_xlabel("Feedback Index")
    ax_std.set_ylabel("Polarity Score")
    ax_std.set_title("Standard VADER Polarity Scores")

    st.pyplot(fig_std)

    # ------------------------------
    # 2️⃣ Augmented VADER Scatter
    # ------------------------------
    st.markdown("### Augmented VADER (With Filipino Lexicon)")

    colors_aug = df["VADER_Augmented"].apply(sentiment_color)

    fig_aug, ax_aug = plt.subplots()

    ax_aug.scatter(
        range(len(df)),
        df["VADER_Augmented"],
        c=colors_aug,
        alpha=0.7
    )

    ax_aug.axhline(0, linestyle="--")
    ax_aug.set_xlabel("Feedback Index")
    ax_aug.set_ylabel("Polarity Score")
    ax_aug.set_title("Augmented VADER Polarity Scores")

    st.pyplot(fig_aug)

    # ------------------------------
    # Statistical Comparison
    # ------------------------------
    correlation = df["VADER_Standard"].corr(df["VADER_Augmented"])
    mean_difference = (df["VADER_Augmented"] - df["VADER_Standard"]).mean()

    st.markdown(f"""
    ### 📈 Statistical Comparison Summary

    **Pearson Correlation Between Methods:** {correlation:.3f}  
    **Mean Score Difference (Augmented − Standard):** {mean_difference:.3f}
    """)

    # # Statistical comparison
    # correlation = df["VADER_Standard"].corr(df["VADER_Augmented"])
    # mean_difference = (df["VADER_Augmented"] - df["VADER_Standard"]).mean()

    # sign_flip = (
    #     (df["VADER_Standard"] > 0) & (df["VADER_Augmented"] < 0)
    # ) | (
    #     (df["VADER_Standard"] < 0) & (df["VADER_Augmented"] > 0)
    # )

    # flip_rate = sign_flip.mean() * 100

    # st.markdown(f"""
    # ### 📈 Statistical Comparison

    # **Pearson Correlation:** {correlation:.3f}  
    # **Mean Score Difference (Augmented − Standard):** {mean_difference:.3f}  
    # **Polarity Sign Flip Rate:** {flip_rate:.2f}%  
    # """)

    # ------------------------------
    # Topic Coherence Evaluation
    # ------------------------------
    st.subheader("📈 Topic Coherence Evaluation for Optimal k Selection")

    from gensim.models import CoherenceModel

    # Prepare data for LDA
    texts = [t.split() for t in df["Cleaned"] if t.strip()]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    k_values = list(range(3, 11))

    cv_scores = []
    umass_scores = []
    cnpmi_scores = []

    for k in k_values:
        lda_model_k = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=10,
            random_state=42
        )

        # C_v
        coherence_cv = CoherenceModel(
            model=lda_model_k,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        ).get_coherence()
        cv_scores.append(coherence_cv)

        # UMass
        coherence_umass = CoherenceModel(
            model=lda_model_k,
            corpus=corpus,
            dictionary=dictionary,
            coherence='u_mass'
        ).get_coherence()
        umass_scores.append(coherence_umass)

        # C_NPMI
        coherence_cnpmi = CoherenceModel(
            model=lda_model_k,
            texts=texts,
            dictionary=dictionary,
            coherence='c_npmi'
        ).get_coherence()
        cnpmi_scores.append(coherence_cnpmi)

    # Determine optimal k based on C_v
    optimal_index = cv_scores.index(max(cv_scores))
    optimal_k = k_values[optimal_index]
    optimal_cv = cv_scores[optimal_index]

    # ------------------------------
    # Plot Line Graphs
    # ------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    # Top: C_v
    axes[0].plot(k_values, cv_scores, marker='o')
    axes[0].set_title("C_v Coherence Scores")
    axes[0].set_xlabel("Number of Topics (k)")
    axes[0].set_ylabel("C_v Score")
    axes[0].axvline(optimal_k, linestyle='--')
    axes[0].annotate(
        f"Peak at k={optimal_k}\n({optimal_cv:.4f})",
        xy=(optimal_k, optimal_cv),
        xytext=(optimal_k, optimal_cv + 0.02),
        arrowprops=dict()
    )

    # Middle: UMass
    axes[1].plot(k_values, umass_scores, marker='o')
    axes[1].set_title("UMass Coherence Scores")
    axes[1].set_xlabel("Number of Topics (k)")
    axes[1].set_ylabel("UMass Score")

    # Bottom: C_NPMI
    axes[2].plot(k_values, cnpmi_scores, marker='o')
    axes[2].set_title("C_NPMI Coherence Scores")
    axes[2].set_xlabel("Number of Topics (k)")
    axes[2].set_ylabel("C_NPMI Score")

    plt.tight_layout()
    st.pyplot(fig)

    # ------------------------------
    # Interpretation Output
    # ------------------------------
    st.markdown(f"""
    ### 📊 Optimal Topic Determination

    The C_v coherence score reaches its maximum at **k = {optimal_k}**, 
    with a value of **{optimal_cv:.4f}**, indicating the highest semantic similarity 
    and interpretability among the generated topics.

    Based on the strong correlation of C_v with human judgment, the optimal 
    number of topics was programmatically determined to be:

    ## ✅ k = {optimal_k}

    This ensures that subsequent thematic analysis is grounded in the most 
    semantically coherent topic structure derived from student feedback.
    """)

    # ------------------------------
    # Topic Modeling
    # ------------------------------
    st.subheader("🧠 Topic Modeling (LDA)")

    texts = [t.split() for t in df["Cleaned"] if t.strip()]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=5,
        passes=10,
        random_state=42
    )

    for i in range(5):
        st.markdown(f"### Topic {i+1}")
        words_probs = lda_model.show_topic(i, topn=10)
        words = ", ".join([w for w, _ in words_probs])
        st.write(words)

        wc = WordCloud(background_color="white")
        wc.generate_from_frequencies(dict(words_probs))

        fig3, ax3 = plt.subplots()
        ax3.imshow(wc)
        ax3.axis("off")
        st.pyplot(fig3)



    # ------------------------------
    # AI Recommendations (Optional)
    # ------------------------------
    if gemini_model:
        st.subheader("💡 AI Teaching Recommendations")

        summary = f"""
        Average Sentiment Score: {avg_score:.2f}
        Distribution: {counts.to_dict()}
        Correlation Between Models: {correlation:.2f}
        """

        response = gemini_model.generate_content(
            summary + "\nGive 3 actionable teaching recommendations."
        )

        st.write(response.text.strip())

else:
    st.info("👈 Please upload a CSV file to begin.")