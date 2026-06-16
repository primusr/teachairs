import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from wordcloud import WordCloud
import base64

# Import utility functions
from utils import (
    load_nltk,
    configure_gemini,
    vader_standard,
    vader_augmented,
    update_vader_lexicon,
    preprocess,
    get_standard_vader,
    get_augmented_vader,
    label_from_score,
    classify_sentiment,
    sentiment_color,
    filipino_keyword_sentiment,
    SENTIMENT_SCORE_MAP
)
from utils import train_lda, get_topic_keywords, load_filipino_vader_lexicon
from utils import STOP_WORDS
from utils import APIRateLimiter

# Load NLTK resources
load_nltk()

# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(
    page_title="TeachAIRs",
    page_icon="🧊",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Developed by Neo under the supervision of the Oracle. Watch this short video for a tutorial on how to use the app:  https://www.youtube.com/shorts/OvRlMiYURhM",
        "Get Help": "https://www.linkedin.com/in/unclebreaker/",
        "Report a bug": "https://www.linkedin.com/in/unclebreaker/"
    }    
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background-color: #0E1117;
}
[data-testid="stSidebar"] {
    background-color: #161A23;
}
</style>
""", unsafe_allow_html=True)

st.title("TeachAIRs: Student Feedback Analyzer with AI Recommendations")

# ------------------------------
# Gemini API Setup
# ------------------------------
api_key = st.secrets["api_key"]
gemini_model = configure_gemini(api_key)
rate_limiter = APIRateLimiter()

# ------------------------------
# Filipino Lexicon Upload
# ------------------------------
filipino_lexicon_file = st.file_uploader(
    "📤 Upload Filipino VADER Lexicon CSV (word, score)",
    type=["csv"]
)

if filipino_lexicon_file:
    success, message = update_vader_lexicon(filipino_lexicon_file)
    if success:
        st.success(message)
    else:
        st.error(message)

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

    st.divider()
    st.header("Feedback Dataset Overview")
    st.dataframe(df.head())
  
    df["Cleaned"] = df["Feedback"].apply(preprocess)
    df["VADER_Standard"] = df["Feedback"].apply(get_standard_vader)
    df["VADER_Augmented"] = df["Feedback"].apply(get_augmented_vader)
    df["Score"] = df["VADER_Augmented"]
    df["Label"] = df["Score"].apply(label_from_score)

    # Optional: Topic modeling
    st.divider()
    st.header("Topic Modeling (Optional)")
    lda_model = None
    dictionary = None
    corpus = None
    topics = {}
    if st.checkbox("Run LDA topic modeling on cleaned feedback"):
        tokenized = df["Cleaned"].apply(lambda x: x.split()).tolist()
        tokenized = [t for t in tokenized if t]
        if not tokenized:
            st.warning("No tokenized text available for LDA.")
        else:
            num_topics = st.number_input("Number of topics", min_value=2, max_value=12, value=4)
            lda_model, dictionary, corpus = train_lda(tokenized, num_topics=int(num_topics))
            if lda_model is None:
                st.error("LDA could not be trained (too few terms).")
            else:
                topics = get_topic_keywords(lda_model, topn=6)
                st.subheader("Top keywords per topic")
                for tid, kws in topics.items():
                    st.write(f"Topic {tid}: {', '.join(kws)}")
                # Generate word clouds for each topic
                try:
                    st.subheader("Topic Word Clouds")
                    for tid in range(lda_model.num_topics):
                        # get (word, weight) pairs for topic
                        pairs = lda_model.show_topic(tid, topn=40)
                        freqs = {word: float(weight) for word, weight in pairs}
                        wc = WordCloud(width=600, height=360, background_color='white').generate_from_frequencies(freqs)
                        fig_wc, ax_wc = plt.subplots(figsize=(6, 4))
                        ax_wc.imshow(wc, interpolation='bilinear')
                        ax_wc.axis('off')
                        # include top keywords in the figure title
                        top_keywords = ', '.join([w for w, _ in pairs[:8]])
                        title = f"Topic {tid} — {top_keywords}"
                        ax_wc.set_title(title, fontsize=10)
                        st.pyplot(fig_wc)
                except Exception:
                    st.warning("Could not generate topic word clouds.")
  
    st.divider()
    st.header("Sentiment Distribution (Augmented Model)")
    counts = df["Label"].value_counts()
    sentiment_order = ["Positive", "Neutral", "Negative"]
    counts = counts.reindex(sentiment_order, fill_value=0)
    
    color_map = {
        "Positive": "green",
        "Neutral": "blue",
        "Negative": "red"
    }
    colors = [color_map[label] for label in counts.index]
    fig1, ax1 = plt.subplots()
    counts.plot(kind="bar", ax=ax1, color=colors)
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Sentiment")
    ax1.set_title("Sentiment Distribution")
    st.pyplot(fig1)

    # Overall System Sentiment Scores & Distribution
    st.divider()
    st.header("Overall System Sentiment Scores & Distribution")
    avg_std_score = df["VADER_Standard"].mean()
    avg_aug_score = df["VADER_Augmented"].mean()
    avg_score = df["Score"].mean() if "Score" in df else avg_aug_score
    st.markdown(f"""
    **Average Standard VADER Score:** {avg_std_score:.3f}  
    **Average Augmented VADER Score:** {avg_aug_score:.3f}  
    **Overall Sentiment (Augmented):** {'Positive' if avg_score > 0.05 else 'Negative' if avg_score < -0.05 else 'Neutral'}
    """)

    dist_std = df["VADER_Standard"].apply(label_from_score).value_counts().reindex(sentiment_order, fill_value=0)
    dist_aug = df["VADER_Augmented"].apply(label_from_score).value_counts().reindex(sentiment_order, fill_value=0)
    dist_fil = df["Cleaned"].apply(filipino_keyword_sentiment).value_counts().reindex(sentiment_order, fill_value=0)
    summary_df = pd.DataFrame({
        "Standard VADER": dist_std,
        "Augmented VADER": dist_aug,
        "Filipino Keywords": dist_fil
    })
    summary_df.index.name = "Sentiment"
    st.table(summary_df)

    avg_score = df["Score"].mean()
    st.markdown(f"""
    **Average Sentiment Score:** {avg_score:.3f}  
    **Overall Sentiment:** {'Positive' if avg_score > 0.05 else 'Negative' if avg_score < -0.05 else 'Neutral'}
    """)

    # VADER vs Augmented VADER Comparison
    st.divider()
    st.header("VADER vs Augmented VADER Comparison")
    # derive labels
    df['Std_Label'] = df['VADER_Standard'].apply(label_from_score)
    df['Aug_Label'] = df['VADER_Augmented'].apply(label_from_score)
    # agreement rate
    try:
        agreement_rate = (df['Std_Label'] == df['Aug_Label']).mean()
    except Exception:
        agreement_rate = 0.0
    st.write(f"Agreement rate: {agreement_rate:.2%}")

    # Confusion matrix (Std rows, Aug cols)
    conf = pd.crosstab(df['Std_Label'], df['Aug_Label'])
    st.subheader("Label Confusion Matrix")
    st.table(conf)

    # Show counts of changed labels
    changes = df[df['Std_Label'] != df['Aug_Label']]
    st.write(f"Number of comments with different labels: {len(changes)}")
    if not changes.empty:
        st.write("Sample changes (first 5):")
        st.dataframe(changes[['Feedback','Std_Label','Aug_Label']].head())

    # Comparison scatter plot for sentiment polarity scores across methods
    df['Comment_Index'] = list(range(len(df)))
    category_colors = {'Positive': 'green', 'Neutral': 'orange', 'Negative': 'red'}
    std_colors = df['VADER_Standard'].apply(label_from_score).map(category_colors)
    aug_colors = df['VADER_Augmented'].apply(label_from_score).map(category_colors)

    fig_cmp, (ax_std, ax_aug) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    ax_std.scatter(df['Comment_Index'], df['VADER_Standard'], c=std_colors, alpha=0.7)
    ax_std.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax_std.set_xlabel('Feedback Comment Index')
    ax_std.set_ylabel('Sentiment Polarity Score')
    ax_std.set_title('Standard VADER')
    ax_std.set_ylim(-1.05, 1.05)

    ax_aug.scatter(df['Comment_Index'], df['VADER_Augmented'], c=aug_colors, alpha=0.7)
    ax_aug.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax_aug.set_xlabel('Feedback Comment Index')
    ax_aug.set_title('Augmented VADER')
    ax_aug.set_ylim(-1.05, 1.05)

    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='green', label='Positive'),
        mpatches.Patch(color='orange', label='Neutral'),
        mpatches.Patch(color='red', label='Negative')
    ]
    ax_aug.legend(handles=legend_handles, title='Category', loc='upper right')
    st.pyplot(fig_cmp)

    st.markdown("**Additional insight:** Left panel shows the Standard VADER score clustering; right panel shows how Augmented VADER shifts sentiment toward the neutral band.")

    # Gemini AI Recommendations
    st.divider()
    st.header("AI Recommendations (Gemini)")
    if not gemini_model:
        st.info("Gemini not configured. Set `api_key` in Streamlit secrets to enable AI recommendations.")
    else:
        if st.button("Generate Overall AI Recommendations"):
            # build a concise summary
            with st.spinner("Generating AI recommendations..."):
                avg_score = df["Score"].mean() if "Score" in df else 0
                summary = f"Overall Avg Augmented VADER Score: {avg_score:.3f}\n"
                counts = df["Label"].value_counts().to_dict()
                summary += "Distribution:\n"
                for k, v in counts.items():
                    summary += f" - {k}: {v}\n"
                # include topics if available
                if lda_model is not None:
                    topics = get_topic_keywords(lda_model, topn=6)
                    summary += "\nTop Topics:\n"
                    for tid, kws in topics.items():
                        summary += f" - Topic {tid}: {', '.join(kws)}\n"
                # include sample positive/negative
                pos_sample = df[df["Label"] == 'Positive']["Feedback"].head(1).tolist()
                neg_sample = df[df["Label"] == 'Negative']["Feedback"].head(1).tolist()
                summary += f"\nExample Positive: {pos_sample[0] if pos_sample else 'N/A'}\n"
                summary += f"Example Negative: {neg_sample[0] if neg_sample else 'N/A'}\n"

                prompt = (
                    "You are an educational consultant. Given the feedback summary below, provide 3 actionable teaching recommendations with short headings and rationale.\n\n"
                    f"Summary:\n{summary}\n"
                    "Output as plain text."
                )

                try:
                    rate_limiter.wait_if_needed()
                    resp = gemini_model.generate_content(prompt)
                    rec_text = resp.text if hasattr(resp, 'text') else str(resp)
                    st.subheader("AI Recommendations")
                    st.text(rec_text)
                    # allow download
                    st.download_button("Download Recommendations", rec_text, file_name="ai_recommendations.txt")
                except Exception as e:
                    st.error(f"AI request failed: {e}")