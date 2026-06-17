import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os

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

DEFAULT_FEEDBACK_COLUMN_NAMES = ["feedback", "comment", "comments", "Feedback", "Comment", "Strengths"]
OUTPUT_SENTIMENT_COMPARISON_CSV = "sentiment_analysis_results_comparison.csv"


def load_csv_with_encodings(uploaded_file, encodings=None):
    encodings = encodings or ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'unicode_escape']
    last_error = None
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=0, encoding=enc)
            st.write(f"✅ Successfully read CSV with encoding: {enc}")
            return df
        except UnicodeDecodeError as e:
            last_error = e
            st.write(f"⚠️ Failed to decode with {enc}, trying next...")
        except Exception as e:
            last_error = e
            st.write(f"⚠️ Error reading with encoding {enc}: {e}")
    raise ValueError("Could not read CSV. See previous messages.") from last_error


def preprocess_text(text, for_lda=False):
    return preprocess(text)


def clean_dataframe(df, preferred_text_column_names):
    original_text_column = None
    for col_name in preferred_text_column_names:
        if col_name in df.columns:
            original_text_column = col_name
            break
    if not original_text_column:
        if df.shape[1] > 0 and (df.columns[0] == 0 or isinstance(df.columns[0], int)):
            original_text_column = df.columns[0]
        elif any(df[col].dtype == 'object' for col in df.columns):
            potential_cols = [col for col in df.columns if df[col].dtype == 'object']
            original_text_column = max(potential_cols, key=lambda col: df[col].astype(str).str.len().mean(), default=None)
    if not original_text_column and df.shape[1] > 0:
        original_text_column = df.columns[0]
    if not original_text_column:
        raise ValueError("DataFrame empty or no suitable text column.")

    df['Original_Text'] = df[original_text_column].copy()
    df = df[df[original_text_column].notnull() & (df[original_text_column].astype(str) != '0')]
    df.rename(columns={str(original_text_column): 'Feedback_Text'}, inplace=True)
    return df[['Original_Text', 'Feedback_Text']]


def get_vader_sentiment_english(text_for_vader_eng):
    if not isinstance(text_for_vader_eng, str):
        text_for_vader_eng = ""
    compound_score = get_standard_vader(text_for_vader_eng)
    if compound_score >= 0.05:
        sentiment_label = "Positive (VADER Eng)"
    elif compound_score <= -0.05:
        sentiment_label = "Negative (VADER Eng)"
    else:
        sentiment_label = "Neutral (VADER Eng)"
    return compound_score, sentiment_label


def get_vader_sentiment_augmented(cleaned_text_for_vader_aug):
    if not isinstance(cleaned_text_for_vader_aug, str):
        cleaned_text_for_vader_aug = ""
    compound_score = get_augmented_vader(cleaned_text_for_vader_aug)
    if compound_score >= 0.05:
        sentiment_label = "Positive (VADER Aug)"
    elif compound_score <= -0.05:
        sentiment_label = "Negative (VADER Aug)"
    else:
        sentiment_label = "Neutral (VADER Aug)"
    return compound_score, sentiment_label

filipino_positive_keywords = ['magaling', 'mahusay', 'matalino', 'mabait', 'matulungin', 'okay', 'malinaw', 'masaya', 'galing', 'husay', 'saya', 'maayos', 'excellent', 'creative', 'practical', 'professional', 'humble', 'understanding', 'motivating', 'caring', 'cool', 'responsible', 'proficient', 'warm-hearted', 'systematically', 'efficient', 'up to date', 'kind', 'fair', 'considerate', 'beautiful', 'interesting', 'funny', 'awesome', 'best', 'open', 'competent', 'helpful', 'nice', 'approachable']
filipino_negative_keywords = ['hindi', 'di', 'pangit', 'masama', 'mahirap', 'hindi marunong', 'malabo', 'nakakabagot', 'hindi malinaw', 'magulo', 'ayaw', 'fast paced', 'ineffective', 'too much', 'noisy', 'threatening', 'favoritism', 'boring']


def get_filipino_keyword_sentiment(cleaned_text):
    if not isinstance(cleaned_text, str):
        cleaned_text = ""
    score = 0
    positive_matches = []
    negative_matches = []
    tokens = cleaned_text.lower().split()
    for word in filipino_positive_keywords:
        if word in tokens:
            score += 1
            positive_matches.append(word)
    for word in filipino_negative_keywords:
        if word in tokens:
            score -= 1
            negative_matches.append(word)
    if score > 0:
        sentiment_label = "Positive (Filipino Keywords)"
    elif score < 0:
        sentiment_label = "Negative (Filipino Keywords)"
    else:
        sentiment_label = "Neutral (Filipino Keywords)"
    return score, sentiment_label, ", ".join(sorted(set(positive_matches))), ", ".join(sorted(set(negative_matches)))

# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(
    page_title="TeachAIRs",
    page_icon="💽",
    layout="centered",
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
# ------------------------------
# Upload Feedback Dataset
uploaded_file = st.file_uploader("📤 Upload Feedback CSV File", type=["csv"])

if uploaded_file:
    try:
        df_raw = load_csv_with_encodings(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    try:
        cleaned_df = clean_dataframe(df_raw, DEFAULT_FEEDBACK_COLUMN_NAMES)
    except Exception as e:
        st.error(f"Could not identify a feedback text column: {e}")
        st.stop()

    if cleaned_df.empty:
        st.warning("The uploaded file contains no valid feedback rows after initial cleaning.")
        st.stop()

    cleaned_df['Cleaned_Text_Main'] = cleaned_df['Feedback_Text'].apply(lambda x: preprocess_text(x, for_lda=True))
    cleaned_df = cleaned_df[cleaned_df['Cleaned_Text_Main'].astype(str).str.strip() != ''].copy()
    cleaned_df.dropna(subset=['Cleaned_Text_Main'], inplace=True)

    if cleaned_df.empty:
        st.warning("DataFrame is empty after cleaning. Cannot proceed.")
        st.stop()

    cleaned_df[['VADER_Score_Eng', 'VADER_Sentiment_Eng']] = cleaned_df['Feedback_Text'].apply(lambda x: pd.Series(get_vader_sentiment_english(x)))
    cleaned_df[['VADER_Score_Aug', 'VADER_Sentiment_Aug']] = cleaned_df['Cleaned_Text_Main'].apply(lambda x: pd.Series(get_vader_sentiment_augmented(x)))
    cleaned_df[['Filipino_Keyword_Score', 'Filipino_Keyword_Sentiment',
                'Filipino_Positive_Keywords_Found', 'Filipino_Negative_Keywords_Found']] = cleaned_df['Cleaned_Text_Main'].apply(lambda x: pd.Series(get_filipino_keyword_sentiment(x)))

    df = cleaned_df.rename(columns={'Feedback_Text': 'Feedback'}).copy()
    df['Cleaned'] = df['Cleaned_Text_Main']
    df['VADER_Standard'] = df['VADER_Score_Eng']
    df['VADER_Augmented'] = df['VADER_Score_Aug']
    df['Score'] = df.get('VADER_Augmented', df.get('VADER_Standard', 0.0))
    if 'Label' not in df.columns:
        df['Label'] = df['Score'].apply(label_from_score)
    comparison_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download sentiment analysis results",
        comparison_csv,
        file_name=OUTPUT_SENTIMENT_COMPARISON_CSV,
        mime="text/csv"
    )

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
            st.subheader("Evaluate Optimal Number of Topics")
            topic_start = st.number_input("Start number of topics", min_value=2, max_value=12, value=3, step=1)
            topic_end = st.number_input("End number of topics", min_value=topic_start, max_value=12, value=5, step=1)
            topic_step = st.number_input("Step size", min_value=1, max_value=5, value=1, step=1)
            if st.button("Evaluate LDA coherence scores"):
                coherence_values_cv = []
                coherence_values_umass = []
                coherence_values_cnpmi = []
                topic_counts = list(range(topic_start, topic_end + 1, topic_step))
                st.info(f"Evaluating LDA for {topic_start} to {topic_end} topics (step {topic_step})...")
                for num_topics_iter in topic_counts:
                    try:
                        model_iter, dictionary_iter, corpus_iter = train_lda(tokenized, num_topics=num_topics_iter)
                        if model_iter is None or dictionary_iter is None or corpus_iter is None:
                            raise ValueError("LDA model could not be trained for this topic count.")
                        coherencemodel_cv = CoherenceModel(model=model_iter, texts=tokenized, dictionary=dictionary_iter, coherence='c_v')
                        cv_score = coherencemodel_cv.get_coherence()
                        coherencemodel_umass = CoherenceModel(model=model_iter, dictionary=dictionary_iter, corpus=corpus_iter, coherence='u_mass')
                        umass_score = coherencemodel_umass.get_coherence()
                        coherencemodel_cnpmi = CoherenceModel(model=model_iter, texts=tokenized, dictionary=dictionary_iter, coherence='c_npmi')
                        cnpmi_score = coherencemodel_cnpmi.get_coherence()
                        coherence_values_cv.append(cv_score)
                        coherence_values_umass.append(umass_score)
                        coherence_values_cnpmi.append(cnpmi_score)
                        st.write(f"  Coherence for {num_topics_iter} topics: C_v={cv_score:.4f}, UMass={umass_score:.4f}, C_NPMI={cnpmi_score:.4f}")
                    except Exception as e_lda_optim:
                        st.warning(f"Error evaluating {num_topics_iter} topics: {e_lda_optim}. Skipping.")
                        coherence_values_cv.append(np.nan)
                        coherence_values_umass.append(np.nan)
                        coherence_values_cnpmi.append(np.nan)
                if any(not np.isnan(val) for val in coherence_values_cv):
                    coherence_df = pd.DataFrame({
                        "C_v": coherence_values_cv,
                        "UMass": coherence_values_umass,
                        "C_NPMI": coherence_values_cnpmi
                    }, index=topic_counts)
                    coherence_df.index.name = "Num Topics"
                    st.subheader("LDA Coherence by Topic Count")
                    st.line_chart(coherence_df)
                    st.dataframe(coherence_df.style.format("{:.4f}"))
                else:
                    st.warning("No valid coherence scores were generated.")

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

    sentiment_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Sentiment Analysis Results",
        sentiment_csv,
        file_name="sentiment_results.csv",
        mime="text/csv"
    )

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
                    st.download_button(
                        "Download Gemini Recommendations",
                        rec_text,
                        file_name="ai_recommendations.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"AI request failed: {e}")