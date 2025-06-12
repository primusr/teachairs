import streamlit as st
import pandas as pd
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from googletrans import Translator
from gensim import corpora
from gensim.models import LdaModel
from wordcloud import WordCloud
import google.generativeai as genai

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

nltk.download('wordnet')

st.set_page_config(page_title="Sentiment & Topic Analysis", layout="wide")
st.title("📊 Student Feedback Analyzer with AI Recommendations")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
translator = Translator()
vader_eng = SentimentIntensityAnalyzer()

filipino_positive_keywords = ['magaling', 'mahusay', 'matalino', 'mabait']
filipino_negative_keywords = ['hindi', 'pangit', 'masama', 'mahirap']

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

def get_vader_sentiment(text):
    try:
        lang = detect(text)
        if lang != 'en':
            text = translator.translate(text, dest='en').text
    except:
        pass
    score = vader_eng.polarity_scores(text)['compound']
    label = 'Positive' if score > 0.05 else 'Negative' if score < -0.05 else 'Neutral'
    return score, label

def get_filipino_keyword_sentiment(text):
    score = 0
    words = text.split()
    for w in filipino_positive_keywords:
        if w in words: score += 1
    for w in filipino_negative_keywords:
        if w in words: score -= 1
    label = 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'
    return score, label

@st.cache_resource
def configure_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('models/gemini-1.5-flash')
    except:
        return None

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
api_key = st.text_input("🔑 Enter your Gemini API Key (for AI recommendations):", type="password")
gemini_model = configure_gemini(api_key) if api_key else None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty or unreadable.")
        st.stop()
    default_cols = ['feedback', 'comments', 'Comment', 'Strengths']
    feedback_col = next((col for col in default_cols if col in df.columns), df.columns[0])
    df = df[[feedback_col]].rename(columns={feedback_col: "Feedback"})
    df.dropna(subset=['Feedback'], inplace=True)
    st.subheader("📄 Sample Feedback")


    df['Cleaned'] = df['Feedback'].apply(preprocess)
    df[['VADER_Score', 'VADER_Label']] = df['Feedback'].apply(lambda x: pd.Series(get_vader_sentiment(x)))
    df['AUG_VADER_Score'] = df['Cleaned'].apply(lambda x: vader_eng.polarity_scores(x)['compound'])
    df[['Fil_Score', 'Fil_Label']] = df['Cleaned'].apply(lambda x: pd.Series(get_filipino_keyword_sentiment(x)))
    st.dataframe(df[['Feedback', 'Cleaned', 'VADER_Score', 'AUG_VADER_Score', 'VADER_Label', 'Fil_Score', 'Fil_Label']].head())


    st.subheader("✅ Sentiment Results")

    st.markdown("""
**📝 Summary:** This chart visualizes how the sentiment scores of individual comments fluctuate. Higher points indicate positive sentiment, while lower points indicate negative sentiment. This gives you a timeline-style view of emotional tone across the entire feedback dataset.
""")
    vader_counts = df['VADER_Label'].value_counts()
    total = vader_counts.sum()

    positive = vader_counts.get('Positive', 0)
    neutral = vader_counts.get('Neutral', 0)
    negative = vader_counts.get('Negative', 0)

    positive_pct = 100 * positive / total
    neutral_pct = 100 * neutral / total
    negative_pct = 100 * negative / total

    vader_avg_score = df['VADER_Score'].mean()
    summary_text = f"""
    **Augmented VADER (with Filipino Lexicon)**  
    **Methodology:** Avg of Aug VADER scores (on Cleaned_Text_Main). **Score:** {vader_avg_score:.4f}  
    **Interpretation:** Overall sentiment (Aug VADER) is generally {'positive' if vader_avg_score > 0.05 else 'neutral' if vader_avg_score > -0.05 else 'negative'}.  
    **Dominant Category:** Positive (VADER Aug) ({positive}/{total} comments)  
    **Distribution (Aug VADER):**  
    - Positive (VADER Aug): {positive} comments ({positive_pct:.2f}%)  
    - Neutral (VADER Aug): {neutral} comments ({neutral_pct:.2f}%)  
    - Negative (VADER Aug): {negative} comments ({negative_pct:.2f}%)
    """
    st.markdown(summary_text)

    # Add Filipino Keyword Summary
    fil_counts = df['Fil_Label'].value_counts()
    total_fil = fil_counts.sum()

    fil_positive = fil_counts.get('Positive', 0)
    fil_neutral = fil_counts.get('Neutral', 0)
    fil_negative = fil_counts.get('Negative', 0)

    fil_positive_pct = 100 * fil_positive / total_fil
    fil_neutral_pct = 100 * fil_neutral / total_fil
    fil_negative_pct = 100 * fil_negative / total_fil

    fil_avg_score = df['Fil_Score'].mean()
    fil_summary_text = f"""
    **Filipino Keyword Method**  
    **Methodology:** Keyword scoring (positive-negative matches) on preprocessed feedback. **Score:** {fil_avg_score:.4f}  
    **Interpretation:** Overall sentiment (Fil Keywords) is generally {'positive' if fil_avg_score > 0 else 'neutral' if fil_avg_score == 0 else 'negative'}.  
    **Dominant Category:** {'Positive' if fil_positive > fil_neutral and fil_positive > fil_negative else 'Neutral' if fil_neutral >= fil_positive and fil_neutral >= fil_negative else 'Negative'} ({max(fil_positive, fil_neutral, fil_negative)}/{total_fil} comments)  
    **Distribution (Fil Keywords):**  
    - Positive: {fil_positive} comments ({fil_positive_pct:.2f}%)  
    - Neutral: {fil_neutral} comments ({fil_neutral_pct:.2f}%)  
    - Negative: {fil_negative} comments ({fil_negative_pct:.2f}%)
    """
    st.markdown(fil_summary_text)

    st.subheader("📈 Sentiment Analysis of Comments")

    st.markdown("**🔍 Scatterplot: VADER vs AUG_VADER Over Comment Index**")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(df.index, df['VADER_Score'], color='blue', label='VADER Score', alpha=0.6)
    ax.scatter(df.index, df['AUG_VADER_Score'], color='green', label='AUG_VADER Score', alpha=0.6)
    ax.axhline(0, linestyle='--', color='gray')
    ax.set_title('VADER vs AUG_VADER Score Over Comments')
    ax.set_xlabel('Comment Index')
    ax.set_ylabel('Sentiment Score')
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    **📊 VADER vs AUG_VADER Score Comparison**
    This chart compares the standard VADER score (based on original/translated text) with the AUG_VADER score (based on cleaned text).
    """)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['VADER_Score'], label='VADER Score', marker='o', linestyle='-', color='blue', alpha=0.7)
    ax.set_title('VADER Sentiment Score Over Comments')
    ax.set_xlabel('Comment Index')
    ax.set_ylabel('VADER Score')
    ax.axhline(0, linestyle='--', color='gray')
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['AUG_VADER_Score'], label='AUG_VADER Score', marker='x', linestyle='--', color='green', alpha=0.7)
    ax.set_title('AUG_VADER Sentiment Score Over Comments')
    ax.set_xlabel('Comment Index')
    ax.set_ylabel('AUG_VADER Score')
    ax.axhline(0, linestyle='--', color='gray')
    ax.legend()
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['VADER_Score'], label='VADER', marker='o')
    ax.plot(df['Fil_Score'], label='Filipino Keywords', marker='x')
    ax.set_title('Sentiment Analysis Over Comments')
    ax.set_xlabel('Comment Index')
    ax.set_ylabel('Sentiment Score')
    ax.axhline(0, linestyle='--', color='gray')
    ax.legend()
    st.pyplot(fig)

    st.subheader("📊 Visual Summary of Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['VADER_Score'], kde=True, color='blue', label='VADER', ax=ax)
    sns.histplot(df['Fil_Score'], kde=True, color='red', label='Filipino', ax=ax)
    ax.legend()
    st.pyplot(fig)
    st.dataframe(df.head())

    st.download_button("📥 Download Sentiment Results as CSV", data=df.to_csv(index=False), file_name="sentiment_results.csv")

    st.subheader("📊 Sentiment Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='VADER_Label', palette='Set2', ax=ax)
        ax.set_title('VADER Sentiment')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Fil_Label', palette='Set1', ax=ax)
        ax.set_title('Filipino Keyword Sentiment')
        st.pyplot(fig)

    st.subheader("📈 Sentiment Score Comparison")
    fig, ax = plt.subplots()
    ax.scatter(df.index, df['VADER_Score'], label='VADER', alpha=0.7)
    ax.scatter(df.index, df['Fil_Score'], label='Filipino Keywords', alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title('Sentiment Polarity Scores')
    ax.set_xlabel('Comment Index')
    ax.set_ylabel('Polarity Score')
    ax.legend()
    st.pyplot(fig)

    st.subheader("🧠 Topic Modeling with LDA")
    texts = [text.split() for text in df['Cleaned'] if text.strip()]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10, random_state=42)

    doc_topics = [max(lda_model.get_document_topics(bow), key=lambda x: x[1])[0] if len(bow) > 0 else -1 for bow in corpus]
    df = df[df['Cleaned'].str.strip().astype(bool)].reset_index(drop=True)
    df = df.iloc[:len(doc_topics)].copy()
    df['Dominant_Topic'] = doc_topics

    topic_sentiments = []
    for i in range(5):
        topic_subset = df[df['Dominant_Topic'] == i]
        if topic_subset.empty:
            continue
        st.markdown(f"**Topic #{i+1}**")
        words_probs = lda_model.show_topic(i, topn=10)
        words = ', '.join([w for w, _ in words_probs])
        st.write("Top Words:", words)
        fig, ax = plt.subplots()
        wc = WordCloud(background_color='white').fit_words(dict(words_probs))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        topic_subset = df[df['Dominant_Topic'] == i]
        if not topic_subset.empty:
            vader_avg = topic_subset['VADER_Score'].mean()
            fil_avg = topic_subset['Fil_Score'].mean()
            topic_sentiments.append({
                'Topic': f'Topic #{i+1}',
                'Avg VADER Score': round(vader_avg, 2),
                'Avg Filipino Score': round(fil_avg, 2),
                'Comment Count': len(topic_subset)
            })
            st.markdown(f"- Avg VADER Sentiment: **{vader_avg:.2f}**")
            st.markdown(f"- Avg Filipino Keyword Score: **{fil_avg:.2f}**")

        if gemini_model:
            prompt = f"Suggest a 3-5 word label for the following topic words: {words}"
            with st.spinner("Getting label from Gemini..."):
                response = gemini_model.generate_content(prompt)
                st.success(f"Gemini Label: {response.text.strip()}")

    st.subheader("📋 Overall Sentiment Per Topic")
    detailed_topic_df = pd.DataFrame(topic_sentiments)
    if 'Gemini Label' not in detailed_topic_df.columns:
        detailed_topic_df['Gemini Label'] = ''
    if 'Top Keywords' not in detailed_topic_df.columns:
        detailed_topic_df['Top Keywords'] = ''

    for i in range(5):
        topic_subset = df[df['Dominant_Topic'] == i]
        if topic_subset.empty:
            continue
        words_probs = lda_model.show_topic(i, topn=5)
        keywords = ', '.join([w for w, _ in words_probs])
        dist_vader = topic_subset['VADER_Label'].value_counts(normalize=True).apply(lambda x: round(x * 100, 2)).to_dict()
        dist_aug = topic_subset['VADER_Label'].value_counts(normalize=True).apply(lambda x: round(x * 100, 2)).to_dict()
        dist_fil = topic_subset['Fil_Label'].value_counts(normalize=True).apply(lambda x: round(x * 100, 2)).to_dict()

        detailed_topic_df.loc[i, 'Topic ID'] = i
        detailed_topic_df.loc[i, 'Top Keywords'] = keywords
        detailed_topic_df.loc[i, 'Num Comments'] = len(topic_subset)
        detailed_topic_df.loc[i, 'VADER Eng Dist (%)'] = str(dist_vader)
        detailed_topic_df.loc[i, 'VADER Aug Dist (%)'] = str(dist_aug)
        detailed_topic_df.loc[i, 'Fil. Keyword Dist (%)'] = str(dist_fil)

        if gemini_model:
            prompt = f"Suggest a 3-5 word label for the following topic words: {keywords}"
            with st.spinner("Getting Gemini label..."):
                response = gemini_model.generate_content(prompt)
                detailed_topic_df.loc[i, 'AI Label'] = response.text.strip()

    if not detailed_topic_df.empty:
        show_cols = ['Topic ID', 'AI Label', 'Top Keywords', 'Num Comments', 'Avg VADER Score', 'Avg Filipino Score', 'VADER Eng Dist (%)', 'VADER Aug Dist (%)', 'Fil. Keyword Dist (%)']
        st.dataframe(detailed_topic_df[show_cols])
    if topic_sentiments:
        st.dataframe(pd.DataFrame(topic_sentiments))

    st.subheader("💡 Overall AI Recommendations (Gemini)")
    if gemini_model:
        summary = "Here is a summary of the sentiment analysis:\n"
        vader_avg = df['VADER_Score'].mean()
        vader_summary = 'Positive' if vader_avg > 0.05 else 'Negative' if vader_avg < -0.05 else 'Neutral'
        summary += f"\nAverage VADER Sentiment Score: {vader_avg:.2f} ({vader_summary})"
        fil_score_avg = df['Fil_Score'].mean()
        fil_summary = 'Positive' if fil_score_avg > 0 else 'Negative' if fil_score_avg < 0 else 'Neutral'
        summary += f"\nAverage Filipino Keyword Score: {fil_score_avg:.2f} ({fil_summary})"

        topics_summary = "\n\nKey Topics:\n"
        for i in range(5):
            words_probs = lda_model.show_topic(i, topn=5)
            topic_words = ', '.join([w for w, _ in words_probs])
            topics_summary += f"  - Topic {i+1}: {topic_words}\n"
        summary += topics_summary

        prompt = summary + "Based on this analysis, suggest 3–5 actionable teaching recommendations."
        with st.spinner("Generating recommendations..."):
            recs = gemini_model.generate_content(prompt)

        st.subheader("📌 Individual Topic Recommendations")
        for i in range(5):
            topic_subset = df[df['Dominant_Topic'] == i]
            if topic_subset.empty:
                continue
            words_probs = lda_model.show_topic(i, topn=5)
            keywords = ', '.join([w for w, _ in words_probs])
            sentiment_summary = topic_subset['VADER_Label'].value_counts().to_dict()

            topic_prompt = f"Topic {i+1} with keywords: {keywords}Sentiment distribution: {sentiment_summary}Suggest 2–3 actionable teaching strategies for this topic based on sentiment."
            with st.spinner(f"Gemini: Topic {i+1}..."):
                topic_response = gemini_model.generate_content(topic_prompt)
                st.markdown(f"**Topic {i+1}: {keywords}**")
                st.write(topic_response.text.strip())
            st.success("AI Recommendations:")
            st.write(recs.text.strip())
    else:
        st.warning("Enter your Gemini API key above to enable AI recommendations.")

    st.subheader("🗣 Manual Feedback Input for AI Suggestion")
    manual_feedback = st.text_area("Enter a feedback comment:")
    if manual_feedback and gemini_model:
        cleaned_manual = preprocess(manual_feedback)
        vs, vl = get_vader_sentiment(manual_feedback)
        fs, fl = get_filipino_keyword_sentiment(cleaned_manual)

        summary = f"Manual Feedback:\n{manual_feedback}\n\nPreprocessed: {cleaned_manual}\nVADER: {vl} ({vs:.2f})\nFilipino Score: {fl} ({fs})"
        st.markdown(summary)

        prompt = f"Based on this feedback: {manual_feedback}\n\nVADER Sentiment: {vl}, Score: {vs:.2f}\nFilipino Keyword Sentiment: {fl}, Score: {fs}\n\nGive 2-3 teaching suggestions."
        with st.spinner("Generating AI suggestions..."):
            response = gemini_model.generate_content(prompt)
            st.success("Suggestions:")
            st.write(response.text.strip())

else:
    st.info("👈 Please upload a CSV file to begin.")
