# ==============================
# TeachAIRs: Sentiment & Topic Analysis
# With VADER Method Comparison
# ==============================

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import textwrap
import io
import zipfile
import base64

from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from wordcloud import WordCloud
from weasyprint import HTML

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

GLOBAL_DF = pd.DataFrame()
GLOBAL_FEEDBACK = pd.DataFrame()
GLOBAL_SENTIMENTTOPIC = pd.DataFrame()
GLOBAL_WORDCLOUDS = {}
GLOBAL_TOPIC_RECOMMENDATIONS = []
GLOBAL_OVERALL_AI_RECOMMENDATION = ""
GLOBAL_FIGURES = {}
GLOBAL_SENTIMENT_SUMMARY = ""
GLOBAL_TOPIC_SUMMARY = pd.DataFrame()


def _escape_html(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_recommendation_html(text):
    escaped_text = _escape_html(text)
    escaped_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", escaped_text)

    html_parts = []
    in_list = False
    for line in escaped_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            continue
        if stripped.startswith("* "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{stripped[2:]}</li>")
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<p>{stripped.replace(chr(10), '<br>')}</p>")

    if in_list:
        html_parts.append("</ul>")

    return "".join(html_parts) or "<p>No recommendations available.</p>"


def build_report_pdf():
    feedback_table = GLOBAL_FEEDBACK.to_html(index=False, escape=False) if not GLOBAL_FEEDBACK.empty else "<p>No feedback data available.</p>"
    sentiment_table = GLOBAL_SENTIMENTTOPIC.to_html(index=False, escape=False) if not GLOBAL_SENTIMENTTOPIC.empty else "<p>No sentiment topic data available.</p>" 
    
    preferred_figures = [
        "Sentiment Distribution",
        "Standard VADER Polarity Scores",
        "Augmented VADER Polarity Scores",
    ]
    ordered_figure_names = [
        name for name in preferred_figures if name in GLOBAL_FIGURES
    ] + [
        name for name in GLOBAL_FIGURES if name not in preferred_figures
    ]

    figure_html = ""
    for fig_name in ordered_figure_names:
        image_data = GLOBAL_FIGURES.get(fig_name)
        if image_data:
            figure_html += f"<div class='report-section figure-section'><h3>{fig_name}</h3><img src='{image_data}' style='max-width: 80%; height: auto;'/></div>"

    wordcloud_html = ""
    for topic_id, image_data in sorted(GLOBAL_WORDCLOUDS.items()):
        if image_data:
            wordcloud_html += f"<div class='report-section'><h3>Topic {topic_id}</h3><img src='{image_data}' style='max-width: 80%; height: auto;'/></div>"

    


    recommendations_html = ""
    for rec in GLOBAL_TOPIC_RECOMMENDATIONS:
        recommendations_html += f"<div class='report-section'><h3>Topic {rec.get('id', '')}: {rec.get('label', '')}</h3><div>{_format_recommendation_html(rec.get('text', ''))}</div></div>"

   
    sentiment_summary_html = f"<div class='report-section'><pre>{_escape_html(GLOBAL_SENTIMENT_SUMMARY or 'No sentiment summary available.')}</pre></div>"
    topic_summary_table = GLOBAL_TOPIC_SUMMARY.to_html(index=False, escape=False) if not GLOBAL_TOPIC_SUMMARY.empty else "<p>No topic sentiment summary available.</p>"
    overall_html = f"<div class='report-section overall-recommendations'>{_format_recommendation_html(GLOBAL_OVERALL_AI_RECOMMENDATION or 'No overall AI recommendation generated.')}</div>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset='utf-8'>
        <style>
          body {{ font-family: Arial, sans-serif; padding: 5px; line-height: 1.0, font-size:8px; margin: 5px; }}
          h1, h2 {{ color: #1f4e79; page-break-after: avoid; break-after: avoid; }}
          h2 {{ page-break-after: allow; }}
          table {{ border-collapse: collapse; width: 100%; max-width: 100%; font-size: 8px; table-layout: fixed; word-wrap: break-word; align: center; }}
          th, td {{ border: 1px solid #ccc; padding: 2px; text-align: center; vertical-align: center; overflow-wrap: anywhere; align: center;  }}
          img {{ max-width: 80%; height: auto; display: block; margin: 0 auto; }}
          div, p, li {{ overflow-wrap: anywhere; }}
          pre {{ white-space: pre-wrap; word-wrap: break-word; overflow-wrap: anywhere; }}
          .report-section {{ page-break-inside: avoid; break-inside: avoid; margin-bottom: 8px; }}
          .figure-section {{ page-break-inside: avoid; break-inside: avoid; }}
       
         </style>
      </head>
      <body>
        <h1>TeachAIRs Report</h1>
        <h2>Sentiment Summary</h2>
        {sentiment_summary_html}
        <h2>Overall Sentiment per Topic</h2>
        <div class='report-section'>{topic_summary_table}</div>
        <h2>Figures</h2>
        {figure_html or '<p>No figures available.</p>'}
        <h2>Word Clouds</h2>
        {wordcloud_html or '<p>No word clouds available.</p>'}

      
        <h2>AI Recommendations per Topic</h2>
        {recommendations_html or '<p>No topic recommendations available.</p>'}
        <h2>Overall AI Recommendations</h2>
        {overall_html or '<p>No topic recommendations available.</p>'}
      </body>
    </html>
    """

    pdf_buffer = io.BytesIO()
    HTML(string=html_content).write_pdf(pdf_buffer)
    return pdf_buffer.getvalue()


# Load NLTK resources
load_nltk()

# ------------------------------
# Streamlit Config
# ------------------------------

st.set_page_config(
    page_title="TeachAIRs",
    page_icon="🧊",
    layout="wide",
    menu_items={
        "About": "Developed by Neo under the supervision of the Oracle. Watch this short video for a tutorial on how to use the app:  https://www.youtube.com/shorts/OvRlMiYURhM",
        "Get Help": "https://www.linkedin.com/in/unclebreaker/",
        "Report a bug": "https://www.linkedin.com/in/unclebreaker/"
    }    
)

st.title("📊TeachAIRs: Student Feedback Analyzer with AI Recommendations")

# ------------------------------
# Gemini API (Optional)
# ------------------------------
api_key = st.secrets["api_key"]
gemini_model = configure_gemini(api_key)

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
    GLOBAL_WORDCLOUDS = {}
    GLOBAL_TOPIC_RECOMMENDATIONS = []
    GLOBAL_OVERALL_AI_RECOMMENDATION = ""
    GLOBAL_SENTIMENTTOPIC = pd.DataFrame()
    GLOBAL_FEEDBACK = pd.DataFrame()
    GLOBAL_FIGURES = {}
    GLOBAL_SENTIMENT_SUMMARY = ""

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
    st.dataframe(df, use_container_width=True)
    GLOBAL_DF = df.copy()
    GLOBAL_FEEDBACK = df.copy()

    df["Cleaned"] = df["Feedback"].apply(preprocess)
    df["VADER_Standard"] = df["Feedback"].apply(get_standard_vader)
    df["VADER_Augmented"] = df["Feedback"].apply(get_augmented_vader)
    df["Score"] = df["VADER_Augmented"]
    df["Label"] = df["Score"].apply(label_from_score)
    
    GLOBAL_FEEDBACK = df.copy()
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
    fig_buffer = io.BytesIO()
    fig1.savefig(fig_buffer, format="png", bbox_inches="tight", dpi=150)
    fig_buffer.seek(0)
    GLOBAL_FIGURES["Sentiment Distribution"] = f"data:image/png;base64,{base64.b64encode(fig_buffer.read()).decode()}"
    plt.close(fig1)
    avg_score = df["Score"].mean()
    st.markdown(f"""
    **Average Sentiment Score:** {avg_score:.3f}  
    **Overall Sentiment:** {'Positive' if avg_score > 0.05 else 'Negative' if avg_score < -0.05 else 'Neutral'}
    """)

    total_comments = len(df)

    # Standard VADER
    df["Label_Std"] = df["VADER_Standard"].apply(classify_sentiment)
    st.divider() # Make sure chart/metrics have structural padding
    std_avg = df["VADER_Standard"].mean()
    std_counts = df["Label_Std"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    std_dominant = std_counts.idxmax()

    # Augmented VADER
    df["Label_Aug"] = df["VADER_Augmented"].apply(classify_sentiment)
    aug_avg = df["VADER_Augmented"].mean()
    aug_counts = df["Label_Aug"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    aug_dominant = aug_counts.idxmax()

    # Filipino Keywords
    df["Label_Filipino"] = df["Feedback"].apply(filipino_keyword_sentiment)
    fil_counts = df["Label_Filipino"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    fil_dominant = fil_counts.idxmax()

    # Display formatted summaries
    std_sentiment = "Positive" if std_avg > 0.05 else "Negative" if std_avg < -0.05 else "Neutral"
    aug_sentiment = "Positive" if aug_avg > 0.05 else "Negative" if aug_avg < -0.05 else "Neutral"

    std_text = f"""Standard VADER (English/Translated)
Methodology: Avg of Std VADER scores (on Feedback_Text). Score: {std_avg:.4f}
Interpretation: Overall sentiment (Eng VADER) is generally {std_sentiment}.
Dominant Category: {std_dominant} (VADER Eng) ({std_counts[std_dominant]}/{total_comments} comments)
Distribution (Std VADER):
 - Positive (VADER Eng): {std_counts['Positive']} comments ({(std_counts['Positive'] / total_comments * 100):.2f}%)
 - Neutral (VADER Eng): {std_counts['Neutral']} comments ({(std_counts['Neutral'] / total_comments * 100):.2f}%)
 - Negative (VADER Eng): {std_counts['Negative']} comments ({(std_counts['Negative'] / total_comments * 100):.2f}%)
"""

    aug_text = f"""Augmented VADER (with Filipino Lexicon)
Methodology: Avg of Aug VADER scores (on Cleaned_Text_Main). Score: {aug_avg:.4f}
Interpretation: Overall sentiment (Aug VADER) is generally {aug_sentiment}.
Dominant Category: {aug_dominant} (VADER Aug) ({aug_counts[aug_dominant]}/{total_comments} comments)
Distribution (Aug VADER):
 - Positive (VADER Aug): {aug_counts['Positive']} comments ({(aug_counts['Positive'] / total_comments * 100):.2f}%)
 - Neutral (VADER Aug): {aug_counts['Neutral']} comments ({(aug_counts['Neutral'] / total_comments * 100):.2f}%)
 - Negative (VADER Aug): {aug_counts['Negative']} comments ({(aug_counts['Negative'] / total_comments * 100):.2f}%)
"""

    GLOBAL_SENTIMENT_SUMMARY = std_text + "\n" + "-" * 30 + "\n" + aug_text + "\n" + "-" * 30
    st.code(GLOBAL_SENTIMENT_SUMMARY)
    

    # ------------------------------
    # Sentiment Polarity Distribution Across Methods (plots)
    # ------------------------------
    st.divider()
    st.header("Sentiment Polarity Distribution Across Methods")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Standard VADER (English Only)")
        colors_std = df["VADER_Standard"].apply(sentiment_color)
        fig_std, ax_std = plt.subplots()
        ax_std.scatter(range(len(df)), df["VADER_Standard"], c=colors_std, alpha=0.7)
        ax_std.axhline(0, linestyle="--")
        ax_std.set_xlabel("Feedback Index")
        ax_std.set_ylabel("Polarity Score")
        ax_std.set_title("Standard VADER Polarity Scores")
        st.pyplot(fig_std)
        fig_std_buffer = io.BytesIO()
        fig_std.savefig(fig_std_buffer, format="png", bbox_inches="tight", dpi=150)
        fig_std_buffer.seek(0)
        GLOBAL_FIGURES["Standard VADER Polarity Scores"] = f"data:image/png;base64,{base64.b64encode(fig_std_buffer.read()).decode()}"
        plt.close(fig_std)

    with col2:
        st.markdown("### Augmented VADER (With Filipino Lexicon)")
        colors_aug = df["VADER_Augmented"].apply(sentiment_color)
        fig_aug, ax_aug = plt.subplots()
        ax_aug.scatter(range(len(df)), df["VADER_Augmented"], c=colors_aug, alpha=0.7)
        ax_aug.axhline(0, linestyle="--")
        ax_aug.set_xlabel("Feedback Index")
        ax_aug.set_ylabel("Polarity Score")
        ax_aug.set_title("Augmented VADER Polarity Scores")
        st.pyplot(fig_aug)
        fig_aug_buffer = io.BytesIO()
        fig_aug.savefig(fig_aug_buffer, format="png", bbox_inches="tight", dpi=150)
        fig_aug_buffer.seek(0)
        GLOBAL_FIGURES["Augmented VADER Polarity Scores"] = f"data:image/png;base64,{base64.b64encode(fig_aug_buffer.read()).decode()}"
        plt.close(fig_aug)

    # Statistical Comparison Calculations
    correlation = df["VADER_Standard"].corr(df["VADER_Augmented"])
    df = df[df["Cleaned"].str.strip() != ""].reset_index(drop=True)

   

    texts = df["Cleaned"].apply(lambda x: x.split()).tolist()
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    k_values = list(range(3, 11))
    cv_scores = []

    for k in k_values:
        lda_model_k = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=10, random_state=42)
        coherence_cv = CoherenceModel(model=lda_model_k, texts=texts, dictionary=dictionary, coherence='c_v').get_coherence()
        cv_scores.append(coherence_cv)

    optimal_index = cv_scores.index(max(cv_scores))
    optimal_k = k_values[optimal_index]

    # ------------------------------
    # Topic Modeling & Summary Dataframe Construction
    # ------------------------------
    st.divider()
    st.header("Overall Sentiment per Topic")

    try:
        final_k = optimal_k
    except:
        final_k = 4

    lda_model_final = LdaModel(corpus=corpus, id2word=dictionary, num_topics=final_k, passes=10, random_state=42)

    def get_dominant_topic(bow):
        topics = lda_model_final.get_document_topics(bow)
        return max(topics, key=lambda x: x[1])[0]

    df["Topic_ID"] = [get_dominant_topic(bow) for bow in corpus]
    df["Filipino_Score"] = df["Label_Filipino"].map(SENTIMENT_SCORE_MAP)

    GLOBAL_SENTIMENTTOPIC = df.copy()
    topic_rows = []
    for topic_id in range(final_k):
        topic_df = df[df["Topic_ID"] == topic_id]
        num_comments = len(topic_df)
        if num_comments == 0:
            continue

        words_probs = lda_model_final.show_topic(topic_id, topn=5)
        top_keywords = ", ".join([w for w, _ in words_probs])

        if gemini_model:
            prompt = f"Provide a concise 3-word academic topic label for: {top_keywords}"
            response = gemini_model.generate_content(prompt)
            ai_label = response.text.strip().replace("**", "")
        else:
            ai_label = f"Topic {topic_id}"

        std_dist = topic_df["Label_Std"].value_counts(normalize=True) * 100
        std_dist = std_dist.reindex(["Positive", "Neutral", "Negative"], fill_value=0)

        aug_dist = topic_df["Label_Aug"].value_counts(normalize=True) * 100
        aug_dist = aug_dist.reindex(["Positive", "Neutral", "Negative"], fill_value=0)

        fil_dist = topic_df["Label_Filipino"].value_counts(normalize=True) * 100
        fil_dist = fil_dist.reindex(["Positive", "Neutral", "Negative"], fill_value=0)

        topic_rows.append({
            "Topic ID": topic_id,
            "AI Label": ai_label,
            "Top Keywords": top_keywords,
            "Num Comments": num_comments,
            "Avg VADER Eng Score": round(topic_df["VADER_Standard"].mean(), 2),
            "VADER Eng Dist (%)": f"Positive: {std_dist['Positive']:.1f}, Neutral: {std_dist['Neutral']:.1f}, Negative: {std_dist['Negative']:.1f}",
            "Avg VADER Aug Score": round(topic_df["VADER_Augmented"].mean(), 2),
            "VADER Aug Dist (%)": f"Positive: {aug_dist['Positive']:.1f}, Neutral: {aug_dist['Neutral']:.1f}, Negative: {aug_dist['Negative']:.1f}",
            "Avg Fil. Keyword Score": round(topic_df["Filipino_Score"].mean(), 2),
            "Fil. Keyword Dist (%)": f"Positive: {fil_dist['Positive']:.1f}, Neutral: {fil_dist['Neutral']:.1f}, Negative: {fil_dist['Negative']:.1f}"
        })
    
   
    topic_summary_df = pd.DataFrame(topic_rows)
    GLOBAL_TOPIC_SUMMARY = topic_summary_df.copy()

    pre_generated_wordclouds = {}

    if not topic_summary_df.empty:
        # st.markdown("Overall Sentiment Per Topic")
        st.markdown(topic_summary_df.to_html(index=False, escape=False), unsafe_allow_html=True)

        # ------------------------------
        # Topic Word Clouds Rendering Loop
        # ------------------------------
        topic_ids_to_plot = [row["Topic ID"] for row in topic_rows][:4]
        if topic_ids_to_plot:
            st.divider()
            st.header("Word Cloud for Identified Topics")
            st.markdown("Word clouds for the topics identified by TeachAIRs LDA model...")
            
            for i in range(0, len(topic_ids_to_plot), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(topic_ids_to_plot):
                        break
                    topic_idx = topic_ids_to_plot[idx]
                    words_probs = lda_model_final.show_topic(topic_idx, topn=15)
                    words = ", ".join([w for w, _ in words_probs])
                    ai_title = next((r["AI Label"] for r in topic_rows if r["Topic ID"] == topic_idx), f"Topic {topic_idx}")
                    
                    with col:
                        #st.markdown(f'<div style="text-align: center;"><b>Topic {topic_idx}: {ai_title}</b><br>Keywords: {words}</div>', unsafe_allow_html=True)
                        
                        wc = WordCloud(background_color="white", width=400, height=300)
                        wc.generate_from_frequencies(dict(words_probs))
                        fig, ax = plt.subplots(figsize=(6, 4))
                        fig.text(
                            0.5,
                            0.97,
                            f"Topic {topic_idx}: {ai_title}\nKeywords: {words}",
                            ha="center",
                            va="top",
                            fontsize=5,
                            wrap=True,
                        )
                        fig.subplots_adjust(top=0.82)
                        ax.imshow(wc, interpolation="bilinear")
                        ax.axis("off")
                        st.pyplot(fig)
                        
                        # Cache the exact base64 image representation right now to prevent regeneration tasks later
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                        buf.seek(0)
                        image_data = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
                        pre_generated_wordclouds[topic_idx] = image_data
                        plt.close(fig)

                        GLOBAL_WORDCLOUDS[topic_idx] = image_data
    else:
        st.info("No topic sentiment summary available.")

    # ------------------------------
    # AI Recommendations per Topic Rendering Loop
    # ------------------------------
    pre_generated_topic_recs = []

    if topic_summary_df.shape[0] > 0:
        selected_topics = sorted(topic_rows, key=lambda x: x["Avg VADER Aug Score"])[:4]
        st.divider()
        st.header("AI Recommendations for Selected Topics")
        st.markdown("Recommendations for the 4 topic(s) with the most negative average Augmented VADER sentiment.")

        def _sentiment_context(avg_score):
            if avg_score > 0.05:
                return f"A VADER score of {avg_score:.2f} suggests a slightly positive sentiment, indicating that the topic is being received fairly well but may still benefit from refinement."
            if avg_score < -0.05:
                return f"A VADER score of {avg_score:.2f} suggests a negative sentiment, indicating that the topic is creating concern or confusion for learners."
            return f"A VADER score of {avg_score:.2f} suggests a neutral sentiment, indicating that the topic is neither strongly praised nor criticized."

        def _build_topic_recommendations(row):
            topic_label = row['AI Label']
            keywords = row['Top Keywords']
            avg_score = row['Avg VADER Aug Score']
            sentiment_context = _sentiment_context(avg_score)
            rec_1 = f"**1. Improve clarity around {keywords.split(', ')[0]} and related concepts:**"
            rec_2 = f"**2. Reinforce student mastery through targeted examples and checks:**"
            rec_3 = f"**3. Collect and act on short-cycle feedback from learners:**"
            return f"""Recommendations for Topic {row['Topic ID']} ({topic_label}):
Here are 2-3 actionable teaching recommendations based on the topic "{topic_label}".
**Understanding the Context:** {sentiment_context}
**Actionable Teaching Recommendations:**
{rec_1}
   * **Action:** Use the top keywords such as "{keywords}" to design brief, focused instruction and real examples.
   * **Rationale:** Anchoring the lesson in familiar terms helps students connect feedback language to the core idea.
   * **Measurement:** Track student questions and comprehension checks for the highlighted concept.
{rec_2}
   * **Action:** Break the topic into smaller lesson chunks and include a quick guided practice or explain-back moment.
   * **Rationale:** Students often respond better when they can see how each part of the topic builds toward mastery.
   * **Measurement:** Observe student confidence in follow-up tasks and the accuracy of responses.
{rec_3}
   * **Action:** Ask learners to share what they found clear or unclear after the lesson and adjust the next class accordingly.
   * **Rationale:** Rapid feedback clarifies whether the teaching approach is aligning with student needs for this topic.
   * **Measurement:** Compare learner feedback and engagement before and after implementing the changes."""

        for row in selected_topics:
            st.subheader(f"Topic {row['Topic ID']}: {row['AI Label']}")
            rec_text = _build_topic_recommendations(row)
            st.markdown(rec_text)
            st.markdown("---")
            
            # Cache the text blocks so they transfer natively to the PDF template
            pre_generated_topic_recs.append({
                "id": row['Topic ID'],
                "label": row['AI Label'],
                "text": rec_text
            })

    GLOBAL_TOPIC_RECOMMENDATIONS = pre_generated_topic_recs

    # ------------------------------
    # Overall AI Recommendations (Gemini Executive Action Strategy)
    # ------------------------------
    gemini_recommendation_text = ""
    if gemini_model:
        st.header("Overall AI Recommendations (based on CSV Analysis)")
        
        try:
            corr_value = f"{correlation:.2f}"
        except:
            corr_value = "Not computed"

        summary = f"Average Sentiment Score: {avg_score:.2f}\nSentiment Distribution: {counts.to_dict()}\nStandard VADER Average: {std_avg:.2f}\nAugmented VADER Average: {aug_avg:.2f}\nFilipino Dominant Sentiment: {fil_dominant}\nCorrelation Between Models: {corr_value}"
        selected_topics = sorted(topic_rows, key=lambda x: x["Avg VADER Aug Score"])[:4] if topic_rows else []

        def _format_selected_topics(topics):
            if not topics:
                return "No selected topic data is available."
            return "\n".join([
                f"Topic {row['Topic ID']} - {row['AI Label']}: {row['Top Keywords']} | Avg VADER Aug Score: {row['Avg VADER Aug Score']:.2f} | VADER Aug Dist: {row['VADER Aug Dist (%)']}"
                for row in topics
            ])

        selected_topics_text = _format_selected_topics(selected_topics)

        structured_prompt = f"""
You are an academic assistant analyzing student feedback data.
Generate teaching recommendations for the selected topics using the requested structured format.

STRICT FORMAT FOR EACH SELECTED TOPIC:
- Provide a clear topic title line.
- Provide one summary sentence describing the context from sentiment and keywords.
- Provide exactly 2-3 actionable teaching recommendations.
- Each recommendation block MUST include:
  * Action
  * Rationale
  * Measurement

OUTPUT FORMAT FOR EACH TOPIC:
Recommendations for Topic <N> (<Topic Label>):
Here are 2-3 actionable teaching recommendations based on the topic "<Topic Label>".
**Understanding the Context:** <context sentence>
**Actionable Teaching Recommendations:**
1. <recommendation text>
   * **Action:** ...
   * **Rationale:** ...
   * **Measurement:** ...
2. <recommendation text>
   * **Action:** ...
   * **Rationale:** ...
   * **Measurement:** ...

DATA SUMMARY:
{summary}

SELECTED TOPICS:
{selected_topics_text}
"""

        response = gemini_model.generate_content(structured_prompt)
        gemini_recommendation_text = response.text.strip()
        st.markdown(gemini_recommendation_text)
        GLOBAL_OVERALL_AI_RECOMMENDATION = gemini_recommendation_text

    if not GLOBAL_FEEDBACK.empty and not GLOBAL_SENTIMENTTOPIC.empty:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            # archive.writestr("02_Feedback.csv", GLOBAL_FEEDBACK.to_csv(index=False).encode("utf-8"))
            archive.writestr("03_Sentiment.csv", GLOBAL_SENTIMENTTOPIC.to_csv(index=False).encode("utf-8"))
            archive.writestr("01_Output.pdf", build_report_pdf())
        st.subheader("Download Report Package")
        st.download_button(
            "Download Report ZIP",
            zip_buffer.getvalue(),
            file_name="TeachAIRs_Report_Package.zip",
            mime="application/zip",
        )
    else:
        st.info("Upload and analyze a dataset to enable the report download.")

else:
    st.info("Please upload a CSV file to begin.")
    st.divider()