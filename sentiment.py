# ==============================
# TeachAIRs: Sentiment & Topic Analysis
# With VADER Method Comparison
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import textwrap

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

# Load NLTK resources
load_nltk()

# ------------------------------
# Streamlit Config
# ------------------------------

st.set_page_config(
    page_title="TeachAIRs",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Developed by Neo under the supervision of the Oracle. Watch this short video for a tutorial on how to use the app:  https://www.youtube.com/shorts/OvRlMiYURhM",
        "Get Help": "https://www.linkedin.com/in/unclebreaker/",
        "Report a bug": "https://www.linkedin.com/in/unclebreaker/"
    }    
  
)

# st.markdown("""
# <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

# <style>
# html, body, [class*="css"]  {
#     font-family: 'Poppins', sans-serif;
# }

# h1, h2, h3, h4, h5, h6 {
#     font-family: 'Poppins', sans-serif;
# }

# .stApp {
#     background-color: #0E1117;
# }
# [data-testid="stSidebar"] {
#     background-color: #161A23;
# }

# /* Responsive tables and code blocks */
# [data-testid="stDataFrame"],
# .stDataFrame {
#     width: 100% !important;
#     max-width: 100% !important;
# }

# [data-testid="stDataFrame"] td,
# [data-testid="stDataFrame"] th,
# .stDataFrame td,
# .stDataFrame th {
#     white-space: normal !important;
#     overflow-wrap: anywhere !important;
#     word-break: break-word !important;
#     max-width: 300px;
# }

# [data-testid="stDataFrame"] div[role="grid"],
# .stDataFrame div[role="grid"] {
#     overflow-x: auto !important;
#     width: 100% !important;
# }

# code,
# pre {
#     white-space: pre-wrap !important;
#     word-break: break-word !important;
#     overflow-wrap: anywhere !important;
#     max-width: 100% !important;
# }

# </style>
# """, unsafe_allow_html=True)

st.title("TeachAIRs: Student Feedback Analyzer with AI Recommendations")

# ------------------------------
# Gemini API (Optional)
# ------------------------------
api_key = st.secrets["api_key"]
#st.text_input("🔑 Enter Gemini API Key (Optional)", type="password") 
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

    df = pd.read_csv(uploaded_file)

    # Auto-detect feedback column
    possible_cols = ["feedback", "comment", "comments", "Feedback"]
    feedback_col = next(
        (c for c in possible_cols if c in df.columns),
        df.columns[0]
    )

    df = df[[feedback_col]].rename(columns={feedback_col: "Feedback"})
    df.dropna(inplace=True)

    # # ------------------------------
    # # Sample VADER Output (Head & Tail) - formatted
    # # ------------------------------
    # st.subheader("Sample VADER Sentiment Output (Head & Tail)")

    # def _format_sentiment_block(series, n=5):
    #     """Return a formatted string showing Original_Text, Cleaned_Text, Sentiment_Score and labels."""
    #     s = series.head(n) if len(series) >= n else series.head(len(series))
    #     df_block = s.to_frame(name="Original_Text")
    #     df_block["Cleaned_Text"] = df_block["Original_Text"].apply(preprocess)
    #     df_block["Sentiment_Score"] = df_block["Original_Text"].apply(get_augmented_vader)
    #     # Preserve original indices in the string representation
    #     table_str = df_block.to_string()
    #     labels_str = "\n\nSentiment\n" + "\n".join(
    #         f"{idx} {label_from_score(score)}" for idx, score in zip(df_block.index, df_block["Sentiment_Score"])    
    #     )
    #     return table_str + labels_str

    # head_block = _format_sentiment_block(df["Feedback"], n=5)
    # tail_block = _format_sentiment_block(df["Feedback"].tail(5), n=5)

    # st.code("Head of the sentiment analysis results:\n" + head_block)
    # st.markdown("---")
    # st.code("Tail of the sentiment analysis results:\n" + tail_block)

    st.divider()
    st.header("Feedback Dataset Overview")
    st.dataframe(df.head(), use_container_width=True)
  
    df["Cleaned"] = df["Feedback"].apply(preprocess)
    df["VADER_Standard"] = df["Feedback"].apply(get_standard_vader)
    df["VADER_Augmented"] = df["Feedback"].apply(get_augmented_vader)
    df["Score"] = df["VADER_Augmented"]
    df["Label"] = df["Score"].apply(label_from_score)
  
    st.divider()
    st.header("Sentiment Distribution (Augmented Model)")
    counts = df["Label"].value_counts()
    # Ensure consistent order
    sentiment_order = ["Positive", "Neutral", "Negative"]
    counts = counts.reindex(sentiment_order, fill_value=0)
    # Define custom colors
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
    avg_score = df["Score"].mean()
    st.markdown(f"""
    **Average Sentiment Score:** {avg_score:.3f}  
    **Overall Sentiment:** {'Positive' if avg_score > 0.05 else 'Negative' if avg_score < -0.05 else 'Neutral'}
    """)

    # ------------------------------
    # Separate Scatter Plots (Color-Coded)
    # ------------------------------
    # ------------------------------
    # Overall System Sentiment Scores & Distributions
    # # ------------------------------
    # st.divider()
    # st.header("Overall System Sentiment Scores & Distributions")
    total_comments = len(df)

    # Standard VADER
    df["Label_Std"] = df["VADER_Standard"].apply(classify_sentiment)
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
    std_sentiment = f"Positive" if std_avg > 0.05 else "Negative" if std_avg < -0.05 else "Neutral"
    aug_sentiment = f"Positive" if aug_avg > 0.05 else "Negative" if aug_avg < -0.05 else "Neutral"

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

    fil_text = f"""-- Filipino Keyword Sentiment (Direct Count) --
Dominant Category: {fil_dominant} (Filipino Keywords) ({fil_counts[fil_dominant]}/{total_comments} comments)
Distribution (Filipino Keywords):
 - Neutral (Filipino Keywords): {fil_counts['Neutral']} comments ({(fil_counts['Neutral'] / total_comments * 100):.2f}%)
 - Positive (Filipino Keywords): {fil_counts['Positive']} comments ({(fil_counts['Positive'] / total_comments * 100):.2f}%)
 - Negative (Filipino Keywords): {fil_counts['Negative']} comments ({(fil_counts['Negative'] / total_comments * 100):.2f}%)
"""

    st.code(
        std_text
        + "\n"
        + "-" * 30
        + "\n"
        + aug_text
        + "\n"
        + "-" * 30)
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
    with col2:
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

    # st.markdown(f"""
    # ### 📈 Statistical Comparison Summary

    # **Pearson Correlation Between Methods:** {correlation:.3f}  
    # **Mean Score Difference (Augmented − Standard):** {mean_difference:.3f}
    # """)

    # # # Statistical comparison
    # # correlation = df["VADER_Standard"].corr(df["VADER_Augmented"])
    # # mean_difference = (df["VADER_Augmented"] - df["VADER_Standard"]).mean()

    sign_flip = (
         (df["VADER_Standard"] > 0) & (df["VADER_Augmented"] < 0)
     ) | (
         (df["VADER_Standard"] < 0) & (df["VADER_Augmented"] > 0)
    )

    flip_rate = sign_flip.mean() * 100

    # # st.markdown(f"""
    # # ### 📈 Statistical Comparison

    # # **Pearson Correlation:** {correlation:.3f}  
    # # **Mean Score Difference (Augmented − Standard):** {mean_difference:.3f}  
    # # **Polarity Sign Flip Rate:** {flip_rate:.2f}%  
    # # """)

    # # ------------------------------
    # # Topic Coherence Evaluation
    # # ------------------------------
    # st.subheader("📈 Topic Coherence Evaluation for Optimal k Selection")

    from gensim.models import CoherenceModel

    # # Prepare data for LDA
    # Ensure no empty cleaned rows
    df = df[df["Cleaned"].str.strip() != ""].reset_index(drop=True)

    texts = df["Cleaned"].apply(lambda x: x.split()).tolist()

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

        coherence_cv = CoherenceModel(
            model=lda_model_k,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        ).get_coherence()
        cv_scores.append(coherence_cv)

   
        coherence_umass = CoherenceModel(
            model=lda_model_k,
            corpus=corpus,
            dictionary=dictionary,
            coherence='u_mass'
        ).get_coherence()
        umass_scores.append(coherence_umass)

   
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

    # # ------------------------------
    # # Plot Line Graphs
    # # ------------------------------
    # fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    # # Top: C_v
    # axes[0].plot(k_values, cv_scores, marker='o')
    # axes[0].set_title("C_v Coherence Scores")
    # axes[0].set_xlabel("Number of Topics (k)")
    # axes[0].set_ylabel("C_v Score")
    # axes[0].axvline(optimal_k, linestyle='--')
    # axes[0].annotate(
    #     f"Peak at k={optimal_k}\n({optimal_cv:.4f})",
    #     xy=(optimal_k, optimal_cv),
    #     xytext=(optimal_k, optimal_cv + 0.02),
    #     arrowprops=dict()
    # )

    # # Middle: UMass
    # axes[1].plot(k_values, umass_scores, marker='o')
    # axes[1].set_title("UMass Coherence Scores")
    # axes[1].set_xlabel("Number of Topics (k)")
    # axes[1].set_ylabel("UMass Score")

    # # Bottom: C_NPMI
    # axes[2].plot(k_values, cnpmi_scores, marker='o')
    # axes[2].set_title("C_NPMI Coherence Scores")
    # axes[2].set_xlabel("Number of Topics (k)")
    # axes[2].set_ylabel("C_NPMI Score")

    # plt.tight_layout()
    # st.pyplot(fig)

    # # ------------------------------
    # # Interpretation Output
    # # ------------------------------
    # st.markdown(f"""
    # ### 📊 Optimal Topic Determination

    # The C_v coherence score reaches its maximum at **k = {optimal_k}**, 
    # with a value of **{optimal_cv:.4f}**, indicating the highest semantic similarity 
    # and interpretability among the generated topics.

    # Based on the strong correlation of C_v with human judgment, the optimal 
    # number of topics was programmatically determined to be:

    # ## ✅ k = {optimal_k}

    # This ensures that subsequent thematic analysis is grounded in the most 
    # semantically coherent topic structure derived from student feedback.
    # """)

    # ------------------------------
    # Topic Modeling
    # ------------------------------

    # ------------------------------
    # Overall Sentiment per Topic (Tabular)
    # ------------------------------
    st.divider()
    st.header("Overall Sentiment per Topic")

    # Ensure final LDA model exists (using optimal_k if computed earlier)
    try:
        final_k = optimal_k
    except:
        final_k = 4  # fallback if not computed

    lda_model_final = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=final_k,
        passes=10,
        random_state=42
    )

    # Assign dominant topic to each comment
    def get_dominant_topic(bow):
        topics = lda_model_final.get_document_topics(bow)
        return max(topics, key=lambda x: x[1])[0]

    df["Topic_ID"] = [get_dominant_topic(bow) for bow in corpus]

    # Convert Filipino label to numeric score
    df["Filipino_Score"] = df["Label_Filipino"].map(SENTIMENT_SCORE_MAP)

    # Prepare results table
    topic_rows = []

    for topic_id in range(final_k):

        topic_df = df[df["Topic_ID"] == topic_id]
        num_comments = len(topic_df)

        if num_comments == 0:
            continue

        # Top Keywords
        words_probs = lda_model_final.show_topic(topic_id, topn=5)
        top_keywords = ", ".join([w for w, _ in words_probs])

        # AI Label (if Gemini available)
        if gemini_model:
            prompt = f"Provide a concise 3-word academic topic label for: {top_keywords}"
            response = gemini_model.generate_content(prompt)
            ai_label = response.text.strip().replace("**", "")
        else:
            ai_label = f"Topic {topic_id}"

        # --------------------------
        # Standard VADER
        # --------------------------
        avg_std = topic_df["VADER_Standard"].mean()
        std_dist = topic_df["Label_Std"].value_counts(normalize=True) * 100
        std_dist = std_dist.reindex(["Positive", "Neutral", "Negative"], fill_value=0)
        std_dist_str = (
            "{"
            f"'Positive (VADER Eng)': {std_dist['Positive']:.1f}, "
            f"'Neutral (VADER Eng)': {std_dist['Neutral']:.1f}, "
            f"'Negative (VADER Eng)': {std_dist['Negative']:.1f}"
            "}"
        )

        # --------------------------
        # Augmented VADER
        # --------------------------
        avg_aug = topic_df["VADER_Augmented"].mean()
        aug_dist = topic_df["Label_Aug"].value_counts(normalize=True) * 100
        aug_dist = aug_dist.reindex(["Positive", "Neutral", "Negative"], fill_value=0)
        aug_dist_str = (
            "{"
            f"'Positive (VADER Aug)': {aug_dist['Positive']:.1f}, "
            f"'Neutral (VADER Aug)': {aug_dist['Neutral']:.1f}, "
            f"'Negative (VADER Aug)': {aug_dist['Negative']:.1f}"
            "}"
        )

        # --------------------------
        # Filipino Keyword
        # --------------------------
        avg_fil = topic_df["Filipino_Score"].mean()
        fil_dist = topic_df["Label_Filipino"].value_counts(normalize=True) * 100
        fil_dist = fil_dist.reindex(["Positive", "Neutral", "Negative"], fill_value=0)
        fil_dist_str = (
            f"Neutral (Filipino Keywords): {fil_dist['Neutral']:.1f}, "
            f"Positive (Filipino Keywords): {fil_dist['Positive']:.1f}, "
            f"Negative (Filipino Keywords): {fil_dist['Negative']:.1f}"
        )

        topic_rows.append({
            "Topic ID": topic_id,
            "AI Label": ai_label,
            "Top Keywords": top_keywords,
            "Num Comments": num_comments,
            "Avg VADER Eng Score": round(avg_std, 2),
            "VADER Eng Dist (%)": (
                f"Positive (VADER Eng): {std_dist['Positive']:.1f}, "
                f"Neutral (VADER Eng): {std_dist['Neutral']:.1f}, "
                f"Negative (VADER Eng): {std_dist['Negative']:.1f}"
            ),
            "Avg VADER Aug Score": round(avg_aug, 2),
            "VADER Aug Dist (%)": (
                f"Positive (VADER Aug): {aug_dist['Positive']:.1f}, "
                f"Neutral (VADER Aug): {aug_dist['Neutral']:.1f}, "
                f"Negative (VADER Aug): {aug_dist['Negative']:.1f}"
            ),
            "Avg Fil. Keyword Score": round(avg_fil, 2),
            "Fil. Keyword Dist (%)": fil_dist_str
        })

    topic_summary_df = pd.DataFrame(topic_rows)

    if not topic_summary_df.empty:
        topic_summary_df = topic_summary_df[
            [
                "Topic ID",
                "AI Label",
                "Top Keywords",
                "Num Comments",
                "Avg VADER Eng Score",
                "VADER Eng Dist (%)",
                "Avg VADER Aug Score",
                "VADER Aug Dist (%)",
                "Avg Fil. Keyword Score",
                "Fil. Keyword Dist (%)",
            ]
        ]
        st.markdown("Overall Sentiment Per Topic")
        st.markdown(
            topic_summary_df.to_html(index=False, escape=False),
            unsafe_allow_html=True,
        )

        sentiment_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Sentiment Analysis Results",
            sentiment_csv,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv"
        )

        # ------------------------------
        # Topic Word Clouds (First 4 LDA Topics)
        # ------------------------------
        topic_ids_to_plot = [row["Topic ID"] for row in topic_rows][:4]
        if topic_ids_to_plot:
            st.divider()
            st.header("Word Cloud for Identified Topics")
            st.markdown(
                "Word clouds for the topics identified by TeachAIRs LDA model. These visualizations complement the quantitative analysis by providing an intuitive understanding of the core concepts and terms that define each theme."
            )
            for i in range(0, len(topic_ids_to_plot), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(topic_ids_to_plot):
                        break
                    topic_idx = topic_ids_to_plot[idx]
                    words_probs = lda_model_final.show_topic(topic_idx, topn=15)
                    words = ", ".join([w for w, _ in words_probs])
                    ai_title = next(
                        (r["AI Label"] for r in topic_rows if r["Topic ID"] == topic_idx),
                        f"Topic {topic_idx}"
                    )
                    with col:
                        wc = WordCloud(background_color="white", width=400, height=300)
                        wc.generate_from_frequencies(dict(words_probs))
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.imshow(wc, interpolation="bilinear")
                        ax.axis("off")
                        metadata_text = f"Topic {topic_idx}: {ai_title}\nKeywords: {words}"
                        wrapped_text = "\n".join(textwrap.wrap(metadata_text, width=45))
                        ax.text(
                            0.02,
                            0.98,
                            wrapped_text,
                            transform=ax.transAxes,
                            fontsize=9,
                            fontweight='bold',
                            color='black',
                            va='top',
                            ha='left',
                            linespacing=1.2,
                            bbox=dict(facecolor='white', alpha=0.88, edgecolor='black', linewidth=0.75, pad=6)
                        )
                        st.pyplot(fig)

    else:
        st.info("No topic sentiment summary available.")

   
    
   

    
    

    # ------------------------------
    # AI Recommendations for Selected Topics
    # ------------------------------
    if topic_summary_df.shape[0] > 0:
        selected_topics = sorted(topic_rows, key=lambda x: x["Avg VADER Aug Score"])[:4]
        st.divider()
        st.header("AI Recommendations for Selected Topics")
        st.markdown(
            "Recommendations for the 4 topic(s) with the most negative average Augmented VADER sentiment."
        )

        def _sentiment_context(avg_score):
            if avg_score > 0.05:
                return f"A VADER score of {avg_score:.2f} suggests a slightly positive sentiment, indicating that the topic is being received fairly well but may still benefit from refinement."
            if avg_score < -0.05:
                return f"A VADER score of {avg_score:.2f} suggests a negative sentiment, indicating that the topic is creating concern or confusion for learners."
            return f"A VADER score of {avg_score:.2f} suggests a neutral sentiment, indicating that the topic is neither strongly praised nor criticized and may need clearer emphasis."

        def _build_topic_recommendations(row):
            topic_label = row['AI Label']
            keywords = row['Top Keywords']
            avg_score = row['Avg VADER Aug Score']
            sentiment_context = _sentiment_context(avg_score)
            rec_1 = f"**1. Improve clarity around {keywords.split(', ')[0]} and related concepts:**"
            rec_2 = f"**2. Reinforce student mastery through targeted examples and checks:**"
            rec_3 = f"**3. Collect and act on short-cycle feedback from learners:**"
            return f"""
Recommendations for Topic {row['Topic ID']} ({topic_label}):
Here are 2-3 actionable teaching recommendations based on the topic \"{topic_label}\".
**Understanding the Context:** {sentiment_context}
**Actionable Teaching Recommendations:**
{rec_1}
   * **Action:** Use the top keywords such as \"{keywords}\" to design brief, focused instruction and real examples.
   * **Rationale:** Anchoring the lesson in familiar terms helps students connect feedback language to the core idea.
   * **Measurement:** Track student questions and comprehension checks for the highlighted concept.
{rec_2}
   * **Action:** Break the topic into smaller lesson chunks and include a quick guided practice or explain-back moment.
   * **Rationale:** Students often respond better when they can see how each part of the topic builds toward mastery.
   * **Measurement:** Observe student confidence in follow-up tasks and the accuracy of responses.
{rec_3}
   * **Action:** Ask learners to share what they found clear or unclear after the lesson and adjust the next class accordingly.
   * **Rationale:** Rapid feedback clarifies whether the teaching approach is aligning with student needs for this topic.
   * **Measurement:** Compare learner feedback and engagement before and after implementing the changes.
"""

        for row in selected_topics:
            st.subheader(f"Topic {row['Topic ID']}: {row['AI Label']}")
            st.markdown(_build_topic_recommendations(row))
            st.markdown("---")


    # ------------------------------
    # AI Recommendations (Optional)
    # ------------------------------
    if gemini_model:
        st.divider()
        st.header("Generating Overall AI Recommendations (based on CSV Analysis)")
        st.subheader("Overall AI Recommendations:")

        # Ensure correlation exists
        try:
            corr_value = f"{correlation:.2f}"
        except:
            corr_value = "Not computed"

        # Include ALL important computed metrics
        summary = f"""
        Average Sentiment Score: {avg_score:.2f}
        Sentiment Distribution: {counts.to_dict()}
        Standard VADER Average: {std_avg:.2f}
        Augmented VADER Average: {aug_avg:.2f}
        Filipino Dominant Sentiment: {fil_dominant}
        Correlation Between Models: {corr_value}
        """

        selected_topics = sorted(topic_rows, key=lambda x: x["Avg VADER Aug Score"])[:4] if topic_rows else []

        def _format_selected_topics(topics):
            if not topics:
                return "No selected topic data is available."
            lines = []
            for row in topics:
                lines.append(
                    f"Topic {row['Topic ID']} - {row['AI Label']}: {row['Top Keywords']} | Avg VADER Aug Score: {row['Avg VADER Aug Score']:.2f} | VADER Aug Dist: {row['VADER Aug Dist (%)']}"
                )
            return "\n".join(lines)

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

        st.download_button(
            "Download Gemini Recommendations",
            gemini_recommendation_text,
            file_name="gemini_recommendations.txt",
            mime="text/plain"
        )

    # # Use markdown to preserve formatting
    # st.markdown(response.text)
    # if gemini_model:
    #     st.divider()
    #     st.header("Generating Overall AI Recommendations (based on CSV Analysis)")
    #     st.subheader("Overall AI Recommendations:")  

    #     # Ensure correlation exists
    #     try:
    #         corr_value = f"{correlation:.2f}"
    #     except:
    #         corr_value = "Not computed"

    #     summary = f"""
    #     Average Sentiment Score: {avg_score:.2f}
    #     Distribution: {counts.to_dict()}
    #     Correlation Between Models: {corr_value}
    #     """

    #     response = gemini_model.generate_content(
    #         summary + "\nGive 3 actionable teaching recommendations."
    #     )

    #     st.write(response.text.strip())

else:
    st.info("Please upload a CSV file to begin.")
    st.divider()
