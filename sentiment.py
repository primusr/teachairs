# ==============================
# TeachAIRs: Sentiment & Topic Analysis
# With VADER Method Comparison
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from wordcloud import WordCloud

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
       'About': "Developed by TheMatrix"
    }
)

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

    st.divider()
    st.header("Feedback Dataset Overview")
    st.dataframe(df.head())
  
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
    st.divider()
    st.header("Sentiment Polarity Distribution Across Methods")
    
    # 1️⃣ Standard VADER Scatter
    st.markdown("## Standard VADER (English Only)")

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
    st.markdown("## Augmented VADER (With Filipino Lexicon)")
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
    # Overall System Sentiment Scores & Distributions
    # ------------------------------
    st.divider()
    st.header("Overall System Sentiment Scores & Distributions")
    total_comments = len(df)

    # ==============================
    # Helper: Label from score
    # ==============================
    # Using classify_sentiment from utils

    # ==============================
    # 1️⃣ Standard VADER
    # ==============================
    df["Label_Std"] = df["VADER_Standard"].apply(classify_sentiment)

    std_avg = df["VADER_Standard"].mean()
    std_counts = df["Label_Std"].value_counts()
    std_counts = std_counts.reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    std_dominant = std_counts.idxmax()

    # ==============================
    # 2️⃣ Augmented VADER
    # ==============================
    df["Label_Aug"] = df["VADER_Augmented"].apply(classify_sentiment)

    aug_avg = df["VADER_Augmented"].mean()
    aug_counts = df["Label_Aug"].value_counts()
    aug_counts = aug_counts.reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    aug_dominant = aug_counts.idxmax()

    # ==============================
    # 3️⃣ Filipino Keyword Direct Count
    # ==============================
    # Using filipino_keyword_sentiment from utilss
    df["Label_Filipino"] = df["Feedback"].apply(filipino_keyword_sentiment)

    fil_counts = df["Label_Filipino"].value_counts()
    fil_counts = fil_counts.reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    fil_dominant = fil_counts.idxmax()

    # ------------------------------
    # DISPLAY RESULTS
    # ------------------------------

    st.markdown("## Standard VADER (English / Translated)")
    st.markdown(f"""
    **Methodology:** Average of Standard VADER scores (on Feedback_Text)  
    **Score:** {std_avg:.4f}  

    **Interpretation:** Overall sentiment (Eng VADER) is generally 
    {'positive' if std_avg > 0.05 else 'negative' if std_avg < -0.05 else 'neutral'}.

    **Dominant Category:** {std_dominant} (VADER Eng) ({std_counts[std_dominant]}/{total_comments} comments)
    """)

    st.markdown("**Distribution (Std VADER):**")
    for label in ["Positive", "Neutral", "Negative"]:
        count = std_counts[label]
        percent = (count / total_comments) * 100
        st.markdown(f"- {label} (VADER Eng): {count} comments ({percent:.2f}%)")

    # ------------------------------

    st.markdown("## Augmented VADER (With Filipino Lexicon)")

    st.markdown(f"""
    **Methodology:** Average of Augmented VADER scores (on Cleaned_Text_Main)  
    **Score:** {aug_avg:.4f}  
    **Interpretation:** Overall sentiment (Aug VADER) is generally 
    {'positive' if aug_avg > 0.05 else 'negative' if aug_avg < -0.05 else 'neutral'}.
    **Dominant Category:** {aug_dominant} (VADER Aug) ({aug_counts[aug_dominant]}/{total_comments} comments)
    """)

    st.markdown("**Distribution (Aug VADER):**")
    for label in ["Positive", "Neutral", "Negative"]:
        count = aug_counts[label]
        percent = (count / total_comments) * 100
        st.markdown(f"- {label} (VADER Aug): {count} comments ({percent:.2f}%)")

    # ------------------------------

    st.markdown("## Filipino Keyword Sentiment (Direct Count)")

    st.markdown(f"""
    **Dominant Category:** {fil_dominant} (Filipino Keywords) ({fil_counts[fil_dominant]}/{total_comments} comments)
    """)

    st.markdown("**Distribution (Filipino Keywords):**")
    for label in ["Positive", "Neutral", "Negative"]:
        count = fil_counts[label]
        percent = (count / total_comments) * 100
        st.markdown(f"- {label} (Filipino Keywords): {count} comments ({percent:.2f}%)")

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
            ai_label = response.text.strip()
        else:
            ai_label = f"Topic {topic_id}"

        # --------------------------
        # Standard VADER
        # --------------------------
        avg_std = topic_df["VADER_Standard"].mean()
        std_dist = topic_df["Label_Std"].value_counts(normalize=True) * 100
        std_dist = std_dist.reindex(["Positive", "Neutral", "Negative"], fill_value=0)
        std_dist_str = f"P:{std_dist['Positive']:.1f}% | N:{std_dist['Neutral']:.1f}% | Neg:{std_dist['Negative']:.1f}%"

        # --------------------------
        # Augmented VADER
        # --------------------------
        avg_aug = topic_df["VADER_Augmented"].mean()
        aug_dist = topic_df["Label_Aug"].value_counts(normalize=True) * 100
        aug_dist = aug_dist.reindex(["Positive", "Neutral", "Negative"], fill_value=0)
        aug_dist_str = f"P:{aug_dist['Positive']:.1f}% | N:{aug_dist['Neutral']:.1f}% | Neg:{aug_dist['Negative']:.1f}%"

        # --------------------------
        # Filipino Keyword
        # --------------------------
        avg_fil = topic_df["Filipino_Score"].mean()
        fil_dist = topic_df["Label_Filipino"].value_counts(normalize=True) * 100
        fil_dist = fil_dist.reindex(["Positive", "Neutral", "Negative"], fill_value=0)
        fil_dist_str = f"P:{fil_dist['Positive']:.1f}% | N:{fil_dist['Neutral']:.1f}% | Neg:{fil_dist['Negative']:.1f}%"

        topic_rows.append({
            #"Topic ID": topic_id,
            "AI Label": ai_label,
            "Top Keywords": top_keywords,
            "Num Comments": num_comments,
            "Avg VADER Eng Score": round(avg_std, 4),
            "VADER Eng Dist (%)": std_dist_str,
            "Avg VADER Aug Score": round(avg_aug, 4),
            "VADER Aug Dist (%)": aug_dist_str,
            "Avg Fil. Keyword Score": round(avg_fil, 4),
            "Fil. Keyword Dist (%)": fil_dist_str
        })

    topic_summary_df = pd.DataFrame(topic_rows)

    st.dataframe(topic_summary_df, use_container_width=True,)
    st.divider()
    st.header("Topic Modeling (LDA)")

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
        st.divider()
        st.subheader("💡AI Teaching Recommendations")

        # Ensure correlation exists
        try:
            corr_value = f"{correlation:.2f}"
        except:
            corr_value = "Not computed"

        summary = f"""
        Average Sentiment Score: {avg_score:.2f}
        Distribution: {counts.to_dict()}
        Correlation Between Models: {corr_value}
        """

        response = gemini_model.generate_content(
            summary + "\nGive 3 actionable teaching recommendations."
        )

        st.write(response.text.strip())

else:
    st.info("Please upload a CSV file to begin.")
    st.divider()
