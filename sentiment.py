# ==============================
# TeachAIRs: Sentiment & Topic Analysis
# With VADER Method Comparison
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import textwrap
from io import BytesIO
import subprocess
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from wordcloud import WordCloud
import base64
import zipfile

import streamlit.components.v1 as components

# Import WeasyPrint for pristine HTML-to-PDF conversion
from weasyprint import HTML

from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.colors import HexColor, whitesmoke, beige, lightblue, black

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

# Streamlit Config
st.set_page_config(
    page_title="TeachAIRs",
    page_icon="📊",
    layout="wide",
    menu_items={
        "About": "Developed by Neo under the supervision of the Oracle. Watch this short video for a tutorial on how to use the app:  https://www.youtube.com/shorts/OvRlMiYURhM"}    
)

st.title("📊TeachAIRs: Student Feedback Analyzer with AI Recommendations")

# Global CSS to style browser printing alongside WeasyPrint
st.markdown("""
    <style>
    @media print {
        header, footer, [data-testid="stSidebar"], [data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton {
            display: none !important;
        }
        .print-page-break {
            page-break-before: always !important;
            break-before: page !important;
            display: block;
            height: 0px;
        }
    }
    </style>
""", unsafe_allow_html=True)

api_key = st.secrets["api_key"]
gemini_model = configure_gemini(api_key)

# Filipino Lexicon Upload
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

# Upload Feedback Dataset
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
    st.dataframe(df, use_container_width=True)
    df["Cleaned"] = df["Feedback"].apply(preprocess)
    df["VADER_Standard"] = df["Feedback"].apply(get_standard_vader)
    df["VADER_Augmented"] = df["Feedback"].apply(get_augmented_vader)
    df["Score"] = df["VADER_Augmented"]
    df["Label"] = df["Score"].apply(label_from_score)
    st.divider()
    st.header("Sentiment Distribution (Augmented Model)")
    counts = df["Label"].value_counts()
    sentiment_order = ["Positive", "Neutral", "Negative"]
    counts = counts.reindex(sentiment_order, fill_value=0)
    
    color_map = {"Positive": "green", "Neutral": "blue", "Negative": "red"}
    colors = [color_map[label] for label in counts.index]
    fig1, ax1 = plt.subplots()
    counts.plot(kind="bar", ax=ax1, color=colors)
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Sentiment")
    ax1.set_title("Sentiment Distribution")
    fig1.tight_layout()
    fig1_buffer = BytesIO()
    fig1.savefig(fig1_buffer, format='png', bbox_inches='tight')
    fig1_buffer.seek(0)
    st.pyplot(fig1)
    
    avg_score = df["Score"].mean()
    st.markdown(f"""
    **Average Sentiment Score:** {avg_score:.3f}  
    **Overall Sentiment:** {'Positive' if avg_score > 0.05 else 'Negative' if avg_score < -0.05 else 'Neutral'}
    """)
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

    st.code(std_text + "\n" + "-" * 30 + "\n" + aug_text + "\n" + "-" * 30)
    
    st.markdown('<div class="print-page-break"></div>', unsafe_allow_html=True)
    st.divider()
    st.header("Sentiment Polarity Distribution Across Methods")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text("Standard VADER (English Only)")
        colors_std = df["VADER_Standard"].apply(sentiment_color)
        fig_std, ax_std = plt.subplots()
        ax_std.scatter(range(len(df)), df["VADER_Standard"], c=colors_std, alpha=0.7)
        ax_std.axhline(0, linestyle="--")
        ax_std.set_xlabel("Feedback Index")
        ax_std.set_ylabel("Polarity Score")
        ax_std.set_title("Standard VADER Polarity Scores")
        fig_std.tight_layout()
        fig_std_buffer = BytesIO()
        fig_std.savefig(fig_std_buffer, format='png', bbox_inches='tight')
        fig_std_buffer.seek(0)
        st.pyplot(fig_std)

    with col2:
        st.text("Augmented VADER (With Filipino Lexicon)")
        colors_aug = df["VADER_Augmented"].apply(sentiment_color)
        fig_aug, ax_aug = plt.subplots()
        ax_aug.scatter(range(len(df)), df["VADER_Augmented"], c=colors_aug, alpha=0.7)
        ax_aug.axhline(0, linestyle="--")
        ax_aug.set_xlabel("Feedback Index")
        ax_aug.set_ylabel("Polarity Score")
        ax_aug.set_title("Augmented VADER Polarity Scores")
        fig_aug.tight_layout()
        fig_aug_buffer = BytesIO()
        fig_aug.savefig(fig_aug_buffer, format='png', bbox_inches='tight')
        fig_aug_buffer.seek(0)
        st.pyplot(fig_aug)

    correlation = df["VADER_Standard"].corr(df["VADER_Augmented"])
    mean_difference = (df["VADER_Augmented"] - df["VADER_Standard"]).mean()
    sign_flip = ((df["VADER_Standard"] > 0) & (df["VADER_Augmented"] < 0)) | ((df["VADER_Standard"] < 0) & (df["VADER_Augmented"] > 0))
    flip_rate = sign_flip.mean() * 100

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

    st.markdown('<div class="print-page-break"></div>', unsafe_allow_html=True)
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

        avg_std = topic_df["VADER_Standard"].mean()
        std_dist = topic_df["Label_Std"].value_counts(normalize=True) * 100
        std_dist = std_dist.reindex(["Positive", "Neutral", "Negative"], fill_value=0)

        avg_aug = topic_df["VADER_Augmented"].mean()
        aug_dist = topic_df["Label_Aug"].value_counts(normalize=True) * 100
        aug_dist = aug_dist.reindex(["Positive", "Neutral", "Negative"], fill_value=0)

        avg_fil = topic_df["Filipino_Score"].mean()
        fil_dist = topic_df["Label_Filipino"].value_counts(normalize=True) * 100
        fil_dist = fil_dist.reindex(["Positive", "Neutral", "Negative"], fill_value=0)
        fil_dist_str = f"Neutral: {fil_dist['Neutral']:.1f}%, Positive: {fil_dist['Positive']:.1f}%, Negative: {fil_dist['Negative']:.1f}%"

        topic_rows.append({
            "Topic ID": topic_id,
            "AI Label": ai_label,
            "Top Keywords": top_keywords,
            "Num Comments": num_comments,
            "Avg VADER Eng Score": round(avg_std, 2),
            "VADER Eng Dist (%)": f"Pos: {std_dist['Positive']:.1f}%, Neu: {std_dist['Neutral']:.1f}%, Neg: {std_dist['Negative']:.1f}%",
            "Avg VADER Aug Score": round(avg_aug, 2),
            "VADER Aug Dist (%)": f"Pos: {aug_dist['Positive']:.1f}%, Neu: {aug_dist['Neutral']:.1f}%, Neg: {aug_dist['Negative']:.1f}%",
            "Avg Fil. Keyword Score": round(avg_fil, 2),
            "Fil. Keyword Dist (%)": fil_dist_str
        })

    topic_summary_df = pd.DataFrame(topic_rows)

    if not topic_summary_df.empty:
        topic_summary_df = topic_summary_df[[
            "Topic ID", "AI Label", "Top Keywords", "Num Comments",
            "Avg VADER Eng Score", "VADER Eng Dist (%)", "Avg VADER Aug Score",
            "VADER Aug Dist (%)", "Avg Fil. Keyword Score", "Fil. Keyword Dist (%)"
        ]]
        st.markdown("Overall Sentiment Per Topic")
        st.markdown(topic_summary_df.to_html(index=False, escape=False), unsafe_allow_html=True)

        st.markdown('<div class="print-page-break"></div>', unsafe_allow_html=True)
        topic_wordcloud_buffers = []
        topic_ids_to_plot = [row["Topic ID"] for row in topic_rows][:4]
        if topic_ids_to_plot:
            st.divider()
            st.header("Word Cloud for Identified Topics")
            for i in range(0, len(topic_ids_to_plot), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(topic_ids_to_plot):
                        break
                    topic_idx = topic_ids_to_plot[idx]
                    words_probs = lda_model_final.show_topic(topic_idx, topn=10)
                    words = ", ".join([w for w, _ in words_probs])
                    ai_title = next((r["AI Label"] for r in topic_rows if r["Topic ID"] == topic_idx), f"Topic {topic_idx}")
                    with col:
                        st.markdown(f"Keywords: {words}")
                        wc = WordCloud(background_color="white", width=400, height=300)
                        wc.generate_from_frequencies(dict(words_probs))
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.imshow(wc, interpolation="bilinear")
                        ax.axis("off")
                        fig.suptitle(f"Topic {topic_idx}: {ai_title}", fontsize=10, y=0.95)
                        st.pyplot(fig)

                        fig_wc_buffer = BytesIO()
                        fig.savefig(fig_wc_buffer, format='png', bbox_inches='tight')
                        fig_wc_buffer.seek(0)
                        topic_wordcloud_buffers.append((topic_idx, ai_title, fig_wc_buffer))
    else:
        st.info("No topic sentiment summary available.")

    st.markdown('<div class="print-page-break"></div>', unsafe_allow_html=True)
    if topic_summary_df.shape[0] > 0:
        selected_topics = sorted(topic_rows, key=lambda x: x["Avg VADER Aug Score"])[:4]
        st.divider()
        st.header("AI Recommendations for Selected Topics")

        def _sentiment_context(avg_score):
            if avg_score > 0.05:
                return f"A score of {avg_score:.2f} suggests slightly positive sentiment."
            if avg_score < -0.05:
                return f"A score of {avg_score:.2f} suggests negative sentiment."
            return f"A score of {avg_score:.2f} suggests neutral sentiment."

        def _build_topic_recommendations(row):
            topic_label = row['AI Label']
            keywords = row['Top Keywords']
            avg_score = row['Avg VADER Aug Score']
            context = _sentiment_context(avg_score)
            return f"""
<strong>Topic {row['Topic ID']} ({topic_label}):</strong><br>
Context: {context}<br>
• <strong>Action:</strong> Target the keywords "{keywords}" inside custom instruction structures.<br>
• <strong>Rationale:</strong> Concrete application paths minimize comprehension errors.<br>
• <strong>Measurement:</strong> Monitor quiz diagnostics for performance variance.
"""
        for row in selected_topics:
            st.subheader(f"Topic {row['Topic ID']}: {row['AI Label']}")
            st.markdown(_build_topic_recommendations(row), unsafe_allow_html=True)

    st.markdown('<div class="print-page-break"></div>', unsafe_allow_html=True)
    st.divider()
    if gemini_model:
        st.header("Overall AI Recommendations (based on CSV Analysis)")
        try:
            corr_value = f"{correlation:.2f}"
        except:
            corr_value = "Not computed"

        summary = f"Average Sentiment: {avg_score:.2f} | Standard VADER Average: {std_avg:.2f} | Augmented VADER Average: {aug_avg:.2f}"
        selected_topics_text = "\n".join([f"Topic {r['Topic ID']} ({r['AI Label']}): {r['Top Keywords']}" for r in selected_topics])

        structured_prompt = f"Analyze feedback data and provide 3 general summary academic teaching actions based on this info:\n{summary}\nTopics:\n{selected_topics_text}"
        response = gemini_model.generate_content(structured_prompt)
        gemini_recommendation_text = response.text.strip()
        st.markdown(gemini_recommendation_text)

        # ==========================================
        # WEASYPRINT GENERATION IMPLEMENTATION ENGINE
        # ==========================================
        st.divider()
        st.header("📄 Export Analysis Portfolio")

        # Convert state images to Base64 data blocks to decouple filesystems
        def get_b64(bio_buf):
            bio_buf.seek(0)
            return "data:image/png;base64," + base64.b64encode(bio_buf.read()).decode('utf-8')

        img_b64_dist = get_b64(fig1_buffer)
        img_b64_std = get_b64(fig_std_buffer)
        img_b64_aug = get_b64(fig_aug_buffer)

        # Generate custom layout loop structure for word cloud blocks
        wc_html_blocks = ""
        for t_idx, t_title, wc_buf in topic_wordcloud_buffers:
            wc_html_blocks += f"""
            <div class="card col-6">
                <h3>Topic {t_idx}: {t_title}</h3>
                <img src="{get_b64(wc_buf)}" style="width:100%; height:auto;" />
            </div>
            """

        # Generate custom layout loop structure for isolated topic matrices
        rec_html_blocks = ""
        for row in selected_topics:
            rec_html_blocks += f'<div class="callout">{_build_topic_recommendations(row)}</div>'

        # Construct a condensed dataframe layout specifically optimized to fit PDF width profiles
        pdf_topic_df = topic_summary_df.copy()
        pdf_topic_df.columns = [
            "ID", "AI Label", "Top Keywords", "Count",
            "Eng Score", "Eng Dist", "Aug Score", 
            "Aug Dist", "Fil Score", "Fil Dist"
        ]

        # Document Compilation Wrapper String
        html_document_payload = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>TeachAIRs Feedback Portfolio Analysis</title>
            <style>
                @page {{
                    size: letter portrait;
                    margin: 20mm 15mm 20mm 15mm;
                    @bottom-right {{
                        content: "Page " counter(page) " of " counter(pages);
                        font-size: 9pt;
                        color: #666;
                    }}
                }}
                body {{
                    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                    color: #2D3748;
                    line-height: 1.6;
                    font-size: 10.5pt;
                }}
                h1 {{ color: #003366; font-size: 24pt; margin-bottom: 5px; text-align: center; }}
                .subtitle {{ text-align: center; color: #718096; font-size: 11pt; margin-bottom: 30px; }}
                h2 {{ color: #003366; font-size: 16pt; border-bottom: 2px solid #E2E8F0; padding-bottom: 5px; margin-top: 25px; page-break-after: avoid; }}
                h3 {{ color: #2B6CB0; font-size: 12pt; margin-top: 10px; }}
                .page-break {{ page-break-before: always; }}
                
                /* Compact table style to ensure full width matrix fits perfectly inside page walls */
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 15px 0; 
                    font-size: 7.5pt; 
                    page-break-inside: avoid;
                    table-layout: fixed;
                }}
                th {{ 
                    background-color: #003366; 
                    color: white; 
                    padding: 5px 3px; 
                    font-weight: bold; 
                    text-align: left; 
                    word-wrap: break-word;
                    word-break: break-all;
                }}
                td {{ 
                    padding: 5px 3px; 
                    border-bottom: 1px solid #E2E8F0; 
                    vertical-align: top; 
                    word-wrap: break-word;
                    word-break: break-all;
                }}
                tr:nth-child(even) td {{ background-color: #F7FAFC; }}
                
                .metric-row {{ display: flex; justify-content: space-between; margin-bottom: 15px; }}
                .badge {{ background-color: #EBF8FF; color: #2B6CB0; padding: 4px 8px; border-radius: 4px; font-size: 9.5pt; font-weight: bold; }}
                .callout {{ background-color: #FFFAF0; border-left: 4px solid #DD6B20; padding: 12px; margin: 12px 0; border-radius: 4px; page-break-inside: avoid; }}
                .row {{ display: flex; flex-wrap: wrap; margin: 0 -10px; }}
                .col-6 {{ width: 50%; padding: 0 10px; box-sizing: border-box; }}
                .card {{ text-align: center; margin-bottom: 15px; page-break-inside: avoid; }}
                .visual-container {{ text-align: center; margin: 15px 0; page-break-inside: avoid; }}
                .visual-container img {{ max-width: 85%; height: auto; border: 1px solid #E2E8F0; border-radius: 6px; }}
                pre {{ white-space: pre-wrap; background-color: #F7FAFC; padding: 12px; border: 1px solid #E2E8F0; border-radius: 4px; font-size: 9.5pt; font-family: monospace; }}
            </style>
        </head>
        <body>

            <h1>📊 TeachAIRs Analysis Portfolio</h1>
            <div class="subtitle">Comprehensive Analytics & AI Academic Diagnosis Pipeline</div>

            <h2>1. Primary Sentiment Metric Allocations</h2>
            <div class="metric-row">
                <span class="badge">Aggregate Dataset Evaluation: {avg_score:.3f}</span>
                <span class="badge">Data Records Population Vector: {total_comments} rows</span>
            </div>
            <div class="visual-container">
                <img src="{img_b64_dist}" />
            </div>

            <div class="page-break"></div>

            <h2>2. Methodology Framework Polarity Comparison</h2>
            <div class="row">
                <div class="card col-6">
                    <h3>Standard VADER Domain</h3>
                    <img src="{img_b64_std}" style="width:100%;" />
                </div>
                <div class="card col-6">
                    <h3>Augmented Filipino VADER Domain</h3>
                    <img src="{img_b64_aug}" style="width:100%;" />
                </div>
            </div>
            <div class="callout" style="border-left-color: #2B6CB0; background-color: #EBF8FF;">
                <strong>System Transition Correlation Matrix Variance:</strong><br>
                • Pearson Product-Moment Correlation Value: {correlation:.3f}<br>
                • Model Convergence Discrepancy Margin: {mean_difference:.3f}<br>
                • Directional Boundary Sign Deviation Ratio: {flip_rate:.2f}%
            </div>

            <div class="page-break"></div>

            <h2>3. Document Sentiment Matrices Per Topic</h2>
            {pdf_topic_df.to_html(index=False, classes="table", escape=False)}

            <h2>4. Topic Context Lexicon Word Clouds</h2>
            <div class="row">
                {wc_html_blocks}
            </div>

            <div class="page-break"></div>

            <h2>5. Target Action Pedagogical Recommendations</h2>
            {rec_html_blocks}

            <h2>6. Generative AI Comprehensive Synthesis Report</h2>
            <pre>{gemini_recommendation_text}</pre>

        </body>
        </html>
        """

        # Generate the pristine WeasyPrint PDF content
        weasy_pdf_bytes = HTML(string=html_document_payload).write_pdf()

        # Convert the full, un-truncated topic summary dataframe to CSV data
        csv_topic_buffer = BytesIO()
        topic_summary_df.to_csv(csv_topic_buffer, index=False, encoding='utf-8')
        csv_topic_bytes = csv_topic_buffer.getvalue()

        # Convert the full feedback dataset dataframe to CSV data
        csv_full_buffer = BytesIO()
        df.to_csv(csv_full_buffer, index=False, encoding='utf-8')
        csv_full_bytes = csv_full_buffer.getvalue()

        # Compile PDF and both CSV matrices into an in-memory ZIP package
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("TeachAIRs_WeasyPrint_Report.pdf", weasy_pdf_bytes)
            zip_file.writestr("Full_Sentiment_Per_Topic.csv", csv_topic_bytes)
            zip_file.writestr("Full_Feedback_Dataset.csv", csv_full_bytes)
        zip_buffer.seek(0)

        # Unified download layout package delivery channel
        st.download_button(
            label="🚀 Download Complete Portfolio Package (ZIP: PDF + 2 Data CSVs)",
            data=zip_buffer.getvalue(),
            file_name="TeachAIRs_Analysis_Bundle.zip",
            mime="application/zip"
        )
else:
    st.info("Please upload a CSV file to begin.")
    st.divider()