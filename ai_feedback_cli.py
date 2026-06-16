"""AI Teacher Feedback CLI
Simple command-line tool to run sentiment + optional LDA on a CSV feedback file.
Usage: python ai_feedback_cli.py path/to/feedback.csv [--lexicon path/to/lexicon.csv] [--topics 4]
"""
import argparse
import pandas as pd
from utils import preprocess, get_standard_vader, get_augmented_vader, train_lda, get_topic_keywords, update_vader_lexicon


def main():
    parser = argparse.ArgumentParser(description="Process feedback CSV")
    parser.add_argument("csv", help="Path to feedback CSV")
    parser.add_argument("--lexicon", help="Optional Filipino lexicon CSV to augment VADER")
    parser.add_argument("--topics", type=int, default=0, help="Number of LDA topics to train (0 to skip)")
    parser.add_argument("--out", default="sentiment_results.csv", help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, header=0)
    # guess text column
    text_col = next((c for c in ["feedback","comment","comments","Feedback","Comment"] if c in df.columns), df.columns[0])
    df = df[[text_col]].rename(columns={text_col: 'Feedback'})
    df.dropna(inplace=True)

    if args.lexicon:
        success, msg = update_vader_lexicon(args.lexicon)
        print(msg)

    df['Cleaned'] = df['Feedback'].apply(preprocess)
    df['VADER_Standard'] = df['Feedback'].apply(get_standard_vader)
    df['VADER_Augmented'] = df['Feedback'].apply(get_augmented_vader)

    if args.topics and args.topics > 0:
        tokenized = df['Cleaned'].apply(lambda x: x.split()).tolist()
        tokenized = [t for t in tokenized if t]
        if tokenized:
            lda_model, dictionary, corpus = train_lda(tokenized, num_topics=args.topics)
            if lda_model:
                topics = get_topic_keywords(lda_model, topn=8)
                print("Top keywords per topic:")
                for tid, kws in topics.items():
                    print(f"Topic {tid}: {', '.join(kws)}")
        else:
            print("No tokenized texts for LDA.")

    df.to_csv(args.out, index=False)
    print(f"Results saved to {args.out}")


if __name__ == '__main__':
    main()
