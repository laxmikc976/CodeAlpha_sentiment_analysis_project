import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex


# Download VADER lexicon (only first time)
nltk.download("vader_lexicon")

# Load dataset
df = pd.read_csv("reviews.csv")
df["date"] = pd.to_datetime(df["date"])  # ensure date format

# --- Sentiment Analysis ---
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(str(text))
    compound = score["compound"]
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["review_text"].apply(get_sentiment)

# --- Emotion Detection ---
def get_emotions(text):
    emotion = NRCLex(str(text))
    return dict(emotion.raw_emotion_scores)

df["Emotions"] = df["review_text"].apply(get_emotions)

# Save results
df.to_csv("reviews_with_sentiment.csv", index=False)
print("✅ Analysis complete! Results saved to reviews_with_sentiment.csv")

# --- Visualization 1: Sentiment Distribution ---
plt.figure(figsize=(6,4))
df["Sentiment"].value_counts().plot(kind="bar", color=["green","red","gray"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# --- Visualization 2: Emotion Distribution ---
emotion_counts = {}
for emotions in df["Emotions"].dropna():
    if isinstance(emotions, str):
        emotions = eval(emotions)
    for emotion, count in emotions.items():
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + count

plt.figure(figsize=(8,4))
plt.bar(emotion_counts.keys(), emotion_counts.values(), color="skyblue")
plt.title("Emotion Distribution")
plt.xlabel("Emotions")
plt.ylabel("Frequency")
plt.show()

# --- Visualization 3: WordCloud ---
text = " ".join([str(e) for e in df["review_text"].dropna()])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud of Reviews")
plt.show()

# --- Visualization 4: Sentiment Trend Over Time ---
trend = df.groupby(["date", "Sentiment"]).size().unstack(fill_value=0)

plt.figure(figsize=(10,5))
for sentiment in ["Positive","Negative","Neutral"]:
    if sentiment in trend.columns:
        plt.plot(trend.index, trend[sentiment], marker="o", label=sentiment)

plt.title("Sentiment Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Reviews")
plt.legend()
plt.grid(True)
plt.show()
