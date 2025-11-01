# Step 1: Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load Dataset
# Download from Kaggle: Sentiment140 Dataset
# https://www.kaggle.com/datasets/kazanova/sentiment140
print("📥 Loading dataset...")
df = pd.read_csv('training.1600000.processed.noemoticon.csv.zip', encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ['polarity', 'text']

# Step 3: Keep Only Positive and Negative Sentiments
df = df[df.polarity != 2]
df['polarity'] = df['polarity'].map({0: 0, 4: 1})
print("✅ Positive & Negative tweets filtered:", df['polarity'].value_counts().to_dict())

# Step 4: Clean Tweets
def clean_text(text):
    return text.lower()

df['clean_text'] = df['text'].apply(clean_text)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['polarity'],
    test_size=0.2,
    random_state=42
)
print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

# Step 6: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("🔢 TF-IDF Shapes — Train:", X_train_tfidf.shape, " Test:", X_test_tfidf.shape)

# Step 7: Bernoulli Naive Bayes
print("\n🧠 Training Bernoulli Naive Bayes Model...")
bnb = BernoulliNB()
bnb.fit(X_train_tfidf, y_train)
bnb_pred = bnb.predict(X_test_tfidf)
print("Bernoulli Naive Bayes Accuracy:", accuracy_score(y_test, bnb_pred))
print("\nBernoulliNB Classification Report:\n", classification_report(y_test, bnb_pred))

# Step 8: Support Vector Machine (SVM)
print("\n🧠 Training SVM Model...")
svm = LinearSVC(max_iter=1000)
svm.fit(X_train_tfidf, y_train)
svm_pred = svm.predict(X_test_tfidf)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_pred))

# Step 9: Logistic Regression
print("\n🧠 Training Logistic Regression Model...")
logreg = LogisticRegression(max_iter=100)
logreg.fit(X_train_tfidf, y_train)
logreg_pred = logreg.predict(X_test_tfidf)
print("Logistic Regression Accuracy:", accuracy_score(y_test, logreg_pred))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, logreg_pred))

# Step 10: Test on Sample Tweets
sample_tweets = ["I love this!", "I hate that!", "It was okay, not great."]
sample_vec = vectorizer.transform(sample_tweets)

print("\n💬 Sample Predictions:")
print("BernoulliNB:", bnb.predict(sample_vec))
print("SVM:", svm.predict(sample_vec))
print("Logistic Regression:", logreg.predict(sample_vec))

print("\n✅ Sentiment Analysis Completed Successfully!")
