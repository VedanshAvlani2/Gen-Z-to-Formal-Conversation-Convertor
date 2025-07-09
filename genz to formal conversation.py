import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
import joblib

# -------------------------------
# 1. Load & Prepare Dataset
# -------------------------------
df = pd.read_csv("genz_to_formal_dataset.csv")

# Filter only Gen Z → Formal samples
df = df[df['Category'].str.lower() == 'gen z → formal']
X = df['Original']
y = df['Transformed']

# -------------------------------
# 2. Train Style Transfer Model
# -------------------------------
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', RidgeClassifier())
])

pipeline.fit(X, y)

# -------------------------------
# 3. User Input for Inference
# -------------------------------
while True:
    user_input = input("\n💬 Enter a Gen Z-style sentence (or 'exit' to quit):\n> ")
    if user_input.strip().lower() == "exit":
        print("👋 Goodbye!")
        break

    prediction = pipeline.predict([user_input])[0]
    print("\n🎩 Formal Version:\n" + prediction)
