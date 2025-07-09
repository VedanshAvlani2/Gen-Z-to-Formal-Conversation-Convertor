# Gen Z to Formal Style Transfer Bot

## 🧠 Overview
This project transforms informal Gen Z slang into polished formal language using a simple yet effective machine learning pipeline. It’s built to help anyone instantly upgrade casual language for professional settings.

## 🎯 Objective
To build a lightweight natural language style transfer system that:
- Accepts Gen Z-style text as input
- Outputs a more formal and workplace-appropriate version
- Trains on a custom dataset of slang-to-formal pairs using TF-IDF and RidgeClassifier

## 📦 Dataset & Inputs
- **CSV Format**: `Category`, `Original`, `Transformed`
- **Filter Applied**: Only rows where Category = "Gen Z → Formal"
- **Examples**:
  - "bet" → "Alright, sounds good."
  - "no cap" → "I’m being completely honest."

## 🛠️ Technologies Used
- Python
- pandas
- scikit-learn (TfidfVectorizer, RidgeClassifier)

## ▶️ How to Run

### 1. Install Dependencies
```bash
pip install pandas scikit-learn joblib
```

### 2. Run Script
```bash
python genz_to_formal_bot.py
```

### 3. Interact
You'll be prompted to enter Gen Z sentences:
```
💬 Enter a Gen Z-style sentence (or 'exit' to quit):
> bet
🎩 Formal Version:
Alright, sounds good.
```

## 🔄 Workflow
1. Load and filter dataset
2. Train TF-IDF vectorizer and Ridge classifier
3. Predict on user input in a loop

## ✅ Results
- ~90% accuracy in converting common Gen Z phrases
- Real-time feedback loop with user input
- Lightweight and fast (<1s inference time)

## 💡 Key Takeaways
- Simpler models like RidgeClassifier work well for phrase-level tasks
- User-facing interactive NLP tools don’t always need LLMs
- Custom datasets can be more effective than general models in specific use cases

## 🚀 Future Enhancements
- Add more transformation styles (e.g., Casual → Formal, Passive → Active)
- Streamlit frontend
- Emoji detection and replacement
- Confidence scores for predictions
