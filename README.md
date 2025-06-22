# Fake News Detection using Machine Learning

Detect whether a news article is fake or real using NLP and ML.

## Tech Stack
- Python, Pandas, Scikit-learn, Streamlit
- Logistic Regression with TF-IDF
- Dataset: Kaggle Fake and Real News

## 📌 Features

- ✅ Classifies news as **Real** or **Fake**
- 💬 Built using **TF-IDF Vectorizer** and **Logistic Regression**
- ⚡ Interactive **Streamlit app**
- 🧠 Trained on real-world datasets (`Fake.csv`, `True.csv`)
- 💾 Model saved with `pickle` for fast deployment  

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Train model: run `notebook/main.ipynb`
4. Run app: `streamlit run app/app.py`

## Demo
Enter a news article in the text area, and the app will tell you if it's fake or real!
