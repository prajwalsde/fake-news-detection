# Fake News Detection using Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange)
![Platform](https://img.shields.io/badge/Platform-MacOS%20%7C%20Linux%20%7C%20Windows-informational)
![GitHub Repo stars](https://img.shields.io/github/stars/prajwalsde/fake-news-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/prajwalsde/fake-news-detection?style=social)

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
