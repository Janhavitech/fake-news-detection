# 📰 Fake News Detection using NLP & Machine Learning

A machine learning web app that classifies news articles as **Fake** or **Real**, using TF-IDF text vectorization and a Logistic Regression classifier — achieving ~92% accuracy on test data.

## 🎯 Problem It Solves
Misinformation spreads quickly online. This project applies NLP techniques to flag news articles that show text patterns typical of fake news, trained on a labeled dataset of real and fake articles.

## 📊 Dataset
This project uses the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle (~45,000 articles total — real articles from Reuters, fake articles from sources flagged by fact-checkers).

> **Note:** the raw dataset files (`Fake.csv`, `True.csv`) are not included in this repo because of their size (~110 MB combined). Download them from the Kaggle link above and place them in a `data/` folder if you want to re-run the training notebook.

## 🛠️ Tech Stack
- **Python** – pandas, numpy
- **scikit-learn** – TF-IDF vectorization, Logistic Regression
- **Streamlit** – web app interface
- **joblib** – model serialization

## ⚙️ Approach / Pipeline
1. Load and merge real (`True.csv`) and fake (`Fake.csv`) articles, labeling them 1/0
2. Clean text: lowercase, remove punctuation/special characters and stopwords
3. Convert cleaned text into numerical features using **TF-IDF**
4. Train a **Logistic Regression** classifier on the vectorized text
5. Evaluate accuracy on a held-out test set (~92%)
6. Save the trained model and vectorizer as `.pkl` files for reuse
7. Build a Streamlit app where a user pastes any news text and gets an instant Fake/Real prediction

## 📁 Project Structure
```
fake-news-detection/
├── app.py                       # Streamlit app
├── nlp.ipynb                    # Full training notebook (EDA, cleaning, TF-IDF, model training)
├── models/
│   ├── fake_news_model.pkl      # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl     # Fitted TF-IDF vectorizer
├── requirements.txt
└── README.md
```

## 🚀 Setup & Installation
1. Clone the repository
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app
   ```bash
   streamlit run app.py
   ```
4. Paste any news article text into the app and click **Analyze** to see the prediction.

## 📈 Results
- Model: Logistic Regression with TF-IDF features
- Accuracy: ~92% on test data

## 🔮 Future Improvements
- Compare against other models (Random Forest, XGBoost, BERT-based classifiers)
- Add explainability — highlight which words influenced the prediction
- Extend to detect fake news directly from a news article URL

## 👩‍💻 Author
Janhavi — Data Science graduate, NLP & Machine Learning projects.
