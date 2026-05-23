##  Fake News Detection using Natural Language Processing (NLP)
Fake News Detection using Natural Language Processing and Machine Learning to classify news as real or fake.

This project is a Fake News Detection System built using Natural Language Processing (NLP) and Machine Learning. It classifies news articles as real or fake based on text content analysis.

The system processes raw text, extracts important features using TF-IDF, and applies a trained ML model to make predictions.


## Objective

The main objective of this project is to:

   Detect fake news automatically
   
   Reduce misinformation on digital platforms
   
   Improve trust in online news sources



## Technologies Used
Python 

Pandas

NumPy

Scikit-learn

Natural Language Processing (NLP)

TF-IDF Vectorizer

Jupyter Notebook  

## Workflow
Data Collection (News dataset)

Data Preprocessing (cleaning, stopwords removal)

Feature Extraction using TF-IDF

Model Training (Logistic Regression / Naive Bayes)

Prediction (Real or Fake News)

## Machine Learning Model

Algorithm: Logistic Regression (or Naive Bayes)

Input: News text

Output: Real / Fake classification


## How to Run the Project
1. Clone the repository

git clone https://github.com/Hanumakshi/fake-news-detection-nlp.git

2. Install dependencies
   
pip install -r requirements.txt

3. Run the notebook

Open Jupyter Notebook and run:

Fake_News_Detection.ipynb


## Project Structure

fake-news-detection-nlp/
│
├── Fake_News_Detection.ipynb
├── dataset.csv
├── model.pkl
├── requirements.txt
└── README.md

##  Results

The model successfully classifies news articles with good accuracy using NLP-based feature extraction.

<img width="1025" height="860" alt="image" src="https://github.com/user-attachments/assets/b691ef0d-b74e-4e5f-84ef-580e2dcc8201" />


## Future Improvements

Deploy using Streamlit / Flask web app

Improve accuracy with deep learning (LSTM, BERT)

Real-time news API integration


## Conclusion

This project demonstrates how Machine Learning and NLP can be used to detect fake news effectively and help in reducing misinformation online.



