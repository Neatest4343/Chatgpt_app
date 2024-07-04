import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define a function to preprocess text
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Function to compute cosine similarity between two vectors
def cosine_similarity_vector(v1, v2):
    return cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]

# Function to get the most relevant sentence to a query
def get_most_relevant_sentence(query, text):
    query = preprocess(query)
    sentences = nltk.sent_tokenize(text)
    preprocessed_sentences = [preprocess(sentence) for sentence in sentences]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    query_tfidf = vectorizer.transform([query])
    similarities = []
    for i in range(len(sentences)):
        similarity = cosine_similarity_vector(tfidf_matrix[i], query_tfidf)
        similarities.append((sentences[i], similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    most_relevant_sentence = similarities[0][0]
    return most_relevant_sentence

# Function to generate a response based on the query
def chatbot(query, text):
    relevant_sentence = get_most_relevant_sentence(query, text)
    response = f"The most relevant sentence to your query is:\n'{relevant_sentence}'"
    return response

# Streamlit app
def main():
    st.title("Pride and Prejudice Chatbot")
    st.write("Welcome to the Pride and Prejudice chatbot! Ask a question about the novel and get relevant answers.")

    # Read the text file
    with open('pride_and_prejudice.txt.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    # User input for question
    user_question = st.text_input("Ask a question:")

    if user_question:
        try:
            bot_response = chatbot(user_question, text)
            st.write("User:", user_question)
            st.write("Chatbot:", bot_response)
        except Exception as e:
            st.write("An error occurred:", e)

if __name__ == "__main__":
    main()
