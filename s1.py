import streamlit as st
import os
import PyPDF2
from keybert import KeyBERT
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import re

nltk.download('wordnet')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Normalize text
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# Lemmatize keywords
def lemmatize_keywords(keywords):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(keyword) for keyword in keywords]

# Extract keywords
def extract_keywords(text, model, num_keywords=20):
    text = normalize_text(text)
    keywords = model.extract_keywords(text, top_n=num_keywords)
    lemmatized_keywords = lemmatize_keywords([kw[0] for kw in keywords])
    return lemmatized_keywords

# Process PDFs and generate keywords
def process_pdfs(files, model):
    all_keywords = []
    for pdf_file in files:
        text = extract_text_from_pdf(pdf_file)
        keywords = extract_keywords(text, model)
        all_keywords.extend(keywords)
    return all_keywords

# Compare keywords
def compare_keywords(new_keywords, stored_keywords):
    present_keywords = [word for word in stored_keywords if word in new_keywords]
    missing_keywords = [word for word in stored_keywords if word not in new_keywords]
    return present_keywords, missing_keywords

# Streamlit app
def main():
    st.title("PDF Keyword Generator and Comparator")

    # Part 1: Generate Keywords
    st.header("Generate Keywords from PDFs")
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")

    if st.button("Generate Keywords"):
        if uploaded_files:
            model = KeyBERT()
            keywords = process_pdfs(uploaded_files, model)
            with open('keywords_list.pkl', 'wb') as file:
                pickle.dump(keywords, file)
            st.success("Keywords generated and saved.")
            st.write(f"Extracted Keywords: {keywords}")
        else:
            st.warning("Please upload at least one PDF file.")

    # Part 2: Compare Keywords
    st.header("Compare Keywords with New PDF")
    comparison_file = st.file_uploader("Upload PDF for Comparison", type="pdf")

    if st.button("Compare Keywords"):
        if comparison_file:
            try:
                with open('keywords_list.pkl', 'rb') as file:
                    stored_keywords = pickle.load(file)

                model = KeyBERT()
                text = extract_text_from_pdf(comparison_file)
                new_keywords = extract_keywords(text, model)

                present_keywords, missing_keywords = compare_keywords(new_keywords, stored_keywords)
                
                st.write(f"Extracted Keywords from new PDF: {new_keywords}")
                st.write(f"Stored Keywords: {stored_keywords}")
                st.write(f"Keywords matching in the list: {present_keywords}")
                st.write(f"Keywords missing from the list: {missing_keywords}")

            except FileNotFoundError:
                st.error("No stored keywords found. Please generate keywords first.")
        else:
            st.warning("Please upload a PDF file for comparison.")

if __name__ == "__main__":
    main()
