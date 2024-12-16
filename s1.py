import streamlit as st
import yake
import PyPDF2
import re
from sentence_transformers import SentenceTransformer, util

# Load a more relevant model
model = SentenceTransformer('all-MiniLM-L12-v2')

# Function to normalize cosine similarity scores
def normalize_relevance(relevance):
    return (relevance + 1) / 2  # Normalize from -1 to 1 to 0 to 1

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return text

# Function to extract keywords using YAKE
def extract_keywords_with_yake(text, top_n=20):
    # Initialize YAKE keyword extractor
    yake_extractor = yake.KeywordExtractor(lan="en", n=1, top=top_n, features=None)
    keywords = yake_extractor.extract_keywords(text)
    return [keyword[0] for keyword in keywords]

# Function to compare keywords between base and check document
def compare_keywords(base_keywords, check_keywords, check_text):
    base_embeddings = model.encode(base_keywords, convert_to_tensor=True)
    check_embeddings = model.encode(check_text, convert_to_tensor=True)
    
    keyword_comparison = []
    for i, keyword in enumerate(base_keywords):
        presence = keyword in check_keywords
        keyword_embedding = base_embeddings[i].unsqueeze(0)
        relevance = util.cos_sim(keyword_embedding, check_embeddings).max().item()
        normalized_relevance = normalize_relevance(relevance)
        keyword_comparison.append((keyword, presence, normalized_relevance))
    
    return keyword_comparison

# Function to calculate overall relevance
def calculate_overall_relevance(keyword_comparison, threshold=0.3):
    relevant_scores = [relevance for _, presence, relevance in keyword_comparison if relevance > threshold]
    return sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0

# Streamlit app
st.title("Document Topic Checker")

# Base document upload
base_file = st.file_uploader("Upload base document", type="pdf")

if base_file:
    base_text = extract_text_from_pdf(base_file)
    base_text = preprocess_text(base_text)
    base_keywords = extract_keywords_with_yake(base_text)
    st.write("Base document keywords:", base_keywords)
else:
    st.write("Please upload a base document")

# Check document upload
check_file = st.file_uploader("Upload check document", type="pdf")

if base_file and check_file:
    check_text = extract_text_from_pdf(check_file)
    check_text = preprocess_text(check_text)
    
    # Extract keywords from the check document using YAKE
    check_keywords = extract_keywords_with_yake(check_text)
    st.write("Check document keywords:", check_keywords)
    
    # Compare keywords between base and check documents
    keyword_comparison = compare_keywords(base_keywords, check_keywords, check_text)
    
    st.write("Keyword Analysis:")
    for keyword, presence, relevance in keyword_comparison:
        status = "Present" if presence else "Missing"
        st.write(f"- {keyword}: {status} (Relevance: {relevance:.2f})")
        if not presence and relevance < 0.5:
            st.write(f"  Suggestion: Consider including information about '{keyword}' in your document.")
    
    overall_relevance = calculate_overall_relevance(keyword_comparison, threshold=0.3)
    st.write(f"\nOverall document relevance: {overall_relevance:.2f}")

    if overall_relevance == 0:
        st.write("The check document does not include any relevant information from the base document.")
    elif overall_relevance < 0.4:
        st.write("The check document seems to cover some topics, but the relevance is very low. Consider adding more details.")
    elif overall_relevance < 0.7:
        st.write("The check document covers most key topics but could better match the base document's content.")
    else:
        st.write("The check document appears to cover all key topics from the base document well.")
