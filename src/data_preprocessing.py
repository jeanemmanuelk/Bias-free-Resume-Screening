import pandas as pd
import re
from PyPDF2 import PdfReader  # Import PdfReader from PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data from PDF using PyPDF2
def load_pdf(uploaded_file):
    text = ""
    pdf_reader = PdfReader(uploaded_file)  # Read directly from the UploadedFile object
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Extract text from each page, handling potential None values
    return text

# Clean text and remove sensitive data
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags if any
    text = re.sub(r'\n', ' ', text)    # Replace new lines with spaces
    text = re.sub(r'\W', ' ', text)    # Remove non-word characters
    text = re.sub(r'\b(?:gender|race|age|name|address)\b', '', text, flags=re.IGNORECASE)  # Remove sensitive keywords
    return text.lower()
# Extract features using TF-IDF and label encoding
def preprocess_data(df):
    df['Resume_str'] = df['Resume_str'].apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df['Resume_str']).toarray()
    le = LabelEncoder()
    y = le.fit_transform(df['Category'])
    return X, y, vectorizer, le