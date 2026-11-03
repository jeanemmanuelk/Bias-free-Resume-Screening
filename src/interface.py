import streamlit as st
import joblib
from src.data_preprocessing import clean_text, load_pdf

# Load trained model, vectorizer, and label encoder
model = joblib.load('models/trained_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

def main():
    st.title("Bias-free Resume Screening Tool")
    st.write("Upload a resume in PDF format. The system will process it, remove sensitive information, and predict the job category.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Load and clean text from PDF
        resume_text = load_pdf(uploaded_file)
        cleaned_resume = clean_text(resume_text)
        resume_features = vectorizer.transform([cleaned_resume])
        
        # Predict category
        prediction = model.predict(resume_features)
        predicted_category = label_encoder.inverse_transform(prediction)[0]
        
        st.write(f"Predicted Job Category: {predicted_category}")

if __name__ == "__main__":
    main()