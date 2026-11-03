# Bias-free Resume Screening Tool

A small Streamlit app that reads a resume (PDF), cleans the text, and predicts a job category. It started as my MBA capstone project (MBA 590, Fall 2024), "Addressing Algorithmic Bias in AI-Powered Recruitment Systems".

## What it does
1. You upload a resume as a PDF.
2. The app pulls out the text with PyPDF2 and cleans it. The cleaning removes HTML tags, punctuation, and the sensitive keywords gender, race, age, name and address.
3. The text is turned into TF-IDF features (3,000 features).
4. A Random Forest model predicts the job category and the app shows it.

## How the model was trained
- **Data:** the public Kaggle "Resume Dataset" (uploaded by Snehaan Bhawal). Each resume has a job category label. The file is not included here, so download it and save it as `data/Resume.csv` (it needs the columns `Resume_str` and `Category`).
- **Split:** 70% train and 30% test, with a fixed random seed.
- **Class imbalance:** SMOTE on the training data only.
- **Model:** Random Forest with 100 trees.

## Results
On the test split the model reached about 69% accuracy, with weighted precision and recall also near 69%. That is a modest result. Some job categories have few examples, and the model does worse on those.

## Limitations
- The "bias-free" cleaning only deletes a few keywords. It does not remove real names, locations, schools or other details that can still carry sensitive information.
- I discussed fairness measures such as demographic parity and equalized odds in my capstone paper, but this code does not calculate them.
- The app shows only the predicted category. It does not show a confidence score.
- The app runs locally. It is not hosted online.
- It is a learning project and should not be used to make real hiring decisions.

## Run it yourself
Python 3.9 was used.

```bash
git clone https://github.com/jeanemmanuelk/Bias-free-Resume-Screening.git
cd Bias-free-Resume-Screening
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Add the Kaggle file as `data/Resume.csv`, then train the model and start the app:

```bash
python train_model.py
streamlit run main.py
```

Training saves the model, the vectorizer and the label encoder in the `models/` folder, which the app loads when it starts.

## Project layout
```
main.py                  starts the Streamlit app
train_model.py           trains and evaluates the model
train_model.ipynb        notebook version of training
src/data_preprocessing.py   PDF reading, text cleaning, TF-IDF features
src/model_training.py       SMOTE, Random Forest, metrics, saving
src/interface.py            the Streamlit page
```

## Contact
Jean-Emmanuel Kouadio, emmanuelkj5@gmail.com
