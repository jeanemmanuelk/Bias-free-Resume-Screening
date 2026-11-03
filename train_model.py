import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import preprocess_data
from src.model_training import train_model, evaluate_model, save_model

# Load dataset
df = pd.read_csv('data/Resume.csv')

# Preprocess data
X, y, vectorizer, label_encoder = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = train_model(X_train, y_train)

# Evaluate model
accuracy, precision, recall = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

# Save model, vectorizer, and label encoder
save_model(model, vectorizer, label_encoder)