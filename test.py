import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
dataset = pd.read_csv('indian_liver_patient.csv')

# Preprocess the dataset
dataset['Gender'] = dataset['Gender'].map({'Female': 0, 'Male': 1})
dataset['Dataset'] = dataset['Dataset'].map({1: 1, 2: 0})
dataset = dataset.dropna()

# Split the dataset into features and target variable
X = dataset.drop('Dataset', axis=1)
y = dataset['Dataset']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Create a Tkinter window
window = tk.Tk()
window.title("Liver Disease Prediction")
window.geometry("500x500")
window.configure(bg='#f2f2f2')

# Create a frame for the input fields
input_frame = tk.Frame(window, bg='#f2f2f2')
input_frame.pack(pady=20)

# Create labels and input fields for the features
features = [
    ("Age", "age_entry"),
    ("Gender", "gender_entry"),
    ("Total Bilirubin", "total_bilirubin_entry"),
    ("Direct Bilirubin", "direct_bilirubin_entry"),
    ("Alkaline Phosphotase", "alkaline_phosphotase_entry"),
    ("Alamine Aminotransferase", "alamine_aminotransferase_entry"),
    ("Aspartate Aminotransferase", "aspartate_aminotransferase_entry"),
    ("Total Proteins", "total_proteins_entry"),
    ("Albumin", "albumin_entry"),
    ("Albumin Globulin Ratio", "albumin_globulin_ratio_entry")
]

for feature, entry_name in features:
    label = tk.Label(input_frame, text=feature+":", bg='#f2f2f2')
    label.grid(row=features.index((feature, entry_name)), column=0, padx=10, pady=5, sticky="w")
    entry = tk.Entry(input_frame, width=10, bd=2, relief="groove")
    entry.grid(row=features.index((feature, entry_name)), column=1, padx=10, pady=5)
    locals()[entry_name] = entry

def predict():
    # Get the input values
    age = float(age_entry.get())
    gender = float(gender_entry.get())
    total_bilirubin = float(total_bilirubin_entry.get())
    direct_bilirubin = float(direct_bilirubin_entry.get())
    alkaline_phosphotase = float(alkaline_phosphotase_entry.get())
    alamine_aminotransferase = float(alamine_aminotransferase_entry.get())
    aspartate_aminotransferase = float(aspartate_aminotransferase_entry.get())
    total_proteins = float(total_proteins_entry.get())
    albumin = float(albumin_entry.get())
    albumin_globulin_ratio = float(albumin_globulin_ratio_entry.get())


    # Create a feature vector from the input values
    features = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Total_Bilirubin': [total_bilirubin],
        'Direct_Bilirubin': [direct_bilirubin],
        'Alkaline_Phosphotase': [alkaline_phosphotase],
        'Alamine_Aminotransferase': [alamine_aminotransferase],
        'Aspartate_Aminotransferase': [aspartate_aminotransferase],
        'Total_Protiens': [total_proteins],
        'Albumin': [albumin],
        'Albumin_and_Globulin_Ratio': [albumin_globulin_ratio]
    })

    # Make the prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    # Show the prediction in a message box
    if prediction[0] == 1:
        messagebox.showinfo("Prediction", f"You are likely to have liver disease. Probability: {probability[0][1]}")
    else:
        messagebox.showinfo("Prediction", f"You are unlikely to have liver disease. Probability: {probability[0][0]}")


# Create a button to trigger the prediction
predict_button = tk.Button(window, text="Predict", command=predict, bg='#4caf50', fg='white', bd=0, padx=10, pady=5)
predict_button.pack(pady=10)

# Start the GUI event loop
window.mainloop()
