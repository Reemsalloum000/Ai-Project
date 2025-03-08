import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import tkinter as tk
from tkinter import Text, font, Entry, Label
from tkinter import ttk  

# Function to preprocess data and train a model.
def train_and_evaluate_classifier(classifier, X_train, X_test, y_train, y_test, label_encoders, class_label):
    # Train the classifier.
    classifier.fit(X_train, y_train)

    # Make predictions on the test set.
    y_pred = classifier.predict(X_test)

    # Determine the encoded label for the positive class.
    positive_label = label_encoders['Diagnosis'].transform([class_label])[0]

    # Evaluate metrics.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=positive_label, average='binary')
    recall = recall_score(y_test, y_pred, pos_label=positive_label, average='binary')

    return accuracy, precision, recall

# Read data from Excel file.
file_path = "DiabetesDataset.xlsx"
df = pd.read_excel(file_path)

# Create a LabelEncoder for each categorical column.
label_columns = ['Gender', 'FamilyHistory', 'PhysicalActivity', 'Symptoms', 'Region', 'Diagnosis']
label_encoders = {}

for col in label_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target.
X = df.drop(['PatientID', 'Diagnosis'], axis=1)
y = df['Diagnosis']

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to balance the dataset.
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Create Decision Tree classifier.
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train and evaluate the Decision Tree model.
dt_classifier.fit(X_train_resampled, y_train_resampled)
y_pred_dt = dt_classifier.predict(X_test_scaled)

# Evaluate metrics for Decision Tree.
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(y_test, y_pred_dt, pos_label=label_encoders['Diagnosis'].transform(['Positive'])[0], average='binary')
dt_recall = recall_score(y_test, y_pred_dt, pos_label=label_encoders['Diagnosis'].transform(['Positive'])[0], average='binary')

# Create ANN classifier.
ann_classifier = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=5000, random_state=42)
ann_classifier.fit(X_train_scaled, y_train)
y_pred_ann = ann_classifier.predict(X_test_scaled)

# Evaluate metrics for ANN.
ann_accuracy = accuracy_score(y_test, y_pred_ann)
ann_precision = precision_score(y_test, y_pred_ann, pos_label=label_encoders['Diagnosis'].transform(['Positive'])[0], average='binary')
ann_recall = recall_score(y_test, y_pred_ann, pos_label=label_encoders['Diagnosis'].transform(['Positive'])[0], average='binary')

# Function to perform analysis using Decision Tree and display results in a new window.
def analysis_decision_tree():
    # Create a new window.
    results_window = tk.Toplevel(root)
    results_window.title("Decision Tree Analysis Results")
    results_window.geometry("400x200")

    # Create a Text widget to display results.
    text_widget = Text(results_window, wrap="word", font=font.Font(family="Times New Roman", size=12))

    # Insert results into the Text widget
    text_widget.insert(tk.END, f"Accuracy: {dt_accuracy:.2f}\n")
    text_widget.insert(tk.END, f"Precision: {dt_precision:.2f}\n")
    text_widget.insert(tk.END, f"Recall: {dt_recall:.2f}\n")

    # Disable editing in the Text widget.
    text_widget.config(state=tk.DISABLED)

    # Pack the Text widget.
    text_widget.pack()

# Function to perform analysis using ANN and display results in a new window.
def analysis_ann():
    ann_classifier = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=5000, random_state=42)
    ann_classifier.fit(X_train_scaled, y_train)

    ann_accuracy, ann_precision, ann_recall = train_and_evaluate_classifier(
        ann_classifier, X_train_scaled, X_test_scaled, y_train, y_test, label_encoders, 'Positive'
    )

    # Create a new window.
    results_window = tk.Toplevel(root)
    results_window.title("ANN Analysis Results")
    results_window.geometry("400x200")

    # Create a Text widget to display results.
    text_widget = Text(results_window, wrap="word", font=font.Font(family="Times New Roman", size=12))

    # Insert results into the Text widget
    text_widget.insert(tk.END, f"Accuracy: {ann_accuracy:.2f}\n")
    text_widget.insert(tk.END, f"Precision: {ann_precision:.2f}\n")
    text_widget.insert(tk.END, f"Recall: {ann_recall:.2f}\n")

    # Disable editing in the Text widget.
    text_widget.config(state=tk.DISABLED)

    # Pack the Text widget.
    text_widget.pack()

# Function to open a window for Diabetes prediction.
def open_diabetes_prediction_window():
    diabetes_prediction_window = tk.Toplevel(root)
    diabetes_prediction_window.title("Diabetes Prediction")
    diabetes_prediction_window.geometry("500x500")

    # Label and Entry for Patient ID.
    label_patient_id = Label(diabetes_prediction_window, text="Patient ID:")
    label_patient_id.grid(row=0, column=0, padx=10, pady=5)
    Patient_ID = Entry(diabetes_prediction_window)
    Patient_ID.grid(row=0, column=1, padx=10, pady=5)

    # Label and Entry for Age.
    label_age = Label(diabetes_prediction_window, text="Age:")
    label_age.grid(row=1, column=0, padx=10, pady=5)
    Age = Entry(diabetes_prediction_window)
    Age.grid(row=1, column=1, padx=10, pady=5)

    # Label and ComboBox for Gender.
    label_gender = Label(diabetes_prediction_window, text="Gender:")
    label_gender.grid(row=2, column=0, padx=10, pady=5)
    Gender = tk.StringVar()
    Gender.set("")
    combo_gender = ttk.Combobox(diabetes_prediction_window, textvariable=Gender, values=['Male', 'Female'])
    combo_gender.grid(row=2, column=1, padx=10, pady=5)

    # Label and Entry for BMI.
    label_bmi = Label(diabetes_prediction_window, text="BMI:")
    label_bmi.grid(row=3, column=0, padx=10, pady=5)
    BMI = Entry(diabetes_prediction_window)
    BMI.grid(row=3, column=1, padx=10, pady=5)

    # Label and Entry for Blood Pressure.
    label_bp = Label(diabetes_prediction_window, text="Blood Pressure:")
    label_bp.grid(row=4, column=0, padx=10, pady=5)
    BloodPressure = Entry(diabetes_prediction_window)
    BloodPressure.grid(row=4, column=1, padx=10, pady=5)

    # Label and Entry for Fasting Glucose.
    label_fg = Label(diabetes_prediction_window, text="Fasting Glucose:")
    label_fg.grid(row=5, column=0, padx=10, pady=5)
    FastingGlucose = Entry(diabetes_prediction_window)
    FastingGlucose.grid(row=5, column=1, padx=10, pady=5)

    # Label and ComboBox for Family History.
    label_family_history = Label(diabetes_prediction_window, text="Family History:")
    label_family_history.grid(row=6, column=0, padx=10, pady=5)
    FamilyHistory = tk.StringVar()
    FamilyHistory.set("")
    combo_family_history = ttk.Combobox(diabetes_prediction_window, textvariable=FamilyHistory, values=['Yes', 'No'])
    combo_family_history.grid(row=6, column=1, padx=10, pady=5)

    # Label and ComboBox for Physical Activity.
    label_physical_activity = Label(diabetes_prediction_window, text="Physical Activity:")
    label_physical_activity.grid(row=7, column=0, padx=10, pady=5)
    PhysicalActivity = tk.StringVar()
    PhysicalActivity.set("")
    combo_physical_activity = ttk.Combobox(diabetes_prediction_window, textvariable=PhysicalActivity, values=['Low', 'Medium', 'High'])
    combo_physical_activity.grid(row=7, column=1, padx=10, pady=5)

    # Label and ComboBox for Region.
    label_region = Label(diabetes_prediction_window, text="Region:")
    label_region.grid(row=8, column=0, padx=10, pady=5)
    Region = tk.StringVar()
    Region.set("")
    combo_region = ttk.Combobox(diabetes_prediction_window, textvariable=Region, values=['Al-Bireh', 'Ramallah', 'Beituniya'])
    combo_region.grid(row=8, column=1, padx=10, pady=5)

    # Label and Entry for Symptoms.
    label_symptoms = Label(diabetes_prediction_window, text="Symptoms:")
    label_symptoms.grid(row=9, column=0, padx=10, pady=5)
    Symptoms = ttk.Combobox(diabetes_prediction_window, values=['Yes', 'No'])
    Symptoms.grid(row=9, column=1, padx=10, pady=5)

    # Function to predict diabetes based on inputs.
    def predict_diabetes():
        # Collect inputs from the user interface.
        patient_data = {
            'Age': int(Age.get()),
            'Gender': Gender.get(),
            'BMI': float(BMI.get()),
            'BloodPressure': int(BloodPressure.get()),
            'FastingGlucose': int(FastingGlucose.get()),
            'FamilyHistory': FamilyHistory.get(),
            'PhysicalActivity': PhysicalActivity.get(),
            'Region': Region.get(),
            'Symptoms': Symptoms.get()
        }

        # Encode categorical values and prepare for prediction.
        patient_df = pd.DataFrame([patient_data])
        for col in ['Gender', 'FamilyHistory', 'PhysicalActivity', 'Region', 'Symptoms']:
            patient_df[col] = label_encoders[col].transform(patient_df[col])

        # Reorder columns to match training data
        patient_df = patient_df[X.columns]

        # Scale the numeric values.
        patient_scaled = scaler.transform(patient_df)

        # Make prediction using Decision Tree.
        prediction_dt = dt_classifier.predict(patient_scaled)[0]
        result_dt = label_encoders['Diagnosis'].inverse_transform([prediction_dt])[0]

        # Make prediction using ANN.
        prediction_ann = ann_classifier.predict(patient_scaled)[0]
        result_ann = label_encoders['Diagnosis'].inverse_transform([prediction_ann])[0]

        # Display the result in a popup.
        result_window = tk.Toplevel(diabetes_prediction_window)
        result_window.title("Prediction Result")
        result_label = Label(result_window, text=f"Decision Tree Prediction: {result_dt}\nANN Prediction: {result_ann}", font=("Times New Roman", 14))
        result_label.pack(padx=20, pady=20)

    # Button for prediction.
    button_predict_diabetes = tk.Button(diabetes_prediction_window, text="Predict Diabetes Result", command=predict_diabetes)
    button_predict_diabetes.grid(row=10, column=0, columnspan=2, pady=10)

# Create GUI.
root = tk.Tk()
root.title("Diabetes Analysis")
root.geometry("400x300")

# Welcome labels.
welcome_label = tk.Label(root, text="Welcome to Diabetes Project!", font=("Times New Roman", 14))
welcome_label.pack(pady=(20, 10)) 

subtitle_label = tk.Label(root, text="Predict Diabetes using Machine Learning", font=("Times New Roman", 12))
subtitle_label.pack(pady=(0, 10))

# Button to perform Decision Tree analysis.
button_decision_tree = tk.Button(root, text="Analysis using Decision Tree", command=analysis_decision_tree)
button_decision_tree.pack(pady=10)

# Button to perform ANN analysis.
button_ann = tk.Button(root, text="Analysis using ANN", command=analysis_ann)
button_ann.pack(pady=10)

# Button for Diabetes prediction.
button_diabetes_prediction = tk.Button(root, text="Diabetes Prediction", command=open_diabetes_prediction_window)
button_diabetes_prediction.pack(pady=10)

root.mainloop()