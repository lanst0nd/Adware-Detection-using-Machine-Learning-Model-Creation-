from flask import Flask, render_template, request
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and preprocess the dataset
file_path = 'cleaned_dataset.csv'  # Update this path accordingly
data = pd.read_csv(file_path)
data['Label'] = data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# Split the dataset
X = data.drop('Label', axis=1)
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Load the models
rf_model = load('random_forest_model.joblib')
dt_model = load('decision_tree_model.joblib')
svm_model = load('svm_model.joblib')
knn_model = load('knn_model.joblib')  # Load the KNN model

# Calculate accuracies
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))
svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))  # Calculate accuracy for KNN

# Function to make predictions
def make_prediction(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    rf_prediction = rf_model.predict(input_df)[0]
    dt_prediction = dt_model.predict(input_df)[0]
    svm_prediction = svm_model.predict(input_df)[0]
    knn_prediction = knn_model.predict(input_df)[0]  # KNN prediction

    total_weight = rf_accuracy + dt_accuracy + svm_accuracy + knn_accuracy
    weighted_sum = (rf_prediction * rf_accuracy) + (dt_prediction * dt_accuracy) + (svm_prediction * svm_accuracy) + (knn_prediction * knn_accuracy)
    combined_prediction = round(weighted_sum / total_weight)

    return {
        "rf": {"prediction": rf_prediction, "accuracy": rf_accuracy},
        "dt": {"prediction": dt_prediction, "accuracy": dt_accuracy},
        "svm": {"prediction": svm_prediction, "accuracy": svm_accuracy},
        "knn": {"prediction": knn_prediction, "accuracy": knn_accuracy},  # Include KNN results
        "combined": combined_prediction,
        "calculation_steps": f"({rf_prediction} * {rf_accuracy}) + ({dt_prediction} * {dt_accuracy}) + ({svm_prediction} * {svm_accuracy}) + ({knn_prediction} * {knn_accuracy}) / {total_weight} = {combined_prediction}"
    }

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = [request.form.get(f'field{i}') for i in range(1, len(X.columns) + 1)]
        input_data = [float(i) for i in input_data]
        predictions = make_prediction(input_data)
        return render_template('index.html', predictions=predictions)
    return render_template('index.html', predictions=None)

if __name__ == '__main__':
    app.run(debug=True)
