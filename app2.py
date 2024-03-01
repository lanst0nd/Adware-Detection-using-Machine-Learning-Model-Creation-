from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump  

app = Flask(__name__)

# Load and preprocess the dataset
file_path = 'cleaned_dataset.csv'  # Update this path accordingly
data = pd.read_csv(file_path)
data['Label'] = data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

X = data.drop('Label', axis=1)
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train models
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))

svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))

dump(rf_model, 'random_forest_model')
dump(dt_model, 'ecision_tree_model.joblib')
dump(svm_model, 'svm_model.joblib')
dump(knn_model, 'knn_model.joblib')
# Function to make predictions
def make_prediction(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    rf_prediction = rf_model.predict(input_df)[0]
    dt_prediction = dt_model.predict(input_df)[0]
    svm_prediction = svm_model.predict(input_df)[0]

    total_weight = rf_accuracy + dt_accuracy + svm_accuracy
    weighted_sum = (rf_prediction * rf_accuracy) + (dt_prediction * dt_accuracy) + (svm_prediction * svm_accuracy)
    combined_prediction = round(weighted_sum / total_weight)

    return {
        "rf": {"prediction": rf_prediction, "accuracy": rf_accuracy},
        "dt": {"prediction": dt_prediction, "accuracy": dt_accuracy},
        "svm": {"prediction": svm_prediction, "accuracy": svm_accuracy},
        "combined": combined_prediction,
        "calculation_steps": f"({rf_prediction} * {rf_accuracy}) + ({dt_prediction} * {dt_accuracy}) + ({svm_prediction} * {svm_accuracy}) / {total_weight} = {combined_prediction}"
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
