import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Set your DAGsHub credentials
os.environ['MLFLOW_TRACKING_USERNAME'] = 'T-dot-prog' 
os.environ['MLFLOW_TRACKING_PASSWORD'] = '1ee33c16bb54a7f0a5aefc523277430895b24f83'  

# Initialize DAGsHub
import dagshub
dagshub.init(repo_owner='T-dot-prog', repo_name='mlflow', mlflow=True)

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 8
n_estimators = 5

# Set experiment name
mlflow.set_experiment('Wine-Classification')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Log artifacts
    mlflow.log_artifact("E:/mlflow/Confusion-matrix.png")
    mlflow.log_artifact(__file__)

    # Set tags
    mlflow.set_tags({
        "Author": 'Vikash',
        "Project": "Wine Classification",
        "Model": "Random Forest",
        "Dataset": "Wine"
    })

    # Log the model
    mlflow.sklearn.log_model(rf, "Random-Forest-Model")

    print(f"Model accuracy: {accuracy:.4f}")