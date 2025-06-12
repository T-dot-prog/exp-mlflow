import mlflow
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# set the mlflow track
mlflow.set_tracking_uri('http://localhost:5000')

# load the data
wine = load_wine()

# distribute to X and y 
X = wine.data
y = wine.target

# split into train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# params 
max_depth = 7
n_estimators = 11

# Initialize classifier with named parameters
classifier = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

report = classification_report(y_test, y_pred)
score = accuracy_score(y_test, y_pred)




with mlflow.start_run(experiment_id= 198901936389755012):
    mlflow.log_metric('accuracy', score)

    mlflow.log_params({
        "max_depth": max_depth,
        "n_estimators": n_estimators
    })
    mlflow.log_artifact(__file__)
    
    mlflow.sklearn.log_model(classifier, "random-forest-classifier-model")

    mlflow.set_tags(
        {"Author": 'Tahasin',
         "Project": "Wine-classification"}
    )


    
    