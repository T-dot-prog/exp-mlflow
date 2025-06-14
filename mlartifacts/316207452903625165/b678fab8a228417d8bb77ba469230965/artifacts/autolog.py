import mlflow
import mlflow.sklearn 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,  confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



mlflow.set_tracking_uri('http://127.0.0.1:5000')

wine = load_wine()

X = wine.data
y = wine.target 

X_train , X_test, y_train , y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# set hyperparameter 
max_depth = 10
n_estimators = 5

#autolog the metrcis 
mlflow.autolog()

mlflow.set_experiment("experiment02")

with mlflow.start_run():
    gf = GradientBoostingClassifier(max_depth= max_depth, n_estimators= n_estimators)
    gf.fit(X_train, y_train)

    y_pred = gf.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix-gradient-boost-classifier.png')
    mlflow.log_artifact('confusion_matrix-gradient-boost-classifier.png')

    #log the file 
    mlflow.log_artifact(__file__)

    score = gf.score(y_test, y_pred)

    print(f'Score: {score}')