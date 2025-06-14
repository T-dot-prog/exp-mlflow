import mlflow.data
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd 
import mlflow


mlflow.set_tracking_uri('http://127.0.0.1:5000')


data = load_breast_cancer()

X = pd.DataFrame(data.data, columns= data.feature_names)
y = pd.Series(data.target, name= "target")

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size= 0.2 , random_state= 42)

model = RandomForestClassifier(random_state= 42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30]
}


mlflow.set_experiment(experiment_name= 'breat_cancer_classification')

with mlflow.start_run() as parent:
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)

    for i in range(len(grid_search.cv_results_["params"])):

        with mlflow.start_run(nested= True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])
    
    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Log parameters and metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_score", best_score)
    
    #log the train-data 
    train_df = X_train.copy()
    train_df['target'] = y_train

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "training")

    #log the test-data 
    test_df = X_test.copy()
    test_df['target'] = y_test

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "testing")

    #log the artifacts 
    mlflow.log_artifact(__file__)

    #log the model 
    mlflow.sklearn.log_model(grid_search.best_estimator_, 'random_forest')

    # set the tags 
    mlflow.set_tags({
        "Author": "tahasin"
    })

    print(best_params)
    print(best_score)



    