from sklearn.model_selection import GridSearchCV

param_grids = {
    "Logistic_Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
        "class_weight": [None, "balanced"],
    },
    "MultinomialNB": {"alpha": [0.001, 0.01, 0.1, 1]},
    "GaussianNB": {},
    "BernoulliNB": {"alpha": [0.001, 0.01, 0.1, 1]},
    "RandomForestClassifier": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True, False],
        "class_weight": [None, "balanced", "balanced_subsample"],
    },
    "GradientBoostingClassifier": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "DecisionTreeClassifier": {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "SVC": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear", "poly"],
        "gamma": ["scale", "auto"],
        "degree": [3, 4, 5],
        "class_weight": [None, "balanced"],
    },
    "Perceptron": {
        "penalty": [None, "l2", "l1", "elasticnet"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "fit_intercept": [True, False],
        "max_iter": [1000, 1500, 2000],
    },
    "SGDClassifier": {
        "penalty": ["l2", "l1", "elasticnet"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "loss": ["hinge", "log", "modified_huber"],
        "fit_intercept": [True, False],
        "max_iter": [1000, 1500, 2000],
    },
}


def tune_hyperparameters(model, param_grid, X_train, y_train, cv):
    """
    This function takes a model instance, a parameter grid specific to the model,
    and training data to perform hyperparameter tuning using cross-validation with k=5.

    :param model: The model instance to be tuned.
    :param param_grid: The parameter grid specific to the model.
    :param X_train: Training features.
    :param y_train: Training labels.
    :param cv: Number of folds for cross-validation.
    :return: The tuned model instance.
    """
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {type(model).__name__}: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")
    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
    )
