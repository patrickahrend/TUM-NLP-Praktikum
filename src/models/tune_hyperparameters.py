import logging
from sklearn.model_selection import GridSearchCV

# Define the parameter grids for the different models
param_grids = {
    "Logistic_Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
        "class_weight": [None, "balanced"],
    },
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


def tune_hyperparameters(model: object, param_grid, X_train, y_train, cv: int) :
    """
    Tune the hyperparameters of a given model using GridSearchCV.

    Parameters
    ----------
    model : estimator object
        This is assumed to implement the scikit-learn estimator interface.
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
    X_train : array-like of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and n_features is the number of features.
    y_train : array-like of shape (n_samples,)
        Target relative to X for classification or regression.
    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy.

    Returns
    -------
    best_estimator_ : estimator object
        Estimator that was chosen by the search, i.e. estimator which gave highest score on the left out data.
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
    best_score_ : float
        Mean cross-validated score of the best_estimator.
    """
    # Perform grid search on the model
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1
    )
    # Fit the grid search object to the data
    grid_search.fit(X_train, y_train)
    # Log the best parameters and the associated score
    logging.info(f"Best parameters for {type(model).__name__}: {grid_search.best_params_}")
    logging.info(f"Best cross-validation score: {grid_search.best_score_}")
    # Return the best estimator, its parameters, and the associated score
    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
    )
