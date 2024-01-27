import pickle


class ModelBase:
    def __init__(self, model_name, model_instance):
        self.model_name = model_name
        self.model = model_instance

    def save_model(
        self,
        directory,
    ):
        filepath = directory / f"{self.model_name}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print(f"{self.model_name} trained successfully.")

    def predict(self, X):
        return self.model.predict(X)

    def set_params(self, **params):
        self.model.set_params(**params)