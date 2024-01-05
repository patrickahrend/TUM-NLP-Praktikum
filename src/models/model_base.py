import pickle


class ModelBase:
    def __init__(self, model_name, model_instance):
        self.model_name = model_name
        self.model = model_instance

    def save_model(self, directory, tuned=False):
        filepath = directory / f"{self.model_name}_{tuned}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print(f"{self.model_name} trained successfully.")

    def predict(self, X):
        return self.model.predict(X)
