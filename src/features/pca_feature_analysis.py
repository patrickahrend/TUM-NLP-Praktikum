from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Custom import
from make_processed_dataset import load_pickle


class FineTuningProcessor:
    def __init__(self, n_components=0.95):
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.feature_importances_ = None

    def fit_transform_pca(self, X):
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        print(
            f"Reduced number of features to {self.pca.n_components_} from {X.shape[1]}"
        )
        return X_pca

    def fit_random_forest(self, X, y):
        rf = RandomForestClassifier()
        rf.fit(X, y)
        self.feature_importances_ = rf.feature_importances_
        return rf

    def get_feature_importance(self, n_top_features=10):
        cos_sim_importance = self.feature_importances_[-1]
        word_freq_importances = self.feature_importances_[:-1]

        sorted_indices = np.argsort(self.feature_importances_)[::-1]
        top_indices = sorted_indices[:n_top_features]

        print(f"Cosine similarity feature importance: {cos_sim_importance}")
        print(f"Word frequency features importances: {word_freq_importances}")
        print("Top 10 most important features indices and importances:")
        for idx in top_indices:
            print(f"Feature index: {idx}, Importance: {self.feature_importances_[idx]}")

        return top_indices, cos_sim_importance, word_freq_importances


def main():
    project_dir = Path(__file__).resolve().parents[2]
    processed_dataset_path = project_dir / "data/processed/datasets"
    train_labels_path = project_dir / "data/processed/training_data_preprocessed.csv"

    y_train = pd.read_csv(train_labels_path)["Label"]

    ft_processor = FineTuningProcessor(n_components=0.95)

    pca_results = []
    for emb_type in ["gpt", "ft", "w2v", "glove", "bert"]:
        X_train = load_pickle(processed_dataset_path / f"{emb_type}_train_separate.pkl")

        X_train_pca = ft_processor.fit_transform_pca(X_train)

        rf_model = ft_processor.fit_random_forest(X_train_pca, y_train)
        (
            top_indices,
            cos_sim_importance,
            word_freq_importances,
        ) = ft_processor.get_feature_importance()

        pca_results.append(
            {
                "embedding_type": emb_type,
                "n_components": ft_processor.pca.n_components_,
                "top_feature_indices": top_indices,
                "cos_sim_importance": cos_sim_importance,
                "word_freq_importances": word_freq_importances.tolist(),
            }
        )

    results_df = pd.DataFrame(pca_results)
    results_df.to_csv(
        project_dir / "references/pca_feature_importance_results.csv", index=False
    )


if __name__ == "__main__":
    main()
