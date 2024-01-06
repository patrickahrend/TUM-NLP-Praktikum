import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Custom import
from make_processed_dataset import load_pickle, save_pickle


class FineTuningProcessor:
    def __init__(self, n_components=0.95):
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.feature_importances_ = None
        self.original_feature_count_ = 0
        self.additional_features_importance_ = None

    def fit_transform_pca(self, X):
        self.original_feature_count_ = X.shape[1] - 2
        # X_embeddings = X[:, :-101]
        # X_additional = X[:, -101:]

        # Standardize the embedding features
        X_embeddings_scaled = self.scaler.fit_transform(X)
        # Fit PCA on the embedding features only
        X_embeddings_pca = self.pca.fit_transform(X_embeddings_scaled)

        # # Concatenate the PCA components with the additional features
        # X_pca_combined = np.hstack((X_embeddings_pca, X_additional))

        # print(
        #     f"Reduced number of features to {self.pca.n_components_} from {self.original_feature_count_} "
        #     f"and added {X_additional.shape[1]} additional features"
        # )
        return X_embeddings_pca

    def fit_random_forest(self, X, y):
        rf = RandomForestClassifier()
        rf.fit(X, y)
        self.feature_importances_ = rf.feature_importances_
        return rf

    def get_feature_importance(self, n_top_features=10):
        # Get the importance of the additional features
        self.additional_features_importance_ = self.feature_importances_[-101:]

        # Get the importance of the PCA components
        pca_components_importance = self.feature_importances_[:-101]

        # Sort the PCA components by importance
        sorted_indices = np.argsort(pca_components_importance)[::-1]
        top_indices = sorted_indices[:n_top_features]

        print(
            f"Importance of additional features: {self.additional_features_importance_}"
        )
        print(
            f"Top {n_top_features} most important PCA components indices and importances:"
        )
        for idx in top_indices:
            print(
                f"PCA component index: {idx}, Importance: {pca_components_importance[idx]}"
            )

        return (
            top_indices,
            self.additional_features_importance_,
            pca_components_importance,
        )


def main():
    project_dir = Path(__file__).resolve().parents[2]
    processed_dataset_path = project_dir / "data/processed/datasets"
    train_labels_path = project_dir / "data/processed/training_data_preprocessed.csv"
    pca_dataset_path = project_dir / "data/processed/datasets/pca"

    os.makedirs(pca_dataset_path, exist_ok=True)

    y_train = pd.read_csv(train_labels_path)["Label"]

    ft_processor = FineTuningProcessor(n_components=0.95)

    pca_results = []
    for emb_type in ["gpt", "ft", "w2v", "glove", "bert", "tfidf"]:
        # X_train = load_pickle(
        #     processed_dataset_path / f"{emb_type}_train_separate_with_features.pkl"
        # )
        #
        # X_train_pca = ft_processor.fit_transform_pca(X_train)
        #
        # rf_model = ft_processor.fit_random_forest(X_train_pca, y_train)
        # (
        #     top_indices,
        #     additional_features_importance_,
        #     pca_components_importance,
        # ) = ft_processor.get_feature_importance()
        #
        # pca_results.append(
        #     {
        #         "embedding_type": emb_type,
        #         "orginal_feature_count": ft_processor.original_feature_count_,
        #         "n_components": ft_processor.pca.n_components_,
        #         "top_indices": top_indices,
        #         "additional_features_importance": additional_features_importance_,
        #         "pca_components_importance": pca_components_importance,
        #     }
        # )

        X_train = load_pickle(processed_dataset_path / f"{emb_type}_train_separate.pkl")

        X_test = load_pickle(processed_dataset_path / f"{emb_type}_test_separate.pkl")

        X_train_pca = ft_processor.fit_transform_pca(X_train)

        X_test_pca = ft_processor.pca.transform(X_test)

        # Save the PCA-transformed dataset
        pca_dataset_filename = f"{emb_type}_train_separate_pca.pkl"
        save_pickle(X_train_pca, pca_dataset_path / pca_dataset_filename)

        pca_test_filename = f"{emb_type}_test_separate_pca.pkl"
        save_pickle(X_test_pca, pca_dataset_path / pca_test_filename)

    results_df = pd.DataFrame(pca_results)
    os.makedirs(project_dir / "references/feature importance", exist_ok=True)
    results_df.to_csv(
        project_dir
        / "references/feature importance/pca_feature_importance_results_dt_4.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
