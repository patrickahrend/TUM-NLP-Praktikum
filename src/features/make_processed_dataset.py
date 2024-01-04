import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_pickle(obj, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def combine_features(embedding, feature_list):
    combined_features = np.hstack([embedding] + feature_list)
    return combined_features


def scale_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features


def main():
    project_dir = Path(__file__).resolve().parents[2]
    embeddings_path = project_dir / "data/processed/embeddings"
    features_path = project_dir / "data/processed/features"
    processed_dataset_path = project_dir / "data/processed/datasets"

    train_labels_path = project_dir / "data/processed/training_data_preprocessed.csv"
    test_labels_path = project_dir / "data/evaluation/gold_standard_preprocessed.csv"

    # Process each type of embedding
    for emb_type in ["gpt", "ft", "w2v", "glove", "bert"]:
        # Load separate embeddings
        X_train_proc = load_pickle(embeddings_path / f"{emb_type}_train_proc_desc.pkl")
        X_train_legal = load_pickle(
            embeddings_path / f"{emb_type}_train_legal_text.pkl"
        )
        X_test_proc = load_pickle(embeddings_path / f"{emb_type}_test_proc_desc.pkl")
        X_test_legal = load_pickle(embeddings_path / f"{emb_type}_test_legal_text.pkl")

        # Load combined embeddings
        X_train_emb = load_pickle(embeddings_path / f"{emb_type}_train.pkl")
        X_test_emb = load_pickle(embeddings_path / f"{emb_type}_test.pkl")

        # Load additional features
        cos_sim_train = load_pickle(features_path / f"cos_sim_{emb_type}_train.pkl")
        cos_sim_test = load_pickle(features_path / f"cos_sim_{emb_type}_test.pkl")
        freqs_proc_desc_train = load_pickle(
            features_path / f"train_proc_desc_freqs.pkl"
        )
        freqs_proc_desc_test = load_pickle(features_path / f"test_proc_desc_freqs.pkl")
        freqs_legal_text_train = load_pickle(
            features_path / f"train_legal_text_freqs.pkl"
        )
        freqs_legal_text_test = load_pickle(
            features_path / f"test_legal_text_freqs.pkl"
        )

        # Scale additional features
        cos_sim_train_scaled = scale_features(cos_sim_train.reshape(-1, 1))
        cos_sim_test_scaled = scale_features(cos_sim_test.reshape(-1, 1))
        freqs_proc_desc_train_scaled = scale_features(freqs_proc_desc_train)
        freqs_proc_desc_test_scaled = scale_features(freqs_proc_desc_test)
        freqs_legal_text_train_scaled = scale_features(freqs_legal_text_train)
        freqs_legal_text_test_scaled = scale_features(freqs_legal_text_test)

        print("X_train_emb shape:", X_train_emb.shape)
        print("cos_sim_train_scaled shape:", cos_sim_train_scaled.shape)
        print("freqs_proc_desc_train_scaled shape:", freqs_proc_desc_train_scaled.shape)
        print(
            "freqs_legal_text_train_scaled shape:", freqs_legal_text_train_scaled.shape
        )

        # Dataset 1: Just combined embeddings -> | X_train_emb |
        save_pickle(
            X_train_emb, processed_dataset_path / f"{emb_type}_train_combined.pkl"
        )
        save_pickle(
            X_test_emb, processed_dataset_path / f"{emb_type}_test_combined.pkl"
        )

        # Dataset 2: Separate process and legal text embeddings -> | X_train_proc | X_train_legal_features |
        X_train_separate = np.concatenate((X_train_proc, X_train_legal), axis=1)
        X_test_separate = np.concatenate((X_test_proc, X_test_legal), axis=1)
        save_pickle(
            X_train_separate, processed_dataset_path / f"{emb_type}_train_separate.pkl"
        )
        save_pickle(
            X_test_separate, processed_dataset_path / f"{emb_type}_test_separate.pkl"
        )

        # Dataset 3: Combined embeddings with additional features -> | X_train_emb | cos_sim_train | freq_train |
        X_train_combined_features = combine_features(
            X_train_emb,
            [
                cos_sim_train_scaled,
                freqs_proc_desc_train_scaled,
                freqs_legal_text_train_scaled,
            ],
        )
        X_test_combined_features = combine_features(
            X_test_emb,
            [
                cos_sim_test_scaled,
                freqs_proc_desc_test_scaled,
                freqs_legal_text_test_scaled,
            ],
        )
        save_pickle(
            X_train_combined_features,
            processed_dataset_path / f"{emb_type}_train_combined_features.pkl",
        )
        save_pickle(
            X_test_combined_features,
            processed_dataset_path / f"{emb_type}_test_combined_features.pkl",
        )

        # Dataset 4: Separate embeddings with additional features for each -> | X_train_proc | X_train_legal_features | cos_sim_train | freq_train |
        X_train_with_features = np.concatenate(
            (
                X_train_separate,
                cos_sim_train_scaled,
                freqs_proc_desc_train_scaled,
                freqs_legal_text_train_scaled,
            ),
            axis=1,
        )
        X_test_with_features = np.concatenate(
            (
                X_test_separate,
                cos_sim_test_scaled,
                freqs_proc_desc_test_scaled,
                freqs_legal_text_test_scaled,
            ),
            axis=1,
        )

        save_pickle(
            X_train_with_features,
            processed_dataset_path / f"{emb_type}_train_separate_with_features.pkl",
        )
        save_pickle(
            X_test_with_features,
            processed_dataset_path / f"{emb_type}_test_separate_with_features.pkl",
        )


if __name__ == "__main__":
    main()
