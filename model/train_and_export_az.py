"""Train FunkSVD + Content Hybrid Recommender on Azerbaijani Movies dataset
and export artifacts compatible with model/recommender.py to checkpoints_az/.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


class FunkSVD:
    def __init__(self, n_factors=30, n_epochs=25, lr=0.015, reg=0.05, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state

    def fit(self, df, all_user_ids, all_movie_ids):
        rng = np.random.default_rng(self.random_state)
        self.u_map = {int(u): k for k, u in enumerate(all_user_ids)}
        self.i_map = {int(m): k for k, m in enumerate(all_movie_ids)}
        self.mu = float(df["rating"].mean())
        self.P = rng.normal(0, 0.1, (len(self.u_map), self.n_factors)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, (len(self.i_map), self.n_factors)).astype(np.float32)
        self.bu = np.zeros(len(self.u_map), dtype=np.float32)
        self.bi = np.zeros(len(self.i_map), dtype=np.float32)

        u_idx = df["userId"].map(self.u_map).to_numpy()
        i_idx = df["movieId"].map(self.i_map).to_numpy()
        r = df["rating"].to_numpy(dtype=np.float32)
        P, Q, bu, bi = self.P, self.Q, self.bu, self.bi
        mu, lr, reg = self.mu, self.lr, self.reg

        for _ in range(self.n_epochs):
            order = rng.permutation(len(r))
            for u, i, rating_val in zip(u_idx[order], i_idx[order], r[order]):
                pu = P[u].copy()
                qi = Q[i]
                err = rating_val - (mu + bu[u] + bi[i] + pu @ qi)
                bu[u] += lr * (err - reg * bu[u])
                bi[i] += lr * (err - reg * bi[i])
                P[u] += lr * (err * qi - reg * pu)
                Q[i] += lr * (err * pu - reg * qi)

        return self

    def predict(self, df):
        u = df["userId"].map(self.u_map).to_numpy(dtype=float)
        i = df["movieId"].map(self.i_map).to_numpy(dtype=float)
        pred = np.full(len(df), self.mu)
        ku, ki = ~np.isnan(u), ~np.isnan(i)
        pred[ku] += self.bu[u[ku].astype(int)]
        pred[ki] += self.bi[i[ki].astype(int)]
        both = ku & ki
        pred[both] += np.einsum("ij,ij->i", self.P[u[both].astype(int)], self.Q[i[both].astype(int)])
        return np.clip(pred, 0.5, 5.0)


def main():
    root = Path(__file__).resolve().parent.parent
    az_dir = root / "datasets" / "processed_az"
    out_dir = root / "model" / "checkpoints_az"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {az_dir}...")
    movies_df = pd.read_parquet(az_dir / "movies.parquet")
    if "movieId" not in movies_df.columns:
        movies_df["movieId"] = movies_df["movie_idx"]

    # Also load the raw Azerbaijani catalog for original title and overview
    raw_az_csv = root / "datasets" / "raw" / "az" / "az_movies_raw.csv"
    if raw_az_csv.exists():
        raw_az = pd.read_csv(raw_az_csv)
        script_fixes = {
            "Кавказ": "Qafqaz",
            "آتابای": "Atabay",
            "پوست": "Dəri",
            "موقعیت مهدی": "Mehdinin Mövqeyi",
        }
        for orig, fixed in script_fixes.items():
            raw_az.loc[raw_az["original_title"] == orig, "original_title"] = fixed

        def format_az_title(row):
            ot = str(row["original_title"]).strip() if pd.notna(row["original_title"]) else str(row["title"]).strip()
            yr = int(row["year"]) if pd.notna(row["year"]) and row["year"] > 0 else None
            return f"{ot} ({yr})" if yr else ot

        raw_az["az_title"] = raw_az.apply(format_az_title, axis=1)
        raw_az["en_title"] = raw_az["title"]

        keep_cols = ["az_title", "en_title", "original_title", "year", "overview"]
        existing = [c for c in keep_cols if c in raw_az.columns]
        if "title" in raw_az.columns:
            movies_df = movies_df.merge(raw_az[["title"] + existing].drop_duplicates("title"), on="title", how="left")
            movies_df["title"] = movies_df["az_title"].fillna(movies_df["title"])


    train_df = pd.read_parquet(az_dir / "ratings_train.parquet")
    test_df = pd.read_parquet(az_dir / "ratings_test.parquet")
    all_ratings = pd.read_parquet(az_dir / "ratings_clean.parquet")

    for df in [train_df, test_df, all_ratings]:
        if "userId" not in df.columns:
            df["userId"] = df["user_idx"]
        if "movieId" not in df.columns:
            df["movieId"] = df["movie_idx"]

    all_user_ids = np.sort(all_ratings["userId"].unique())
    all_movie_ids = movies_df["movieId"].to_numpy()

    print(f"Dataset stats: {len(all_user_ids)} users, {len(all_movie_ids)} movies, {len(all_ratings)} ratings.")

    # 1. Train FunkSVD
    n_factors = 30
    n_epochs = 30
    print(f"Training FunkSVD (k={n_factors}, epochs={n_epochs})...")
    svd = FunkSVD(n_factors=n_factors, n_epochs=n_epochs, lr=0.015, reg=0.04)
    svd.fit(train_df, all_user_ids, all_movie_ids)

    # Evaluate RMSE
    test_preds = svd.predict(test_df)
    test_rmse = float(np.sqrt(np.mean((test_df["rating"] - test_preds) ** 2)))
    print(f"Test RMSE: {test_rmse:.4f}")

    # 2. Compute genre TF-IDF matrix
    genre_series = movies_df["genres"].fillna("").apply(lambda g: g.split("|") if g else [])
    tfidf = TfidfVectorizer(analyzer=lambda x: x)
    G = tfidf.fit_transform(genre_series)
    Gd = np.asarray(G.todense(), dtype=np.float32)

    item_norm = np.linalg.norm(Gd, axis=1, keepdims=True)
    item_norm[item_norm == 0] = 1.0
    Gd_normed = (Gd / item_norm).astype(np.float32)

    # 3. User profiles
    movie_row = {int(m): k for k, m in enumerate(all_movie_ids)}
    rows = train_df["userId"].map(svd.u_map).to_numpy()
    cols = train_df["movieId"].map(movie_row).to_numpy()
    w = (train_df["rating"].to_numpy() - 2.5).astype(np.float32)
    W = sparse.csr_matrix((w, (rows, cols)), shape=(len(all_user_ids), len(all_movie_ids)))

    prof = W @ Gd
    norm = np.linalg.norm(prof, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    profiles = (prof / norm).astype(np.float32)

    # 4. Save checkpoint
    print(f"Saving checkpoint to {out_dir}...")
    np.savez(
        out_dir / "model.npz",
        P=svd.P,
        Q=svd.Q,
        bu=svd.bu,
        bi=svd.bi,
        mu=svd.mu,
        user_ids=all_user_ids,
        item_ids=all_movie_ids,
        Gd_normed=Gd_normed,
        profiles=profiles,
    )

    save_cols = ["movieId", "title", "genres"]
    for opt_col in ["poster_url", "en_title", "original_title", "year", "overview"]:
        if opt_col in movies_df.columns:
            save_cols.append(opt_col)

    movies_df[save_cols].to_parquet(out_dir / "movies.parquet", index=False)
    all_ratings[["userId", "movieId", "rating"]].to_parquet(out_dir / "ratings.parquet", index=False)

    meta = {
        "model": "FunkSVD (biased MF, SGD) + genre TF-IDF hybrid",
        "dataset": "Azerbaijani Cinema (TMDB top-100 catalog + synthetic ratings)",
        "catalog_type": "az",
        "n_users": len(all_user_ids),
        "n_items": len(all_movie_ids),
        "n_ratings": len(all_ratings),
        "n_users_sampled": len(all_user_ids),
        "min_movie_ratings": 1,
        "k": n_factors,
        "best_alpha": 0.7,
        "epochs": n_epochs,
        "trained_on": "train temporal split (80%)",
        "test_rmse": round(test_rmse, 3),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Done! Artifacts saved successfully.")


if __name__ == "__main__":
    main()
