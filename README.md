# CineMatch 🎬
### Hybrid Movie Recommendation System with Azerbaijani Cinema Integration

> A dual-engine, Netflix-styled recommendation platform combining FunkSVD matrix factorization and TF-IDF content-based filtering — bringing AI-powered personalization to both global cinema and the historically underrepresented Azerbaijani film heritage.

🔗 **Live Repository:** [https://github.com/Fariz-Isg/CineMatch](https://github.com/Fariz-Isg/CineMatch)

---

## 👥 Team Members

| Name | Role | Responsibilities |
|---|---|---|
| **Fariz Isgandarli** | ML Inference & Systems Architect | Serving-side inference engine, real-time rating adaptation loop, checkpoint serialization, deployment |
| **[Member 2]** | Latent Factor & CF Engineer | FunkSVD algorithm, SGD optimization, bias modeling, RMSE benchmarks |
| **[Member 3]** | NLP & Cold-Start Specialist | TF-IDF genre vectors, hybrid blending, item-to-item similarity, cold-start resolution |
| **[Member 4]** | Data Pipeline & ETL Architect | Azerbaijani cinema pipeline, TMDB enrichment, synthetic ratings, temporal data splits |

---

## ❓ Problems CineMatch Solves

### 1. The Cold-Start Problem
Traditional Collaborative Filtering (CF) algorithms are completely blind to new users who have no rating history. Without prior interaction data, these systems fall back to generic popularity-based suggestions — resulting in poor first impressions and weak user retention.

### 2. Filter Bubbles & Popularity Bias
Pure CF over-amplifies popularity trends, repeatedly recommending the same blockbusters. Pure content-based filtering traps users in a narrow genre loop. Neither alone delivers genuine serendipitous discovery.

### 3. Regional Data Scarcity & Underrepresentation
While Hollywood has multi-million-row benchmark datasets (e.g., MovieLens), **Azerbaijani cinema had zero publicly available machine-learning rating datasets and zero AI-driven recommendation engines**. Iconic cinematic masterpieces were entirely absent from modern discovery platforms.

CineMatch addresses all three through its hybrid architecture and a fully engineered synthetic Azerbaijani data pipeline.

---

## ✨ Project Features

- **🇦🇿 Azerbaijani Cinema Catalog:** 100 authentic Azerbaijani films with native-language titles and release years (*Qorxma, mən səninləyəm (1981)*, *Arşın mal alan (1945)*, *Qaynana (1978)*, *O Olmasın, Bu Olsun (1958)*, *Babək (1979)*, etc.)
- **🌍 Global MovieLens Catalog:** Extensive international catalog with 7,359 titles.
- **Dual-Catalog Switcher:** Seamlessly toggle between both catalogs from the sidebar.
- **Hybrid Recommendation Engine:** Tunable blend of collaborative and content-based filtering via a live $\alpha$ slider.
- **Live Movie Rating Panel:** Rate any movie (1.0 – 5.0 ★) and watch personalized recommendations update instantly.
- **Cold-Start Matching:** New users without history get on-the-fly content-based recommendations from a few selected favorites.
- **Item-to-Item Discovery:** "Because you watched…" similarity suggestions using cosine distance and popularity tie-breaking.
- **TMDB Poster Artwork:** High-resolution movie backdrops loaded directly from TMDB for both catalogs.
- **Match-Percentage Badges:** Green percentage overlays showing how strongly each recommendation matches the user profile.
- **Netflix-Style Dark UI:** Custom Streamlit interface with glassmorphism card layouts, hover animations, and responsive design.

---

## 🔬 Architecture & How It Works

### Hybrid Recommendation Formula

$$\text{Score}(u, i) = \alpha \cdot \hat{r}_{\text{CF}}(u, i) + (1 - \alpha) \cdot \text{Sim}_{\text{Content}}(u, i)$$

Where:
- **$\alpha = 1$:** Pure collaborative filtering (exploits similar user tastes)
- **$\alpha = 0$:** Pure content-based filtering (exploits genre similarity)
- **Best $\alpha = 0.7–0.8$:** Optimal balance validated on held-out test data

---

### 1. Collaborative Filtering — FunkSVD (Biased Matrix Factorization)

$$\hat{r}(u, i) = \mu + b_u + b_i + P_u \cdot Q_i^T$$

| Symbol | Meaning |
|---|---|
| $\mu$ | Global average rating across all users and items |
| $b_u$ | User bias (tendency to rate above/below average) |
| $b_i$ | Item bias (generally well-rated or poorly-rated movies) |
| $P_u \in \mathbb{R}^k$ | User latent factor vector ($k=30$ dimensions) |
| $Q_i \in \mathbb{R}^k$ | Item latent factor vector |

**Training:** Stochastic Gradient Descent (SGD) with $L_2$ regularization, optimizing only over observed ratings in a sparse 98%+ empty matrix. Avoids the dense matrix decomposition requirement of classical SVD.

**SGD Update Rules:**
```
error = rating - (mu + bu + bi + Pu · Qi)
bu += lr * (error - reg * bu)
bi += lr * (error - reg * bi)
Pu += lr * (error * Qi - reg * Pu)
Qi += lr * (error * Pu - reg * Qi)
```

---

### 2. Content-Based Filtering — TF-IDF Genre Embeddings

Each movie is mapped to an $L_2$-normalized genre vector using **TF-IDF weighting**:

$$\hat{g}_i = \frac{G_i}{\|G_i\|_2}$$

**User Genre Profiles** are synthesized as weighted genre preferences from their rating history:

$$\text{profile}_u = \frac{\sum_i w_{u,i} \cdot G_i}{\|\sum_i w_{u,i} \cdot G_i\|_2}$$

Where $w_{u,i} = \text{rating}_{u,i} - 2.5$ (centering ratings around zero).

**Content similarity** is then cosine distance between user profiles and all item genre vectors.

---

### 3. Cold-Start Resolution

For users with zero history, a genre profile is built on-the-fly from a few selected liked movies, enabling instant recommendations without any historical data.

---

### 4. Azerbaijani Cinema Data Pipeline

Since no public dataset exists, a fully engineered 3-stage pipeline was developed:

| Stage | Script | Output |
|---|---|---|
| 1. Fetch Catalog | `az_01_fetch_movies.py` | 100 Azerbaijani films from TMDB API |
| 2. Generate Ratings | `az_02_generate_synthetic.py` | 15,205 taste-matched synthetic ratings, 400 users |
| 3. Temporal Split | `az_03_temporal_split.py` | 80/20 train/test split (no data leakage) |

**Taste-matching approach:** Synthetic users are assigned random genre preference profiles. Each user's ratings are then biased toward movies whose genre vectors have high cosine similarity with that user's profile — producing statistically plausible user-item interactions.

---

## 📊 Model Evaluation

| Catalog | Model | Test RMSE |
|---|---|---|
| Azerbaijani Cinema | FunkSVD (k=30, 30 epochs) | **0.693** |
| Global MovieLens | FunkSVD (k=100, 20 epochs) | **0.82** |

Evaluation used an **80/20 per-user temporal split**: the most recent 20% of each user's ratings form the test set. This mirrors real production environments where the model must predict future preferences without seeing future data.

---

## 🛠️ Technologies Used

### Machine Learning & Mathematics
- **NumPy** — Dense linear algebra (latent factor matrices, dot products, min-max normalization)
- **SciPy** — Sparse matrix representations during training
- **Scikit-Learn** — TF-IDF Vectorizer, cosine similarity metrics

### Data Engineering & Storage
- **Pandas** — Tabular data manipulation and processing
- **Apache Arrow / PyArrow** — High-speed Parquet columnar serialization for model artifacts
- **TMDB REST API** — Movie metadata, poster images, genres, and Azerbaijani film discovery

### Web Application & Interface
- **Streamlit** — Reactive Python web application framework
- **Vanilla CSS** — Custom Netflix dark theme, glassmorphism card components, hover animations

### DevOps & Version Control
- **Git & GitHub** — Version control (`Fariz-Isg/CineMatch`)
- **GitHub CLI (`gh`)** — Repository creation and management
- **Streamlit Community Cloud** — Free production deployment

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Fariz-Isg/CineMatch.git
cd CineMatch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Web App

```bash
streamlit run streamlit/app.py
```

Open **`http://localhost:8501`** in your browser.

### 3. (Optional) Regenerate the Azerbaijani Dataset & Retrain

```bash
# Re-run the data pipeline
python3 datasets/scripts/az_run_pipeline.py

# Retrain and export the Azerbaijani model checkpoint
python3 model/train_and_export_az.py
```

> **Note:** Regenerating the Azerbaijani film catalog from TMDB requires a free API key. Set `TMDB_API_KEY` in `datasets/.env` (see `datasets/.env.example`).

---

## 📁 Project Structure

```text
CineMatch/
├── streamlit/
│   └── app.py                      # Netflix-styled Streamlit interface & real-time rating
├── model/
│   ├── recommender.py              # Serving-side hybrid recommendation engine (no training)
│   ├── train_and_export_az.py      # FunkSVD training & checkpoint export for Azerbaijani data
│   ├── Movie_Recommendation_System_v3_20M.ipynb  # Original MovieLens training notebook
│   ├── checkpoints/                # Trained Global (MovieLens) model artifacts
│   │   ├── model.npz               # P, Q, bu, bi, mu, Gd_normed, profiles
│   │   ├── movies.parquet          # Movie catalog with TMDB poster URLs
│   │   ├── ratings.parquet         # Sampled rating interactions
│   │   └── meta.json               # Hyperparameters and RMSE metrics
│   └── checkpoints_az/             # Trained Azerbaijani model artifacts (same schema)
├── datasets/
│   ├── raw/az/
│   │   └── az_movies_raw.csv       # 100 Azerbaijani films fetched from TMDB
│   ├── processed_az/               # Parquet output of the AZ pipeline
│   └── scripts/
│       ├── az_01_fetch_movies.py   # Stage 1: TMDB catalog fetcher
│       ├── az_02_generate_synthetic.py  # Stage 2: Taste-matched rating generator
│       ├── az_03_temporal_split.py # Stage 3: Train/test temporal split
│       ├── az_run_pipeline.py      # Orchestrator: runs all 3 stages in order
│       └── config.py               # Shared configuration paths and constants
└── requirements.txt                # Project dependencies
```

---

## 🗺️ Future Plans

1. **Persistent User Accounts** — PostgreSQL / Supabase integration to save multi-session watchlists, rating histories, and custom preference profiles.
2. **Deep Learning Upgrade** — Transition from FunkSVD to **Neural Collaborative Filtering (NCF)** two-tower embeddings in PyTorch to model complex non-linear user-movie interactions.
3. **Azerbaijani Catalog Expansion** — Scale from 100 to 500+ films including vintage archives, short films, and documentaries. Add knowledge graph embeddings for famous directors (Rasim Ojagov, Tofiq Taghizade) and actors.
4. **Director & Actor Signal Embeddings** — Incorporate cast/crew graph networks to recommend films by auteur style, ensemble, or cultural period.
5. **Mobile PWA** — Progressive Web App packaging for seamless iOS and Android experience.
6. **Negative Feedback Loop** — Low star ratings actively demote genre vectors to penalize disliked content in real-time.

---

## 📜 Dataset Credits

- **MovieLens Dataset:** F. M. Harper & J. A. Konstan (2015), *The MovieLens Datasets: History and Context*, ACM TiiS 5(4).
- **Azerbaijani Cinema Catalog:** Enriched via the [TMDB API](https://www.themoviedb.org/) (language: `az`, origin country: `AZ`). All rights belong to respective rights holders.