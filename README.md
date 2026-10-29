# CineMatch 🎬

A hybrid movie recommendation platform built with **FunkSVD matrix factorization** and **content-based genre filtering**, complete with a Netflix-styled **Streamlit** user interface.

Featuring dual-catalog support:
1. **🇦🇿 Azerbaijani Cinema** — 100 native Azerbaijani films enriched via TMDB with taste-matched user ratings.
2. **🌍 Global MovieLens** — Extensive international catalog based on MovieLens.

---

## ✨ Features

- **Hybrid Recommendation Engine:** Blends collaborative filtering (FunkSVD with user & item biases) and content-based filtering (normalized genre TF-IDF embeddings) via a tunable weight $\alpha$.
- **Personalized Recommendations:** Top picks tailored to individual user watch and rating history.
- **Cold-Start Handling:** On-the-fly preference matching for new users based on a few selected favorites.
- **Item-to-Item Similarity:** Discover similar films based on genre profiles and popularity.
- **Netflix-Style UI:** Responsive, modern dark-mode frontend with poster backdrops, match percentages, and bilingual title support.

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

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

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```text
CineMatch/
├── streamlit/
│   └── app.py                     # Netflix-styled Streamlit interface
├── model/
│   ├── recommender.py             # Serving-side hybrid recommendation engine
│   ├── train_and_export_az.py     # Training & export script for Azerbaijani catalog
│   ├── checkpoints/               # Trained MovieLens model artifacts
│   └── checkpoints_az/            # Trained Azerbaijani model artifacts
├── datasets/
│   ├── raw/az/                    # Real Azerbaijani film catalog from TMDB
│   ├── processed_az/              # Processed train/test parquet splits
│   └── scripts/                   # Data pipeline scripts (download, enrich, split)
└── requirements.txt               # Project dependencies
```

---

## 🔬 Recommender Architecture

The hybrid score for a user $u$ and item $i$ is calculated as:

$$\text{Score}(u, i) = \alpha \cdot \hat{r}_{\text{CF}}(u, i) + (1 - \alpha) \cdot \text{Sim}_{\text{Content}}(u, i)$$

- **Collaborative Filtering ($\hat{r}_{\text{CF}}$):**
  $$\hat{r}(u, i) = \mu + b_u + b_i + P_u \cdot Q_i^T$$
  Trained with Stochastic Gradient Descent (SGD) with $L_2$ regularization.
- **Content-Based ($\text{Sim}_{\text{Content}}$):**
  Cosine similarity between user genre profile vector and item genre vector.
- **$\alpha$ Slider:** Interactively adjust between pure genre matching ($\alpha = 0$) and collaborative taste matching ($\alpha = 1$).