"""Reusable SousChef inference code, matched to souschef-1.ipynb.

In demo mode, the retriever is deliberately TF-IDF, not a trained embedding model.
With exported Kaggle artifacts it uses all-MiniLM-L6-v2 and FAISS.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def normal(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s\-]", " ", str(text).lower())).strip()


def split_items(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [normal(x) for x in value if normal(x)]
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [normal(x) for x in parsed if normal(x)]
        except (SyntaxError, ValueError):
            pass
        return [normal(x) for x in re.split(r"[,\n]", value) if normal(x)]
    return []


def user_items(text):
    return [normal(x) for x in re.split(r"[,\n]+", str(text)) if normal(x)]


MEAT = {"beef", "chicken", "pork", "fish", "shrimp", "salmon", "tuna", "anchovy", "lamb", "turkey", "bacon", "sausage", "gelatin"}
DAIRY_EGG = {"egg", "eggs", "milk", "butter", "cheese", "cream", "paneer", "feta", "parmesan", "yogurt", "ghee"}
GLUTEN = {"wheat", "flour", "bread", "pasta", "tortilla", "noodles", "couscous", "barley", "semolina"}


def contains_terms(items, terms):
    phrase = " ".join(items)
    return any(re.search(r"\b" + re.escape(term) + r"\b", phrase) for term in terms)


def prep(df):
    df = df.copy().reset_index(drop=True)
    if "recipe_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "recipe_id"})
    if "recipe_id" not in df.columns:
        df["recipe_id"] = np.arange(len(df))
    for col in ["name", "ingredients", "steps", "tags"]:
        if col not in df:
            df[col] = ""
    for col, source in [("ingredients_list", "ingredients"), ("steps_list", "steps"), ("tags_list", "tags")]:
        if col not in df:
            df[col] = df[source].map(split_items)
        else:
            df[col] = df[col].map(split_items)
    for col, source in [("ingredient_text", "ingredients_list"), ("step_text", "steps_list"), ("tag_text", "tags_list")]:
        if col not in df:
            df[col] = df[source].map(" ".join)
    if "recipe_text" not in df:
        df["recipe_text"] = (df["name"].astype(str) + " ingredients " + df["ingredient_text"] + " tags " + df["tag_text"] + " instructions " + df["step_text"]).map(normal)
    df["minutes_clean"] = pd.to_numeric(df.get("minutes_clean", df.get("minutes", 30)), errors="coerce").fillna(30).clip(lower=1)
    df["rating"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(0)
    df["review_count"] = pd.to_numeric(df.get("review_count", 0), errors="coerce").fillna(0)
    if "bayesian_rating" not in df:
        rated = df.loc[df.rating > 0, "rating"]
        mean = rated.mean() if len(rated) else 4.0
        n = df["review_count"]
        prior = 10.0
        df["bayesian_rating"] = (n * df["rating"] + prior * mean) / (n + prior)
    for name, banned in [("is_vegetarian", MEAT), ("is_vegan", MEAT | DAIRY_EGG), ("is_gluten_free", GLUTEN)]:
        if name not in df:
            df[name] = ~df["ingredients_list"].map(lambda xs: contains_terms(xs, banned))
    return df


class TextIndex:
    def __init__(self, texts, faiss_path=None, model_name=EMBEDDING_MODEL):
        self.texts = [str(x) for x in texts]
        self.faiss_path = faiss_path
        self.mode = "faiss" if faiss_path is not None else "tfidf"
        self.model_name = model_name
        if self.mode == "faiss":
            import faiss
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.index = faiss.read_index(str(faiss_path))
            if self.index.ntotal != len(self.texts):
                raise ValueError("Saved FAISS index and metadata have different row counts. Re-export from Kaggle.")
        else:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=40000)
            self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query, top_k=30):
        k = min(max(int(top_k), 1), len(self.texts))
        if self.mode == "faiss":
            emb = self.model.encode([query], normalize_embeddings=True)
            scores, indices = self.index.search(np.asarray(emb, dtype="float32"), k)
            return indices[0], scores[0]
        vec = self.vectorizer.transform([query])
        scores = cosine_similarity(vec, self.matrix).ravel()
        indices = np.argsort(scores)[::-1][:k]
        return indices, scores[indices]


class SousChefEngine:
    def __init__(self, base=ARTIFACTS):
        base = Path(base)
        self.demo = not (base / "manifest.json").exists()
        self.cf = None
        self.cuisine_model = None
        self.metrics = {}
        self.cuisine_results = pd.DataFrame()
        self.reranker = None
        self.manifest = {}
        if self.demo:
            self.recipes = prep(pd.read_csv(ROOT / "data/sample_recipes.csv"))
            self.docs = pd.read_csv(ROOT / "data/culinary_knowledge.csv")
            self.recipe_search = TextIndex(self.recipes["recipe_text"].tolist())
            self.knowledge_search = TextIndex(self.docs["text"].tolist())
        else:
            required = ["recipes.parquet", "recipe_index.faiss", "rag_docs.parquet", "rag_index.faiss"]
            missing = [x for x in required if not (base / x).exists()]
            if missing:
                raise FileNotFoundError(f"Incomplete Kaggle export: missing {', '.join(missing)}")
            self.manifest = json.loads((base / "manifest.json").read_text())
            self.recipes = prep(pd.read_parquet(base / "recipes.parquet"))
            self.docs = pd.read_parquet(base / "rag_docs.parquet")
            model_name = self.manifest.get("embedding_model", EMBEDDING_MODEL)
            self.recipe_search = TextIndex(self.recipes["recipe_text"].tolist(), base / "recipe_index.faiss", model_name)
            # Same SentenceTransformer model as training; FAISS files follow notebook's ordering.
            self.knowledge_search = TextIndex(self.docs["text"].tolist(), base / "rag_index.faiss", model_name)
            self.knowledge_search.model = self.recipe_search.model
            if (base / "cuisine_model.joblib").exists():
                import joblib
                self.cuisine_model = joblib.load(base / "cuisine_model.joblib")
            if (base / "collab_factors.npz").exists():
                with np.load(base / "collab_factors.npz", allow_pickle=False) as npz:
                    self.cf = {k: npz[k] for k in npz.files}
                self.cf_user_index = {x: i for i, x in enumerate(self.cf["users"].astype(str))}
                self.cf_recipe_index = {x: i for i, x in enumerate(self.cf["items"].astype(str))}
            if (base / "metrics.json").exists():
                self.metrics = json.loads((base / "metrics.json").read_text())
            if (base / "cuisine_results.csv").exists():
                self.cuisine_results = pd.read_csv(base / "cuisine_results.csv")

    @staticmethod
    def _scale(values):
        values = pd.Series(values, dtype=float)
        if values.empty:
            return values
        return (values - values.min()) / (values.max() - values.min()) if values.max() != values.min() else pd.Series(.5, index=values.index)

    def _cf_scores(self, user, recipe_ids):
        neutral = np.full(len(recipe_ids), .5)
        if self.cf is None or str(user).strip() not in self.cf_user_index:
            return neutral, False
        u = self.cf["user_factors"][self.cf_user_index[str(user).strip()]]
        raw = np.array([float(np.dot(u, self.cf["item_factors"][self.cf_recipe_index[str(x)]])) if str(x) in self.cf_recipe_index else np.nan for x in recipe_ids])
        if np.isfinite(raw).sum() < 2:
            return neutral, True
        v = raw[np.isfinite(raw)]
        if v.max() != v.min():
            neutral[np.isfinite(raw)] = (v - v.min()) / (v.max() - v.min())
        return neutral, True

    def recommend(self, ingredients, goal, diet="Any", max_minutes=60, exclusions="", user_id="", top_k=8, rerank=False):
        available = user_items(ingredients)
        excluded = user_items(exclusions)
        query = (goal + " " + " ".join(available)).strip() or "easy recipe"
        ids, values = self.recipe_search.search(query, top_k=min(300, len(self.recipes)))
        df = self.recipes.iloc[ids].copy().reset_index(drop=True)
        df["semantic_score"] = values
        df = df.loc[df["minutes_clean"] <= max_minutes]
        diet_col = {"Vegetarian": "is_vegetarian", "Vegan": "is_vegan", "Gluten-free": "is_gluten_free"}.get(diet)
        if diet_col:
            df = df.loc[df[diet_col]]
        if excluded:
            df = df.loc[~df["ingredients_list"].map(lambda x: contains_terms(x, excluded))]
        if df.empty:
            return df
        df = df.copy()
        available_set = set(available)
        df["matched_ingredients"] = df["ingredients_list"].map(lambda xs: sorted(set(xs) & available_set))
        df["missing_ingredients"] = df["ingredients_list"].map(lambda xs: sorted(set(xs) - available_set))
        df["ingredient_score"] = df.apply(lambda x: len(x.matched_ingredients) / max(len(set(x.ingredients_list)), 1), axis=1)
        df["bayesian_score"] = self._scale(df["bayesian_rating"]).values
        df["time_score"] = (1 - df["minutes_clean"] / max(max_minutes, 1)).clip(0, 1)
        df["cf_score"], is_personalized = self._cf_scores(user_id, df["recipe_id"].tolist())
        df["rerank_score"] = self._scale(df["semantic_score"]).values
        if rerank and not self.demo:
            from sentence_transformers import CrossEncoder
            if self.reranker is None:
                self.reranker = CrossEncoder(RERANKER_MODEL)
            head = df.head(min(60, len(df)))
            raw = self.reranker.predict([(query, s) for s in head["recipe_text"].tolist()], batch_size=32, show_progress_bar=False)
            df.loc[head.index, "rerank_score"] = self._scale(raw).values
        if is_personalized:
            df["final_score"] = .25 * self._scale(df.rerank_score).values + .25 * df.ingredient_score + .15 * df.bayesian_score + .1 * df.time_score + .25 * df.cf_score
        else:
            df["final_score"] = .35 * self._scale(df.rerank_score).values + .3 * df.ingredient_score + .2 * df.bayesian_score + .15 * df.time_score
        return df.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)

    def evidence(self, question, top_k=5):
        idx, scores = self.knowledge_search.search(question, top_k)
        df = self.docs.iloc[idx].copy().reset_index(drop=True)
        df["score"] = scores
        return df

    def answer(self, question, llm=False, key="", model="gpt-4o-mini"):
        docs = self.evidence(question)
        plain = "\n\n".join(f"[S{i+1}] {str(r.text)[:550]}" for i, r in enumerate(docs.itertuples()))
        if not llm:
            return "Relevant cooking evidence found:\n\n" + plain, docs
        if not key:
            raise ValueError("Enter an OpenAI API key or switch to evidence-only mode.")
        from openai import OpenAI
        client = OpenAI(api_key=key)
        context = "\n\n".join(f"[S{i+1}] {r.source}: {r.text}" for i, r in enumerate(docs.itertuples()))
        system = ("You are SousChef AI. Answer with practical cooking steps using ONLY the retrieved evidence, "
                  "citing sources [S1], [S2], etc. If the evidence is insufficient, explicitly say so. "
                  "Do not guess allergens, safe food temperatures, or food safety guarantees.")
        resp = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": f"Evidence:\n{context}\n\nQuestion: {question}"}], temperature=0.2, max_tokens=500)
        return resp.choices[0].message.content or "No response produced.", docs

    def cuisine(self, ingredients):
        if self.cuisine_model is None:
            return None
        return str(self.cuisine_model.predict([" ".join(user_items(ingredients))])[0])
