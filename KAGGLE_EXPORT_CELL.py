# PASTE THIS CELL AT THE END of souschef-1.ipynb, AFTER all model and evaluation cells.
# The variables below come from your existing SousChef AI v2 Kaggle notebook.
# This will export the exact trained models (no model retraining in Streamlit).
from pathlib import Path
import json, zipfile
import numpy as np
import pandas as pd
import joblib
import faiss

needed = ["recipes", "recipe_index", "rag_docs", "rag_index", "cf_model"]
missing = [v for v in needed if v not in globals()]
if missing:
    raise RuntimeError(f"Run the training notebook first; missing variables: {missing}")

# Guard against exporting the 3-recipe fallback after a missing Kaggle input.
MIN_EXPORT_RECIPES = 1000
if len(recipes) < MIN_EXPORT_RECIPES:
    raise RuntimeError(
        f"Only {len(recipes)} recipes loaded. This looks like the demo/fallback data. "
        "Attach Food.com Recipes and Interactions in Kaggle, run all training cells again, "
        "verify len(recipes) >= 1000, and then retry this export."
    )
if len(rag_docs) == 0:
    raise RuntimeError("No RAG documents loaded; run the RAG indexing cells first.")

output_dir = Path("/kaggle/working/souschef_streamlit_artifacts")
output_dir.mkdir(parents=True, exist_ok=True)

# Preserve exact row ordering matching the saved FAISS indexes.
recipes.reset_index(drop=True).to_parquet(output_dir / "recipes.parquet", index=False)
rag_docs.reset_index(drop=True).to_parquet(output_dir / "rag_docs.parquet", index=False)

def export_index(obj, destination):
    vectors = np.asarray(obj.embeddings, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) == 0:
        raise RuntimeError(f"No embeddings available for {destination}. Run embedding cell first.")
    # all-MiniLM-L6-v2 embeddings were normalized in the training notebook.
    faiss.normalize_L2(vectors)
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)
    faiss.write_index(idx, str(output_dir / destination))

export_index(recipe_index, "recipe_index.faiss")
export_index(rag_index, "rag_index.faiss")

# A saved sklearn Pipeline can be loaded at inference time; only load trusted artifacts.
if globals().get("cuisine_model") is not None:
    joblib.dump(cuisine_model, output_dir / "cuisine_model.joblib")

if getattr(cf_model, "ready", False):
    users = sorted(cf_model.user_to_idx, key=cf_model.user_to_idx.get)
    items = sorted(cf_model.item_to_idx, key=cf_model.item_to_idx.get)
    np.savez_compressed(
        output_dir / "collab_factors.npz",
        user_factors=np.asarray(cf_model.user_factors, dtype=np.float32),
        item_factors=np.asarray(cf_model.item_factors, dtype=np.float32),
        users=np.asarray([str(x) for x in users]),
        items=np.asarray([str(x) for x in items]),
    )

metrics_export = {}
for name, variable in [("semantic_retrieval", "retrieval_metrics"), ("rag_retrieval", "rag_metrics")]:
    raw = globals().get(variable, {})
    if isinstance(raw, dict):
        metrics_export[name] = {k: float(v) for k, v in raw.items()}
(output_dir / "metrics.json").write_text(json.dumps(metrics_export, indent=2))

if isinstance(globals().get("cuisine_results"), pd.DataFrame) and not cuisine_results.empty:
    cuisine_results.to_csv(output_dir / "cuisine_results.csv", index=False)

manifest = {
    "source": "SousChef AI v2 Kaggle notebook",
    "recipe_count": len(recipes),
    "rag_document_count": len(rag_docs),
    "embedding_model": globals().get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    "has_cuisine_classifier": (output_dir / "cuisine_model.joblib").exists(),
    "has_collaborative_filtering": (output_dir / "collab_factors.npz").exists(),
    "notes": "All saved FAISS indexes and dataframes share row order. Metrics are from the notebook, not calculated in Streamlit.",
}
(output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

archive = Path("/kaggle/working/souschef_streamlit_artifacts.zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for path in output_dir.iterdir():
        if path.is_file():
            z.write(path, arcname=f"artifacts/{path.name}")

print("Export finished:", archive)
print("Files:", [(p.name, round(p.stat().st_size / 1024**2, 2)) for p in output_dir.iterdir()])
print("Download souschef_streamlit_artifacts.zip from Kaggle's Output panel.")
