"""SousChef AI: no FastAPI server required. Run: streamlit run streamlit_app.py"""
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from souschef_core import SousChefEngine

st.set_page_config(page_title="SousChef AI · Your intelligent kitchen", page_icon="🥘", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap');
html,body,[class*="css"] {font-family:'DM Sans',sans-serif;}
[data-testid="stApp"] {background: #FBFAF7; color:#162B24;}
/* Fix: Streamlit/OS dark-mode field colors conflicting with the light dashboard. */
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stTextArea"] [data-baseweb="textarea"],
[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="textarea"] {
    background-color:#FFFFFF !important;
    border-color:#B9CDBF !important;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-baseweb="select"] input,
[data-baseweb="select"] [role="combobox"] {
    color:#183B2C !important;
    -webkit-text-fill-color:#183B2C !important;
    caret-color:#183B2C !important;
    background:#FFFFFF !important;
    opacity:1 !important;
}
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder,
[data-testid="stSidebar"] input::placeholder {
    color:#687E72 !important;
    -webkit-text-fill-color:#687E72 !important;
    opacity:1 !important;
}
[data-baseweb="select"] [data-testid="stMarkdownContainer"],
[data-baseweb="select"] span,
[data-baseweb="select"] svg {color:#183B2C !important; fill:currentColor !important;}
[data-baseweb="popover"] {background-color:#FFFFFF !important; color:#183B2C !important;}
[data-baseweb="menu"] {background-color:#FFFFFF !important;}
[data-baseweb="menu"] li {color:#183B2C !important;}
[data-testid="stAppHeader"] {background:#FBFAF7 !important;}
[data-testid="stSidebar"] {color:#183B2C !important;}

[data-testid="stSidebar"] {background:#F0F3EC; border-right:1px solid #DDE5DB;}
[data-testid="stSidebar"] * {color:#1D3B2B;}
.block-container {padding-top:1.6rem; max-width:1390px;}
h1,h2,h3 {font-family:'DM Serif Display',Georgia,serif!important; letter-spacing:-.02em; color:#173D2E!important;}
h1 {font-size:2.7rem!important;}
.hero {padding:28px 33px; border-radius:22px; background:linear-gradient(110deg,#173F30,#2C6247); color:#fff; margin-bottom:19px;}
.hero-title {font-family:'DM Serif Display',Georgia,serif;font-size:2.45rem;line-height:1.1;margin:8px 0;color:#fff;}
.hero-sub {font-size:1.06rem;max-width:650px;color:#E3EFE5;}
.eyebrow {text-transform:uppercase;letter-spacing:2.5px;font-size:.72rem;font-weight:700;color:#B4DFC4;}
.micro {font-size:.78rem;color:#547565;}
.metricline {font-size:1rem;color:#274737;}
[data-testid="stMetric"] {background:#FFFFFF;border:1px solid #E4E8E0;border-radius:14px;padding:10px 15px;}
[data-testid="stMetricLabel"] {color:#566C5D;}
[data-testid="stMetricValue"] {color:#1A5136;}
div[data-baseweb="tab-list"] {gap:8px;}
button[data-baseweb="tab"] {background:#EDF2EB;border-radius:10px;padding:10px 16px;color:#2A5039;}
button[aria-selected="true"][data-baseweb="tab"] {background:#1D593A!important;color:white!important;}
[data-testid="stButton"] button[kind="primary"] {background:#215C3E;border-color:#215C3E;}
[data-testid="stExpander"] {background:#FFFFFF;border-radius:12px;border:1px solid #E4E8E0;}
section[data-testid="stFileUploaderDropzone"] {background:#f3f5ef;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading SousChef recipe indexes…")
def get_engine(source: str):
    if source == "Sample demo (12 recipes)":
        # Force the explicit sample mode even when old Kaggle artifacts are installed.
        return SousChefEngine(base=Path(__file__).resolve().parent / "data" / "no-artifacts-demo")
    return SousChefEngine()

st.markdown("""<section class="hero"><div class="eyebrow">A smarter way to cook</div>
<div class="hero-title">SousChef AI <span style="font-size:1.25rem; font-family:'DM Sans'">✦</span></div>
<div class="hero-sub">From what’s in your pantry to what’s on your plate. Discover recipes, explore cuisines, and ask cooking questions with evidence-backed AI.</div>
</section>""", unsafe_allow_html=True)

artifact_manifest = Path(__file__).resolve().parent / "artifacts" / "manifest.json"
available_sources = (["Kaggle artifacts", "Sample demo (12 recipes)"]
                     if artifact_manifest.exists() else ["Sample demo (12 recipes)"])
source_choice = st.sidebar.selectbox("Recipe data source", available_sources,
                                    help="Kaggle artifacts use your exported trained model; the sample contains 12 demo recipes.")
try:
    engine = get_engine(source_choice)
except Exception as exc:
    st.error("Unable to initialize SousChef AI.")
    st.code(str(exc))
    st.info("Check the Kaggle artifact export files in `artifacts/`, or temporarily move `manifest.json` to launch sample-data mode.")
    st.stop()

with st.sidebar:
    st.markdown("### 🥘 Kitchen settings")
    if engine.demo:
        st.info("**Sample mode** — 12 example recipes. Uses TF-IDF (not your trained FAISS model).")
        st.caption(f"{len(engine.recipes)} sample recipes · {len(engine.docs)} cooking sources")
    else:
        if len(engine.recipes) < 100:
            st.warning(f"**Tiny Kaggle export: only {len(engine.recipes)} recipes.** FAISS is connected, but it cannot recommend recipes that weren't exported. Switch to 'Sample demo' to explore 12 examples, or rerun Kaggle with Food.com attached and re-export 20,000+ recipes.")
        else:
            st.success("**Kaggle models connected** — FAISS semantic search enabled.")
        st.caption(f"{len(engine.recipes):,} recipes · {len(engine.docs):,} RAG sources")
    st.divider()
    st.markdown("**Personalization**")
    default_diet = st.selectbox("Diet preference", ["Any", "Vegetarian", "Vegan", "Gluten-free"])
    max_minutes = st.slider("Maximum cook time", min_value=10, max_value=180, value=45, step=5)
    banned = st.text_input("Ingredients to exclude", placeholder="e.g., peanuts, mushrooms")
    user_id = st.text_input("Food.com user ID (optional)", placeholder="Existing dataset user ID")
    st.caption("Diet/exclusion detection uses keyword heuristics; it does not guarantee allergen safety. Always check full ingredient labels.")
    st.divider()
    if st.button("↻ Reload model artifacts", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    st.caption("Notebook training → Export saved artifacts → Streamlit inference. No separate API server.")

cols = st.columns(4)
for col, label, value in zip(cols,
    ["Indexed recipes", "RAG documents", "Recipe retriever", "Cuisine classifier"],
    [f"{len(engine.recipes):,}", f"{len(engine.docs):,}", "FAISS + embeddings" if not engine.demo else "TF-IDF (demo)", "Trained" if engine.cuisine_model is not None else "Not connected"]):
    col.metric(label, value)

st.write("")
tab_recipe, tab_chat, tab_cuisine, tab_analytics = st.tabs(["🍳 Recipe Discovery", "💬 Ask SousChef", "🌍 Cuisine Predictor", "📊 Model Analytics"])

with tab_recipe:
    st.markdown("### Find your next favorite dish")
    st.caption("Semantic retrieval + ingredient coverage + Bayesian recipe ratings + optional collaborative filtering.")
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        ingredients = st.text_area("What ingredients do you have?", value="tomato, garlic, chickpeas, onion", height=105)
        goal = st.text_input("What are you in the mood for?", value="quick comforting vegetarian dinner")
    with right:
        count = st.slider("Recipes to show", 3, 12, 6)
        enable_rerank = st.checkbox("Cross-encoder reranking", disabled=engine.demo, help="Uses the trained reranking model and may download its weights first time.")
        st.markdown("<div class='micro'>Your dietary, cooking-time and exclusion preferences are in the sidebar.</div>", unsafe_allow_html=True)
    run = st.button("Refresh recommendations →", type="primary", key="discover")
    st.caption("Recommendations update when you change ingredients, goal or filters (finish text input by pressing Enter or clicking outside).")
    # Re-evaluate all results as *any* recipe input or filter changes.
    signature = (source_choice, ingredients, goal, default_diet, max_minutes,
                 banned, user_id, count, enable_rerank)
    if run or signature != st.session_state.get("recipe_signature"):
        try:
            with st.spinner("Finding recipes for your current choices…"):
                results = engine.recommend(ingredients, goal, default_diet, max_minutes,
                                           banned, user_id, count, enable_rerank)
            st.session_state["results"] = results
            st.session_state["recipe_signature"] = signature
        except Exception as exc:
            st.session_state.pop("results", None)
            st.error(f"Recommendation failed: {exc}")
    results = (st.session_state.get("results")
               if st.session_state.get("recipe_signature") == signature else None)
    if results is not None:
        if results.empty:
            st.info("No recipes matched your settings. Increase the maximum time or relax some filters.")
        else:
            st.markdown(f"#### {len(results)} recommended recipes")
            if len(results) < count:
                st.info(f"Showing {len(results)} of {count} requested. Only {len(engine.recipes):,} recipes are loaded, and time/diet/exclusion filters may reduce matches. For more choices, export the full Food.com dataset from Kaggle.")
            for rank, row in enumerate(results.itertuples(), start=1):
                st.markdown(f"**{rank:02d}. {row.name}** · {int(row.minutes_clean)} min · rating {row.bayesian_rating:.2f}/5")
                a, b, c = st.columns(3)
                a.caption(f"Ingredient match · {row.ingredient_score:.0%}")
                b.caption(f"Semantic score · {row.semantic_score:.2f}")
                c.caption(f"Overall ranking · {row.final_score:.2f} (relative score)")
                with st.expander("View ingredients, instructions and why it matched"):
                    st.markdown("**Ingredients:** " + ", ".join(row.ingredients_list))
                    st.markdown("**Already in your pantry:** " + (", ".join(row.matched_ingredients) or "None detected"))
                    st.markdown("**Other ingredients needed:** " + (", ".join(row.missing_ingredients) or "None"))
                    if row.steps_list:
                        st.markdown("**Steps**")
                        for i, s in enumerate(row.steps_list[:18], 1):
                            st.write(f"{i}. {s}")
                    st.caption("Ranked from semantic similarity, ingredient coverage, weighted reviews, cook time, and profile history where available.")
                st.divider()
    else:
        st.info("Enter pantry ingredients to see your personalized results.")

with tab_chat:
    st.markdown("### Your cooking questions, grounded in recipes")
    st.caption("See exactly which culinary notes or recipes were retrieved. Generated replies are optional; evidence retrieval works without an API key.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    colmode, colkey = st.columns([.9, 1.1])
    with colmode:
        response_mode = st.selectbox("Answer mode", ["Evidence only (no API key)", "OpenAI LLM + RAG"])
    api_key = ""
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if response_mode.startswith("OpenAI"):
        with colkey:
            api_key = st.text_input("OpenAI key (not saved)", type="password", help="Or set OPENAI_API_KEY / st.secrets on your own machine.")
        if not api_key:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY", "")
            except Exception:
                api_key = os.getenv("OPENAI_API_KEY", "")
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.write(item["content"])
            if item.get("sources"):
                with st.expander("View retrieved sources"):
                    for s in item["sources"]:
                        st.caption(f"{s['source']} · relevance {s['score']:.3f}")
                        st.write(s["text"][:900])
    question = st.chat_input("Ask about a recipe, a substitute, or a cooking problem…")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        try:
            answer, evidence = engine.answer(question, llm=response_mode.startswith("OpenAI"), key=api_key, model=model_name)
        except Exception as exc:
            answer = f"Could not generate the answer: {exc}"
            evidence = pd.DataFrame()
        sources = evidence[["source", "score", "text"]].to_dict("records") if not evidence.empty else []
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()
    if st.button("Clear conversation", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

with tab_cuisine:
    st.markdown("### Predict the cuisine from its ingredients")
    st.caption("TF-IDF + trained Logistic Regression / Linear SVM from the What's Cooking? Kaggle dataset.")
    ci = st.text_area("Ingredients to classify", value="cumin, coriander, turmeric, ginger, chickpeas", height=100)
    if st.button("Predict cuisine", type="primary", key="predict"):
        if engine.cuisine_model is None:
            st.warning("The trained cuisine classifier has not been exported yet. Export from the Kaggle notebook and reload artifacts here.")
        elif not ci.strip():
            st.warning("Enter some ingredients first.")
        else:
            try:
                predicted = engine.cuisine(ci)
                st.success(f"Predicted cuisine: **{predicted.title()}**")
                st.caption("Cuisine classification is probabilistic inference from ingredient patterns; individual dishes can belong to multiple traditions.")
            except Exception as exc:
                st.error(str(exc))
    if not engine.cuisine_results.empty:
        st.markdown("#### Trained model comparison")
        st.dataframe(engine.cuisine_results, use_container_width=True, hide_index=True)

with tab_analytics:
    st.markdown("### Inside the SousChef models")
    st.caption("Descriptive analytics and real notebook evaluation results. No generated/fabricated accuracy figures.")
    a, b, c = st.columns(3)
    a.metric("Median cook time", f"{engine.recipes.minutes_clean.median():.0f} min")
    b.metric("Recipes with reviews", f"{int((engine.recipes.review_count > 0).sum()):,}")
    c.metric("Vegetarian candidates*", f"{int(engine.recipes.is_vegetarian.sum()):,}")
    plotdata = engine.recipes[engine.recipes.minutes_clean.between(1, 180)]
    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.plotly_chart(px.histogram(plotdata, x="minutes_clean", nbins=30, title="Cooking time distribution", labels={"minutes_clean": "Minutes"}, color_discrete_sequence=["#337453"]), use_container_width=True)
    with p2:
        counts = pd.Series([x for xs in engine.recipes.ingredients_list for x in xs]).value_counts().head(12)
        st.plotly_chart(px.bar(x=counts.values[::-1], y=counts.index[::-1], orientation="h", title="Most common ingredients", labels={"x": "Recipes", "y": "Ingredient"}, color_discrete_sequence=["#8B9D66"]), use_container_width=True)
    st.caption("*Dietary detection is heuristic and must not be used to guarantee avoidance of meat or allergens.")
    st.markdown("#### Evaluation from Kaggle training")
    if engine.metrics:
        for label, val in engine.metrics.items():
            if isinstance(val, dict) and val:
                st.markdown(f"**{label.replace('_', ' ').title()}**")
                st.dataframe(pd.DataFrame([val]), use_container_width=True, hide_index=True)
        if not engine.cuisine_results.empty:
            st.markdown("**Cuisine classification model comparison**")
            st.dataframe(engine.cuisine_results, use_container_width=True, hide_index=True)
    else:
        st.info("Train/evaluate in Kaggle, export artifacts, and your real retrieval and classification metrics will appear here.")
    st.caption("The notebook's retrieval HitRate and MRR measure whether withheld recipe documents can be found—not user satisfaction or food quality.")

st.markdown("<hr style='border:0;border-top:1px solid #DEE8DF;margin-top:2.5rem'><div class='micro'>SousChef AI · Educational cooking support. Always verify allergens, safe cooking temperatures, and food storage separately.</div>", unsafe_allow_html=True)
