"""Quick offline smoke checks for the local twelve-recipe demo.

Run with: python test_dashboard.py
"""
from pathlib import Path
from souschef_core import SousChefEngine

engine = SousChefEngine(base=Path(__file__).resolve().parent / 'data' / 'no-artifacts-demo')
assert engine.demo
assert len(engine.recipes) == 12
assert len(engine.docs) > 0

cases = [
    ('tomato, basil, pasta', 'Italian pasta', 'Vegetarian'),
    ('rice, peas, soy sauce', 'vegetable fried rice', 'Vegan'),
    ('cumin, chickpeas, tomato', 'Indian curry', 'Vegetarian'),
]
tops = []
for pantry, goal, diet in cases:
    result = engine.recommend(pantry, goal, diet=diet, max_minutes=60, top_k=7)
    assert not result.empty
    assert len(result) == 7, (goal, len(result))
    tops.append(result.iloc[0]['name'])
    assert all(result['is_vegetarian'] if diet == 'Vegetarian' else result['is_vegan'])
assert len(set(tops)) == 3, tops
excluded = engine.recommend('mushroom, onion, rice', 'risotto', exclusions='mushroom', max_minutes=60, top_k=10)
assert all('mushroom' not in ' '.join(xs) for xs in excluded['ingredients_list'])
answer, evidence = engine.answer('How do I thicken a watery sauce?')
assert len(evidence) > 0 and '[S1]' in answer
print('PASS: 12 demo recipes, 3 distinct top recommendations, diet and exclusion filters, RAG evidence')
