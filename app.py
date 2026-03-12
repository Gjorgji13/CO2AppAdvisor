import random
import pandas as pd
import requests
import os
from io import StringIO
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- CONFIG & MODELS ---
model = SentenceTransformer("all-MiniLM-L6-v2")

INDICATORS = {
    "drinks": [
        "beverage", "drink", "liquid", "brew", "soda", "coffee", "tea", "juice",
        "beverage_category", "beverage_prep", "serving_size_ml", "caffeine"
    ],
    "food": [
        "food", "ingredient", "item", "product", "name", "shrt_desc",
        "describe", "description", "item_description", "food_name", "common_name",
        "scientific_name", "commodity", "food_item", "main_ingredient", "label",
        "product_name", "generic_name", "food_product", "item_name"
    ],
    "calories": [
        "calories", "kcal", "energy", "energ_kcal", "energy_100g", "caloric_value",
        "kilocalories", "energy_value", "joules", "kilojoules", "kj", "energy_kj"
    ],
    "protein": [
        "protein", "prot", "proteins", "protein_g", "protein_content", "total_protein"
    ],
    "carbs": [
        "carbohydrates", "carbs", "carb", "carbohydrt_g", "total_carbohydrates", "cho"
    ],
    "fat": [
        "fat", "lipids", "lipid", "lipid_tot_g", "total_fat", "fat_content"
    ],
    "co2": [
        "co2", "carbon", "ghg", "greenhouse", "emissions", "kg_co2", "gwp",
        "global_warming", "carbon_footprint", "climate_impact"
    ],
    "water": [
        "water", "h2o", "freshwater", "liters", "irrigation", "aquatic_stress",
        "water_use", "water_footprint", "l_per_kg"
    ],
    "land": [
        "land", "area", "m2", "arable", "land_use", "land_footprint", "sq_meters"
    ],
    "eutro": [
        "eutrophication", "eutro", "po4", "phosphorus", "nitrogen", "gpo4",
        "leaching", "runoff", "nutrient_pollution", "excess_nutrients", "phosphate"
    ],
    "category": [
        "category", "group", "food_group", "type", "class", "sector", "department"
    ]
}

DIET_BASELINES = {
    "mediterranean": {"ghg": 4.88, "water": 1079.9, "land": 14.8, "eutro": 35.5},
    "western": {"ghg": 9.08, "water": 1105.4, "land": 33.1, "eutro": 51.6}
}

# --- GLOBAL STATE ---
foods = {}
recipes = {}


def process_dataset(df):
    global foods
    df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]
    col_names = df.columns.tolist()
    col_vectors = model.encode(col_names)

    mapping_report = {}
    mapping_internal = {}
    used_columns = set()

    # 1. IDENTIFIER MAPPING (Strictly for the name of the item)
    # We only look for the food/drink name first.
    id_priority = ["food", "drinks"]
    for indicator in id_priority:
        keywords = INDICATORS.get(indicator, [])
        keyword_vectors = model.encode(keywords)
        scores = cosine_similarity(col_vectors, keyword_vectors).max(axis=1)

        # We take the best match for the name
        best_idx = scores.argmax()
        if scores[best_idx] > 0.25:
            mapping_internal["name_col"] = col_names[best_idx]
            used_columns.add(col_names[best_idx])
            break

    if "name_col" not in mapping_internal:
        return None

    # 2. METRIC MAPPING (The math values)
    # We exclude food/drinks from here so they don't steal nutrient columns
    metric_priority = ["eutro", "co2", "land", "water", "fat", "protein", "calories", "carbs", "category"]

    for indicator in metric_priority:
        keywords = INDICATORS.get(indicator, [])
        keyword_vectors = model.encode(keywords)
        scores = cosine_similarity(col_vectors, keyword_vectors).max(axis=1)
        sorted_indices = scores.argsort()[::-1]

        # Use a high threshold for nutrients to prevent "total_protein" being grabbed by "fat"
        threshold = 0.30 if indicator in ["eutro", "co2", "land", "water"] else 0.60

        for idx in sorted_indices:
            detected_col = col_names[idx]
            if detected_col not in used_columns and scores[idx] > threshold:
                mapping_internal[indicator] = detected_col
                used_columns.add(detected_col)
                mapping_report[indicator] = {
                    "column": detected_col,
                    "confidence": round(float(scores[idx]), 2)
                }
                break

    # 3. EXTRACTION
    new_foods = {}
    name_col = mapping_internal["name_col"]

    for _, row in df.iterrows():
        if pd.isna(row[name_col]): continue

        # Standardize name
        raw_name = str(row[name_col]).strip().lower()

        # Append prep info if available to keep keys unique
        if "beverage_prep" in df.columns and pd.notnull(row["beverage_prep"]):
            raw_name += f" ({str(row['beverage_prep']).lower().strip()})"

        entry = {}
        for metric in ["calories", "protein", "carbs", "fat", "co2", "water", "land", "eutro", "category"]:
            if metric in mapping_internal:
                col = mapping_internal[metric]
                try:
                    if metric == "category":
                        entry[metric] = str(row[col]).lower().strip()
                    else:
                        val_str = str(row[col]).lower().replace('%', '').replace('g', '').replace('mg', '').strip()
                        val = float(val_str) if val_str not in ['', 'nan', 'trace'] else 0.0

                        # Unit Normalization
                        if "mg" in col: val /= 1000
                        if metric == "calories" and "kj" in col: val /= 4.184
                        if metric in ["co2", "water", "land", "eutro"] and "100g" in col: val *= 10

                        entry[metric] = round(val, 4)
                except:
                    entry[metric] = 0.0 if metric != "category" else "unknown"
            else:
                entry[metric] = 0.0 if metric != "category" else "unknown"

        new_foods[raw_name] = entry

    foods = new_foods
    return mapping_report

def load_initial_data():
    global recipes
    try:
        if os.path.exists("food_dataset.csv"):
            process_dataset(pd.read_csv("food_dataset.csv"))
        if os.path.exists("recipes_dataset.csv"):
            r_df = pd.read_csv("recipes_dataset.csv")
            for _, row in r_df.iterrows():
                recipes.setdefault(row['recipe_name'], []).append({
                    "food": str(row['ingredient']).lower().strip(), "grams": row['grams']
                })
    except Exception as e:
        print(f"Startup Load Error: {e}")


load_initial_data()


# --- ROUTES ---

@app.route("/")
def index():
    return render_template("index.html", foods=list(foods.keys()), predefined_meals=list(recipes.keys()))


@app.route("/api/upload_local", methods=["POST"])
def upload_local():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    df = pd.read_csv(request.files['file'])

    # This now returns the detailed report { 'food': {'column': '...', 'confidence': 0.7}, ... }
    mapping_report = process_dataset(df)

    if mapping_report is None:
        return jsonify({"status": "error", "message": "Could not detect a 'food' column."}), 400

    return jsonify({
        "status": "success",
        "count": len(foods),
        "mapping": mapping_report  # This is the key part!
    })


@app.route("/api/load_external_data", methods=["POST"])
def load_external():
    url = request.json.get("url")
    resp = requests.get(url)
    df = pd.read_csv(StringIO(resp.text))
    mapping = process_dataset(df)
    return jsonify({"status": "success", "count": len(foods)})


@app.route("/api/calculate", methods=["POST"])
def calculate():
    data = request.json
    ings = data.get("ingredients", [])
    servings = max(1, data.get("servings", 1))
    base_key = data.get("baseline", "mediterranean")
    base = DIET_BASELINES.get(base_key, DIET_BASELINES["mediterranean"])

    totals = {k: 0.0 for k in ["co2", "water", "land", "eutro", "calories", "protein", "carbs", "fat"]}
    details = []

    for ing in ings:
        name = ing.get("food", "").lower().strip()
        grams = float(ing.get("grams", 0))
        if name in foods:
            f = foods[name]
            kg, g100 = grams / 1000, grams / 100
            for k in ["co2", "water", "land", "eutro"]: totals[k] += f.get(k, 0) * kg
            for k in ["calories", "protein", "carbs", "fat"]: totals[k] += f.get(k, 0) * g100
            details.append({"food": name, "co2": round(f.get("co2", 0) * kg, 2)})

    return jsonify({
        "total_co2": round(totals["co2"], 2),
        "total_water": round(totals["water"], 2),
        "total_land": round(totals["land"], 2),
        "total_eutro": round(totals["eutro"], 2),
        "per_serving": {k: round(v / servings, 2) for k, v in totals.items()},
        "comparison": {"vs_baseline_co2_pct": round((totals["co2"] / base["ghg"]) * 100, 1)},
        "details": details
    })


@app.route("/api/optimize", methods=["POST"])
def optimize():
    ingredients = request.json.get("ingredients", [])
    suggestions = []
    for ing in ingredients:
        name = ing.get("food", "").lower().strip()
        if name not in foods: continue

        curr = foods[name]
        curr_cat = curr.get("category", "unknown")  # New Guardrail
        curr_co2 = max(curr.get("co2", 0), 0.001)
        curr_eni = (curr.get("protein", 0) + 0.5 * curr.get("carbs", 0)) / curr_co2

        for alt_name, alt_info in foods.items():
            if alt_name == name: continue

            # --- NUTRITIONAL GUARDRAIL ---
            # Only suggest within same category or if protein is comparable
            alt_cat = alt_info.get("category", "unknown")
            is_same_group = (curr_cat == alt_cat and curr_cat != "unknown")
            protein_match = alt_info.get("protein", 0) >= (curr.get("protein", 0) * 0.7)

            if not (is_same_group or protein_match): continue

            alt_co2 = max(alt_info.get("co2", 0), 0.001)
            alt_eni = (alt_info.get("protein", 0) + 0.5 * alt_info.get("carbs", 0)) / alt_co2

            if alt_eni > curr_eni:
                savings = round(((curr_co2 - alt_co2) / curr_co2) * 100, 1)
                if savings > 15:  # Higher threshold for research significance
                    suggestions.append({
                        "original": name,
                        "replacement": alt_name,
                        "co2_saved_pct": savings,
                        "reason": "Similar Protein" if protein_match else f"Same Group ({curr_cat})"
                    })
    return jsonify({"suggestions": suggestions})


@app.route("/api/get_meal", methods=["POST"])
def get_meal():
    m_name = request.json.get("meal")
    return jsonify({"ingredients": recipes.get(m_name, [])})


if __name__ == "__main__":
    app.run(debug=True, port=5050)