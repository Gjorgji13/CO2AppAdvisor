from flask import jsonify, request

from app import foods


@app.route("/api/calculate", methods=["POST"])
def api_calculate():
    data = request.get_json()
    selected_foods = data.get("foods", [])

    total_co2 = 0
    total_calories = 0
    total_protein = 0
    total_carbs = 0
    total_fat = 0
    details = []

    for item in selected_foods:
        food = item.get("food")
        grams = item.get("grams", 0)
        if not food or grams <= 0:
            continue
        data_food = foods[food]
        co2 = (grams/1000) * data_food["co2_per_kg"]
        calories = (grams/100) * float(str(data_food["calories"]).split()[0])
        protein = (grams/100) * float(str(data_food["protein"]).split()[0])
        carbs = (grams/100) * float(str(data_food["carbs"]).split()[0])
        fat = (grams/100) * float(str(data_food["fat"]).split()[0])

        total_co2 += co2
        total_calories += calories
        total_protein += protein
        total_carbs += carbs
        total_fat += fat

        details.append({
            "food": food,
            "grams": grams,
            "co2": round(co2,2)
        })

    score = max(0, 100 - total_co2*10)

    return jsonify({
        "total_co2": round(total_co2,2),
        "calories": round(total_calories,2),
        "protein": round(total_protein,2),
        "carbs": round(total_carbs,2),
        "fat": round(total_fat,2),
        "score": score,
        "details": details
    })