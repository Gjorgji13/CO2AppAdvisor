class CO2Service:

    def __init__(self, foods):
        self.foods = foods

    def calculate_meal(self, meal):

        total_co2 = 0
        total_calories = 0
        protein = 0
        fat = 0
        carbs = 0

        for item in meal:

            food = self.foods[item["food"]]
            kg = item["grams"] / 1000

            total_co2 += kg * food["co2_per_kg"]
            total_calories += food["calories"] * (item["grams"]/100)
            protein += food["protein"] * (item["grams"]/100)
            fat += food["fat"] * (item["grams"]/100)
            carbs += food["carbs"] * (item["grams"]/100)

        return {
            "co2": total_co2,
            "calories": total_calories,
            "protein": protein,
            "fat": fat,
            "carbs": carbs
        }