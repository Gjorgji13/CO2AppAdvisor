# CO2 MyPlate - Food Carbon & Nutrition Advisor

CO2 MyPlate is a Python Flask web application that allows users to analyze the environmental impact (CO₂, water, land, eutrophication) and nutritional content of foods and recipes. Users can upload local datasets or use predefined meals, and the app provides detailed calculations and optimization suggestions for more sustainable and nutritious choices.

---

## Features

- **Environmental Impact Analysis**: Calculate CO₂ emissions, water use, land use, and eutrophication for foods and recipes.
- **Nutritional Analysis**: Track calories, protein, carbs, and fat per serving.
- **Recipe Optimization**: Suggest ingredient substitutions to reduce environmental impact while maintaining nutritional value.
- **Flexible Dataset Handling**: Upload local CSV datasets; supports multiple column names via intelligent column mapping using NLP embeddings.
- **Predefined Meals**: Quick analysis for existing recipes.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Gjorgji13/CO2AppAdvisor.git
cd CO2AppAdvisor

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python app.py

