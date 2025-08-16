# tagfilter.py
def get_filtered_foods(meal_type, liked_tags, target_energy, meal_macros, foods, user_type="veg"):
    total = meal_macros['protein_g'] + meal_macros['carbs_g'] + meal_macros['fat_g']
    ideal_macros = [
        meal_macros['protein_g'] / total,
        meal_macros['carbs_g'] / total,
        meal_macros['fat_g'] / total
    ]

    required_tags = {meal_type}  # Meal type is mandatory
    preferred_tags = set()

    # Add regional preferences as preferred but not mandatory
    if 'south-indian' in liked_tags:
        preferred_tags.add('south-indian')
    if 'north-indian' in liked_tags:
        preferred_tags.add('north-indian')

    # Enforce vegetarian filter if user_type is 'veg'
    if user_type == "veg":
        required_tags.add('veg')

    filtered = []
    for food in foods:
        food_tags = set(food.get("tags", []))
        if required_tags.issubset(food_tags):
            filtered.append(food)

    # Sort by preferred tags if any
    if preferred_tags:
        filtered = sorted(
            filtered,
            key=lambda x: len(set(x.get("tags", [])).intersection(preferred_tags)),
            reverse=True
        )

    print(f"→ {len(filtered)} foods selected for {meal_type} after tag filtering (user_type: {user_type}).")
    if not filtered:
        print(f"⚠️ Warning: No foods found for {meal_type}. Falling back to meal_type only.")
        filtered = [food for food in foods if meal_type in food.get("tags", [])]
        if user_type == "veg":
            filtered = [food for food in filtered if 'veg' in food.get("tags", [])]

    return filtered, ideal_macros