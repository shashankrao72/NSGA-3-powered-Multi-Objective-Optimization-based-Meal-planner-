import pickle
import os
import pandas as pd
import random
from foods_dataset import foods
from fruits_dataset import fruits
from beverages_dataset import beverages
from tagfilter import get_filtered_foods
from binary_nsga_diverse_solution import NSGA3BinaryMealPlanner
from compatibility import analyze_all_meals
from onboarding_final import SarsaFoodRecommender, FoodEnvironment
from Phase0final import get_user_input_and_calculate

def run_pipeline(use_cached_input=False, reset_history=False, strict_filtering=False):
    if use_cached_input and os.path.exists("user_data.pkl"):
        with open("user_data.pkl", "rb") as f:
            user_data = pickle.load(f)
        print("✅ Loaded cached user input.")
    else:
        user_data = get_user_input_and_calculate()
        with open("user_data.pkl", "wb") as f:
            pickle.dump(user_data, f)

    adjusted_bmr = user_data['adjusted_bmr']
    budget = user_data['budget_inr']
    meal_breakdown = user_data['meal_breakdown']
    meal_distribution = user_data['meal_distribution']
    health_goal = user_data['health_goal']
    selected_fruit = user_data['fruit']
    selected_beverage = user_data['beverage']

    # Log fruit and beverage from user_data
    fruit_name = selected_fruit['name'] if selected_fruit else 'None'
    fruit_energy = selected_fruit['energy'] if selected_fruit else 0
    beverage_name = selected_beverage['name'] if selected_beverage else 'None'
    beverage_energy = selected_beverage['energy'] if selected_beverage else 0
    print(f"\n🍎 Selected Fruit: {fruit_name} (Energy: {fruit_energy:.2f} kcal)")
    print(f"🥤 Selected Beverage: {beverage_name} (Energy: {beverage_energy:.2f} kcal)")

    # Subtract fruit and beverage energy (already done in Phase0final)
    total_subtracted_energy = fruit_energy + beverage_energy
    remaining_energy = adjusted_bmr - total_subtracted_energy

    # Log meal distribution and energy targets
    print("\n📊 Energy Targets (based on user preferences):")
    for tag in ['breakfast', 'lunch', 'dinner']:
        meal_energy = (
            (meal_breakdown[tag]['carbs_g'] * 4) +
            (meal_breakdown[tag]['protein_g'] * 4) +
            (meal_breakdown[tag]['fat_g'] * 9)
        )
        print(f"{tag.capitalize()} ({meal_distribution[tag]}): {meal_energy:.2f} kcal")

    # Onboarding and NSGA-III optimization
    print("\n🍽️ Starting Onboarding...")
    actions = ['recommend', 'not_recommend']
    agent = SarsaFoodRecommender(actions)
    q_table_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.csv")
    feedback_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_history.csv")

    if reset_history:
        for file in [q_table_path, feedback_file]:
            if os.path.exists(file):
                os.remove(file)
                print(f"✅ Cleared {file}")

    user_type = user_data.get('user_type', 'veg')
    if use_cached_input and os.path.exists(q_table_path) and not reset_history:
        agent.load_q_table(q_table_path)
        print("✅ Loaded existing Q-table.")
    else:
        env = FoodEnvironment()
        while True:
            user_type = input("Are you vegetarian or non-vegetarian? (Enter 'veg' or 'non-veg'): ").strip().lower()
            if user_type in ["veg", "non-veg"]:
                break
            print("❗ Please enter a valid option: 'veg' or 'non-veg'.")
        user_data['user_type'] = user_type
        with open("user_data.pkl", "wb") as f:
            pickle.dump(user_data, f)

        subset = env.fixed_onboarding_foods_veg if user_type == "veg" else env.fixed_onboarding_foods_nonveg
        env.display_options(subset)
        selected_indices = env.get_user_selection(subset)

        liked_tags = set()
        liked_names = set()
        for i, item in enumerate(subset):
            name = item['name']
            tags = item.get('tags', [])
            action = 'recommend' if i in selected_indices else 'not_recommend'
            reward = 1 if i in selected_indices else -1
            agent.learn(name, action, reward)
            if reward == 1:
                liked_tags.update(tags)
                liked_names.add(name)

        for item in env.foods:
            name = item['name']
            tags = set(item.get('tags', []))
            if name in liked_names:
                continue
            if tags.intersection(liked_tags.intersection({"sweet", "spicy", "savoury"})):
                agent.learn(name, 'recommend', 0.5)
            elif tags.intersection(liked_tags.intersection({"south-indian", "north-indian"})):
                agent.learn(name, 'recommend', 0.4)
            elif tags.intersection(liked_tags.intersection({"rice", "chapati"})):
                agent.learn(name, 'recommend', 0.2)

        agent.save_q_table(q_table_path)

    if os.path.exists(feedback_file) and not reset_history:
        feedback_df = pd.read_csv(feedback_file)
        for _, row in feedback_df.iterrows():
            meal_type = row['meal_type']
            meal_items = row['meal_items'].split(',')
            feedback = row['feedback']
            reward = 1 if feedback == 'yes' else -1
            for food in meal_items:
                if food.strip():
                    agent.learn(food.strip(), 'recommend' if reward == 1 else 'not_recommend', reward)
        print("✅ Loaded previous feedback from feedback_history.csv.")

    filtered_foods = [item for item in foods if 'veg' in item.get('tags', [])] if user_type == "veg" else foods[:]
    print(f"✅ Using {len(filtered_foods)} foods after filtering for '{user_type}' preference.")

    # Initialize set for previously used dishes (for original filter)
    feedback_dishes = set()
    final_meals_csv = "final_selected_meals.csv"
    if os.path.exists(final_meals_csv):
        feedback_df = pd.read_csv(final_meals_csv)
        feedback_df.columns = feedback_df.columns.str.strip()
        for _, row in feedback_df.iterrows():
            dishes = [str(item).strip() for item in row[1:] if pd.notna(item)]  # Skip MealType column
            feedback_dishes.update(dishes)
        print(f"✅ Loaded {len(feedback_dishes)} unique dishes from previous meals in {final_meals_csv}")

    def filter_meal_file(filename, feedback_dishes):
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()

        # Identify all item columns (e.g., Item1, Item2, ...)
        item_columns = [col for col in df.columns if col.lower().startswith('item')]

        # Check each meal and remove at most two that contain a feedback dish
        mask = [False] * len(df)
        deletion_count = 0
        for i, row in df[item_columns].iterrows():
            if deletion_count >= 2:
                break
            items = [str(item).strip() for item in row if pd.notna(item)]
            matched_items = [item for item in items if item in feedback_dishes]
            if matched_items:
                print(f"🗑️ Removed meal {df['Meal'][i]} from {filename} due to matched item(s): {', '.join(matched_items)}")
                mask[i] = True
                deletion_count += 1

        filtered_df = df[~pd.Series(mask)]
        filtered_df.to_csv(filename, index=False)

    def filter_meal_file_strict(filename, feedback_file):
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()

        # Identify all item columns (e.g., Item1, Item2, ...)
        item_columns = [col for col in df.columns if col.lower().startswith('item')]

        # Load feedback history to get previous meals
        previous_meals = []
        if os.path.exists(feedback_file):
            feedback_df = pd.read_csv(feedback_file)
            for _, row in feedback_df.iterrows():
                items = set(str(item).strip() for item in row['meal_items'].split(',') if item.strip())
                if items:  # Only include non-empty meals
                    previous_meals.append(items)

        # Check each meal and remove at most two that match a previous meal
        mask = [False] * len(df)
        deletion_count = 0
        for i, row in df[item_columns].iterrows():
            if deletion_count >= 2:
                break
            current_items = set(str(item).strip() for item in row if pd.notna(item))
            if any(current_items == prev_meal for prev_meal in previous_meals):
                print(f"🗑️ Removed meal {df['Meal'][i]} from {filename} due to exact match with previous meal")
                mask[i] = True
                deletion_count += 1

        filtered_df = df[~pd.Series(mask)]
        filtered_df.to_csv(filename, index=False)

    meal_tags = ['breakfast', 'lunch', 'dinner']
    final_meals = []
    feedback_data = []

    for tag in meal_tags:
        print(f"\n⚙️ Generating meal candidates for: {tag.upper()}")
        filtered_meal_foods, ideal_macros = get_filtered_foods(tag, set(), adjusted_bmr, meal_breakdown[tag], filtered_foods, user_type=user_type)
        meal_energy_target = (
            (meal_breakdown[tag]['carbs_g'] * 4) +
            (meal_breakdown[tag]['protein_g'] * 4) +
            (meal_breakdown[tag]['fat_g'] * 9)
        )
        planner = NSGA3BinaryMealPlanner(filtered_meal_foods, meal_energy_target, ideal_macros)

        meal_rows = []
        all_selected_indices = set()

        for i in range(5):
            population, objectives, _ = planner.optimize(pop_size=100, generations=100, avoid_set=all_selected_indices, candidate_num=i+1)
            best_idx = sorted(range(len(objectives)), key=lambda i: sum(objectives[i]))[0]
            individual = population[best_idx]
            selected_indices = [i for i, val in enumerate(individual) if val > 0.5]
            meal = [filtered_meal_foods[i]['name'] for i in selected_indices]
            meal_rows.append([f"{tag}_meal{i+1}"] + meal)
            all_selected_indices.update(selected_indices)

        max_len = max(len(row) for row in meal_rows)
        for row in meal_rows:
            while len(row) < max_len:
                row.append("")

        header = ["Meal"] + [f"Item{i+1}" for i in range(max_len - 1)]
        df = pd.DataFrame(meal_rows, columns=header)
        temp_csv = f"{tag}_candidates.csv"
        df.to_csv(temp_csv, index=False)

        # Choose filtering function based on strict_filtering flag
        if strict_filtering:
            filter_meal_file_strict(temp_csv, feedback_file)
        else:
            filter_meal_file(temp_csv, feedback_dishes)

        compat_results = analyze_all_meals(temp_csv, filtered_foods)
        compat_results.sort(key=lambda x: x['compatibility_score'], reverse=True)
        # Remove at most one meal with lowest compatibility score, ensuring at least 2 meals remain
        if len(compat_results) > 2:
            removed_meal = compat_results[-1]
            removed_meal_name = df.loc[df['Meal'] == removed_meal['meal'][0], 'Meal'].iloc[0] if not df.loc[df['Meal'] == removed_meal['meal'][0], 'Meal'].empty else 'unknown'
            rules = removed_meal.get('applied_rules', [])
            rules_str = ', '.join([f"{r['rule']}: {r['score']:+.2f} ({r['type']})" for r in rules]) if rules else 'none'
            print(f"🗑️ Removed meal  due to low compatibility: score={removed_meal['compatibility_score']:.2f}, rules={rules_str}")
            top_meals = compat_results[:-1]
        else:
            top_meals = compat_results

        cleaned_meals = [res['meal'] for res in top_meals]
        if not cleaned_meals:
            print(f"⚠️ Warning: No compatible meals for {tag}. Falling back to a random candidate meal.")
            df = pd.read_csv(temp_csv)
            cleaned_meals = [[str(item).strip() for item in row[1:] if pd.notna(item)] for _, row in df.iterrows()][:1]  # Take first meal

        scored_meals = []
        for meal in cleaned_meals:
            score = sum(agent.q_table.loc[f, 'recommend'] if f in agent.q_table.index else 0 for f in meal)
            scored_meals.append((meal, score))

        best_plan, _ = max(scored_meals, key=lambda x: x[1], default=(cleaned_meals[0], 0))  # Fallback to first meal
        final_meals.append([tag] + best_plan)

    # Interactive feedback loop
    print("\n📋 Final Selected Meal Plan:")
    for meal in final_meals:
        meal_type = meal[0]
        items = [item for item in meal[1:] if item]
        print(f"{meal_type.capitalize()}:")
        for item in items:
            print(f" - {item}")

        while True:
            feedback = input(f"Did you like the {meal_type} meal? (yes/no): ").strip().lower()
            if feedback in ['yes', 'no']:
                break
            print("Please enter 'yes' or 'no'.")

        reward = 1 if feedback == 'yes' else -1
        for food in items:
            if food:
                agent.learn(food, 'recommend' if reward == 1 else 'not_recommend', reward)

        feedback_data.append({
            'meal_type': meal_type,
            'meal_items': ','.join(items),
            'feedback': feedback
        })

    feedback_df = pd.DataFrame(feedback_data)
    feedback_df.to_csv(feedback_file, index=False)
    print(f"✅ Feedback saved to {feedback_file}")
    agent.save_q_table(q_table_path)

    # Calculate total energy, cost, and macro breakdown
    nutrient_lookup = {item['name'] if 'name' in item else item.get('Name', ''): {
        'energy': item.get('energy', item.get('Energy', item.get('Energy (kcal)', 0))),
        'cost': item.get('cost', item.get('Cost', item.get('Cost (INR)', 0))),
        'carbs_g': item.get('carbs_g', item.get('Carbs', item.get('Carbohydrates', item.get('Carbs (g)', 0)))),
        'protein_g': item.get('protein_g', item.get('Protein', item.get('Protein (g)', 0))),
        'fat_g': item.get('fat_g', item.get('Fat', item.get('Fats', item.get('Fat (g)', 0))))
    } for item in foods + fruits + beverages}

    total_energy = 0
    total_cost = 0
    total_carbs = 0
    total_protein = 0
    total_fat = 0

    # Add fruit and beverage contributions
    if fruit_name in nutrient_lookup:
        total_energy += nutrient_lookup[fruit_name]['energy']
        total_cost += nutrient_lookup[fruit_name]['cost']
        total_carbs += nutrient_lookup[fruit_name]['carbs_g']
        total_protein += nutrient_lookup[fruit_name]['protein_g']
        total_fat += nutrient_lookup[fruit_name]['fat_g']
    if beverage_name in nutrient_lookup:
        total_energy += nutrient_lookup[beverage_name]['energy']
        total_cost += nutrient_lookup[beverage_name]['cost']
        total_carbs += nutrient_lookup[beverage_name]['carbs_g']
        total_protein += nutrient_lookup[beverage_name]['protein_g']
        total_fat += nutrient_lookup[beverage_name]['fat_g']

    # Add meal contributions
    for meal in final_meals:
        items = meal[1:]
        for item in items:
            if item and item in nutrient_lookup:
                total_energy += nutrient_lookup[item]['energy']
                total_cost += nutrient_lookup[item]['cost']
                total_carbs += nutrient_lookup[item]['carbs_g']
                total_protein += nutrient_lookup[item]['protein_g']
                total_fat += nutrient_lookup[item]['fat_g']

    print(f"\n🔢 Total Energy for the Day: {round(total_energy)} kcal")
    #print(f"💪 Macro Breakdown: Carbs: {round(total_carbs, 2)}g, Protein: {round(total_protein, 2)}g, Fat: {round(total_fat, 2)}g")
    print(f"💰 Total Cost for the Day: ₹{round(total_cost, 2)}")

    max_len = max(len(m) for m in final_meals)
    for row in final_meals:
        while len(row) < max_len:
            row.append("")

    df = pd.DataFrame(final_meals, columns=["MealType"] + [f"Item{i+1}" for i in range(max_len - 1)])
    df.to_csv("final_selected_meals.csv", index=False)
    print("\n✅ Final selected meals saved to final_selected_meals.csv")

if __name__ == "__main__":
    import sys
    reset = "--reset-history" in sys.argv
    use_cache = "--use-cache" in sys.argv
    strict_filtering = "--strict-filtering" in sys.argv
    run_pipeline(use_cached_input=use_cache, reset_history=reset, strict_filtering=strict_filtering)