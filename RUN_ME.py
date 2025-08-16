import sys
import pandas as pd
from foods_dataset import foods
from fruits_dataset import fruits
from beverages_dataset import beverages
import os
import pickle
import multimealplanner

EXAMPLE_CSV = "final_selected_meals.csv"

# Create fast cost lookup, handling both 'name' and 'Name' keys
cost_lookup = {item['name'] if 'name' in item else item.get('Name', ''): item.get('cost', 0) for item in foods + fruits + beverages}

def calculate_total_cost(csv_file, fruit_name, beverage_name):
    total = cost_lookup.get(fruit_name, 0) + cost_lookup.get(beverage_name, 0)
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            for food in row[1:]:  # Skip MealType
                if pd.isna(food):
                    continue
                food = food.strip()
                if food in cost_lookup:
                    total += cost_lookup[food]
    return round(total, 2)

def main(use_cache=False, reset_history=False, strict_filtering=False, max_attempts=2):
    attempts = 0
    budget = None
    fruit_name = None
    beverage_name = None
    run_data = []  # Store (cost, df) for each run

    while attempts < max_attempts:
        multimealplanner.run_pipeline(use_cached_input=use_cache or attempts > 0, 
                                    reset_history=reset_history and attempts == 0, 
                                    strict_filtering=strict_filtering)

        if not os.path.exists(EXAMPLE_CSV):
            print("❌ Meal plan CSV not found after execution.")
            return

        # Get fruit, beverage, and budget from user_data.pkl
        if os.path.exists("user_data.pkl"):
            with open("user_data.pkl", "rb") as f:
                user_data = pickle.load(f)
            fruit_name = user_data['fruit']['name'] if user_data['fruit'] and 'name' in user_data['fruit'] else user_data['fruit'].get('Name', 'None') if user_data['fruit'] else 'None'
            beverage_name = user_data['beverage']['name'] if user_data['beverage'] and 'name' in user_data['beverage'] else user_data['beverage'].get('Name', 'None') if user_data['beverage'] else 'None'
            budget = user_data.get('budget_inr')
            if budget is None:
                print("❌ budget_inr not found in user_data.pkl")
                return
        else:
            print("❌ user_data.pkl not found. Please ensure Phase0final writes the user data.")
            return

        total_cost = calculate_total_cost(EXAMPLE_CSV, fruit_name, beverage_name)
        final_df = pd.read_csv(EXAMPLE_CSV) if os.path.exists(EXAMPLE_CSV) else None
        run_data.append((total_cost, final_df))

        if total_cost <= budget:
            print(f"✅ Meal plan fits within budget: ₹{total_cost} <= ₹{budget}")
            print("\n📋 Final Meal Plan:")
            print(final_df.to_string(index=False))
            return

        print("⚠️ Budget was insufficient so rerunning")
        attempts += 1

    # If budget is still exceeded, select the run with the lowest cost
    if run_data:
        min_cost, min_cost_df = min(run_data, key=lambda x: x[0])
        print(f"\n❌ Failed to generate a meal plan within budget after {max_attempts} attempts. Lowest cost: ₹{min_cost}, Budget: ₹{budget}")
        if min_cost_df is not None:
            print("\n📋 Cheapest Meal Plan (from run {0}):".format(run_data.index((min_cost, min_cost_df)) + 1))
            print(min_cost_df.to_string(index=False))
        else:
            print("❌ No valid meal plan generated.")
    else:
        print(f"\n❌ Failed to generate a meal plan within budget after {max_attempts} attempts. No plans generated.")
        print("❌ No meal plan generated.")

if __name__ == "__main__":
    use_cache = "--use-cache" in sys.argv
    reset_history = "--reset-history" in sys.argv
    strict_filtering = "--strict-filtering" in sys.argv
    main(use_cache=use_cache, reset_history=reset_history, strict_filtering=strict_filtering)