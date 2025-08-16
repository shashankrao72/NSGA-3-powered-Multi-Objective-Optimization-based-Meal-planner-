import numpy as np
import pandas as pd
import random
import os
from foods_dataset import foods  # Make sure this exists and contains the food data

# === Environment ===
class FoodEnvironment:
    def __init__(self):
        
        self.foods = foods
        self.n_items = len(self.foods)
        
        # Optimized set of 10 foods covering maximum tag diversity
        # inside FoodEnvironment.__init__()

        # === Fixed onboarding sets for veg and non-veg ===
        self.fixed_onboarding_foods_veg = [
            {"name": "Plain Dosa", "energy": 130, "cost": 60, "protein": 2.7, "carbs": 30, "fat": 2.5, "tags": ["breakfast", "veg", "savoury", "south-indian"]},
            {"name": "Vegetable Khichdi (1 plate)", "energy": 190, "cost": 75, "protein": 5.5, "carbs": 28, "fat": 6, "tags": ["lunch", "veg", "savoury", "rice"]},
            {"name": "Sprouts Salad (1 bowl)", "energy": 140, "cost": 35, "protein": 10, "carbs": 16, "fat": 2, "tags": ["protein", "veg"]},
            {"name": "Aloo Paratha (1 pc)", "energy": 290, "cost": 70, "protein": 6.5, "carbs": 45, "fat": 8.5, "tags": ["breakfast", "veg", "gluten", "chapati"]},
            {"name": "1 Plain Roti + Dal Tadka", "energy": 244, "cost": 85, "protein": 9.5, "carbs": 35, "fat": 4.7, "tags": ["veg", "gluten", "iron"]},
            {"name": "Idli with Sambar", "energy": 210, "cost": 50, "protein": 6.1, "carbs": 38, "fat": 2.2, "tags": ["breakfast", "veg", "south-indian"]},
            {"name": "1 Roti + Palak Paneer", "energy": 284, "cost": 135, "protein": 12.8, "carbs": 23.5, "fat": 11.5, "tags": ["veg", "dairy", "north-indian"]},
            {"name": "Curd Rice", "energy": 210, "cost": 70, "protein": 6, "carbs": 32, "fat": 6, "tags": ["rice", "veg", "dairy"]},
            {"name": "Oats Porridge with Milk", "energy": 210, "cost": 40, "protein": 6, "carbs": 30, "fat": 6, "tags": ["veg", "dairy", "breakfast"]},
            {"name": "Poha", "energy": 180, "cost": 50, "protein": 3.8, "carbs": 35, "fat": 3.5, "tags": ["veg", "breakfast", "savoury"]}
        ]

        self.fixed_onboarding_foods_nonveg = [
    # 1. Non-veg + Breakfast + Protein
    {"name": "Boiled Egg (1 pc)", "energy": 77, "cost": 7, "protein": 6.3, "carbs": 0.6, "fat": 5.3,
     "tags": ["breakfast", "egg", "non-veg", "protein"]},

    # 2. Premium Chicken + Rice
    {"name": "Chicken Biryani", "energy": 350, "cost": 150, "protein": 18, "carbs": 45, "fat": 12,
     "tags": ["non-veg", "rice", "lunch", "savoury", "premium"]},


    # 3. North Indian + Chicken + Chapati
    {"name": "1 Roti + Butter Chicken", "energy": 415, "cost": 168, "protein": 20, "carbs": 23, "fat": 22.9,
     "tags": ["non-veg", "chapati", "north-indian", "savoury"]},

    # 4. High Protein Veg + Breakfast
    {"name": "Sprouts Salad (1 bowl)", "energy": 140, "cost": 35, "protein": 10, "carbs": 16, "fat": 2,
     "tags": ["veg", "protein", "healthy", "breakfast"]},

    # 5. Traditional South Indian Veg + Breakfast
    {"name": "Plain Dosa", "energy": 130, "cost": 60, "protein": 2.7, "carbs": 30, "fat": 2.5,
     "tags": ["breakfast", "veg", "south-indian", "savoury"]},

    # 6. Comfort Veg + Rice + Dairy
    {"name": "Curd Rice", "energy": 210, "cost": 70, "protein": 6, "carbs": 32, "fat": 6,
     "tags": ["veg", "dairy", "rice", "lunch", "cooling"]},

    # 7. Chapati + Paneer Combo (veg alternative to butter chicken)
    {"name": "1 Roti + Palak Paneer", "energy": 284, "cost": 135, "protein": 12.8, "carbs": 23.5, "fat": 11.5,
     "tags": ["veg", "north-indian", "dairy", "chapati", "iron"]},

    # 8. Sweet + Dairy + Breakfast
    {"name": "Oats Porridge with Milk", "energy": 210, "cost": 40, "protein": 6, "carbs": 30, "fat": 6,
     "tags": ["veg", "breakfast", "dairy", "sweet"]},

    # 9. Non-veg + Rice + Curry (keep one premium combo)
    {"name": "Brown Rice + Chicken Curry", "energy": 421, "cost": 175, "protein": 20, "carbs": 28, "fat": 23.3,
     "tags": ["non-veg", "rice", "dinner", "premium", "spicy"]},

    # 10. Non-veg North Indian + Chapati
    {"name": "1 Butter Naan + Butter Chicken", "energy": 450, "cost": 176.25, "protein": 21, "carbs": 26.25, "fat": 26.25,
     "tags": ["non-veg", "chapati", "north-indian", "dairy", "premium"]}
]



    def get_random_foods(self, n=10):
        return random.sample(self.foods, n)

    def display_options(self, subset):
        print("\nSelect your favorite foods by entering their numbers (comma-separated):")
        for i, item in enumerate(subset):
            print(f"{i+1}. {item['name']}")
        print()

    def get_user_selection(self, subset):
        selected = input("Enter numbers of dishes you like (e.g., 1,4,6): ")
        indices = set()
        for x in selected.split(","):
            x = x.strip()
            if x.isdigit():
                idx = int(x) - 1
                if 0 <= idx < len(subset):
                    indices.add(idx)
        return indices

# === Agent ===
class SarsaFoodRecommender:
    def __init__(self, actions, learning_rate=0.1, gamma=0.9):
        self.q_table = pd.DataFrame(columns=actions)
        self.lr = learning_rate
        self.gamma = gamma
        self.actions = actions

    def initialize_q_table(self, all_food_names):
        for name in all_food_names:
            self._check_state(name)


    def _check_state(self, state):
        if state not in self.q_table.index:
            new_row = pd.Series([0] * len(self.actions), index=self.actions, name=state)
            self.q_table = pd.concat([self.q_table, new_row.to_frame().T])

    def learn(self, state, action, reward):
        self._check_state(state)
        predict = self.q_table.loc[state, action]
        target = reward
        updated_value = predict + self.lr * (target - predict)
        self.q_table.loc[state, action] = round(updated_value, 2)  # <- Force rounding here


    def save_q_table(self, filename):
        """Save Q-table to CSV file with rounded values"""
        rounded_q_table = self.q_table.copy().round(2)
        rounded_q_table.to_csv(filename, index=True)
        print(f"✅ Q-table saved to {filename}")


    def load_q_table(self, filename):
        """Load Q-table from CSV file"""
        if os.path.exists(filename):
            self.q_table = pd.read_csv(filename, index_col=0)
            print(f"✅ Q-table loaded from {filename}")
        else:
            print(f"❌ Q-table file {filename} not found")

# === MAIN EXECUTION ===
def main():
    print("🍽️  Welcome to Food Preference Onboarding!")
    print("This will help us understand your food preferences.\n")
    
    env = FoodEnvironment()
    actions = ['recommend', 'not_recommend']
    agent = SarsaFoodRecommender(actions)
    # Ensure all foods have initial Q-values (0, 0)
    all_food_names = [item['name'] for item in env.foods]
    agent.initialize_q_table(all_food_names)


        # === STEP 1: Onboarding ===
        # Ask user for dietary preference
    while True:
        user_type = input("Are you vegetarian or non-vegetarian? (Enter 'veg' or 'non-veg'): ").strip().lower()
        if user_type in ["veg", "non-veg"]:
            break
        else:
            print("❗ Please enter a valid option: 'veg' or 'non-veg'.")

    # Select quiz foods based on preference
    if user_type == "veg":
        random_subset = env.fixed_onboarding_foods_veg
    else:
        random_subset = env.fixed_onboarding_foods_nonveg

    env.display_options(random_subset)
    selected_indices = env.get_user_selection(random_subset)

    liked_tags = set()
    liked_names = set()

    print("\n🔄 Processing your preferences...")

    for i, item in enumerate(random_subset):
        name = item['name']
        tags = item.get('tags', [])
        action = 'recommend' if i in selected_indices else 'not_recommend'
        reward = 1 if i in selected_indices else -1

        agent.learn(name, action, reward)

        if reward == 1:
            liked_tags.update(tags)
            liked_names.add(name)

    print(f"✨ You liked {len(liked_names)} items with tags: {', '.join(liked_tags)}")

    # Propagate reward to similar-tag foods
    propagated_count = 0
    for item in env.foods:
        name = item['name']
        tags = set(item.get('tags', []))
        if name in liked_names:
            continue

        # Custom propagation based on tag type
        if tags.intersection(liked_tags.intersection({"sweet", "spicy", "savoury"})):
            agent.learn(name, 'recommend', 0.5)  # strong propagation
            propagated_count += 1
        elif tags.intersection(liked_tags.intersection({"south-indian", "north-indian"})):
            agent.learn(name, 'recommend', 0.4)
            propagated_count += 1
        elif tags.intersection(liked_tags.intersection({"rice", "chapati"})):
            agent.learn(name, 'recommend', 0.2)  # softer propagation
            propagated_count += 1


    print(f"🔗 Propagated preferences to {propagated_count} similar foods")

    # Save Q-table
    script_dir = os.path.dirname(os.path.abspath(__file__))
    q_table_path = os.path.join(script_dir, "q_table.csv")
    agent.save_q_table(q_table_path)

    print("\n✅ Onboarding complete! Your preferences have been saved.")
    print("📊 Final Q-table preview:")
    print(agent.q_table.head(10))
    
    print(f"\n🎯 Total foods in Q-table: {len(agent.q_table)}")
    recommend_count = (agent.q_table['recommend'] > 0).sum()
    print(f"🌟 Foods with positive recommendation scores: {recommend_count}")

if __name__ == "__main__":
    main()