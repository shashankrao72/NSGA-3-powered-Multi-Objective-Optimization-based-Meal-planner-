import numpy as np
import pandas as pd
import random
import os
from foods_dataset import foods  # Make sure this exists and contains the food data

# === Utility ===
def load_meal_plans_from_csv(filename):
    df = pd.read_csv(filename)
    meal_plans = []
    for _, row in df.iterrows():
        meal = [str(item).strip() for item in row.values if pd.notna(item)]
        meal_plans.append(meal[1:] if row.get('Meal') else meal)  # skip Meal name if present
    return meal_plans

# === Environment ===
class FoodEnvironment:
    def __init__(self):
        self.foods = foods
        self.n_items = len(self.foods)

# === Agent ===
class SarsaFoodRecommender:
    def __init__(self, actions, learning_rate=0.1, gamma=0.9):
        self.q_table = pd.DataFrame(columns=actions)
        self.lr = learning_rate
        self.gamma = gamma
        self.actions = actions

    def _check_state(self, state):
        if state not in self.q_table.index:
            new_row = pd.Series([0] * len(self.actions), index=self.actions, name=state)
            self.q_table = pd.concat([self.q_table, new_row.to_frame().T])

    def learn(self, state, action, reward):
        self._check_state(state)
        predict = self.q_table.loc[state, action]
        target = reward
        updated_value = predict + self.lr * (target - predict)
        self.q_table.loc[state, action] = round(updated_value, 2) 

    def save_q_table(self, filename):
        """Save Q-table to CSV file"""
        self.q_table.to_csv(filename, index=True)
        print(f"✅ Q-table updated and saved to {filename}")

    def load_q_table(self, filename):
        """Load Q-table from CSV file"""
        if os.path.exists(filename):
            self.q_table = pd.read_csv(filename, index_col=0)
            print(f"✅ Q-table loaded from {filename}")
            return True
        else:
            print(f"❌ Q-table file {filename} not found")
            print("🔔 Please run onboarding.py first to create your preference profile!")
            return False

# === Meal Scoring ===
def score_meal_plan(meal, q_table):
    score = 0
    for food in meal:
        if food in q_table.index:
            score += q_table.loc[food, 'recommend']
    return score

# === MAIN EXECUTION ===
def main():
    print("🍽️  Welcome to Personalized Meal Recommendations!")
    print("Loading your food preferences...\n")
    
    env = FoodEnvironment()
    actions = ['recommend', 'not_recommend']
    agent = SarsaFoodRecommender(actions)

    # Load existing Q-table
    script_dir = os.path.dirname(os.path.abspath(__file__))
    q_table_path = os.path.join(script_dir, "q_table.csv")
    
    if not agent.load_q_table(q_table_path):
        return  # Exit if Q-table doesn't exist

    # Load meal plans
    meal_file_path = os.path.join(script_dir, "meal_plans.csv")
    
    if not os.path.exists(meal_file_path):
        print(f"❌ Meal plans file {meal_file_path} not found.")
        print("Please create meal_plans.csv with your meal combinations.")
        return

    meal_plans = load_meal_plans_from_csv(meal_file_path)

    # Filter out invalid meals
    valid_names = {item["name"] for item in env.foods}
    meal_plans = [[food for food in meal if food in valid_names] for meal in meal_plans]
    meal_plans = [m for m in meal_plans if m]

    if not meal_plans:
        print("❌ No valid meal plans found.")
        print("Please check that your meal_plans.csv contains valid food names from the dataset.")
        return

    print(f"📋 Loaded {len(meal_plans)} valid meal plans")
    
    # Show initial Q-table stats
    recommend_count = (agent.q_table['recommend'] > 0).sum()
    print(f"🌟 Foods with positive recommendation scores: {recommend_count}")
    print(f"📊 Total foods in preference profile: {len(agent.q_table)}\n")

    # === STEP 2: Recommendation Loop ===
    shown_meals = set()
    session_count = 0

    while True:
        remaining_meals = [meal for meal in meal_plans if tuple(meal) not in shown_meals]
        if not remaining_meals:
            print("\n🚫 No new meal plans left to suggest.")
            print("💡 Consider generating more meal plans or clearing your history.")
            break

        # Sort meals by Q-table scores (best first)
        remaining_meals.sort(key=lambda m: score_meal_plan(m, agent.q_table), reverse=True)
        best_plan = remaining_meals[0]
        best_score = score_meal_plan(best_plan, agent.q_table)
        shown_meals.add(tuple(best_plan))
        session_count += 1

        print(f"\n🌟 Recommended meal plan #{session_count} (Score: {best_score:.2f}):")
        for item in best_plan:
            print(f" - {item}")

        # Get user feedback
        while True:
            feedback = input("\nDid you like this meal plan? (yes/no/quit): ").strip().lower()
            if feedback in ['yes', 'no', 'quit']:
                break
            print("Please enter 'yes', 'no', or 'quit'")

        if feedback == 'quit':
            print("\n👋 Thanks for using the meal recommender!")
            break

        reward = 1 if feedback == 'yes' else -1

        # Collect all tags from the meal plan
        all_tags = set()
        for food in best_plan:
            food_obj = next((f for f in env.foods if f['name'] == food), None)
            if food_obj:
                all_tags.update(food_obj.get('tags', []))

        # Update Q-table based on feedback
        print("🔄 Updating your preferences...")
        
        # Direct feedback on meal items
        for food in best_plan:
            agent.learn(food, 'recommend' if reward == 1 else 'not_recommend', reward)

        # Negative propagation for disliked meals
        if reward == -1:
            propagated = 0
            for item in env.foods:
                if item['name'] in best_plan:
                    continue
                item_tags = set(item.get('tags', []))

                # Custom negative propagation
                if item_tags.intersection(all_tags.intersection({"sweet", "spicy", "savoury"})):
                    agent.learn(item['name'], 'recommend', -0.2)
                    propagated += 1
                elif item_tags.intersection(all_tags.intersection({"south-indian", "north-indian"})):
                    agent.learn(item['name'], 'recommend', -0.2)
                    propagated += 1
                elif item_tags.intersection(all_tags.intersection({"rice", "chapati"})):
                    agent.learn(item['name'], 'recommend', -0.1)
                    propagated += 1


        # Save updated Q-table
        agent.save_q_table(q_table_path)

        print(f"\n📊 Updated preference profile:")
        new_recommend_count = (agent.q_table['recommend'] > 0).sum()
        print(f"🌟 Foods with positive scores: {new_recommend_count}")

        if reward == 1:
            print("\n✅ Great! We'll remember you liked this combination.")
            
            # Ask if they want more recommendations
            continue_rec = input("Would you like more recommendations? (yes/no): ").strip().lower()
            if continue_rec != 'yes':
                print("\n👋 Thanks for using the meal recommender!")
                break
        else:
            print("\n🔁 Looking for a better meal plan for you...\n")

if __name__ == "__main__":
    main()