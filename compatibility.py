#compatibility.py
import pandas as pd
import numpy as np
from typing import List, Dict
#from foods_dataset import foods
import os

class MealCompatibilityEngine:
    def __init__(self, foods_dataset):
        """Initialize with foods dataset to get tags"""
        self.foods = foods_dataset
        self.food_tags = {food['name']: food['tags'] for food in self.foods}
        
        # Define compatibility rules
        self.compatibility_rules = {
            # Positive combinations (add points)
           #('rice', 'curry'): 1.0,
           #('chapati', 'curry'): 0.8,
           # ('veg', 'veg'): 0.5,
            
            # Negative combinations (subtract points)
            #('rice', 'chapati'): -1.0,
            ('iron', 'dairy'): -0.8,
            ('iron', 'calcium'): -0.3,
            ('rice', 'rice'): -1.5,           # New: Avoid too much rice
            ('chapati', 'chapati'): -0.8 
        }
    
    def get_meal_tags(self, meal_items: List[str]) -> List[List[str]]:
        """Get tags for all items in a meal"""
        meal_tags = []
        for item in meal_items:
            item_tags = self.food_tags.get(item, [])
            meal_tags.append(item_tags)
        return meal_tags
    
    def calculate_compatibility_score(self, meal_items: List[str]) -> float:
        """Calculate compatibility score based on tag combinations"""
        if not meal_items:
            return 0.0
        
        # Get all tags in the meal
        meal_tags = self.get_meal_tags(meal_items)
        all_tags = set([tag for item_tags in meal_tags for tag in item_tags])
        
        compatibility_score = 0.0
        
        # Check each compatibility rule
        for (tag1, tag2), score in self.compatibility_rules.items():
            if tag1 in all_tags and tag2 in all_tags:
                compatibility_score += score
        
        return compatibility_score
    
    def analyze_meal_plan(self, meal_items: List[str]) -> Dict:
        """Analyze a meal plan and return detailed compatibility info"""
        meal_tags = self.get_meal_tags(meal_items)
        all_tags = set([tag for item_tags in meal_tags for tag in item_tags])
        
        compatibility_score = self.calculate_compatibility_score(meal_items)
        
        # Find which rules applied
        applied_rules = []
        for (tag1, tag2), score in self.compatibility_rules.items():
            if tag1 in all_tags and tag2 in all_tags:
                applied_rules.append({
                    'rule': f"{tag1} + {tag2}",
                    'score': score,
                    'type': 'positive' if score > 0 else 'negative'
                })
        
        return {
            'meal': meal_items,
            'tags': list(all_tags),
            'compatibility_score': compatibility_score,
            'applied_rules': applied_rules
        }
    

def load_meal_plans_from_csv(filename):
    df = pd.read_csv(filename)
    meal_plans = []
    for _, row in df.iterrows():
        # If there's a 'Meal' column, skip it
        if 'Meal' in df.columns:
            meal = [str(row[col]).strip() for col in df.columns if col != 'Meal' and pd.notna(row[col])]
        else:
            meal = [str(item).strip() for item in row.values if pd.notna(item)]
        meal_plans.append(meal)
    return meal_plans

# Example usage with your data
def analyze_all_meals(meal_plans_csv: str, foods_dataset: List[Dict]) -> List[Dict]:
    """Analyze all meals from CSV and return compatibility scores"""
    
    # Initialize compatibility engine
    compatibility_engine = MealCompatibilityEngine(foods_dataset)
    
    # Load meal plans
    meal_plans = load_meal_plans_from_csv(meal_plans_csv)
    
    # Analyze each meal
    results = []
    for i, meal in enumerate(meal_plans, 1):
        analysis = compatibility_engine.analyze_meal_plan(meal)
        analysis['meal_id'] = f"meal{i}"
        results.append(analysis)
    
    return results

# Function to display results nicely
def display_compatibility_results(results: List[Dict]):
    """Display compatibility analysis results"""
    print("🍽️ Meal Compatibility Analysis")
    print("=" * 50)
    
    for result in results:
        print(f"\n{result['meal_id'].upper()}: {', '.join(result['meal'])}")
        print(f"🏷️  Tags: {', '.join(result['tags'])}")
        print(f"⭐ Compatibility Score: {result['compatibility_score']:.1f}")
        
        if result['applied_rules']:
            print("📋 Applied Rules:")
            for rule in result['applied_rules']:
                emoji = "✅" if rule['type'] == 'positive' else "❌"
                print(f"   {emoji} {rule['rule']}: {rule['score']:+.1f}")
        else:
            print("📋 No compatibility rules applied")
        print("-" * 30)

# Run analysis
if __name__ == "__main__":
    # Get the path to meal_plans.csv in the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    meal_plans_csv_path = os.path.join(current_dir, "meal_plans.csv")
    results = analyze_all_meals(meal_plans_csv_path, foods)
    display_compatibility_results(results)
