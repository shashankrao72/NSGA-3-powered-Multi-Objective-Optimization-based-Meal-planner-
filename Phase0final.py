import random
import pickle
from foods_dataset import foods

def get_user_input_and_calculate():
    print("Welcome to the Meal Planner!")
    weight = float(input("Enter your weight (kg): "))
    height = float(input("Enter your height (cm): "))
    age = int(input("Enter your age: "))
    gender = input("Enter your gender (male/female): ").strip().lower()

    # Activity level input
    print("Enter activity level (1=sedentary, 2=light, 3=moderate, 4=active, 5=very_active): ")
    activity_map = {
        '1': 'sedentary',
        '2': 'light',
        '3': 'moderate',
        '4': 'active',
        '5': 'very_active'
    }
    activity_input = input().strip()
    activity_level = activity_map.get(activity_input, 'light')  # Default to 'light' if invalid

    # Meal distribution input
    print("\nMeal Distribution Preferences (l=light, m=medium, h=heavy):")
    meal_map = {'l': 'light', 'm': 'medium', 'h': 'heavy'}
    meal_distribution = {
        "breakfast": meal_map.get(input(f"Breakfast preference (l/m/h): ").strip().lower(), 'medium'),
        "lunch": meal_map.get(input(f"Lunch preference (l/m/h): ").strip().lower(), 'medium'),
        "dinner": meal_map.get(input(f"Dinner preference (l/m/h): ").strip().lower(), 'medium')
    }

    # Modified health goal input
    print("Enter health goal (wl=weight_loss, mg=muscle_gain, m=maintenance, lc=low_carb): ")
    health_goal_map = {
        'wl': 'weight_loss',
        'mg': 'muscle_gain',
        'm': 'maintenance',
        'lc': 'low_carb'
    }
    health_goal_input = input().strip().lower()
    health_goal = health_goal_map.get(health_goal_input, 'maintenance')  # Default to 'maintenance' if invalid

    # Calculate BMR (Mifflin-St Jeor Equation)
    if gender == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    # Adjust BMR based on activity level
    activity_multipliers = {
        "sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9
    }
    adjusted_bmr = bmr * activity_multipliers.get(activity_level, 1.2)
    # Log total energy requirement
    print(f"\nTotal Daily Energy Requirement (Adjusted BMR): {adjusted_bmr:.2f} kcal")

    # Select random fruit and beverage
    fruits = [f for f in foods if 'fruit' in f.get('tags', [])]
    beverages = [f for f in foods if 'beverage' in f.get('tags', [])]
    selected_fruit = random.choice(fruits) if fruits else None
    selected_beverage = random.choice(beverages) if beverages else None
    fruit_calories = selected_fruit["energy"] if selected_fruit else 0
    beverage_calories = selected_beverage["energy"] if selected_beverage else 0
    fruit_name = selected_fruit["name"] if selected_fruit else 0
    beverage_name = selected_beverage["name"] if selected_beverage else 0

    # Adjust energy split based on meal distribution
    total_meal_energy = adjusted_bmr - (fruit_calories + beverage_calories)
    allocation_rules = {"light": 0.15, "medium": 0.30, "heavy": 0.45}
    energy_split = {
        meal: total_meal_energy * allocation_rules.get(pref, 0.30)
        for meal, pref in meal_distribution.items()
    }
    # Normalize to ensure sum equals total_meal_energy
    total = sum(energy_split.values())
    if total > 0:
        energy_split = {meal: (calories / total) * total_meal_energy for meal, calories in energy_split.items()}
    else:
        # Fallback to 4:4:9 if invalid
        energy_split = {
            "breakfast": total_meal_energy * (4/17),
            "lunch": total_meal_energy * (4/17),
            "dinner": total_meal_energy * (9/17)
        }


    # Log total energy and remaining after fruit/beverage
    print(f"\nTotal Daily Energy Requirement: {adjusted_bmr:.2f} kcal")
    print(f"Energy After Subtracting Fruit [{fruit_name}:{fruit_calories:.2f} kcal] and Beverage[{beverage_name}:{beverage_calories:.2f} kcal]  {total_meal_energy:.2f} kcal")


    # # Log energy split across meals
    # print("\nEnergy Split Across Meals:")
    # for meal, energy in energy_split.items():
    #     print(f"{meal.capitalize()}: {energy:.2f} kcal")

    # Set macro ratios based on health goal
    macro_ratios = {
        "weight_loss": {"carbs": 0.3, "protein": 0.4, "fat": 0.3},
        "muscle_gain": {"carbs": 0.4, "protein": 0.3, "fat": 0.3},
        "maintenance": {"carbs": 0.4, "protein": 0.3, "fat": 0.3},
        "low_carb": {"carbs": 0.1, "protein": 0.25, "fat": 0.65}
    }
    ratios = macro_ratios.get(health_goal, macro_ratios["maintenance"])
    carb_cal_per_g, protein_cal_per_g, fat_cal_per_g = 4, 4, 9

    meal_breakdown = {
        meal: {
            "carbs_g": (energy * ratios["carbs"]) / carb_cal_per_g,
            "protein_g": (energy * ratios["protein"]) / protein_cal_per_g,
            "fat_g": (energy * ratios["fat"]) / fat_cal_per_g
        } for meal, energy in energy_split.items()
    }

    budget = float(input("Enter your daily meal budget (INR): "))

    user_data = {
        "weight": weight,
        "height": height,
        "age": age,
        "gender": gender,
        "activity_level": activity_level,
        "adjusted_bmr": adjusted_bmr,
        "budget_inr": budget,
        "fruit": selected_fruit,
        "beverage": selected_beverage,
        "meal_distribution": meal_distribution,
        "health_goal": health_goal,
        "meal_breakdown": meal_breakdown
    }

    with open("user_data.pkl", "wb") as f:
        pickle.dump(user_data, f)

    with open("budget.txt", "w") as f:
        f.write(str(budget))

    return user_data

if __name__ == "__main__":
    user_data = get_user_input_and_calculate()
    print("\nUser Data:")
    print(f"Adjusted BMR: {user_data['adjusted_bmr']:.2f} kcal")
    print(f"Budget: ₹{user_data['budget_inr']}")
    print(f"Fruit: {user_data['fruit']['name'] if user_data['fruit'] else 'None'}")
    print(f"Beverage: {user_data['beverage']['name'] if user_data['beverage'] else 'None'}")
    print("Meal Breakdown:")
    for meal, macros in user_data['meal_breakdown'].items():
        print(f"{meal.capitalize()}: {macros['carbs_g']:.2f}g carbs, {macros['protein_g']:.2f}g protein, {macros['fat_g']:.2f}g fat")