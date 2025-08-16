import plotly.graph_objs as go
import numpy as np
import random
from itertools import combinations_with_replacement
from foods_dataset import foods
import numpy as np
import random
from tabulate import tabulate

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available. Plotting functionality will be disabled.")
    MATPLOTLIB_AVAILABLE = False

class NSGA3BinaryMealPlanner:
    def __init__(self, food_data, target_energy, ideal_macros, min_foods=2, max_foods=2):
        self.food_data = food_data
        self.target_energy = target_energy
        self.ideal_macros = ideal_macros
        self.min_foods = min_foods
        self.max_foods = max_foods
        self.dim = len(food_data)
        self.nobj = 3

    def is_valid_meal(self, individual):
        selected_count = np.sum(individual > 0.5)
        return self.min_foods <= selected_count <= self.max_foods

    def repair_individual(self, individual, avoid_set=None):
        binary_individual = (individual > 0.5).astype(float)
        
        # Ensure avoided foods are not selected
        if avoid_set:
            for idx in avoid_set:
                binary_individual[idx] = 0.0
        
        selected_count = int(np.sum(binary_individual))
        
        if selected_count < self.min_foods:
            if avoid_set:
                available_indices = list(set(range(self.dim)) - avoid_set)
            else:
                available_indices = list(range(self.dim))
            
            unselected_indices = [i for i in available_indices if binary_individual[i] == 0]
            foods_to_add = min(self.min_foods - selected_count, len(unselected_indices))
            
            if foods_to_add > 0:
                add_indices = np.random.choice(unselected_indices, foods_to_add, replace=False)
                binary_individual[add_indices] = 1
        
        elif selected_count > self.max_foods:
            selected_indices = np.where(binary_individual == 1)[0]
            foods_to_remove = selected_count - self.max_foods
            remove_indices = np.random.choice(selected_indices, foods_to_remove, replace=False)
            binary_individual[remove_indices] = 0
        
        return binary_individual
    
    def uniform_crossover(self, p1, p2, prob=0.5, avoid_set=None):
        c1 = np.copy(p1)
        c2 = np.copy(p2)
        for i in range(len(p1)):
            if random.random() < prob:
                c1[i], c2[i] = c2[i], c1[i]
        
        c1 = self.repair_individual(c1, avoid_set=avoid_set)
        c2 = self.repair_individual(c2, avoid_set=avoid_set)
        return c1, c2

    def bit_flip_mutation(self, individual, prob=0.1, avoid_set=None):
        mutant = np.copy(individual)
        
        for i in range(len(individual)):
            if random.random() < prob:
                mutant[i] = 1 - mutant[i]
        
        # Repair with avoid_set consideration
        mutant = self.repair_individual(mutant, avoid_set=avoid_set)
        return mutant

    def calculate_objectives(self, population):
        objectives = []
        for individual in population:
            selected = individual > 0.5
            if not self.is_valid_meal(individual):
                objectives.append([1e6, 1e6, 1e6])
                continue
            total_energy = 0
            total_cost = 0
            total_protein = 0
            total_carbs = 0
            total_fat = 0
            for is_selected, food in zip(selected, self.food_data):
                if is_selected:
                    total_energy += food["energy"]
                    total_cost += food["cost"]
                    total_protein += food["protein"]
                    total_carbs += food["carbs"]
                    total_fat += food["fat"]
            energy_deviation = abs(self.target_energy - total_energy)
            cost = total_cost
            total_macros = total_protein + total_carbs + total_fat + 1e-6
            actual_ratios = [
                total_protein / total_macros,
                total_carbs / total_macros,
                total_fat / total_macros
            ]
            macro_deviation = sum(abs(a - b) for a, b in zip(actual_ratios, self.ideal_macros))
            objectives.append([energy_deviation, cost, macro_deviation])
        return np.array(objectives)

    def generate_reference_points(self, p=12):
        def get_points(n, k):
            for c in combinations_with_replacement(range(n + 1), k):
                if sum(c) == n:
                    yield np.array(c, dtype=float)
        points = list(get_points(p, k=self.nobj))
        return np.array([point / p for point in points])

    def normalize_objectives(self, objectives):
        ideal = np.min(objectives, axis=0)
        nadir = np.max(objectives, axis=0)
        range_obj = nadir - ideal
        range_obj[range_obj < 1e-6] = 1e-6
        normalized = (objectives - ideal) / range_obj
        return normalized, ideal, nadir

    def associate_to_reference_points(self, normalized_objectives, reference_points):
        associations = []
        distances = []
        for obj in normalized_objectives:
            min_distance = float('inf')
            closest_ref = -1
            for j, ref_point in enumerate(reference_points):
                if np.linalg.norm(ref_point) > 1e-6:
                    distance = np.linalg.norm(obj - ref_point)
                else:
                    distance = np.linalg.norm(obj)
                if distance < min_distance:
                    min_distance = distance
                    closest_ref = j
            associations.append(closest_ref)
            distances.append(min_distance)
        return associations, distances

    def environmental_selection(self, population, objectives, reference_points, target_size):
        normalized_objectives, _, _ = self.normalize_objectives(objectives)
        associations, distances = self.associate_to_reference_points(normalized_objectives, reference_points)
        niche_counts = [0] * len(reference_points)
        selected = []
        selected_indices = []
        for ref_idx in range(len(reference_points)):
            candidates = [i for i, assoc in enumerate(associations) if assoc == ref_idx and i not in selected_indices]
            if candidates and len(selected) < target_size:
                best_candidate = min(candidates, key=lambda i: distances[i])
                selected.append(population[best_candidate])
                selected_indices.append(best_candidate)
                niche_counts[ref_idx] += 1
        while len(selected) < target_size:
            min_niche = min(niche_counts)
            min_niche_refs = [i for i, count in enumerate(niche_counts) if count == min_niche]
            selected_ref = random.choice(min_niche_refs)
            candidates = [i for i, assoc in enumerate(associations) if assoc == selected_ref and i not in selected_indices]
            if candidates:
                best_candidate = min(candidates, key=lambda i: distances[i])
                selected.append(population[best_candidate])
                selected_indices.append(best_candidate)
                niche_counts[selected_ref] += 1
            else:
                niche_counts[selected_ref] += 1
        return selected[:target_size]

    def create_initial_population(self, pop_size, avoid_set=None):
        population = []
        for _ in range(pop_size):
            individual = np.zeros(self.dim)
            
            if avoid_set:
                # Use only foods NOT in avoid_set
                allowed_pool = list(set(range(self.dim)) - avoid_set)
            else:
                allowed_pool = list(range(self.dim))
            
            num_foods = random.randint(self.min_foods, self.max_foods)
            if len(allowed_pool) >= num_foods:
                selected_indices = random.sample(allowed_pool, num_foods)
            else:
                selected_indices = allowed_pool  # Use all available
            
            for idx in selected_indices:
                individual[idx] = 1.0
            population.append(individual)
        
        return population

    def calculate_deviations(self, individual):
        """Calculate detailed deviation metrics for an individual meal"""
        selected = individual > 0.5
        if not self.is_valid_meal(individual):
            return None
        
        total_energy = 0
        total_cost = 0
        total_protein = 0
        total_carbs = 0
        total_fat = 0
        
        for is_selected, food in zip(selected, self.food_data):
            if is_selected:
                total_energy += food["energy"]
                total_cost += food["cost"]
                total_protein += food["protein"]
                total_carbs += food["carbs"]
                total_fat += food["fat"]
        
        # Energy deviation
        energy_deviation = abs(self.target_energy - total_energy)
        energy_deviation_pct = (energy_deviation / self.target_energy) * 100
        
        # Macro deviations
        total_macros = total_protein + total_carbs + total_fat + 1e-6
        actual_ratios = [
            total_protein / total_macros,
            total_carbs / total_macros,
            total_fat / total_macros
        ]
        
        protein_deviation = abs(actual_ratios[0] - self.ideal_macros[0])
        carbs_deviation = abs(actual_ratios[1] - self.ideal_macros[1])
        fat_deviation = abs(actual_ratios[2] - self.ideal_macros[2])
        macro_deviation_total = protein_deviation + carbs_deviation + fat_deviation
        
        return {
            'energy_deviation': energy_deviation,
            'energy_deviation_pct': energy_deviation_pct,
            'protein_deviation': protein_deviation,
            'carbs_deviation': carbs_deviation,
            'fat_deviation': fat_deviation,
            'macro_deviation_total': macro_deviation_total,
            'cost': total_cost,
            'actual_energy': total_energy,
            'actual_protein_ratio': actual_ratios[0],
            'actual_carbs_ratio': actual_ratios[1],
            'actual_fat_ratio': actual_ratios[2],
            'target_protein_ratio': self.ideal_macros[0],
            'target_carbs_ratio': self.ideal_macros[1],
            'target_fat_ratio': self.ideal_macros[2]
        }

    def write_meals_to_csv(self, meals, filename="meal_plan.csv"):
        """Write meals to CSV file in the specified format"""
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['Meal'] + [f'Item{i+1}' for i in range(10)]
            writer.writerow(header)
            
            # Write each meal
            for i, meal in enumerate(meals, 1):
                row = [f'meal{i}']
                foods = meal['foods']
                
                # Add food items (up to 10)
                for j in range(10):
                    if j < len(foods):
                        row.append(foods[j])
                    else:
                        row.append('')
                
                writer.writerow(row)
        
        print(f"\n✅ Meals saved to {filename}")

    def optimize(self, pop_size=100, generations=100, avoid_set=None, candidate_num=1):
        population = self.create_initial_population(pop_size, avoid_set=avoid_set)
        reference_points = self.generate_reference_points(p=12)
        history = {'energy_dev': [], 'cost': [], 'macro_dev': [], 'combined': []}
        for gen in range(generations):
            objectives = self.calculate_objectives(population)
            offspring = []
            while len(offspring) < pop_size:
                p1, p2 = random.sample(population, 2)
                c1, c2 = self.uniform_crossover(p1, p2)
                c1 = self.bit_flip_mutation(c1, avoid_set=avoid_set)
                c2 = self.bit_flip_mutation(c2, avoid_set=avoid_set)
                offspring.extend([c1, c2])
            offspring = offspring[:pop_size]
            combined_population = population + offspring
            combined_objectives = np.vstack([objectives, self.calculate_objectives(offspring)])
            elite_fraction = 0.05
            elite_count = max(1, int(pop_size * elite_fraction))
            total_scores = np.sum(combined_objectives, axis=1)
            elite_indices = np.argsort(total_scores)[:elite_count]
            elites = [combined_population[i] for i in elite_indices]
            selected = self.environmental_selection(combined_population, combined_objectives, reference_points, pop_size - elite_count)
            population = selected + elites
            current_objectives = self.calculate_objectives(population)
            valid_objectives = current_objectives[current_objectives[:, 0] < 1e6]
            if len(valid_objectives) > 0:
                best_idx = np.argmin(np.sum(valid_objectives, axis=1))
                best_obj = valid_objectives[best_idx]
            else:
                best_obj = current_objectives[np.argmin(np.sum(current_objectives, axis=1))]
            history['energy_dev'].append(best_obj[0])
            history['cost'].append(best_obj[1])
            history['macro_dev'].append(best_obj[2])
            history['combined'].append(sum(best_obj))
        # Log the best individual for this candidate
        print(f"Meal candidate {candidate_num} generated: Energy dev={best_obj[0]:.2f} cost=₹{best_obj[1]:.2f} macro dev={best_obj[2]:.3f}")
        return population, current_objectives, history

def test_binary_optimization():
    target_energy = 1200
    ideal_macros = [0.25, 0.55, 0.2]
    min_foods = 2
    max_foods = 2
    mode = "exclude_only"
    required_tags = {"veg", "breakfast"}
    excluded_tags = {"fruit", "beverage"}

    if mode == "all":
        working_dataset = foods
    elif mode == "exclude_only":
        working_dataset = [food for food in foods if excluded_tags.isdisjoint(food["tags"])]
    elif mode == "include_exclude":
        working_dataset = [food for food in foods if required_tags.issubset(food["tags"]) and excluded_tags.isdisjoint(food["tags"])]
    else:
        raise ValueError("Invalid mode selected.")

    optimizer = NSGA3BinaryMealPlanner(working_dataset, target_energy, ideal_macros, min_foods, max_foods)

    meals = []
    all_selected = []

    print("🍽️  MEAL OPTIMIZATION RESULTS")
    print("=" * 60)

    for meal_num in range(3):
        print(f"\n🔍 Generating Meal {meal_num + 1}")
        print("-" * 30)
        
        avoid = None
        if meal_num == 1:
            avoid = all_selected[0]
        elif meal_num == 2:
            avoid = all_selected[0].union(all_selected[1])

        pop, obj, _ = optimizer.optimize(pop_size=100, generations=100, avoid_set=avoid, candidate_num=meal_num+1)
        best = pop[np.argmin(np.sum(obj, axis=1))]
        selected_indices = set(np.where(best > 0.5)[0])
        all_selected.append(selected_indices)

        # Calculate deviations
        deviations = optimizer.calculate_deviations(best)
        
        print(f"\n📋 MEAL {meal_num + 1} ITEMS:")
        total_energy = 0
        total_cost = 0
        total_protein = 0
        total_carbs = 0
        total_fat = 0

        food_names = []
        for idx in selected_indices:
            food = working_dataset[idx]
            food_names.append(food['name'])
            print(f"   • {food['name']}")
            total_energy += food["energy"]
            total_cost += food["cost"]
            total_protein += food["protein"]
            total_carbs += food["carbs"]
            total_fat += food["fat"]

        total_macros = total_protein + total_carbs + total_fat
        
        print(f"\n📊 NUTRITIONAL SUMMARY:")
        print(f"   Energy: {total_energy:.1f} kcal (Target: {target_energy})")
        print(f"   Cost: ₹{total_cost:.2f}")
        print(f"   Protein: {total_protein:.1f}g ({(total_protein / total_macros) * 100:.1f}%)")
        print(f"   Carbs: {total_carbs:.1f}g ({(total_carbs / total_macros) * 100:.1f}%)")
        print(f"   Fat: {total_fat:.1f}g ({(total_fat / total_macros) * 100:.1f}%)")

        if deviations:
            print(f"\n📈 DEVIATION ANALYSIS:")
            print(f"   Energy Deviation: {deviations['energy_deviation']:.1f} kcal ({deviations['energy_deviation_pct']:.1f}%)")
            print(f"   Protein Deviation: {deviations['protein_deviation']:.3f} ({deviations['protein_deviation']:.1%})")
            print(f"   Carbs Deviation: {deviations['carbs_deviation']:.3f} ({deviations['carbs_deviation']:.1%})")
            print(f"   Fat Deviation: {deviations['fat_deviation']:.3f} ({deviations['fat_deviation']:.1%})")
            print(f"   Total Macro Deviation: {deviations['macro_deviation_total']:.3f}")
            
            print(f"\n🎯 TARGET vs ACTUAL RATIOS:")
            print(f"   Protein: {deviations['target_protein_ratio']:.1%} → {deviations['actual_protein_ratio']:.1%}")
            print(f"   Carbs: {deviations['target_carbs_ratio']:.1%} → {deviations['actual_carbs_ratio']:.1%}")
            print(f"   Fat: {deviations['target_fat_ratio']:.1%} → {deviations['actual_fat_ratio']:.1%}")

        meals.append({
            "foods": food_names,
            "energy": total_energy,
            "cost": total_cost,
            "protein": total_protein,
            "carbs": total_carbs,
            "fat": total_fat,
            "deviations": deviations
        })

    # Write to CSV file
    optimizer.write_meals_to_csv(meals, "meal_plan.csv")
    
    print(f"\n🎉 OPTIMIZATION COMPLETE!")
    print(f"📁 Results saved to meal_plan.csv")
    
    return meals

if __name__ == "__main__":
    results = test_binary_optimization()