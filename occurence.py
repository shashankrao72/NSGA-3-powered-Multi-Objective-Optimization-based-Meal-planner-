import pandas as pd

# Step 1: Extract all unique dishes from feedback
feedback_df = pd.read_csv('feedback_history.csv')
feedback_df.columns = feedback_df.columns.str.strip()

feedback_dishes = set()
for items in feedback_df['meal_items']:
    dishes = [dish.strip() for dish in str(items).split(',')]
    feedback_dishes.update(dishes)

# Step 2: Filter function (works for any meal)
def filter_meal_file(filename):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()

    # Identify all item columns (e.g., Item1, Item2, ...)
    item_columns = [col for col in df.columns if col.lower().startswith('item')]

    # Remove rows where ANY item in the row matches a feedback dish
    mask = df[item_columns].apply(lambda row: any(str(item).strip() in feedback_dishes for item in row), axis=1)
    filtered_df = df[~mask]

    # Overwrite the same file
    filtered_df.to_csv(filename, index=False)
    print(f"✅ {filename} updated: {len(df)} → {len(filtered_df)} rows")

# Step 3: Apply to all candidate files
filter_meal_file('breakfast_candidates.csv')
filter_meal_file('lunch_candidates.csv')
filter_meal_file('dinner_candidates.csv')
