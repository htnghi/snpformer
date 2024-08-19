import pandas as pd

data = {
  "Sample_id": [0, 1, 2],
  "3_10": [420, 380, 390],
  "3_2": [7, 8, 12],
  "1_5": [5, 1, 89],
  "2_1": [2, 4, 6],
  "1_8": [7, 8, 12],
  "1_3": [7, 8, 2],
  "2_4": [7, 5, 12]
}


# Load data into a DataFrame object:
df = pd.DataFrame(data)

# Exclude "Sample_id" column from sorting
sorted_columns = sorted(df.columns[1:], key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1])))

# Sort the DataFrame based on the sorted columns
sorted_df = df[["Sample_id"] + sorted_columns]

# Split data into separate DataFrames based on type number
type_dfs = {}
for column in sorted_columns:
    type_num = column.split("_")[0]
    if type_num not in type_dfs:
        type_dfs[type_num] = sorted_df[["Sample_id", column]]
    else:
        type_dfs[type_num][column] = sorted_df[column]

# Print sorted DataFrame
print("Sorted DataFrame:")
print(sorted_df)

# Print separate DataFrames based on type number
print("\nSeparate DataFrames based on type number:")
for type_num, type_df in type_dfs.items():
    print(f"\nType number {type_num}:")
    print(type_df)
exit(1)
# Load data into a DataFrame object
df = pd.DataFrame(data)

# Extract column names excluding "Sample_id"
columns_to_sort = [col for col in df.columns if col != "Sample_id"]

# Sort the column names
sorted_columns = sorted(columns_to_sort, key=lambda x: tuple(map(int, x.split('_'))))

# Reorder the DataFrame columns based on the sorted column names
df = df.reindex(["Sample_id"] + sorted_columns, axis=1)

# Display the sorted DataFrame
print(df)


from collections import Counter

# Extract the type numbers before "_" from the column names
type_numbers = [int(col.split('_')[0]) for col in df.columns]

# Count the occurrences of each type number
type_number_counts = Counter(type_numbers)

# Display the counts
for type_number, count in type_number_counts.items():
    print(f"Type number {type_number} has {count} occurrences")

