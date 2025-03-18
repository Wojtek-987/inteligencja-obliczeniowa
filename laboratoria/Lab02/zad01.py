import pandas as pd

df = pd.read_csv("../Dane/iris_with_errors.csv")

print("Initial missing data counts:\n", df.isnull().sum())

numeric_columns = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nMissing data counts after converting numeric columns:\n", df.isnull().sum())




# Replace values outside the range (0, 15) with the mean of the valid data in that column.
for col in numeric_columns:
    invalid_mask = (df[col] <= 0) | (df[col] >= 15)
    if invalid_mask.any():
        valid_mean = df.loc[~invalid_mask, col].mean()
        count_invalid = invalid_mask.sum()
        print(f"\nColumn '{col}': Found {count_invalid} values outside (0, 15). Replacing them with mean {valid_mean:.2f}.")
        df.loc[invalid_mask, col] = valid_mean




def fix_species(s: str) -> str:
    s = s.strip()
    s_lower = s.lower()

    if s_lower == "setosa":
        return "Setosa"
    elif s_lower in ["versicolor", "versicolour"]:
        return "Versicolor"
    elif s_lower == "virginica":
        return "Virginica"
    else:
        return s


df["variety"] = df["variety"].apply(fix_species)
