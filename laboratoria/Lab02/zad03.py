import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn import datasets

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df = df.iloc[:, :2]
df['Species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


def print_stats(data, name):
    print(f"--- {name} ---")
    print("Min values:\n", data.min())
    print("Max values:\n", data.max())
    print("Mean values:\n", data.mean())
    print("Standard deviations:\n", data.std())
    print("\n")


print_stats(df.iloc[:, :2], "Original Data")


# Min-max normalization: (x - min) / (max - min)
df_minmax = df.copy()
for col in df.columns[:2]:
    min_val = df_minmax[col].min()
    max_val = df_minmax[col].max()
    df_minmax[col] = (df_minmax[col] - min_val) / (max_val - min_val)
print_stats(df_minmax.iloc[:, :2], "Min-Max Normalized Data")


# Z-score standardization: (x - mean) / std
df_zscore = df.copy()
for col in df.columns[:2]:
    mean_val = df_zscore[col].mean()
    std_val = df_zscore[col].std()
    df_zscore[col] = (df_zscore[col] - mean_val) / std_val
print_stats(df_zscore.iloc[:, :2], "Z-Score Standardized Data")


def scatter_plot(data, title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    for species in data['Species'].unique():
        subset = data[data['Species'] == species]
        plt.scatter(subset.iloc[:, 0], subset.iloc[:, 1],
                    label=species, edgecolor='k')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title="Species")
    plt.grid(True)
    plt.show()


# Original data plot
scatter_plot(df, "Original Data: Sepal Length vs. Sepal Width",
             "Sepal Length (cm)", "Sepal Width (cm)")

# Min-Max Normalized data plot
scatter_plot(df_minmax, "Min-Max Normalized Data: Sepal Length vs. Sepal Width",
             "Normalized Sepal Length", "Normalized Sepal Width")

# Z-Score Standardized data plot
scatter_plot(df_zscore, "Z-Score Standardized Data: Sepal Length vs. Sepal Width",
             "Standardized Sepal Length", "Standardized Sepal Width")
