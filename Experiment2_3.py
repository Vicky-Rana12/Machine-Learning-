import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine


wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap - Wine Dataset")
plt.show()

corr = df.drop("target", axis=1).corr()
corr_pairs = corr.unstack()
corr_pairs = corr_pairs[corr_pairs < 1]  # remove self-correlation (1.0)
strongest = corr_pairs.sort_values(ascending=False).head(1)

print("\n-- Strongest Positive Correlation Pair --")
print(strongest)