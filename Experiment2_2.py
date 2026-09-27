import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine


wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target
df.drop("target", axis=1).plot(
    kind="box",
    subplots=True,
    layout=(4, 4),
    figsize=(16, 12),
    sharex=False,
    sharey=False,
)
plt.suptitle("Boxplots of Wine Dataset Attributes")
plt.tight_layout()
plt.show()