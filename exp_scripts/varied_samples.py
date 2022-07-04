import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("C:\\Users\\L.G.G.E\\Desktop\\sample.xlsx", sheet_name="EPR", header=0)
print(df)
sns.boxplot(data=df, y="val", x="number", hue='dataset', linewidth=1.5, palette='Set2')
plt.show()
