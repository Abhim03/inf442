import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

fname = '/users/eleves-b/2022/abderrahim.sadegh/inf442/inf442/td4/csv/iris.csv'
data = pd.read_csv(fname, header = 0)

tree = linkage(data[['x', 'y']])
plt.figure(figsize=(10, 5))
D = dendrogram(tree, labels = data['name'].to_numpy(), orientation = 'left')

plt.tick_params(axis='y', which='both', labelleft=True)

plt.show()
