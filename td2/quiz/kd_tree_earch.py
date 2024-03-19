import pandas as pd
from sklearn.neighbors import KDTree

fname = "/users/eleves-b/2022/abderrahim.sadegh/inf442/td2/csv/iris_large.csv"
data = pd.read_csv(fname, delimiter=' ', header=0)
print(data)

print('\nbuilding kd-tree...', end='')
kdt = KDTree(data.iloc[:,0:4], leaf_size=1, metric='euclidean')
print(' done')

x = [41.1, 32.2, 23.3, 14.4]
print('\n  my measurements [sepal_len, sepal_wid, petal_len, petal_wid] are: ', x)


# Modify the kdt.query call to search for the 5 nearest neighbors
distances, indices = kdt.query([x], k=5)

# Calculate the average distance
average_distance = distances[0].mean()

print(f'\t Index/indices of the {len(indices[0])} closest point(s): {indices[0]}')
print(f'\t Distance(s) to the {len(distances[0])} closest point(s): {distances[0]}')
print(f'\t Average distance to the {len(distances[0])} closest point(s): {average_distance}')


