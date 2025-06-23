import pandas as pd
import matplotlib.pyplot as plt
import gower
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


data = pd.read_csv('Books.csv')

print(data.head())
data.info()
data.describe()

feature_cols =['author','pages','average_rating','ratings_count']
data_selected = data[feature_cols].copy()

data_selected['pages'] = pd.to_numeric(data_selected['pages'], errors='coerce')
data_selected['average_rating'] = pd.to_numeric(data_selected['average_rating'], errors='coerce')

data_cleaned = data_selected.dropna()


print("data yang digunakan untuk clustering")
print(data_selected.head())

distance_matrix = gower.gower_matrix(data_cleaned)

condensed_distance = squareform(distance_matrix)
linked = linkage(condensed_distance, method='ward')


plt.figure(figsize=(10,5))
plt.title("Dendrogram clustering buku")

dendrogram(linked, truncate_mode='lastp', p=10)

plt.ylabel("jarak antar buku")
plt.xlabel("ukuran cluster")
plt.show()