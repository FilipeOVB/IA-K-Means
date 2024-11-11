import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# Função para executar o K-means e calcular o Silhouette
def kmeans_silhouette(dados, k):
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans_model.fit_predict(dados)
    centroids = kmeans_model.cluster_centers_
    silhouette = silhouette_score(dados, clusters)
    return silhouette, clusters, centroids

# Carregar o dataset iris
df = pd.read_csv('iris.csv')

# Separar as características
dados = df.drop(columns=['Id', 'Species']).values

# Encontrar o melhor valor de K com base no Silhouette
best_k = 0
best_silhouette = -1
best_clusters = None
best_centroids = None


print(f"K-Means auto\n")
# Testar o K-means para diferentes valores de k usando sklearn e calcular o silhouette score
for k in [2, 3, 5]:

    # Medir o tempo de execução utilizando timeit
    exec_time = timeit.timeit(lambda: kmeans_silhouette(dados, k), number=10) / 10

    # Executar o K-means e obter o Silhouette
    silhouette, clusters, centroids = kmeans_silhouette(dados, k)

    # Armazenar o melhor k
    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_k = k
        best_clusters = clusters
        best_centroids = centroids


    print(f"Para k={k}, Silhouette Score: {silhouette:.4f}")
    print(f"Tempo de execução: {exec_time:.4f} segundos\n")

    # Visualizar os clusters e centróides
    plt.scatter(dados[:, 0], dados[:, 1], c=clusters, cmap='viridis', marker='o', label='Data Points')
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=100, label='Centroids')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()
    plt.title(f'K-Means auto com k = {k}')
    plt.show()


print(f"\nTeste PCA - melhor K = {best_k}\n")
# Reduzir a dimensionalidade para 1 e 2 componentes
for n_components in [1, 2]:

    pca = PCA(n_components=n_components)
    dados_reduzidos = pca.fit_transform(dados)
    
    # Medir o tempo de execução utilizando timeit
    exec_time = timeit.timeit(lambda: kmeans_silhouette(dados_reduzidos, best_k), number=10) / 10

    # Executar o best_k-means e obter o Silhouette
    silhouette, clusters, centroids = kmeans_silhouette(dados_reduzidos, best_k)

    print(f"Silhouette Score com {n_components} componentes: {silhouette:.4f}")
    print(f"Tempo de execução: {exec_time:.4f} segundos\n")

    # Plotar os resultados
    plt.figure(figsize=(8, 5))
    if n_components == 1:
        plt.scatter(dados_reduzidos[:, 0], np.zeros_like(dados_reduzidos[:, 0]), c=clusters, cmap='viridis', marker='o', label='Data Points')
        plt.scatter(centroids[:, 0], np.zeros_like(centroids[:, 0]), color='red', marker='X', s=100, label='Centroids')
        plt.xlabel('Componente Principal 1')
        plt.yticks([])
    else:
        plt.scatter(dados_reduzidos[:, 0], dados_reduzidos[:, 1], c=clusters, cmap='viridis', marker='o', label='Data Points')
        plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=100, label='Centroids')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
    
    plt.legend()
    plt.title(f"K-Means auto com {n_components} componente(s) e k = {best_k}")
    plt.show()