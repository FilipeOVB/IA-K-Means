import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def euclidian_distance(a,b):
    return np.sqrt(np.sum((a-b) ** 2))


def initialize_centroids(dados,k):
    np.random.seed(42) 
    # definindo uma seed de randomizaçao padrao para todas execuções
    random_index = np.random.choice(len(dados), size=k, replace=False)
    centroids = dados[random_index]
    return centroids


def assing_clusters(dados, centroids):
    clusters = []
    for point in dados:

        # Calcula a Distancia Euclidiana de cada ponto para cada um dos centroids
        distances = [euclidian_distance(point, centroid) for centroid in centroids]

        # Armazena o indice do centroid com a menor distancia encontrada em distances
        cluster = np.argmin(distances)

        clusters.append(cluster) 
    
    return np.array(clusters)


def update_centroids(dados, clusters, k):
    new_centroids = np.array([dados[clusters == i].mean(axis=0) for i in range(k)])
    return new_centroids


def kmeans(dados, k):

    max_iters = 100 # Número maximo de iterações caso não atinja a convergência
    tolerance = 1e-4 # Criterio de convergencia

    # Define os centroids
    centroids = initialize_centroids(dados,k)

    # Calcula os clusters e ajusta os centroids até atingir o max_iters ou a convergencia
    for _ in range(max_iters):
        clusters = assing_clusters(dados, centroids)
        new_centroids = update_centroids(dados, clusters, k)

        # Se a mudança no centroid for menor que 0.0001 consideramos que atingiu a convergencia
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            break

        centroids = new_centroids

    # Calcular a silhouete score para k
    silhouette = silhouette_score(dados, clusters)

    return silhouette, clusters, centroids


# Carregar o dataset iris
df = pd.read_csv('iris.csv')

# Separar as características e o rótulo
dados = df.drop(columns=['Id', 'Species']).values

# Encontrar o melhor valor de K com base no Silhouette
best_k = 0
best_silhouette = -1
best_clusters = None
best_centroids = None


print(f"K-Means hardcore\n")
# Testar o K-means para diferentes valores de k
for k in [2,3,5]:

    # Medir o tempo de execução utilizando timeit
    exec_time = timeit.timeit(lambda: kmeans(dados, k), number=10) / 10

    # Executar o K-means
    silhouette, clusters, centroids = kmeans(dados,k)

    # Armazenar o melhor k
    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_k = k
        best_clusters = clusters
        best_centroids = centroids


    print(f"Silhouette Score para k = {k}: {silhouette:.4}")
    print(f"Tempo de execução: {exec_time:.4f} segundos\n")

    # Formação dos graficos
    plt.scatter(dados[:,0], dados[:,1], c = clusters, cmap='viridis', marker='o', label='Data Points')
    plt.scatter(centroids[:,0], centroids[:,1], color='red', marker='X', s=100, label='Centroids')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()
    plt.title(f"K-Means hardcore com k = {k}")
    plt.show()


print(f"\nTeste PCA - melhor K = {best_k}\n")
# Reduzir a dimensionalidade para 1 e 2 componentes
for n_components in [1, 2]:

    pca = PCA(n_components=n_components)
    dados_reduzidos = pca.fit_transform(dados)
    
    # Medir o tempo de execução utilizando timeit
    exec_time = timeit.timeit(lambda: kmeans(dados_reduzidos, best_k), number=10) / 10

    # Executar o K-means nos dados reduzidos
    silhouette, clusters, centroids = kmeans(dados_reduzidos,best_k)

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
    plt.title(f"K-Means hardcore com {n_components} componente(s) e k = {best_k}")
    plt.show()