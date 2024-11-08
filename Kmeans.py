import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.metrics import silhouette_score


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

    return clusters, centroids


# Carregar o dataset iris
df = pd.read_csv('iris.csv')

# Separar as características e o rótulo
dados = df.drop(columns=['Id', 'Species']).values

# Testar o K-means para diferentes valores de k
for k in [3,5]:

    # Iniciar a contagem de tempo
    start_time = time.time()

    # Executar o K-means
    clusters, centroids = kmeans(dados,k)

    # Calcular o tempo gasto
    elapsed_time = time.time() - start_time 

    # Calcular a silhouete score para k
    silhouette = silhouette_score(dados, clusters)
    print(f"Silhouette Score para k={k}: {silhouette:.4}")

    # Formação dos graficos
    plt.scatter(dados[:,0], dados[:,1], c = clusters, cmap='viridis', marker='o', label='Data Points')
    plt.scatter(centroids[:,0], centroids[:,1], color='red', marker='X', s=100, label='Centroids')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()
    plt.show()

    print(f"Tempo de execução: {elapsed_time:.4f} segundos\n")