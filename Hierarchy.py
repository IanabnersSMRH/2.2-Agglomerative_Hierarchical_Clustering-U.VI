import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Datos de ejemplo (150 clientes en 3 grupos; reemplaza esto con tu carga real de X_clientes)
X_raw, _ = make_blobs(
    n_samples=150, centers=3, cluster_std=1.1, random_state=42
)

# 2. Escalado de variables (clave cuando se usa distancia euclidiana y enlace 'ward')
scaler = StandardScaler()
X_clientes = scaler.fit_transform(X_raw)

# 3. Clustering Jerárquico Aglomerativo con enlace Ward
hierarchical = AgglomerativeClustering(
    n_clusters=3, metric="euclidean", linkage="ward"
)
labels_hierarchical = hierarchical.fit_predict(X_clientes)

# 4. Métricas y resumen de resultados
silueta_h = silhouette_score(X_clientes, labels_hierarchical)

print("=== CLUSTERIZACIÓN JERÁRQUICA AGLOMERATIVA ===")
print(f"Número de Clusters asignados: {hierarchical.n_clusters_}")
print("Métrica de Distancia / Enlace: Euclidiana / Ward")
print(f"Coeficiente de Silueta:       {silueta_h:.4f}")
print(
    "Distribución de elementos por cluster:", np.bincount(labels_hierarchical)
)

# 5. Dendrograma con SciPy para visualizar la estructura jerárquica
Z = linkage(X_clientes, method="ward", metric="euclidean")

plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode="lastp", p=15, leaf_rotation=45, leaf_font_size=10)
plt.title("Dendrograma - Agrupación Jerárquica (Ward)")
plt.xlabel("Observaciones / Clústeres fusionados")
plt.ylabel("Distancia Euclidiana / Incremento de varianza")
plt.tight_layout()
plt.show()