import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil, floor, pi
from seaborn import color_palette
from prince import PCA as PCA_Prince
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import umap

from eda import EDA   # clase base, carpeta eda/


class NoSupervisado(EDA):
    """
    Métodos no supervisados del framework. Hereda de EDA, así que
    también tiene todos los métodos de análisis exploratorio.
    - ACP: plano principal, círculo de correlación e integrado (sobreposición)
    - Agrupamiento: K-means (con codo), HAC (con dendrograma)
    - Proyecciones: t-SNE y UMAP
    """

    def __init__(self, dataframe: pd.DataFrame):
        super().__init__(dataframe)
        # resultados del último ACP (se llenan al llamar ACP())
        self.__modelo = None
        self.__correlacion_var = None
        self.__coordenadas_ind = None
        self.__contribucion_ind = None
        self.__cos2_ind = None
        self.__var_explicada = None

    # --- Propiedades del ACP ---
    @property
    def modelo(self):
        return self.__modelo

    @property
    def correlacion_var(self):
        return self.__correlacion_var

    @property
    def coordenadas_ind(self):
        return self.__coordenadas_ind

    @property
    def contribucion_ind(self):
        return self.__contribucion_ind

    @property
    def cos2_ind(self):
        return self.__cos2_ind

    @property
    def var_explicada(self):
        return self.__var_explicada

    # =================================================================
    # ACP
    # =================================================================
    def ACP(self, n_componentes=5, estandarizar=True):
        """
        Aplica ACP (prince) sobre las variables numéricas.
        estandarizar=True usa la matriz de correlaciones; False la de covarianzas.
        Retorna la varianza explicada y acumulada de cada componente.
        """
        datos = self.obtener_datos_numericos()
        self.__modelo = PCA_Prince(n_components=n_componentes, rescale_with_std=estandarizar).fit(datos)
        self.__correlacion_var = self.__modelo.column_correlations
        self.__coordenadas_ind = self.__modelo.row_coordinates(datos)
        self.__contribucion_ind = self.__modelo.row_contributions_
        self.__cos2_ind = self.__modelo.row_cosine_similarities(datos)
        self.__var_explicada = self.__modelo.percentage_of_variance_
        var = np.array(self.__var_explicada)
        return pd.DataFrame({
            "Componente": [f"PC{i+1}" for i in range(n_componentes)],
            "Varianza explicada (%)": var.round(2),
            "Acumulada (%)": var.cumsum().round(2)
        })

    def plano_principal(self, ejes=[0, 1], ind_labels=True, colores=None, titulo='Plano Principal'):
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        plt.style.use('seaborn-v0_8-bright')
        if colores is None:
            plt.scatter(x, y, color='gray')
        else:
            # colorea cada individuo con otra variable o con los clusters
            sc = plt.scatter(x, y, c=colores, cmap='viridis', s=8)
            plt.colorbar(sc)
        plt.title(titulo)
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))

    def circulo(self, ejes=[0, 1], var_labels=True, titulo='Círculo de Correlación'):
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-v0_8-bright')
        c = plt.Circle((0, 0), radius=1, color='steelblue', fill=False)
        plt.gca().add_patch(c)
        plt.axis('scaled')
        plt.title(titulo)
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * 0.95, cor[i, 1] * 0.95, color='steelblue',
                      alpha=0.5, head_width=0.05, head_length=0.05)
            if var_labels:
                plt.text(cor[i, 0] * 1.05, cor[i, 1] * 1.05, self.correlacion_var.index[i],
                         color='steelblue', ha='center', va='center')

    def integrado(self, ejes=[0, 1], ind_labels=True, var_labels=True, titulo='Sobreposición Plano-Círculo'):
        # plano principal y círculo de correlación en un mismo gráfico
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        cor = self.correlacion_var.iloc[:, ejes]
        scale = min((max(x) - min(x) / (max(cor[ejes[0]]) - min(cor[ejes[0]]))),
                    (max(y) - min(y) / (max(cor[ejes[1]]) - min(cor[ejes[1]])))) * 0.7
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-v0_8-bright')
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        plt.scatter(x, y, color='gray')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale, color='steelblue',
                      alpha=0.5, head_width=0.05, head_length=0.05)
            if var_labels:
                plt.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15,
                         self.correlacion_var.index[i],
                         color='steelblue', ha='center', va='center')
        plt.title(titulo)

    # =================================================================
    # Agrupamiento / clustering
    # =================================================================
    def datos_para_clustering(self, n_acp=None):
        # datos estandarizados; si se pide, se reducen antes con ACP
        datos = self.estandarizar()
        if n_acp:
            coords = PCA(n_components=n_acp, random_state=42).fit_transform(datos)
            datos = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(n_acp)])
        return datos

    def kmeans(self, n_clusters=3, init="k-means++", n_acp=None, graficar=True):
        """
        K-means sobre los datos estandarizados (o sobre n_acp componentes).
        Retorna etiquetas y silhouette score.
        """
        datos = self.datos_para_clustering(n_acp)
        modelo = KMeans(n_clusters=n_clusters, init=init, n_init=10, random_state=42)
        etiquetas = modelo.fit_predict(datos)
        score = silhouette_score(datos, etiquetas)
        print(f"K-means k={n_clusters} init={init} -> Silhouette: {score:.4f} | Inercia: {modelo.inertia_:.1f}")
        if graficar:
            NoSupervisado.bar_plot(modelo.cluster_centers_, datos.columns)
            NoSupervisado.radar_plot(modelo.cluster_centers_, datos.columns)
        return etiquetas, score

    def codo(self, k_max=10, n_acp=None):
        # inercia y silhouette para k = 2..k_max, para escoger k
        datos = self.datos_para_clustering(n_acp)
        inercias, siluetas = [], []
        for k in range(2, k_max + 1):
            modelo = KMeans(n_clusters=k, n_init=10, random_state=42).fit(datos)
            inercias.append(modelo.inertia_)
            siluetas.append(silhouette_score(datos, modelo.labels_))
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(range(2, k_max + 1), inercias, marker="o")
        ax[0].set_title("Método del codo")
        ax[0].set_xlabel("k")
        ax[0].set_ylabel("Inercia")
        ax[1].plot(range(2, k_max + 1), siluetas, marker="o", color="darkred")
        ax[1].set_title("Silhouette por k")
        ax[1].set_xlabel("k")
        return pd.DataFrame({"k": range(2, k_max + 1), "inercia": np.round(inercias, 1),
                             "silhouette": np.round(siluetas, 4)})

    def hac(self, n_clusters=3, enlace="ward", n_acp=None, graficar=True):
        """
        Agrupamiento jerárquico aglomerativo.
        enlace: 'ward', 'complete', 'average' o 'single'.
        Retorna etiquetas y silhouette score, y dibuja el dendrograma.
        """
        datos = self.datos_para_clustering(n_acp)
        modelo = AgglomerativeClustering(n_clusters=n_clusters, linkage=enlace)
        etiquetas = modelo.fit_predict(datos)
        score = silhouette_score(datos, etiquetas)
        print(f"HAC {enlace} k={n_clusters} -> Silhouette: {score:.4f} | "
              f"Tamaños: {sorted(np.bincount(etiquetas).tolist(), reverse=True)}")
        if graficar:
            enlaces = linkage(datos, method=enlace)
            plt.figure(figsize=(12, 5))
            dendrogram(enlaces, truncate_mode="lastp", p=30)
            corte = (enlaces[-n_clusters, 2] + enlaces[-n_clusters + 1, 2]) / 2
            plt.axhline(y=corte, color="red", linestyle="--")
            plt.title(f"Dendrograma ({enlace}), corte en k={n_clusters}")
            plt.xlabel("Grupos (cantidad de individuos)")
            plt.ylabel("Distancia")
        return etiquetas, score

    def comparar_kmeans(self, configuraciones):
        # configuraciones: lista de dicts, ej. [{"n_clusters": 3}, {"n_clusters": 4, "init": "random"}]
        filas = []
        for conf in configuraciones:
            etiquetas, score = self.kmeans(graficar=False, **conf)
            filas.append({**conf, "silhouette": round(score, 4),
                          "tamaños": sorted(pd.Series(etiquetas).value_counts().tolist(), reverse=True)})
        return pd.DataFrame(filas)

    def comparar_hac(self, configuraciones):
        filas = []
        for conf in configuraciones:
            etiquetas, score = self.hac(graficar=False, **conf)
            filas.append({**conf, "silhouette": round(score, 4),
                          "tamaños": sorted(pd.Series(etiquetas).value_counts().tolist(), reverse=True)})
        return pd.DataFrame(filas)

    def perfil_clusters(self, etiquetas, columnas=None):
        # promedio de cada variable por cluster (en la escala original)
        datos = self.mostrar() if columnas is None else self.mostrar()[columnas]
        perfil = datos.groupby(etiquetas).mean().T
        perfil.loc["cantidad"] = pd.Series(etiquetas).value_counts().sort_index().values
        return perfil.round(3)

    @staticmethod
    def bar_plot(centros, labels, scale=False, cluster=None, var=None):
        fig, ax = plt.subplots(1, 1, figsize=(15, 8), dpi=200)
        centros = np.copy(centros)
        if scale:
            for col in range(centros.shape[1]):
                centros[:, col] = centros[:, col] / max(centros[:, col])
        colores = color_palette()
        minimo = floor(centros.min()) if floor(centros.min()) < 0 else 0

        def inside_plot(valores, labels, titulo):
            plt.barh(range(len(valores)), valores, 1/1.5, color=colores)
            plt.xlim(minimo, ceil(centros.max()))
            plt.title(titulo)

        if var is not None:
            centros = np.array([n[[x in var for x in labels]] for n in centros])
            colores = [colores[x % len(colores)] for x, i in enumerate(labels) if i in var]
            labels = labels[[x in var for x in labels]]

        if cluster is None:
            for i in range(centros.shape[0]):
                plt.subplot(1, centros.shape[0], i + 1)
                inside_plot(centros[i].tolist(), labels, ('Cluster ' + str(i)))
                plt.yticks(range(len(labels)), labels) if i == 0 else plt.yticks([])
        else:
            pos = 1
            for i in cluster:
                plt.subplot(1, len(cluster), pos)
                inside_plot(centros[i].tolist(), labels, ('Cluster ' + str(i)))
                plt.yticks(range(len(labels)), labels) if pos == 1 else plt.yticks([])
                pos += 1

    @staticmethod
    def radar_plot(centros, labels):
        fig, ax = plt.subplots(1, 1, figsize=(15, 8), dpi=200)
        centros = np.array([((n - min(n)) / (max(n) - min(n)) * 100) if
                            max(n) != min(n) else (n/n * 50) for n in centros.T])
        angulos = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
        angulos += angulos[:1]
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angulos[:-1], labels)
        ax.set_rlabel_position(0)
        plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                   ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"],
                   color="grey", size=8)
        plt.ylim(-10, 100)
        for i in range(centros.shape[1]):
            valores = centros[:, i].tolist()
            valores += valores[:1]
            ax.plot(angulos, valores, linewidth=1, linestyle='solid', label='Cluster ' + str(i))
            ax.fill(angulos, valores, alpha=0.3)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # =================================================================
    # t-SNE y UMAP (solo sirven para visualizar en 2D)
    # =================================================================
    @staticmethod
    def graficar_2d(coords, colores, titulo):
        plt.figure(figsize=(7, 6))
        if colores is None:
            plt.scatter(coords[:, 0], coords[:, 1], s=6, color="gray")
        else:
            sc = plt.scatter(coords[:, 0], coords[:, 1], s=6, c=colores, cmap="viridis")
            plt.colorbar(sc)
        plt.title(titulo)
        plt.xticks([])
        plt.yticks([])

    def tsne(self, perplexity=30, colores=None, titulo="t-SNE"):
        modelo = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        coords = modelo.fit_transform(self.estandarizar())
        print(f"t-SNE perplexity={perplexity} -> KL: {modelo.kl_divergence_:.4f}")
        NoSupervisado.graficar_2d(coords, colores, titulo)
        return coords

    def umap(self, n_neighbors=15, min_dist=0.1, colores=None, titulo="UMAP"):
        modelo = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        coords = modelo.fit_transform(self.estandarizar())
        print(f"UMAP n_neighbors={n_neighbors} min_dist={min_dist}")
        NoSupervisado.graficar_2d(coords, colores, titulo)
        return coords
