import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('TkAgg')


def visualize_pca(features, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="Set2", s=50)
    plt.title("PCA - Wizualizacja podpisów")
    plt.savefig("pca_visualization.png")
    plt.close()


def visualize_tsne(features, labels):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="Set2", s=50)
    plt.title("t-SNE - Wizualizacja podpisów")
    plt.savefig("tsne_visualization.png")
    plt.close()