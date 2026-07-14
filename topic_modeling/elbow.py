import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import KMeans

def set_global_determinism(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_global_determinism(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def plot_elbow_method(sistemas, col_texto="Descrição do chamado", k_min=2, k_max=25):
    print(f"Carregando modelo de embeddings no device: {DEVICE.upper()}...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)

    for sis in sistemas:
        csv_path = f"../data/chamados_{sis}.csv"
        output_dir = f"bertopic_resultados/analise_cotovelo"
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(csv_path):
            print(f"Arquivo não encontrado: {csv_path}. Pulando...")
            continue

        print(f"\n--- Processando Regra do Cotovelo para: {sis.upper()} ---")
        df = pd.read_csv(csv_path, low_memory=False)
        df_clean = df.dropna(subset=[col_texto])
        docs = df_clean[col_texto].astype(str).tolist()

        if len(docs) < k_max:
            print(f"[{sis.upper()}] Documentos insuficientes ({len(docs)}). Pulando...")
            continue

        print(f"[{sis.upper()}] Gerando embeddings para {len(docs)} documentos...")
        embeddings = embedding_model.encode(docs, show_progress_bar=True)

        print(f"[{sis.upper()}] Reduzindo dimensionalidade com UMAP...")
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        reduced_embeddings = umap_model.fit_transform(embeddings)

        print(f"[{sis.upper()}] Calculando K-Means para K de {k_min} até {k_max}...")
        wcss = []
        K_range = range(k_min, k_max + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(reduced_embeddings)
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(K_range, wcss, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)

        plt.title(f'Método do Cotovelo (Elbow Method) - {sis.upper()}', fontsize=16, fontweight='bold')
        plt.xlabel('Número de Clusters (k)', fontsize=14)
        plt.ylabel('Soma dos Quadrados Intra-Cluster (WCSS)', fontsize=14)
        plt.xticks(K_range)
        plt.grid(True, linestyle='--', alpha=0.7)

        plot_path = os.path.join(output_dir, f'grafico_cotovelo_{sis}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[{sis.upper()}] Concluído! Gráfico salvo em: {plot_path}")


if __name__ == "__main__":
    sistemas_alvo = ["siape", "siass", "sigepe", "sougov", "totais"]
    plot_elbow_method(sistemas_alvo, k_min=2, k_max=25)