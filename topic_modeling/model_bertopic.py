"""
BERTopic - Modelagem de Tópicos para Chamados de Suporte
=========================================================
Reprodutibilidade garantida via seeds fixas em todos os componentes estocásticos.
O número de tópicos por sistema foi determinado pelo método do cotovelo (inércia do KMeans).
Os hiperparâmetros do UMAP são selecionados via Grid Search usando o Índice de Silhueta.

Seeds fixas aplicadas em:
  - numpy       (SEED)
  - Python random (SEED)
  - PyTorch     (SEED)
  - UMAP        (random_state=SEED)
  - KMeans      (random_state=SEED)
"""

import random
import os
import json
import itertools

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from bertopic import BERTopic as BERTopic_
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Reprodutibilidade global
# ---------------------------------------------------------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Configurações globais
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Número de tópicos determinado pelo método do cotovelo para cada sistema
K_POR_SISTEMA = {
    "SIASS":  6,
    "SOUGOV": 5,
    "SIGEPE": 5,
    "SIAPE":  8,
    "TOTAIS": 10,
}

# Grade de hiperparâmetros do UMAP avaliada via Índice de Silhueta
# n_neighbors: controla o balanço entre estrutura local e global
# n_components: dimensionalidade do espaço reduzido antes do KMeans
# metric: função de distância usada pelo UMAP
UMAP_PARAM_GRID = {
    "n_neighbors":  [15, 30, 50],
    "n_components": [5, 10, 15],
    "metric":       ["cosine", "euclidean"],
}

COL_TEXTO = "Descrição do chamado"
COL_ID    = "Id"


# ---------------------------------------------------------------------------
# Classe BERTopic estendida
# ---------------------------------------------------------------------------
class BERTopic(BERTopic_):
    """BERTopic com métodos auxiliares de persistência e análise de tópicos dominantes."""

    def __init__(self, device: str = DEVICE, **kwargs):
        self.device = device
        super().__init__(**kwargs)

    def save_txt(self, pathfile: str) -> None:
        os.makedirs(os.path.dirname(pathfile), exist_ok=True)
        topic_info = self.get_topic_info()
        with open(pathfile, "w", encoding="utf-8") as f:
            for _, row in topic_info.iterrows():
                f.write(f"\ntopico {row['Topic']}:\n")
                words = row["Representation"]
                if isinstance(words, list):
                    f.write(" ".join(words))
                f.write("\n")

    def save_json(self, pathfile: str) -> None:
        os.makedirs(os.path.dirname(pathfile), exist_ok=True)
        result = {
            topic_id: [[word, float(score)] for word, score in words if word]
            for topic_id, words in self.get_topics().items()
        }
        with open(pathfile, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    def save_params(self, pathfile: str, umap_params: dict = None) -> None:
        os.makedirs(os.path.dirname(pathfile), exist_ok=True)
        params = {
            "seed": SEED,
            "nr_topics": self.nr_topics,
            "calculate_probabilities": self.calculate_probabilities,
            "verbose": self.verbose,
            "umap_params_selecionados": umap_params or {},
            "embedding_model":     str(getattr(self, "embedding_model",  "None")),
            "umap_model":          str(getattr(self, "umap_model",        "None")),
            "kmeans_model":        str(getattr(self, "hdbscan_model",     "None")),
            "vectorizer_model":    str(getattr(self, "vectorizer_model",  "None")),
            "representation_model": (
                {k: str(v) for k, v in self.representation_model.items()}
                if isinstance(getattr(self, "representation_model", None), dict)
                else str(getattr(self, "representation_model", "None"))
            ),
        }
        with open(pathfile, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=4)

    def dominant_topics(self, docs: list, output_dir: str, ids: list) -> None:
        os.makedirs(output_dir, exist_ok=True)
        topic_info = self.get_topic_info()
        topicnames = ["Topico " + str(t) for t in topic_info["Topic"].tolist()]
        papernames = [str(i) for i in ids]

        try:
            topic_dist, _ = self.approximate_distribution(docs)

            df_dt = pd.DataFrame(np.round(topic_dist, 4), columns=topicnames)
            df_dt["id"]             = papernames
            df_dt["dominant_topic"] = self.topics_

            plt.figure(figsize=(10, 6))
            sns.countplot(x=df_dt["dominant_topic"])
            plt.title("Distribuição de Tópicos Dominantes")
            plt.savefig(os.path.join(output_dir, "Topicos_Dominantes.png"))
            plt.close()

            df_dt.to_csv(os.path.join(output_dir, "Topicos_Dominantes.csv"), sep="|", index=False)

            pd.DataFrame({"id": papernames, "dominant_topic": df_dt["dominant_topic"].values}).to_csv(
                os.path.join(output_dir, "Resumo_Topicos_Dominantes.csv"), index=False
            )
            print(f"  Tópicos dominantes salvos em {output_dir}")
        except Exception as e:
            print(f"  Erro ao calcular tópicos dominantes: {e}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def run_bertopic(docs: list, ids: list, sistema_nome: str, nr_topics: int) -> BERTopic:
    print(f"\n--- [{sistema_nome}] {nr_topics} tópicos | device: {DEVICE.upper()} ---")

    model_dir   = f"models/{sistema_nome}"
    model_path  = f"{model_dir}/modelo"
    results_dir = f"bertopic_resultados/{sistema_nome}"
    graph_dir   = f"bertopic_graphs/{sistema_nome}"

    for d in (model_dir, results_dir, graph_dir):
        os.makedirs(d, exist_ok=True)

    # Carrega modelo existente ou treina do zero
    if os.path.exists(model_path) and os.path.exists(f"{model_path}/config.json"):
        print(f"  Modelo encontrado em {model_path}. Carregando...")
        topic_model = BERTopic.load(model_path)
        melhores_params = None  # params já foram salvos na execução original

    else:
        # Embeddings calculados uma única vez e reutilizados em todo o grid
        print(f"  Pré-calculando embeddings com SentenceTransformer no {DEVICE.upper()}...")
        embedding_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE
        )
        embeddings = embedding_model.encode(docs, show_progress_bar=True)

        # --- Grid Search via Índice de Silhueta ---
        keys, values  = zip(*UMAP_PARAM_GRID.items())
        combinacoes   = [dict(zip(keys, v)) for v in itertools.product(*values)]
        n_combinacoes = len(combinacoes)

        print(f"  Iniciando Grid Search ({n_combinacoes} combinações)...")

        melhor_silhueta = -1.0
        melhores_params = None
        resultados_grid = []

        for idx, config in enumerate(combinacoes, start=1):
            umap_model   = UMAP(**config, min_dist=0.0, random_state=SEED)
            kmeans_model = KMeans(n_clusters=nr_topics, random_state=SEED, n_init=10)

            # Modelo leve para avaliação: sem representation_model para ganhar velocidade
            test_model = BERTopic(
                device=DEVICE,
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=kmeans_model,
                nr_topics=nr_topics,
                verbose=False,
            )
            test_model.fit_transform(docs, embeddings)

            # Silhueta calculada no espaço UMAP reduzido
            reduced_emb = test_model.umap_model.transform(embeddings)
            labels      = np.array(test_model.topics_)
            silhueta    = round(float(silhouette_score(reduced_emb, labels)), 4)

            resultados_grid.append({**config, "silhouette_score": silhueta})
            print(f"    [{idx}/{n_combinacoes}] {config} -> Silhueta: {silhueta}")

            if silhueta > melhor_silhueta:
                melhor_silhueta = silhueta
                melhores_params = config

        print(f"  > Melhor configuração: {melhores_params} | Silhueta: {melhor_silhueta}")

        # Salva tabela completa do grid para o artigo
        pd.DataFrame(resultados_grid).to_csv(
            f"{results_dir}/grid_search_resultados.csv", index=False
        )

        # --- Treinamento definitivo com os melhores parâmetros ---
        print("  Treinando modelo definitivo...")
        umap_final   = UMAP(**melhores_params, min_dist=0.0, random_state=SEED)
        kmeans_final = KMeans(n_clusters=nr_topics, random_state=SEED, n_init=20)

        representation_model = {
            "KeyBERTInspired": KeyBERTInspired(),
            "MMR": MaximalMarginalRelevance(diversity=0.3),
        }

        topic_model = BERTopic(
            device=DEVICE,
            embedding_model=embedding_model,
            umap_model=umap_final,
            hdbscan_model=kmeans_final,
            representation_model=representation_model,
            nr_topics=nr_topics,
            calculate_probabilities=True,
            verbose=True,
        )
        topic_model.fit_transform(docs, embeddings)

        # Silhueta do modelo definitivo
        final_reduced   = topic_model.umap_model.transform(embeddings)
        final_labels    = np.array(topic_model.topics_)
        silhueta_final  = round(float(silhouette_score(final_reduced, final_labels)), 4)

        with open(f"{results_dir}/metricas_validacao.json", "w", encoding="utf-8") as f:
            json.dump({
                "silhouette_score":        silhueta_final,
                "melhores_parametros_umap": melhores_params,
            }, f, ensure_ascii=False, indent=4)

        print(f"  Salvando modelo em {model_path}...")
        topic_model.save(model_path, serialization="safetensors", save_ctfidf=True)

    # Persistência de resultados
    topic_model.save_txt(   f"{results_dir}/topicos.txt")
    topic_model.save_json(  f"{results_dir}/topicos.json")
    topic_model.save_params(f"{results_dir}/parametros.json", umap_params=melhores_params)
    topic_model.dominant_topics(docs, results_dir, ids)

    # Visualizações interativas
    try:
        topic_model.visualize_topics()                          .write_html(f"{graph_dir}/intertopic_map.html")
        topic_model.visualize_barchart(top_n_topics=nr_topics)  .write_html(f"{graph_dir}/bar_graphs.html")
        topic_model.visualize_hierarchy()                       .write_html(f"{graph_dir}/hierarchy.html")
        topic_model.get_topic_info()                            .to_csv(    f"{graph_dir}/topic_info.csv", index=False)
    except Exception as e:
        print(f"  Erro ao gerar visualizações: {e}")

    return topic_model


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for sistema, k in K_POR_SISTEMA.items():
        csv_path = f"../data/chamados_{sistema.lower()}.csv"

        if not os.path.exists(csv_path):
            print(f"Arquivo não encontrado: {csv_path}")
            continue

        df   = pd.read_csv(csv_path, low_memory=False).dropna(subset=[COL_TEXTO])
        docs = df[COL_TEXTO].astype(str).tolist()
        ids  = df[COL_ID].tolist()

        if len(docs) < 100:
            print(f"[{sistema}] Ignorado: apenas {len(docs)} documentos disponíveis.")
            continue

        run_bertopic(docs, ids, sistema, nr_topics=k)

    print("\nProcessamento finalizado!")