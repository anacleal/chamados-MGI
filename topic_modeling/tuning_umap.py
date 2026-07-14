import os
import json
import itertools

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP
from sentence_transformers import SentenceTransformer

from model_bertopic import (
    BERTopic,
    SEED,
    DEVICE,
    K_POR_SISTEMA,
    COL_TEXTO,
    COL_ID,
)

UMAP_PARAM_GRID = {
    "n_neighbors":  [15, 30, 50],
    "n_components": [5, 10, 15],
    "metric":       ["cosine", "euclidean"],
}


def grid_search_umap(docs: list, sistema_nome: str, nr_topics: int) -> dict:
    print(f"\n--- [Tuning: {sistema_nome}] {nr_topics} tópicos | device: {DEVICE.upper()} ---")

    results_dir = f"bertopic_resultados/{sistema_nome}"
    os.makedirs(results_dir, exist_ok=True)

    print(f"  Pré-calculando embeddings com SentenceTransformer no {DEVICE.upper()}...")
    embedding_model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE
    )
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

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

        #sem representation_model
        test_model = BERTopic(
            device=DEVICE,
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=kmeans_model,
            nr_topics=nr_topics,
            verbose=False,
        )
        test_model.fit_transform(docs, embeddings)

        reduced_emb = test_model.umap_model.transform(embeddings)
        labels      = np.array(test_model.topics_)
        silhueta    = round(float(silhouette_score(reduced_emb, labels)), 4)

        resultados_grid.append({**config, "silhouette_score": silhueta})
        print(f"    [{idx}/{n_combinacoes}] {config} -> Silhueta: {silhueta}")

        if silhueta > melhor_silhueta:
            melhor_silhueta = silhueta
            melhores_params = config

    print(f"  > Melhor configuração: {melhores_params} | Silhueta: {melhor_silhueta}")

    pd.DataFrame(resultados_grid).to_csv(
        f"{results_dir}/grid_search_resultados.csv", index=False
    )

    with open(f"{results_dir}/metricas_validacao.json", "w", encoding="utf-8") as f:
        json.dump({
            "silhouette_score":        melhor_silhueta,
            "melhores_parametros_umap": melhores_params,
        }, f, ensure_ascii=False, indent=4)

    # Salva os parâmetros tunados para o model_bertopic.py consumir depois
    with open(f"{results_dir}/melhores_parametros_umap.json", "w", encoding="utf-8") as f:
        json.dump(melhores_params, f, ensure_ascii=False, indent=4)

    return melhores_params


def gerar_visualizacoes(sistema_nome: str) -> None:
    model_path = f"models/{sistema_nome}/modelo"
    graph_dir  = f"bertopic_graphs/{sistema_nome}"

    if not (os.path.exists(model_path) and os.path.exists(f"{model_path}/config.json")):
        print(f"  [AVISO] Modelo não encontrado em {model_path}. "
              f"Rode model_bertopic.py primeiro.")
        return

    os.makedirs(graph_dir, exist_ok=True)
    topic_model = BERTopic.load(model_path)

    try:
        topic_model.visualize_topics()                                .write_html(f"{graph_dir}/intertopic_map.html")
        topic_model.visualize_barchart(top_n_topics=topic_model.nr_topics or 10).write_html(f"{graph_dir}/bar_graphs.html")
        topic_model.visualize_hierarchy()                              .write_html(f"{graph_dir}/hierarchy.html")
        topic_model.get_topic_info()                                   .to_csv(    f"{graph_dir}/topic_info.csv", index=False)
        print(f"  Gráficos salvos em {graph_dir}")
    except Exception as e:
        print(f"  Erro ao gerar visualizações: {e}")


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

        grid_search_umap(docs, sistema, nr_topics=k)
        gerar_visualizacoes(sistema)

    print("\nTuning e gráficos finalizados!")