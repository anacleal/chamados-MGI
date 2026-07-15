import random
import os
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from bertopic import BERTopic as BERTopic_
from sklearn.cluster import KMeans
from umap import UMAP
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K_POR_SISTEMA = {
    "SIASS":  6,
    "SOUGOV": 5,
    "SIGEPE": 5,
    "SIAPE":  8,
    "TOTAIS": 10,
}

COL_TEXTO = "Descrição do chamado"
COL_ID    = "Id"

class BERTopic(BERTopic_):
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

    def save_topic_coordinates_2d(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

        topic_embeddings = getattr(self, "topic_embeddings_", None)
        if topic_embeddings is None:
            print("  [AVISO] topic_embeddings_ não disponível — pulei coordenadas 2D.")
            return

        topic_info = self.get_topic_info()
        topic_ids  = topic_info["Topic"].tolist()
        topic_sizes = dict(zip(topic_info["Topic"], topic_info["Count"]))

        ids_validos  = [t for t, m in zip(topic_ids, mask) if m]
        emb_validos  = np.array(topic_embeddings)[mask]

        if len(ids_validos) < 3:
            print("  [AVISO] Poucos tópicos válidos para projeção 2D — pulei.")
            return
        n_neighbors_2d = max(2, min(len(ids_validos) - 1, 2))

        umap_2d = UMAP(
            n_neighbors=n_neighbors_2d,
            n_components=2,
            metric="cosine",
            min_dist=0.1,
            random_state=SEED,
        )
        coords = umap_2d.fit_transform(emb_validos)

        df_coords = pd.DataFrame({
            "topico": ids_validos,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "n_documentos": [topic_sizes.get(t, 0) for t in ids_validos],
        })

        df_coords.to_csv(os.path.join(output_dir, "topic_coordinates_2d.csv"), index=False)
        print(f"  Coordenadas 2D (embeddings semânticos) salvas em {output_dir}/topic_coordinates_2d.csv")


def _carregar_umap_params(results_dir: str) -> dict:
    tuned_path = os.path.join(results_dir, "melhores_parametros_umap.json")
    if not os.path.exists(tuned_path):
        raise FileNotFoundError(
            f"Parâmetros do UMAP não encontrados em {tuned_path}. "
            f"Rode tuning_bertopic.py para esse sistema antes de treinar o modelo final."
        )
    with open(tuned_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    print(f"  Usando parâmetros UMAP tunados: {params}")
    return params


def run_bertopic(docs: list, ids: list, sistema_nome: str, nr_topics: int) -> BERTopic:
    print(f"\n--- [{sistema_nome}] {nr_topics} tópicos | device: {DEVICE.upper()} ---")

    model_dir   = f"models/{sistema_nome}"
    model_path  = f"{model_dir}/modelo"
    results_dir = f"bertopic_resultados/{sistema_nome}"

    for d in (model_dir, results_dir):
        os.makedirs(d, exist_ok=True)

    umap_params = None

    if os.path.exists(model_path) and os.path.exists(f"{model_path}/config.json"):
        print(f"  Modelo encontrado em {model_path}. Carregando...")
        topic_model = BERTopic.load(model_path)

    else:
        umap_params = _carregar_umap_params(results_dir)

        print(f"  Calculando embeddings com SentenceTransformer no {DEVICE.upper()}...")
        embedding_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE
        )
        embeddings = embedding_model.encode(docs, show_progress_bar=True)

        print("  Treinando modelo definitivo...")
        umap_final   = UMAP(**umap_params, min_dist=0.0, random_state=SEED)
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

        print(f"  Salvando modelo em {model_path}...")
        topic_model.save(model_path, serialization="safetensors", save_ctfidf=True)

    # Persistência de resultados essenciais ao dashboard
    topic_model.save_txt(   f"{results_dir}/topicos.txt")
    topic_model.save_json(  f"{results_dir}/topicos.json")
    topic_model.save_params(f"{results_dir}/parametros.json", umap_params=umap_params)
    topic_model.dominant_topics(docs, results_dir, ids)
    topic_model.save_topic_coordinates_2d(results_dir)

    return topic_model

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