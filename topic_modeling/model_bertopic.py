import numpy as np
import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt
import os
import torch
from bertopic import BERTopic as BERTopic_
from sklearn.cluster import KMeans
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
import nltk

# Detecção global de GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BERTopic(BERTopic_):
    def __init__(self, device=DEVICE, **kwargs):
        super().__init__(**kwargs)
        self.encoder = SentenceTransformer('all-mpnet-base-v2', device=device)
        
    def print_params(self):
        print("Parâmetros do modelo:")
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
            
    def save_txt(self, pathfile: str) -> None:
        os.makedirs(os.path.dirname(pathfile), exist_ok=True)
        with open(pathfile, 'w', encoding='utf-8') as f:
            topic_info = self.get_topic_info()
            for i in range(len(topic_info)):
                t = topic_info['Topic'][i]
                f.write(f'\ntopico {t}:\n')
                words = topic_info['Representation'][i]
                if isinstance(words, list):
                    for word in words:
                        f.write(f'{word} ')
                f.write('\n')
                    
    def save_json(self, pathfile: str) -> None:
        os.makedirs(os.path.dirname(pathfile), exist_ok=True)
        topics = self.get_topics() # retorna um dicionario em py (key - id do topico, value - lista de palavras)
        result = {}
        for topic_id, words in topics.items():
            # BERTopic retorna as palavras como tuplas (palavra, score)
            result[topic_id] = [[word, float(value)] for word, value in words if word != ""]

        with open(pathfile, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=4)

    def save_params(self, pathfile: str) -> None:
        os.makedirs(os.path.dirname(pathfile), exist_ok=True)
        params = {
            "nr_topics": self.nr_topics,
            "calculate_probabilities": self.calculate_probabilities,
            "verbose": self.verbose,
            "embedding_model": str(self.embedding_model) if hasattr(self, 'embedding_model') else "None",
            "umap_model": str(self.umap_model) if hasattr(self, 'umap_model') else "None",
            "hdbscan_model": str(self.hdbscan_model) if hasattr(self, 'hdbscan_model') else "None",
            "vectorizer_model": str(self.vectorizer_model) if hasattr(self, 'vectorizer_model') else "None",
            "representation_model": {k: str(v) for k, v in self.representation_model.items()} if isinstance(getattr(self, 'representation_model', None), dict) else str(getattr(self, 'representation_model', "None"))
        }
        with open(pathfile, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=4)

    def evaluate_model(self, docs: list) -> dict:
        metrics = {}

        # 2. diversidade de topicos
        # mede a proporção de palavras únicas entre os top 10 termos de todos os tópicos
        all_words = []
        topics = self.get_topics()
        for t in topics:
            if t != -1: #ignora outliers
                words = [w[0] for w in topics[t][:10]]
                all_words.extend(words)
        
        if len(all_words) > 0:
            unique_words = set(all_words)
            metrics['topic_diversity'] = round(len(unique_words) / len(all_words), 4)
        else:
            metrics['topic_diversity'] = 0

        # 3. num de tópicos
        metrics['num_topics'] = len([t for t in topics])
        return metrics
                    
    def dominant_topics(self, data: list, path: str, ids: list) -> None:
        output_dir = f'./bertopic_resultados/{path}'
        os.makedirs(output_dir, exist_ok=True)
        
        topic_info = self.get_topic_info()
        topicnames = ['Topico ' + str(i) for i in topic_info["Topic"].values.tolist()]
        papernames = [str(i) for i in ids]
        
        try:
            topic_dist, _ = self.approximate_distribution(data)
            #porcentagem de semelhança de cada chamado com cada um dos topicos criados
            if "Topico -1" in topicnames:
                temp_array = 1 - topic_dist.sum(axis=1)
                topic_dist = np.insert(topic_dist, 0, temp_array, axis=1)
                #se somar as porcentagens e dar por ex. 85%, significa que os 15% restantes vao para o -1 (outlier)

            # cria o dataframe
            df_document_topic = pd.DataFrame(np.round(topic_dist, 4), columns=topicnames)
            df_document_topic['id'] = papernames
            df_document_topic['dominant_topic'] = self.topics_

            plt.figure(figsize=(10, 6))
            sns.countplot(x=df_document_topic.dominant_topic)
            plt.title(f'Distribuição de Tópicos Dominantes - {path}')
            plt.savefig(f'{output_dir}/Topicos_Dominantes.png')
            plt.close()

            df_document_topic.to_csv(f'{output_dir}/Topicos_Dominantes.csv', sep="|", index=False)
            
            resumo = pd.DataFrame()
            resumo['id'] = papernames
            resumo['dominant_topic'] = df_document_topic['dominant_topic'].values
            resumo.to_csv(f'{output_dir}/Resumo_Topicos_Dominantes.csv', index=False)
            print(f"Resultados de tópicos dominantes salvos em {output_dir}")
        except Exception as e:
            print(f"Erro ao calcular tópicos dominantes: {e}")

def run_bertopic(docs, ids, sistema_nome, nr_topics):
    print(f"\n--- [{sistema_nome}] Iniciando Processamento ({nr_topics} tópicos) | Device: {DEVICE.upper()} ---")

    # caminhos para pastas e arquivos
    graph_dir = f"bertopic_graphs/{sistema_nome}/{nr_topics}_topicos"
    model_dir = "../models"
    model_path = f"{model_dir}/model_{sistema_nome.lower()}_{nr_topics}"
    results_dir = f"bertopic_resultados/{sistema_nome}/{nr_topics}_topicos"

    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if os.path.exists(model_path):
        print(f"[{sistema_nome}] Modelo ({nr_topics} tópicos) encontrado! Carregando do disco...")
        topic_model = BERTopic.load(model_path)
    else:
        print(f"[{sistema_nome}] Modelo ({nr_topics} tópicos) não encontrado. Iniciando treinamento...")

        # 1. embeddings
        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)

        # 2. dimensionality reduction
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

        # 3. clustering
        hdbscan_model = KMeans(n_clusters=nr_topics, random_state=42)

        # 5. Representation Models
        representation_model = {
            "KeyBERTInspired": KeyBERTInspired(),
            "MMR": MaximalMarginalRelevance(diversity=0.3)
        }

        topic_model = BERTopic(
            device=DEVICE,
            embedding_model=embedding_model,
            umap_model=umap_model, 
            hdbscan_model=hdbscan_model, 
            representation_model=representation_model,
            nr_topics=nr_topics, # Custom topic count
            calculate_probabilities=True, 
            verbose=True
        )

        topics, probs = topic_model.fit_transform(docs)

    #json save
    topic_model.save_txt(f"{results_dir}/topicos.txt")
    topic_model.save_json(f"{results_dir}/topicos.json")
    topic_model.save_params(f"{results_dir}/parametros.json")
    
    metrics = topic_model.evaluate_model(docs)
    with open(f"{results_dir}/metricas.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    
    topic_model.dominant_topics(docs, f"{sistema_nome}/{nr_topics}_topicos", ids)

    try:
        topic_model.visualize_topics().write_html(f"{graph_dir}/intertopic_map.html")
        topic_model.visualize_barchart(top_n_topics=10).write_html(f"{graph_dir}/bar_graphs.html")
        topic_model.visualize_hierarchy().write_html(f"{graph_dir}/hierarchy.html")
        topic_model.get_topic_info().to_csv(f"{graph_dir}/topic_info.csv", index=False)
    except Exception as e:
        print(f"[{sistema_nome}] Erro ao gerar gráficos: {e}")

    return topic_model

if __name__ == "__main__":
    sistemas = ["siape", "siass", "sigepe", "sougov", "totais"]
    col_texto = "Descrição do chamado"
    col_id = "Id"

    for sis in sistemas:
        csv_path = f"../data/chamados_{sis}.csv"

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, low_memory=False)
            df_clean = df.dropna(subset=[col_texto])

            docs = df_clean[col_texto].astype(str).tolist()
            ids = df_clean[col_id].tolist()

            if len(docs) > 20:
                for n_topics in [5, 10, 15]:
                    run_bertopic(docs, ids, sis.upper(), nr_topics=n_topics)
            else:
                print(f"Pulo: {sis.upper()} possui poucos documentos ({len(docs)}).")
        else:
            print(f"Arquivo não encontrado: {csv_path}")

    print("\nProcessamento em lote finalizado!")
