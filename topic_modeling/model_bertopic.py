from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd
import os

def run_bertopic(docs):
    # cria a pasta dos graficos
    os.makedirs("bertopic_graphs", exist_ok=True)

    # uniform mainfold approximation and projection (UMAP) para redução de dimensionalidade
    # similaridade de cosseno, e fixa random state para que toda vez que a gnt rodar ele dê o mesmo resultado
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

    # hiearchical density-based spatial clustering of applications with noise (HDBSCAN) para clusterização
    # baseado em densidade, não precisa definir o número de clusters, e é robusto a ruídos
    # se tiver menos de 10 documentos parecidos, ele considera ruído (min_cluster_size)
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom',
                            prediction_data=True)

    # inicializa o modelo
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, calculate_probabilities=True,
                           verbose=True)

    # treina o modelo e retorna os tópicos e as probabilidades de cada documento pertencer a cada tópico
    topics, probs = topic_model.fit_transform(docs)

    # visualizações
    print("Gerando visualizações...")
    topic_model.visualize_topics().write_html("bertopic_graphs/intertopic_distance_map.html")
    topic_model.visualize_barchart(top_n_topics=10).write_html("bertopic_graphs/bar_graphs.html")
    topic_model.visualize_hierarchy().write_html("bertopic_graphs/hierarchical_clustering.html")

    return topic_model, topics, probs

if __name__ == "__main__":
    csv_path = "../data/data_pre_processed.csv"

    df = pd.read_csv(csv_path)

    # Seleção da coluna: busca automática pela coluna 'clean' com 'descri'
    col = ["Descrição do chamado"][0]
    print(f"Processando coluna: {col}")

    docs = df[col].dropna().astype(str).tolist()

    # Execução
    topic_model, topics, probs = run_bertopic(docs)

    # Salva resumo
    topic_model.get_topic_info().to_csv("bertopic_graphs/topic_info.csv", index=False)
    print("Sucesso! Resultados em 'bertopic_graphs/'.")
