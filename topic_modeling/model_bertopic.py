from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd
import os

def run_bertopic(docs, sistema_nome):

    print(f"\n--- [{sistema_nome}] Iniciando Processamento ---")

    # caminhos para pastas e arquivos
    graph_dir = f"bertopic_graphs/{sistema_nome}"
    model_dir = "../models"
    model_path = f"{model_dir}/model_{sistema_nome.lower()}"

    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # carregar ou treinar
    if os.path.exists(model_path):
        print(f"[{sistema_nome}] Modelo encontrado! Carregando do disco para poupar tempo...")
        topic_model = BERTopic.load(model_path)
    else:
        print(f"[{sistema_nome}] Modelo não encontrado. Iniciando treinamento pesado...")


    # uniform mainfold approximation and projection (UMAP) para redução de dimensionalidade
    # similaridade de cosseno, e fixa random state para que toda vez que a gnt rodar ele dê o mesmo resultado
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

    # hiearchical density-based spatial clustering of applications with noise (HDBSCAN) para clusterização
    # baseado em densidade, não precisa definir o número de clusters, e é robusto a ruídos
    # se tiver menos de 10 documentos parecidos, ele considera ruído (min_cluster_size)
    hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom',
                            prediction_data=True)

    # inicializa o modelo
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, nr_topics=15, calculate_probabilities=True,
                           verbose=True)

    # treina o modelo e retorna os tópicos e as probabilidades de cada documento pertencer a cada tópico
    topics, probs = topic_model.fit_transform(docs)

    topic_model.save(model_path)
    print(f"[{sistema_nome}] Modelo treinado e salvo em: {model_path}")

    # graficos
    print(f"[{sistema_nome}] gerando/atualizando visualizações...")

    try:
        topic_model.visualize_topics().write_html(f"{graph_dir}/intertopic_map.html")
        topic_model.visualize_barchart(top_n_topics=10).write_html(f"{graph_dir}/bar_graphs.html")
        topic_model.visualize_hierarchy().write_html(f"{graph_dir}/hierarchy.html")
        topic_model.get_topic_info().to_csv(f"{graph_dir}/topic_info.csv", index=False)
        print(f"[{sistema_nome}] Gráficos salvos em '{graph_dir}/'.")
    except Exception as e:
        print(f"[{sistema_nome}] Erro ao gerar gráficos: {e}")

    return topic_model, topics, probs


if __name__ == "__main__":
    sistemas = ["siape", "siass", "sigepe", "sougov"]
    col_texto = "Descrição do chamado"

    for sis in sistemas:
        csv_path = f"../data/chamados_{sis}.csv"

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, low_memory=False)
            docs = df[col_texto].dropna().astype(str).tolist()

            if len(docs) > 50:
                run_bertopic(docs, sis.upper())
            else:
                print(f"Pulo: {sis.upper()} possui poucos documentos ({len(docs)}).")
        else:
            print(f"Arquivo não encontrado: {csv_path}")

    print("\nProcessamento em lote finalizado!")


# if __name__ == "__main__":
#     csv_path = "../data/data_pre_processed.csv"
#
#     df = pd.read_csv(csv_path)
#
#     # Seleção da coluna: busca automática pela coluna 'clean' com 'descri'
#     col = ["Descrição do chamado"][0]
#     print(f"Processando coluna: {col}")
#
#     docs = df[col].dropna().astype(str).tolist()
#
#     # Execução
#     topic_model, topics, probs = run_bertopic(docs)
#
#     # Salva resumo
#     topic_model.get_topic_info().to_csv("bertopic_graphs/topic_info.csv", index=False)
#     print("Sucesso! Resultados em 'bertopic_graphs/'.")
