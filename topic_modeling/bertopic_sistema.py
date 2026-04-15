from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd
import os


def processar_sistema_bertopic(docs, sistema_nome):
    print(f"\n--- [{sistema_nome}] Iniciando Processamento ---")

    # Caminhos para pastas e arquivos
    graph_dir = f"bertopic_graphs/{sistema_nome}"
    model_dir = "../models"
    model_path = f"{model_dir}/model_{sistema_nome.lower()}"

    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # LÓGICA DE PERSISTÊNCIA: Carregar ou Treinar
    if os.path.exists(model_path):
        print(f"[{sistema_nome}] Modelo encontrado! Carregando do disco para poupar tempo...")
        topic_model = BERTopic.load(model_path)
    else:
        print(f"[{sistema_nome}] Modelo não encontrado. Iniciando treinamento pesado...")

        # Configurações do Pipeline
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom',
                                prediction_data=True)

        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=15,
            calculate_probabilities=True,
            verbose=True
        )

        # Treinamento
        topic_model.fit_transform(docs)

        # SALVAR o modelo para a próxima vez
        topic_model.save(model_path)
        print(f"[{sistema_nome}] Modelo treinado e salvo em: {model_path}")

    # GERAÇÃO DE GRÁFICOS (Isso rodará sempre, seja carregando ou treinando)
    print(f"[{sistema_nome}] Gerando/Atualizando visualizações...")

    try:
        topic_model.visualize_topics().write_html(f"{graph_dir}/intertopic_map.html")
        topic_model.visualize_barchart(top_n_topics=10).write_html(f"{graph_dir}/bar_graphs.html")
        topic_model.visualize_hierarchy().write_html(f"{graph_dir}/hierarchy.html")
        topic_model.get_topic_info().to_csv(f"{graph_dir}/topic_info.csv", index=False)
        print(f"[{sistema_nome}] Gráficos salvos em '{graph_dir}/'.")
    except Exception as e:
        print(f"[{sistema_nome}] Erro ao gerar gráficos: {e}")


if __name__ == "__main__":
    sistemas = ["siape", "siass", "sigepe", "sougov"]
    col_texto = "Descrição do chamado"

    for sis in sistemas:
        csv_path = f"../data/chamados_{sis}.csv"

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, low_memory=False)
            docs = df[col_texto].dropna().astype(str).tolist()

            if len(docs) > 50:
                processar_sistema_bertopic(docs, sis.upper())
            else:
                print(f"Pulo: {sis.upper()} possui poucos documentos ({len(docs)}).")
        else:
            print(f"Arquivo não encontrado: {csv_path}")

    print("\n✅ Processamento em lote finalizado!")