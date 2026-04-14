from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd
import os
from generate_graphs import generate_bertopic_visualizations

def train_model(docs, model_path="model"):

    #o bertopic usa
    # uniform mainfold approximation and projection (UMAP) para redução de dimensionalidade
    # similaridade de cosseno, e fixa random state para que toda vez que a gnt rodar ele dê o mesmo resultado
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

    # hiearchical density-based spatial clustering of applications with noise (HDBSCAN) para clusterização
    # baseado em densidade, não precisa definir o número de clusters, e é robusto a ruídos
    # se tiver menos de 15 documentos parecidos, ele considera ruído (min_cluster_size)
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom',
                            prediction_data=True)

    # inicializa o modelo
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, nr_topics=15, calculate_probabilities=True,
                           verbose=True)

    # treina o modelo
    print("Treinando o modelo bertopic...")
    topics, probs = topic_model.fit_transform(docs)

    # Garante que a pasta model existe
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print(f"Salvando o modelo em {model_path}...")
    topic_model.save(model_path)

    return topic_model, topics, probs

if __name__ == "__main__":
    csv_path = "../data/data_pre_processed.csv"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "model", "model_bertopic")

    # --- DEBUG DE CAMINHO ---
    diretorio_salvamento = os.path.dirname(model_path)
    print(f"DEBUG: O modelo será salvo na pasta: {os.path.abspath(diretorio_salvamento)}")

    # Testa se a pasta existe, se não, tenta criar
    try:
        os.makedirs(diretorio_salvamento, exist_ok=True)
        # Tenta criar um arquivo temporário só para testar permissão de escrita
        test_file = os.path.join(diretorio_salvamento, "test_perm.txt")
        with open(test_file, "w") as f:
            f.write("teste")
        os.remove(test_file)
        print("DEBUG: Permissão de escrita confirmada! ✅")
    except Exception as e:
        print(f"DEBUG: ERRO de permissão ou caminho: {e} ❌")
        exit()  # Para a execução antes de começar o treino demorado
    # -------------------------

    df = pd.read_csv(csv_path)

    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo {csv_path} não encontrado! Verifique o caminho.")
    else:
        df = pd.read_csv(csv_path)

        col = "Descrição do chamado"
        if col not in df.columns:
            print("Nome da coluna incorreto.")
            
        print(f"Processando coluna: {col}")

        # Remove vazios e converte para string
        docs = df[col].dropna().astype(str).tolist()

        # treino
        topic_model, topics, probs = train_model(docs, model_path=model_path)

        print("Gerando visualizações e salvando topic_info.csv...")
        output_graphs_dir = "./bertopic_graphs"
        generate_bertopic_visualizations(topic_model, docs=docs, output_dir=output_graphs_dir)

        print("Sucesso! Modelo e gráficos (com docs representativos) salvos.")