from bertopic import BERTopic
import pandas as pd
import numpy as np
import os

def generate_bertopic_visualizations(topic_model, docs=None, output_dir="bertopic_graphs"):
    #docs - documentos originais para gerar hierarquia detalhada

    os.makedirs(output_dir, exist_ok=True)
    print("Gerando visualizações...")

    print("Salvando mapa de distância intertópicos...")
    topic_model.visualize_topics().write_html(os.path.join(output_dir, "intertopic_distance_map.html"))

    print("Salvando gráfico de barras dos tópicos...")
    topic_model.visualize_barchart(top_n_topics=10).write_html(os.path.join(output_dir, "bar_graphs.html"))

    print("Processando hierarquia...")
    topic_model.visualize_hierarchy().write_html(os.path.join(output_dir, "hierarchical_clustering.html"))

    print("Salvando informações dos tópicos...")
    topic_model.get_topic_info().to_csv(os.path.join(output_dir, "topic_info.csv"), index=False)
    topic_model.get_representative_docs()

    print(f"Sucesso! Gráficos gerados na pasta: {output_dir}")


def load_and_visualize(model_path="/model", output_dir="bertopic_graphs"):
    if not os.path.exists(model_path):
        print(f"Erro: Modelo não encontrado em {model_path}. Treine o modelo primeiro.")
        return

    print(f"Carregando o modelo de {model_path}...")
    topic_model = BERTopic.load(model_path)
    
    generate_bertopic_visualizations(topic_model, output_dir=output_dir)


if __name__ == "__main__":
    # Quando executado diretamente de topic_modeling/
    load_and_visualize()