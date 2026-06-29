import os
import json
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rag_data_loader import carregar_dados

device = "cuda" if torch.cuda.is_available() else "cpu"

def carregar_modelo_embedding():
    base_dir = Path(__file__).resolve().parent.parent
    metadata_path = base_dir / "chatbot" / "models" / "metadata.json"
    
    caminho_modelo_local = None
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                caminho_modelo_local = metadata.get("embedding_model_local_path")
        except Exception as e:
            print(f"Aviso: Erro ao ler metadados: {e}")

    if caminho_modelo_local and os.path.exists(caminho_modelo_local):
        modelo_embedding_path = caminho_modelo_local
    else:
        modelo_embedding_path = 'paraphrase-multilingual-MiniLM-L12-v2'

    embedding_model = SentenceTransformer(modelo_embedding_path, device=device)
    return embedding_model

def get_embeddings(df_topicos, df_chamados, embedding_model):
    base_dir = Path(__file__).resolve().parent.parent
    vetores_topicos_path = base_dir / "chatbot" / "artifacts" / "embeddings_topicos.npy"
    vetores_chamados_path = base_dir / "chatbot" / "artifacts" / "embeddings_chamados.npy"
    
    vetores_topicos_path.parent.mkdir(parents=True, exist_ok=True)
    
    if vetores_topicos_path.exists():
        embeddings_topicos = np.load(vetores_topicos_path)
    else:
        print("Calculando embeddings dos TÓPICOS pela primeira vez...")
        embeddings_topicos = embedding_model.encode(
            df_topicos['texto_para_embedding'].tolist(),
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        ).astype(np.float32)
        np.save(vetores_topicos_path, embeddings_topicos)
        
    if vetores_chamados_path.exists():
        embeddings_chamados = np.load(vetores_chamados_path)
    else:
        print("Calculando embeddings dos CHAMADOS pela primeira vez...")
        embeddings_chamados = embedding_model.encode(
            df_chamados['texto_completo'].tolist(),
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        ).astype(np.float32)
        np.save(vetores_chamados_path, embeddings_chamados)
        
    return embeddings_topicos, embeddings_chamados

if __name__ == "__main__":
    df_topicos, df_chamados = carregar_dados()
    model = carregar_modelo_embedding()
    emb_t, emb_c = get_embeddings(df_topicos, df_chamados, model)
    print(f"Shape Topicos: {emb_t.shape}")
    print(f"Shape Chamados: {emb_c.shape}")
