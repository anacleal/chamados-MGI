import faiss
import numpy as np
import pandas as pd
from collections import defaultdict
from embeddings_manager import carregar_modelo_embedding

class SearchEngine:
    def __init__(self, df_topicos, df_chamados, embeddings_topicos, embeddings_chamados, embedding_model):
        self.df_topicos = df_topicos
        self.df_chamados = df_chamados
        self.embeddings_topicos = embeddings_topicos
        self.embeddings_chamados = embeddings_chamados
        self.embedding_model = embedding_model
        
        # Mapeamento
        self.mapa_topico_para_indices = defaultdict(list)
        for idx, valor_topico in enumerate(self.df_chamados['topico_id_unico']):
            if pd.isna(valor_topico) or valor_topico == "NAO_CLASSIFICADO":
                continue
            lista_topicos = str(valor_topico).split(",")
            for t in lista_topicos:
                self.mapa_topico_para_indices[t.strip()].append(idx)
                
        self.mapa_topico_para_indices = {k: np.array(v) for k, v in self.mapa_topico_para_indices.items()}
        
        # Criação de índices
        self.indice_topicos = faiss.IndexFlatIP(self.embeddings_topicos.shape[1])
        self.indice_topicos.add(self.embeddings_topicos)
        
        self.indice_chamados = faiss.IndexFlatIP(self.embeddings_chamados.shape[1])
        self.indice_chamados.add(self.embeddings_chamados)

    def buscar_topicos_relevantes(self, pergunta: str, top_k: int = 3) -> list:
        emb_pergunta = self.embedding_model.encode(
            [pergunta], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.indice_topicos.search(emb_pergunta, top_k)

        resultados = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            topico = self.df_topicos.iloc[idx].to_dict()
            topico['score_topico'] = float(score)
            resultados.append(topico)

        return resultados

    def aplicar_prioridade_sistema(self, topicos_recuperados: list) -> list:
        especificos = [t for t in topicos_recuperados if t['sistema_origem'] != 'Geral']
        gerais = [t for t in topicos_recuperados if t['sistema_origem'] == 'Geral']

        if not especificos:
            return topicos_recuperados
        if not gerais:
            return especificos

        melhor_especifico = max(t['score_topico'] for t in especificos)
        gerais_relevantes = [
            t for t in gerais
            if t['score_topico'] > melhor_especifico + 0.10
        ]

        return especificos + gerais_relevantes

    def buscar_chamados_hierarquico(self, pergunta: str, top_k_topicos: int = 3, top_k_chamados: int = 5, score_min_topico: float = 0.30, filtro_sistema: str = None) -> dict:
        busca_k = top_k_topicos * 30 if filtro_sistema and filtro_sistema != "TOTAIS" else top_k_topicos
        topicos_brutos = self.buscar_topicos_relevantes(pergunta, top_k=busca_k)

        if filtro_sistema and filtro_sistema != "TOTAIS":
            topicos_brutos = [
                t for t in topicos_brutos 
                if str(t.get('sistema_origem', '')).upper() == str(filtro_sistema).upper()
            ][:top_k_topicos]

        topicos_brutos = [t for t in topicos_brutos if t['score_topico'] >= score_min_topico]

        if not topicos_brutos:
            return {
                "chamados": [],
                "topicos_usados": [],
                "sistema_detectado": filtro_sistema
            }

        if filtro_sistema and filtro_sistema != "TOTAIS":
            topicos_finais = topicos_brutos
        else:
            topicos_finais = self.aplicar_prioridade_sistema(topicos_brutos)
            
        ids_topicos = [t['topico_id_unico'] for t in topicos_finais]
        
        arrays_de_indices = [
            self.mapa_topico_para_indices[tid] for tid in ids_topicos
            if tid in self.mapa_topico_para_indices
        ]

        if not arrays_de_indices:
            return {
                "chamados": [],
                "topicos_usados": topicos_finais,
                "sistema_detectado": topicos_finais[0]['sistema_origem']
            }

        indices_subconjunto = np.unique(np.concatenate(arrays_de_indices))
        
        embs_subconjunto = self.embeddings_chamados[indices_subconjunto]
        indice_temp = faiss.IndexFlatIP(embs_subconjunto.shape[1])
        indice_temp.add(embs_subconjunto)

        emb_pergunta = self.embedding_model.encode(
            [pergunta], normalize_embeddings=True
        ).astype(np.float32)

        k_busca = min(top_k_chamados, len(indices_subconjunto))
        scores, pos_locais = indice_temp.search(emb_pergunta, k_busca)

        chamados_resultado = []
        for score, pos_local in zip(scores[0], pos_locais[0]):
            if pos_local == -1:
                continue
            idx_global = indices_subconjunto[pos_local]
            chamado = self.df_chamados.iloc[idx_global].to_dict()
            chamado['score_similaridade'] = float(score)
            chamados_resultado.append(chamado)

        sistema_detectado = max(topicos_finais, key=lambda t: t['score_topico'])['sistema_origem']

        return {
            "chamados": chamados_resultado,
            "topicos_usados": topicos_finais,
            "sistema_detectado": sistema_detectado,
            "n_chamados_filtrados": len(indices_subconjunto)
        }
