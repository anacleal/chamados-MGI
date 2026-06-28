import time
import numpy as np

SCORE_ALTO   = 0.70
SCORE_MEDIO  = 0.50

class RecommenderSystem:
    def __init__(self, search_engine, df_chamados):
        self.search_engine = search_engine
        self.df_chamados = df_chamados

    def buscar_chamados_similares_suporte(
        self,
        pergunta: str,
        top_k: int = 10,
        filtro_time: str = None,
        score_minimo: float = 0.20,
        top_k_topicos: int = 5,
        filtro_sistema: str = None
    ) -> list:
        resultado_hier = self.search_engine.buscar_chamados_hierarquico(
            pergunta,
            top_k_topicos=top_k_topicos,
            top_k_chamados=top_k * 3 if filtro_time else top_k,
            score_min_topico=0.20,
            filtro_sistema=filtro_sistema
        )

        chamados = resultado_hier.get('chamados', [])
        chamados = [c for c in chamados if c.get('score_similaridade', 0) >= score_minimo]

        if filtro_time:
            chamados = [
                c for c in chamados
                if filtro_time.upper() in str(c.get('Time', '')).upper()
            ]

        return chamados[:top_k]

    def buscar_por_id(self, id_chamado: str, top_k: int = 10) -> list:
        mascara = self.df_chamados['Id'].astype(str) == str(id_chamado).strip()

        if not mascara.any():
            return []

        chamado_ref = self.df_chamados[mascara].iloc[0]

        texto_ref = (
            f"{chamado_ref.get('Título', '')} "
            f"{chamado_ref.get('Descrição do chamado', '')}"
        )

        return self.buscar_chamados_similares_suporte(texto_ref, top_k=top_k)

    def analisar_padrao(self, chamados: list) -> dict:
        if not chamados:
            return {}

        times = [c.get('Time', 'N/A') for c in chamados]
        tempos = [c.get('Tempo em horas do atendimento', 0) for c in chamados]
        scores = [c.get('score_similaridade', 0) for c in chamados]
        categorias = [c.get('Categoria', 'N/A') for c in chamados]

        time_mais_freq = max(set(times), key=times.count)
        freq_time = times.count(time_mais_freq)
        categoria_freq = max(set(categorias), key=categorias.count)
        
        tempos_validos = [t for t in tempos if isinstance(t, (int, float)) and not np.isnan(t) and t > 0]
        tempo_medio = float(np.mean(tempos_validos)) if tempos_validos else 0.0
        score_medio = float(np.mean(scores)) if scores else 0.0
        concentrado = freq_time >= len(chamados) * 0.6

        return {
            "time_mais_frequente": time_mais_freq,
            "frequencia_time": freq_time,
            "total_chamados": len(chamados),
            "categoria_frequente": categoria_freq,
            "tempo_medio_h": tempo_medio,
            "score_medio": score_medio,
            "padrao_concentrado": concentrado,
        }

    def recomendar_solucao(self, pergunta: str, top_k: int = 3, filtro_time: str = None, filtro_sistema: str = None) -> dict:
        inicio = time.time()
        chamados = self.buscar_chamados_similares_suporte(
            pergunta, top_k=top_k, filtro_time=filtro_time, filtro_sistema=filtro_sistema
        )
        
        melhor_score = chamados[0]['score_similaridade'] if chamados else 0.0
        padrao = self.analisar_padrao(chamados)
        
        if not chamados:
            confianca = "Nenhuma"
        elif melhor_score >= SCORE_ALTO:
            confianca = "Alta"
        elif melhor_score >= SCORE_MEDIO:
            confianca = "Média"
        else:
            confianca = "Baixa"
            
        return {
            "confianca": confianca,
            "melhor_score": melhor_score,
            "chamados_recomendados": chamados,
            "padrao": padrao,
            "tempo_s": time.time() - inicio
        }
