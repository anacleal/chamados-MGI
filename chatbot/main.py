from rag_data_loader import carregar_dados
from embeddings_manager import carregar_modelo_embedding, get_embeddings
from search_engine import SearchEngine
from recommender import RecommenderSystem

def exibir_recomendacoes(resultado: dict):
    print("\n" + "=" * 70)
    print("SISTEMA DE RECOMENDAÇÃO DE SUPORTE")
    print("=" * 70)
    
    chamados = resultado.get('chamados_recomendados', [])
    
    if not chamados:
        print("Status: Nenhum chamado similar encontrado para recomendação.")
        print("=" * 70)
        return
        
    print(f"Confiança do Match : {resultado['confianca']} (Score máximo: {resultado['melhor_score']:.1%})")
    
    padrao = resultado.get("padrao", {})
    if padrao:
        print(f"Time provável      : {padrao.get('time_mais_frequente', 'N/A')} "
              f"({padrao.get('frequencia_time', 0)}/{padrao.get('total_chamados', 0)} chamados parecidos)")
              
    print("\n" + "-" * 70)
    print(f"MELHORES SOLUÇÕES HISTÓRICAS ({len(chamados)} listadas):")
    print("-" * 70)
    
    for i, c in enumerate(chamados, 1):
        id_ = str(c.get('Id', 'N/A'))
        titulo = str(c.get('Título', 'N/A'))
        time_ = str(c.get('Time', 'N/A'))
        score = c.get('score_similaridade', 0)
        
        desc = str(c.get('Descrição do chamado', 'Sem descrição')).replace('\n', ' ')
        acao = str(c.get('Última ação de acompanhamento', 'Sem ação registrada')).replace('\n', ' ')
        
        if len(desc) > 180: desc = desc[:177] + "..."
        if len(acao) > 180: acao = acao[:177] + "..."
            
        print(f"[{i}] RECOMENDAÇÃO (Match: {score:.0%}) | ID: {id_} | Time: {time_}")
        print(f"    Título   : {titulo}")
        print(f"    Problema : {desc}")
        print(f"    Solução  : {acao}\n")
        
    print("-" * 70)
    print(f"Tempo de busca: {resultado['tempo_s']:.3f}s")
    print("=" * 70)

def recomendador_interativo_suporte(recommender, df_chamados):
    historico = []
    filtro_time = None
    top_k_atual = 3

    print("\n" + "=" * 70)
    print("  SISTEMA RECOMENDADOR DE SUPORTE TÉCNICO (MODO TESTE)")
    print("=" * 70)
    print("""
  Descreva o problema ou use um dos comandos abaixo:
  /time <nome>   filtra recomendação para um time especifico
  /id <numero>   recomenda baseado em um ID existente
  /top <numero>  define quantos chamados exibir (ex: /top 5)
  /times         lista todos os times disponíveis
  sair           encerra a sessao
  """ + "-" * 70)

    while True:
        try:
            entrada = input("\nVocê: ").strip()
        except EOFError:
            break

        if not entrada:
            continue

        if entrada.lower() in ['sair', 'exit', 'quit']:
            print(f"\nSessão encerrada. Consultas realizadas: {len(historico)}\n")
            break

        if entrada.lower() == '/times':
            print("\nTimes disponíveis na base:")
            for time_nome, count in df_chamados['Time'].value_counts().items():
                print(f"  {time_nome:<45} {count:>6} chamados")
            continue

        if entrada.lower().startswith('/top '):
            novo_top = entrada[5:].strip()
            if novo_top.isdigit() and int(novo_top) > 0:
                top_k_atual = int(novo_top)
                print(f"  [Configuração] O sistema agora listará as {top_k_atual} melhores recomendações.")
            else:
                print("  [Erro] Por favor, informe um número válido maior que zero. (Ex: /top 5)")
            continue

        if entrada.lower().startswith('/id '):
            id_chamado = entrada[4:].strip()
            print(f"\n  Buscando as {top_k_atual} melhores recomendações baseadas no ID {id_chamado}...")
            
            chamados_similares = recommender.buscar_por_id(id_chamado, top_k=top_k_atual)
            
            if not chamados_similares:
                print(f"  ID {id_chamado} não encontrado.")
                continue
                
            melhor_score = chamados_similares[0]['score_similaridade'] if chamados_similares else 0
            resultado_id = {
                "confianca": "Busca Exata por ID",
                "melhor_score": melhor_score,
                "chamados_recomendados": chamados_similares,
                "padrao": recommender.analisar_padrao(chamados_similares),
                "tempo_s": 0.01
            }
            exibir_recomendacoes(resultado_id)
            historico.append(entrada)
            continue

        filtro_ativo = None
        pergunta = entrada

        if entrada.lower().startswith('/time '):
            resto = entrada[6:].strip()
            partes = resto.split(' ', 1)
            filtro_ativo = partes[0]
            pergunta = partes[1] if len(partes) > 1 else ""
            if not pergunta:
                print(f"  Filtro ativado para o time: {filtro_ativo}")
                filtro_time = filtro_ativo
                continue
            print(f"\n  Procurando no time: {filtro_ativo}")

        print(f"\n  Procurando as {top_k_atual} melhores soluções...")
        
        resultado = recommender.recomendar_solucao(
            pergunta,
            top_k=top_k_atual,
            filtro_time=filtro_ativo or filtro_time
        )
        
        exibir_recomendacoes(resultado)
        historico.append(entrada)
        
        if filtro_ativo:
            filtro_time = None

if __name__ == "__main__":
    print("Inicializando o sistema recomendador...")
    df_topicos, df_chamados = carregar_dados()
    model = carregar_modelo_embedding()
    emb_t, emb_c = get_embeddings(df_topicos, df_chamados, model)
    
    search_engine = SearchEngine(df_topicos, df_chamados, emb_t, emb_c, model)
    recommender = RecommenderSystem(search_engine, df_chamados)
    
    print("Inicialização concluída.\n")
    recomendador_interativo_suporte(recommender, df_chamados)
