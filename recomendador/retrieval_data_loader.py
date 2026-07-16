import pandas as pd
import textwrap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def criar_texto_combinado(row):
    partes = [
        f"Título: {row.get('Título', '')}",
        f"Descrição: {row.get('Descrição do chamado', '')}"
    ]
    return " | ".join(partes)

def limpar_texto(texto):
    texto = texto.strip()
    texto = ' '.join(texto.split())
    return texto

def carregar_dados():
    base_dir = Path(__file__).resolve().parent.parent
    path_topicos = base_dir / 'data' / 'recomendador' / 'df_topicos.csv'
    path_chamados = base_dir / 'data' / 'recomendador' / 'df_chamados.csv'
    
    # Nível 1 — Tópicos
    df_topicos = pd.read_csv(path_topicos)
    df_topicos = df_topicos.rename(columns={
        'Sistema':         'sistema_origem',
        'Tópico':          'topico_id_original',
        'Título':          'label_curto',
        'Nº Documentos':   'n_documentos',
    })
    
    df_topicos['topico_id_unico'] = (
        df_topicos['sistema_origem'].str.upper().str.replace(' ', '') +
        '_' +
        df_topicos['topico_id_original'].astype(str)
    )
    
    df_topicos['texto_para_embedding'] = (
        df_topicos['label_curto'] + '. ' +
        df_topicos['padrao_dominante'].fillna('') + ' ' +
        df_topicos['impacto_operacional'].fillna('')
    ).str.strip()
    
    # Nível 2 — Chamados
    df_chamados = pd.read_csv(path_chamados)
    df_chamados['texto_completo'] = df_chamados.apply(criar_texto_combinado, axis=1)
    df_chamados['texto_completo'] = df_chamados['texto_completo'].apply(limpar_texto)
    
    return df_topicos, df_chamados

if __name__ == "__main__":
    df_t, df_c = carregar_dados()
    print(f"Topicos carregados: {len(df_t)}")
    print(f"Chamados carregados: {len(df_c)}")
