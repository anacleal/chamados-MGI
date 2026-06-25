from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from chatbot.config import ARTIFACTS_DIR


COL_DUVIDA = "Descrição do chamado"
COL_SOLUCAO = "Última ação de acompanhamento"
SEED = 42

RESULTADOS_DIR = ARTIFACTS_DIR.parent / "resultados_avaliacao"

TEXTO_INTRODUCAO = """Seção 1 de 2
Validação Especializada de Inteligência Artificial: Gestão de Chamados da Folha (MGI)

Prezada equipe,
Estamos testando ferramentas de Inteligência Artificial para otimizar o atendimento dos chamados nos nossos sistemas estruturantes. Como a área de Folha de Pagamento tem alta demanda, sua expertise é fundamental para validarmos a qualidade técnica das respostas geradas pela IA.

Como vai funcionar? Você avaliará 10 cenários reais (anonimizados). Para cada um, a IA gerou duas respostas:
- Chatbot Interno: Um "copiloto" para a equipe técnica, que resume casos históricos parecidos para acelerar a resolução do chamado.
- Chatbot Público: Um "autoatendimento" focado em resolver a dúvida do usuário na origem, diminuindo a abertura de chamados simples.

O que avaliar? Para cada resposta, pedimos que você avalie 3 frentes (Compreensão, Utilidade/Eficácia e Clareza) atribuindo uma nota em uma escala de 1 a 5, onde 1 significa "Discordo Totalmente" e 5 significa "Concordo Totalmente".

Nota de Confidencialidade: Todos os dados pessoais dos chamados reais foram anonimizados.

Agradecemos imensamente o seu tempo!"""


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed)


def criar_formularios(rag: ChatbotRAG, seed: int = SEED) -> dict[str, pd.DataFrame]:
    df_work = rag._ensure_artifacts().df_chamados.copy()

    if COL_DUVIDA not in df_work.columns:
        raise KeyError(f"Coluna obrigatoria ausente: {COL_DUVIDA}")
    if COL_SOLUCAO not in df_work.columns:
        df_work[COL_SOLUCAO] = ""

    df_work[COL_DUVIDA] = df_work[COL_DUVIDA].fillna("").astype(str).str.strip()
    df_work[COL_SOLUCAO] = df_work[COL_SOLUCAO].fillna("").astype(str).str.strip()

    mask_sol_preenchida = (df_work[COL_SOLUCAO] != "") & (df_work[COL_SOLUCAO].str.lower() != "nan")
    mask_sol_detalhada = df_work[COL_SOLUCAO].str.len() >= 30

    kw_criticos = "caiu|parou|urgente|servidor fora|não abre|nao abre|trava|ruim|lixo"
    mask_criticos_len = df_work[COL_DUVIDA].str.len() < 30
    mask_criticos_kw = df_work[COL_DUVIDA].str.contains(kw_criticos, case=False, na=False, regex=True)
    df_criticos_total = df_work[mask_criticos_len | mask_criticos_kw]
    amostra_criticos = _sample(df_criticos_total, 15, seed)

    kw_complexos = "erro|falha|sistema|integração|integracao|folha|carregar|bug|atualização|atualizacao"
    mask_complexos_len = df_work[COL_DUVIDA].str.len() > 100
    mask_complexos_kw = df_work[COL_DUVIDA].str.contains(kw_complexos, case=False, na=False, regex=True)
    df_complexos_total = df_work[(mask_complexos_len) & (mask_complexos_kw) & (mask_sol_detalhada)].drop(
        amostra_criticos.index, errors="ignore"
    )
    amostra_complexos = _sample(df_complexos_total, 20, seed)

    kw_simples = "senha|acesso|contracheque|desbloquear|entrar|login|cadastro"
    mask_simples_len = df_work[COL_DUVIDA].str.len() < 100
    mask_simples_kw = df_work[COL_DUVIDA].str.contains(kw_simples, case=False, na=False, regex=True)
    indices_usados = amostra_criticos.index.union(amostra_complexos.index)
    df_simples_total = df_work[(mask_simples_len | mask_simples_kw) & (mask_sol_preenchida)].drop(
        indices_usados, errors="ignore"
    )
    amostra_simples = _sample(df_simples_total, 15, seed)

    formularios = {}
    for i in range(5):
        slice_simples = amostra_simples.iloc[i * 3 : (i + 1) * 3]
        slice_complexos = amostra_complexos.iloc[i * 4 : (i + 1) * 4]
        slice_criticos = amostra_criticos.iloc[i * 3 : (i + 1) * 3]
        form_atual = pd.concat([slice_simples, slice_complexos, slice_criticos])
        form_atual = form_atual.sample(frac=1, random_state=seed + i).reset_index(drop=True)
        formularios[f"form_{i + 1}"] = form_atual

    total = sum(len(df) for df in formularios.values())
    print(f"Amostragem concluida: {total} chamados divididos em {len(formularios)} formularios.")
    return formularios


def gerar_resposta_copiloto_interno(rag: ChatbotRAG, pergunta_usuario: str) -> str:
    res = rag.chatbot_rag_suporte(pergunta_usuario, top_k=10, verbose=False)
    return res["resposta"]


def gerar_resposta_autoatendimento_publico(rag: ChatbotRAG, pergunta_usuario: str) -> str:
    res = rag.chatbot_rag(pergunta_usuario, top_k=10, verbose=False)
    return res["resposta"]


def preencher_respostas(
    rag: ChatbotRAG,
    formularios: dict[str, pd.DataFrame],
    out_dir: Path = RESULTADOS_DIR,
) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for nome_form, df_form in formularios.items():
        print(f"\nProcessando {nome_form}...")
        respostas_internas = []
        respostas_publicas = []
        iterator = df_form.iterrows()
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(df_form), desc=nome_form)

        for _, row in iterator:
            pergunta = str(row.get(COL_DUVIDA, ""))
            try:
                resp_a = gerar_resposta_copiloto_interno(rag, pergunta)
            except Exception as exc:
                resp_a = f"[ERRO IA - INTERNO]: {exc}"

            try:
                resp_b = gerar_resposta_autoatendimento_publico(rag, pergunta)
            except Exception as exc:
                resp_b = f"[ERRO IA - PUBLICO]: {exc}"

            respostas_internas.append(resp_a)
            respostas_publicas.append(resp_b)

        df_form = df_form.copy()
        df_form["Resposta_Chatbot_Interno"] = respostas_internas
        df_form["Resposta_Chatbot_Publico"] = respostas_publicas
        formularios[nome_form] = df_form

        nome_arquivo = out_dir / f"{nome_form}_Tabela_IA.xlsx"
        df_form.to_excel(nome_arquivo, index=False)
        print(f"Excel salvo: {nome_arquivo}")

    return formularios


def salvar_amostras_sem_respostas(
    formularios: dict[str, pd.DataFrame],
    out_dir: Path = RESULTADOS_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for nome_form, df_form in formularios.items():
        nome_arquivo = out_dir / f"{nome_form}_Amostra.xlsx"
        df_form.to_excel(nome_arquivo, index=False)
        print(f"Amostra salva: {nome_arquivo}")


def gerar_txts_formulario(
    formularios: dict[str, pd.DataFrame],
    out_dir: Path = RESULTADOS_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for nome_form, df_form in formularios.items():
        caminho_arquivo = out_dir / f"{nome_form}_texto_para_copiar.txt"
        with caminho_arquivo.open("w", encoding="utf-8") as f:
            f.write(TEXTO_INTRODUCAO + "\n=========================================\nSeção 2 de 2\n")

            for i, row in df_form.iterrows():
                chamado = row.get(COL_DUVIDA, "")
                resp_a = row.get("Resposta_Chatbot_Interno", "")
                resp_b = row.get("Resposta_Chatbot_Publico", "")

                bloco = f"""
-----------------------------------------------------------------------------
Cenário {i + 1}/10
-----------------------------------------------------------------------------
CHAMADO ORIGINAL DO USUÁRIO:
{chamado}

RESPOSTA MODELO A (CHATBOT INTERNO)
Texto da IA:
{resp_a}

Avalie o Modelo A (Escala 1 a 5):
[ ] Compreensão e Relevância: O retorno demonstra que o chamado foi compreendido.
[ ] Utilidade Prática: O plano de ação sugerido aceleraria a resolução deste chamado.
[ ] Clareza e Objetividade: O texto vai direto ao ponto, sem redundâncias.

RESPOSTA MODELO B (CHATBOT PÚBLICO)
Texto da IA:
{resp_b}

Avalie o Modelo B (Escala 1 a 5):
[ ] Compreensão e Relevância: O retorno demonstra que o chamado foi compreendido.
[ ] Eficácia: A resposta resolve a dúvida de forma autônoma, evitando abertura de chamado.
[ ] Clareza e Objetividade: A linguagem é acessível e estruturada.

"""
                f.write(bloco)
        print(f"TXT salvo: {caminho_arquivo}")


def gerar_formularios_avaliacao(
    rag: ChatbotRAG,
    out_dir: Path = RESULTADOS_DIR,
    preencher_ia: bool = True,
) -> dict[str, pd.DataFrame]:
    formularios = criar_formularios(rag)
    if preencher_ia:
        formularios = preencher_respostas(rag, formularios, out_dir=out_dir)
    else:
        salvar_amostras_sem_respostas(formularios, out_dir=out_dir)
    gerar_txts_formulario(formularios, out_dir=out_dir)
    print(f"\nProcesso finalizado. Arquivos salvos em: {out_dir}")
    return formularios


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera formularios de avaliacao do chatbot.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--out-dir", default=str(RESULTADOS_DIR))
    parser.add_argument(
        "--somente-amostra",
        action="store_true",
        help="Gera Excel/TXT sem chamar os dois chatbots para preencher respostas.",
    )
    args = parser.parse_args()

    rag = ChatbotRAG(device=args.device)
    gerar_formularios_avaliacao(
        rag,
        out_dir=Path(args.out_dir),
        preencher_ia=not args.somente_amostra,
    )


if __name__ == "__main__":
    main()
