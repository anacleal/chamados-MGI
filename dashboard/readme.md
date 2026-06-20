# Dashboard de Análise de Chamados — MGI

Painel para identificação de gargalos operacionais nos sistemas SIASS, SIAPE,
SIGEPE, SOUGOV e TOTAIS, a partir da modelagem de tópicos (BERTopic) e da
sumarização automática (Llama 3.1 via Ollama).

## Estrutura de pastas esperada

```
raiz-do-projeto/
├── data/
│   └── chamados_{sistema}.csv          # com coluna "Data de abertura"
├── topic_modeling/
│   ├── bertopic_resultados/{SISTEMA}/
│   │   ├── Resumo_Topicos_Dominantes.csv
│   │   ├── Topicos_Dominantes.csv
│   │   └── topicos.json
│   └── models/{SISTEMA}/modelo          # modelo BERTopic salvo
├── summarization/
│   └── outLLM/detailed_summarization/{SISTEMA}/
│       ├── summary_topic_{N}.txt
│       └── titulo_topic_{N}.txt
└── dashboard/                           # <- esta pasta
    ├── app.py
    ├── data_loader.py
    ├── assets/style.css
    └── requirements.txt
```

## Como rodar

```bash
cd dashboard
pip install -r requirements.txt
python app.py
```

Acesse http://127.0.0.1:8050

## O que cada arquivo faz

- **data_loader.py** — única fonte de leitura de dados. Lê os CSVs, JSONs e
  .txt gerados pelo pipeline (`model_bertopic.py`, `summarization.py`,
  `title.py`) e monta as tabelas consolidadas que o dashboard consome.
  Se você mudar algum caminho de saída no pipeline, ajuste aqui.

- **app.py** — layout e callbacks do Dash. Não lê arquivo nenhum diretamente;
  tudo passa por `data_loader.py`.

- **assets/style.css** — carregado automaticamente pelo Dash (qualquer CSS
  dentro de `assets/` é injetado sem precisar referenciar manualmente).

## Mapa intertópicos

As coordenadas 2D são calculadas a partir do modelo BERTopic salvo
(`topic_embeddings_` reduzido pelo `umap_model` do próprio modelo treinado).
Se o modelo não for encontrado em `topic_modeling/models/{SISTEMA}/modelo`,
o dashboard cai automaticamente em um layout circular simples (apenas para
não quebrar a visualização) — nesse caso, treine/salve o modelo com
`model_bertopic.py` para ter o posicionamento real.

## Evolução temporal

Usa a coluna `Data de abertura` do CSV de chamados, cruzada com o tópico
dominante de cada chamado (via `Resumo_Topicos_Dominantes.csv`). Se essa
coluna não existir ou o cruzamento de IDs falhar, o painel mostra um aviso
em vez de quebrar.

## Navegação

- Clique no nome de um sistema na barra lateral para trocar de sistema
  (reseta a seleção de tópico).
- Clique em um círculo no mapa de tópicos, ou em uma barra no ranking, para
  abrir o cartão de diagnóstico (Padrão Dominante + Impacto Operacional) e
  filtrar a evolução mensal por aquele tópico especificamente.
- A tabela no rodapé sempre mostra todos os tópicos do sistema atual,
  com busca e ordenação nativas.