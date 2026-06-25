# Chatbot RAG de Suporte

## Gerar CSVs para o chatbot

```bash
python -m chatbot.build_datasets
```

Saídas:

- `data/chatbot/df_topicos.csv`
- `data/chatbot/df_chamados.csv`

## Criar embeddings e índices FAISS

```bash
pip install -r chatbot/requirements.txt
python -m chatbot.rag --build-index --device cpu
```

Saídas:

- `chatbot/artifacts/indice_topicos.faiss`
- `chatbot/artifacts/indice_chamados.faiss`
- `chatbot/artifacts/embeddings_topicos.npy`
- `chatbot/artifacts/embeddings_chamados.npy`
- `chatbot/artifacts/modelos_utilizados.json`

## Salvar modelos localmente

```bash
python -m chatbot.model_store --embedding
python -m chatbot.model_store --llm
```

Em uma CPU AMD Ryzen 5 5500, o modelo de embeddings deve rodar bem para gerar e consultar
vetores, embora a primeira indexacao dos 76 mil chamados possa demorar alguns minutos. O LLM
`Qwen/Qwen2.5-3B-Instruct` em `transformers` roda em tese na CPU, mas nao e o melhor formato:
em `float32` pode passar de 8 GB de RAM e a geracao tende a ser lenta. Para uso local sem GPU,
prefira uma versao quantizada via Ollama/llama.cpp, ou troque o `llm_model_name` por um modelo
menor/quantizado.

## Consulta rápida

```bash
python -m chatbot.rag --pergunta "nao consigo acessar o sougov" --top-k 5 --device cpu
```

## Chat interativo

Autoatendimento publico:

```bash
python -m chatbot.rag --interactive publico --top-k 5 --device cpu
```

Modulo interno para analistas:

```bash
python -m chatbot.rag --interactive suporte --top-k 7 --device cpu
```

Comandos do modo publico: `historico`, `stats`, `sair`.
Comandos do modo suporte: `/times`, `/time <nome>`, `/id <numero>`, `historico`, `stats`, `sair`.

## Diagnosticar recuperacao

Mostra topicos e chamados recuperados sem chamar o LLM:

```bash
python -m chatbot.rag --pergunta "nao consigo acessar o sougov" --top-k 5 --device cpu --debug-retrieval
```

## Formularios de avaliacao

Gerar amostras e preencher respostas dos dois chatbots:

```bash
python -m chatbot.forms --device cpu
```

Gerar somente as amostras, sem esperar a inferencia do LLM:

```bash
python -m chatbot.forms --device cpu --somente-amostra
```
