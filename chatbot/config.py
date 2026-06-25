from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CHATBOT_DATA_DIR = DATA_DIR / "chatbot"
ARTIFACTS_DIR = ROOT_DIR / "chatbot" / "artifacts"
MODELS_DIR = ROOT_DIR / "chatbot" / "models"

TOPICOS_CSV = CHATBOT_DATA_DIR / "df_topicos.csv"
CHAMADOS_CSV = CHATBOT_DATA_DIR / "df_chamados.csv"

TOPIC_INDEX_PATH = ARTIFACTS_DIR / "indice_topicos.faiss"
TICKET_INDEX_PATH = ARTIFACTS_DIR / "indice_chamados.faiss"
TOPIC_EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings_topicos.npy"
TICKET_EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings_chamados.npy"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "modelos_utilizados.json"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

K_POR_SISTEMA = {
    "SIASS": 6,
    "SIAPE": 8,
    "SIGEPE": 5,
    "SOUGOV": 5,
    "TOTAIS": 10,
}

