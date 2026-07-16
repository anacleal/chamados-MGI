<h1 align="center"> An NLP-Based Pipeline for Exploring, Topic Modeling, and Semantic Recommendation of Support Tickets </h1>

### 💻 _Project Description_

This repository contains the code related to the scientific article on analyzing public service support tickets from the Ministry of Management and Innovation in Public Services (MGI). We propose an integrated Natural Language Processing (NLP) and Large Language Model (LLM) pipeline to sanitize sensitive data, automatically cluster tickets into topics, extract diagnostic summaries (Title, Pattern, and Impact) using local LLMs, and recommend resolutions via semantic search.

### 📁 _Running the project_

The Python version recommended is **3.10+**.

Activate the virtual environment:
* On Windows (PowerShell):
  ```powershell
  .venv\Scripts\activate
  ```
* On Linux/macOS:
  ```bash
  source .venv/bin/activate
  ```

Install all required dependencies listed in the `requirements.txt` file with:
```bash
pip install -r requirements.txt
```

Ensure you have [Ollama](https://ollama.com/) installed and running locally for the LLM processing steps.

Run the interactive dashboard using:
```bash
python dashboard/app.py
```

### ⚙️ _NLP Modules_

This project includes four core NLP modules:
1. **Preprocessing & Anonymization**  
   Cleanses the text by removing emails, URLs, punctuation, and stopwords, and automatically removes personal names based on the IBGE dataset to comply with data privacy regulations (LGPD).
2. **Topic Modeling (BERTopic + K-Means)**  
   Groups ticket descriptions into semantically coherent topics using BERTopic combined with the K-Means clustering algorithm, utilizing the Elbow Method to determine the optimal number of clusters.
3. **Cognitive Summarization (LLM)**  
   Integrates local LLMs (via Ollama) to analyze each topic cluster and automatically generate a representative Title, identify the Dominant Pattern of error or doubt, and assess the Operational Impact.
4. **Semantic Search & Recommendation**  
   Encodes tickets into dense vector embeddings using SentenceTransformers and indexes them with FAISS. The module calculates cosine similarity to recommend similar historical tickets and their corresponding solutions.
