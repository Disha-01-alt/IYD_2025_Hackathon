# IYD 2025 Hackathon
# Ramayana Fact-Checking System
**Team Name:** FactSetu

**Bridging User Belief with Ramayana Wisdom**

[![Built with](https://img.shields.io/badge/Built%20with-Python-blueviolet)](https://www.python.org/)
[![Uses](https://img.shields.io/badge/Uses-SBERT%20%7C%20FAISS%20%7C%20Mistral%20LLM-ff69b4)](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
[![Model Format](https://img.shields.io/badge/Model%20Format-GGUF-orange)](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
## Project Overview & Goal

In an era of pervasive information, verifying statements, especially those related to extensive and ancient texts like the Valmiki Ramayana, can be a significant challenge. Traditional methods are time-consuming and prone to human error or reliance on secondary interpretations.

Our project, the **Ramayana Fact-Checking System**, tackles this problem head-on. We've developed an automated system that leverages cutting-edge Natural Language Processing (NLP) and AI techniques – specifically **Semantic Search** and **Large Language Models (LLMs)** within a Retrieval-Augmented Generation (RAG) framework.

Our core goal is to enable users to take any statement related to the Valmiki Ramayana and automatically verify its truthfulness *strictly* against the authoritative text. The system provides a clear verdict ("TRUE", "FALSE", or "NOT RELEVANT") and cites the specific verses from the source that support or contradict the statement. It's designed to bridge user beliefs with the actual wisdom recorded in the Ramayana text.

## System Pipeline

The Ramayana Fact-Checking System processes input statements in batches from a CSV file. It operates via a robust pipeline that first sets up all necessary components and then iteratively processes each statement.

Here's a high-level view of the process:

```text
+---------------------+
|    Input: CSV       |
|    (input.csv)      |
+----------+----------+
           |
           v
+---------------------+
|       main()        |
+----------+----------+
           |
           v
+---------------------+
|  Validate Input:    |
|  - File Exists      |
|  - 'Statement' Col  |
+----------+----------+
           |
           v
+---------------------+
| setup_models_and_data|
|    (One-Time Setup) |
+----------+----------+
           |
           v
+---------------------+
|  Loop: For Each     |
|    Statement        |
|  in Input CSV       |
+----------+----------+
           |
           v
+---------------------+
|  get_llm_decision() |
|  (for current stmt) |
+----------+----------+
           |
           v
+---------------------+
|  retrieve_top_verses|
|  (Semantic Search)  |
+----------+----------+
           |
           v
+---------------------+
| Construct LLM Prompt|
| (Statement + Verses)|
+----------+----------+
           |
           v
+---------------------+
|  Feed Prompt to LLM |
|     (Mistral GGUF)  |
+----------+----------+
           |
           v
+---------------------+
|  Parse LLM Response |
|  (TRUE/FALSE/UNDET.)|
+----------+----------+
           |
+----------+----------+ <----+
|Map Decision to String|     |
|("TRUE"/"FALSE"/    |     |
|"NOT RELEVANT")     |     |
+----------+----------+     |
           |                |
           +----------------+ (Loop continues for next statement)
           |
(After all statements processed)
           v
+---------------------+
|  Save all results to|
| prediction_output.csv|
+----------+----------+
           |
           v
+---------------------+
|   Clean Up LLM      |
|  (Release Memory)   |
+---------------------+
           |
           v
+---------------------+
|     Process         |
|     Complete        |
+---------------------+
