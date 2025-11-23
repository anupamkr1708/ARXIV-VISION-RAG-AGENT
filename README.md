# Agentic RAG System Using CVPR Research Papers

This repository contains a complete workflow for building a locally hosted Retrieval-Augmented Generation (RAG) system backed by research papers from the CVPR domain.
The project covers the entire pipeline—from data collection to fine-tuning and an interactive agentic interface using Langflow.

The system is designed for developers, researchers, and practitioners who want a self-contained environment to explore computer vision papers, generate insights, and build intelligent applications powered by locally running LLMs.

# Key Features

Automated extraction of CVPR research papers (metadata + PDFs)

End-to-end text processing and dataset preparation

Instruction-format dataset generation for LLaMA fine-tuning

Local RAG pipeline using Ollama, Chroma, and LangChain

Agentic RAG workflow using Langflow

Fully offline, locally hosted architecture

# Workflow Overview

1. Task 1: Data Collection

The data pipeline begins here.

# script_1.py

Scrapes CVPR accepted papers

Collects titles, authors, links

Identifies arXiv references

# script_2.py

Uses arXiv IDs to download PDFs

Extracts full text using PyMuPDF

Saves dataset as CSV with metadata + extracted text

# Instruction_output_script.py

Converts extracted content into instruction → input → output triples

These are later used for supervised fine-tuning

2. Task 2: Model Fine-Tuning

# Task_2_Fine_Tune.ipynb

Loads the instruction dataset from Task 1

Performs LoRA/QLoRA fine-tuning of LLaMA

Generates a domain-adapted CVPR reasoning model

This model is used downstream in the RAG system.

3. Task 3: Local RAG System

This stage prepares the retrieval layer and local chat interface.

# Vector_Database.py

Loads cleaned text from Task 1

Splits into semantic chunks

Generates embeddings using Ollama's embedding model

Stores vectors inside a Chroma persistent database

# models.py

Defines embedding and chat model configurations for Ollama

# chat.py

Builds a LangChain retrieval pipeline

Serves a CLI chat interface that:

Retrieves relevant CVPR content

Synthesizes answers using the local LLM

Displays chunk-level sources

4. Task 4: Agentic RAG using Langflow

# agentic_rag_workflow.json
This file represents a full end-to-end RAG workflow visually designed in Langflow.
It includes:

Document loader

Text splitter

Embedding generator

Chroma retriever

Prompting layer

LLM

Agent node for advanced reasoning

Interactive Chat Input/Output