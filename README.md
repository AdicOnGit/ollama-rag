# Document Processing with ChromaDB and Ollama

This Python script provides a framework for processing text documents, embedding sentences, and querying embedded sentences using ChromaDB and Ollama APIs. It allows the manipulation of text to be stored in a searchable format and queries it through neural embeddings.

## Features

- **Document Processing**: Read and process documents to create manageable text chunks.
- **Text Embedding**: Convert text chunks into embeddings and store them in ChromaDB.
- **Text Querying**: Query the ChromaDB with natural language inputs and receive contextual outputs using Ollama's models.

## Requirements

- Python 3.x
- Libraries: `chromadb`, `nltk`, `ollama`, `os`

## Installation

Before running the script, you need to install the necessary Python libraries. You can install these libraries using pip:

```bash
pip install chromadb nltk ollama
```

## Usage

1. **Set Up Parameters**: Configure the host, port, and model names in the `main()` function. Ensure ChromaDB and Ollama services are running and accessible.

2. **Prepare Your Document**: Place the text file you want to process in the script's directory or update the `text_file` path in the `main()` function.

3. **Run the Script**: Execute the script using Python. It will read the text file, create a collection in ChromaDB, chunk the text, embed the chunks, and store these embeddings. Then, it allows querying the collection:

```bash
python main.py
```

4. **Querying**: Follow the prompt to input your query for searching the document collections. The script will output relevant text based on your input.

## Configuration

- **Host and Port**: Set the host and port for ChromaDB where your database is hosted.
- **Embedding Model**: The embedding model to use, specified as `embed_model` in the script.
- **Main Model**: The main Ollama model for generating responses, specified as `main_model`.

## Files

- **source.txt**: Example text file to be processed. Replace or rename as needed based on your specific document.
