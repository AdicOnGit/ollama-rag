import chromadb
from nltk.tokenize import sent_tokenize
import ollama
import os


class DocumentProcessor:
    def __init__(self, host, port, embed_model):
        self.host = host
        self.port = port
        self.embed_model = embed_model

    def create_collection(self, collection_name):
        """Creates a new collection in ChromaDB with the given name."""
        chroma = chromadb.HttpClient(host=self.host, port=self.port)
        return chroma.get_or_create_collection(name=collection_name)

    def read_text(self, text_file):
        """Reads text from a file."""
        try:
            with open(text_file, 'r') as file:
                text = file.read()
            return text
        except FileNotFoundError:
            raise Exception(f"The file {text_file} was not found.")
        except Exception as e:
            raise Exception(
                f"An error occurred while reading {text_file}: {str(e)}")

    def chunk_text_by_sentences(self, text, sentence_per_chunk, overlap, language="english"):
        """Splits text into overlapping chunks of sentences."""
        if not text:
            raise ValueError("The input text cannot be empty.")
        if sentence_per_chunk <= overlap:
            raise ValueError(
                "sentence_per_chunk must be greater than overlap.")

        sentences = sent_tokenize(text, language=language)
        chunks = []
        i = 0
        while i < len(sentences):
            end = min(i + sentence_per_chunk, len(sentences))
            chunks.append(" ".join(sentences[i:end]))
            i += sentence_per_chunk - overlap
        return chunks

    def embed_sentences(self, sentence_chunks, collection):
        """Embeds each chunk of sentences and adds them to a ChromaDB collection."""
        for index, chunk in enumerate(sentence_chunks):
            try:
                embed = ollama.embeddings(model=self.embed_model, prompt=chunk)[
                    "embedding"]
                collection.add(str([index]), embeddings=[
                               embed], documents=[chunk])
            except KeyError:
                raise Exception(
                    "Embedding failed, likely due to an issue with the Ollama service.")
            except Exception as e:
                raise Exception(
                    f"An unexpected error occurred while embedding: {str(e)}")


class QueryProcessor:
    def __init__(self, collection, main_model, embed_model):
        self.collection = collection
        self.main_model = main_model
        self.embed_model = embed_model

    def query_collection(self):
        """Queries the collection based on user input and uses Ollama to generate a response."""
        user_query = input("Enter your query: ")
        try:
            query_embedding = ollama.embeddings(
                model=self.embed_model, prompt=user_query)["embedding"]
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=5)["documents"][0]
            if results:
                docs = "\n\n".join(results)
                model_query = f"{user_query} answer that question using the following text as resource: {docs}"
                stream = ollama.generate(
                    model=self.main_model, prompt=model_query, stream=True)
                for chunk in stream:
                    if chunk["response"]:
                        print(chunk["response"], end="", flush=True)
            else:
                print("No relevant documents found.")
        except KeyError:
            raise Exception(
                "Query failed, possibly due to issues with embeddings.")
        except Exception as e:
            raise Exception(
                f"An unexpected error occurred during the query: {str(e)}")


def main():
    host = "localhost"
    port = 8080
    embed_model = "nomic-embed-text"
    main_model = "llama3"
    text_file = "source.txt"  # Update with the path to the file you want to process

    document_processor = DocumentProcessor(host, port, embed_model)
    text = document_processor.read_text(text_file)
    collection_name = os.path.basename(text_file).split('.')[0]
    collection = document_processor.create_collection(collection_name)

    sentence_chunks = document_processor.chunk_text_by_sentences(text, 7, 0)
    document_processor.embed_sentences(sentence_chunks, collection)

    query_processor = QueryProcessor(collection, main_model, embed_model)
    query_processor.query_collection()


if __name__ == "__main__":
    main()
