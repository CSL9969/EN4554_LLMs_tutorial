"""Builds a FAISS-backed retriever over a folder of documents using LangChain.
"""

import argparse
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def get_chunks(docs_dir: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    return chunks

class VectorStore:
    """Wraps a FAISS vector store: building, saving, loading, and querying it."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        """Initializes the VectorStore class.

        Args:
            index_dir (str): Path where the FAISS index is saved/loaded from.
            chunk_size (int): Max characters per document chunk.
            chunk_overlap (int): Overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = HuggingFaceEmbeddings()
        self.vectorstore = None  # type: ignore

    def build_from_docs(self, docs_dir: str) -> "VectorStore":
        """Loads .txt documents, splits, embeds, and builds the FAISS index.

        Args:
            docs_dir (str): Path to a folder containing .txt files to index.

        Returns:
            VectorStore: self, for chaining.
        """
        loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s).")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)


    def retrieve(self, query: str, k: int = 4) -> list:
        """Retrieves the top-k most relevant chunks for a query.

        Args:
            query (str): The search query.
            k (int): Number of chunks to retrieve.

        Returns:
            list: The top-k matching documents, each with .page_content and .metadata.
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Call build_from_docs() first.")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)


if __name__ == "__main__":

    # store = VectorStore()
    # store.build_from_docs("/home/chamindu/EN4554_LLMs_tutorial/docs")

    # query = "What is attention?"
    # results = store.retrieve(query, k=4)

    # print(f"\nTop {len(results)} result(s) for: {query!r}\n")
    # for i, doc in enumerate(results, start=1):
    #     source = doc.metadata.get("source", "unknown")
    #     print(f"--- Result {i} (source: {source}) ---")
    #     print(doc.page_content.strip())
    #     print()

    chunks = get_chunks("/home/chamindu/EN4554_LLMs_tutorial/docs")

    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i} \n\n")
        print(chunk.page_content)
        print("\n\n")
        if i == 10:
            break