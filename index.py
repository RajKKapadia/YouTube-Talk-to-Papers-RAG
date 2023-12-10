import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.vectorstores.qdrant import Qdrant
import gradio as gr

import config


def create_index(query: str) -> str:
    gr.Info('Started vector generation process.')
    dir_path = os.path.join(
        config.DOWNLOAD_DIR_PATH,
        query.replace(' ', '')
    )
    loader = DirectoryLoader(
        dir_path,
        glob='**/*.pdf',
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3072,
        chunk_overlap=64
    )
    texts = text_splitter.split_documents(documents)
    Qdrant.from_documents(
        texts,
        config.EMBEDDING_FUNCTION,
        collection_name=config.COLLECTION_NAME,
        url=config.QDRANT_URL
    )
    return 'Documents uploaded and index created successfully. You can chat now.'
