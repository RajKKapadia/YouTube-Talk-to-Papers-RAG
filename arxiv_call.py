import time
from urllib.error import HTTPError
import os

import arxiv
from tqdm import tqdm
import gradio as gr

import config

def download_and_save_papers(query: str, max_results: float) -> str:
    gr.Info('Downloading process started...')
    search = arxiv.Search(
        query=query,
        max_results=int(max_results),
    )
    search_results = arxiv.Client().results(search)
    dir_path = os.path.join(
        config.DOWNLOAD_DIR_PATH,
        query.replace(' ', '')
    )
    os.makedirs(dir_path, exist_ok=True)
    for result in tqdm(search_results):
        while True:
            try:
                result.download_pdf(dirpath=dir_path)
                break
            except FileNotFoundError:
                break
            except HTTPError:
                break
            except ConnectionResetError as e:
                time.sleep(5)
    return 'Paper downloaded successfully. You can talk to the agent.'
               