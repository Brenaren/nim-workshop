"""Utility functions for document processing and caching."""

import os
import time
from IPython.display import clear_output
def contains_safetensors(directory, file_to_match):
    """
    Recursively search for any file with the .safetensors extension in the given directory.
    
    Parameters:
    directory (str): The directory path to search.

    Returns:
    bool: True if a .safetensors file is found, False otherwise.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_to_match):
                return True
    return False

def safetensors_downloaded(directory_path):
    '''
    Wait for the safetensors [1,2,3] to be downloaded in the nim cache for mistral 7b
        ├── model-00001-of-00003.safetensors -> ../../blobs/0333a0721c4a8dbec4c999388ccbacb5
        ├── model-00002-of-00003.safetensors -> ../../blobs/56af40423ceef170f2ca98548a39e7cb
        ├── model-00003-of-00003.safetensors -> ../../blobs/1db82681c646faff57b870f9055315f7
    '''
    rv = True
    print(f"Searching {directory_path} for safetensors")
    # Llama3.1 has 4 safetensor files
    # mistral  has 3 safetensor files
    for i in range(1,5):
        result = contains_safetensors(directory_path, f"{i}-of-00004.safetensors")
        print(f"Found {i}-of-00004.safetensors file in {directory_path}: {result}")
        if(result == False) :
            print(f"setting result {result}")
            rv = False
    print(f"returning {rv}")
    return rv


import requests
import os
def check_model_running(url, model_id):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    except requests.exceptions.RequestException:
        print(f"error NIM at url {url} not yet running.  Please wait while loading ...")
        return False

    if response.status_code == 200:
        data = response.json()
        for model in data.get("data", []):
            if model.get("id") == model_id:
                return True
    return False

# For Fine Tuning Notebook

import json
import re
from typing import Tuple

import json
import re
from typing import Tuple, List
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_predictions(file_name: str, postprocessing: bool = True) -> Tuple[List[str], List[str], float]:
    predictions = []
    labels = []

    with open(file_name, 'r') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            pred_value = data["pred"].strip()
            true_value = data["label"].strip()

            if postprocessing:
                pred_value = re.split(r'[ \.\n]', pred_value)[0]
            # print(f"example {i} : appending pred_value {pred_value} and label {true_value}")
            predictions.append(pred_value)
            labels.append(true_value)

    accuracy = sum(1 for p, l in zip(predictions, labels) if p == l) / len(labels) if labels else 0

    return predictions, labels, accuracy

def plot_confusion_matrix(predictions: List[str], labels: List[str], classes: List[str]):
    cm = confusion_matrix(labels, predictions, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

import pickle
import os
from pathlib import Path
from glob import glob
from langchain.docstore.document import Document
from unstructured.partition.pdf import partition_pdf

def load_or_create_corpus_documents(
    pdfs_path: str,
    cache_file: str = 'corpus_documents.pkl'
) -> List[Document]:
    """Load corpus_documents from cache if available, otherwise create and cache them.
    
    Args:
        pdfs_path: Path to PDF files
        cache_file: Path to cache file
        
    Returns:
        List of Document objects
    """
    # Check if cache exists
    if os.path.exists(cache_file):
        print(f"Loading corpus_documents from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # If no cache, create corpus_documents
    print("Now parsing PDF data using OCR. This will take about 6 minutes...")
    corpus_documents = []
    
    for filename in glob(pdfs_path):
        file_chunks = [
            str(chunk) for chunk in partition_pdf(filename=filename, url=None)
        ]
        
        # Print the first 500 parsed characters of each doc
        print(f"Parsed {filename=}:")
        print(f"{chr(10).join(file_chunks):.500}")
        
        doc = Document(
            page_content=str(file_chunks),
            metadata={"source": filename}
        )
        corpus_documents.append(doc)
    
    # Cache the results
    print(f"Caching corpus_documents to: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(corpus_documents, f)
    
    return corpus_documents