import csv
from multiprocessing import current_process, Queue
from time import sleep

from sklearn.model_selection import train_test_split

from EventsBroadcaster import broadcast_data
from algorithms.faithfulness_algorithm import FaithfulnessAlgorithm
from indicators.corpus_training import CorpusTrainer
from indicators.dataset_splitter import DatasetSplitter
from indicators.embedder import Embedder
from indicators.faithfulness_indicator import FaithfulnessIndicator
from indicators.xdnn_classifier import xDNNClassifier
from indicators.xdnn_trainer import xDNNTrainer
from myutils import pre_process_text

callback_queue = Queue()


if __name__ == "__main__":
    print(f"main pid: [{current_process().pid}]")

    with open("./data/dataset.csv", newline='\n', encoding="utf8") as csvfile:
        documents_reader = csv.reader(csvfile, delimiter=';', quotechar='"')
        next(documents_reader)
        data = list(documents_reader)

    cleaned_cases = []
    labels = []

    for row in data:
        if row[2] is None or row[2] == ' ':
            continue

        label = int(row[2])
        labels.append(label)

        cleaned_case = pre_process_text(row[1])
        cleaned_cases.append(cleaned_case)

    corpus_trainer = CorpusTrainer()
    xdnn_trainer = xDNNTrainer()
    xdnn_classifier = xDNNClassifier()

    faithfulness_indicator = FaithfulnessIndicator()
    embedder = Embedder()
    dataset_splitter = DatasetSplitter()

    corpus_file_path = r'C:\Users\SKIKK\PycharmProjects\XAI\data\all_orgs_documents.csv'
    broadcast_data({"corpus_file_path": corpus_file_path,
                    "embeddings_file_path": r'C:\Users\SKIKK\PycharmProjects\XAI\dataset_embeddings.csv'})

    broadcast_data({
        "labels": labels,
        "cases": cleaned_cases
    })