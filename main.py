import csv
from multiprocessing import current_process, Queue
from time import sleep

from sklearn.model_selection import train_test_split

from EventsBroadcaster import broadcast_data
from algorithms.faithfulness_algorithm import FaithfulnessAlgorithm
from indicators.corpus_training import CorpusTrainer
from indicators.embedder import Embedder
from indicators.faithfulness_indicator import FaithfulnessIndicator
from indicators.xdnn_classifier import xDNNClassifier
from indicators.xdnn_trainer import xDNNTrainer
from myutils import pre_process_text

callback_queue = Queue()

def some_func(data):
    print(f"some_func {data}")
    callback_queue.put(None)

if __name__ == "__main__":
    print(f"main pid: [{current_process().pid}]")

    with open("./data/spring_org_xDNN.csv", newline='\n', encoding="utf8") as csvfile:
        documents_reader = csv.reader(csvfile, delimiter=';', quotechar='"')
        data = list(documents_reader)

    cleaned_cases = []
    labels = []

    for row in data:
        cleaned_case = pre_process_text(row[1])
        cleaned_cases.append(cleaned_case)
        labels.append(row[2])

    X_train, X_test, y_train, y_test = train_test_split(cleaned_cases, labels, test_size = 0.33, random_state = 42)

    corpus_trainer = CorpusTrainer()
    xdnn_trainer = xDNNTrainer()
    xdnn_classifier = xDNNClassifier()

    faithfulness_indicator = FaithfulnessIndicator()
    embedder = Embedder()

    corpus_file_path = r'C:\Users\SKIKK\PycharmProjects\XAI\data\all_orgs_documents.csv'
    broadcast_data({"corpus_file_path": corpus_file_path})

    broadcast_data({"cases": X_train,
        "labels": y_train
    })