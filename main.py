import csv
from multiprocessing import current_process

from EventsBroadcaster import broadcast_data
from dataset_cleaner import filter_allowed_words
from indicators.adherence_to_knowledge_limits_indicator import AdherenceToKnowledgeLimitsIndicator
from indicators.audience_acceptability_indicator import AudienceAcceptabilityIndicator
from indicators.audience_aspects_extractor import AudienceAspectsExtractor
from indicators.corpus_training import CorpusTrainer
from indicators.dataset_output import DatasetOutput
from indicators.dataset_splitter import DatasetSplitter
from indicators.embedder import Embedder
from indicators.explanation_accuracy_indicator import ExplanationAccuracyIndicator
from indicators.explanations_generator import ExplanationsGenerator
from indicators.faithfulness_indicator import FaithfulnessIndicator
from indicators.meaningfulness_indicator import MeaningfulnessIndicator
from indicators.xdnn_classifier import xDNNClassifier
from indicators.xdnn_trainer import xDNNTrainer
from myutils import pre_process_text


if __name__ == "__main__":
    print(f"main pid: [{current_process().pid}]")

    with open("./data/dataset.csv", newline='\n', encoding="utf8") as csvfile:
        documents_reader = csv.reader(csvfile, delimiter=';', quotechar='"')
        next(documents_reader)
        data = list(documents_reader)

    original_cases = []
    cleaned_cases = []
    labels = []

    for i, row in enumerate(data):
        if row[2] is None or row[2] == ' ':
            continue

        if i > 100:
            break

        label = int(row[2]) - 1
        labels.append(label)

        case = row[1]

        cleaned_case = " ".join(str(case).split())
        cleaned_case.replace(';', '')
        cleaned_case.replace('\n', '')
        cleaned_cases.append(cleaned_case)

    corpus_trainer = CorpusTrainer()
    xdnn_trainer = xDNNTrainer()
    xdnn_classifier = xDNNClassifier()

    faithfulness_indicator = FaithfulnessIndicator()
    embedder = Embedder()
    dataset_splitter = DatasetSplitter()
    explanation_generator = ExplanationsGenerator()
    audience_knowledge_graphs_extractor = AudienceAspectsExtractor()
    audience_acceptability_indicator = AudienceAcceptabilityIndicator()
    explanation_accuracy_indicator = ExplanationAccuracyIndicator()
    meaningfulness_indicator = MeaningfulnessIndicator()
    adherence_to_knowledge_limits_indicator = AdherenceToKnowledgeLimitsIndicator()
    dataset_output = DatasetOutput()

    corpus_file_path = r'C:\Users\SKIKK\PycharmProjects\XAI\data\all_orgs_documents.csv'
    broadcast_data({"corpus_file_path": corpus_file_path,
                    "embeddings_file_path": r'C:\Users\SKIKK\PycharmProjects\XAI\dataset_embeddings.csv'})

    broadcast_data({
        "labels": labels,
        "cases": cleaned_cases,
    })

    outsider_questions = """What is a user story?
    What is an example of a user story?
    What was the prediction for this user story?"""

    practitioner_questions = """What was the prediction for this user story? 
    What features of the user story contributed to the story points prediction? 
    How can a user story be refused? 
    Which specific process was used to automatcally determine the amount of story points for a user story?
    """

    expert_questions = """What is a user story?
    What is an example of a user story?
    What was the prediction for this user story?
    What features of the user story contributed to the story points prediction?
    How can a user story be refused?
    Which specific process was used to automatcally determine the amount of story points for a user story?
    What are known issues of the technology used to automatically determine the amount of story points for a user story?"""

    broadcast_data({"outsider_questions": outsider_questions,
                    "practitioner_questions": practitioner_questions,
                    "expert_questions": expert_questions})