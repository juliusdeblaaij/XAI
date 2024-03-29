import re
import numpy as np
from nltk import sent_tokenize
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

from dataset_cleaner import filter_allowed_words_in_sentences
from doxpy.models.knowledge_extraction.knowledge_graph_extractor import KnowledgeGraphExtractor
import confuse
import random

def pre_process_text(text):
    # text = re.sub('[^a-zA-Z0-9 -]', '', str(text))
    text = " ".join(str(text).split())
    text = text.lower()

    return text


def pad_array(arr):
    # Find the length of the longest element in the array
    max_length = 200

    # Create an empty result array with the same shape as the input array
    result = np.empty_like(arr, dtype=object)

    # Iterate over each row in the array
    # Pad each element in the row to match the length of the longest element
    if len(arr) > max_length:
        arr = arr[:max_length]
    padded_arr = np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=0)

    return padded_arr

config = confuse.Configuration('XAI', __name__)
graph_builder_options = config["graph_builder_options"].get(dict)
graph_cleaning_options = config["graph_cleaning_options"].get(dict)
graph_extraction_options = config["graph_extraction_options"].get(dict)


def extract_knowledge_graph(text: str):
    sentences_in_text = sent_tokenize(text)
    filtered_sentences_in_text = filter_allowed_words_in_sentences(sentences_in_text, keep_stop_words=True)

    knowledge_graph = KnowledgeGraphExtractor(graph_builder_options).set_content_list(
        content_list=filtered_sentences_in_text,
        **graph_cleaning_options).build(
        **graph_extraction_options)

    return knowledge_graph


def sort_with_indices(arr):
    sorted_indices = np.argsort(arr)
    sorted_array = np.sort(arr)
    return sorted_array, sorted_indices.tolist()


def shuffle_with_indices(arr):
    x = list(enumerate(arr))
    random.shuffle(x)
    indices, arr = zip(*x)
    return arr, indices


def label_to_story_point(label: int) -> int:
    story_point = label

    if label == 4:
        story_point = 5
    elif label == 5:
        story_point = 8

    return story_point


def split_text_into_sentences(text: str) -> list:
    if text is None:
        return [""]

    punkt_params = PunktParameters()
    tokenizer = PunktSentenceTokenizer(punkt_params)

    try:
        sentences = tokenizer.tokenize(str(text))
    except:
        return [""]

    return sentences