from os import path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from non_blocking_process import AbstractNonBlockingProcess
import numpy as np

from run_xdnn import RunxDNN
from d2v import doc2vec
from myutils import *

class TfidfVectorizerAlgorithm(AbstractNonBlockingProcess):

    def _do_work(self, mode=None, corpus_file_path=None, cases=None, vectorizer_file_path=None):
        if mode is None:
            raise ValueError("Attempted to run TfidfVectorizer algortihm without specifying 'mode'")

        if mode == "Learning":
            if corpus_file_path is None:
                raise ValueError("Attempted to run TfidfVectorizer algortihm without specifying 'file'")

            vectorizer = TfidfVectorizer(input='filename', lowercase=True)
            vectorizer.fit([corpus_file_path])

            vectorizer_file_path = path.join(r"C:\Users\SKIKK\PycharmProjects\XAI", "vectorizer.pkl")
            joblib.dump(vectorizer, vectorizer_file_path)

            return vectorizer_file_path

        if mode == "Embedding":
            if vectorizer_file_path is None:
                raise ValueError(
                    "Attepted to run TfidfVectorizer embedding while TfidfVectorizer is None. (is it not trained yet?)")
            if cases is None:
                raise ValueError(f"Attempted to run TfidfVectorizer algortihm with mode='{mode}', without specifying 'cases'")

            vectorizer = joblib.load(vectorizer_file_path)
            vectorizer.input = 'content'

            embeddings = []

            for case in cases:
                embedding = vectorizer.transform(cases)
                embeddings.append(embedding.data)
            return embeddings
