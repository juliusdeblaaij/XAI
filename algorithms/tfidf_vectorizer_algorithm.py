import csv
from os import path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from myutils import pad_array
from non_blocking_process import AbstractNonBlockingProcess

class TfidfVectorizerAlgorithm(AbstractNonBlockingProcess):

    def _do_work(self, mode=None, corpus_file_path=None, embeddings_file_path=None, cases=None, vectorizer_file_path=None):
        if mode is None:
            raise ValueError("Attempted to run TfidfVectorizer algortihm without specifying 'mode'")

        if mode == "Learning":
            if corpus_file_path is None:
                raise ValueError("Attempted to run TfidfVectorizer algortihm without specifying 'file'")

            vectorizer = TfidfVectorizer(input='filename', lowercase=True, min_df=0.1)
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
            if embeddings_file_path is None:
                raise ValueError(f"Attempted to run TfidfVectorizer algortihm with mode='{mode}', without specifying 'embeddings_file_path'")

            # TODO: give error when vectorizer isnt found
            vectorizer = joblib.load(vectorizer_file_path)
            vectorizer.input = 'content'

            embeddings = []

            if path.isfile(embeddings_file_path):
                with open(embeddings_file_path, 'r', newline='\n') as csvfile:
                    embeddings_reader = csv.reader(csvfile, delimiter=',',
                                                   quotechar='"',
                                                   quoting=csv.QUOTE_NONNUMERIC)

                    embeddings = list(embeddings_reader)

            if not path.isfile(embeddings_file_path) or embeddings == []:
                for i, case in enumerate(cases):
                    embedding = vectorizer.transform([case])
                    padded_embedding = pad_array(embedding.data)
                    embeddings.append(padded_embedding)
                    print(f'Embedded {i}/{len(cases)} strings.')

                with open(embeddings_file_path, 'w+', newline='\n') as csvfile:
                    embeddings_writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='"',
                                            quoting=csv.QUOTE_NONNUMERIC)

                    embeddings_writer.writerows(embeddings)

            return embeddings
