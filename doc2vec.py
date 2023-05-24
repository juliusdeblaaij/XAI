from gensim.models import Doc2Vec
from nltk.tokenize import word_tokenize
import csv
import numpy

from myutils import pre_process_text

#doc2vec parameters
vector_size = 300
window_size = 5
min_count = 20
sampling_threshold = 1e-6
negative_size = 5
train_epoch = 10
dm = 0 #0 = dbow; 1 = dmpv
worker_count = 8 #number of parallel processes

max_epochs = 1000


def clean_corpus(corpus_file_name):
    data = []
    with open(corpus_file_name, newline='\n', encoding="utf8") as csvfile:
        documents_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        data = list(documents_reader)

    cleaned_data = []

    for row in data:
        cleaned_string = pre_process_text([row[0]])
        cleaned_data.append([cleaned_string])

    with open("cleaned_" + corpus_file_name, newline='\n', encoding="utf8", mode="w") as csvfile:
        documents_writer = csv.writer(csvfile, delimiter=',', quotechar='"')

        documents_writer.writerows(cleaned_data)


from gensim.models import word2vec, Doc2Vec

from gensim.models.callbacks import CallbackAny2Vec

from gensim.test.utils import get_tmpfile

import time

class callback(CallbackAny2Vec):

   """Callback to print loss after each epoch."""

   def __init__(self):

       self.path_prefix = "d2v"

       self.epoch = 0

   def on_epoch_start(self, model):

       print(f"Start:{self.epoch} epoch at {time.ctime()}")

   def on_epoch_end(self, model):

       print(f"Done:{self.epoch} epoch at {time.ctime()}")

       self.epoch += 1

       if self.epoch % 10 is not 0:
           return


       output_path = get_tmpfile(f"{self.path_prefix}_epoch{self.epoch}.model")

       model.save(output_path)

       print("Saved:", output_path)

def train(path_to_corpus, model_file_name):

    model = Doc2Vec(vector_size=vector_size,
                    window=window_size,
                    min_count=min_count,
                    min_alpha=sampling_threshold,
                    negative=negative_size,
                    dm=dm,
                    workers=worker_count,
                    epochs=max_epochs,
                    )

    model.build_vocab(corpus_file=path_to_corpus)

    print(f"Number of Documents:{model.corpus_count}")

    print(f"Total Number of Words:{model.corpus_total_words}")

    model.train(corpus_file=path_to_corpus ,
                total_examples = model.corpus_count,
                total_words=model.corpus_total_words,
                epochs=model.epochs,
                callbacks=[callback()])


    model.save(model_file_name)
    print(f"Model saved to: {model_file_name}")


def doc2vec(text, model_file_path):
    text = pre_process_text(text)

    words = text.split(' ')
    model = Doc2Vec.load(model_file_path)
    return model.infer_vector(doc_words=words)

if __name__ == "__main__":
    train('cleaned_all_orgs_documents.csv', 'd2v_23k_dbow.model')
    # clean_corpus('all_orgs_documents.csv')