from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import csv

#doc2vec parameters
vector_size = 300
window_size = 15
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0 #0 = dbow; 1 = dmpv
worker_count = 8 #number of parallel processes

max_epochs = 100

def train(path_to_docs, model_file_name):
    with open(path_to_docs, newline='\n', encoding="utf8") as csvfile:
        documents_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        data = list(documents_reader)

    tagged_data = [TaggedDocument(doc, [i]) for i, doc in enumerate(data)]

    model = Doc2Vec(size=vector_size,
                    window=window_size,
                    min_count=min_count,
                    min_alpha=sampling_threshold,
                    negative=negative_size,
                    dm=dm,
                    workers=worker_count,
                    )

    model.build_vocab(tagged_data)

    print(f"Training Doc2Vec with {len(tagged_data)} words from {len(data)} documents .")

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(sentences = tagged_data,
                    total_examples = model.corpus_count)
        # decrease the learning rate
        model.alpha -= 0.0002

    model.save(model_file_name)
    print(f"Model saved to: {model_file_name}")

def doc2vec(text, model_file_path):

    text = text.split(' ')
    model = Doc2Vec.load(model_file_path)
    return model.infer_vector(text)
