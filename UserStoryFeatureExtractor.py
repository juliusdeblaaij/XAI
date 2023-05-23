# Slpit the data into training and test sets
import json
import re

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
from doc2vec import doc2vec
from queue import Queue
import threading


X_user_stories = []
y_user_stories = []
X_non_user_stories = []
y_non_user_stories = []


def process_row(row):
    id = ""
    embedding = []
    story_point = ""

    for j, variable in enumerate(row):
        variable = re.sub('[^a-zA-Z0-9 -]', '', variable)

        if j == 0:
            id = str(variable)
        if j == 1:
            description = " ".join(str(variable).split())
            description.lower()

            embedding = doc2vec(description, 'd2v.model')

        if j == 2:
            story_point = str(variable)

            if story_point == "-1":
                X_non_user_stories.append(embedding)
                y_non_user_stories.append(story_point)
            else:
                X_user_stories.append(embedding)
                y_user_stories.append(story_point)

def worker():
    while True:
        row = queue.get()
        if row is None:
            break
        process_row(row)
        print(f'\nRemaining rows: {round(queue.qsize()/len(rows)*100, 1)}')
        queue.task_done()

# Create a queue to hold the rows
queue = Queue()

# Read the CSV file and put each row into the queue
with open('./data/spring_org_xDNN.csv', newline='\n', encoding="utf8") as csvfile:
    documents_reader = csv.reader(csvfile, delimiter=';', quotechar='"')
    rows = list(documents_reader)

    for i, row in enumerate(rows):
        if i == 0:
            continue
        queue.put(row)

# Create worker threads
num_threads = 12  # Choose the number of threads you want to use
threads = []
for _ in range(num_threads):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

# Wait for all tasks in the queue to be processed
queue.join()

# Stop worker threads
for _ in range(num_threads):
    queue.put(None)
for t in threads:
    t.join()

X_train_user_stories, X_test_user_stories, y_train_user_stories, y_test_user_stories = train_test_split(
    X_user_stories,
    y_user_stories,
    train_size=0.8,
    random_state=42)

X_train_non_user_stories, X_test_non_user_stories, y_train_non_user_stories, y_test_non_user_stories = train_test_split(
    X_non_user_stories,
    y_non_user_stories,
    train_size=0.8,
    random_state=42)

print(f'User stories for training: {len(X_train_user_stories)} user stories for testing: {len(X_test_user_stories)}')
print(
    f'Non user stories for training: {len(X_train_non_user_stories)} user stories for testing: {len(X_test_non_user_stories)}')


def combine_and_shuffle_X_y(X_data_1, y_data_1, X_data_2, y_data_2):
    X_data_y_data = []

    for i, datapoint in enumerate(X_data_1):
        X_data_y_data.append([datapoint, y_data_1[i]])

    for i, datapoint in enumerate(X_data_2):
        X_data_y_data.append([datapoint, y_data_2[i]])

    shuffled_data = shuffle(X_data_y_data, random_state=42)

    X_data = []
    y_data = []

    for datapoint in shuffled_data:
        X_data.append(datapoint[0])
        y_data.append(datapoint[1])

    return X_data, y_data


X_train, y_train = combine_and_shuffle_X_y(X_train_user_stories, y_train_user_stories,
                                           X_train_non_user_stories, y_train_non_user_stories)

X_test, y_test = combine_and_shuffle_X_y(X_test_user_stories, y_test_user_stories,
                                         X_test_non_user_stories, y_test_non_user_stories)

print(f"X_train = {len(X_train)}, y_train = {len(y_train)} | X_test = {len(X_test)} y_test = {len(X_test)}")

data_df_X_train = pd.DataFrame(X_train)
data_df_y_train = pd.DataFrame(y_train)
data_df_X_test = pd.DataFrame(X_test)
data_df_y_test = pd.DataFrame(y_test)

data_df_X_train.to_csv('data_df_X_train.csv',header=False,index=False)
data_df_y_train.to_csv('data_df_y_train.csv',header=False,index=False)
data_df_X_test.to_csv('data_df_X_test.csv',header=False,index=False)
data_df_y_test.to_csv('data_df_y_test.csv',header=False,index=False)