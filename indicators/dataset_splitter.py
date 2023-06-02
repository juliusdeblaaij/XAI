from sklearn.model_selection import train_test_split

from DataEvent import DataEvent
from dataset_cleaner import filter_allowed_words
from indicators.CompositeIndicator import CompositeIndicator
from EventsBroadcaster import broadcast_data
from myutils import shuffle_with_indices


class DatasetSplitter(CompositeIndicator):

    def __init__(self):
        super().__init__()

    _input_data = {}

    def input_data(self) -> dict:
        return self._input_data

    _local_data = {}

    def local_data(self) -> dict:
        return self._local_data

    def input_signature(self) -> dict:
        return {"features": [], "labels": [], "cases": []}

    def run_algorithm(self, **kwargs):
        self.input_data().clear()

        cases = kwargs.get("cases")
        features = kwargs.get("features")
        labels = kwargs.get("labels")

        user_story_original_cases = []
        non_user_story_original_cases = []

        user_story_cases = []
        non_user_story_cases = []

        user_story_features = []
        non_user_story_features = []

        user_story_labels = []
        non_user_story_labels = []

        for i, label in enumerate(labels):
            if label == 0:
                non_user_story_original_cases.append(cases[i])
                non_user_story_cases.append(filter_allowed_words(cases[i]))
                non_user_story_features.append(features[i])
                non_user_story_labels.append(label)
            else:
                user_story_original_cases.append(cases[i])
                user_story_cases.append(filter_allowed_words(cases[i]))
                user_story_features.append(features[i])
                user_story_labels.append(label)

        test_size = 0.2

        non_user_story_training_features, non_user_story_testing_features, non_user_story_training_labels, non_user_story_testing_labels, non_user_story_training_cases, non_user_story_testing_cases, non_user_story_training_original_cases, non_user_story_testing_original_cases = \
            train_test_split(non_user_story_features, non_user_story_labels, non_user_story_cases, non_user_story_original_cases, test_size=test_size, random_state=42,
                             shuffle=True)

        user_story_training_features, user_story_testing_features, user_story_training_labels, user_story_testing_labels, user_story_training_cases, user_story_testing_cases, user_story_training_original_cases, user_story_testing_original_cases = \
            train_test_split(user_story_features, user_story_labels, user_story_cases, user_story_original_cases, test_size=test_size, random_state=42,
                             shuffle=True)

        training_features = []
        training_features.extend(non_user_story_training_features)
        training_features.extend(user_story_training_features)

        testing_features = []
        testing_features.extend(non_user_story_testing_features)
        testing_features.extend(user_story_testing_features)

        training_labels = []
        training_labels.extend(non_user_story_training_labels)
        training_labels.extend(user_story_training_labels)

        testing_labels = []
        testing_labels.extend(non_user_story_testing_labels)
        testing_labels.extend(user_story_testing_labels)

        training_cases = []
        training_cases.extend(non_user_story_training_cases)
        training_cases.extend(user_story_training_cases)

        testing_cases = []
        testing_cases.extend(non_user_story_testing_cases)
        testing_cases.extend(user_story_testing_cases)

        testing_original_cases = []
        testing_original_cases.extend(non_user_story_testing_original_cases)
        testing_original_cases.extend(user_story_testing_original_cases)

        shuffled_training_features, shuffled_training_indices = shuffle_with_indices(training_features)
        shuffled_testing_features, shuffled_testing_indices = shuffle_with_indices(testing_features)

        shuffled_training_labels = [training_labels[i] for i in shuffled_training_indices]
        shuffled_testing_labels = [testing_labels[i] for i in shuffled_testing_indices]

        shuffled_training_cases = [training_cases[i] for i in shuffled_training_indices]
        shuffled_testing_cases = [testing_cases[i] for i in shuffled_testing_indices]

        shuffled_testing_original_cases = [testing_original_cases[i] for i in shuffled_testing_indices]

        broadcast_data({"training_features":list(shuffled_training_features)})
        broadcast_data({"testing_features": list(shuffled_testing_features)})

        broadcast_data({"training_labels": list(shuffled_training_labels)})
        broadcast_data({"testing_labels": list(shuffled_testing_labels)})

        broadcast_data({"training_cases": list(shuffled_training_cases)})
        broadcast_data({"testing_cases": list(shuffled_testing_cases)})

        broadcast_data({"testing_original_cases": list(shuffled_testing_original_cases)})

    def on_event_happened(self, data_event: DataEvent):
        super().on_event_happened(data_event.value())
