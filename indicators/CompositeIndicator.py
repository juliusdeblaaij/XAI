from abc import ABC, abstractmethod

from EventsBroadcaster import subscribe_to_data


class CompositeIndicator(ABC):

    @abstractmethod
    def __init__(self):
        subscribe_to_data(self.on_event_happened)

    @property
    @abstractmethod
    def input_signature(self) -> dict:
        pass

    @property
    @abstractmethod
    def input_data(self) -> dict:
        pass

    def on_event_happened(self, data: dict):
        if type(data) != dict:
            return
        else:
            for key in data.keys():
                if key in self.input_signature().keys():
                    value = data[key]

                    if type(self.input_signature()[key]) == type(data[key]):
                        self.input_data().update({key: value})
                    else:
                        raise Exception(f'Event raised for {key} has data type: {type(data[key])}, instead of: {type(self.input_signature()[key])}')

            input_signature = self.input_signature()
            if input_signature.keys() == self.input_data().keys():
                input_data = self.input_data().copy()
                self.run_algorithm(input_data)
        pass

    @abstractmethod
    def run_algorithm(self, data: dict):
        pass
