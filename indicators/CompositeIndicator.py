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

    @property
    @abstractmethod
    def local_data(self) -> dict:
        pass

    def on_event_happened(self, data: dict):
        if not isinstance(data, dict):
            return
        else:
            input_signature = self.input_signature()
            input_data = self.input_data()

            for key in data.keys():
                if key in input_signature.keys():
                    value = data[key]

                    if isinstance(value, type(input_signature[key])):
                        input_data[key] = value
                    else:
                        raise Exception(
                            f'Event raised for {key} has data type: {type(value)}, instead of: {type(input_signature[key])}')

            if set(input_signature.keys()) == set(input_data.keys()):
                self.run_algorithm(**input_data)
        pass

    @abstractmethod
    def run_algorithm(self, data: dict):
        pass
