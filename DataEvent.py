from whistle import Event


class DataEvent(Event):
    _data = {}

    def __init__(self, data):
        self._data = data

    def value(self) -> dict:
        return self._data
