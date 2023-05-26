from whistle import EventDispatcher, Event

from DataEvent import DataEvent

dispatcher = EventDispatcher()


def broadcast(event_name: str, data):
    dispatcher.dispatch(event_name, DataEvent(data))


def broadcast_data(data):
    dispatcher.dispatch("data_sent", DataEvent(data))


def subscribe(event_name: str, method):
    dispatcher.add_listener(event_name, method)

def subscribe_to_data(method):
    dispatcher.add_listener('data_sent', method)
