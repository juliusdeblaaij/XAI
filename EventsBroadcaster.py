from whistle import EventDispatcher, Event

from DataEvent import DataEvent

dispatcher = EventDispatcher()


def broadcast(event_name: str, data):
    dispatcher.dispatch(event_name, DataEvent(data))


def subscribe(event_name: str, method):
    dispatcher.add_listener(event_name, method)
