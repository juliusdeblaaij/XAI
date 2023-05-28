import inspect

from whistle import EventDispatcher, Event

from DataEvent import DataEvent

dispatcher = EventDispatcher()
debug = True


def broadcast(event_name: str, data):
    if debug is True:
        if type(data) == type({}):
            print(f"Function {str(inspect.stack()[1].function)} broadcasted '{event_name}' ({data.keys()})")
        else:
            print(f"Function {str(inspect.stack()[1].function)} broadcasted '{event_name}' ({type(data)})")

    dispatcher.dispatch(event_name, DataEvent(data))


def broadcast_data(data):
    if debug is True:
        if type(data) == type({}):
            print(f"Function {str(inspect.stack()[1].function)} broadcasted 'data' ({data.keys()})")
        else:
            print(f"Function {str(inspect.stack()[1].function)} broadcasted 'data' ({type(data)})")

    dispatcher.dispatch("data_sent", DataEvent(data))


def subscribe(event_name: str, method):
    dispatcher.add_listener(event_name, method)

def subscribe_to_data(method):
    dispatcher.add_listener('data_sent', method)
