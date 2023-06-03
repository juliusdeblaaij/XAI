import operator
import threading

from whistle.event import Event


class EventDispatcher(object):
    def __init__(self):
        self._listeners = {}
        self._sorted = {}
        self._lock = threading.Lock()

    def dispatch(self, event_id, event=None):
        if event is None:
            event = Event()

        event.dispatcher = self
        event.name = event_id

        if not event_id in self._listeners:
            return event

        listeners = self.get_listeners(event_id)
        self.run_listeners_async(listeners, event)

        return event

    def get_listeners(self, event_id=None):
        if event_id is not None:
            if not event_id in self._sorted:
                self.sort_listeners(event_id)

            return self._sorted[event_id]

        for event_id in self._listeners:
            if not event_id in self._sorted:
                self.sort_listeners(event_id)

        return self._sorted

    def has_listeners(self, event_id=None):
        return bool(len(self.get_listeners(event_id)))

    def add_listener(self, event_id, listener, priority=0):
        with self._lock:
            if not event_id in self._listeners:
                self._listeners[event_id] = {}
            if not priority in self._listeners[event_id]:
                self._listeners[event_id][priority] = []
            self._listeners[event_id][priority].append(listener)

            if event_id in self._sorted:
                del self._sorted[event_id]

    def listen(self, event_id, priority=0):
        def wrapper(listener):
            self.add_listener(event_id, listener, priority)
            return listener

        return wrapper

    def remove_listener(self, event_id, listener):
        with self._lock:
            if not event_id in self._listeners:
                return

            for priority, listeners in self._listeners[event_id].items():
                if listener in self._listeners[event_id][priority]:
                    self._listeners[event_id][priority].remove(listener)
                    if event_id in self._sorted:
                        del self._sorted[event_id]

    def do_dispatch(self, listeners, event):
        for listener in listeners:
            listener(event)
            if event.propagation_stopped:
                break

    def run_listeners_async(self, listeners, event):
        for listener in listeners:
            thread = threading.Thread(target=listener, args=(event,))
            thread.start()

    def sort_listeners(self, event_id):
        self._sorted[event_id] = []
        if event_id in self._listeners:
            self._sorted[event_id] = [
                listener for listeners in sorted(self._listeners[event_id].items(), key=operator.itemgetter(0))
                for listener in listeners[1]
            ]
