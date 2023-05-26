import asyncio
from algorithms.AlgorithmAdapter import AlgorithmAdapter
from EventsBroadcaster import broadcast


class AdditionAlgorithm(AlgorithmAdapter):

    def __init__(self):
        pass

    async def algorithm_coroutine(self, **kwargs):
        if "A" in kwargs and "B" in kwargs:
            return kwargs["A"] + kwargs["B"]
        pass

    def on_output_received(self, output):
        if type(output) == int:
            broadcast('data_sent', {'addition': output})
        else:
            return