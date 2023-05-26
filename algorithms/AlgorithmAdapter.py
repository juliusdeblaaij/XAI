from abc import abstractmethod, ABC
import asyncio

class AlgorithmAdapter(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    async def algorithm_coroutine(self, **kwargs):
        pass

    @abstractmethod
    def on_output_received(self, result):
        pass

    def run(self, **kwargs):
        output = asyncio.run(self.algorithm_coroutine(**kwargs))
        self.on_output_received(output)
