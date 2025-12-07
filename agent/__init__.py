# agent/__init__.py
import abc

class Agent(abc.ABC):
    def reset(self): pass

    @abc.abstractmethod
    def train(self, training: bool = True): ...

    @abc.abstractmethod
    def update(self, replay_buffer, logger, step): ...

    @abc.abstractmethod
    def get_action(self, obs, sample: bool = False): ...
