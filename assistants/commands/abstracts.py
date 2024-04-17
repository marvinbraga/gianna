from abc import abstractmethod, ABCMeta


class AbstractCommand(metaclass=ABCMeta):
    activation_key_words = ("say", "here")

    @abstractmethod
    def execute(self, **kwargs):
        pass
