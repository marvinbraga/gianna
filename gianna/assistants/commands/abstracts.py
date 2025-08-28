from abc import ABCMeta, abstractmethod


class AbstractCommand(metaclass=ABCMeta):
    activation_key_words = ("say", "here")

    @abstractmethod
    def execute(self, **kwargs):
        pass
