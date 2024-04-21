from abc import ABCMeta, abstractmethod


class AbstractLLMFactory(metaclass=ABCMeta):
    """
    An abstract base class for language model factories.
    """

    def __init__(self, model_enum):
        """
        Initialize the factory with the specified language model enumeration.

        Args:
            model_enum (Enum): The enumeration representing the language model.
        """
        self.model_enum = model_enum

    @abstractmethod
    def create(self, **kwargs):
        """
        Create an instance of the language model.

        Args:
            **kwargs: Additional keyword arguments for creating the language model.

        Returns:
            The created language model instance.
        """
        pass


class AbstractCommandFactory(metaclass=ABCMeta):
    """
    An abstract base class for command factories.
    """
    command_class = None

    def __init__(self, command_name: str):
        self.command_name = command_name

    @abstractmethod
    def create(self, **kwargs):
        """
        Create an instance of the command.

        Args:
            **kwargs: Additional keyword arguments for creating the command.

        Returns:
            The created command instance.
        """
        pass
