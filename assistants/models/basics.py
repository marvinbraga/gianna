from abc import ABCMeta, abstractmethod
from enum import Enum

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate


class ModelsEnum(Enum):
    """
    An enumeration class for language models.
    """

    def __init__(self, index: int, model_name: str):
        """
        Initialize the enumeration with an index and model name.

        Args:
            index (int): The index of the language model.
            model_name (str): The name of the language model.
        """
        self.index = index
        self.model_name = model_name


class AbstractBasicChain(metaclass=ABCMeta):
    """
    An abstract base class for basic language model chains.
    """

    def __init__(self, prompt_template: PromptTemplate):
        """
        Initialize the chain with a prompt template.

        Args:
            prompt_template (PromptTemplate): The prompt template for the chain.
        """
        self._prompt_template: PromptTemplate = prompt_template
        self._output: str = ""
        self._chain: LLMChain = None

    @property
    def output(self) -> str:
        """
        Get the output of the chain.

        Returns:
            str: The output of the chain.
        """
        return self._output

    @property
    def chain(self) -> LLMChain:
        """
        Get the chain.

        Returns:
            LLMChain: The chain instance.
        """
        return self._chain

    @abstractmethod
    def _get_chain(self) -> LLMChain:
        """
        Get the language model chain.

        Returns:
            LLMChain: The language model chain.
        """
        pass

    def process(self, input_data, **kwargs):
        """
        Process the input data using the language model chain.

        Args:
            input_data: The input data for the chain.
            **kwargs: Additional keyword arguments for the chain.

        Returns:
            AbstractBasicChain: The chain instance.
        """
        self._chain = self._get_chain()
        response = self._chain.invoke(input_data)
        self._output = response
        return self
