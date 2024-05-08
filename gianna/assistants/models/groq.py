import os

from dotenv import load_dotenv, find_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from gianna.assistants.models.abstracts import AbstractLLMFactory
from gianna.assistants.models.basics import ModelsEnum, AbstractBasicChain
from gianna.assistants.models.registers import LLMRegister

load_dotenv(find_dotenv())


class GroqModelsEnum(ModelsEnum):
    """
    An enumeration class for Groq language models.
    """
    mixtral = 0, "mixtral-8x7b-32768"


class GroqChain(AbstractBasicChain):
    """
    A basic chain class for Groq language models.
    """

    def __init__(self, model: GroqModelsEnum, prompt: str, temperature: float = 0.0,
                 verbose: bool = False, max_tokens=2048):
        """
        Initialize the Groq chain with the specified model, prompt, temperature, verbosity, and max tokens.

        Args:
            model (GroqModelsEnum): The Groq language model to use.
            prompt (str): The prompt for the chain.
            temperature (float): The temperature for generating responses (default: 0.0).
            verbose (bool): Whether to enable verbose output (default: False).
            max_tokens (int): The maximum number of tokens to generate (default: 2048).
        """
        self._max_tokens = max_tokens
        self._verbose = verbose
        self._temperature = temperature
        self._model = model
        super().__init__(
            prompt_template=PromptTemplate.from_template(prompt)
        )

    def _get_chain(self) -> LLMChain:
        """
        Get the language model chain for the Groq model.

        Returns:
            LLMChain: The language model chain.
        """
        chain = self._prompt_template | ChatGroq(
            model=self._model.model_name,
            temperature=self._temperature,
            verbose=self._verbose,
            max_tokens=self._max_tokens,
            api_key=os.environ["GROQ_API_KEY"]
        ) | StrOutputParser()
        return chain


class GroqFactory(AbstractLLMFactory):
    """
    A factory class for creating Groq chains.
    """

    def create(self, prompt: str):
        """
        Create a Groq chain with the specified prompt.

        Args:
            prompt (str): The prompt for the chain.

        Returns:
            GroqChain: The created Groq chain.
        """
        return GroqChain(self.model_enum, prompt)


def register_groq_chains():
    """
    Register the Groq chains with the LLMRegister.
    This method should always be instantiated in the __init__.py file of the package.
    """
    register = LLMRegister()
    register.register_factory(
        model_name="groq_mixtral",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.mixtral
    )
