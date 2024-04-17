import os

import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from assistants.models.abstracts import AbstractLLMFactory
from assistants.models.basics import ModelsEnum, AbstractBasicChain
from assistants.models.registers import LLMRegister

load_dotenv(find_dotenv())

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY"),
)


class GoogleModelsEnum(ModelsEnum):
    """
    An enumeration class for Google language models.
    """
    gemini = 0, "gemini-pro"


class GoogleChain(AbstractBasicChain):
    """
    A basic chain class for Google language models.
    """

    def __init__(self, model: GoogleModelsEnum, prompt: str, temperature: float = 0.0,
                 verbose: bool = False):
        """
        Initialize the Google chain with the specified model, prompt, temperature, and verbosity.

        Args:
            model (GoogleModelsEnum): The Google language model to use.
            prompt (str): The prompt for the chain.
            temperature (float): The temperature for generating responses (default: 0.0).
            verbose (bool): Whether to enable verbose output (default: False).
        """
        self._verbose = verbose
        self._temperature = temperature
        self._model = model
        super().__init__(
            prompt_template=PromptTemplate.from_template(prompt)
        )

    def _get_chain(self) -> LLMChain:
        """
        Get the language model chain for the Google model.

        Returns:
            LLMChain: The language model chain.
        """
        chain = self._prompt_template | ChatGoogleGenerativeAI(
            model=self._model.model_name,
            temperature=self._temperature,
            verbose=self._verbose,
        ) | StrOutputParser()
        return chain


class GoogleFactory(AbstractLLMFactory):
    """
    A factory class for creating Google chains.
    """

    def create(self, prompt: str):
        """
        Create a Google chain with the specified prompt.

        Args:
            prompt (str): The prompt for the chain.

        Returns:
            GoogleChain: The created Google chain.
        """
        return GoogleChain(self.model_enum, prompt)


def register_google_chains():
    """
    Register the Google chains with the LLMRegister.
    This method should always be instantiated in the __init__.py file of the package.
    """
    register = LLMRegister()
    register.register_factory(
        model_name="gemini",
        factory_class=GoogleFactory,
        model_enum=GoogleModelsEnum.gemini
    )
