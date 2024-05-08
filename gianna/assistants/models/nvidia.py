import os

from dotenv import load_dotenv, find_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from gianna.assistants.models.abstracts import AbstractLLMFactory
from gianna.assistants.models.basics import ModelsEnum, AbstractBasicChain
from gianna.assistants.models.registers import LLMRegister

load_dotenv(find_dotenv())


class NVIDIAModelsEnum(ModelsEnum):
    """
    An enumeration class for NVIDIA language models.
    """
    mixtral = 0, "mixtral_8x7b"


class NVIDIAChain(AbstractBasicChain):
    """
    A basic chain class for NVIDIA language models.
    """

    def __init__(self, model: NVIDIAModelsEnum, prompt: str, temperature: float = 0.0,
                 verbose: bool = False):
        """
        Initialize the NVIDIA chain with the specified model, prompt, temperature, and verbosity.

        Args:
            model (NVIDIAModelsEnum): The NVIDIA language model to use.
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
        Get the language model chain for the NVIDIA model.

        Returns:
            LLMChain: The language model chain.
        """
        chain = self._prompt_template | ChatNVIDIA(
            model=self._model.model_name,
            temperature=self._temperature,
            streaming=False,
            verbose=self._verbose,
            api_key=os.environ["NVIDIA_API_KEY"]
        ) | StrOutputParser()
        return chain


class NVIDIAFactory(AbstractLLMFactory):
    """
    A factory class for creating NVIDIA chains.
    """

    def create(self, prompt: str):
        """
        Create an NVIDIA chain with the specified prompt.

        Args:
            prompt (str): The prompt for the chain.

        Returns:
            NVIDIAChain: The created NVIDIA chain.
        """
        return NVIDIAChain(self.model_enum, prompt)


def register_nvidia_chains():
    """
    Register the NVIDIA chains with the LLMRegister.
    This method should always be instantiated in the __init__.py file of the package.
    """
    register = LLMRegister()
    register.register_factory(
        model_name="nvidia_mixtral",
        factory_class=NVIDIAFactory,
        model_enum=NVIDIAModelsEnum.mixtral
    )
