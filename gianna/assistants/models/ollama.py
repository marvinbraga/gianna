from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from gianna.assistants.models.abstracts import AbstractLLMFactory
from gianna.assistants.models.basics import ModelsEnum, AbstractBasicChain
from gianna.assistants.models.registers import LLMRegister


class OllamaModelsEnum(ModelsEnum):
    """
    An enumeration class for Ollama language models.
    """
    mixtral = 0, "mixtral"
    mistral = 1, "mistral"
    llama2 = 2, "llama2"


class OllamaChain(AbstractBasicChain):
    """
    A basic chain class for Ollama language models.
    """

    def __init__(self, model: OllamaModelsEnum, prompt: str, temperature: float = 0.0,
                 verbose: bool = False):
        """
        Initialize the Ollama chain with the specified model, prompt, temperature, and verbosity.

        Args:
            model (OllamaModelsEnum): The Ollama language model to use.
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
        Get the language model chain for the Ollama model.

        Returns:
            LLMChain: The language model chain.
        """
        chain = self._prompt_template | Ollama(
            model=self._model.model_name,
            temperature=self._temperature,
            verbose=self._verbose,
        ) | StrOutputParser()
        return chain


class OllamaFactory(AbstractLLMFactory):
    """
    A factory class for creating Ollama chains.
    """

    def create(self, prompt: str):
        """
        Create an Ollama chain with the specified prompt.

        Args:
            prompt (str): The prompt for the chain.

        Returns:
            OllamaChain: The created Ollama chain.
        """
        return OllamaChain(self.model_enum, prompt)


def register_ollama_chains():
    """
    Register the Ollama chains with the LLMRegister.
    This method should always be instantiated in the __init__.py file of the package.
    """
    register = LLMRegister()
    register.register_factory(
        model_name="ollama_mistral",
        factory_class=OllamaFactory,
        model_enum=OllamaModelsEnum.mistral
    )
    register.register_factory(
        model_name="ollama_mixtral",
        factory_class=OllamaFactory,
        model_enum=OllamaModelsEnum.mixtral
    )
    register.register_factory(
        model_name="ollama_llama2",
        factory_class=OllamaFactory,
        model_enum=OllamaModelsEnum.llama2
    )
