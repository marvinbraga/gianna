from dotenv import load_dotenv, find_dotenv
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

from gianna.assistants.models.abstracts import AbstractLLMFactory
from gianna.assistants.models.basics import ModelsEnum, AbstractBasicChain
from gianna.assistants.models.registers import LLMRegister

load_dotenv(find_dotenv())


class OpenAIModelsEnum(ModelsEnum):
    """
    An enumeration class for OpenAI language models.
    """
    gpt35_1106 = 0, "gpt-3.5-turbo-1106"
    gpt4_1106 = 1, "gpt-4-1106-preview"


class OpenAIChain(AbstractBasicChain):
    """
    A basic chain class for OpenAI language models.
    """

    def __init__(self, model: OpenAIModelsEnum, prompt: str, temperature: float = 0.0,
                 verbose: bool = False):
        """
        Initialize the OpenAI chain with the specified model, prompt, temperature, and verbosity.

        Args:
            model (OpenAIModelsEnum): The OpenAI language model to use.
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
        Get the language model chain for the OpenAI model.

        Returns:
            LLMChain: The language model chain.
        """
        chain = self._prompt_template | ChatOpenAI(
            model_name=self._model.model_name,
            temperature=self._temperature,
            verbose=self._verbose,
        ) | StrOutputParser()
        return chain


class OpenAIFactory(AbstractLLMFactory):
    """
    A factory class for creating OpenAI chains.
    """

    def create(self, prompt: str):
        """
        Create an OpenAI chain with the specified prompt.

        Args:
            prompt (str): The prompt for the chain.

        Returns:
            OpenAIChain: The created OpenAI chain.
        """
        return OpenAIChain(self.model_enum, prompt)


def register_openai_chains():
    """
    Register the OpenAI chains with the LLMRegister.
    This method should always be instantiated in the __init__.py file of the package.
    """
    register = LLMRegister()
    register.register_factory(
        model_name="gpt35",
        factory_class=OpenAIFactory,
        model_enum=OpenAIModelsEnum.gpt35_1106
    )
    register.register_factory(
        model_name="gpt4",
        factory_class=OpenAIFactory,
        model_enum=OpenAIModelsEnum.gpt4_1106
    )
