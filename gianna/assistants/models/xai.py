from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_xai import ChatXAI

from gianna.assistants.models.abstracts import AbstractLLMFactory
from gianna.assistants.models.basics import AbstractBasicChain, ModelsEnum
from gianna.assistants.models.registers import LLMRegister

load_dotenv(find_dotenv())


class XAIModelsEnum(ModelsEnum):
    """
    An enumeration class for xAI language models.
    """

    # Grok 2 models
    grok_2_1212 = 0, "grok-2-1212"
    grok_2_vision_1212 = 1, "grok-2-vision-1212"
    grok_2_image_1212 = 2, "grok-2-image-1212"

    # Grok 3 models
    grok_3 = 3, "grok-3"
    grok_3_fast = 4, "grok-3-fast"
    grok_3_mini = 5, "grok-3-mini"
    grok_3_mini_fast = 6, "grok-3-mini-fast"

    # Grok 4 models
    grok_4_0709 = 7, "grok-4-0709"
    grok_4_0709_eu = 8, "grok-4-0709-eu"

    # Grok code model
    grok_code_fast_1 = 9, "grok-code-fast-1"


class XAIChain(AbstractBasicChain):
    """
    A basic chain class for xAI language models.
    """

    def __init__(
        self,
        model: XAIModelsEnum,
        prompt: str,
        temperature: float = 0.0,
        verbose: bool = False,
    ):
        """
        Initialize the xAI chain with the specified model, prompt, temperature, and verbosity.

        Args:
            model (XAIModelsEnum): The xAI language model to use.
            prompt (str): The prompt for the chain.
            temperature (float): The temperature for generating responses (default: 0.0).
            verbose (bool): Whether to enable verbose output (default: False).
        """
        self._verbose = verbose
        self._temperature = temperature
        self._model = model
        super().__init__(prompt_template=PromptTemplate.from_template(prompt))

    def _get_chain(self) -> LLMChain:
        """
        Get the language model chain for the xAI model.

        Returns:
            LLMChain: The language model chain.
        """
        chain = (
            self._prompt_template
            | ChatXAI(
                model=self._model.model_name,
                temperature=self._temperature,
                verbose=self._verbose,
            )
            | StrOutputParser()
        )
        return chain


class XAIFactory(AbstractLLMFactory):
    """
    A factory class for creating xAI chains.
    """

    def create(self, prompt: str):
        """
        Create an xAI chain with the specified prompt.

        Args:
            prompt (str): The prompt for the chain.

        Returns:
            XAIChain: The created xAI chain.
        """
        return XAIChain(self.model_enum, prompt)


def register_xai_chains():
    """
    Register the xAI chains with the LLMRegister.
    This method should always be instantiated in the __init__.py file of the package.
    """
    register = LLMRegister()
    register.register_factory(
        model_name="grok-2",
        factory_class=XAIFactory,
        model_enum=XAIModelsEnum.grok_2_1212,
    )
    register.register_factory(
        model_name="grok-3", factory_class=XAIFactory, model_enum=XAIModelsEnum.grok_3
    )
    register.register_factory(
        model_name="grok-3-mini",
        factory_class=XAIFactory,
        model_enum=XAIModelsEnum.grok_3_mini,
    )
    register.register_factory(
        model_name="grok-4",
        factory_class=XAIFactory,
        model_enum=XAIModelsEnum.grok_4_0709,
    )
