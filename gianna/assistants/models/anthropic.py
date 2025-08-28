from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from gianna.assistants.models.abstracts import AbstractLLMFactory
from gianna.assistants.models.basics import AbstractBasicChain, ModelsEnum
from gianna.assistants.models.registers import LLMRegister

load_dotenv(find_dotenv())


class AnthropicModelsEnum(ModelsEnum):
    """
    An enumeration class for Anthropic language models.
    """

    # Claude Haiku models
    claude_3_haiku = 0, "claude-3-haiku-20240307"
    claude_35_haiku = 1, "claude-3-5-haiku-20241022"

    # Claude Sonnet models
    claude_35_sonnet_old = 2, "claude-3-5-sonnet-20240620"
    claude_35_sonnet_new = 3, "claude-3-5-sonnet-20241022"
    claude_37_sonnet = 4, "claude-3-7-sonnet-20250219"
    claude_sonnet_4 = 5, "claude-sonnet-4-20250514"

    # Claude Opus models
    claude_3_opus = 6, "claude-3-opus-20240229"
    claude_opus_4 = 7, "claude-opus-4-20250514"
    claude_opus_41 = 8, "claude-opus-4-1-20250805"


class AnthropicChain(AbstractBasicChain):
    """
    A basic chain class for Anthropic language models.
    """

    def __init__(
        self,
        model: AnthropicModelsEnum,
        prompt: str,
        temperature: float = 0.0,
        verbose: bool = False,
    ):
        """
        Initialize the Anthropic chain with the specified model, prompt, temperature, and verbosity.

        Args:
            model (AnthropicModelsEnum): The Anthropic language model to use.
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
        Get the language model chain for the Anthropic model.

        Returns:
            LLMChain: The language model chain.
        """
        chain = (
            self._prompt_template
            | ChatAnthropic(
                model=self._model.model_name,
                temperature=self._temperature,
                verbose=self._verbose,
            )
            | StrOutputParser()
        )
        return chain


class AnthropicFactory(AbstractLLMFactory):
    """
    A factory class for creating Anthropic chains.
    """

    def create(self, prompt: str):
        """
        Create an Anthropic chain with the specified prompt.

        Args:
            prompt (str): The prompt for the chain.

        Returns:
            AnthropicChain: The created Anthropic chain.
        """
        return AnthropicChain(self.model_enum, prompt)


def register_anthropic_chains():
    """
    Register the Anthropic chains with the LLMRegister.
    This method should always be instantiated in the __init__.py file of the package.
    """
    register = LLMRegister()
    register.register_factory(
        model_name="claude-haiku",
        factory_class=AnthropicFactory,
        model_enum=AnthropicModelsEnum.claude_35_haiku,
    )
    register.register_factory(
        model_name="claude-sonnet",
        factory_class=AnthropicFactory,
        model_enum=AnthropicModelsEnum.claude_35_sonnet_new,
    )
    register.register_factory(
        model_name="claude-opus",
        factory_class=AnthropicFactory,
        model_enum=AnthropicModelsEnum.claude_3_opus,
    )
    register.register_factory(
        model_name="claude-4",
        factory_class=AnthropicFactory,
        model_enum=AnthropicModelsEnum.claude_sonnet_4,
    )
