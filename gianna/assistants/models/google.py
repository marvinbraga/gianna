from dotenv import find_dotenv, load_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from gianna.assistants.models.abstracts import AbstractLLMFactory
from gianna.assistants.models.basics import AbstractBasicChain, ModelsEnum
from gianna.assistants.models.registers import LLMRegister

load_dotenv(find_dotenv())


class GoogleModelsEnum(ModelsEnum):
    """
    An enumeration class for Google language models.
    """

    # Legacy model
    gemini = 0, "gemini-pro"

    # Gemini 1.5 Flash models
    gemini_15_flash = 1, "gemini-1.5-flash"
    gemini_15_flash_002 = 2, "gemini-1.5-flash-002"
    gemini_15_flash_8b = 3, "gemini-1.5-flash-8b"
    gemini_15_flash_latest = 4, "gemini-1.5-flash-latest"

    # Gemini 1.5 Pro models
    gemini_15_pro = 5, "gemini-1.5-pro"
    gemini_15_pro_002 = 6, "gemini-1.5-pro-002"
    gemini_15_pro_latest = 7, "gemini-1.5-pro-latest"

    # Gemini 2.0 Flash models
    gemini_20_flash = 8, "gemini-2.0-flash"
    gemini_20_flash_001 = 9, "gemini-2.0-flash-001"
    gemini_20_flash_exp = 10, "gemini-2.0-flash-exp"
    gemini_20_flash_lite = 11, "gemini-2.0-flash-lite"
    gemini_20_flash_thinking_exp = 12, "gemini-2.0-flash-thinking-exp"

    # Gemini 2.0 Pro model
    gemini_20_pro_exp = 13, "gemini-2.0-pro-exp"

    # Gemini 2.5 Flash models
    gemini_25_flash = 14, "gemini-2.5-flash"
    gemini_25_flash_lite = 15, "gemini-2.5-flash-lite"
    gemini_25_flash_preview = 16, "gemini-2.5-flash-preview"

    # Gemini 2.5 Pro models
    gemini_25_pro = 17, "gemini-2.5-pro"
    gemini_25_pro_preview = 18, "gemini-2.5-pro-preview"

    # Gemma 3 models
    gemma_3_1b_it = 19, "gemma-3-1b-it"
    gemma_3_4b_it = 20, "gemma-3-4b-it"
    gemma_3_12b_it = 21, "gemma-3-12b-it"
    gemma_3_27b_it = 22, "gemma-3-27b-it"

    # Embedding models
    embedding_001 = 23, "embedding-001"
    gemini_embedding_001 = 24, "gemini-embedding-001"
    text_embedding_004 = 25, "text-embedding-004"

    # AQA model
    aqa = 26, "aqa"


class GoogleChain(AbstractBasicChain):
    """
    A basic chain class for Google language models.
    """

    def __init__(
        self,
        model: GoogleModelsEnum,
        prompt: str,
        temperature: float = 0.0,
        verbose: bool = False,
    ):
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
        super().__init__(prompt_template=PromptTemplate.from_template(prompt))

    def _get_chain(self) -> LLMChain:
        """
        Get the language model chain for the Google model.

        Returns:
            LLMChain: The language model chain.
        """
        chain = (
            self._prompt_template
            | ChatGoogleGenerativeAI(
                model=self._model.model_name,
                temperature=self._temperature,
                verbose=self._verbose,
            )
            | StrOutputParser()
        )
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
        model_enum=GoogleModelsEnum.gemini,
    )
