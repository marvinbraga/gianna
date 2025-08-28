from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from gianna.assistants.models.abstracts import AbstractLLMFactory
from gianna.assistants.models.basics import AbstractBasicChain, ModelsEnum
from gianna.assistants.models.registers import LLMRegister

load_dotenv(find_dotenv())


class CohereModelsEnum(ModelsEnum):
    """
    An enumeration class for Cohere language models.
    """

    # Aya Expanse models
    c4ai_aya_expanse_32b = 0, "c4ai-aya-expanse-32b"
    c4ai_aya_expanse_8b = 1, "c4ai-aya-expanse-8b"

    # Aya Vision models
    c4ai_aya_vision_32b = 2, "c4ai-aya-vision-32b"
    c4ai_aya_vision_8b = 3, "c4ai-aya-vision-8b"

    # Command A models
    command_a_03_2025 = 4, "command-a-03-2025"
    command_a_reasoning_08_2025 = 5, "command-a-reasoning-08-2025"
    command_a_vision_07_2025 = 6, "command-a-vision-07-2025"

    # Command Light models
    command_light = 7, "command-light"
    command_light_nightly = 8, "command-light-nightly"
    command_nightly = 9, "command-nightly"

    # Command R models
    command_r = 10, "command-r"
    command_r_plus = 11, "command-r-plus"
    command_r_plus_08_2024 = 12, "command-r-plus-08-2024"

    # Command R7B models
    command_r7b_12_2024 = 13, "command-r7b-12-2024"
    command_r7b_arabic_02_2025 = 14, "command-r7b-arabic-02-2025"

    # Embed English models
    embed_english_light_v3_0 = 15, "embed-english-light-v3.0"
    embed_english_light_v3_0_image = 16, "embed-english-light-v3.0-image"
    embed_english_v3_0_image = 17, "embed-english-v3.0-image"

    # Embed Multilingual models
    embed_multilingual_light_v3_0_image = 18, "embed-multilingual-light-v3.0-image"
    embed_multilingual_v2_0 = 19, "embed-multilingual-v2.0"

    # Rerank models
    rerank_multilingual_v3_0 = 20, "rerank-multilingual-v3.0"


class CohereChain(AbstractBasicChain):
    """
    A basic chain class for Cohere language models.
    """

    def __init__(
        self,
        model: CohereModelsEnum,
        prompt: str,
        temperature: float = 0.0,
        verbose: bool = False,
    ):
        """
        Initialize the Cohere chain with the specified model, prompt, temperature, and verbosity.

        Args:
            model (CohereModelsEnum): The Cohere language model to use.
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
        Get the language model chain for the Cohere model.

        Returns:
            LLMChain: The language model chain.
        """
        chain = (
            self._prompt_template
            | ChatCohere(
                model=self._model.model_name,
                temperature=self._temperature,
                verbose=self._verbose,
            )
            | StrOutputParser()
        )
        return chain


class CohereFactory(AbstractLLMFactory):
    """
    A factory class for creating Cohere chains.
    """

    def create(self, prompt: str):
        """
        Create a Cohere chain with the specified prompt.

        Args:
            prompt (str): The prompt for the chain.

        Returns:
            CohereChain: The created Cohere chain.
        """
        return CohereChain(self.model_enum, prompt)


def register_cohere_chains():
    """
    Register the Cohere chains with the LLMRegister.
    This method should always be instantiated in the __init__.py file of the package.
    """
    register = LLMRegister()
    register.register_factory(
        model_name="command-r",
        factory_class=CohereFactory,
        model_enum=CohereModelsEnum.command_r,
    )
    register.register_factory(
        model_name="command-r-plus",
        factory_class=CohereFactory,
        model_enum=CohereModelsEnum.command_r_plus,
    )
    register.register_factory(
        model_name="command-light",
        factory_class=CohereFactory,
        model_enum=CohereModelsEnum.command_light,
    )
    register.register_factory(
        model_name="aya-expanse-32b",
        factory_class=CohereFactory,
        model_enum=CohereModelsEnum.c4ai_aya_expanse_32b,
    )
    register.register_factory(
        model_name="aya-vision-32b",
        factory_class=CohereFactory,
        model_enum=CohereModelsEnum.c4ai_aya_vision_32b,
    )
