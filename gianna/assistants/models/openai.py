from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

from gianna.assistants.models.abstracts import AbstractLLMFactory
from gianna.assistants.models.basics import AbstractBasicChain, ModelsEnum
from gianna.assistants.models.registers import LLMRegister

load_dotenv(find_dotenv())


class OpenAIModelsEnum(ModelsEnum):
    """
    An enumeration class for OpenAI language models.
    """

    # ChatGPT models
    chatgpt_4o_latest = 0, "chatgpt-4o-latest"

    # GPT-3.5 models
    gpt35_turbo = 1, "gpt-3.5-turbo"
    gpt35_turbo_0125 = 2, "gpt-3.5-turbo-0125"
    gpt35_1106 = 3, "gpt-3.5-turbo-1106"
    gpt35_turbo_16k = 4, "gpt-3.5-turbo-16k"
    gpt35_turbo_instruct = 5, "gpt-3.5-turbo-instruct"

    # GPT-4 models
    gpt4 = 6, "gpt-4"
    gpt4_0125_preview = 7, "gpt-4-0125-preview"
    gpt4_0613 = 8, "gpt-4-0613"
    gpt4_1106 = 9, "gpt-4-1106-preview"
    gpt4_turbo = 10, "gpt-4-turbo"

    # GPT-4.1 models
    gpt41 = 11, "gpt-4.1"
    gpt41_mini = 12, "gpt-4.1-mini"
    gpt41_nano = 13, "gpt-4.1-nano"
    gpt41_mini_2025_04_14 = 14, "gpt-4.1-mini-2025-04-14"
    gpt41_nano_2025_04_14 = 15, "gpt-4.1-nano-2025-04-14"

    # GPT-4o models
    gpt4o = 16, "gpt-4o"
    gpt4o_2024_05_13 = 17, "gpt-4o-2024-05-13"
    gpt4o_2024_08_06 = 18, "gpt-4o-2024-08-06"
    gpt4o_2024_11_20 = 19, "gpt-4o-2024-11-20"
    gpt_4o_mini = 20, "gpt-4o-mini"
    gpt4o_audio_preview = 21, "gpt-4o-audio-preview"
    gpt4o_realtime_preview = 22, "gpt-4o-realtime-preview"
    gpt4o_search_preview = 23, "gpt-4o-search-preview"

    # GPT-5 models
    gpt5 = 24, "gpt-5"
    gpt5_mini = 25, "gpt-5-mini"
    gpt5_nano = 26, "gpt-5-nano"
    gpt5_chat_latest = 27, "gpt-5-chat-latest"

    # o1 models
    o1 = 28, "o1"
    o1_mini = 29, "o1-mini"
    o1_pro = 30, "o1-pro"

    # DALL-E models
    dalle_2 = 31, "dall-e-2"
    dalle_3 = 32, "dall-e-3"

    # TTS models
    tts_1 = 33, "tts-1"
    tts_1_hd = 34, "tts-1-hd"

    # Whisper models
    whisper_1 = 35, "whisper-1"


class OpenAIChain(AbstractBasicChain):
    """
    A basic chain class for OpenAI language models.
    """

    def __init__(
        self,
        model: OpenAIModelsEnum,
        prompt: str,
        temperature: float = 0.0,
        verbose: bool = False,
    ):
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
        super().__init__(prompt_template=PromptTemplate.from_template(prompt))

    def _get_chain(self) -> LLMChain:
        """
        Get the language model chain for the OpenAI model.

        Returns:
            LLMChain: The language model chain.
        """
        chain = (
            self._prompt_template
            | ChatOpenAI(
                model_name=self._model.model_name,
                temperature=self._temperature,
                verbose=self._verbose,
            )
            | StrOutputParser()
        )
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
        model_enum=OpenAIModelsEnum.gpt35_1106,
    )
    register.register_factory(
        model_name="gpt4",
        factory_class=OpenAIFactory,
        model_enum=OpenAIModelsEnum.gpt4_1106,
    )
    register.register_factory(
        model_name="gpt-4o-mini",
        factory_class=OpenAIFactory,
        model_enum=OpenAIModelsEnum.gpt_4o_mini,
    )
