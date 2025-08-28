import os

from dotenv import find_dotenv, load_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from gianna.assistants.models.abstracts import AbstractLLMFactory
from gianna.assistants.models.basics import AbstractBasicChain, ModelsEnum
from gianna.assistants.models.registers import LLMRegister

load_dotenv(find_dotenv())


class GroqModelsEnum(ModelsEnum):
    """
    An enumeration class for Groq language models.
    """

    # Legacy model
    mixtral = 0, "mixtral-8x7b-32768"

    # Whisper models
    whisper_large_v3 = 1, "whisper-large-v3"
    whisper_large_v3_turbo = 2, "whisper-large-v3-turbo"

    # Gemma models
    gemma2_9b_it = 3, "gemma2-9b-it"

    # Llama 3.1 models
    llama_3_1_8b_instant = 4, "llama-3.1-8b-instant"
    llama_3_3_70b_versatile = 5, "llama-3.3-70b-versatile"

    # Llama 3 models
    llama3_70b_8192 = 6, "llama3-70b-8192"
    llama3_8b_8192 = 7, "llama3-8b-8192"

    # Meta Llama specialized models
    llama_prompt_guard_2_22m = 8, "llama-prompt-guard-2-22m"
    llama_prompt_guard_2_86m = 9, "llama-prompt-guard-2-86m"
    llama_guard_4_12b = 10, "llama-guard-4-12b"
    llama_4_scout_17b_16e_instruct = 11, "llama-4-scout-17b-16e-instruct"
    llama_4_maverick_17b_128e_instruct = 12, "llama-4-maverick-17b-128e-instruct"

    # Compound models
    compound_beta = 13, "compound-beta"
    compound_beta_mini = 14, "compound-beta-mini"

    # DeepSeek models
    deepseek_r1_distill_llama_70b = 15, "deepseek-r1-distill-llama-70b"

    # Command models
    command_r7b_12_2024 = 16, "command-r7b-12-2024"

    # Allam models
    allam_2_7b = 17, "allam-2-7b"


class GroqChain(AbstractBasicChain):
    """
    A basic chain class for Groq language models.
    """

    def __init__(
        self,
        model: GroqModelsEnum,
        prompt: str,
        temperature: float = 0.0,
        verbose: bool = False,
        max_tokens=2048,
    ):
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
        super().__init__(prompt_template=PromptTemplate.from_template(prompt))

    def _get_chain(self) -> LLMChain:
        """
        Get the language model chain for the Groq model.

        Returns:
            LLMChain: The language model chain.
        """
        chain = (
            self._prompt_template
            | ChatGroq(
                model=self._model.model_name,
                temperature=self._temperature,
                verbose=self._verbose,
                max_tokens=self._max_tokens,
                api_key=os.environ["GROQ_API_KEY"],
            )
            | StrOutputParser()
        )
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

    # Legacy model
    register.register_factory(
        model_name="groq_mixtral",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.mixtral,
    )

    # Whisper models
    register.register_factory(
        model_name="groq_whisper_large_v3",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.whisper_large_v3,
    )
    register.register_factory(
        model_name="groq_whisper_large_v3_turbo",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.whisper_large_v3_turbo,
    )

    # Gemma models
    register.register_factory(
        model_name="groq_gemma2_9b_it",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.gemma2_9b_it,
    )

    # Llama 3.1 models
    register.register_factory(
        model_name="groq_llama_3_1_8b_instant",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.llama_3_1_8b_instant,
    )
    register.register_factory(
        model_name="groq_llama_3_3_70b_versatile",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.llama_3_3_70b_versatile,
    )

    # Llama 3 models
    register.register_factory(
        model_name="groq_llama3_70b_8192",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.llama3_70b_8192,
    )
    register.register_factory(
        model_name="groq_llama3_8b_8192",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.llama3_8b_8192,
    )

    # Meta Llama specialized models
    register.register_factory(
        model_name="groq_llama_prompt_guard_2_22m",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.llama_prompt_guard_2_22m,
    )
    register.register_factory(
        model_name="groq_llama_prompt_guard_2_86m",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.llama_prompt_guard_2_86m,
    )
    register.register_factory(
        model_name="groq_llama_guard_4_12b",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.llama_guard_4_12b,
    )
    register.register_factory(
        model_name="groq_llama_4_scout_17b_16e_instruct",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.llama_4_scout_17b_16e_instruct,
    )
    register.register_factory(
        model_name="groq_llama_4_maverick_17b_128e_instruct",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.llama_4_maverick_17b_128e_instruct,
    )

    # Compound models
    register.register_factory(
        model_name="groq_compound_beta",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.compound_beta,
    )
    register.register_factory(
        model_name="groq_compound_beta_mini",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.compound_beta_mini,
    )

    # DeepSeek models
    register.register_factory(
        model_name="groq_deepseek_r1_distill_llama_70b",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.deepseek_r1_distill_llama_70b,
    )

    # Command models
    register.register_factory(
        model_name="groq_command_r7b_12_2024",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.command_r7b_12_2024,
    )

    # Allam models
    register.register_factory(
        model_name="groq_allam_2_7b",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.allam_2_7b,
    )
