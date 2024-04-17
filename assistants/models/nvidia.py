import os

from dotenv import load_dotenv, find_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from assistants.models.abstracts import AbstractLLMFactory
from assistants.models.basics import ModelsEnum, AbstractBasicChain
from assistants.models.registers import LLMRegister

load_dotenv(find_dotenv())


class NVIDIAModelsEnum(ModelsEnum):
    mixtral = 0, "mixtral_8x7b"


class NVIDIAChain(AbstractBasicChain):
    def __init__(self, model: NVIDIAModelsEnum, prompt: str, temperature: float = 0.0,
                 verbose: bool = False):
        self._verbose = verbose
        self._temperature = temperature
        self._model = model
        super().__init__(
            prompt_template=PromptTemplate.from_template(prompt)
        )

    def _get_chain(self) -> LLMChain:
        chain = self._prompt_template | ChatNVIDIA(
            model=self._model.model_name,
            temperature=self._temperature,
            streaming=False,
            verbose=self._verbose,
            api_key=os.environ["NVIDIA_API_KEY"]
        ) | StrOutputParser()
        return chain


class NVIDIAFactory(AbstractLLMFactory):
    def create(self, prompt: str):
        return NVIDIAChain(self.model_enum, prompt)


def register_nvidia_chains():
    register = LLMRegister()
    register.register_factory(
        model_name="nvidia_mixtral",
        factory_class=NVIDIAFactory,
        model_enum=NVIDIAModelsEnum.mixtral
    )
