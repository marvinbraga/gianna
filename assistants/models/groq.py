import os

from dotenv import load_dotenv, find_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from assistants.models.abstracts import AbstractLLMFactory
from assistants.models.basics import ModelsEnum, AbstractBasicChain
from assistants.models.registers import LLMRegister

load_dotenv(find_dotenv())


class GroqModelsEnum(ModelsEnum):
    mixtral = 0, "mixtral-8x7b-32768"


class GroqChain(AbstractBasicChain):
    def __init__(self, model: GroqModelsEnum, prompt: str, temperature: float = 0.0,
                 verbose: bool = False, max_tokens=2048):
        self._max_tokens = max_tokens
        self._verbose = verbose
        self._temperature = temperature
        self._model = model
        super().__init__(
            prompt_template=PromptTemplate.from_template(prompt)
        )

    def _get_chain(self) -> LLMChain:
        chain = self._prompt_template | ChatGroq(
            model=self._model.model_name,
            temperature=self._temperature,
            verbose=self._verbose,
            max_tokens=self._max_tokens,
            api_key=os.environ["GROQ_API_KEY"]
        ) | StrOutputParser()
        return chain


class GroqFactory(AbstractLLMFactory):
    def create(self, prompt: str):
        return GroqChain(self.model_enum, prompt)


def register_groq_chains():
    register = LLMRegister()
    register.register_factory(
        model_name="groq_mixtral",
        factory_class=GroqFactory,
        model_enum=GroqModelsEnum.mixtral
    )
