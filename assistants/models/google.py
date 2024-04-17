import os

import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from assistants.models.abstracts import AbstractLLMFactory
from assistants.models.basics import ModelsEnum, AbstractBasicChain
from assistants.models.registers import LLMRegister

load_dotenv(find_dotenv())

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY"),
)


class GoogleModelsEnum(ModelsEnum):
    gemini = 0, "gemini-pro"


class GoogleChain(AbstractBasicChain):
    def __init__(self, model: GoogleModelsEnum, prompt: str, temperature: float = 0.0,
                 verbose: bool = False):
        self._verbose = verbose
        self._temperature = temperature
        self._model = model
        super().__init__(
            prompt_template=PromptTemplate.from_template(prompt)
        )

    def _get_chain(self) -> LLMChain:
        chain = self._prompt_template | ChatGoogleGenerativeAI(
            model=self._model.model_name,
            temperature=self._temperature,
            verbose=self._verbose,
        ) | StrOutputParser()
        return chain


class GoogleFactory(AbstractLLMFactory):
    def create(self, prompt: str):
        return GoogleChain(self.model_enum, prompt)


def register_google_chains():
    register = LLMRegister()
    register.register_factory(
        model_name="gemini",
        factory_class=GoogleFactory,
        model_enum=GoogleModelsEnum.gemini
    )
