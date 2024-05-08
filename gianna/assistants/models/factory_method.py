from gianna.assistants.models.basics import AbstractBasicChain
from gianna.assistants.models.registers import LLMRegister


def get_chain_instance(model_registered_name: str, prompt: str) -> AbstractBasicChain:
    register = LLMRegister()
    return register.get_factory(model_registered_name).create(prompt)
