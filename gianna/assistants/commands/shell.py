# assistants/commands/shell.py
import os
from textwrap import dedent

import pyperclip
from loguru import logger

from gianna.assistants.commands.abstracts import AbstractCommand
from gianna.assistants.commands.register import CommandRegister
from gianna.assistants.commands.speech import SpeechCommand, SpeechType
from gianna.assistants.models.abstracts import AbstractCommandFactory
from gianna.assistants.models.factory_method import get_chain_instance


class ShellCommandPrompt:
    def __init__(self, name: str, human_companion_name: str, prompt: str):
        self.content = dedent(
            f"""
            You are a highly efficient, code-savvy AI assistant named '{name}'.
            You work with your human companion '{human_companion_name}' to build valuable experience through software.
            Your task is to provide a JSON response with the following format: {{command_to_run}}: '' detailing the 
            shell command for MacOS bash to based on this question: {prompt}. 
            After generating the response, your command will be attached DIRECTLY to your human companions clipboard 
            to be run.
            """
        )


class ShellCommandCompletionPrompt:
    def __init__(self, name: str, human_companion_name: str):
        self.content = dedent(
            f"""
            You are a friendly, ultra helpful, attentive, concise AI assistant named '{name}'.
            You work with your human companion '{human_companion_name}' to build valuable experience through 
            software.
            We both like short, concise, back-and-forth conversations.
            You've just attached the command '{{command_to_run}}' to your human companion's clipboard like 
            they've requested.
            Let your human companion know you've attached it and let them know you're ready for the  next task.
            """
        )


class ShellCommand(AbstractCommand):
    activation_key_words = ("shell", "shell command", "run shell")

    def __init__(self, name: str, human_companion_name: str, text_to_speech: SpeechCommand):
        self.name = name
        self.human_companion_name = human_companion_name
        self.text_to_speech = text_to_speech

    def execute(self, prompt: str, **kwargs):
        shell_command_prompt = ShellCommandPrompt(self.name, self.human_companion_name, prompt).content
        chain_processor = get_chain_instance(
            model_registered_name=os.environ["LLM_DEFAULT_MODEL"],
            prompt=shell_command_prompt,
        )
        response = chain_processor.process({"command_to_run": prompt}).output
        self._clip_copy(response)._talk(self._run(), response)

    def _clip_copy(self, response):
        try:
            pyperclip.copy(response)
        except Exception as e:
            logger.error(dedent(f"ERRO:\n\n {e}"))
            logger.info(dedent(
                f"""
                Response was: 
                {response}
                """
            ))
        return self

    def _run(self):
        completion_prompt = ShellCommandCompletionPrompt(
            self.name,
            self.human_companion_name,
        ).content
        completion_chain_processor = get_chain_instance(
            model_registered_name=os.environ["LLM_DEFAULT_MODEL"],
            prompt=completion_prompt,
        )
        return completion_chain_processor

    def _talk(self, completion_chain_processor, response):
        completion_response = completion_chain_processor.process({"command_to_run": response}).output
        self.text_to_speech.execute(
            text=completion_response,
            speech_type=SpeechType(os.environ["TTS_DEFAULT_TYPE"]),
        )
        return self


class ShellCommandFactory(AbstractCommandFactory):
    command_class = ShellCommand

    def create(self, name: str, human_companion_name: str, text_to_speech: SpeechCommand, **kwargs):
        return self.command_class(
            name=name, human_companion_name=human_companion_name, text_to_speech=text_to_speech
        )


def register_shell_command():
    """
    Register the Shell Command with the CommandRegister.
    This method should always be instantiated in the __init__.py file of the package.
    """
    register = CommandRegister()
    register.register_factory(
        command_name="shell",
        factory_class=ShellCommandFactory,
    )
