import os
from textwrap import dedent

import pyperclip

from assistants.commands.abstracts import AbstractCommand
from assistants.commands.speech import SpeechCommand, SpeechType
from assistants.models.factory_method import get_chain_instance
from loguru import logger


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

        completion_prompt = ShellCommandCompletionPrompt(
            self.name,
            self.human_companion_name,
        ).content
        completion_chain_processor = get_chain_instance(
            model_registered_name=os.environ["LLM_DEFAULT_MODEL"],
            prompt=completion_prompt,
        )
        completion_response = completion_chain_processor.process({"command_to_run": response}).output
        self.text_to_speech.execute(
            text=completion_response,
            speech_type=SpeechType(os.environ["TTS_DEFAULT_TYPE"]),
        )
