from pydantic import BaseModel


class BashCommandResponse(BaseModel):
    bash_command: str


class ShellCommandModel(BaseModel):
    shell_command: str
