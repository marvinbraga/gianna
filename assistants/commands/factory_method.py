from assistants.commands.register import CommandRegister


def get_command(activation_key_word: str, **kwargs):
    command_factory = CommandRegister().get_factory(activation_key_word)
    cmd = command_factory.create(**kwargs)
    return cmd


def list_commands():
    commands = CommandRegister.list()
    return [activation_key for _, activation_key in commands]
