# Using the `get_command` Method and Registering Classes with `CommandRegister`

This guide provides detailed instructions on how to use the `get_command` method from the `factory_method.py` module to
dynamically instantiate commands and how to register a new command class using `CommandRegister` within the context of
virtual assistants project.

## Using the `get_command` Method

The `get_command` method is designed to facilitate the creation of command instances based on an activation keyword.
This allows for a modular and extensible approach to adding new commands to the virtual assistant.

### Example Usage

To use the `get_command` method, follow the steps below:

1. **Import the Method**: First, import the `get_command` method from the `factory_method` module.

    ```python
    from assistants.commands.factory_method import get_command
    ```

2. **Call the Method**: Next, call the `get_command` method, passing the activation keyword of the desired command. For
   example, to activate the `shell` command, you would do:

    ```python
    cmd = get_command('shell', name='Assistant', human_companion_name='John Doe')
    ```

3. **Execute the Command**: After obtaining the command instance, you can execute it by calling the `execute` method
   with the necessary parameters.

    ```python
    cmd.execute(prompt='What command do you wish to run?')
    ```

## Registering a New Command Class with `CommandRegister`

To add a new command to the system, you must create a command class and register it using `CommandRegister`. This
involves defining a command class, a factory for that class, and registering the factory.

### Steps for Registration

1. **Define the Command Class**: Create a new command class extending `AbstractCommand`. Implement the required methods,
   such as `execute`.

    ```python
    class MyCommand(AbstractCommand):
        activation_key_words = ('mycommand',)

        def execute(self, **kwargs):
            # Command logic
            pass
    ```

2. **Create a Factory for the Command**: Define a factory extending `AbstractCommandFactory` and implement the `create`
   method to return an instance of your command.

    ```python
    class MyCommandFactory(AbstractCommandFactory):
        command_class = MyCommand

        def create(self, **kwargs):
            return self.command_class(**kwargs)
    ```

3. **Register the Command**: Use the `register_factory` method of `CommandRegister` to register your command factory.
   This is typically done at the entry point of your package or module.

    ```python
    def register_my_command():
        CommandRegister.register_factory('mycommand', MyCommandFactory)
    ```

   Make sure to call `register_my_command()` during your application's initialization.

## Listing Available Commands

To list all available registered commands in the system, you can use the `list_commands` method:

```python
from assistants.commands.factory_method import list_commands

available_commands = list_commands()
print(available_commands)
```

This method returns a list of activation keywords for all registered commands, allowing you to see which commands are
available for use.

---

