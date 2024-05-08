class CommandRegister:
    _instance = None
    _factories = {}

    def __new__(cls):
        """
        Ensure only one instance of the class is created.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register_factory(cls, command_name, factory_class):
        for activation_key_word in factory_class.command_class.activation_key_words:
            cls._factories[activation_key_word] = (factory_class, command_name)

    @classmethod
    def get_factory(cls, activation_key_word):
        if activation_key_word not in cls._factories:
            raise ValueError(f'No factory registered for the command "{activation_key_word}".')
        factory_class, command_name = cls._factories[activation_key_word]
        return factory_class(command_name)

    @classmethod
    def list(cls):
        """
        Get a list of all registered commands.

        Returns:
            list: A list of tuples containing the command name and corresponding factory class for each registered command.

        Example:
            >>> commands = CommandRegister.list()
            >>> for command_name, factory_class in commands:
            ...     print(f"Command: {command_name}, Factory: {factory_class}")
        """
        return [
            (command_name, factory_class)
            for factory_class, command_name in cls._factories.items()
        ]
