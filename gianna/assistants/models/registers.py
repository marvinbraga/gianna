class LLMRegister:
    """
    A singleton class that registers and retrieves factories for language models.
    """
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
    def register_factory(cls, model_name, factory_class, model_enum):
        """
        Register a factory for a specific language model.

        Args:
            model_name (str): The name of the language model.
            factory_class (type): The factory class for creating the language model.
            model_enum (Enum): The enumeration representing the language model.
        """
        cls._factories[model_name] = (factory_class, model_enum)

    @classmethod
    def get_factory(cls, model_name):
        """
        Retrieve the factory for a specific language model.

        Args:
            model_name (str): The name of the language model.

        Returns:
            The factory instance for the specified language model.

        Raises:
            ValueError: If no factory is registered for the specified language model.
        """
        if model_name not in cls._factories:
            raise ValueError(f'No factory registered for the model "{model_name}".')
        factory_class, model_enum = cls._factories[model_name]
        return factory_class(model_enum)

    @classmethod
    def list(cls):
        """
        Get a list of all registered language models.

        Returns:
            list: A list of tuples containing the model name and corresponding factory class for each registered language model.

        Example:
            >>> models = LLMRegister.list()
            >>> for model_name, factory_class in models:
            ...     print(f"Model: {model_name}, Factory: {factory_class}")
        """
        return [(model_name, factory_class) for model_name, (factory_class, _) in cls._factories.items()]
