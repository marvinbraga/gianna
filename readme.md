# Gianna: Generative Intelligent Artificial Neural Network Assistant

Gianna is a voice assistant that utilizes CrewAI and Langchain to perform complex tasks. It provides an easy way to create chain instances for working with a simple prompt, a string parser, and various LLM models such as OpenAI, Google, NVIDIA, Groq, and Ollama.

## Installation

To get started with Gianna, you'll need to have Poetry installed. Poetry is a dependency management and packaging tool for Python projects. If you don't have Poetry installed, you can follow the installation instructions from the [official Poetry documentation](https://python-poetry.org/docs/#installation).

Once you have Poetry installed, follow these steps to set up Gianna:

1. Clone the Gianna repository:
   ```
   git clone https://github.com/your-username/gianna.git
   ```

2. Navigate to the project directory:
   ```
   cd gianna
   ```

3. Install the project dependencies using Poetry:
   ```
   poetry install
   ```

4. Activate the virtual environment created by Poetry:
   ```
   poetry shell
   ```

5. Run Gianna:
   ```
   python main.py
   ```

## Usage

Gianna provides a simple way to create chain instances for different registered LLM models. You can use the `get_chain_instance` function to create a chain instance with a simple prompt and the desired registered model name.

Here are some examples of how to use Gianna:

```python
from dotenv import load_dotenv, find_dotenv
from assistants.models.registers import LLMRegister
from assistants.models.factory_method import get_chain_instance

# Load environment variables
load_dotenv(find_dotenv())

# List all registered LLM models
models = sorted([model_name for model_name, _ in LLMRegister.list()])
print(models)
# Output: ['gemini', 'gpt35', 'gpt4', 'groq_mixtral', 'nvidia_mixtral', 'ollama_llama2', 'ollama_mistral', 'ollama_mixtral']

# Create a chain instance using the Groq Mixtral model
chain_processor = get_chain_instance(
    model_registered_name="groq_mixtral",
    prompt="Act as a virtual assistant that specifically talks about coffee and nothing else.",
)

# Process user input
response = chain_processor.process({"input": "Talk about football."})
print(response.output)
# Output: I apologize, but as a virtual assistant focused solely on coffee, I am not able to discuss topics like football. My knowledge is limited to coffee-related subjects such as coffee beans, brewing methods, roasting techniques, and coffee culture. If you have any questions or would like to have a conversation about coffee, I would be more than happy to assist you. However, for information on other topics like sports, I recommend seeking out resources or assistants that specialize in those areas.
```

In this example, we first load the environment variables using `load_dotenv`. Then, we list all the registered LLM models using `LLMRegister.list()`.

Next, we create a chain instance using the Groq Mixtral model with a specific prompt using the `get_chain_instance` function.

Finally, we process user input by calling the `process` method on the chain instance, passing the user's input as a dictionary with the key `"input"`. The response from the model is printed to the console.

## Contributing

We welcome contributions from the community to make Gianna even better! If you'd like to contribute to the project, please follow these steps:

1. Fork the Gianna repository on GitHub.

2. Create a new branch for your feature or bug fix:
   ```
   git checkout -b feature/your-feature-name
   ```

3. Make your changes and commit them with descriptive commit messages.

4. Push your changes to your forked repository:
   ```
   git push origin feature/your-feature-name
   ```

5. Open a pull request on the main Gianna repository, describing your changes and why they should be merged.

We appreciate your contributions and will review your pull request as soon as possible. Together, let's make Gianna an even more powerful and versatile voice assistant!

If you have any questions or need further assistance, please don't hesitate to reach out to our friendly community on [GitHub Discussions](https://github.com/marvinbraga/gianna/discussions) or [Discord](https://discord.gg/xXaqSaYS).

Happy coding!