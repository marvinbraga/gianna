from dotenv import load_dotenv, find_dotenv

from assistants.models.factory_method import get_chain_instance
from assistants.models.registers import LLMRegister

load_dotenv(find_dotenv())

# Create a chain instance using the Groq Mixtral model
chain_processor = get_chain_instance(
    model_registered_name="groq_mixtral",
    prompt="Act as a virtual assistant that specifically talks about coffee and nothing else.",
)

# List all registered LLM models
models = sorted([model_name for model_name, _ in LLMRegister.list()])
print(models)
# Output:
# [
#     'gemini',
#     'gpt35', 'gpt4',
#     'groq_mixtral',
#     'nvidia_mixtral',
#     'ollama_llama2', 'ollama_mistral', 'ollama_mixtral'
# ]

# Process user input
response = chain_processor.process({"input": "Talk about football."})
print(response.output)
