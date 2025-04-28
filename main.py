#!/usr/bin/env python3
"""
Gianna - Generative Intelligent Artificial Neural Network Assistant
Main entry point for the application.
"""

import os
from dotenv import load_dotenv
from loguru import logger
from gianna.assistants.models.factory_method import get_chain_instance
from gianna.assistants.commands.factory_method import list_commands


def main():
    """Main entry point for Gianna assistant."""
    # Load environment variables
    load_dotenv()
    
    # Initialize the LLM chain
    model_name = os.getenv("DEFAULT_MODEL", "openai")
    logger.info(f"Initializing Gianna with model: {model_name}")
    
    try:
        chain = get_chain_instance(model_name)
        
        # List available commands
        commands = list_commands()
        logger.info(f"Available commands: {', '.join(commands)}")
        
        # Application loop
        logger.info("Gianna is ready. Type 'exit' to quit.")
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                    
                # Process input with LLM
                response = chain.invoke({"input": user_input})
                print(f"Gianna: {response['output']}")
                
            except KeyboardInterrupt:
                print("\nExiting Gianna...")
                break
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                print("Sorry, I encountered an error. Please try again.")
    except Exception as e:
        logger.error(f"Error initializing Gianna: {e}")
        print(f"Failed to initialize Gianna: {e}")


if __name__ == "__main__":
    main()