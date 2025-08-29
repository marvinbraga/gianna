"""
LangGraph Factory for Gianna AI Assistant

This module provides factory classes for creating LangGraph chains
and integrating them with the existing model registration system.
"""

from enum import Enum
from typing import Any, Dict, Optional, Union

from loguru import logger

from ..assistants.models.abstracts import AbstractLLMFactory
from ..assistants.models.basics import AbstractBasicChain, ModelsEnum
from ..assistants.models.registers import LLMRegister

try:
    from .langgraph_chain import LANGGRAPH_AVAILABLE, LangGraphChain
    from .migration_utils import BackwardCompatibilityWrapper
except ImportError:
    LANGGRAPH_AVAILABLE = False
    LangGraphChain = None
    BackwardCompatibilityWrapper = None


class LangGraphModelsEnum(ModelsEnum):
    """
    Enumeration class for LangGraph-enabled language models.

    This enum mirrors the existing model enums but with LangGraph support.
    """

    # General LangGraph models - these use the model name as the identifier
    langgraph_claude_haiku = 0, "claude-haiku"
    langgraph_claude_sonnet = 1, "claude-sonnet"
    langgraph_claude_opus = 2, "claude-opus"
    langgraph_claude_4 = 3, "claude-4"
    langgraph_gpt35 = 4, "gpt35"
    langgraph_gpt4 = 5, "gpt4"
    langgraph_gpt4o_mini = 6, "gpt4o-mini"
    langgraph_gemini = 7, "gemini"
    langgraph_ollama_llama2 = 8, "ollama_llama2"
    langgraph_ollama_mistral = 9, "ollama_mistral"
    langgraph_command_r = 10, "command-r"
    langgraph_command_r_plus = 11, "command-r-plus"


class LangGraphChainFactory(AbstractLLMFactory):
    """
    Factory class for creating LangGraph chains with backward compatibility.
    """

    def __init__(self, model_enum: LangGraphModelsEnum):
        """
        Initialize the factory with a model enum.

        Args:
            model_enum: LangGraphModelsEnum specifying the model
        """
        self.model_enum = model_enum

    def create(
        self, prompt: str, **kwargs
    ) -> Union[LangGraphChain, AbstractBasicChain]:
        """
        Create a LangGraph chain instance.

        Args:
            prompt: Prompt template string
            **kwargs: Additional arguments for chain creation

        Returns:
            LangGraphChain if available, otherwise falls back to traditional chain
        """
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available, falling back to traditional chain")
            return self._create_fallback_chain(prompt, **kwargs)

        try:
            # Extract the underlying model name (remove 'langgraph_' prefix)
            underlying_model_name = self.model_enum.model_name

            # Get the traditional chain to extract LLM instance
            llm_instance = None
            try:
                register = LLMRegister()
                traditional_factory = register.get_factory(underlying_model_name)
                traditional_chain = traditional_factory.create(prompt)

                # Try to extract LLM instance
                if hasattr(traditional_chain, "_get_chain"):
                    chain = traditional_chain._get_chain()
                    if hasattr(chain, "llm"):
                        llm_instance = chain.llm
                    elif hasattr(chain, "model"):
                        llm_instance = chain.model

            except Exception as e:
                logger.warning(
                    f"Could not extract LLM instance for {underlying_model_name}: {e}"
                )

            # Create LangGraph chain
            langgraph_chain = LangGraphChain(
                model_name=underlying_model_name,
                prompt=prompt,
                llm_instance=llm_instance,
                enable_state_management=kwargs.get("enable_state_management", True),
                **kwargs,
            )

            logger.info(f"Created LangGraph chain for model: {underlying_model_name}")
            return langgraph_chain

        except Exception as e:
            logger.error(f"Error creating LangGraph chain: {e}")
            return self._create_fallback_chain(prompt, **kwargs)

    def _create_fallback_chain(self, prompt: str, **kwargs) -> AbstractBasicChain:
        """
        Create a fallback traditional chain when LangGraph is not available.

        Args:
            prompt: Prompt template string
            **kwargs: Additional arguments

        Returns:
            Traditional chain instance
        """
        try:
            # Get the underlying model name
            underlying_model_name = self.model_enum.model_name

            # Create traditional chain
            register = LLMRegister()
            traditional_factory = register.get_factory(underlying_model_name)
            traditional_chain = traditional_factory.create(prompt)

            logger.info(
                f"Created fallback traditional chain for model: {underlying_model_name}"
            )
            return traditional_chain

        except Exception as e:
            logger.error(f"Error creating fallback chain: {e}")
            raise


class HybridChainFactory(AbstractLLMFactory):
    """
    Factory that creates chains with automatic LangGraph/traditional selection.

    This factory automatically decides whether to use LangGraph or traditional
    chains based on availability and requirements.
    """

    def __init__(self, model_enum: Union[ModelsEnum, LangGraphModelsEnum]):
        """
        Initialize the factory with a model enum.

        Args:
            model_enum: Model enum (either traditional or LangGraph)
        """
        self.model_enum = model_enum

    def create(
        self, prompt: str, **kwargs
    ) -> Union[LangGraphChain, AbstractBasicChain]:
        """
        Create the most appropriate chain type.

        Args:
            prompt: Prompt template string
            **kwargs: Additional arguments for chain creation

        Returns:
            Most appropriate chain instance
        """
        # Check if LangGraph features are requested
        needs_langgraph = (
            kwargs.get("enable_state_management", False)
            or kwargs.get("session_id") is not None
            or kwargs.get("maintain_history", False)
            or kwargs.get("use_langgraph", False)
        )

        # Force traditional if explicitly requested
        if kwargs.get("use_langgraph") is False:
            needs_langgraph = False

        if needs_langgraph and LANGGRAPH_AVAILABLE:
            # Create LangGraph chain
            try:
                # Extract model name
                if hasattr(self.model_enum, "model_name"):
                    model_name = self.model_enum.model_name
                else:
                    model_name = str(self.model_enum)

                # Remove 'langgraph_' prefix if present
                if model_name.startswith("langgraph_"):
                    model_name = model_name[10:]  # Remove 'langgraph_' prefix

                langgraph_factory = LangGraphChainFactory(
                    LangGraphModelsEnum(0, model_name)  # Create temporary enum
                )
                return langgraph_factory.create(prompt, **kwargs)

            except Exception as e:
                logger.warning(f"Failed to create LangGraph chain, falling back: {e}")

        # Create traditional chain
        try:
            # Extract model name
            if hasattr(self.model_enum, "model_name"):
                model_name = self.model_enum.model_name
            else:
                model_name = str(self.model_enum)

            # Remove 'langgraph_' prefix if present
            if model_name.startswith("langgraph_"):
                model_name = model_name[10:]

            register = LLMRegister()
            traditional_factory = register.get_factory(model_name)
            return traditional_factory.create(prompt)

        except Exception as e:
            logger.error(f"Failed to create traditional chain: {e}")
            raise


def register_langgraph_chains():
    """
    Register LangGraph chains with the LLMRegister.

    This function registers LangGraph variants of all major models,
    allowing users to opt into LangGraph features by using the
    'langgraph_' prefixed model names.
    """
    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not available, skipping LangGraph chain registration")
        return

    register = LLMRegister()

    # Register LangGraph variants of popular models
    models_to_register = [
        ("langgraph-claude-haiku", LangGraphModelsEnum.langgraph_claude_haiku),
        ("langgraph-claude-sonnet", LangGraphModelsEnum.langgraph_claude_sonnet),
        ("langgraph-claude-opus", LangGraphModelsEnum.langgraph_claude_opus),
        ("langgraph-claude-4", LangGraphModelsEnum.langgraph_claude_4),
        ("langgraph-gpt35", LangGraphModelsEnum.langgraph_gpt35),
        ("langgraph-gpt4", LangGraphModelsEnum.langgraph_gpt4),
        ("langgraph-gpt4o-mini", LangGraphModelsEnum.langgraph_gpt4o_mini),
        ("langgraph-gemini", LangGraphModelsEnum.langgraph_gemini),
        ("langgraph-ollama-llama2", LangGraphModelsEnum.langgraph_ollama_llama2),
        ("langgraph-ollama-mistral", LangGraphModelsEnum.langgraph_ollama_mistral),
        ("langgraph-command-r", LangGraphModelsEnum.langgraph_command_r),
        ("langgraph-command-r-plus", LangGraphModelsEnum.langgraph_command_r_plus),
    ]

    for model_name, model_enum in models_to_register:
        try:
            register.register_factory(
                model_name=model_name,
                factory_class=LangGraphChainFactory,
                model_enum=model_enum,
            )
            logger.debug(f"Registered LangGraph chain: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to register LangGraph chain {model_name}: {e}")

    logger.info(f"Registered {len(models_to_register)} LangGraph chain variants")


def get_enhanced_chain_instance(
    model_registered_name: str, prompt: str, prefer_langgraph: bool = True, **kwargs
) -> Union[LangGraphChain, AbstractBasicChain]:
    """
    Enhanced factory function that intelligently chooses between chain types.

    Args:
        model_registered_name: Name of the registered model
        prompt: Prompt template string
        prefer_langgraph: Whether to prefer LangGraph when available
        **kwargs: Additional arguments

    Returns:
        Most appropriate chain instance
    """
    # Check if user explicitly requested LangGraph variant
    if model_registered_name.startswith("langgraph-"):
        kwargs["use_langgraph"] = True

    # Set preference
    if prefer_langgraph and LANGGRAPH_AVAILABLE:
        kwargs["use_langgraph"] = kwargs.get("use_langgraph", True)

    # Use the updated factory method
    from ..assistants.models.factory_method import get_chain_instance

    return get_chain_instance(
        model_registered_name=model_registered_name, prompt=prompt, **kwargs
    )


def create_compatible_chain(
    model_name: str, prompt: str, backward_compatible: bool = True, **kwargs
) -> Union[BackwardCompatibilityWrapper, AbstractBasicChain]:
    """
    Create a chain with optional backward compatibility wrapper.

    Args:
        model_name: Name of the model
        prompt: Prompt template
        backward_compatible: Whether to wrap in compatibility layer
        **kwargs: Additional arguments

    Returns:
        Chain instance, optionally wrapped for compatibility
    """
    # Create the chain
    chain = get_enhanced_chain_instance(model_name, prompt, **kwargs)

    # Wrap for backward compatibility if requested
    if backward_compatible and BackwardCompatibilityWrapper:
        return BackwardCompatibilityWrapper(chain)

    return chain


def get_langgraph_capabilities() -> Dict[str, Any]:
    """
    Get information about LangGraph capabilities and availability.

    Returns:
        Dictionary with capability information
    """
    capabilities = {"available": LANGGRAPH_AVAILABLE, "features": {}, "models": []}

    if LANGGRAPH_AVAILABLE:
        capabilities["features"] = {
            "state_management": True,
            "conversation_history": True,
            "session_persistence": True,
            "workflow_orchestration": True,
            "checkpointing": True,
            "error_recovery": True,
        }

        # List available LangGraph models
        register = LLMRegister()
        all_models = register.list()
        langgraph_models = [
            model_name
            for model_name, _ in all_models
            if model_name.startswith("langgraph-")
        ]
        capabilities["models"] = langgraph_models

    return capabilities
