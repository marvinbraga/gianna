from typing import Optional, Union

from gianna.assistants.models.basics import AbstractBasicChain
from gianna.assistants.models.registers import LLMRegister

try:
    from gianna.core.langgraph_chain import LANGGRAPH_AVAILABLE, LangGraphChain
except ImportError:
    LANGGRAPH_AVAILABLE = False
    LangGraphChain = None


def get_chain_instance(
    model_registered_name: str,
    prompt: str,
    use_langgraph: Optional[bool] = None,
    enable_state_management: bool = True,
    **kwargs,
) -> AbstractBasicChain:
    """
    Create a chain instance with optional LangGraph support.

    Args:
        model_registered_name: Name of the registered model
        prompt: Prompt template string
        use_langgraph: Force LangGraph usage (None=auto-detect, True=force, False=disable)
        enable_state_management: Enable persistent state management (LangGraph only)
        **kwargs: Additional arguments passed to chain constructor

    Returns:
        AbstractBasicChain: Either traditional chain or LangGraphChain
    """
    # Determine whether to use LangGraph
    should_use_langgraph = _should_use_langgraph(use_langgraph, **kwargs)

    if should_use_langgraph and LANGGRAPH_AVAILABLE:
        # Create LangGraphChain with backward compatibility
        register = LLMRegister()
        factory = register.get_factory(model_registered_name)

        # Get the traditional chain to extract the LLM instance if possible
        traditional_chain = factory.create(prompt)
        llm_instance = None

        # Try to extract LLM instance for better performance
        if hasattr(traditional_chain, "_chain") and traditional_chain._chain:
            chain = traditional_chain._get_chain()
            if hasattr(chain, "llm"):
                llm_instance = chain.llm
            elif hasattr(chain, "model"):
                llm_instance = chain.model

        return LangGraphChain(
            model_name=model_registered_name,
            prompt=prompt,
            llm_instance=llm_instance,
            enable_state_management=enable_state_management,
            **kwargs,
        )
    else:
        # Fall back to traditional chain
        register = LLMRegister()
        return register.get_factory(model_registered_name).create(prompt)


def _should_use_langgraph(use_langgraph: Optional[bool], **kwargs) -> bool:
    """
    Determine whether to use LangGraph based on various factors.

    Args:
        use_langgraph: Explicit preference (None=auto, True=force, False=disable)
        **kwargs: Additional arguments that may influence the decision

    Returns:
        bool: Whether to use LangGraph
    """
    # Explicit preference takes precedence
    if use_langgraph is not None:
        return use_langgraph and LANGGRAPH_AVAILABLE

    # Auto-detection logic
    if not LANGGRAPH_AVAILABLE:
        return False

    # Use LangGraph if state management features are requested
    if kwargs.get("enable_state_management", True):
        return True

    # Use LangGraph if session management is requested
    if "session_id" in kwargs:
        return True

    # Use LangGraph if conversation history is requested
    if kwargs.get("maintain_history", False):
        return True

    # Default to traditional chains for now (conservative approach)
    return False


def create_langgraph_chain(
    model_registered_name: str, prompt: str, **kwargs
) -> Union[LangGraphChain, AbstractBasicChain]:
    """
    Explicitly create a LangGraph chain if available.

    Args:
        model_registered_name: Name of the registered model
        prompt: Prompt template string
        **kwargs: Additional arguments

    Returns:
        LangGraphChain if available, otherwise traditional chain
    """
    return get_chain_instance(
        model_registered_name=model_registered_name,
        prompt=prompt,
        use_langgraph=True,
        **kwargs,
    )


def create_traditional_chain(
    model_registered_name: str, prompt: str
) -> AbstractBasicChain:
    """
    Explicitly create a traditional chain (no LangGraph).

    Args:
        model_registered_name: Name of the registered model
        prompt: Prompt template string

    Returns:
        Traditional AbstractBasicChain instance
    """
    return get_chain_instance(
        model_registered_name=model_registered_name, prompt=prompt, use_langgraph=False
    )


def get_available_features() -> dict:
    """
    Get information about available features.

    Returns:
        Dictionary with feature availability information
    """
    return {
        "langgraph_available": LANGGRAPH_AVAILABLE,
        "state_management": LANGGRAPH_AVAILABLE,
        "conversation_history": LANGGRAPH_AVAILABLE,
        "session_persistence": LANGGRAPH_AVAILABLE,
    }
