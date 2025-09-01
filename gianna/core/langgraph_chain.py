"""
LangGraph Chain Implementation for Gianna AI Assistant

This module provides a LangGraph-based replacement for AbstractBasicChain,
maintaining 100% backward compatibility while leveraging StateGraph workflows.
"""

from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Graceful fallback when LangGraph is not available
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    START = None

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from loguru import logger

from ..assistants.models.basics import AbstractBasicChain
from .state import (
    AudioState,
    CommandState,
    ConversationState,
    GiannaState,
    create_initial_state,
)
from .state_manager import StateManager


class LangGraphChain(AbstractBasicChain):
    """
    LangGraph-based replacement for AbstractBasicChain with full backward compatibility.

    This class provides a StateGraph-based workflow while maintaining the same
    interface as AbstractBasicChain, ensuring existing code continues to work.
    """

    def __init__(
        self,
        model_name: str,
        prompt: str,
        llm_instance: Optional[BaseLanguageModel] = None,
        enable_state_management: bool = True,
        **kwargs,
    ):
        """
        Initialize the LangGraphChain with a model and prompt.

        Args:
            model_name: Name of the language model to use
            prompt: Prompt template string
            llm_instance: Optional pre-configured LLM instance
            enable_state_management: Whether to use StateManager for persistence
            **kwargs: Additional arguments for compatibility
        """
        super().__init__(PromptTemplate.from_template(prompt))

        self.model_name = model_name
        self._llm_instance = llm_instance
        self.enable_state_management = enable_state_management
        self._kwargs = kwargs

        # Initialize state management
        if enable_state_management and LANGGRAPH_AVAILABLE:
            self.state_manager = StateManager()
        else:
            self.state_manager = None

        # Build the workflow graph
        self._graph = None
        self._built = False

        # Compatibility fields
        self._last_input = None
        self._last_output = None
        self._session_id = None

        logger.info(f"LangGraphChain initialized for model: {model_name}")

    def _ensure_graph_built(self):
        """Ensure the workflow graph is built when needed."""
        if not self._built and LANGGRAPH_AVAILABLE:
            self._graph = self._build_workflow()
            self._built = True
        elif not LANGGRAPH_AVAILABLE:
            logger.warning(
                "LangGraph not available, falling back to compatibility mode"
            )

    def _build_workflow(self) -> Optional[Any]:
        """
        Build the LangGraph StateGraph workflow.

        Returns:
            Compiled StateGraph or None if LangGraph not available
        """
        if not LANGGRAPH_AVAILABLE:
            return None

        try:
            # Create the state graph
            graph = StateGraph(GiannaState)

            # Add workflow nodes
            graph.add_node("process_input", self._process_input)
            graph.add_node("llm_processing", self._llm_processing)
            graph.add_node("format_output", self._format_output)

            # Define workflow edges
            graph.set_entry_point("process_input")
            graph.add_edge("process_input", "llm_processing")
            graph.add_edge("llm_processing", "format_output")
            graph.add_edge("format_output", END)

            # Compile with checkpointer if state management enabled
            if self.state_manager:
                compiled_graph = graph.compile(
                    checkpointer=self.state_manager.checkpointer
                )
            else:
                compiled_graph = graph.compile()

            logger.info("LangGraph workflow compiled successfully")
            return compiled_graph

        except Exception as e:
            logger.error(f"Error building LangGraph workflow: {e}")
            return None

    def _process_input(self, state: GiannaState) -> GiannaState:
        """
        Process and validate input data.

        Args:
            state: Current GiannaState

        Returns:
            Updated GiannaState
        """
        try:
            # Update session activity
            if self.state_manager and state["conversation"].session_id:
                self.state_manager.update_last_activity(
                    state["conversation"].session_id
                )

            # Add processing metadata
            state["metadata"]["processing_stage"] = "input_processed"
            state["metadata"]["last_input_time"] = str(
                uuid4()
            )  # Simple timestamp alternative

            logger.debug("Input processing completed")
            return state

        except Exception as e:
            logger.error(f"Error in input processing: {e}")
            # Return state with error metadata
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "input_error"
            return state

    def _llm_processing(self, state: GiannaState) -> GiannaState:
        """
        Process the input through the language model.

        Args:
            state: Current GiannaState

        Returns:
            Updated GiannaState with LLM response
        """
        try:
            # Get the latest message from conversation
            messages = state["conversation"].messages
            if not messages:
                # No messages to process
                state["metadata"]["processing_stage"] = "llm_no_input"
                return state

            # Get the user input from the last message
            last_message = messages[-1]
            user_input = last_message.get("content", "")

            # Process through LLM using the existing chain mechanism
            if self._llm_instance:
                # Use provided LLM instance directly
                formatted_prompt = self._prompt_template.format(input=user_input)
                response = self._llm_instance.invoke(formatted_prompt)
            else:
                # Fall back to chain processing
                chain = self._get_chain()
                if hasattr(chain, "invoke"):
                    response = chain.invoke({"input": user_input})
                else:
                    # Handle older chain format
                    response = chain.run(input=user_input)

            # Extract text response
            if isinstance(response, dict):
                response_text = response.get("text", str(response))
            else:
                response_text = str(response)

            # Add assistant response to conversation
            state["conversation"].messages.append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "metadata": {
                        "model": self.model_name,
                        "processing_time": "calculated_elsewhere",  # Placeholder
                    },
                }
            )

            state["metadata"]["processing_stage"] = "llm_completed"
            state["metadata"]["last_response"] = response_text

            logger.debug("LLM processing completed")
            return state

        except Exception as e:
            logger.error(f"Error in LLM processing: {e}")
            # Add error response
            error_msg = f"Error processing request: {str(e)}"
            state["conversation"].messages.append(
                {"role": "assistant", "content": error_msg, "metadata": {"error": True}}
            )
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "llm_error"
            return state

    def _format_output(self, state: GiannaState) -> GiannaState:
        """
        Format the final output for return.

        Args:
            state: Current GiannaState with LLM response

        Returns:
            Final GiannaState
        """
        try:
            # Save state if state management enabled
            if self.state_manager and state["conversation"].session_id:
                self.state_manager.save_state(state["conversation"].session_id, state)

            state["metadata"]["processing_stage"] = "completed"
            state["metadata"]["output_formatted"] = True

            logger.debug("Output formatting completed")
            return state

        except Exception as e:
            logger.error(f"Error in output formatting: {e}")
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "format_error"
            return state

    def _convert_input_to_state(
        self, input_data: Union[str, Dict[str, Any]], session_id: str = None
    ) -> GiannaState:
        """
        Convert various input formats to GiannaState.

        Args:
            input_data: Input in various formats (string, dict, etc.)
            session_id: Optional session identifier

        Returns:
            GiannaState: Properly formatted state
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid4())

        # Load existing state if available
        existing_state = None
        if self.state_manager:
            existing_state = self.state_manager.load_state(session_id)

        if existing_state:
            state = existing_state
        else:
            # Create new state
            state = create_initial_state(session_id)

        # Convert input to message format
        if isinstance(input_data, str):
            # Simple string input
            message = {
                "role": "user",
                "content": input_data,
                "timestamp": str(uuid4()),  # Simple ID for now
            }
        elif isinstance(input_data, dict):
            # Dictionary input - extract content
            content = input_data.get(
                "input", input_data.get("content", str(input_data))
            )
            message = {
                "role": "user",
                "content": content,
                "timestamp": str(uuid4()),
                "metadata": {
                    k: v for k, v in input_data.items() if k not in ["input", "content"]
                },
            }
        else:
            # Fallback for other types
            message = {
                "role": "user",
                "content": str(input_data),
                "timestamp": str(uuid4()),
            }

        # Add message to conversation
        state["conversation"].messages.append(message)

        # Store for compatibility
        self._last_input = input_data
        self._session_id = session_id

        return state

    def _convert_state_to_output(self, state: GiannaState) -> Dict[str, Any]:
        """
        Convert GiannaState to backward-compatible output format.

        Args:
            state: Final GiannaState

        Returns:
            Dictionary in compatible format
        """
        # Extract the last assistant message
        messages = state["conversation"].messages
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]

        if assistant_messages:
            last_response = assistant_messages[-1]["content"]
        else:
            last_response = "No response generated"

        # Store for compatibility
        self._last_output = last_response

        # Return in expected format
        return {
            "output": last_response,
            "text": last_response,  # Alternative field name for compatibility
            # Additional fields for enhanced compatibility
            "messages": messages,
            "session_id": state["conversation"].session_id,
            "metadata": state.get("metadata", {}),
        }

    def _get_chain(self):
        """
        Get the language model chain (compatibility method).

        Returns:
            A chain-like object for LLM processing
        """
        # This method is called by the parent class
        # Return a minimal chain-like object for compatibility
        if self._llm_instance:
            return self._prompt_template | self._llm_instance
        else:
            # Import the actual model chain for this model
            try:
                from ..assistants.models.factory_method import get_chain_instance

                actual_chain = get_chain_instance(
                    self.model_name, str(self._prompt_template.template)
                )
                return actual_chain._get_chain()
            except Exception as e:
                logger.error(f"Error getting chain: {e}")

                # Return a simple fallback
                class FallbackChain:
                    def invoke(self, input_data):
                        return {
                            "text": f"Error: Unable to process with model {self.model_name}"
                        }

                    def run(self, input):
                        return f"Error: Unable to process with model {self.model_name}"

                return FallbackChain()

    def invoke(
        self, input_data: Union[str, Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke the chain with backward compatibility.

        This method maintains the same interface as AbstractBasicChain.invoke()
        while using LangGraph workflows internally.

        Args:
            input_data: Input in various formats
            **kwargs: Additional arguments (session_id, config, etc.)

        Returns:
            Dictionary with output in compatible format
        """
        try:
            session_id = kwargs.get("session_id", "default")

            # Ensure graph is built
            self._ensure_graph_built()

            # If LangGraph not available, fall back to parent implementation
            if not LANGGRAPH_AVAILABLE or not self._graph:
                logger.info("Falling back to traditional chain processing")
                result = super().invoke(input_data, **kwargs)
                # Ensure we return the expected format
                if hasattr(result, "output"):
                    return {"output": result.output}
                return result

            # Convert input to state
            state_input = self._convert_input_to_state(input_data, session_id)

            # Get LangGraph config
            config = (
                self.state_manager.get_config(session_id) if self.state_manager else {}
            )
            config.update(kwargs.get("config", {}))

            # Process through LangGraph
            result_state = self._graph.invoke(state_input, config)

            # Convert back to compatible format
            output = self._convert_state_to_output(result_state)

            # Update parent output for compatibility
            self._output = output["output"]

            logger.debug(f"LangGraph processing completed for session: {session_id}")
            return output

        except Exception as e:
            logger.error(f"Error in LangGraph invoke: {e}")
            # Fallback to parent implementation
            try:
                result = super().invoke(input_data, **kwargs)
                if hasattr(result, "output"):
                    return {"output": result.output}
                return result
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return {"output": f"Error processing request: {str(e)}"}

    def process(self, input_data: Union[str, Dict[str, Any]], **kwargs):
        """
        Process method for backward compatibility.

        Args:
            input_data: Input data
            **kwargs: Additional arguments

        Returns:
            Self for chaining
        """
        self.invoke(input_data, **kwargs)
        return self

    @property
    def output(self) -> str:
        """Get the last output (backward compatibility)."""
        return self._last_output or self._output or ""

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._session_id

    @property
    def has_state_management(self) -> bool:
        """Check if state management is enabled."""
        return self.state_manager is not None

    def get_conversation_history(self, session_id: str = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session ID (uses current session if not provided)

        Returns:
            List of conversation messages
        """
        if not self.state_manager:
            return []

        target_session = session_id or self._session_id
        if not target_session:
            return []

        state = self.state_manager.load_state(target_session)
        if state:
            return state["conversation"].messages
        return []

    def clear_session(self, session_id: str = None) -> bool:
        """
        Clear a session's data.

        Args:
            session_id: Session ID (uses current session if not provided)

        Returns:
            True if session was cleared successfully
        """
        if not self.state_manager:
            return False

        target_session = session_id or self._session_id
        if not target_session:
            return False

        return self.state_manager.delete_session(target_session)


def create_langgraph_chain(
    model_name: str,
    prompt: str,
    llm_instance: Optional[BaseLanguageModel] = None,
    **kwargs,
) -> LangGraphChain:
    """
    Factory function to create a LangGraphChain instance.

    Args:
        model_name: Name of the language model
        prompt: Prompt template string
        llm_instance: Optional pre-configured LLM instance
        **kwargs: Additional arguments

    Returns:
        LangGraphChain instance
    """
    return LangGraphChain(
        model_name=model_name, prompt=prompt, llm_instance=llm_instance, **kwargs
    )
