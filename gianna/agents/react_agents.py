"""
Core ReAct Agent Implementations for Gianna AI Assistant

This module implements the specialized ReAct agents using langgraph.prebuilt.create_react_agent.
Each agent is designed for specific domain tasks while integrating with the Gianna
state management system and tool ecosystem.
"""

from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from loguru import logger

try:
    from langgraph.prebuilt import create_react_agent

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available - ReAct agents will use fallback mode")

from ..core.state import GiannaState
from ..tools import (
    AudioProcessorTool,
    FileSystemTool,
    MemoryTool,
    ShellExecutorTool,
    STTTool,
    TTSTool,
)
from .base_agent import AgentConfig, BaseReActAgent


class GiannaReActAgent(BaseReActAgent):
    """
    Base implementation for Gianna ReAct agents using LangGraph.

    This class provides the common ReAct agent setup and lifecycle
    management for all specialized Gianna agents.
    """

    def __init__(
        self,
        name: str,
        llm: BaseLanguageModel,
        tools: List,
        config: Optional[AgentConfig] = None,
    ):
        """
        Initialize the Gianna ReAct agent.

        Args:
            name: Agent name/identifier
            llm: Language model instance
            tools: List of tools available to the agent
            config: Optional agent configuration
        """
        super().__init__(name, llm, tools, config)

        # Initialize the ReAct agent if LangGraph is available
        if LANGGRAPH_AVAILABLE:
            try:
                self.agent = create_react_agent(
                    llm, tools, state_modifier=self._prepare_agent_state
                )
                logger.info(f"LangGraph ReAct agent initialized for {name}")
            except Exception as e:
                logger.error(f"Failed to initialize LangGraph agent for {name}: {e}")
                self.agent = None
        else:
            self.agent = None
            logger.warning(f"LangGraph not available - {name} will use fallback mode")

    @property
    def system_message(self) -> str:
        """Base system message - should be overridden by subclasses."""
        return """Você é um assistente AI inteligente usando o padrão ReAct (Reasoning and Acting).

        Siga este processo:
        1. THOUGHT: Analise a solicitação e planeje sua abordagem
        2. ACTION: Execute a ação necessária usando as ferramentas disponíveis
        3. OBSERVATION: Analise o resultado da ação
        4. Repita até completar a tarefa

        Sempre explique seu raciocínio e forneça resultados úteis ao usuário."""

    def _prepare_agent_state(self, state: GiannaState) -> Dict[str, Any]:
        """
        Prepare the GiannaState for ReAct agent execution.

        Args:
            state: Current Gianna system state

        Returns:
            Dict[str, Any]: State prepared for ReAct agent
        """
        # Extract relevant information from GiannaState
        prepared_state = {
            "conversation_history": state["conversation"].messages,
            "session_id": state["conversation"].session_id,
            "user_preferences": state["conversation"].user_preferences,
            "audio_mode": state["audio"].current_mode,
            "language": state["audio"].language,
            "command_history": state["commands"].execution_history,
            "metadata": state["metadata"],
        }

        return prepared_state

    def _process_agent_output(self, output: Any, state: GiannaState) -> Dict[str, Any]:
        """
        Process ReAct agent output back into Gianna format.

        Args:
            output: Output from the ReAct agent
            state: Current Gianna system state

        Returns:
            Dict[str, Any]: Processed output in Gianna format
        """
        # Extract messages from LangGraph output
        messages = []
        if isinstance(output, dict) and "messages" in output:
            messages = output["messages"]

        # Process the final message
        final_content = ""
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                final_content = last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                final_content = last_message["content"]

        return {
            "agent_name": self.name,
            "content": final_content,
            "messages": messages,
            "raw_output": output,
            "success": True,
        }

    def _execute_agent(
        self, input_data: Any, agent_state: Dict[str, Any], **kwargs
    ) -> Any:
        """Execute the ReAct agent with fallback support."""
        if self.agent is None:
            # Fallback mode - direct LLM invocation
            return self._execute_fallback(input_data, agent_state, **kwargs)

        # Use LangGraph ReAct agent
        try:
            graph_input = {"messages": [{"role": "user", "content": str(input_data)}]}

            result = self.agent.invoke(graph_input)
            self.current_iteration += 1
            return result

        except Exception as e:
            logger.error(f"ReAct agent execution failed for {self.name}: {e}")
            # Fall back to direct LLM
            return self._execute_fallback(input_data, agent_state, **kwargs)

    def _execute_fallback(
        self, input_data: Any, agent_state: Dict[str, Any], **kwargs
    ) -> Any:
        """
        Fallback execution when LangGraph is not available.

        Args:
            input_data: Input for processing
            agent_state: Prepared state
            **kwargs: Additional parameters

        Returns:
            Any: Fallback response
        """
        logger.info(f"Using fallback mode for {self.name}")

        # Create a simple prompt with system message
        prompt = f"{self.system_message}\n\nUser: {input_data}\n\nAssistant:"

        try:
            # Direct LLM invocation
            response = self.llm.invoke(prompt)

            return {
                "messages": [
                    {"role": "user", "content": str(input_data)},
                    {"role": "assistant", "content": str(response)},
                ],
                "fallback_used": True,
            }

        except Exception as e:
            logger.error(f"Fallback execution failed for {self.name}: {e}")
            raise


class CommandAgent(GiannaReActAgent):
    """
    Specialized ReAct agent for shell commands and system operations.

    This agent excels at understanding command requests, validating them
    for safety, executing them appropriately, and providing clear feedback
    about the results.
    """

    def __init__(self, llm: BaseLanguageModel, config: Optional[AgentConfig] = None):
        """
        Initialize the Command Agent.

        Args:
            llm: Language model instance
            config: Optional agent configuration
        """
        # Initialize tools for command operations
        tools = [
            ShellExecutorTool(),
            FileSystemTool(),
        ]

        # Set default config if not provided
        if config is None:
            config = AgentConfig(
                name="command_agent",
                description="Specialized agent for shell commands and system operations",
                max_iterations=15,
                safety_checks=True,
                validate_inputs=True,
            )

        super().__init__("command_agent", llm, tools, config)

    @property
    def system_message(self) -> str:
        """System message specific to command execution."""
        return """Você é um especialista em comandos shell e operações de sistema usando o padrão ReAct.

        Sua função é:
        1. Analisar solicitações de comando do usuário com cuidado
        2. Validar comandos para segurança antes da execução
        3. Gerar comandos shell apropriados e seguros
        4. Executar comandos usando as ferramentas disponíveis
        5. Interpretar resultados e fornecer feedback claro

        PROCESSO ReAct:
        - THOUGHT: Analise a solicitação e planeje o comando necessário
        - ACTION: Use as ferramentas para executar o comando de forma segura
        - OBSERVATION: Interprete o resultado e determine próximos passos

        REGRAS DE SEGURANÇA:
        - SEMPRE explicar o que cada comando faz antes de executar
        - Verificar se o comando é seguro e não destrutivo
        - Solicitar confirmação para operações potencialmente perigosas
        - Fornecer alternativas mais seguras quando apropriado
        - Nunca executar comandos que possam comprometer o sistema

        Use as ferramentas ShellExecutorTool e FileSystemTool para executar operações."""

    def _prepare_agent_state(self, state: GiannaState) -> Dict[str, Any]:
        """Prepare state with command-specific context."""
        base_state = super()._prepare_agent_state(state)

        # Add command-specific context
        base_state.update(
            {
                "command_mode": "interactive",
                "safety_level": "high",
                "execution_context": {
                    "pending_operations": state["commands"].pending_operations,
                    "recent_commands": (
                        state["commands"].execution_history[-5:]
                        if state["commands"].execution_history
                        else []
                    ),
                    "current_directory": state["metadata"].get(
                        "current_directory", "unknown"
                    ),
                },
            }
        )

        return base_state


class AudioAgent(GiannaReActAgent):
    """
    Specialized ReAct agent for audio processing operations.

    This agent handles text-to-speech, speech-to-text, audio playback,
    recording, and other audio-related tasks with intelligent reasoning
    about audio context and user preferences.
    """

    def __init__(self, llm: BaseLanguageModel, config: Optional[AgentConfig] = None):
        """
        Initialize the Audio Agent.

        Args:
            llm: Language model instance
            config: Optional agent configuration
        """
        # Initialize tools for audio operations
        tools = [
            AudioProcessorTool(),
            TTSTool(),
            STTTool(),
        ]

        # Set default config if not provided
        if config is None:
            config = AgentConfig(
                name="audio_agent",
                description="Specialized agent for audio processing and voice operations",
                max_iterations=10,
                conversation_memory=True,
            )

        super().__init__("audio_agent", llm, tools, config)

    @property
    def system_message(self) -> str:
        """System message specific to audio processing."""
        return """Você é um especialista em processamento de áudio usando o padrão ReAct.

        Sua função é:
        1. Processar entrada de voz do usuário com precisão
        2. Converter texto em fala natural e expressiva
        3. Gerenciar gravação e reprodução de áudio
        4. Ajustar configurações de áudio conforme preferências
        5. Fornecer feedback sobre operações de áudio

        PROCESSO ReAct:
        - THOUGHT: Analise a solicitação de áudio e determine a melhor abordagem
        - ACTION: Use as ferramentas de áudio para processar a solicitação
        - OBSERVATION: Verifique a qualidade e adequação do resultado

        RECURSOS DISPONÍVEIS:
        - Síntese de voz (TTS) em múltiplas vozes e idiomas
        - Reconhecimento de voz (STT) com alta precisão
        - Processamento e análise de áudio
        - Configuração dinâmica de parâmetros de voz

        CONSIDERAÇÕES:
        - Adapte a voz e entonação ao contexto da conversa
        - Mantenha consistência na personalidade vocal
        - Considere preferências do usuário para velocidade e tom
        - Forneça feedback sobre qualidade do áudio

        Use as ferramentas AudioProcessorTool, TTSTool e STTTool para operações."""

    def _prepare_agent_state(self, state: GiannaState) -> Dict[str, Any]:
        """Prepare state with audio-specific context."""
        base_state = super()._prepare_agent_state(state)

        # Add audio-specific context
        base_state.update(
            {
                "audio_context": {
                    "current_mode": state["audio"].current_mode,
                    "voice_settings": state["audio"].voice_settings,
                    "speech_type": state["audio"].speech_type,
                    "language": state["audio"].language,
                    "preferred_voice": state["conversation"].user_preferences.get(
                        "voice", "default"
                    ),
                    "volume_level": state["conversation"].user_preferences.get(
                        "volume", 0.8
                    ),
                    "speech_rate": state["conversation"].user_preferences.get(
                        "speech_rate", 1.0
                    ),
                }
            }
        )

        return base_state


class ConversationAgent(GiannaReActAgent):
    """
    Specialized ReAct agent for natural dialogue and conversation management.

    This agent focuses on maintaining engaging, contextually appropriate
    conversations while coordinating with other agents when specialized
    tasks are needed.
    """

    def __init__(self, llm: BaseLanguageModel, config: Optional[AgentConfig] = None):
        """
        Initialize the Conversation Agent.

        Args:
            llm: Language model instance
            config: Optional agent configuration
        """
        # Conversation agent uses minimal tools - focuses on dialogue
        tools = [
            MemoryTool(),
        ]

        # Set default config if not provided
        if config is None:
            config = AgentConfig(
                name="conversation_agent",
                description="Specialized agent for natural dialogue and conversation",
                max_iterations=8,
                conversation_memory=True,
                max_context_length=15000,
            )

        super().__init__("conversation_agent", llm, tools, config)

    @property
    def system_message(self) -> str:
        """System message specific to conversation management."""
        return """Você é Gianna, uma assistente AI amigável e inteligente usando o padrão ReAct.

        Sua função principal é:
        1. Manter conversas naturais e envolventes com o usuário
        2. Entender o contexto e intenções por trás das mensagens
        3. Fornecer respostas úteis e contextualmente apropriadas
        4. Identificar quando especialização técnica é necessária
        5. Coordenar com outros agentes quando apropriado

        PROCESSO ReAct:
        - THOUGHT: Analise a mensagem do usuário e o contexto da conversa
        - ACTION: Use ferramentas de memória para manter contexto relevante
        - OBSERVATION: Avalie se a resposta está adequada ao contexto

        PERSONALIDADE:
        - Amigável e prestativa, mas profissional
        - Curiosa e interessada no bem-estar do usuário
        - Clara na comunicação, evitando jargões desnecessários
        - Empática e compreensiva com as necessidades do usuário

        HABILIDADES DE CONVERSA:
        - Manter contexto de conversas anteriores
        - Fazer perguntas de esclarecimento quando necessário
        - Sugerir soluções proativas
        - Reconhecer quando encaminhar para agentes especializados

        QUANDO ENCAMINHAR:
        - Comandos shell → CommandAgent
        - Operações de áudio → AudioAgent
        - Gerenciamento de memória complexo → MemoryAgent

        Use a ferramenta MemoryTool para gerenciar contexto conversacional."""

    def _prepare_agent_state(self, state: GiannaState) -> Dict[str, Any]:
        """Prepare state with conversation-specific context."""
        base_state = super()._prepare_agent_state(state)

        # Add conversation-specific context
        base_state.update(
            {
                "conversation_context": {
                    "session_duration": len(state["conversation"].messages),
                    "user_name": state["conversation"].user_preferences.get(
                        "name", "usuário"
                    ),
                    "conversation_tone": state["conversation"].user_preferences.get(
                        "tone", "friendly"
                    ),
                    "context_summary": state["conversation"].context_summary,
                    "recent_topics": self._extract_recent_topics(
                        state["conversation"].messages
                    ),
                    "user_mood": self._assess_user_mood(state["conversation"].messages),
                }
            }
        )

        return base_state

    def _extract_recent_topics(self, messages: List[Dict[str, str]]) -> List[str]:
        """Extract recent conversation topics."""
        topics = []
        recent_messages = messages[-10:] if len(messages) > 10 else messages

        # Simple topic extraction - could be enhanced with NLP
        for msg in recent_messages:
            if msg.get("role") == "user":
                content = msg.get("content", "").lower()
                if "comando" in content or "shell" in content:
                    topics.append("comandos")
                elif "áudio" in content or "voz" in content:
                    topics.append("áudio")
                elif "arquivo" in content or "pasta" in content:
                    topics.append("arquivos")

        return list(set(topics))

    def _assess_user_mood(self, messages: List[Dict[str, str]]) -> str:
        """Simple mood assessment from recent messages."""
        if not messages:
            return "neutral"

        recent_user_messages = [
            msg["content"] for msg in messages[-5:] if msg.get("role") == "user"
        ]

        if not recent_user_messages:
            return "neutral"

        # Simple sentiment analysis
        text = " ".join(recent_user_messages).lower()

        positive_words = ["obrigado", "ótimo", "perfeito", "legal", "bom"]
        negative_words = ["problema", "erro", "ruim", "não funciona", "difícil"]

        positive_count = sum(word in text for word in positive_words)
        negative_count = sum(word in text for word in negative_words)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "frustrated"
        else:
            return "neutral"


class MemoryAgent(GiannaReActAgent):
    """
    Specialized ReAct agent for context and memory management.

    This agent handles long-term memory, context summarization,
    user preference learning, and information retrieval across
    conversation sessions.
    """

    def __init__(self, llm: BaseLanguageModel, config: Optional[AgentConfig] = None):
        """
        Initialize the Memory Agent.

        Args:
            llm: Language model instance
            config: Optional agent configuration
        """
        # Initialize tools for memory operations
        tools = [
            MemoryTool(),
        ]

        # Set default config if not provided
        if config is None:
            config = AgentConfig(
                name="memory_agent",
                description="Specialized agent for context and memory management",
                max_iterations=5,
                persistent_memory=True,
                max_context_length=20000,
            )

        super().__init__("memory_agent", llm, tools, config)

    @property
    def system_message(self) -> str:
        """System message specific to memory management."""
        return """Você é um especialista em gerenciamento de memória e contexto usando o padrão ReAct.

        Sua função é:
        1. Gerenciar memória de longo prazo entre sessões
        2. Sumarizar contextos de conversa importantes
        3. Identificar e armazenar preferências do usuário
        4. Recuperar informações relevantes quando necessário
        5. Manter consistência contextual ao longo do tempo

        PROCESSO ReAct:
        - THOUGHT: Analise que informações são importantes para lembrar
        - ACTION: Use MemoryTool para armazenar ou recuperar informações
        - OBSERVATION: Verifique se o contexto foi adequadamente preservado

        TIPOS DE MEMÓRIA:
        - Preferências do usuário (voz, idioma, estilo de comunicação)
        - Fatos importantes sobre o usuário
        - Contexto de projetos e tarefas em andamento
        - Padrões de uso e comportamentos
        - Histórico de problemas e soluções

        ESTRATÉGIAS DE MEMORIZAÇÃO:
        - Identificar informações que se repetem
        - Destacar preferências explicitamente mencionadas
        - Sumarizar conversas longas em pontos-chave
        - Conectar informações relacionadas
        - Remover informações obsoletas

        RECUPERAÇÃO INTELIGENTE:
        - Buscar contexto relevante para solicitações atuais
        - Antecipar necessidades baseadas no histórico
        - Sugerir continuações baseadas em padrões
        - Personalizar respostas com base em preferências conhecidas

        Use a ferramenta MemoryTool para todas as operações de memória."""

    def _prepare_agent_state(self, state: GiannaState) -> Dict[str, Any]:
        """Prepare state with memory-specific context."""
        base_state = super()._prepare_agent_state(state)

        # Add memory-specific context
        base_state.update(
            {
                "memory_context": {
                    "session_age": len(state["conversation"].messages),
                    "user_preferences": state["conversation"].user_preferences,
                    "context_summary": state["conversation"].context_summary,
                    "memory_capacity": self.config.max_context_length,
                    "persistent_enabled": self.config.persistent_memory,
                    "key_information": self._extract_key_information(state),
                    "memory_priorities": [
                        "user_preferences",
                        "ongoing_projects",
                        "frequent_tasks",
                        "personal_info",
                    ],
                }
            }
        )

        return base_state

    def _extract_key_information(self, state: GiannaState) -> List[str]:
        """Extract key information that should be remembered."""
        key_info = []

        # Extract from user preferences
        prefs = state["conversation"].user_preferences
        if prefs:
            key_info.extend([f"Preference: {k}={v}" for k, v in prefs.items()])

        # Extract from recent successful commands
        recent_commands = state["commands"].execution_history[-5:]
        successful_commands = [
            cmd for cmd in recent_commands if cmd.get("success", False)
        ]

        if successful_commands:
            key_info.append(f"Recent successful operations: {len(successful_commands)}")

        return key_info
