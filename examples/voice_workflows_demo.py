#!/usr/bin/env python3
"""
Voice Workflows Demo for Gianna AI Assistant

This script demonstrates how to use the new voice workflow implementations:
1. VoiceInteractionWorkflow - Complete voice pipeline with ReAct agents
2. StreamingVoiceWorkflow - Real-time streaming voice interactions

The demo shows:
- Basic workflow setup and configuration
- Voice processing with STT/TTS integration
- ReAct agent integration for intelligent responses
- Event handling and callbacks
- State management and persistence
- Error handling and recovery
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add the parent directory to the path to import gianna
sys.path.insert(0, str(Path(__file__).parent.parent))

from gianna.assistants.models.factory_method import get_chain_instance
from gianna.core.state import create_initial_state
from gianna.core.state_manager import StateManager
from gianna.workflows import (
    StreamingEvent,
    StreamingVoiceWorkflow,
    StreamingWorkflowConfig,
    VoiceInteractionWorkflow,
    VoiceWorkflowConfig,
    create_streaming_voice_workflow,
    create_voice_interaction_workflow,
)


class VoiceWorkflowDemo:
    """Demo class for voice workflow functionality."""

    def __init__(self):
        """Initialize the demo."""
        self.state_manager = StateManager("demo_voice_workflows.db")

        # Try to get LLM instance for ReAct agents
        try:
            self.llm_instance = get_chain_instance(
                "gpt35", "Você é um assistente de voz inteligente."
            )
        except Exception as e:
            print(f"⚠️  LLM não disponível, usando modo fallback: {e}")
            self.llm_instance = None

    def demo_voice_interaction_workflow(self):
        """Demonstrate the VoiceInteractionWorkflow."""
        print("\n" + "=" * 60)
        print("🎤 DEMO: Voice Interaction Workflow")
        print("=" * 60)

        try:
            # Create configuration
            config = VoiceWorkflowConfig(
                name="demo_voice_interaction",
                stt_engine="whisper",
                stt_language="pt-br",
                tts_engine="google",
                tts_language="pt-br",
                use_react_agents=True,
                enable_checkpointing=True,
                enable_error_recovery=True,
            )

            print(f"📋 Configuração criada: {config.name}")

            # Create workflow
            workflow = VoiceInteractionWorkflow(
                config=config,
                state_manager=self.state_manager,
                llm_instance=self.llm_instance,
            )

            print(f"🔧 Workflow inicializado com {len(workflow._agents)} agentes ReAct")

            # Set up callbacks
            workflow.set_callback("on_stt_start", self._on_stt_start)
            workflow.set_callback("on_stt_complete", self._on_stt_complete)
            workflow.set_callback("on_agent_start", self._on_agent_processing)
            workflow.set_callback("on_agent_complete", self._on_agent_response)
            workflow.set_callback("on_tts_complete", self._on_tts_complete)
            workflow.set_callback("on_error", self._on_error)

            print("📞 Callbacks configurados")

            # Compile workflow
            compiled_workflow = workflow.compile()
            print("⚙️  Workflow compilado com sucesso")

            # Create test state with simulated audio input
            session_id = "demo_session_001"
            initial_state = create_initial_state(session_id)

            # Simulate audio input
            audio_data = {
                "format": "wav",
                "sample_rate": 16000,
                "channels": 1,
                "duration": 2.5,
                "file_path": "simulated_audio.wav",
                "transcript": "Olá, como você está hoje?",
                "stt_confidence": 0.95,
                "timestamp": "2024-01-15T10:30:00",
            }

            print("🎧 Simulando entrada de áudio...")
            print(f"   Transcrição: '{audio_data['transcript']}'")

            # Execute workflow
            print("▶️  Executando workflow...")
            result_state = workflow.execute(
                initial_state=initial_state,
                session_id=session_id,
                audio_data=audio_data,
            )

            # Display results
            print("\n📊 RESULTADOS:")
            print(f"   Estado final: {result_state['audio'].current_mode}")
            print(
                f"   Mensagens na conversa: {len(result_state['conversation'].messages)}"
            )

            if result_state["conversation"].messages:
                last_message = result_state["conversation"].messages[-1]
                if last_message.get("role") == "assistant":
                    print(
                        f"   Resposta do assistente: '{last_message['content'][:100]}...'"
                    )

            # Show workflow info
            info = workflow.get_workflow_info()
            print(f"\n🔍 INFO DO WORKFLOW:")
            print(f"   Nome: {info['name']}")
            print(f"   Agentes disponíveis: {info['agents']['available']}")
            print(f"   Nodes do grafo: {info['graph']['nodes']}")

            print("✅ Demo Voice Interaction Workflow concluído com sucesso!")

        except Exception as e:
            print(f"❌ Erro no demo Voice Interaction: {e}")
            import traceback

            traceback.print_exc()

    async def demo_streaming_voice_workflow(self):
        """Demonstrate the StreamingVoiceWorkflow."""
        print("\n" + "=" * 60)
        print("🌊 DEMO: Streaming Voice Workflow")
        print("=" * 60)

        try:
            # Create streaming configuration
            config = StreamingWorkflowConfig(
                name="demo_streaming_voice",
                enable_continuous_listening=True,
                vad_threshold=0.02,
                min_silence_duration=1.0,
                stt_engine="whisper",
                stt_language="pt-br",
                tts_engine="google",
                tts_language="pt-br",
                use_react_agents=True,
                enable_events=True,
                enable_async_processing=True,
            )

            print(f"📋 Configuração de streaming criada: {config.name}")

            # Create streaming workflow
            workflow = StreamingVoiceWorkflow(
                config=config,
                state_manager=self.state_manager,
                llm_instance=self.llm_instance,
            )

            print("🌊 Streaming workflow inicializado")

            # Set up event handlers
            workflow.add_event_handler(
                StreamingEvent.VOICE_DETECTED, self._on_voice_detected
            )
            workflow.add_event_handler(
                StreamingEvent.SPEECH_COMPLETED, self._on_speech_completed
            )
            workflow.add_event_handler(
                StreamingEvent.PROCESSING_STARTED, self._on_processing_started
            )
            workflow.add_event_handler(
                StreamingEvent.RESPONSE_GENERATED, self._on_response_generated
            )
            workflow.add_event_handler(
                StreamingEvent.SPEAKING_STARTED, self._on_speaking_started
            )
            workflow.add_event_handler(
                StreamingEvent.ERROR_OCCURRED, self._on_streaming_error
            )

            print("📡 Event handlers configurados")

            # Compile workflow
            workflow.compile()
            print("⚙️  Streaming workflow compilado")

            # Start streaming (simulation)
            print("▶️  Iniciando streaming workflow...")

            # In a real implementation, this would start continuous listening
            # For demo purposes, we'll simulate a short streaming session
            session_id = await workflow.start_streaming("demo_streaming_session")
            print(f"🎙️  Streaming iniciado para sessão: {session_id}")

            # Simulate some streaming events
            print("🔄 Simulando eventos de streaming...")

            # Get status
            status = workflow.get_streaming_status(session_id)
            print(f"\n📈 STATUS DO STREAMING:")
            print(f"   Running: {status['is_running']}")
            print(f"   Estado atual: {status['current_state']}")
            print(f"   Sessões ativas: {status['active_sessions']}")
            print(f"   Pipeline inicializado: {status['pipeline']['initialized']}")
            print(f"   Eventos na fila: {status['events']['queue_size']}")

            # Wait a moment to let events process
            await asyncio.sleep(2)

            # Stop streaming
            print("⏹️  Parando streaming workflow...")
            await workflow.stop_streaming(session_id)

            print("✅ Demo Streaming Voice Workflow concluído com sucesso!")

        except Exception as e:
            print(f"❌ Erro no demo Streaming Voice: {e}")
            import traceback

            traceback.print_exc()

    def demo_factory_functions(self):
        """Demonstrate the factory functions."""
        print("\n" + "=" * 60)
        print("🏭 DEMO: Factory Functions")
        print("=" * 60)

        try:
            # Simple voice workflow
            print("Creating simple voice workflow...")
            simple_workflow = create_voice_interaction_workflow()
            print(f"✅ Simple workflow created: {simple_workflow.config.name}")

            # Simple streaming workflow
            print("Creating simple streaming workflow...")
            simple_streaming = create_streaming_voice_workflow()
            print(f"✅ Simple streaming created: {simple_streaming.config.name}")

            print("✅ Factory functions demo concluído!")

        except Exception as e:
            print(f"❌ Erro no demo Factory Functions: {e}")

    # Callback methods for voice interaction workflow

    def _on_stt_start(self, state):
        """Callback for STT processing start."""
        print("   🎤 Iniciando conversão de fala para texto...")

    def _on_stt_complete(self, state, transcript):
        """Callback for STT processing completion."""
        print(f"   ✅ STT concluído: '{transcript[:50]}...'")

    def _on_agent_processing(self, state):
        """Callback for agent processing start."""
        print("   🧠 Iniciando processamento com agente ReAct...")

    def _on_agent_response(self, state, response):
        """Callback for agent response."""
        print(f"   💬 Resposta gerada: '{response[:50]}...'")

    def _on_tts_complete(self, state, tts_result):
        """Callback for TTS completion."""
        engine = tts_result.get("engine", "unknown")
        print(f"   🔊 TTS concluído ({engine})")

    def _on_error(self, error, node):
        """Callback for errors."""
        print(f"   ⚠️  Erro em {node}: {error}")

    # Event handlers for streaming workflow

    async def _on_voice_detected(self, event_data):
        """Handle voice detected event."""
        print("   👂 Voz detectada pelo VAD")

    async def _on_speech_completed(self, event_data):
        """Handle speech completed event."""
        print("   ✅ Fala completa detectada")

    async def _on_processing_started(self, event_data):
        """Handle processing started event."""
        print("   🔄 Processamento iniciado")

    async def _on_response_generated(self, event_data):
        """Handle response generated event."""
        response = event_data.get("data", {}).get("response", "")
        print(f"   💭 Resposta gerada: '{response[:40]}...'")

    async def _on_speaking_started(self, event_data):
        """Handle speaking started event."""
        print("   🗣️  Iniciando reprodução de áudio")

    async def _on_streaming_error(self, event_data):
        """Handle streaming error event."""
        error = event_data.get("data", {}).get("error", "Unknown error")
        print(f"   ❌ Erro no streaming: {error}")


async def main():
    """Run the voice workflows demo."""
    print("🎙️  GIANNA VOICE WORKFLOWS DEMO")
    print("=" * 60)
    print("Este demo mostra os novos workflows de voz implementados:")
    print("1. VoiceInteractionWorkflow - Pipeline completo de voz")
    print("2. StreamingVoiceWorkflow - Interações de voz em tempo real")
    print("3. Factory Functions - Funções de conveniência")

    demo = VoiceWorkflowDemo()

    try:
        # Demo 1: Voice Interaction Workflow
        demo.demo_voice_interaction_workflow()

        # Demo 2: Streaming Voice Workflow
        await demo.demo_streaming_voice_workflow()

        # Demo 3: Factory Functions
        demo.demo_factory_functions()

        print("\n" + "=" * 60)
        print("🎉 TODOS OS DEMOS CONCLUÍDOS COM SUCESSO!")
        print("=" * 60)
        print("\nPróximos passos:")
        print("- Integre os workflows em suas aplicações")
        print("- Configure os agentes ReAct conforme necessário")
        print("- Ajuste as configurações de áudio para seu ambiente")
        print("- Implemente callbacks personalizados conforme necessário")

    except KeyboardInterrupt:
        print("\n👋 Demo interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro geral no demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main())
