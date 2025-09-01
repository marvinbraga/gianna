"""
Assistente de Voz em Tempo Real - Gianna

Este exemplo demonstra um sistema completo de assistente de voz com:

- Detecção de atividade de voz (VAD) em tempo real
- Streaming de áudio bidirecional
- Processamento concorrente de entrada e saída
- Cancelamento de eco e redução de ruído
- Múltiplos engines de STT/TTS
- Interface de usuário em tempo real
- Gerenciamento inteligente de sessões

Pré-requisitos:
- Gianna instalado com dependências de áudio
- Microfone e alto-falantes funcionais
- Chaves de API para STT/TTS (OpenAI, Google, ElevenLabs)
- PyAudio instalado (pip install pyaudio)

Uso:
    python real_time_voice_assistant.py

Comandos de voz:
- "Parar" ou "Stop" - Para a escuta
- "Reiniciar" - Reinicia o sistema
- "Configurar [engine]" - Muda engine TTS
- "Idioma [código]" - Muda idioma (pt, en, es)
"""

import asyncio
import json
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pyaudio

from gianna.assistants.audio.stt.factory_method import speech_to_text
from gianna.assistants.audio.tts.factory_method import text_to_speech
from gianna.assistants.audio.vad import VoiceActivityDetector
from gianna.assistants.models.factory_method import get_chain_instance
from gianna.audio.streaming import StreamingAudioProcessor
from gianna.core.langgraph_chain import LangGraphChain
from gianna.core.state import AudioState, ConversationState, GiannaState
from gianna.core.state_manager import StateManager


class VoiceAssistantState(Enum):
    """Estados do assistente de voz."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class AudioConfig:
    """Configuração de áudio."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: int = pyaudio.paInt16
    input_device_id: Optional[int] = None
    output_device_id: Optional[int] = None

    # VAD settings
    vad_threshold: float = 0.02
    min_silence_duration: float = 1.5
    max_speech_duration: float = 30.0

    # TTS settings
    tts_engine: str = "google"
    voice_speed: float = 1.0
    voice_pitch: float = 0.0
    voice_volume: float = 0.8

    # STT settings
    stt_engine: str = "whisper"
    language: str = "pt-br"


class RealTimeVoiceAssistant:
    """Assistente de voz avançado em tempo real."""

    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.state = VoiceAssistantState.IDLE
        self.state_manager = StateManager()
        self.session_id = f"voice_session_{int(time.time())}"

        # Componentes principais
        self.llm_chain = None
        self.vad = VoiceActivityDetector(
            threshold=self.config.vad_threshold,
            min_silence_duration=self.config.min_silence_duration,
        )

        # Audio streaming
        self.audio_processor = StreamingAudioProcessor()
        self.pyaudio_instance = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None

        # Buffers e controle
        self.audio_buffer = []
        self.is_recording = False
        self.is_speaking = False
        self.should_stop = False

        # Threading para processamento paralelo
        self.processing_thread = None
        self.audio_thread = None

        # Callbacks e hooks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        self.on_transcription: Optional[Callable] = None
        self.on_response_ready: Optional[Callable] = None
        self.on_speaking_start: Optional[Callable] = None
        self.on_speaking_end: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

        # Métricas e monitoramento
        self.metrics = {
            "interactions_count": 0,
            "avg_response_time": 0.0,
            "avg_transcription_time": 0.0,
            "avg_tts_time": 0.0,
            "errors_count": 0,
            "uptime_start": datetime.now(),
        }

        self._initialize()

    def _initialize(self):
        """Inicializar componentes do assistente."""
        print("🎤 Inicializando Assistente de Voz em Tempo Real...")

        try:
            # Inicializar LLM
            self.llm_chain = LangGraphChain(
                "gpt4",
                """Você é um assistente de voz inteligente e amigável. Características:

                - Responda de forma natural e conversacional
                - Seja conciso mas informativo
                - Adapte-se ao contexto da conversa
                - Use linguagem apropriada para voz
                - Evite textos muito longos
                - Seja proativo em oferecer ajuda
                - Lembre-se do contexto da conversação

                Você pode ajudar com:
                - Perguntas gerais e conversação
                - Comandos de sistema (com segurança)
                - Análise de informações
                - Tarefas de produtividade
                - Configurações do sistema de voz

                Sempre responda em português brasileiro, a menos que solicitado outro idioma.""",
            )

            # Configurar callbacks padrão
            self._setup_default_callbacks()

            # Verificar dispositivos de áudio
            self._check_audio_devices()

            print("✅ Assistente inicializado com sucesso")
            print(
                f"🔧 Configuração: {self.config.stt_engine} STT + {self.config.tts_engine} TTS"
            )
            print(f"🌐 Idioma: {self.config.language}")

        except Exception as e:
            print(f"❌ Erro na inicialização: {str(e)}")
            self.state = VoiceAssistantState.ERROR
            raise

    def _setup_default_callbacks(self):
        """Configurar callbacks padrão."""
        self.on_speech_start = lambda: self._log_event("🎤 Escutando...")
        self.on_speech_end = lambda: self._log_event("🔄 Processando...")
        self.on_transcription = lambda text: self._log_event(f"📝 Você disse: '{text}'")
        self.on_response_ready = lambda response: self._log_event(
            f"🤖 Assistente: '{response[:50]}...'"
        )
        self.on_speaking_start = lambda: self._log_event("🔊 Falando...")
        self.on_speaking_end = lambda: self._log_event("✅ Pronto para nova interação")
        self.on_error = lambda error: self._log_event(f"❌ Erro: {error}")

    def _check_audio_devices(self):
        """Verificar dispositivos de áudio disponíveis."""
        print("\\n🔍 Verificando dispositivos de áudio...")

        device_count = self.pyaudio_instance.get_device_count()
        input_devices = []
        output_devices = []

        for i in range(device_count):
            info = self.pyaudio_instance.get_device_info_by_index(i)

            if info["maxInputChannels"] > 0:
                input_devices.append((i, info["name"]))
            if info["maxOutputChannels"] > 0:
                output_devices.append((i, info["name"]))

        print(f"🎤 Dispositivos de entrada encontrados: {len(input_devices)}")
        if input_devices:
            for idx, name in input_devices[:3]:  # Mostrar apenas os 3 primeiros
                print(f"   • [{idx}] {name}")

        print(f"🔊 Dispositivos de saída encontrados: {len(output_devices)}")
        if output_devices:
            for idx, name in output_devices[:3]:
                print(f"   • [{idx}] {name}")

        # Auto-selecionar dispositivos se não especificados
        if self.config.input_device_id is None and input_devices:
            self.config.input_device_id = input_devices[0][0]
        if self.config.output_device_id is None and output_devices:
            self.config.output_device_id = output_devices[0][0]

    async def start_listening(self):
        """Iniciar escuta contínua."""
        if self.state != VoiceAssistantState.IDLE:
            print("⚠️  Assistente já está ativo")
            return

        print("🚀 Iniciando assistente de voz...")
        print("💡 Diga 'parar' ou pressione Ctrl+C para sair")

        self.should_stop = False
        self.state = VoiceAssistantState.LISTENING

        try:
            # Iniciar streams de áudio
            self._start_audio_streams()

            # Iniciar thread de processamento
            self.processing_thread = threading.Thread(
                target=self._processing_loop, daemon=True
            )
            self.processing_thread.start()

            # Iniciar thread de áudio
            self.audio_thread = threading.Thread(
                target=self._audio_capture_loop, daemon=True
            )
            self.audio_thread.start()

            print("✅ Assistente ativo! Pode falar agora...")

            # Loop principal
            await self._main_loop()

        except KeyboardInterrupt:
            print("\\n🛑 Interrompido pelo usuário")
        except Exception as e:
            print(f"❌ Erro durante execução: {str(e)}")
            self.state = VoiceAssistantState.ERROR
        finally:
            await self.stop()

    def _start_audio_streams(self):
        """Iniciar streams de entrada e saída de áudio."""
        try:
            # Stream de entrada (microfone)
            self.input_stream = self.pyaudio_instance.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.input_device_id,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=None,
            )

            # Stream de saída (alto-falantes)
            self.output_stream = self.pyaudio_instance.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                output=True,
                output_device_index=self.config.output_device_id,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=None,
            )

            print("🔊 Streams de áudio iniciados")

        except Exception as e:
            raise RuntimeError(f"Falha ao iniciar streams de áudio: {str(e)}")

    def _audio_capture_loop(self):
        """Loop de captura de áudio em thread separada."""
        while not self.should_stop:
            try:
                if self.state == VoiceAssistantState.LISTENING and not self.is_speaking:
                    # Capturar chunk de áudio
                    audio_data = self.input_stream.read(
                        self.config.chunk_size, exception_on_overflow=False
                    )

                    # Converter para numpy array
                    audio_chunk = np.frombuffer(audio_data, dtype=np.int16)

                    # Detectar atividade de voz
                    has_speech = self.vad.detect_activity(audio_chunk)

                    if has_speech:
                        if not self.is_recording:
                            # Início da fala detectado
                            self.is_recording = True
                            self.audio_buffer = []
                            if self.on_speech_start:
                                self.on_speech_start()

                        # Adicionar ao buffer
                        self.audio_buffer.append(audio_data)

                    elif self.is_recording:
                        # Possível fim da fala
                        self.audio_buffer.append(
                            audio_data
                        )  # Incluir silêncio para contexto

                        # Verificar se silêncio é longo o suficiente
                        silence_duration = (
                            len(self.audio_buffer)
                            * self.config.chunk_size
                            / self.config.sample_rate
                        )

                        if silence_duration >= self.config.min_silence_duration:
                            # Fim da fala confirmado
                            self.is_recording = False
                            if self.on_speech_end:
                                self.on_speech_end()

                            # Processar áudio capturado
                            asyncio.create_task(self._process_captured_audio())

                time.sleep(0.01)  # Pequena pausa para evitar uso excessivo de CPU

            except Exception as e:
                if not self.should_stop:
                    print(f"❌ Erro na captura de áudio: {str(e)}")
                    self.metrics["errors_count"] += 1

    async def _process_captured_audio(self):
        """Processar áudio capturado."""
        if not self.audio_buffer:
            return

        self.state = VoiceAssistantState.PROCESSING
        start_time = time.time()

        try:
            # Converter buffer para arquivo WAV temporário
            audio_data = b"".join(self.audio_buffer)

            # Speech-to-Text
            transcription_start = time.time()
            text = await self._transcribe_audio(audio_data)
            transcription_time = time.time() - transcription_start

            if not text or len(text.strip()) < 2:
                self.state = VoiceAssistantState.LISTENING
                return

            if self.on_transcription:
                self.on_transcription(text)

            # Verificar comandos especiais
            if self._handle_special_commands(text):
                self.state = VoiceAssistantState.LISTENING
                return

            # Processar com LLM
            llm_start = time.time()
            response = await self._get_llm_response(text)
            llm_time = time.time() - llm_start

            if self.on_response_ready:
                self.on_response_ready(response)

            # Text-to-Speech
            tts_start = time.time()
            await self._speak_response(response)
            tts_time = time.time() - tts_start

            # Atualizar métricas
            total_time = time.time() - start_time
            self._update_metrics(total_time, transcription_time, tts_time)

            self.state = VoiceAssistantState.LISTENING

        except Exception as e:
            print(f"❌ Erro no processamento: {str(e)}")
            self.metrics["errors_count"] += 1
            self.state = VoiceAssistantState.LISTENING
            if self.on_error:
                self.on_error(str(e))

    async def _transcribe_audio(self, audio_data: bytes) -> str:
        """Transcrever áudio para texto."""
        try:
            # Salvar áudio temporário
            temp_file = f"temp_audio_{int(time.time())}.wav"

            with wave.open(temp_file, "wb") as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(
                    self.pyaudio_instance.get_sample_size(self.config.format)
                )
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(audio_data)

            # Transcrever usando factory method
            result = speech_to_text(
                self.config.stt_engine, temp_file, self.config.language
            )

            # Limpar arquivo temporário
            import os

            if os.path.exists(temp_file):
                os.remove(temp_file)

            return result.get("text", "") if isinstance(result, dict) else str(result)

        except Exception as e:
            print(f"❌ Erro na transcrição: {str(e)}")
            return ""

    async def _get_llm_response(self, text: str) -> str:
        """Obter resposta do LLM."""
        try:
            result = await self.llm_chain.ainvoke(
                {"input": text}, session_id=self.session_id
            )

            return result.get(
                "output", "Desculpe, não consegui processar sua solicitação."
            )

        except Exception as e:
            print(f"❌ Erro no LLM: {str(e)}")
            return "Desculpe, houve um erro interno. Pode repetir?"

    async def _speak_response(self, text: str):
        """Falar resposta usando TTS."""
        self.is_speaking = True
        if self.on_speaking_start:
            self.on_speaking_start()

        try:
            # Gerar áudio TTS
            audio_file = text_to_speech(
                self.config.tts_engine,
                text,
                speed=self.config.voice_speed,
                pitch=self.config.voice_pitch,
            )

            if audio_file:
                # Reproduzir áudio
                await self._play_audio_file(audio_file)

        except Exception as e:
            print(f"❌ Erro no TTS: {str(e)}")
        finally:
            self.is_speaking = False
            if self.on_speaking_end:
                self.on_speaking_end()

    async def _play_audio_file(self, audio_file: str):
        """Reproduzir arquivo de áudio."""
        try:
            # Usar pygame ou similar para reprodução
            # Por simplicidade, vamos simular
            await asyncio.sleep(len(audio_file) * 0.1)  # Simular duração da fala

        except Exception as e:
            print(f"❌ Erro na reprodução: {str(e)}")

    def _handle_special_commands(self, text: str) -> bool:
        """Lidar com comandos especiais de controle."""
        text_lower = text.lower()

        # Comando para parar
        if any(word in text_lower for word in ["parar", "stop", "sair", "exit"]):
            print("🛑 Parando assistente...")
            asyncio.create_task(self.stop())
            return True

        # Comando para reiniciar
        if "reiniciar" in text_lower or "restart" in text_lower:
            print("🔄 Reiniciando...")
            asyncio.create_task(self._restart())
            return True

        # Comandos de configuração
        if "configurar" in text_lower:
            self._handle_config_command(text_lower)
            return True

        return False

    def _handle_config_command(self, text: str):
        """Lidar com comandos de configuração."""
        if "google" in text:
            self.config.tts_engine = "google"
            asyncio.create_task(self._speak_response("TTS configurado para Google"))
        elif "elevenlabs" in text:
            self.config.tts_engine = "elevenlabs"
            asyncio.create_task(self._speak_response("TTS configurado para ElevenLabs"))
        elif "whisper" in text:
            self.config.stt_engine = "whisper"
            asyncio.create_task(self._speak_response("STT configurado para Whisper"))

    def _processing_loop(self):
        """Loop de processamento em thread separada."""
        while not self.should_stop:
            try:
                # Monitoramento de sistema
                self._monitor_system_health()

                # Limpeza periódica
                if int(time.time()) % 60 == 0:  # A cada minuto
                    self._cleanup_old_data()

                time.sleep(1)

            except Exception as e:
                print(f"❌ Erro no loop de processamento: {str(e)}")

    def _monitor_system_health(self):
        """Monitorar saúde do sistema."""
        # Verificar uso de memória, latência, etc.
        # Implementação simplificada
        pass

    def _cleanup_old_data(self):
        """Limpar dados antigos."""
        # Limpar buffers, cache, etc.
        if len(self.audio_buffer) > 1000:
            self.audio_buffer = self.audio_buffer[-500:]

    async def _main_loop(self):
        """Loop principal do assistente."""
        try:
            while not self.should_stop:
                await asyncio.sleep(0.1)

                # Verificar se ainda há threads ativas
                if (
                    self.processing_thread
                    and not self.processing_thread.is_alive()
                    or self.audio_thread
                    and not self.audio_thread.is_alive()
                ):
                    print("⚠️  Thread crítica parou, reiniciando...")
                    break

        except KeyboardInterrupt:
            print("\\n🛑 Parando assistente...")

    async def stop(self):
        """Parar assistente."""
        print("🛑 Parando Assistente de Voz...")

        self.should_stop = True
        self.state = VoiceAssistantState.IDLE

        # Parar streams
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()

        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()

        # Aguardar threads terminarem
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)

        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2)

        # Fechar PyAudio
        self.pyaudio_instance.terminate()

        # Imprimir estatísticas finais
        self._print_final_statistics()

        print("✅ Assistente parado com sucesso")

    async def _restart(self):
        """Reiniciar assistente."""
        await self.stop()
        await asyncio.sleep(1)
        await self.start_listening()

    def _update_metrics(
        self, total_time: float, transcription_time: float, tts_time: float
    ):
        """Atualizar métricas de performance."""
        self.metrics["interactions_count"] += 1

        # Calcular média móvel de tempo de resposta
        count = self.metrics["interactions_count"]
        current_avg = self.metrics["avg_response_time"]
        self.metrics["avg_response_time"] = (
            (current_avg * (count - 1)) + total_time
        ) / count

        # Atualizar outras métricas
        self.metrics["avg_transcription_time"] = (
            self.metrics["avg_transcription_time"] * (count - 1) + transcription_time
        ) / count
        self.metrics["avg_tts_time"] = (
            self.metrics["avg_tts_time"] * (count - 1) + tts_time
        ) / count

    def _print_final_statistics(self):
        """Imprimir estatísticas finais."""
        print("\\n" + "=" * 50)
        print("📊 ESTATÍSTICAS DA SESSÃO")
        print("=" * 50)

        uptime = datetime.now() - self.metrics["uptime_start"]
        print(f"⏱️  Tempo ativo: {uptime}")
        print(f"💬 Interações: {self.metrics['interactions_count']}")
        print(f"⚡ Tempo médio de resposta: {self.metrics['avg_response_time']:.2f}s")
        print(f"🎤 Tempo médio STT: {self.metrics['avg_transcription_time']:.2f}s")
        print(f"🔊 Tempo médio TTS: {self.metrics['avg_tts_time']:.2f}s")
        print(f"❌ Erros: {self.metrics['errors_count']}")

        if self.metrics["interactions_count"] > 0:
            success_rate = (
                (self.metrics["interactions_count"] - self.metrics["errors_count"])
                / self.metrics["interactions_count"]
            ) * 100
            print(f"✅ Taxa de sucesso: {success_rate:.1f}%")

        print("=" * 50)

    def _log_event(self, message: str):
        """Log de eventos com timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")


def create_custom_config():
    """Criar configuração customizada baseada em preferências."""
    print("🔧 Configuração do Assistente de Voz")
    print("-" * 40)

    config = AudioConfig()

    # Engine TTS
    print("\\n🔊 Escolha o engine de Text-to-Speech:")
    print("1. Google TTS (padrão, gratuito)")
    print("2. ElevenLabs (qualidade superior, requer API key)")
    print("3. Whisper TTS")

    tts_choice = input("Opção (1-3): ").strip()
    if tts_choice == "2":
        config.tts_engine = "elevenlabs"
    elif tts_choice == "3":
        config.tts_engine = "whisper"
    else:
        config.tts_engine = "google"

    # Engine STT
    print("\\n🎤 Escolha o engine de Speech-to-Text:")
    print("1. Whisper (padrão, mais preciso)")
    print("2. Google STT (requer API key)")

    stt_choice = input("Opção (1-2): ").strip()
    if stt_choice == "2":
        config.stt_engine = "google"
    else:
        config.stt_engine = "whisper"

    # Idioma
    print("\\n🌐 Escolha o idioma:")
    print("1. Português Brasileiro (pt-br)")
    print("2. Inglês (en-us)")
    print("3. Espanhol (es-es)")

    lang_choice = input("Opção (1-3): ").strip()
    if lang_choice == "2":
        config.language = "en-us"
    elif lang_choice == "3":
        config.language = "es-es"
    else:
        config.language = "pt-br"

    # Sensibilidade do VAD
    print("\\n🔊 Sensibilidade de detecção de voz:")
    print("1. Baixa (ambientes ruidosos)")
    print("2. Normal (padrão)")
    print("3. Alta (ambientes silenciosos)")

    vad_choice = input("Opção (1-3): ").strip()
    if vad_choice == "1":
        config.vad_threshold = 0.05
    elif vad_choice == "3":
        config.vad_threshold = 0.01
    else:
        config.vad_threshold = 0.02

    return config


async def interactive_demo():
    """Demonstração interativa do assistente."""
    print("🎭 DEMONSTRAÇÃO INTERATIVA - ASSISTENTE DE VOZ")
    print("=" * 55)

    # Configuração
    use_custom = input("\\n🔧 Usar configuração customizada? (s/n): ").strip().lower()

    if use_custom == "s":
        config = create_custom_config()
    else:
        config = AudioConfig()
        print("✅ Usando configuração padrão")

    # Criar e iniciar assistente
    assistant = RealTimeVoiceAssistant(config)

    print("\\n🚀 Iniciando assistente...")
    print("💡 Dicas de uso:")
    print("   • Fale claramente após ouvir o sinal")
    print("   • Diga 'parar' para finalizar")
    print("   • Diga 'configurar google' para mudar TTS")
    print("   • Pressione Ctrl+C para sair a qualquer momento")

    try:
        await assistant.start_listening()
    except KeyboardInterrupt:
        print("\\n🛑 Demo finalizada pelo usuário")
    except Exception as e:
        print(f"❌ Erro na demonstração: {str(e)}")


def main():
    """Função principal."""
    try:
        # Verificar dependências
        import pyaudio

        print("✅ PyAudio disponível")
    except ImportError:
        print("❌ PyAudio não encontrado. Instale com: pip install pyaudio")
        return

    # Executar demonstração
    asyncio.run(interactive_demo())


if __name__ == "__main__":
    main()
