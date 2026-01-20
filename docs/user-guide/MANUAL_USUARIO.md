# 📖 Gianna - Manual do Usuário

**Versão:** 0.1.4
**Última Atualização:** Janeiro 2026

---

## Índice

1. [Introdução](#1-introdução)
2. [Instalação](#2-instalação)
3. [Configuração Inicial](#3-configuração-inicial)
4. [Uso Básico](#4-uso-básico)
5. [Provedores de LLM](#5-provedores-de-llm)
6. [Sistema de Áudio](#6-sistema-de-áudio)
7. [Sistema de Agentes](#7-sistema-de-agentes)
8. [Memória Semântica](#8-memória-semântica)
9. [CLI Administrativa](#9-cli-administrativa)
10. [Otimização e Performance](#10-otimização-e-performance)
11. [Exemplos Práticos](#11-exemplos-práticos)
12. [Troubleshooting](#12-troubleshooting)
13. [Referência de API](#13-referência-de-api)

---

## 1. Introdução

### O que é o Gianna?

**Gianna** (Generative Intelligent Artificial Neural Network Assistant) é um framework avançado para criação de assistentes de voz inteligentes em Python. Ele oferece:

- 🎤 **Pipeline de voz completo**: Captura → STT → LLM → TTS → Reprodução
- 🤖 **8+ provedores de LLM**: OpenAI, Anthropic, Google, Groq, NVIDIA, xAI, Cohere, Ollama
- 🎯 **Sistema multi-agente**: Agentes especializados com orquestração inteligente
- 🧠 **Memória semântica**: Armazenamento vetorial com embeddings
- 🎙️ **VAD avançado**: 6 algoritmos de detecção de voz
- ⚡ **Otimizado para produção**: Cache, circuit breaker, monitoramento

### Arquitetura Geral

```
┌─────────────────────────────────────────────────────────────┐
│                      ENTRADA DO USUÁRIO                     │
│                    (Voz ou Texto)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PROCESSAMENTO DE VOZ                     │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                │
│   │   VAD   │ -> │   STT   │ -> │  Texto  │                │
│   └─────────┘    └─────────┘    └─────────┘                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ORQUESTRAÇÃO DE AGENTES                   │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│   │ Comando  │  │  Áudio   │  │ Memória  │  │ Conversa │   │
│   │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │   │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PROCESSAMENTO LLM                        │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│   │ OpenAI  │  │Anthropic│  │ Google  │  │  Groq   │ ...   │
│   └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      SAÍDA DE VOZ                           │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                │
│   │  Texto  │ -> │   TTS   │ -> │  Áudio  │                │
│   └─────────┘    └─────────┘    └─────────┘                │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Instalação

### Requisitos

- Python 3.11 ou superior (até 3.13)
- Sistema operacional: Linux, macOS ou Windows
- Microfone e alto-falantes (para funcionalidades de voz)

### Instalação com uv (Recomendado)

```bash
# Instalar uv (se ainda não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clonar o repositório
git clone https://github.com/marvinbraga/gianna.git
cd gianna

# Instalar dependências básicas
uv sync

# Instalar com todas as funcionalidades
uv sync --extra full

# Instalar para desenvolvimento
uv sync --extra dev --extra test
```

### Instalação com pip

```bash
# Instalação básica
pip install -e .

# Com funcionalidades de VAD avançado
pip install -e ".[vad-full]"

# Com todas as funcionalidades
pip install -e ".[full]"
```

### Extras Disponíveis

| Extra | Descrição |
|-------|-----------|
| `vad-basic` | VAD com WebRTC |
| `vad-advanced` | VAD com análise espectral |
| `vad-ml` | VAD com Machine Learning (PyTorch) |
| `vad-full` | Todos os algoritmos VAD |
| `optimization` | Redis + análise de performance |
| `analytics` | Matplotlib para visualizações |
| `dev` | Ferramentas de desenvolvimento |
| `test` | Ferramentas de teste |
| `full` | Todas as funcionalidades |

---

## 3. Configuração Inicial

### 3.1 Configurando Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```bash
# Copiar exemplo
cp .env.example .env

# Editar com suas chaves
nano .env
```

Conteúdo do `.env`:

```env
# Provedor padrão (openai, anthropic, google, groq, etc.)
DEFAULT_MODEL=openai

# Chaves de API dos provedores LLM
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GOOGLE_API_KEY=your-google-api-key
GROQ_API_KEY=gsk_your-groq-key
XAI_API_KEY=your-xai-key
NVIDIA_API_KEY=your-nvidia-key
COHERE_API_KEY=your-cohere-key

# Text-to-Speech
DEFAULT_TTS=google
ELEVEN_LABS_API_KEY=your-elevenlabs-key

# Speech-to-Text
DEFAULT_STT=whisper_local

# Logging
LOGURU_LEVEL=INFO
```

### 3.2 Arquivo de Configuração YAML

Para configurações avançadas, crie `gianna.yaml`:

```yaml
# gianna.yaml - Configuração do Gianna
app_name: "Gianna"
environment: "development"
debug: false

llm:
  default_provider: "openai"
  default_model: "gpt-4"
  temperature: 0.7
  max_tokens: 4096
  timeout_seconds: 30.0
  max_retries: 3

audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 1024
  tts_provider: "google"
  tts_language: "pt-BR"
  stt_provider: "whisper_local"
  stt_language: "pt"
  vad_algorithm: "adaptive"
  vad_threshold: 0.02

memory:
  embedding_provider: "local"
  vectorstore_provider: "in_memory"
  collection_name: "gianna_memory"
  similarity_threshold: 0.7
  max_search_results: 10
  enable_clustering: true

cache:
  enabled: true
  backend: "memory"
  default_ttl_seconds: 3600

monitoring:
  log_level: "INFO"
  log_format: "json"
  enable_metrics: true
```

### 3.3 Validando a Configuração

```bash
# Validar configuração
uv run python -m gianna.cli config validate

# Validar para produção (mais rigoroso)
uv run python -m gianna.cli config validate --strict
```

---

## 4. Uso Básico

### 4.1 Exemplo Mínimo

```python
from gianna.assistants.models.factory_method import get_chain_instance

# Criar uma instância de chat
chain = get_chain_instance()

# Fazer uma pergunta
response = chain.invoke("Olá! Como você pode me ajudar?")
print(response)
```

### 4.2 Chat Interativo

```python
from gianna.assistants.models.factory_method import get_chain_instance

def main():
    # Inicializar o assistente
    chain = get_chain_instance()

    print("Gianna iniciada! Digite 'sair' para encerrar.\n")

    while True:
        # Obter input do usuário
        user_input = input("Você: ").strip()

        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("Até logo!")
            break

        if not user_input:
            continue

        # Obter resposta
        response = chain.invoke(user_input)
        print(f"Gianna: {response}\n")

if __name__ == "__main__":
    main()
```

### 4.3 Usando com Estado (LangGraph)

```python
from gianna.core.langgraph_chain import LangGraphChain
from gianna.core.state import create_initial_state

# Criar chain com suporte a estado
chain = LangGraphChain(
    model_name="openai",
    persist_state=True,
    db_path="./gianna_state.db"
)

# Criar estado inicial
state = create_initial_state(session_id="user_session_001")

# Conversar mantendo contexto
response1 = chain.invoke_with_state("Meu nome é João", state)
response2 = chain.invoke_with_state("Qual é meu nome?", state)
# Gianna lembrará que o nome é João
```

---

## 5. Provedores de LLM

### 5.1 OpenAI

```python
from gianna.assistants.models.factory_method import get_chain_instance

# Usando GPT-4
chain = get_chain_instance(model_name="openai")
response = chain.invoke("Explique quantum computing")

# Modelos disponíveis: gpt-3.5-turbo, gpt-4, gpt-4-turbo
```

### 5.2 Anthropic (Claude)

```python
chain = get_chain_instance(model_name="anthropic")
response = chain.invoke("Escreva um poema sobre IA")

# Modelos: claude-3-opus, claude-3-sonnet, claude-instant
```

### 5.3 Google (Gemini)

```python
chain = get_chain_instance(model_name="google")
response = chain.invoke("Analise esta imagem...")

# Modelos: gemini-pro, gemini-pro-vision
```

### 5.4 Groq (Rápido)

```python
chain = get_chain_instance(model_name="groq")
response = chain.invoke("Responda rapidamente: 2+2?")

# Modelos: llama2-70b, mixtral-8x7b (muito rápidos!)
```

### 5.5 Ollama (Local)

```python
# Primeiro, inicie o Ollama localmente
# ollama run llama2

chain = get_chain_instance(model_name="ollama")
response = chain.invoke("Olá! Funcionando localmente!")

# Modelos: depende do que você baixou no Ollama
```

### 5.6 Usando Circuit Breaker (Resiliência)

```python
from gianna.optimization.circuit_breaker import create_llm_circuit_breaker
from gianna.assistants.models.factory_method import get_chain_instance

# Criar circuit breaker para OpenAI
breaker = create_llm_circuit_breaker("openai", failure_threshold=3)

@breaker
def safe_invoke(prompt: str):
    chain = get_chain_instance(model_name="openai")
    return chain.invoke(prompt)

# Se OpenAI falhar 3 vezes, o circuit abre e falha rápido
try:
    response = safe_invoke("Olá!")
except CircuitOpenError:
    print("OpenAI está instável, tente mais tarde")
```

---

## 6. Sistema de Áudio

### 6.1 Text-to-Speech (TTS)

```python
from gianna.assistants.audio.tts import TextToSpeechFactory

# Usando Google TTS
tts = TextToSpeechFactory.create("google")
audio_data = tts.synthesize("Olá! Eu sou a Gianna.", language="pt-BR")
tts.play(audio_data)

# Usando ElevenLabs (voz mais natural)
tts_eleven = TextToSpeechFactory.create("elevenlabs")
audio = tts_eleven.synthesize("Voz super realista!", voice_id="Rachel")
tts_eleven.play(audio)
```

### 6.2 Speech-to-Text (STT)

```python
from gianna.assistants.audio.stt import SpeechToTextFactory

# Usando Whisper local
stt = SpeechToTextFactory.create("whisper_local")

# Transcrever arquivo de áudio
text = stt.transcribe("audio.wav", language="pt")
print(f"Transcrição: {text}")

# Transcrever do microfone
text = stt.transcribe_from_microphone(duration=5)
print(f"Você disse: {text}")
```

### 6.3 Voice Activity Detection (VAD)

```python
from gianna.audio.vad import VADFactory, VADAlgorithm

# Criar VAD adaptativo (recomendado)
vad = VADFactory.create(VADAlgorithm.ADAPTIVE)

# Configurar callbacks
def on_speech_start(result):
    print("🎤 Fala detectada!")

def on_speech_end(result):
    print(f"🔇 Silêncio após {result.speech_duration:.1f}s de fala")

vad.set_callbacks(
    on_speech_start=on_speech_start,
    on_speech_end=on_speech_end
)

# Processar áudio em tempo real
import sounddevice as sd

def audio_callback(indata, frames, time, status):
    result = vad.process_chunk(indata)
    if result.is_voice_active:
        print(f"Energia: {result.energy_level:.4f}")

# Iniciar captura
with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
    print("Ouvindo... Pressione Ctrl+C para parar")
    input()
```

### 6.4 Algoritmos VAD Disponíveis

| Algoritmo | Descrição | Uso Recomendado |
|-----------|-----------|-----------------|
| `ENERGY` | Baseado em RMS | Rápido, ambientes silenciosos |
| `SPECTRAL` | Análise de frequência | Mais preciso, custo médio |
| `WEBRTC` | Google WebRTC VAD | Equilibrado, uso geral |
| `SILERO` | Rede neural | Alta precisão, mais lento |
| `ADAPTIVE` | Fusão multi-algoritmo | **Recomendado** para produção |
| `ML` | Machine Learning | Personalizável |

### 6.5 Pipeline de Voz Completo

```python
from gianna.workflows.voice_interaction import VoiceWorkflow

# Criar workflow de voz
workflow = VoiceWorkflow(
    stt_provider="whisper_local",
    tts_provider="google",
    llm_provider="openai",
    vad_algorithm="adaptive"
)

# Iniciar assistente de voz
async def main():
    await workflow.start()

    print("Assistente de voz ativo! Fale algo...")

    # O workflow gerencia automaticamente:
    # 1. Detecta quando você começa a falar (VAD)
    # 2. Grava sua fala
    # 3. Transcreve para texto (STT)
    # 4. Processa com LLM
    # 5. Converte resposta para voz (TTS)
    # 6. Reproduz o áudio

    await workflow.run_forever()

import asyncio
asyncio.run(main())
```

---

## 7. Sistema de Agentes

### 7.1 Entendendo os Agentes

O Gianna usa um sistema multi-agente com orquestração inteligente:

| Agente | Responsabilidade |
|--------|------------------|
| **CommandAgent** | Executa comandos de sistema |
| **AudioAgent** | Processa áudio (TTS/STT) |
| **MemoryAgent** | Gerencia memória semântica |
| **ConversationAgent** | Conversação geral |

### 7.2 Usando o Orquestrador

```python
from gianna.coordination.orchestrator import AgentOrchestrator
from gianna.agents.react_agents import (
    ConversationAgent,
    CommandAgent,
    AudioAgent,
    MemoryAgent
)
from gianna.core.state import create_initial_state

# Criar orquestrador
orchestrator = AgentOrchestrator()

# Registrar agentes
orchestrator.register_agent("conversation", ConversationAgent())
orchestrator.register_agent("command", CommandAgent())
orchestrator.register_agent("audio", AudioAgent())
orchestrator.register_agent("memory", MemoryAgent())

# Criar estado
state = create_initial_state()

# Processar requisição (roteamento automático)
response = orchestrator.process(
    message="Execute o comando ls -la",
    state=state
)
# O orquestrador detecta keywords "execute" e "comando"
# e roteia para o CommandAgent
```

### 7.3 Roteamento Personalizado

As regras de roteamento estão em `config/routing_rules.yaml`:

```yaml
# config/routing_rules.yaml
version: "1.0"
default_agent: "CONVERSATION"

rules:
  - agent_type: "COMMAND"
    priority: 3
    keywords:
      - "executar"
      - "rodar"
      - "comando"
      - "terminal"
    patterns:
      - "\\bexecute\\s+"
      - "\\brodar\\s+(o\\s+)?(script|programa)"

  - agent_type: "AUDIO"
    priority: 3
    keywords:
      - "falar"
      - "tocar"
      - "gravar"
      - "volume"
```

---

## 8. Memória Semântica

### 8.1 Armazenando Informações

```python
from gianna.memory.semantic_memory import SemanticMemory
from gianna.memory.embeddings import EmbeddingProvider

# Criar memória semântica
memory = SemanticMemory(
    embedding_provider=EmbeddingProvider.LOCAL,
    collection_name="minha_memoria"
)

# Armazenar informações
memory.store(
    content="O usuário prefere respostas curtas e diretas",
    metadata={"type": "preference", "user_id": "user_001"}
)

memory.store(
    content="O projeto X foi concluído em Janeiro de 2026",
    metadata={"type": "fact", "project": "X"}
)
```

### 8.2 Buscando Informações

```python
# Busca por similaridade semântica
results = memory.search(
    query="Quais são as preferências do usuário?",
    max_results=5,
    threshold=0.7
)

for result in results:
    print(f"Conteúdo: {result.content}")
    print(f"Similaridade: {result.score:.2f}")
    print(f"Metadata: {result.metadata}")
    print()
```

### 8.3 Integração com Conversas

```python
from gianna.core.langgraph_chain import LangGraphChain

# Chain com memória semântica integrada
chain = LangGraphChain(
    model_name="openai",
    enable_memory=True,
    memory_config={
        "embedding_provider": "local",
        "vectorstore_provider": "chromadb",
        "persist_directory": "./memory_data"
    }
)

# Conversa com contexto persistente
chain.invoke("Lembre-se: meu aniversário é dia 15 de março")
# ... depois de reiniciar o programa ...
chain.invoke("Quando é meu aniversário?")
# Gianna: Seu aniversário é dia 15 de março!
```

---

## 9. CLI Administrativa

### 9.1 Comandos Disponíveis

```bash
# Mostrar ajuda
uv run python -m gianna.cli --help

# Informações do sistema
uv run python -m gianna.cli info
```

### 9.2 Gerenciamento de Configuração

```bash
# Validar configuração atual
uv run python -m gianna.cli config validate

# Validar para produção (mais rigoroso)
uv run python -m gianna.cli config validate --strict

# Gerar template de configuração
uv run python -m gianna.cli config generate -o minha_config.yaml

# Mostrar configuração atual
uv run python -m gianna.cli config show

# Mostrar apenas seção específica
uv run python -m gianna.cli config show --section llm
```

### 9.3 Health Checks

```bash
# Verificar saúde de todos os componentes
uv run python -m gianna.cli health check

# Verificar componente específico
uv run python -m gianna.cli health check --component llm
uv run python -m gianna.cli health check --component audio

# Com detalhes
uv run python -m gianna.cli health check --verbose
```

### 9.4 Benchmarks

```bash
# Benchmark de VAD
uv run python -m gianna.cli benchmark vad

# Benchmark de algoritmo específico
uv run python -m gianna.cli benchmark vad --algorithm adaptive

# Benchmark com duração customizada
uv run python -m gianna.cli benchmark vad --duration 30
```

### 9.5 Gerenciamento de Cache

```bash
# Ver estatísticas de cache
uv run python -m gianna.cli cache stats

# Limpar cache (requer confirmação)
uv run python -m gianna.cli cache clear --confirm

# Limpar tipo específico de cache
uv run python -m gianna.cli cache clear --type memory --confirm
```

### 9.6 Gerenciamento de Secrets

```bash
# Validar configuração de secrets
uv run python -m gianna.cli secrets validate

# Rotacionar chaves (requer confirmação)
uv run python -m gianna.cli secrets rotate --confirm
```

---

## 10. Otimização e Performance

### 10.1 Circuit Breaker

Protege contra falhas em cascata:

```python
from gianna.optimization.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    get_circuit_breaker
)

# Configuração personalizada
config = CircuitBreakerConfig(
    failure_threshold=5,      # Abre após 5 falhas
    timeout_seconds=60.0,     # Tenta reabrir após 60s
    success_threshold=3,      # Fecha após 3 sucessos
)

# Criar breaker
breaker = get_circuit_breaker("meu_servico", config=config)

@breaker
def chamar_api_externa():
    return requests.get("https://api.exemplo.com/data")

# Verificar status
status = breaker.get_status()
print(f"Estado: {status['state']}")
print(f"Falhas: {status['failure_count']}")
```

### 10.2 Memory Pool para Áudio

Reduz pressão no garbage collector:

```python
from gianna.audio.memory_pool import AudioBufferPool, BufferContext

# Criar pool de buffers
pool = AudioBufferPool(
    chunk_size=1024,
    pool_size=32,
    dtype=np.float32
)

# Usar buffers do pool
for audio_chunk in audio_stream:
    with BufferContext(pool) as buffer:
        buffer[:] = audio_chunk
        process(buffer)
    # Buffer automaticamente retornado ao pool
```

### 10.3 Cache Multi-Layer

```python
from gianna.optimization.caching import MultiLayerCache

# Cache em 3 camadas: Memória → Redis → SQLite
cache = MultiLayerCache(
    memory_size=1000,
    redis_url="redis://localhost:6379",
    sqlite_path="./cache.db"
)

# Armazenar
cache.set("resposta_comum", "Olá! Como posso ajudar?", ttl=3600)

# Recuperar (busca em cada camada)
value = cache.get("resposta_comum")
```

### 10.4 Async/Await para Performance

```python
from gianna.core.async_utils import AsyncExecutor, AsyncRetry

# Executar múltiplas tarefas concorrentemente
async with AsyncExecutor(max_concurrent=5) as executor:
    results = await executor.map(process_item, items)

# Retry automático com backoff
@AsyncRetry(max_retries=3, delay=1.0, backoff_factor=2.0)
async def chamar_api():
    return await client.get("/endpoint")
```

### 10.5 Logging Estruturado

```python
from gianna.monitoring.structured_logging import (
    configure_logging,
    get_logger,
    log_context
)

# Configurar logging JSON
configure_logging(
    json_format=True,
    level="INFO",
    file_path="./logs/gianna.log"
)

logger = get_logger(__name__)

# Log com contexto automático
with log_context(user_id="user_001", session_id="sess_123"):
    logger.info("Processando requisição", action="query")
    # Output JSON inclui user_id e session_id automaticamente
```

---

## 11. Exemplos Práticos

### 11.1 Assistente de Voz Simples

```python
"""
Assistente de voz simples com Gianna.
Fale algo e receba uma resposta em voz.
"""

import asyncio
from gianna.workflows.voice_interaction import VoiceWorkflow

async def main():
    # Criar workflow
    workflow = VoiceWorkflow(
        llm_provider="openai",
        tts_provider="google",
        stt_provider="whisper_local",
        language="pt-BR"
    )

    print("🎤 Assistente de voz iniciado!")
    print("Fale algo ou pressione Ctrl+C para sair.\n")

    try:
        await workflow.run_forever()
    except KeyboardInterrupt:
        print("\n👋 Até logo!")
        await workflow.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### 11.2 Chatbot com Memória

```python
"""
Chatbot que lembra conversas anteriores.
"""

from gianna.core.langgraph_chain import LangGraphChain
from gianna.core.state import create_initial_state

def main():
    # Criar chain com persistência
    chain = LangGraphChain(
        model_name="openai",
        persist_state=True,
        db_path="./chatbot_memory.db"
    )

    # Criar ou recuperar sessão
    session_id = input("ID da sessão (Enter para nova): ").strip()
    if not session_id:
        session_id = f"session_{int(time.time())}"
        print(f"Nova sessão: {session_id}")

    state = create_initial_state(session_id=session_id)

    print("\nChatbot com memória iniciado!")
    print("Digite 'sair' para encerrar.\n")

    while True:
        user_input = input("Você: ").strip()

        if user_input.lower() == 'sair':
            print("Sessão salva. Até logo!")
            break

        response = chain.invoke_with_state(user_input, state)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()
```

### 11.3 Executador de Comandos por Voz

```python
"""
Execute comandos de sistema usando sua voz.
"""

from gianna.agents.react_agents import CommandAgent
from gianna.assistants.audio.stt import SpeechToTextFactory

def main():
    stt = SpeechToTextFactory.create("whisper_local")
    agent = CommandAgent()

    print("🎤 Diga um comando (ex: 'liste os arquivos')")

    while True:
        print("\nOuvindo...")
        text = stt.transcribe_from_microphone(duration=5)

        if not text:
            continue

        print(f"Você disse: {text}")

        if "sair" in text.lower():
            break

        # Processar comando
        result = agent.execute(text)
        print(f"Resultado: {result}")

if __name__ == "__main__":
    main()
```

### 11.4 Análise de Documentos com RAG

```python
"""
Sistema de perguntas e respostas sobre documentos.
"""

from gianna.memory.semantic_memory import SemanticMemory
from gianna.assistants.models.factory_method import get_chain_instance

def main():
    # Criar memória para documentos
    memory = SemanticMemory(
        collection_name="documentos",
        persist_directory="./doc_memory"
    )

    # Carregar documentos (exemplo)
    documents = [
        "A empresa foi fundada em 2020 por João Silva.",
        "O produto principal é um software de gestão.",
        "Temos 50 funcionários e escritórios em SP e RJ.",
    ]

    for doc in documents:
        memory.store(doc, metadata={"source": "manual"})

    print("Documentos carregados. Faça perguntas!\n")

    chain = get_chain_instance()

    while True:
        question = input("Pergunta: ").strip()
        if question.lower() == 'sair':
            break

        # Buscar contexto relevante
        context = memory.search(question, max_results=3)
        context_text = "\n".join([r.content for r in context])

        # Gerar resposta com contexto
        prompt = f"""Contexto:
{context_text}

Pergunta: {question}

Responda baseado apenas no contexto fornecido."""

        response = chain.invoke(prompt)
        print(f"Resposta: {response}\n")

if __name__ == "__main__":
    main()
```

---

## 12. Troubleshooting

### 12.1 Problemas Comuns

#### "API Key não encontrada"

```bash
# Verifique se o .env está configurado
cat .env | grep API_KEY

# Verifique se as variáveis estão sendo carregadas
uv run python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

#### "Microfone não detectado"

```bash
# Linux: Verificar dispositivos de áudio
arecord -l

# Instalar dependências de áudio
sudo apt-get install portaudio19-dev python3-pyaudio
```

#### "Erro de importação do PyTorch"

```bash
# Reinstalar com suporte CUDA (se tiver GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Ou versão CPU
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### "ChromaDB não funciona"

```bash
# Instalar dependências do ChromaDB
pip install chromadb

# Se tiver erro de SQLite
pip install pysqlite3-binary
```

### 12.2 Logs de Debug

```bash
# Ativar logs detalhados
export LOGURU_LEVEL=DEBUG
uv run python seu_script.py

# Ou no código
from loguru import logger
logger.enable("gianna")
```

### 12.3 Verificação de Saúde

```bash
# Executar health check completo
uv run python -m gianna.cli health check --verbose

# Verificar configuração
uv run python -m gianna.cli config validate
```

---

## 13. Referência de API

### 13.1 Módulos Principais

| Módulo | Descrição |
|--------|-----------|
| `gianna.core` | Estado, LangGraph, utilitários async |
| `gianna.assistants.models` | Provedores LLM (OpenAI, Anthropic, etc.) |
| `gianna.assistants.audio` | TTS, STT, players, recorders |
| `gianna.audio.vad` | Voice Activity Detection |
| `gianna.agents` | Agentes ReAct especializados |
| `gianna.coordination` | Orquestração e roteamento |
| `gianna.memory` | Memória semântica com embeddings |
| `gianna.optimization` | Cache, circuit breaker, performance |
| `gianna.monitoring` | Logging estruturado, métricas |
| `gianna.config` | Sistema de configurações |
| `gianna.workflows` | Workflows de voz com LangGraph |

### 13.2 Classes Principais

```python
# LLM
from gianna.assistants.models.factory_method import get_chain_instance

# Estado
from gianna.core.state import create_initial_state, GiannaState
from gianna.core.langgraph_chain import LangGraphChain

# Áudio
from gianna.assistants.audio.tts import TextToSpeechFactory
from gianna.assistants.audio.stt import SpeechToTextFactory
from gianna.audio.vad import VADFactory

# Agentes
from gianna.agents.react_agents import (
    ConversationAgent,
    CommandAgent,
    AudioAgent,
    MemoryAgent
)
from gianna.coordination.orchestrator import AgentOrchestrator

# Memória
from gianna.memory.semantic_memory import SemanticMemory

# Otimização
from gianna.optimization.circuit_breaker import CircuitBreaker
from gianna.audio.memory_pool import AudioBufferPool

# Configuração
from gianna.config import get_settings, Settings
```

### 13.3 Exemplos de Notebooks

Os notebooks tutoriais estão em `notebooks/`:

- `tutorial_fase1_langgraph.ipynb` - Fundamentos com LangGraph
- `tutorial_fase2_multiagent.ipynb` - Sistema multi-agente
- `tutorial_fase3_voice.ipynb` - Pipeline de voz
- `tutorial_complete_workflow.ipynb` - Workflow completo

---

## Suporte

- **GitHub Issues**: https://github.com/marvinbraga/gianna/issues
- **Documentação**: `docs/` no repositório
- **Exemplos**: `examples/` no repositório

---

**Gianna** - *Generative Intelligent Artificial Neural Network Assistant*
Desenvolvido com ❤️ em Python
