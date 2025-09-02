# Guia do Usuário - Gianna

Bem-vindo ao Gianna, seu assistente de voz inteligente! Este guia irá te ensinar como usar todos os recursos do sistema de forma eficiente.

## Índice

1. [Primeiros Passos](#primeiros-passos)
2. [Comandos Básicos](#comandos-básicos)
3. [Interação por Voz](#interação-por-voz)
4. [Comandos de Sistema](#comandos-de-sistema)
5. [Personalização](#personalização)
6. [Solução de Problemas](#solução-de-problemas)

## Primeiros Passos

### Instalação Rápida

1. **Clone o repositório:**
   ```bash
   git clone <repository-url>
   cd gianna
   ```

2. **Configure o ambiente:**
   ```bash
   # Instalar dependências
   poetry install

   # Configurar variáveis de ambiente
   cp .env.example .env
   # Edite o .env com suas chaves de API
   ```

3. **Teste a instalação:**
   ```bash
   python main.py
   ```

### Configuração Inicial

Edite o arquivo `.env` com suas chaves de API:

```env
# APIs dos LLMs (pelo menos uma é necessária)
OPENAI_API_KEY=sua_chave_openai
GOOGLE_API_KEY=sua_chave_google
ELEVEN_LABS_API_KEY=sua_chave_elevenlabs

# Configurações padrão
LLM_DEFAULT_MODEL=gpt4
TTS_DEFAULT_TYPE=google
STT_DEFAULT_TYPE=whisper

# Configurações de idioma
LANGUAGE=pt-br
VOICE_LANGUAGE=pt-BR
```

## Comandos Básicos

### Conversa Simples

```python
from gianna.core.langgraph_chain import LangGraphChain

# Inicializar assistente
assistente = LangGraphChain("gpt4", "Você é um assistente útil em português.")

# Fazer uma pergunta
resposta = assistente.invoke({
    "input": "Qual é a capital do Brasil?"
})

print(resposta["output"])
```

### Comandos de Sistema

```python
from gianna.agents.react_agents import CommandAgent
from gianna.assistants.models.factory_method import get_chain_instance

# Inicializar agente de comandos
llm = get_chain_instance("gpt4", "Você é um especialista em comandos shell.")
agente_comando = CommandAgent(llm)

# Executar comando
resultado = agente_comando.invoke({
    "input": "Liste os arquivos no diretório atual"
})
```

## Interação por Voz

### Conversação por Voz

```python
from gianna.workflows.voice_interaction import VoiceWorkflow
import asyncio

async def exemplo_voz():
    # Configurar workflow de voz
    workflow = VoiceWorkflow()

    # Iniciar conversação
    await workflow.start_conversation()

    print("Diga algo... (pressione Ctrl+C para sair)")

    try:
        while True:
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        await workflow.stop_conversation()

# Executar
asyncio.run(exemplo_voz())
```

### Comandos por Voz Disponíveis

| Comando | Descrição | Exemplo |
|---------|-----------|---------|
| "Execute" | Executar comando shell | "Execute listar arquivos" |
| "Fale" | Síntese de fala | "Fale olá mundo" |
| "Lembre" | Armazenar informação | "Lembre que gosto de café" |
| "Busque" | Buscar na memória | "Busque informações sobre café" |
| "Pare" | Parar operação atual | "Pare de falar" |

### Configurações de Voz

```python
from gianna.core.state import AudioState

# Configurar preferências de voz
audio_config = AudioState(
    voice_settings={
        "speed": 1.0,        # Velocidade da fala (0.5 - 2.0)
        "pitch": 0,          # Tom da voz (-20 a +20)
        "voice_id": "pt-BR", # Idioma/sotaque
        "engine": "google"   # Motor TTS (google, elevenlabs, whisper)
    },
    language="pt-br"
)
```

## Comandos de Sistema

### Execução Segura de Comandos

O Gianna executa comandos shell de forma segura com validação:

```python
# Exemplos de comandos seguros
comandos_seguros = [
    "ls -la",                    # Listar arquivos
    "pwd",                       # Diretório atual
    "df -h",                     # Espaço em disco
    "ps aux | head -10",         # Processos ativos
    "git status",                # Status do Git
    "python --version",          # Versão do Python
]

# O sistema automaticamente:
# 1. Valida comandos antes de executar
# 2. Aplica timeout de 30 segundos
# 3. Copia comandos para clipboard antes da execução
# 4. Retorna saída estruturada (stdout, stderr, exit_code)
```

### Comandos Restringidos

Por segurança, alguns comandos são bloqueados:

```python
comandos_bloqueados = [
    "rm -rf",           # Remoção recursiva
    "sudo",             # Elevação de privilégio
    "chmod 777",        # Permissões perigosas
    "dd",               # Operações de disco baixo nível
    "mkfs",             # Formatação
    "> /dev/null",      # Redirecionamento perigoso
]
```

## Personalização

### Configuração de Modelos

```python
# Modelos disponíveis por provedor
modelos_disponiveis = {
    "OpenAI": ["gpt35", "gpt4", "gpt4-turbo"],
    "Google": ["gemini", "gemini-pro"],
    "Anthropic": ["claude", "claude-instant"],
    "Groq": ["llama2", "mixtral"],
    "Nvidia": ["llama2-70b"]
}

# Configurar modelo preferido
from gianna.core.state_manager import StateManager

state_manager = StateManager()
state_manager.set_preference("llm_model", "gpt4")
state_manager.set_preference("tts_engine", "elevenlabs")
```

### Personalização de Respostas

```python
from gianna.learning.user_adaptation import UserPreferenceLearner

# Sistema aprende suas preferências automaticamente
learner = UserPreferenceLearner()

# Configurações manuais
preferencias = {
    "estilo_resposta": "conciso",      # conciso, detalhado, tecnico
    "idioma": "pt-br",
    "formalidade": "informal",         # formal, informal, tecnico
    "areas_interesse": ["programacao", "linux", "python"]
}

learner.update_preferences("user123", preferencias)
```

## Solução de Problemas

### Problemas Comuns

#### 1. Erro de API Key
```
Erro: OpenAI API key not found
Solução: Verifique se OPENAI_API_KEY está configurada no .env
```

#### 2. Problema de Áudio
```
Erro: Audio device not found
Solução:
- Verifique se o microfone está conectado
- Instale: sudo apt-get install portaudio19-dev (Linux)
- Instale: brew install portaudio (macOS)
```

#### 3. Comando Não Executado
```
Erro: Command execution blocked
Solução: O comando pode estar na lista de bloqueados por segurança
```

### Diagnóstico do Sistema

```python
from gianna.optimization.monitoring import SystemMonitor

# Verificar status do sistema
monitor = SystemMonitor()
status = monitor.get_system_status()

print(f"Status do sistema: {status['overall_health']}")
print(f"Uso de memória: {status['memory_usage']}%")
print(f"Latência média: {status['avg_latency']}ms")
print(f"Modelo ativo: {status['active_model']}")
```

### Logs e Debug

```python
import logging

# Configurar logging detalhado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gianna.log'),
        logging.StreamHandler()
    ]
)

# Executar com debug ativo
assistente = LangGraphChain("gpt4", "Assistente", debug=True)
```

## Recursos Avançados

### 1. Memória Semântica
- O sistema lembra conversas anteriores
- Busca inteligente por contexto
- Aprende suas preferências automaticamente

### 2. Multi-Agentes
- Agente de comandos para operações de sistema
- Agente de áudio para processamento de voz
- Coordenação automática entre agentes

### 3. Otimização Automática
- Cache inteligente de respostas
- Balanceamento de carga entre modelos
- Otimização de performance em tempo real

## Próximos Passos

1. **Explore os exemplos**: Veja `/examples` para casos de uso avançados
2. **Consulte a API**: Documentação completa em `/docs/api`
3. **Participe do desenvolvimento**: Guia em `/docs/developer-guide`

Para mais informações, consulte a [documentação completa](../README.md).
