# Documentação da API - Gianna

Esta pasta contém a documentação completa da API do projeto Gianna, organizadas por módulos e fases de implementação.

## Estrutura da Documentação

### Core APIs (Fase 1)
- [Core State API](./core/state.md) - Sistema de gerenciamento de estado
- [LangGraph Integration](./core/langgraph.md) - Integração com LangGraph
- [Migration Utils](./core/migration.md) - Utilitários de migração

### Agents APIs (Fase 2)
- [Base Agent](./agents/base_agent.md) - Classe base para agentes
- [ReAct Agents](./agents/react_agents.md) - Agentes ReAct especializados
- [Tools Integration](./tools/) - Ferramentas integradas

### Audio APIs (Fase 3)
- [Streaming Audio](./audio/streaming.md) - Sistema de áudio em streaming
- [VAD Integration](./audio/vad.md) - Detecção de atividade de voz
- [Voice Workflows](./workflows/voice.md) - Fluxos de trabalho de voz

### Advanced APIs (Fase 4)
- [Memory System](./memory/) - Sistema de memória semântica
- [Learning System](./learning/) - Sistema de aprendizado adaptativo
- [Coordination](./coordination/) - Coordenação multi-agente

### Production APIs (Fase 5)
- [Optimization](./optimization/) - Otimizações de performance
- [Monitoring](./monitoring/) - Monitoramento e métricas
- [Deployment](./deployment/) - Guias de deployment

## Convenções da API

### Padrões de Resposta
Todas as APIs seguem padrões consistentes de resposta:

```python
# Resposta de sucesso
{
    "success": True,
    "data": {...},
    "metadata": {
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "1.0.0"
    }
}

# Resposta de erro
{
    "success": False,
    "error": {
        "code": "ERROR_CODE",
        "message": "Mensagem descritiva",
        "details": {...}
    },
    "metadata": {
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "1.0.0"
    }
}
```

### Tipos de Estado
- `GiannaState`: Estado principal do sistema
- `ConversationState`: Estado de conversação
- `AudioState`: Estado de áudio
- `CommandState`: Estado de comandos

### Padrões Assíncronos
- Todas as operações de I/O são assíncronas
- Use `await` para operações que podem bloquear
- Implemente timeouts apropriados

## Guias Rápidos

### Inicialização Básica
```python
from gianna.core.state_manager import StateManager
from gianna.core.langgraph_chain import LangGraphChain

# Inicializar gerenciador de estado
state_manager = StateManager()

# Criar chain LangGraph
chain = LangGraphChain("gpt4", "Você é um assistente útil.")
```

### Processamento de Áudio
```python
from gianna.audio.streaming import StreamingAudioProcessor
from gianna.workflows.voice_interaction import VoiceWorkflow

# Configurar processamento de áudio
audio_processor = StreamingAudioProcessor()
voice_workflow = VoiceWorkflow(chain, audio_processor)
```

### Multi-Agentes
```python
from gianna.agents.react_agents import CommandAgent, AudioAgent
from gianna.coordination.orchestrator import AgentOrchestrator

# Configurar agentes
command_agent = CommandAgent(llm)
audio_agent = AudioAgent(llm)

# Coordenar agentes
orchestrator = AgentOrchestrator()
orchestrator.register_agent(command_agent)
orchestrator.register_agent(audio_agent)
```

## Versionamento da API

- **v1.0.0**: Implementação básica (Fases 1-2)
- **v1.1.0**: Recursos de voz (Fase 3)
- **v1.2.0**: Recursos avançados (Fase 4)
- **v1.3.0**: Otimizações de produção (Fase 5)

## Suporte

Para dúvidas sobre a API, consulte:
- [Guia do Desenvolvedor](../developer-guide/)
- [Exemplos de Uso](../../examples/)
- [FAQ](../faq.md)
