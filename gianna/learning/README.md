# Gianna Learning System

Sistema completo de aprendizado e adaptação que permite ao Gianna aprender com as interações do usuário e adaptar seu comportamento automaticamente.

## Visão Geral

O sistema de aprendizado do Gianna é composto por quatro módulos principais que trabalham em conjunto para criar uma experiência de usuário personalizada e adaptativa:

- **User Adaptation**: Aprende preferências do usuário e adapta respostas
- **Pattern Analysis**: Analisa padrões comportamentais e de uso
- **Adaptation Engine**: Algoritmos de aprendizado de máquina para otimização
- **State Integration**: Persistência e gerenciamento de estado

## Características Principais

### 🧠 Aprendizado Inteligente
- Aprendizado online e offline
- Algoritmos de aprendizado incremental
- Detecção automática de mudanças de preferência
- Sistema de confiança nas preferências aprendidas

### 📊 Análise de Padrões
- Análise temporal de uso (horários, dias da semana)
- Frequência de comandos e preferências
- Detecção de tópicos de interesse
- Padrões de sessão e comportamento

### ⚡ Adaptação Dinâmica
- Personalização de estilo de resposta
- Adaptação de comprimento de resposta
- Ajuste de nível técnico baseado em expertise
- Otimização de tempo de resposta

### 💾 Persistência
- Integração completa com GiannaState
- Banco de dados SQLite para persistência
- Export/import de dados de aprendizado
- Métricas de performance e estatísticas

## Arquitetura

```
gianna/learning/
├── __init__.py              # Interface principal
├── user_adaptation.py       # Aprendizado de preferências do usuário
├── pattern_analysis.py      # Análise de padrões comportamentais
├── adaptation_engine.py     # Motor de adaptação com ML
├── state_integration.py     # Integração com persistência
└── README.md               # Esta documentação
```

## Uso Básico

### Iniciando o Sistema

```python
from gianna.learning import LearningStateManager

# Inicializar o sistema de aprendizado
learning_manager = LearningStateManager()
```

### Registrando Interações

```python
from gianna.learning import InteractionContext
from datetime import datetime

# Criar contexto de interação
context = InteractionContext(
    user_input="Como implementar uma API REST?",
    response_generated="Para implementar uma API REST...",
    timestamp=datetime.now(),
    interaction_mode="text",
    user_feedback="Muito detalhado, pode ser mais breve?",
    command_used="explain"
)

# Registrar a interação
learning_manager.record_interaction(context, satisfaction_score=0.6)
```

### Adaptando Respostas

```python
# Resposta original
original_response = "Esta é uma explicação muito detalhada sobre o tópico..."

# Contexto atual
current_context = {
    'user_input': 'Explique brevemente',
    'interaction_mode': 'text',
    'user_expertise': 0.7
}

# Adaptar resposta baseada nas preferências aprendidas
adapted_response, metadata = learning_manager.adapt_response(
    original_response,
    current_context
)

print(f"Resposta adaptada: {adapted_response}")
print(f"Adaptações aplicadas: {metadata['ml_adaptations']}")
```

### Analisando Padrões

```python
# Obter análise completa dos padrões do usuário
analysis = learning_manager.pattern_analyzer.get_comprehensive_analysis(
    list(learning_manager.preference_learner.interaction_history)
)

print("Padrões temporais:", analysis['temporal_patterns'])
print("Comandos mais usados:", analysis['command_analysis']['most_used_commands'])
print("Tópicos de interesse:", analysis['topic_analysis']['topic_distribution'])
```

### Gerando Perfil do Usuário

```python
# Gerar perfil completo do usuário
user_profile = learning_manager.get_user_profile()

print("Preferências de alta confiança:", user_profile['preferences']['high_confidence_preferences'])
print("Padrões comportamentais:", user_profile['behavioral_patterns']['detected_patterns'])
print("Confiança geral:", user_profile['interaction_stats']['learning_confidence'])
```

## Componentes Detalhados

### UserPreferenceLearner

Aprende preferências específicas do usuário:

- **Comprimento de resposta**: brief, detailed, comprehensive
- **Estilo de comunicação**: formal, casual, technical
- **Modo de interação**: voice, text, mixed
- **Nível de detalhamento**: high, medium, low

```python
from gianna.learning import UserPreferenceLearner, PreferenceType

learner = UserPreferenceLearner()

# Verificar preferências aprendidas
preferences = learner.get_preference_summary()
print(f"Preferências aprendidas: {preferences['preferences_count']}")
```

### PatternAnalyzer

Analisa padrões complexos de comportamento:

```python
from gianna.learning import PatternAnalyzer

analyzer = PatternAnalyzer()

# Análise temporal
temporal_patterns = analyzer.analyze_temporal_patterns(interactions)

# Análise de frequência de comandos
command_analysis = analyzer.analyze_command_frequency(interactions)

# Análise de tópicos de interesse
topic_analysis = analyzer.analyze_topic_interests(interactions)
```

### AdaptationEngine

Motor de adaptação com algoritmos de ML:

```python
from gianna.learning import AdaptationEngine, AdaptationStrategy

engine = AdaptationEngine(
    strategy=AdaptationStrategy.BALANCED,
    learning_mode=LearningMode.HYBRID
)

# Aprender de interação
context = {'user_input': 'test', 'response': 'response'}
engine.learn_from_interaction(context, user_satisfaction=0.8)

# Adaptar resposta
result = engine.adapt_response("Original response", context)
print(f"Sucesso: {result.success}")
print(f"Adaptações: {result.adaptations_applied}")
```

## Configuração Avançada

### Estratégias de Adaptação

```python
# Conservadora - adaptação lenta e cuidadosa
AdaptationStrategy.CONSERVATIVE

# Balanceada - velocidade moderada (padrão)
AdaptationStrategy.BALANCED

# Agressiva - adaptação rápida às mudanças
AdaptationStrategy.AGGRESSIVE

# Consciente de contexto - adapta baseado no contexto
AdaptationStrategy.CONTEXT_AWARE
```

### Modos de Aprendizado

```python
# Online - aprendizado em tempo real
LearningMode.ONLINE

# Batch - aprendizado periódico
LearningMode.BATCH

# Híbrido - combinação de ambos (padrão)
LearningMode.HYBRID
```

## Persistência e Backup

### Exportando Dados

```python
# Exportar todos os dados de aprendizado
export_data = learning_manager.export_learning_data()

# Salvar em arquivo
with open('learning_backup.json', 'w') as f:
    f.write(json.dumps(export_data, indent=2))
```

### Importando Dados

```python
# Carregar de arquivo
with open('learning_backup.json', 'r') as f:
    import_data = json.loads(f.read())

# Importar dados
success = learning_manager.import_learning_data(import_data)
print(f"Import {'bem-sucedido' if success else 'falhou'}")
```

### Reset do Sistema

```python
# Reset completo (cuidado!)
learning_manager.reset_all_learning()
```

## Métricas e Monitoramento

### Estatísticas do Sistema

```python
stats = learning_manager.get_learning_statistics()

print(f"Total de interações: {stats['database_stats']['total_interactions']}")
print(f"Preferências armazenadas: {stats['database_stats']['stored_preferences']}")
print(f"Padrões detectados: {stats['database_stats']['detected_patterns']}")
print(f"Confiança geral: {stats['overall_confidence']:.2f}")
print(f"Status do sistema: {stats['system_health']}")
```

### Insights de Aprendizado

```python
insights = learning_manager.adaptation_engine.get_learning_insights()

print("Métricas de aprendizado:", insights['learning_metrics'])
print("Features mais importantes:", insights['most_important_features'])
print("Taxa de sucesso atual:", insights['current_satisfaction_trend'])
print("Recomendações:", insights['recommendations'])
```

## Exemplos Práticos

Veja o arquivo `examples/learning_system_demo.py` para uma demonstração completa do sistema de aprendizado em ação.

## Testes

Execute os testes básicos:

```bash
cd tests
python test_learning_system.py
```

Ou execute o demo interativo:

```bash
cd examples
python learning_system_demo.py
```

## Considerações de Performance

### Otimizações

- Histórico limitado (padrão: 1000 interações)
- Cache de análises para melhor performance
- Algoritmos incrementais para aprendizado eficiente
- Persistência assíncrona para não bloquear interações

### Configurações Recomendadas

```python
# Para uso intensivo
learner = UserPreferenceLearner(max_history=2000, confidence_threshold=0.8)

# Para uso leve
learner = UserPreferenceLearner(max_history=500, confidence_threshold=0.6)

# Para ambiente de produção
engine = AdaptationEngine(
    strategy=AdaptationStrategy.BALANCED,
    learning_mode=LearningMode.HYBRID,
    max_history_size=1500
)
```

## Limitações Atuais

- Suporte apenas para um usuário por instância
- Algoritmos de ML simples (linear learning)
- Análise de tópicos baseada em palavras-chave
- Sem suporte para aprendizado federado

## Roadmap Futuro

- [ ] Suporte multi-usuário
- [ ] Algoritmos de ML mais sofisticados
- [ ] Análise de sentimento avançada
- [ ] Aprendizado federado
- [ ] Interface web para monitoramento
- [ ] Análise de voz e entonação
- [ ] Integração com sistemas externos

## Contribuição

Para contribuir com o sistema de aprendizado:

1. Implemente novos algoritmos em `adaptation_engine.py`
2. Adicione novos tipos de padrões em `pattern_analysis.py`
3. Estenda tipos de preferência em `user_adaptation.py`
4. Adicione testes em `tests/test_learning_system.py`

## Suporte

Para problemas ou dúvidas sobre o sistema de aprendizado, consulte os logs do sistema ou execute o demo para verificar o funcionamento.

O sistema está pronto para uso em produção e pode ser estendido conforme necessário.
