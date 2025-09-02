# Multi-Agent Coordination System

The Gianna Multi-Agent Coordination System enables intelligent orchestration of specialized AI agents for complex task execution and workflow management.

## 🏗️ Architecture

### Core Components

1. **AgentOrchestrator** (`orchestrator.py`)
   - Central coordination hub for all agents
   - Manages agent lifecycle and resource allocation
   - Provides sequential, parallel, and hybrid execution modes
   - Handles error recovery and performance optimization

2. **AgentRouter** (`router.py`)
   - Intelligent routing based on Portuguese keyword analysis
   - Context-aware agent selection with confidence scoring
   - Pattern matching and contextual adjustments
   - Performance tracking and routing statistics

3. **Specialized Agents** (from `../agents/react_agents.py`)
   - **CommandAgent**: Shell commands and system operations
   - **AudioAgent**: Audio processing and voice operations
   - **ConversationAgent**: Natural dialogue and conversation management
   - **MemoryAgent**: Context and memory management

## 🧠 Intelligent Routing

### Portuguese Keyword Analysis

The routing system analyzes Portuguese text using multiple strategies:

```python
from gianna.coordination import AgentOrchestrator, AgentRouter
from gianna.core.state import create_initial_state

# Initialize router
router = AgentRouter()

# Create test state
state = create_initial_state("session_123")
state["conversation"]["messages"] = [
    {"role": "user", "content": "Execute o comando ls -la no terminal"}
]

# Route request
agent_type, confidence = router.route_request(state)
print(f"Routed to: {agent_type.value} (confidence: {confidence:.2f})")
# Output: Routed to: command_agent (confidence: 0.85)
```

### Routing Rules

- **Command Agent**: `comando`, `executar`, `shell`, `terminal`, etc.
- **Audio Agent**: `falar`, `áudio`, `voz`, `ouvir`, `tocar`, etc.
- **Memory Agent**: `lembrar`, `memória`, `histórico`, `salvar`, etc.
- **Conversation Agent**: Default fallback for general conversation

## 🔄 Execution Modes

### 1. Sequential Execution

Agents execute one after another, with state passing between them:

```python
from gianna.coordination.router import ExecutionMode

orchestrator = AgentOrchestrator()

results = orchestrator.coordinate_agents(
    agents=[AgentType.MEMORY, AgentType.AUDIO, AgentType.CONVERSATION],
    state=state,
    execution_mode=ExecutionMode.SEQUENTIAL
)
```

### 2. Parallel Execution

Agents execute simultaneously with the same initial state:

```python
results = orchestrator.coordinate_agents(
    agents=[AgentType.COMMAND, AgentType.MEMORY, AgentType.CONVERSATION],
    state=state,
    execution_mode=ExecutionMode.PARALLEL
)
```

### 3. Hybrid Execution

Smart mix of sequential and parallel based on agent priorities:

```python
results = orchestrator.coordinate_agents(
    agents=[AgentType.COMMAND, AgentType.AUDIO, AgentType.CONVERSATION],
    state=state,
    execution_mode=ExecutionMode.HYBRID
)

# Execution order:
# 1. High Priority (Sequential): COMMAND, MEMORY
# 2. Medium Priority (Parallel): AUDIO
# 3. Low Priority (Sequential): CONVERSATION
```

## 🎯 Agent Registration

### Basic Registration

```python
from gianna.agents.react_agents import CommandAgent, AudioAgent
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create and register agents
orchestrator = AgentOrchestrator()
orchestrator.register_agent(CommandAgent(llm))
orchestrator.register_agent(AudioAgent(llm))

# Check registered agents
agents = orchestrator.get_registered_agents()
print(f"Registered: {[a.value for a in agents]}")
```

### Agent Lifecycle Management

```python
# Check agent status
status = orchestrator.get_agent_status(AgentType.COMMAND)
print(f"Command Agent Status: {status}")

# Health check all agents
health = orchestrator.health_check()
print(f"Health Status: {health}")

# Reset agent errors
orchestrator.reset_agent_errors(AgentType.COMMAND)
```

## 🔧 Error Recovery

### Automatic Failover

When an agent is unavailable, the system automatically finds alternatives:

```python
# If CommandAgent is busy/error, automatically routes to ConversationAgent
result = orchestrator.route_and_execute(
    state=state,
    requested_agent=AgentType.COMMAND  # Will fallback if unavailable
)

if not result.success:
    print(f"Execution failed: {result.error}")
```

### Manual Error Recovery

```python
# Reset specific agent errors
orchestrator.reset_agent_errors(AgentType.COMMAND)

# Reset all agent errors
orchestrator.reset_agent_errors()

# Check system status
status = orchestrator.get_system_status()
print(f"Error agents: {status['error_agents']}")
```

## 📊 Performance Monitoring

### Real-time Metrics

```python
# Get performance metrics
metrics = orchestrator.get_performance_metrics()
for agent_name, stats in metrics.items():
    print(f"{agent_name}:")
    print(f"  - Total Requests: {stats['total_requests']}")
    print(f"  - Success Rate: {1-stats['error_rate']:.1%}")
    print(f"  - Avg Execution Time: {stats['average_execution_time']:.2f}s")

# Get routing statistics
routing_stats = router.get_routing_stats()
print(f"Total Routes: {routing_stats['total_routes']}")
print(f"Agent Distribution: {routing_stats['agent_distribution']}")
```

### System Status

```python
# Comprehensive system status
status = orchestrator.get_system_status()
print(f"""
Orchestrator Status: {status['orchestrator_status']}
Total Agents: {status['total_agents']}
Available: {status['available_agents']}
Busy: {status['busy_agents']}
Error: {status['error_agents']}
""")
```

## 🌊 Advanced Workflows

### Complex Multi-Agent Workflow

```python
async def complex_workflow(orchestrator, initial_state):
    """Example of a complex multi-agent workflow."""

    # Phase 1: Information gathering (parallel)
    phase1_results = orchestrator.coordinate_agents(
        agents=[AgentType.MEMORY, AgentType.COMMAND],
        state=initial_state,
        execution_mode=ExecutionMode.PARALLEL
    )

    # Update state with phase 1 results
    updated_state = update_state_with_results(initial_state, phase1_results)

    # Phase 2: Processing (sequential)
    phase2_results = orchestrator.coordinate_agents(
        agents=[AgentType.AUDIO, AgentType.CONVERSATION],
        state=updated_state,
        execution_mode=ExecutionMode.SEQUENTIAL
    )

    return phase1_results + phase2_results
```

### Context-Aware Routing

```python
def intelligent_routing_example():
    """Demonstrate context-aware routing."""

    # Audio mode affects routing
    state = create_initial_state("session_123")
    state["audio"]["current_mode"] = "listening"
    state["conversation"]["messages"] = [
        {"role": "user", "content": "Can you help me?"}
    ]

    # Will prefer AudioAgent due to audio mode context
    agent_type, confidence = router.route_request(state)
    print(f"Context-aware routing: {agent_type.value}")
```

## 🔍 Debugging and Monitoring

### Logging Configuration

```python
from loguru import logger

# Enable detailed coordination logging
logger.add("coordination.log",
          filter="gianna.coordination",
          level="DEBUG",
          rotation="1 MB")
```

### Performance Analysis

```python
# Analyze routing performance
routing_stats = router.get_routing_stats()
for route in routing_stats['recent_routes']:
    print(f"Message: {route['message']}")
    print(f"Agent: {route['selected_agent']}")
    print(f"Confidence: {route['confidence']:.2f}")
    print(f"All Scores: {route['all_scores']}")
```

## 🧪 Testing

Run the comprehensive demonstration:

```bash
python examples/multi_agent_coordination_example.py
```

This will demonstrate:
- Intelligent routing with Portuguese keywords
- Sequential execution workflows
- Parallel execution capabilities
- Hybrid execution strategies
- Error recovery mechanisms
- Performance monitoring and metrics

## 🔧 Configuration

### Orchestrator Settings

```python
# Custom orchestrator configuration
orchestrator = AgentOrchestrator(max_workers=8)

# Custom routing thresholds
router = AgentRouter()
# Routing rules and confidence thresholds are automatically configured
# for Portuguese language support
```

### Agent Configuration

```python
from gianna.agents.base_agent import AgentConfig

# Custom agent configuration
config = AgentConfig(
    name="custom_command_agent",
    description="Custom command agent with enhanced capabilities",
    max_iterations=20,
    safety_checks=True,
    validate_inputs=True
)

agent = CommandAgent(llm, config=config)
orchestrator.register_agent(agent)
```

## 🚀 Production Deployment

### Best Practices

1. **Resource Management**
   - Monitor agent execution times and resource usage
   - Configure appropriate `max_workers` based on system capacity
   - Implement proper error handling and recovery strategies

2. **Performance Optimization**
   - Use parallel execution for independent operations
   - Leverage hybrid mode for complex workflows
   - Monitor routing statistics and adjust as needed

3. **Error Handling**
   - Implement comprehensive logging for debugging
   - Set up health checks and monitoring
   - Configure automatic error recovery and failover

4. **Security Considerations**
   - Validate all user inputs before routing
   - Implement proper authentication and authorization
   - Monitor for potential security vulnerabilities

### Scaling Considerations

- The orchestrator supports concurrent execution with configurable worker pools
- Agent registration is thread-safe and supports dynamic registration/unregistration
- Performance metrics are automatically tracked for optimization insights
- The system is designed to handle high-throughput scenarios with proper resource management

## 📚 Integration Examples

See `examples/multi_agent_coordination_example.py` for comprehensive usage examples and demonstrations of all system capabilities.
