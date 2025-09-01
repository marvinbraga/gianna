"""
Sistema Avançado de Orquestração Multi-Agente - Gianna

Este exemplo demonstra como criar e coordenar múltiplos agentes especializados
que trabalham em conjunto para resolver tarefas complexas, incluindo:

- Roteamento inteligente de tarefas
- Execução paralela e sequencial
- Comunicação inter-agente
- Agregação de resultados
- Fallback e recovery automático

Pré-requisitos:
- Gianna instalado (poetry install)
- Chaves de API configuradas (OpenAI, Google, etc.)
- Redis (opcional, para cache distribuído)

Uso:
    python multi_agent_orchestration.py
"""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from gianna.agents.react_agents import AudioAgent, CommandAgent
from gianna.assistants.models.factory_method import get_chain_instance
from gianna.coordination.orchestrator import AgentOrchestrator
from gianna.core.langgraph_chain import LangGraphChain
from gianna.core.state_manager import StateManager
from gianna.optimization.monitoring import SystemMonitor


class TaskPriority(Enum):
    """Prioridades de tarefas."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Status de execução de tarefas."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Representa uma tarefa para execução por agentes."""

    id: str
    description: str
    agent_type: str
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


class AdvancedAgentOrchestrator:
    """Orquestrador avançado com recursos empresariais."""

    def __init__(self):
        self.agents = {}
        self.task_queue = []
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.state_manager = StateManager()
        self.monitor = SystemMonitor()

        # Configurações avançadas
        self.max_concurrent_tasks = 5
        self.task_timeout = 300  # 5 minutos
        self.retry_attempts = 3
        self.circuit_breaker_threshold = 5

        # Métricas
        self.metrics = {
            "tasks_processed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "avg_processing_time": 0.0,
            "agent_utilization": {},
        }

        self._initialize_agents()

    def _initialize_agents(self):
        """Inicializar agentes especializados."""
        print("🤖 Inicializando agentes especializados...")

        try:
            # Agente de análise de dados
            self.agents["data_analyst"] = self._create_data_analyst_agent()

            # Agente de processamento de texto
            self.agents["text_processor"] = self._create_text_processor_agent()

            # Agente de comandos do sistema
            self.agents["system_commander"] = self._create_system_commander_agent()

            # Agente de síntese e relatórios
            self.agents["synthesizer"] = self._create_synthesizer_agent()

            print(f"✅ {len(self.agents)} agentes inicializados com sucesso")

        except Exception as e:
            print(f"❌ Erro ao inicializar agentes: {str(e)}")
            raise

    def _create_data_analyst_agent(self):
        """Criar agente especializado em análise de dados."""
        llm = get_chain_instance(
            "gpt4",
            """Você é um analista de dados especializado. Suas capacidades incluem:
            - Análise estatística de datasets
            - Identificação de padrões e tendências
            - Geração de insights acionáveis
            - Visualização de dados
            - Detecção de anomalias

            Sempre forneça análises precisas e baseadas em evidências.""",
        )

        # Configurar ferramentas específicas para análise
        agent = CommandAgent(llm)
        agent.specialized_domain = "data_analysis"
        agent.max_processing_time = 120

        return agent

    def _create_text_processor_agent(self):
        """Criar agente especializado em processamento de texto."""
        llm = get_chain_instance(
            "gpt4",
            """Você é um especialista em processamento de linguagem natural. Suas funções:
            - Análise de sentimentos
            - Extração de entidades
            - Classificação de texto
            - Sumarização inteligente
            - Tradução e localização

            Processe texto com alta precisão e contexto.""",
        )

        agent = AudioAgent(llm)  # Inclui capacidades de áudio
        agent.specialized_domain = "text_processing"
        agent.max_processing_time = 60

        return agent

    def _create_system_commander_agent(self):
        """Criar agente para comandos de sistema."""
        llm = get_chain_instance(
            "gpt4",
            """Você é um especialista em administração de sistemas. Responsabilidades:
            - Execução segura de comandos shell
            - Monitoramento de recursos do sistema
            - Automação de tarefas administrativas
            - Diagnóstico de problemas
            - Backup e manutenção

            Priorize sempre a segurança e estabilidade do sistema.""",
        )

        agent = CommandAgent(llm)
        agent.specialized_domain = "system_administration"
        agent.safety_mode = True
        agent.max_processing_time = 180

        return agent

    def _create_synthesizer_agent(self):
        """Criar agente para síntese e relatórios."""
        llm = get_chain_instance(
            "gpt4",
            """Você é um especialista em síntese de informações. Suas habilidades:
            - Consolidação de resultados de múltiplas fontes
            - Geração de relatórios executivos
            - Identificação de insights inter-domínios
            - Recomendações estratégicas
            - Comunicação clara e estruturada

            Crie resumos precisos e acionáveis.""",
        )

        chain = LangGraphChain("gpt4", llm.prompt_template.template)
        agent = type(
            "SynthesizerAgent",
            (),
            {
                "name": "synthesizer",
                "chain": chain,
                "specialized_domain": "synthesis_reporting",
                "invoke": lambda self, input_data: chain.invoke(input_data),
            },
        )()

        return agent

    async def submit_task(self, task: Task) -> str:
        """Submeter tarefa para execução."""
        print(f"📋 Submetendo tarefa: {task.description}")

        # Validar tarefa
        if not self._validate_task(task):
            raise ValueError(f"Tarefa inválida: {task.id}")

        # Verificar dependências
        if not self._check_dependencies(task):
            print(f"⏳ Tarefa {task.id} aguardando dependências")

        # Adicionar à fila
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)

        print(
            f"✅ Tarefa {task.id} adicionada à fila (posição: {len(self.task_queue)})"
        )
        return task.id

    async def execute_tasks(self):
        """Executar tarefas da fila com orquestração inteligente."""
        print("🚀 Iniciando execução de tarefas...")

        while self.task_queue or self.running_tasks:
            # Executar tarefas prontas
            await self._process_ready_tasks()

            # Verificar tarefas em execução
            await self._check_running_tasks()

            # Aguardar um pouco antes da próxima iteração
            await asyncio.sleep(0.1)

        print("✅ Todas as tarefas foram processadas")
        self._print_final_report()

    async def _process_ready_tasks(self):
        """Processar tarefas prontas para execução."""
        ready_tasks = [
            task
            for task in self.task_queue
            if (
                len(self.running_tasks) < self.max_concurrent_tasks
                and self._check_dependencies(task)
            )
        ]

        for task in ready_tasks[: self.max_concurrent_tasks - len(self.running_tasks)]:
            await self._execute_single_task(task)
            self.task_queue.remove(task)

    async def _execute_single_task(self, task: Task):
        """Executar uma única tarefa."""
        print(f"🔄 Executando tarefa {task.id}: {task.description}")

        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self.running_tasks[task.id] = task

        # Selecionar agente apropriado
        agent = self._select_agent_for_task(task)

        if not agent:
            task.status = TaskStatus.FAILED
            task.error = "Nenhum agente disponível para esta tarefa"
            return

        # Executar tarefa de forma assíncrona
        asyncio.create_task(self._run_task_with_agent(task, agent))

    async def _run_task_with_agent(self, task: Task, agent):
        """Executar tarefa com agente específico."""
        try:
            print(f"🤖 Agente {agent.name} processando tarefa {task.id}")

            # Preparar entrada para o agente
            agent_input = self._prepare_agent_input(task, agent)

            # Executar com timeout
            result = await asyncio.wait_for(
                self._invoke_agent(agent, agent_input), timeout=self.task_timeout
            )

            # Processar resultado
            task.result = self._process_agent_result(result, task)
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            print(f"✅ Tarefa {task.id} completada com sucesso")
            self.metrics["tasks_succeeded"] += 1

        except asyncio.TimeoutError:
            task.error = f"Timeout após {self.task_timeout} segundos"
            task.status = TaskStatus.FAILED
            print(f"⏰ Timeout na tarefa {task.id}")

        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            print(f"❌ Erro na tarefa {task.id}: {str(e)}")
            self.metrics["tasks_failed"] += 1

        finally:
            # Mover para lista apropriada
            self.running_tasks.pop(task.id, None)

            if task.status == TaskStatus.COMPLETED:
                self.completed_tasks[task.id] = task
            else:
                self.failed_tasks[task.id] = task

            self.metrics["tasks_processed"] += 1
            self._update_processing_time_metric(task)

    async def _invoke_agent(self, agent, input_data):
        """Invocar agente de forma assíncrona."""
        if hasattr(agent, "ainvoke"):
            return await agent.ainvoke(input_data)
        else:
            # Executar invoke síncrono em thread separada
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, agent.invoke, input_data)

    def _select_agent_for_task(self, task: Task):
        """Selecionar melhor agente para a tarefa."""
        # Mapear tipos de tarefa para agentes
        agent_mapping = {
            "data_analysis": "data_analyst",
            "text_processing": "text_processor",
            "system_command": "system_commander",
            "synthesis": "synthesizer",
        }

        agent_name = agent_mapping.get(task.agent_type)
        if agent_name and agent_name in self.agents:
            return self.agents[agent_name]

        # Fallback: usar agente mais apropriado baseado no contexto
        return self._intelligent_agent_selection(task)

    def _intelligent_agent_selection(self, task: Task):
        """Seleção inteligente de agente baseada no contexto."""
        description = task.description.lower()

        # Análise de keywords para seleção
        if any(
            word in description
            for word in ["analise", "dados", "estatistica", "grafico"]
        ):
            return self.agents.get("data_analyst")
        elif any(
            word in description
            for word in ["texto", "sentimento", "traducao", "resumo"]
        ):
            return self.agents.get("text_processor")
        elif any(
            word in description
            for word in ["comando", "sistema", "arquivo", "processo"]
        ):
            return self.agents.get("system_commander")
        else:
            return self.agents.get("synthesizer")

    def _prepare_agent_input(self, task: Task, agent) -> Dict[str, Any]:
        """Preparar entrada específica para o agente."""
        base_input = {
            "input": task.description,
            "task_id": task.id,
            "priority": task.priority.name,
            "metadata": task.metadata,
        }

        # Adicionar contexto específico do domínio
        if hasattr(agent, "specialized_domain"):
            base_input["domain"] = agent.specialized_domain

        # Adicionar resultados de dependências
        if task.dependencies:
            dependency_results = {}
            for dep_id in task.dependencies:
                if dep_id in self.completed_tasks:
                    dependency_results[dep_id] = self.completed_tasks[dep_id].result
            base_input["dependencies"] = dependency_results

        return base_input

    def _process_agent_result(self, result, task: Task) -> Dict[str, Any]:
        """Processar resultado do agente."""
        processed_result = {
            "output": result.get("output", ""),
            "agent_used": task.agent_type,
            "processing_time": (datetime.now() - task.started_at).total_seconds(),
            "metadata": result.get("metadata", {}),
            "raw_result": result,
        }

        return processed_result

    async def _check_running_tasks(self):
        """Verificar status de tarefas em execução."""
        current_time = datetime.now()

        for task_id, task in list(self.running_tasks.items()):
            # Verificar timeout
            if task.started_at:
                elapsed = (current_time - task.started_at).total_seconds()
                if elapsed > self.task_timeout:
                    print(f"⏰ Tarefa {task_id} excedeu timeout, cancelando...")
                    task.status = TaskStatus.CANCELLED
                    task.error = "Timeout excedido"
                    self.running_tasks.pop(task_id)
                    self.failed_tasks[task_id] = task

    def _validate_task(self, task: Task) -> bool:
        """Validar se tarefa está bem formada."""
        if not task.id or not task.description:
            return False

        if task.agent_type not in [
            "data_analysis",
            "text_processing",
            "system_command",
            "synthesis",
        ]:
            return False

        return True

    def _check_dependencies(self, task: Task) -> bool:
        """Verificar se dependências da tarefa foram completadas."""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True

    def _update_processing_time_metric(self, task: Task):
        """Atualizar métrica de tempo de processamento."""
        if task.started_at and task.completed_at:
            processing_time = (task.completed_at - task.started_at).total_seconds()

            current_avg = self.metrics["avg_processing_time"]
            total_tasks = self.metrics["tasks_processed"]

            # Calcular nova média
            self.metrics["avg_processing_time"] = (
                current_avg * (total_tasks - 1) + processing_time
            ) / total_tasks

    def _print_final_report(self):
        """Imprimir relatório final de execução."""
        print("\n" + "=" * 60)
        print("📊 RELATÓRIO FINAL DE EXECUÇÃO")
        print("=" * 60)

        print(f"✅ Tarefas completadas: {self.metrics['tasks_succeeded']}")
        print(f"❌ Tarefas falhadas: {self.metrics['tasks_failed']}")
        print(f"📈 Total processado: {self.metrics['tasks_processed']}")
        print(f"⚡ Tempo médio: {self.metrics['avg_processing_time']:.2f}s")

        # Taxa de sucesso
        if self.metrics["tasks_processed"] > 0:
            success_rate = (
                self.metrics["tasks_succeeded"] / self.metrics["tasks_processed"]
            ) * 100
            print(f"🎯 Taxa de sucesso: {success_rate:.1f}%")

        print("\n📋 DETALHES DAS TAREFAS:")

        # Tarefas completadas
        if self.completed_tasks:
            print("\n✅ COMPLETADAS:")
            for task_id, task in self.completed_tasks.items():
                duration = (task.completed_at - task.started_at).total_seconds()
                print(f"  • {task_id}: {task.description[:50]}... ({duration:.2f}s)")

        # Tarefas falhadas
        if self.failed_tasks:
            print("\n❌ FALHADAS:")
            for task_id, task in self.failed_tasks.items():
                print(f"  • {task_id}: {task.error}")

        print("=" * 60)


async def demonstrate_complex_workflow():
    """Demonstrar workflow complexo com múltiplos agentes."""
    print("🚀 Iniciando demonstração de workflow complexo...\n")

    orchestrator = AdvancedAgentOrchestrator()

    # Criar tarefas interdependentes
    tasks = [
        Task(
            id="task_001",
            description="Analisar dados de vendas do último trimestre e identificar tendências",
            agent_type="data_analysis",
            priority=TaskPriority.HIGH,
            metadata={"data_source": "sales_q3_2024.csv", "analysis_type": "trend"},
        ),
        Task(
            id="task_002",
            description="Processar feedback dos clientes e extrair sentimentos principais",
            agent_type="text_processing",
            priority=TaskPriority.HIGH,
            metadata={"source": "customer_reviews.json", "languages": ["pt", "en"]},
        ),
        Task(
            id="task_003",
            description="Verificar status do sistema e gerar relatório de saúde",
            agent_type="system_command",
            priority=TaskPriority.NORMAL,
            metadata={"check_services": ["database", "api", "cache"]},
        ),
        Task(
            id="task_004",
            description="Consolidar análise de vendas, feedback de clientes e status do sistema em relatório executivo",
            agent_type="synthesis",
            priority=TaskPriority.CRITICAL,
            dependencies=["task_001", "task_002", "task_003"],
            metadata={"report_type": "executive_summary", "format": "pdf"},
        ),
        Task(
            id="task_005",
            description="Traduzir relatório executivo para inglês e espanhol",
            agent_type="text_processing",
            priority=TaskPriority.LOW,
            dependencies=["task_004"],
            metadata={"target_languages": ["en", "es"]},
        ),
    ]

    # Submeter tarefas
    print("📋 Submetendo tarefas...")
    for task in tasks:
        await orchestrator.submit_task(task)

    print(f"\n🔄 Executando {len(tasks)} tarefas com orquestração inteligente...")

    # Executar workflow
    start_time = time.time()
    await orchestrator.execute_tasks()
    total_time = time.time() - start_time

    print(f"\n⏱️  Tempo total de execução: {total_time:.2f} segundos")

    # Demonstrar resultados
    print("\n🎯 RESULTADOS FINAIS:")
    for task_id, task in orchestrator.completed_tasks.items():
        if task.result:
            print(f"\n📄 {task_id}:")
            print(f"   Output: {task.result['output'][:100]}...")
            print(f"   Agente: {task.result['agent_used']}")
            print(f"   Tempo: {task.result['processing_time']:.2f}s")


async def demonstrate_parallel_processing():
    """Demonstrar processamento paralelo de tarefas independentes."""
    print("\n🔄 Demonstrando processamento paralelo...\n")

    orchestrator = AdvancedAgentOrchestrator()

    # Criar tarefas paralelas independentes
    parallel_tasks = []
    for i in range(8):
        task = Task(
            id=f"parallel_task_{i:03d}",
            description=f"Processar dataset #{i+1} para análise estatística",
            agent_type="data_analysis",
            priority=TaskPriority.NORMAL,
            metadata={
                "dataset_id": i + 1,
                "operations": ["mean", "std", "correlation"],
            },
        )
        parallel_tasks.append(task)

    # Submeter todas as tarefas
    for task in parallel_tasks:
        await orchestrator.submit_task(task)

    print(f"🚀 Processando {len(parallel_tasks)} tarefas em paralelo...")

    start_time = time.time()
    await orchestrator.execute_tasks()
    parallel_time = time.time() - start_time

    print(f"⚡ Processamento paralelo concluído em {parallel_time:.2f} segundos")
    print(f"💡 Com {orchestrator.max_concurrent_tasks} agentes concorrentes")


async def demonstrate_error_handling():
    """Demonstrar tratamento de erros e recovery."""
    print("\n🛡️  Demonstrando tratamento de erros...\n")

    orchestrator = AdvancedAgentOrchestrator()

    # Criar tarefas que podem falhar
    error_tasks = [
        Task(
            id="error_task_001",
            description="Executar comando inexistente que vai falhar",
            agent_type="system_command",
            priority=TaskPriority.HIGH,
            metadata={"command": "comando_que_nao_existe"},
        ),
        Task(
            id="error_task_002",
            description="Analisar arquivo que não existe",
            agent_type="data_analysis",
            priority=TaskPriority.NORMAL,
            metadata={"file_path": "/arquivo/inexistente.csv"},
        ),
        Task(
            id="recovery_task_001",
            description="Esta tarefa deve ser executada normalmente",
            agent_type="text_processing",
            priority=TaskPriority.HIGH,
            metadata={"text": "Este é um texto normal para processamento"},
        ),
    ]

    for task in error_tasks:
        await orchestrator.submit_task(task)

    await orchestrator.execute_tasks()

    print("\n📊 Resultado do teste de erro:")
    print(f"✅ Sucessos: {orchestrator.metrics['tasks_succeeded']}")
    print(f"❌ Falhas: {orchestrator.metrics['tasks_failed']}")


def main():
    """Função principal para demonstração completa."""
    print("🎭 DEMONSTRAÇÃO AVANÇADA DE ORQUESTRAÇÃO MULTI-AGENTE")
    print("=" * 60)

    async def run_all_demonstrations():
        # Workflow complexo com dependências
        await demonstrate_complex_workflow()

        await asyncio.sleep(1)

        # Processamento paralelo
        await demonstrate_parallel_processing()

        await asyncio.sleep(1)

        # Tratamento de erros
        await demonstrate_error_handling()

        print("\n🎉 Demonstração completa finalizada!")
        print("💡 Este exemplo mostrou:")
        print("   • Coordenação inteligente de agentes")
        print("   • Execução paralela e sequencial")
        print("   • Tratamento de dependências")
        print("   • Recovery automático de erros")
        print("   • Monitoramento e métricas")

    # Executar todas as demonstrações
    asyncio.run(run_all_demonstrations())


if __name__ == "__main__":
    main()
