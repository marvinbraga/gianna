"""
Intelligent Agent Router for Gianna Multi-Agent System

This module implements intelligent routing logic that analyzes Portuguese text
to determine the most appropriate agent for handling specific requests.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

from ..core.state import GiannaState


class AgentType(Enum):
    """Enumeration of available agent types."""

    COMMAND = "command_agent"
    AUDIO = "audio_agent"
    CONVERSATION = "conversation_agent"
    MEMORY = "memory_agent"


@dataclass
class RoutingRule:
    """Represents a routing rule with keywords and priority."""

    agent_type: AgentType
    keywords: List[str]
    patterns: List[str]
    priority: int = 1
    confidence_threshold: float = 0.6


class AgentRouter:
    """
    Intelligent router that determines the best agent for handling requests.

    Uses Portuguese keyword analysis, pattern matching, and contextual
    information to make routing decisions with confidence scoring.
    """

    def __init__(self):
        """Initialize the router with Portuguese routing rules."""
        self.routing_rules = self._build_routing_rules()
        self.routing_history = []

    def _build_routing_rules(self) -> List[RoutingRule]:
        """
        Build comprehensive routing rules for Portuguese language.

        Returns:
            List[RoutingRule]: Complete set of routing rules
        """
        rules = [
            # Command Agent Rules (High Priority)
            RoutingRule(
                agent_type=AgentType.COMMAND,
                keywords=[
                    "comando",
                    "executar",
                    "rodar",
                    "run",
                    "shell",
                    "bash",
                    "terminal",
                    "console",
                    "script",
                    "instalar",
                    "install",
                    "sudo",
                    "chmod",
                    "mkdir",
                    "ls",
                    "cd",
                    "cp",
                    "mv",
                    "rm",
                    "cat",
                    "grep",
                    "find",
                    "wget",
                    "curl",
                    "git",
                    "docker",
                    "npm",
                    "pip",
                    "python",
                    "node",
                    "java",
                    "gcc",
                    "make",
                    "systemctl",
                    "service",
                    "processar",
                    "processo",
                    "kill",
                    "arquivo",
                    "diretório",
                    "pasta",
                    "criar",
                    "deletar",
                    "copiar",
                    "mover",
                    "renomear",
                    "permissão",
                    "configurar",
                    "compilar",
                    "build",
                    "deploy",
                ],
                patterns=[
                    r"\b(execute|run|rodar)\s+['\"]?[\w\-\.\/]+",
                    r"\bsudo\s+\w+",
                    r"\b(cd|ls|mkdir|rm|cp|mv)\s+[\w\-\.\/]+",
                    r"\bgit\s+(clone|pull|push|commit|status|add)",
                    r"\b(npm|pip|apt|yum)\s+(install|update|remove)",
                    r"executar\s+o\s+comando",
                    r"rodar\s+(o\s+)?(script|programa)",
                    r"abrir\s+(o\s+)?terminal",
                    r"no\s+terminal",
                ],
                priority=3,
                confidence_threshold=0.15,
            ),
            # Audio Agent Rules (High Priority)
            RoutingRule(
                agent_type=AgentType.AUDIO,
                keywords=[
                    "falar",
                    "áudio",
                    "voz",
                    "ouvir",
                    "escutar",
                    "som",
                    "reproduzir",
                    "tocar",
                    "pausar",
                    "parar",
                    "volume",
                    "música",
                    "audio",
                    "sound",
                    "play",
                    "speak",
                    "voice",
                    "tts",
                    "stt",
                    "reconhecimento",
                    "síntese",
                    "microfone",
                    "alto-falante",
                    "headphone",
                    "fone",
                    "gravar",
                    "gravação",
                    "recording",
                    "record",
                    "wav",
                    "mp3",
                    "ogg",
                    "flac",
                    "dizer",
                    "pronunciar",
                    "narrar",
                    "ler",
                    "leitura",
                    "velocidade",
                    "tom",
                    "entonação",
                    "idioma",
                    "language",
                    "português",
                    "english",
                    "español",
                    "silêncio",
                    "mudo",
                ],
                patterns=[
                    r"\bfalar\s+(sobre|com|para|em)",
                    r"\bouvir\s+(música|áudio|som)",
                    r"\btocar\s+(música|áudio|arquivo)",
                    r"\bgravar\s+(áudio|voz|som)",
                    r"\bvolume\s+(alto|baixo|médio|\d+)",
                    r"\bvoz\s+(masculina|feminina|robótica)",
                    r"\bidioma\s+(português|inglês|espanhol)",
                    r"reproduzir\s+o\s+áudio",
                    r"me\s+conte\s+sobre",
                    r"leia\s+(para\s+mim|em\s+voz\s+alta)",
                ],
                priority=3,
                confidence_threshold=0.15,
            ),
            # Memory Agent Rules (Medium Priority)
            RoutingRule(
                agent_type=AgentType.MEMORY,
                keywords=[
                    "lembrar",
                    "memória",
                    "histórico",
                    "contexto",
                    "salvar",
                    "guardar",
                    "armazenar",
                    "recuperar",
                    "buscar",
                    "procurar",
                    "anterior",
                    "antes",
                    "passado",
                    "ontem",
                    "semana",
                    "mês",
                    "ano",
                    "data",
                    "quando",
                    "onde",
                    "como",
                    "preferência",
                    "configuração",
                    "setting",
                    "profile",
                    "perfil",
                    "usuário",
                    "user",
                    "personalizar",
                    "customizar",
                    "nota",
                    "anotação",
                    "observação",
                    "importante",
                    "relevante",
                    "esquecer",
                    "apagar",
                    "deletar",
                    "limpar",
                    "reset",
                    "backup",
                    "restore",
                    "exportar",
                    "importar",
                    "sincronizar",
                ],
                patterns=[
                    r"\blembrar\s+(de|que|sobre)",
                    r"\bmemória\s+(de|sobre|do)",
                    r"\bhistórico\s+(de|da|do)",
                    r"\bcontexto\s+(anterior|passado|da\s+conversa)",
                    r"\bsalvar\s+(nas?\s+)?(preferência|memória|histórico)",
                    r"\bguardar\s+(na\s+)?memória",
                    r"\bbuscar\s+(no\s+)?(histórico|memória|contexto)",
                    r"você\s+(lembra|sabe)\s+(de|sobre|que)",
                    r"o\s+que\s+(conversamos|falamos)\s+(sobre|antes)",
                    r"configurar\s+(preferência|perfil|setting)",
                ],
                priority=2,
                confidence_threshold=0.1,
            ),
            # Conversation Agent Rules (Default/Low Priority)
            RoutingRule(
                agent_type=AgentType.CONVERSATION,
                keywords=[
                    "conversar",
                    "conversa",
                    "chat",
                    "falar",
                    "dizer",
                    "pergunta",
                    "resposta",
                    "questão",
                    "dúvida",
                    "ajuda",
                    "explicar",
                    "entender",
                    "compreender",
                    "esclarecer",
                    "orientar",
                    "instruir",
                    "ensinar",
                    "aprender",
                    "saber",
                    "informação",
                    "conhecimento",
                    "curiosidade",
                    "interessante",
                    "legal",
                    "bacana",
                    "ótimo",
                    "perfeito",
                    "excelente",
                    "obrigado",
                    "valeu",
                    "agradecer",
                    "parabéns",
                    "felicitar",
                    "opiniões",
                    "sugestão",
                    "conselho",
                    "dica",
                    "recomendação",
                    "história",
                    "contar",
                    "narrar",
                    "descrever",
                    "relatar",
                ],
                patterns=[
                    r"\bme\s+(explique|conte|diga|fale)\s+(sobre|como|porque)",
                    r"\bo\s+que\s+(é|significa|representa)",
                    r"\bcomo\s+(funciona|fazer|usar|configurar)",
                    r"\bpor\s+que\s+(isso|acontece|funciona)",
                    r"\bqual\s+(é|seria|foi|será)\s+a",
                    r"\bobrigado\s+(pela?\s+)?(ajuda|explicação|resposta)",
                    r"\bme\s+ajude\s+(com|a|na)",
                    r"\btenho\s+uma\s+(dúvida|pergunta|questão)",
                    r"\bvocê\s+(pode|poderia|consegue|sabe)",
                    r"\bestou\s+(com\s+)?(dificuldade|problema|dúvida)",
                ],
                priority=1,
                confidence_threshold=0.05,
            ),
        ]

        logger.info(f"Built {len(rules)} routing rules for Portuguese language")
        return rules

    def route_request(self, state: GiannaState) -> Tuple[AgentType, float]:
        """
        Route a request to the most appropriate agent.

        Args:
            state: Current Gianna system state

        Returns:
            Tuple[AgentType, float]: Selected agent type and confidence score
        """
        if not state["conversation"].messages:
            return AgentType.CONVERSATION, 0.5

        last_message = state["conversation"].messages[-1]
        if last_message.get("role") != "user":
            return AgentType.CONVERSATION, 0.5

        content = last_message.get("content", "").lower().strip()
        if not content:
            return AgentType.CONVERSATION, 0.5

        # Calculate scores for each agent type
        agent_scores = self._calculate_routing_scores(content, state)

        # Select the best agent based on scores and context
        selected_agent, confidence = self._select_best_agent(agent_scores, state)

        # Log routing decision
        self._log_routing_decision(content, selected_agent, confidence, agent_scores)

        # Update routing history
        self.routing_history.append(
            {
                "message": content[:100],  # Truncate for privacy
                "selected_agent": selected_agent.value,
                "confidence": confidence,
                "all_scores": {k.value: v for k, v in agent_scores.items()},
            }
        )

        return selected_agent, confidence

    def _calculate_routing_scores(
        self, content: str, state: GiannaState
    ) -> Dict[AgentType, float]:
        """
        Calculate routing scores for all agent types.

        Args:
            content: Message content to analyze
            state: Current system state for context

        Returns:
            Dict[AgentType, float]: Scores for each agent type
        """
        scores = {agent_type: 0.0 for agent_type in AgentType}

        for rule in self.routing_rules:
            score = self._evaluate_rule(rule, content, state)
            scores[rule.agent_type] += score * rule.priority

        # Apply contextual adjustments
        scores = self._apply_contextual_adjustments(scores, content, state)

        # Normalize scores
        max_possible_score = max(rule.priority for rule in self.routing_rules) * 3
        for agent_type in scores:
            scores[agent_type] = min(1.0, scores[agent_type] / max_possible_score)

        return scores

    def _evaluate_rule(
        self, rule: RoutingRule, content: str, state: GiannaState
    ) -> float:
        """
        Evaluate a single routing rule against the content.

        Args:
            rule: Routing rule to evaluate
            content: Message content
            state: Current system state

        Returns:
            float: Rule evaluation score
        """
        score = 0.0

        # Keyword matching
        keyword_matches = sum(1 for keyword in rule.keywords if keyword in content)
        keyword_score = min(1.0, keyword_matches / max(1, len(rule.keywords) * 0.1))

        # Pattern matching
        pattern_matches = 0
        for pattern in rule.patterns:
            if re.search(pattern, content, re.IGNORECASE):
                pattern_matches += 1

        pattern_score = min(1.0, pattern_matches / max(1, len(rule.patterns)))

        # Combine scores
        score = (keyword_score * 0.6) + (pattern_score * 0.4)

        return score if score >= rule.confidence_threshold else 0.0

    def _apply_contextual_adjustments(
        self, scores: Dict[AgentType, float], content: str, state: GiannaState
    ) -> Dict[AgentType, float]:
        """
        Apply contextual adjustments based on conversation history and state.

        Args:
            scores: Current routing scores
            content: Message content
            state: Current system state

        Returns:
            Dict[AgentType, float]: Adjusted scores
        """
        adjusted_scores = scores.copy()

        # Recent agent usage - slight preference for continuity
        if self.routing_history:
            last_agent = AgentType(self.routing_history[-1]["selected_agent"])
            if adjusted_scores[last_agent] > 0.3:  # Only if already reasonably relevant
                adjusted_scores[last_agent] *= 1.1

        # Audio mode context
        if state["audio"].current_mode in ["listening", "speaking"]:
            adjusted_scores[AgentType.AUDIO] *= 1.2

        # Command history context
        recent_commands = state["commands"].execution_history[-3:]
        if recent_commands and any(
            "error" in str(cmd).lower() for cmd in recent_commands
        ):
            # If recent commands had errors, boost command agent for follow-ups
            if any(
                word in content
                for word in ["erro", "problema", "não", "funciona", "fix"]
            ):
                adjusted_scores[AgentType.COMMAND] *= 1.3

        # Memory-related follow-up detection
        user_prefs = state["conversation"].user_preferences
        if user_prefs and any(
            word in content
            for word in ["mudança", "alterar", "configurar", "preferência"]
        ):
            adjusted_scores[AgentType.MEMORY] *= 1.2

        return adjusted_scores

    def _select_best_agent(
        self, scores: Dict[AgentType, float], state: GiannaState
    ) -> Tuple[AgentType, float]:
        """
        Select the best agent based on scores and system constraints.

        Args:
            scores: Routing scores for each agent
            state: Current system state

        Returns:
            Tuple[AgentType, float]: Selected agent and confidence score
        """
        # Find the highest scoring agent
        best_agent = max(scores, key=scores.get)
        best_score = scores[best_agent]

        # Minimum confidence check
        if best_score < 0.05:
            # Fall back to conversation agent for low confidence
            return AgentType.CONVERSATION, 0.05

        # If multiple agents have similar high scores, apply tie-breaking rules
        high_scores = {k: v for k, v in scores.items() if v > best_score * 0.9}

        if len(high_scores) > 1:
            # Tie-breaking priority: Command > Audio > Memory > Conversation
            priority_order = [
                AgentType.COMMAND,
                AgentType.AUDIO,
                AgentType.MEMORY,
                AgentType.CONVERSATION,
            ]

            for agent_type in priority_order:
                if agent_type in high_scores:
                    return agent_type, high_scores[agent_type]

        return best_agent, best_score

    def _log_routing_decision(
        self,
        content: str,
        selected_agent: AgentType,
        confidence: float,
        all_scores: Dict[AgentType, float],
    ) -> None:
        """
        Log the routing decision for debugging and monitoring.

        Args:
            content: Message content (truncated)
            selected_agent: Selected agent
            confidence: Confidence score
            all_scores: All calculated scores
        """
        content_preview = content[:50] + "..." if len(content) > 50 else content

        logger.info(
            f"Routing decision: '{content_preview}' -> {selected_agent.value} "
            f"(confidence: {confidence:.2f})"
        )

        # Log detailed scores in debug mode
        score_details = ", ".join(
            [f"{agent.value}: {score:.2f}" for agent, score in all_scores.items()]
        )
        logger.debug(f"All routing scores: {score_details}")

    def get_routing_stats(self) -> Dict[str, any]:
        """
        Get routing statistics and performance metrics.

        Returns:
            Dict[str, any]: Routing statistics
        """
        if not self.routing_history:
            return {"total_routes": 0, "agent_distribution": {}, "avg_confidence": 0.0}

        total_routes = len(self.routing_history)
        agent_counts = {}
        total_confidence = 0.0

        for route in self.routing_history:
            agent = route["selected_agent"]
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            total_confidence += route["confidence"]

        agent_distribution = {
            agent: count / total_routes for agent, count in agent_counts.items()
        }

        return {
            "total_routes": total_routes,
            "agent_distribution": agent_distribution,
            "avg_confidence": total_confidence / total_routes,
            "recent_routes": self.routing_history[-10:],  # Last 10 for analysis
        }

    def clear_routing_history(self) -> None:
        """Clear routing history (useful for testing or privacy)."""
        self.routing_history.clear()
        logger.info("Routing history cleared")
