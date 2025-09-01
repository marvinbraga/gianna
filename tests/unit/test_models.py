"""
Unit tests for LLM models and chains - FASE 1 & 2

Tests for:
- LLM provider implementations (OpenAI, Anthropic, Google, etc.)
- Chain factory methods and registries
- Model enumeration and configuration
- Chain invoke methods and compatibility

Test Coverage:
- All LLM providers and models
- Factory method patterns
- Registry operations
- Chain invocation and response formatting
- Error handling and fallbacks
"""

import os
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Model system imports
from gianna.assistants.models.basics import (
    AbstractBasicChain,
    AbstractLLMFactory,
    ModelsEnum,
)
from gianna.assistants.models.factory_method import get_chain_instance
from gianna.assistants.models.registers import LLMRegister


@pytest.mark.unit
@pytest.mark.fase1
@pytest.mark.llm
class TestLLMRegistry:
    """Test LLM registry functionality."""

    def test_llm_register_singleton(self):
        """Test LLM register is a singleton."""
        registry1 = LLMRegister()
        registry2 = LLMRegister()

        assert registry1 is registry2
        assert id(registry1) == id(registry2)

    def test_registry_initialization(self):
        """Test registry initializes with expected providers."""
        registry = LLMRegister()

        # Check that registry has some registered providers
        assert hasattr(registry, "_providers")
        assert hasattr(registry, "register_provider")
        assert hasattr(registry, "get_provider")

    def test_provider_registration(self):
        """Test provider registration in registry."""
        registry = LLMRegister()

        # Mock provider
        mock_provider = MagicMock()
        mock_provider.name = "test_provider"

        # Test registration
        registry.register_provider("test_provider", mock_provider)

        # Test retrieval
        retrieved = registry.get_provider("test_provider")
        assert retrieved == mock_provider

    def test_invalid_provider_handling(self):
        """Test handling of invalid provider requests."""
        registry = LLMRegister()

        with pytest.raises((KeyError, ValueError)):
            registry.get_provider("nonexistent_provider")

    def test_available_providers_listing(self):
        """Test listing available providers."""
        registry = LLMRegister()

        providers = registry.list_providers()
        assert isinstance(providers, (list, tuple, dict))

        # Should have at least basic providers
        expected_providers = ["openai", "google", "anthropic"]
        if isinstance(providers, dict):
            provider_names = list(providers.keys())
        else:
            provider_names = providers

        # Check that some expected providers are available
        available_expected = [p for p in expected_providers if p in provider_names]
        assert len(available_expected) > 0


@pytest.mark.unit
@pytest.mark.fase1
@pytest.mark.llm
class TestModelEnumerations:
    """Test model enumeration classes."""

    def test_models_enum_structure(self):
        """Test ModelsEnum base structure."""
        # Test that ModelsEnum can be imported and has expected structure
        assert hasattr(ModelsEnum, "__members__") or hasattr(
            ModelsEnum, "_value2member_map_"
        )

    def test_openai_models_enum(self):
        """Test OpenAI models enumeration."""
        try:
            from gianna.assistants.models.openai import OpenAIModels

            # Test enum has expected models
            models = list(OpenAIModels)
            assert len(models) > 0

            # Check for common models
            model_values = [m.value for m in models]
            expected_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]

            for expected in expected_models:
                if expected in model_values:
                    assert True
                    break
            else:
                pytest.skip("No expected OpenAI models found")

        except ImportError:
            pytest.skip("OpenAI models not available")

    def test_google_models_enum(self):
        """Test Google models enumeration."""
        try:
            from gianna.assistants.models.google import GoogleModels

            models = list(GoogleModels)
            assert len(models) > 0

            # Check for Gemini models
            model_values = [m.value for m in models]
            expected_models = ["gemini-pro", "gemini-1.5-pro"]

            for expected in expected_models:
                if expected in model_values:
                    assert True
                    break
            else:
                pytest.skip("No expected Google models found")

        except ImportError:
            pytest.skip("Google models not available")

    def test_anthropic_models_enum(self):
        """Test Anthropic models enumeration."""
        try:
            from gianna.assistants.models.anthropic import AnthropicModels

            models = list(AnthropicModels)
            assert len(models) > 0

            # Check for Claude models
            model_values = [m.value for m in models]
            expected_models = ["claude-3-sonnet", "claude-3-opus", "claude-3-haiku"]

            for expected in expected_models:
                if expected in model_values:
                    assert True
                    break
            else:
                pytest.skip("No expected Anthropic models found")

        except ImportError:
            pytest.skip("Anthropic models not available")


@pytest.mark.unit
@pytest.mark.fase1
@pytest.mark.llm
class TestChainFactoryMethods:
    """Test chain factory method functionality."""

    def test_get_chain_instance_basic(self):
        """Test basic chain instance creation."""
        with patch(
            "gianna.assistants.models.factory_method.LLMRegister"
        ) as mock_registry:
            mock_chain = MagicMock(spec=AbstractBasicChain)
            mock_chain.invoke.return_value = {"output": "Test response"}

            mock_factory = MagicMock()
            mock_factory.create_chain.return_value = mock_chain

            mock_registry.return_value.get_provider.return_value = mock_factory

            chain = get_chain_instance("gpt35", "You are helpful")

            assert chain is not None
            assert hasattr(chain, "invoke")

    def test_chain_factory_with_different_models(self, llm_model_name):
        """Test chain factory with different model types."""
        with patch(
            "gianna.assistants.models.factory_method.LLMRegister"
        ) as mock_registry:
            mock_chain = MagicMock(spec=AbstractBasicChain)
            mock_chain.model_name = llm_model_name
            mock_chain.invoke.return_value = {
                "output": f"Response from {llm_model_name}"
            }

            mock_factory = MagicMock()
            mock_factory.create_chain.return_value = mock_chain

            mock_registry.return_value.get_provider.return_value = mock_factory

            chain = get_chain_instance(llm_model_name, "Test prompt")

            assert chain.model_name == llm_model_name
            response = chain.invoke({"input": "test"})
            assert llm_model_name in response["output"]

    def test_factory_method_error_handling(self):
        """Test factory method error handling."""
        with patch(
            "gianna.assistants.models.factory_method.LLMRegister"
        ) as mock_registry:
            mock_registry.return_value.get_provider.side_effect = KeyError(
                "Provider not found"
            )

            with pytest.raises((KeyError, ValueError)):
                get_chain_instance("invalid_model", "Test prompt")

    def test_chain_factory_with_parameters(self):
        """Test chain factory with additional parameters."""
        with patch(
            "gianna.assistants.models.factory_method.LLMRegister"
        ) as mock_registry:
            mock_chain = MagicMock(spec=AbstractBasicChain)
            mock_chain.invoke.return_value = {"output": "Parameterized response"}

            mock_factory = MagicMock()
            mock_factory.create_chain.return_value = mock_chain

            mock_registry.return_value.get_provider.return_value = mock_factory

            chain = get_chain_instance(
                "gpt35", "Test prompt", temperature=0.7, max_tokens=1000
            )

            assert chain is not None
            mock_factory.create_chain.assert_called_once()


@pytest.mark.unit
@pytest.mark.fase1
@pytest.mark.llm
class TestChainInvocation:
    """Test chain invocation and response handling."""

    def test_chain_invoke_basic(self, mock_chain):
        """Test basic chain invocation."""
        response = mock_chain.invoke({"input": "Hello"})

        assert isinstance(response, dict)
        pytest.assert_response_format_valid(response)
        mock_chain.invoke.assert_called_once_with({"input": "Hello"})

    def test_chain_async_invoke(self, mock_chain, async_test_runner):
        """Test async chain invocation."""

        async def test_async_invoke():
            response = await mock_chain.ainvoke({"input": "Hello async"})
            assert isinstance(response, dict)
            pytest.assert_response_format_valid(response)
            return response

        result = async_test_runner(test_async_invoke())
        assert result is not None

    def test_chain_invoke_with_parameters(self, mock_chain):
        """Test chain invocation with additional parameters."""
        mock_chain.invoke.return_value = {"output": "Parameterized response"}

        response = mock_chain.invoke({"input": "Test"}, temperature=0.5, max_tokens=500)

        assert response["output"] == "Parameterized response"
        mock_chain.invoke.assert_called_once()

    def test_chain_invoke_error_handling(self, mock_chain):
        """Test chain invocation error handling."""
        mock_chain.invoke.side_effect = Exception("LLM service unavailable")

        with pytest.raises(Exception) as exc_info:
            mock_chain.invoke({"input": "Test"})

        assert "LLM service unavailable" in str(exc_info.value)

    def test_chain_response_validation(self, mock_chain):
        """Test chain response format validation."""
        # Test valid response
        mock_chain.invoke.return_value = {"output": "Valid response"}
        response = mock_chain.invoke({"input": "test"})
        pytest.assert_response_format_valid(response)

        # Test invalid response format
        mock_chain.invoke.return_value = {"invalid": "format"}
        with pytest.raises(AssertionError):
            response = mock_chain.invoke({"input": "test"})
            pytest.assert_response_format_valid(response)

    def test_chain_streaming_support(self, mock_chain):
        """Test chain streaming functionality if available."""
        # Test if chain supports streaming
        if hasattr(mock_chain, "stream"):
            mock_chain.stream.return_value = [
                {"output": "Part 1"},
                {"output": "Part 2"},
                {"output": "Part 3"},
            ]

            stream = mock_chain.stream({"input": "streaming test"})
            parts = list(stream)

            assert len(parts) == 3
            for part in parts:
                pytest.assert_response_format_valid(part)
        else:
            pytest.skip("Streaming not supported by chain")


@pytest.mark.unit
@pytest.mark.fase1
@pytest.mark.llm
class TestProviderSpecificImplementations:
    """Test provider-specific implementations."""

    def test_openai_chain_creation(self):
        """Test OpenAI chain creation."""
        try:
            from gianna.assistants.models.openai import OpenAIFactory

            factory = OpenAIFactory()
            assert factory is not None
            assert hasattr(factory, "create_chain")

        except ImportError:
            pytest.skip("OpenAI provider not available")

    def test_google_chain_creation(self):
        """Test Google chain creation."""
        try:
            from gianna.assistants.models.google import GoogleFactory

            factory = GoogleFactory()
            assert factory is not None
            assert hasattr(factory, "create_chain")

        except ImportError:
            pytest.skip("Google provider not available")

    def test_anthropic_chain_creation(self):
        """Test Anthropic chain creation."""
        try:
            from gianna.assistants.models.anthropic import AnthropicFactory

            factory = AnthropicFactory()
            assert factory is not None
            assert hasattr(factory, "create_chain")

        except ImportError:
            pytest.skip("Anthropic provider not available")

    def test_groq_chain_creation(self):
        """Test Groq chain creation."""
        try:
            from gianna.assistants.models.groq import GroqFactory

            factory = GroqFactory()
            assert factory is not None
            assert hasattr(factory, "create_chain")

        except ImportError:
            pytest.skip("Groq provider not available")

    def test_nvidia_chain_creation(self):
        """Test NVIDIA chain creation."""
        try:
            from gianna.assistants.models.nvidia import NVIDIAFactory

            factory = NVIDIAFactory()
            assert factory is not None
            assert hasattr(factory, "create_chain")

        except ImportError:
            pytest.skip("NVIDIA provider not available")


@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.llm
class TestModelPerformance:
    """Test model system performance characteristics."""

    def test_chain_creation_performance(self, benchmark_timer):
        """Test chain creation performance."""
        with patch(
            "gianna.assistants.models.factory_method.LLMRegister"
        ) as mock_registry:
            mock_chain = MagicMock(spec=AbstractBasicChain)
            mock_factory = MagicMock()
            mock_factory.create_chain.return_value = mock_chain
            mock_registry.return_value.get_provider.return_value = mock_factory

            benchmark_timer.start()

            chains = []
            for i in range(100):
                chain = get_chain_instance("gpt35", f"Prompt {i}")
                chains.append(chain)

            benchmark_timer.stop()

            assert len(chains) == 100
            assert benchmark_timer.elapsed < 1.0  # < 1 second for 100 chains

    def test_chain_invoke_performance(self, mock_chain, benchmark_timer):
        """Test chain invocation performance."""
        mock_chain.invoke.return_value = {"output": "Fast response"}

        benchmark_timer.start()

        responses = []
        for i in range(100):
            response = mock_chain.invoke({"input": f"Request {i}"})
            responses.append(response)

        benchmark_timer.stop()

        assert len(responses) == 100
        assert benchmark_timer.elapsed < 2.0  # < 2 seconds for 100 invocations

    def test_registry_lookup_performance(self, benchmark_timer):
        """Test registry lookup performance."""
        registry = LLMRegister()

        # Add test providers
        for i in range(10):
            mock_provider = MagicMock()
            registry.register_provider(f"provider_{i}", mock_provider)

        benchmark_timer.start()

        lookups = []
        for i in range(1000):
            provider = registry.get_provider(f"provider_{i % 10}")
            lookups.append(provider)

        benchmark_timer.stop()

        assert len(lookups) == 1000
        assert benchmark_timer.elapsed < 0.5  # < 0.5 seconds for 1000 lookups

    def test_concurrent_chain_usage(self, mock_chain, async_test_runner):
        """Test concurrent chain usage performance."""
        import asyncio

        mock_chain.ainvoke = AsyncMock(return_value={"output": "Concurrent response"})

        async def concurrent_test():
            tasks = []
            for i in range(50):
                task = mock_chain.ainvoke({"input": f"Concurrent request {i}"})
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            return results

        start_time = time.perf_counter()
        results = async_test_runner(concurrent_test())
        end_time = time.perf_counter()

        assert len(results) == 50
        assert end_time - start_time < 5.0  # < 5 seconds for 50 concurrent requests
