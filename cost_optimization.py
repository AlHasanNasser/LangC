# Token Budgeting and Cost Optimization in LangChain

import hashlib
import json
import os
from typing import Optional, Callable
from functools import lru_cache
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable
from dotenv import load_dotenv


load_dotenv()


def _has_google_credentials() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


class _InMemorySemanticCache:
    """Simple local cache used when no external semantic cache backend is configured."""

    def __init__(self):
        self._store = {}

    def _normalize(self, key: str) -> str:
        return key.strip().lower()

    def get(self, key: str):
        return self._store.get(self._normalize(key))

    def set(self, key: str, value):
        self._store[self._normalize(key)] = value


class ModelRouter:
    """Route queries to appropriate model based on complexity."""

    def __init__(self):
        api_key_available = _has_google_credentials()
        self.cheap_model = (
            GoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
            if api_key_available
            else None
        )
        self.expensive_model = (
            GoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
            if api_key_available
            else None
        )
        self.classifier = (
            GoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
            if api_key_available
            else None
        )

    def classify_complexity(self, query: str) -> str:
        """Classify query complexity."""
        if self.classifier is None:
            normalized = query.lower()
            simple_keywords = [
                "what is",
                "what color",
                "who is",
                "when",
                "where",
                "which",
                "2 + 2",
                "calculate",
                "define",
                "sky",
            ]
            if any(keyword in normalized for keyword in simple_keywords):
                return "simple"
            return "complex"

        prompt = ChatPromptTemplate.from_template(
            """
Classify this query's complexity as 'simple' or 'complex'.

Simple: Basic facts, short answers, simple calculations
Complex: Analysis, reasoning, creative tasks, multi-step problems

Query: {query}

Respond with only: simple or complex
"""
        )

        response = self.classifier.invoke(prompt.format(query=query))
        response_text = response.content if hasattr(response, "content") else response
        return response_text.strip().lower()

    @traceable(name="routed_query")
    def invoke(self, query: str) -> tuple[str, str, float]:
        """
        Route and invoke query.
        Returns: (response, model_used, estimated_cost)
        """
        complexity = self.classify_complexity(query)

        if complexity == "simple":
            model = self.cheap_model
            model_name = "gemini-3.1-flash-lite"
            cost_per_1k = 0.00015  # Input cost
        else:
            model = self.expensive_model
            model_name = "gemini-3.8-flash"
            cost_per_1k = 0.0025  # Input cost

        if model is None:
            response_text = (
                "This is a local demo response. Set GOOGLE_API_KEY or GEMINI_API_KEY to use live model output."
                if complexity == "simple"
                else "This is a local demo response for a complex query. Add an API key to enable live model output."
            )
            return response_text, model_name, 0.0

        response = model.invoke(query)

        # Estimate cost (rough)
        tokens = len(query.split()) * 1.3  # Rough token estimate
        estimated_cost = (tokens / 1000) * cost_per_1k

        response_text = response.content if hasattr(response, "content") else response
        return response_text, model_name, estimated_cost


def demo_model_routing():
    """Demonstrate model routing."""

    router = ModelRouter()

    queries = [
        "What is 2 + 2?",  # Simple
        "Analyze the economic implications of AI on the job market.",  # Complex
        "What color is the sky?",  # Simple
    ]

    print("Model Routing Demo:\n")

    total_cost = 0
    for query in queries:
        result, model, cost = router.invoke(query)
        total_cost += cost
        print(f"Query: {query[:50]}...")
        print(f"  Model: {model}")
        print(f"  Est. Cost: ${cost:.6f}")
        print(f"  Response: {result[:50]}...")

    print(f"\nTotal Estimated Cost: ${total_cost:.6f}")

class TokenBudget:
    """Track and limit token usage."""

    def __init__(self, max_tokens_per_request: int = 4000):
        self.max_per_request = max_tokens_per_request
        self.usage = {"total_input": 0, "total_output": 0, "requests": 0}

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (actual would use tiktoken)."""
        return int(len(text.split()) * 1.3)

    def check_budget(self, text: str) -> tuple[bool, int]:
        """Check if request is within budget."""
        tokens = self.estimate_tokens(text)
        return tokens <= self.max_per_request, tokens

    def record_usage(self, input_tokens: int, output_tokens: int):
        """Record token usage."""
        self.usage["total_input"] += input_tokens
        self.usage["total_output"] += output_tokens
        self.usage["requests"] += 1

    def get_stats(self) -> dict:
        return {
            **self.usage,
            "total_tokens": self.usage["total_input"] + self.usage["total_output"],
            "avg_per_request": (
                (self.usage["total_input"] + self.usage["total_output"])
                / max(self.usage["requests"], 1)
            ),
        }


class BudgetedLLM:
    """LLM with token budgeting."""

    def __init__(self, max_tokens: int = 4000):
        self.llm = (
            GoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
            if _has_google_credentials()
            else None
        )
        self.budget = TokenBudget(max_tokens_per_request=max_tokens)

    @traceable(name="budgeted_invoke")
    def invoke(self, query: str) -> str:
        # Check budget
        within_budget, tokens = self.budget.check_budget(query)

        if not within_budget:
            raise ValueError(
                f"Query exceeds token budget: {tokens} > {self.budget.max_per_request}"
            )

        if self.llm is None:
            response_text = f"Offline demo response for: {query[:80]}"
            self.budget.record_usage(tokens, self.budget.estimate_tokens(response_text))
            return response_text

        # Execute
        response = self.llm.invoke(query)
        response_text = response.content if hasattr(response, "content") else response

        # Record usage
        output_tokens = self.budget.estimate_tokens(response_text)
        self.budget.record_usage(tokens, output_tokens)

        return response_text

    def get_stats(self) -> dict:
        return self.budget.get_stats()


def demo_token_budgeting():
    """Demonstrate token budgeting."""

    llm = BudgetedLLM(max_tokens=100)

    queries = [
        "What is AI?",  # Within budget
        "Explain " + "very " * 100 + "complex topic",  # Over budget
    ]

    print("\nToken Budgeting Demo:\n")

    for query in queries:
        try:
            result = llm.invoke(query)
            print(f"✅ {query[:40]}... -> {result[:30]}...")
        except ValueError as e:
            print(f"❌ {query[:40]}... -> {e}")

    print(f"\nUsage: {llm.get_stats()}")




class CachedLLM:
    """LLM wrapper with caching."""

    def __init__(self):
        self.llm = (
            GoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
            if _has_google_credentials()
            else None
        )
        self.cache = _InMemorySemanticCache()
        self.cache_hits = 0
        self.cache_misses = 0

    @traceable(name="cached_invoke")
    def invoke(self, query: str) -> tuple[str, bool]:
        """
        Invoke with caching.
        Returns: (response, from_cache)
        """
        # Check cache
        cached = self.cache.get(query)
        if cached:
            self.cache_hits += 1
            return cached, True

        # Call LLM
        self.cache_misses += 1
        if self.llm is None:
            response_text = f"Offline cached response for: {query}"
        else:
            response = self.llm.invoke(query)
            response_text = response.content if hasattr(response, "content") else response

        # Cache result
        self.cache.set(query, response_text)

        return response_text, False

    def get_stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.1%}",
        }


def demo_caching():
    """Demonstrate caching."""

    llm = CachedLLM()

    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is Python?",  # Cache hit
        "What is python?",  # Cache hit (normalized)
        "What is Rust?",
    ]

    print("\nCaching Demo:\n")

    for query in queries:
        result, from_cache = llm.invoke(query)
        source = "CACHE" if from_cache else "LLM"
        print(f"[{source}] {query} -> {result[:30]}...")

    print(f"\nStats: {llm.get_stats()}")


if __name__ == "__main__":
    #demo_model_routing()
    #demo_token_budgeting()
    demo_caching()

    


