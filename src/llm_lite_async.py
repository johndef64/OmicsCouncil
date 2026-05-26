"""
llm_lite.py - Lightweight text-only LLM inference module with async support

Supports both sync and async operations for efficient concurrent requests.
Works with Groq and OpenRouter models.
"""

import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from openai import OpenAI, AsyncOpenAI

# Load API keys
try:
    with open("api_keys.json") as f:
        api_keys = json.load(f)
    os.environ["GROQ_API_KEY"] = api_keys.get("groq", "")
    os.environ["OPENROUTER_API_KEY"] = api_keys.get("openrouter", "")
except FileNotFoundError:
    pass

# Model dictionaries
GROQ_MODELS = {
    1: "qwen/qwen3-32b",
    2: "meta-llama/llama-4-scout-17b-16e-instruct",
    3: "llama-3.1-8b-instant",
    4: "llama-3.3-70b-versatile",
    5: "openai/gpt-oss-20b",
    6: "openai/gpt-oss-120b",
}

OPENROUTER_MODELS = {
    1: "anthropic/claude-opus-4.5",
    2: "anthropic/claude-sonnet-4.5",
    3: "anthropic/claude-haiku-4.5",
    4: "anthropic/claude-opus-4",
    5: "anthropic/claude-sonnet-4",
    6: "openai/gpt-5.2-pro",
    7: "openai/gpt-5.2",
    8: "openai/gpt-5.2-chat",
    9: "openai/gpt-5.2-codex",
    10: "openai/gpt-5.1-codex-max",
    11: "google/gemini-3-flash-preview",
    12: "x-ai/grok-4",
    13: "x-ai/grok-4.1-fast",
    14: "x-ai/grok-4-fast",
    15: "x-ai/grok-3",
    16: "x-ai/grok-3-mini",
    17: "x-ai/grok-code-fast-1",
    18: "mistralai/mistral-small-creative",
    19: "deepseek/deepseek-v3.2",
    20: "mistralai/ministral-14b-2512",
    21: "mistralai/ministral-8b-2512",
    22: "mistralai/ministral-3b-2512",
    23: "mistralai/devstral-2512",
    24: "allenai/olmo-3.1-32b-instruct",
    25: "allenai/olmo-3.1-32b-think",
    26: "moonshotai/kimi-k2.5",
    27: "writer/palmyra-x5",
    28: "minimax/minimax-m2-her",
    29: "minimax/minimax-m2.1",
    30: "prime-intellect/intellect-3",
    31: "deepcogito/cogito-v2.1-671b",
}

DEFAULT_MODEL = GROQ_MODELS[4]

SYSTEM_PROMPTS = {
    "default": "You are a helpful AI assistant.",
    "summarize": "You are an expert at summarizing information concisely and accurately.",
    "extract_json": "Extract information and return it as valid JSON.",
    "analyze": "You are an analytical assistant. Provide clear, structured analysis.",
}


def get_model(query: str, models_dict: Dict[int, str] = GROQ_MODELS) -> str:
    """Select a model based on query content."""
    import re
    query_parts = [part.lower().strip() for part in query.split() if part.strip()]
    if not query_parts:
        return list(models_dict.values())[0]

    scores = {}
    for key, model_name in models_dict.items():
        model_parts = [part.lower() for part in re.split(r'[-/]', model_name) if part]
        score = 0
        for query_part in query_parts:
            for model_part in model_parts:
                if query_part == model_part:
                    score += 20
                elif '.' in query_part and query_part in model_part:
                    score += 18
                elif query_part in model_part:
                    score += 8
                elif model_part in query_part:
                    score += 6
        if score > 0:
            scores[model_name] = score

    return max(scores.items(), key=lambda x: x[1])[0] if scores else list(models_dict.values())[0]


def get_client(model: str) -> tuple[OpenAI, str]:
    """Get the appropriate OpenAI client for the model."""
    if model in GROQ_MODELS.values():
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("No Groq API key found in environment.")
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1"), "groq"
    elif model in OPENROUTER_MODELS.values():
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("No OpenRouter API key found in environment.")
        return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1"), "openrouter"
    else:
        raise ValueError(f"Model '{model}' not found in available models.")


def get_async_client(model: str) -> tuple[AsyncOpenAI, str]:
    """Get the appropriate AsyncOpenAI client for the model."""
    if model in GROQ_MODELS.values():
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("No Groq API key found in environment.")
        return AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1"), "groq"
    elif model in OPENROUTER_MODELS.values():
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("No OpenRouter API key found in environment.")
        return AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1"), "openrouter"
    else:
        raise ValueError(f"Model '{model}' not found in available models.")


def llm(
    prompt: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str:
    """Simple text-only LLM inference (sync)."""
    system_prompt = system or SYSTEM_PROMPTS["default"]
    client, provider = get_client(model)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    kwargs = {"messages": messages, "model": model, "temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


async def llm_async(
    prompt: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str:
    """Simple text-only LLM inference (async)."""
    system_prompt = system or SYSTEM_PROMPTS["default"]
    client, provider = get_async_client(model)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    kwargs = {"messages": messages, "model": model, "temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def llm_safe(
    prompt: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Safe text-only LLM inference with error handling (sync)."""
    try:
        content = llm(prompt, system, model, temperature, max_tokens)
        _, provider = get_client(model)
        return {
            "success": True,
            "content": content,
            "error": None,
            "provider": provider,
        }
    except Exception as e:
        return {
            "success": False,
            "content": None,
            "error": str(e),
            "provider": None,
        }


async def llm_safe_async(
    prompt: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Safe text-only LLM inference with error handling (async)."""
    try:
        content = await llm_async(prompt, system, model, temperature, max_tokens)
        _, provider = get_async_client(model)
        return {
            "success": True,
            "content": content,
            "error": None,
            "provider": provider,
        }
    except Exception as e:
        return {
            "success": False,
            "content": None,
            "error": str(e),
            "provider": None,
        }


async def llm_batch_async(
    prompts: List[str],
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Process multiple prompts concurrently with async.

    Args:
        prompts: List of user prompts
        system: System prompt
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        List of dicts with keys: success, content, error, provider
    """
    tasks = [
        llm_safe_async(prompt, system, model, temperature, max_tokens)
        for prompt in prompts
    ]
    return await asyncio.gather(*tasks)


# Example usage
if __name__ == "__main__":
    import time

    # Test sync
    print("=== SYNC TEST ===")
    start = time.time()
    response = llm("What is 2+2?")
    print(f"Response: {response}")
    print(f"Time: {time.time() - start:.2f}s\n")

    # Test async single
    print("=== ASYNC SINGLE TEST ===")
    async def test_single():
        start = time.time()
        response = await llm_async("What is 3+3?")
        print(f"Response: {response}")
        print(f"Time: {time.time() - start:.2f}s\n")

    asyncio.run(test_single())

    # Test async batch
    print("=== ASYNC BATCH TEST (3 concurrent requests) ===")
    async def test_batch():
        prompts = [
            "What is the capital of France?",
            "What is the capital of Italy?",
            "What is the capital of Spain?",
        ]
        start = time.time()
        results = await llm_batch_async(prompts)
        for i, result in enumerate(results):
            if result["success"]:
                print(f"Prompt {i+1}: {result['content'][:80]}...")
        print(f"Total time for 3 requests: {time.time() - start:.2f}s")
        print(f"Average time per request: {(time.time() - start)/3:.2f}s\n")

    asyncio.run(test_batch())
