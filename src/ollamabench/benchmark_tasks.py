# core_python/benchmark_tasks.py
BENCHMARK_TASKS = [
    {
        "id": "summarization_short",
        "name": "Short Summary",
        "prompt": "Summarize the following text in exactly one sentence: \"The quick brown fox jumps over the lazy dog, an action that demonstrates agility and a playful spirit. This classic sentence is often used to test typewriters and computer keyboards as it contains all letters of the English alphabet.\"",
        "expected_output_min_tokens": 10,
        "expected_output_max_tokens": 25,
        "temperature": 0.5,
        "seed": 42
    },
    {
        "id": "code_generation_fibonacci",
        "name": "Python Fibonacci Function",
        "prompt": "Write a Python function `fibonacci(n)` that returns the Nth Fibonacci number.",
        "expected_output_min_tokens": 40,
        "expected_output_max_tokens": 80,
        "temperature": 0.7,
        "seed": 42
    },
    {
        "id": "creative_writing_haiku",
        "name": "Haiku Generation",
        "prompt": "Write a short haiku about a rainy day in the city.",
        "expected_output_min_tokens": 15,
        "expected_output_max_tokens": 30,
        "temperature": 0.8,
        "seed": 42
    },
    {
        "id": "Youtubeing_general",
        "name": "General Q&A",
        "prompt": "What is the capital of France?",
        "expected_output_min_tokens": 5,
        "expected_output_max_tokens": 15,
        "temperature": 0.1,
        "seed": 42
    }
]