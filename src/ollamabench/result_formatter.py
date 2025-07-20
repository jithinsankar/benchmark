# core_python/result_formatter.py
import json
from datetime import datetime
import logging
from typing import Dict, Any

def format_results_for_display(results: Dict[str, Any]) -> str:
    """Formats benchmark results into a human-readable string for CLI output."""
    if not results:
        return "No benchmark results available."
    if "error" in results:
        return f"Benchmark encountered an error: {results['error']}"

    output = []

    # Header
    output.append("\n" + "="*50)
    output.append(" Ollama Benchmark Results")
    output.append("="*50)

    # Basic Info
    model_name = results.get("model", "N/A")
    timestamp = datetime.fromtimestamp(results.get("timestamp", datetime.now().timestamp())).strftime("%Y-%m-%d %H:%M:%S")
    output.append(f"Model: {model_name}")
    output.append(f"Timestamp: {timestamp}")
    output.append("-" * 50)

    # System Info
    sys_info = results.get("system_info", {})
    output.append("System Information:")
    output.append(f"  OS: {sys_info.get('os', {}).get('name', 'N/A')} {sys_info.get('os', {}).get('release', 'N/A')} ({sys_info.get('os', {}).get('architecture', 'N/A')})")
    cpu = sys_info.get('cpu', {})
    output.append(f"  CPU: {cpu.get('name', 'N/A')} ({cpu.get('cores', 'N/A')} cores, {cpu.get('threads', 'N/A')} threads, {cpu.get('frequency_ghz', 'N/A')} GHz)")
    ram = sys_info.get('ram', {})
    output.append(f"  RAM: {ram.get('total_gb', 'N/A')} GB")
    gpus = sys_info.get('gpu', [])
    for i, gpu in enumerate(gpus):
        output.append(f"  GPU {i+1}: {gpu.get('name', 'N/A')} ({gpu.get('vendor', 'N/A')}, {gpu.get('vram_mb', 'N/A')} MB VRAM)")
    output.append("-" * 50)

    # Task-specific Results
    output.append("Task-Specific Performance:")
    for task in results.get("tasks", []):
        task_id = task.get("task_id", "N/A")
        task_name = task.get("task_name", "N/A")
        error = task.get("error")
        if error:
            output.append(f"  - {task_name} ({task_id}): Error: {error}")
        else:
            tps_overall = task.get("tokens_per_second_overall", 0)
            ttft_ms = task.get("time_to_first_token_ms", 0)
            prompt_len = task.get("prompt_length_tokens", 0)
            output_len = task.get("output_length_tokens", 0)
            output.append(f"  - {task_name} ({task_id}):")
            output.append(f"    Input Tokens: {prompt_len}, Output Tokens: {output_len}")
            output.append(f"    Tokens/Sec: {tps_overall:.2f}")
            output.append(f"    Time to First Token: {ttft_ms:.2f} ms")
    output.append("-" * 50)

    # Overall Metrics
    overall = results.get("overall_metrics", {})
    output.append("Overall Benchmark Metrics:")
    output.append(f"  Average Tokens/Sec: {overall.get('average_tokens_per_second', 0):.2f}")
    output.append(f"  Average Time to First Token: {overall.get('average_time_to_first_token_ms', 0):.2f} ms")
    output.append(f"  Benchmark Score: {overall.get('benchmark_score', 0):.2f}")
    output.append("="*50 + "\n")

    return "\n".join(output)

def save_results_to_json(results: Dict[str, Any], filepath: str):
    """Saves benchmark results to a JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Results saved to: {filepath}")
    except Exception as e:
        logging.error(f"Failed to save results to {filepath}: {e}")

def create_submission_payload(results: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a simplified payload for web API submission (future)."""
    # Remove raw_ollama_metadata, reduce content length, etc.
    payload = results.copy()
    payload["tasks"] = [
        {
            "task_id": t["task_id"],
            "prompt_length_tokens": t["prompt_length_tokens"],
            "output_length_tokens": t["output_length_tokens"],
            "time_to_first_token_ms": t["time_to_first_token_ms"],
            "total_generation_time_s": t["total_generation_time_s"],
            "tokens_per_second_overall": t["tokens_per_second_overall"],
            "tokens_per_second_prompt_eval": t["tokens_per_second_prompt_eval"],
            "tokens_per_second_response_gen": t["tokens_per_second_response_gen"],
        } for t in payload.get("tasks", []) if "error" not in t
    ]
    return payload