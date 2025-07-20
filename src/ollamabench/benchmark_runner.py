# core_python/benchmark_runner.py
import time
import logging
import ollama 
from typing import Dict, Any, List

from ollamabench.ollama_manager import ensure_ollama_ready, pull_ollama_model, get_ollama_version
from ollamabench.sys_info import get_system_info
from ollamabench.benchmark_tasks import BENCHMARK_TASKS

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_benchmark(model_name: str, num_warmup_runs: int = 1) -> Dict[str, Any]:
    """
    Runs the full benchmark suite for a given Ollama model.
    """
    logging.info(f"\n--- Starting Benchmark for Model: {model_name} ---")

    if not ensure_ollama_ready():
        logging.error("Ollama is not ready. Cannot proceed with benchmark.")
        return {"error": "Ollama not ready"}

    ollama_version = get_ollama_version()
    logging.info(f"Using Ollama version: {ollama_version or 'Unknown'}")

    if not pull_ollama_model(model_name):
        logging.error(f"Failed to pull model '{model_name}'.")
        return {"error": "Model pull failed"}

    try:
        model_info = ollama.show(model_name)
        if not model_info:
            logging.error(f"Could not get details for model '{model_name}'.")
            return {"error": "Failed to get model details"}
    except Exception as e:
        logging.error(f"Error getting model details for '{model_name}': {e}")
        return {"error": f"Failed to get model details: {e}"}

    system_info = get_system_info()
    benchmark_results = {
        "timestamp": time.time(),
        "model": model_name,
        "model_info": {
            "quantization_level": model_info.get("details", {}).get("quantization_level"),
            "family": model_info.get("details", {}).get("family"),
            "parameter_size": model_info.get("details", {}).get("parameter_size"),
            "size": model_info.get("size"),
        },
        "ollama_version": ollama_version,
        "system_info": system_info,
        "tasks": [],
        "overall_metrics": {}
    }

    logging.info(f"Performing {num_warmup_runs} warm-up runs...")
    for _ in range(num_warmup_runs):
        try:
            ollama.generate(model=model_name, prompt="hello", options={"num_predict": 1})
        except Exception as e:
            logging.warning(f"Warm-up failed: {e}. This might indicate a problem.")

    logging.info("Starting actual benchmark tasks...")
    all_tokens_per_second: List[float] = []
    all_ttft_ms: List[float] = []

    for task in BENCHMARK_TASKS:
        task_id = task["id"]
        task_name = task["name"]
        prompt = task["prompt"]
        options = {
            "temperature": task.get("temperature", 0.7),
            "num_predict": task.get("expected_output_max_tokens", -1),
            "seed": task.get("seed", None)
        }

        logging.info(f"  Running task: {task_name} ({task_id})")

        try:
            start_time = time.perf_counter()
            first_token_time = None
            full_response_content = ""
            input_token_count = 0
            output_token_count = 0

            stream_response = ollama.generate(model=model_name, prompt=prompt, options=options, stream=True)
            for chunk in stream_response:
                if first_token_time is None and chunk.get('response'):
                    first_token_time = time.perf_counter()

                if chunk.get('response'):
                    full_response_content += chunk['response']

                if 'eval_count' in chunk:
                    output_token_count = chunk['eval_count']
                    input_token_count = chunk['prompt_eval_count']

            end_time = time.perf_counter()

            total_generation_time_s = end_time - start_time
            ttft_s = first_token_time - start_time if first_token_time else total_generation_time_s

            tps_prompt = input_token_count / (chunk.get('prompt_eval_duration', 1) / 1_000_000_000) if input_token_count else 0
            tps_response = output_token_count / (chunk.get('eval_duration', 1) / 1_000_000_000) if output_token_count else 0
            tps_overall = (input_token_count + output_token_count) / total_generation_time_s if total_generation_time_s else 0

            all_tokens_per_second.append(tps_overall)
            all_ttft_ms.append(ttft_s * 1000)

            task_result = {
                "task_id": task_id,
                "task_name": task_name,
                "prompt_length_tokens": input_token_count,
                "output_length_tokens": output_token_count,
                "response_content_truncated": full_response_content[:200] + "..." if len(full_response_content) > 200 else full_response_content,
                "time_to_first_token_ms": round(ttft_s * 1000, 2),
                "total_generation_time_s": round(total_generation_time_s, 2),
                "tokens_per_second_overall": round(tps_overall, 2),
                "tokens_per_second_prompt_eval": round(tps_prompt, 2),
                "tokens_per_second_response_gen": round(tps_response, 2),
                "raw_ollama_metadata": {
                    "load_duration": chunk.get("load_duration"),
                    "prompt_eval_count": chunk.get("prompt_eval_count"),
                    "prompt_eval_duration": chunk.get("prompt_eval_duration"),
                    "eval_count": chunk.get("eval_count"),
                    "eval_duration": chunk.get("eval_duration"),
                }
            }
            benchmark_results["tasks"].append(task_result)

        except Exception as e:
            logging.error(f"  Error running task '{task_name}': {e}")
            benchmark_results["tasks"].append({
                "task_id": task_id,
                "task_name": task_name,
                "error": str(e)
            })

    if all_tokens_per_second:
        avg_tps = sum(all_tokens_per_second) / len(all_tokens_per_second)
        avg_ttft_ms = sum(all_ttft_ms) / len(all_ttft_ms)
        benchmark_results["overall_metrics"] = {
            "average_tokens_per_second": round(avg_tps, 2),
            "average_time_to_first_token_ms": round(avg_ttft_ms, 2),
            "benchmark_score": round(avg_tps, 2)
        }
    else:
        benchmark_results["overall_metrics"] = {"average_tokens_per_second": 0, "average_time_to_first_token_ms": 0, "benchmark_score": 0}

    logging.info("\n--- Benchmark Complete ---")
    return benchmark_results

if __name__ == "__main__":
    import argparse
    import json
    from ollamabench.submission_client import submit_benchmark_results
    from ollamabench.version_checker import check_for_updates_and_exit

    # --- Version Check ---
    check_for_updates_and_exit("yourusername/benchmark")

    parser = argparse.ArgumentParser(description="Run Ollama model benchmarks.")
    parser.add_argument("model_name", type=str, help="The Ollama model to benchmark (e.g., 'gemma3:1b').")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warm-up runs before actual benchmarking.")
    args = parser.parse_args()

    results = run_benchmark(args.model_name, args.warmup_runs)
    logging.info("\nFull Benchmark Results:")
    logging.info(json.dumps(results, indent=2))

    if results and "error" not in results:
        upload_choice = input("\nDo you want to upload these results to the Ollama Benchmark API? (Y/n): ").strip().lower()
        if upload_choice in ["", "y", "yes"]:
            token = input("Please enter your submission token: ").strip()
            if not token:
                logging.error("Submission token cannot be empty.")
            else:
                logging.info("Uploading results...")
                if submit_benchmark_results(results, token):
                    logging.info("Results uploaded successfully!")
                else:
                    logging.error("Failed to upload results.")
        else:
            logging.info("Results not uploaded.")
