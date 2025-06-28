'''# Ollama Benchmark Tool

This tool provides a comprehensive benchmarking suite for Ollama models, allowing you to measure their performance on your specific hardware. It automates the process of setting up Ollama (if not already present), pulling models, running various benchmark tasks, and collecting detailed system information.

## Features

- **Automated Ollama Management**: Automatically checks for Ollama installation, downloads and installs it if missing, and ensures the Ollama server is running.
- **Ollama Version Detection**: Automatically detects and reports the exact Ollama version used for the benchmark.
- **Model Quantization Detection**: Captures the specific quantization level of the model (e.g., `gemma3:1b-q4_0`) as this impacts performance and memory footprint.
- **Model Pulling**: Seamlessly pulls specified Ollama models before benchmarking.
- **Performance Metrics**: Measures key performance indicators such as:
    - **Tokens per second (TPS)**: Overall, prompt evaluation, and response generation.
    - **Time to First Token (TTFT)**: Latency for the first token generation.
- **System Information Collection**: Gathers detailed hardware information (CPU, RAM, GPU, OS, device model) to provide context for benchmark results.
- **Extensible Benchmark Tasks**: Comes with a set of predefined benchmark tasks and is designed to be easily extensible with new tasks.

## Installation

You can install the `ollamabenchmark` package using pip:

```bash
pip install .
```

## Usage

To run a benchmark for a specific Ollama model, use the following command:

```bash
python -m ollamabenchmark.benchmark_runner <model_name> [--warmup-runs <number_of_runs>]
```

- `<model_name>`: The Ollama model to benchmark (e.g., `gemma3:1b`, `llama2`).
- `--warmup-runs <number_of_runs>`: (Optional) Number of warm-up runs before actual benchmarking. Defaults to 1.

After the benchmark completes, you will be prompted to upload the results to the Ollama Benchmark API. Pressing Enter (or typing 'y'/'yes') will upload the results, while typing 'n'/'no' will skip the upload.

### Example

```bash
python -m ollamabenchmark.benchmark_runner gemma3:1b --warmup-runs 3
```

The benchmark results, including system information, Ollama version, model details, and task-specific metrics, will be printed to the console.

## Project Structure

- `src/ollamabenchmark/benchmark_runner.py`: The main script to run the benchmark.
- `src/ollamabenchmark/ollama_manager.py`: Handles Ollama installation, server management, version detection, and model pulling.
- `src/ollamabenchmark/sys_info.py`: Collects detailed system hardware information.
- `src/ollamabenchmark/benchmark_tasks.py`: Defines the benchmark tasks.
- `src/ollamabenchmark/result_formatter.py`: (If applicable) Formats the benchmark results.
- `src/ollamabenchmark/submission_client.py`: (If applicable) Handles submission of results.

## Dependencies

The project relies on the following Python libraries:

- `ollama`: Python client for Ollama.
- `psutil`: For system and process utilities.
- `requests`: For making HTTP requests (e.g., to Ollama API, GitHub API).
- `wmi` (Windows only): For Windows Management Instrumentation.
- `pynvml` (Windows/Linux, optional): For NVIDIA GPU monitoring.

These dependencies are automatically installed when you install the package using `pip`.
''