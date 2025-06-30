# ollamabenchmark/submission_client.py
import requests
import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

WEB_API_SUBMIT_URL = "https://api.ollamabenchmark.org/submit" # Placeholder URL

def submit_benchmark_results(payload: Dict[str, Any], token: str) -> bool:
    """Submits formatted benchmark results to the web API."""
    logging.info("Attempting to submit results to web API...")
    try:
        # Add the token to the payload
        payload_with_token = payload.copy()
        payload_with_token['submission_token'] = token

        headers = {"Content-Type": "application/json"}
        response = requests.post(WEB_API_SUBMIT_URL, json=payload_with_token, headers=headers, timeout=30)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        logging.info("Results submitted successfully!")
        logging.debug(f"Server response: {response.json()}")
        return True
    except requests.exceptions.ConnectionError:
        logging.error(f"Failed to connect to the submission server at {WEB_API_SUBMIT_URL}. Check your internet connection.")
        return False
    except requests.exceptions.Timeout:
        logging.error(f"Submission to {WEB_API_SUBMIT_URL} timed out.")
        return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Error submitting results: {e}. Server responded with: {getattr(e, 'response', None)}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during submission: {e}")
        return False

if __name__ == "__main__":
    # Example usage (will fail without a live API)
    dummy_results = {
        "model": "llama3",
        "system_info": {"os": {"name": "Windows"}},
        "overall_metrics": {"benchmark_score": 100},
        "tasks": [{"task_id": "test", "tokens_per_second_overall": 50}]
    }
    logging.info("This is a dummy submission client. It will likely fail without a live API.")
    submit_benchmark_results(dummy_results, "your_dummy_token")
