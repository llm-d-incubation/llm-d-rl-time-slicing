import argparse
import os
import time
import sys
import logging
from timeslice.snapshot_agent import SnapshotAgentClient
import requests
import subprocess
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_checkpoint_dir(directory="/mnt/huge-ckpt"):
    """Removes all files in the checkpoint directory to prevent disk pressure."""
    if os.path.exists(directory):
        logger.info(f"Cleaning up old checkpoint files in {directory}...")
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")

def start_vllm_server(model, host="0.0.0.0", port=8000, tensor_parallel_size=1):
    """Starts a vLLM server using subprocess."""
    command = [
        "vllm", "serve", model,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--enforce-eager",
    ]
    logger.info(f"Starting vLLM server with command: {' '.join(command)}")
    return subprocess.Popen(command)


def wait_for_vllm_health(host, port, retries=180, delay=3):
    """Polls the vLLM /health endpoint until it returns 200 or retries are exhausted."""
    url = f"http://{host}:{port}/health"
    logger.info(f"Waiting for vLLM health check at {url}...")
    for i in range(retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info("vLLM is healthy!")
                return True
        except requests.RequestException:
            pass
        
        if i % 10 == 0 and i > 0:
            logger.info(f"Still waiting for vLLM health check... ({i}/{retries})")
        time.sleep(delay)
    
    logger.error(f"vLLM health check failed after {retries} retries.")
    return False


def call_vllm_generate(host, port, prompt, model):
    """Calls the generate endpoint of the vLLM server."""
    url = f"http://{host}:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 0.0,
    }
    try:
        logger.info(f"Calling vLLM (OpenAI) completions at {url} with prompt: {prompt}")
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info(f"vLLM Response: {data}")
        return data
    except Exception as e:
        logger.error(f"Error calling vLLM OpenAI completions: {e}")
    return None


def run(endpoint, job_id, group, interval, backend, vllm_host=None, vllm_port=None, prompt=None, model=None):
    logger.info(f"Starting sampler, calling {endpoint} every {interval}s")
    
    call_vllm_generate(vllm_host, vllm_port, prompt, model)
    
    with SnapshotAgentClient(endpoint) as client:
        while True:
            # 1. Snapshot and wait
            logger.info(f"Triggering snapshot for job {job_id} using backend {backend}...")
            op = client.snapshot_and_wait(job_id, group, backend=backend)
            
            if op.status == "OPERATION_STATUS_FAILED":
                logger.error(f"Snapshot operation failed: {op.error}")
                sys.exit(1)
            
            logger.info(f"Snapshot complete: {op.elapsed_ms}ms, {op.storage_bytes} bytes")

            # 2. If vLLM is configured, restore and test
            if vllm_host and vllm_port and prompt:
                logger.info(f"Triggering restore for job {job_id} using backend {backend}...")
                op = client.restore_and_wait(job_id, group, backend=backend)
                
                if op.status == "OPERATION_STATUS_FAILED":
                    logger.error(f"Restore operation failed: {op.error}")
                    sys.exit(1)
                
                logger.info(f"Restore complete: {op.elapsed_ms}ms")
                
                wait_for_vllm_health(vllm_host, vllm_port)
                logger.info("vLLM server healthy after restore")
                call_vllm_generate(vllm_host, vllm_port, prompt, model)

            logger.info(f"Waiting {interval}s before next cycle...")
            time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sampler script using timeslice library")
    parser.add_argument("--endpoint", default=os.getenv("AGENT_ENDPOINT", "localhost:9001"), help="gRPC endpoint")
    parser.add_argument("--job-id", default=os.getenv("JOB_ID", "test-job"), help="Job ID for snapshot")
    parser.add_argument("--group", default=os.getenv("GROUP", "default"), help="Group for snapshot")
    parser.add_argument("--interval", type=int, default=int(os.getenv("INTERVAL", "30")), help="Interval in seconds")
    parser.add_argument("--vllm-model", default=os.getenv("VLLM_MODEL", "facebook/opt-125m"), help="Model to start vLLM server with (optional)")
    parser.add_argument("--vllm-host", default=os.getenv("VLLM_HOST", "0.0.0.0"), help="vLLM server host")
    parser.add_argument("--vllm-port", type=int, default=int(os.getenv("VLLM_PORT", "8000")), help="vLLM server port")
    parser.add_argument("--prompt", default=os.getenv("PROMPT", "San Francisco is a"), help="Prompt for vLLM generate")
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")), help="Tensor parallel size for vLLM")
    parser.add_argument("--backend", default=os.getenv("BACKEND", "GPU_GCR"), help="Backend to use")

    args = parser.parse_args()

    # Clean up old checkpoints before starting a new run
    cleanup_checkpoint_dir()

    if args.vllm_model:
        vllm_process = start_vllm_server(args.vllm_model, args.vllm_host, args.vllm_port, args.tensor_parallel_size)
        wait_for_vllm_health(args.vllm_host, args.vllm_port)
        logger.info(f"vLLM server started and healthy with PID: {vllm_process.pid}")

    run(args.endpoint, args.job_id, args.group, args.interval, args.backend, args.vllm_host, args.vllm_port, args.prompt, args.vllm_model)
