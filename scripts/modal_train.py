"""
Tahoe-x1 Training on Modal

This script wraps the existing scripts/train.py to run on Modal infrastructure.
It handles:
- Container image definition with all dependencies
- S3 data download and caching to Modal Volumes
- Multi-GPU training with FSDP
- Checkpoint persistence
- Cost-optimized execution

Usage:
    modal run modal_train.py --config configs/modal/modal_test.yaml
"""

import os
import pathlib
import shutil
import subprocess
import sys
from typing import Optional

import modal

# =============================================================================
# MODAL APP AND IMAGE DEFINITION
# =============================================================================

app = modal.App("tahoe-x1-training")

# Use pre-built tahoe-x1 Docker image with all dependencies
# This includes: PyTorch, flash-attn, llm-foundry, mosaicml-streaming, awscli, boto3, etc.
# Base: mosaicml/llm-foundry:2.2.1_cu121_flash2-813d596
image = (
    modal.Image.from_registry("ghcr.io/tahoebio/tahoe-x1:1.0.0")
    # Create symlink for Modal Python detection (python is at /composer-python/python)
    .run_commands("ln -sf /composer-python/python /usr/bin/python")
    .workdir("/root")
    # Add local code for development (overrides the installed tahoe-x1 package)
    # Note: add_local_dir must come absolutely last in the build chain
    .add_local_dir("./tahoe_x1", "/root/tahoe_x1")
    .add_local_dir("./scripts", "/root/scripts")
    .add_local_dir("./configs", "/root/configs")
)

# =============================================================================
# MODAL VOLUMES FOR PERSISTENT STORAGE
# =============================================================================

# Data cache: Store downloaded S3 datasets
data_volume = modal.Volume.from_name(
    "tahoe-x1-data",
    create_if_missing=True,
)

# Checkpoints: Store model checkpoints
checkpoint_volume = modal.Volume.from_name(
    "tahoe-x1-checkpoints",
    create_if_missing=True,
)

# Cache: Vocabulary and other small files
cache_volume = modal.Volume.from_name(
    "tahoe-x1-cache",
    create_if_missing=True,
)

# =============================================================================
# DATA DOWNLOAD FUNCTIONS
# =============================================================================


def copy_concurrent(src: pathlib.Path, dest: pathlib.Path, max_threads: int = 24) -> None:
    """
    Copy files in parallel to increase bandwidth when copying to Modal Volumes.
    Based on Modal's laion400 example.
    """
    from multiprocessing.pool import ThreadPool

    class MultithreadedCopier:
        def __init__(self, max_threads):
            self.pool = ThreadPool(max_threads)
            self.copy_jobs = []

        def copy(self, source, dest):
            res = self.pool.apply_async(
                shutil.copy2,
                args=(source, dest),
                callback=lambda r: print(f"  ✓ {pathlib.Path(source).name} copied"),
                error_callback=lambda exc: print(f"  ✗ {source} failed: {exc}", file=sys.stderr),
            )
            self.copy_jobs.append(res)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.pool.close()
            self.pool.join()

    dest.mkdir(parents=True, exist_ok=True)
    with MultithreadedCopier(max_threads=max_threads) as copier:
        shutil.copytree(src, dest, copy_function=copier.copy, dirs_exist_ok=True)


@app.function(
    image=image,
    volumes={
        "/data": data_volume,
        "/cache": cache_volume,
    },
    timeout=3600 * 12,  # 12 hours for initial download + copy to volume (266GB takes time)
    cpu=8.0,  # More CPUs for parallel download
)
def download_dataset_from_s3(
    s3_path: str,
    local_path: str,
    dataset_name: str = "dataset",
) -> None:
    """
    Download dataset from S3 to Modal Volume.
    Only downloads if data doesn't already exist.

    Args:
        s3_path: S3 URI (e.g., s3://bucket/path/to/data/)
        local_path: Path in Modal Volume (e.g., /data/tahoe-100m/)
        dataset_name: Name for logging
    """
    local_path = pathlib.Path(local_path)

    # Check if already downloaded
    if local_path.exists() and any(local_path.iterdir()):
        print(f"✓ Dataset '{dataset_name}' already cached at {local_path}")
        print(f"  Contents: {list(local_path.glob('*'))[:5]}...")
        return

    print(f"Downloading {dataset_name} from S3...")
    print(f"  Source: {s3_path}")
    print(f"  Destination: {local_path}")

    # Download to /tmp first (faster than writing directly to volume)
    tmp_path = pathlib.Path(f"/tmp/download/{dataset_name}")
    tmp_path.mkdir(parents=True, exist_ok=True)

    # Use AWS CLI for parallel download
    # The --no-sign-request flag is for public buckets
    cmd = [
        "aws",
        "s3",
        "sync",
        s3_path,
        str(tmp_path),
        "--no-sign-request",  # Public bucket
        "--only-show-errors",
    ]

    print(f"Running: {' '.join(cmd)}")
    print("Progress updates every 30 seconds...")

    # Run download with progress monitoring
    import threading
    import time

    download_complete = threading.Event()

    def monitor_progress():
        """Monitor download progress in background thread."""
        last_size = 0
        start_time = time.time()

        while not download_complete.is_set():
            time.sleep(30)  # Check every 30 seconds
            if download_complete.is_set():
                break

            try:
                files = list(tmp_path.rglob("*"))
                file_count = len([f for f in files if f.is_file()])
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                size_gb = total_size / (1024**3)

                # Calculate speed
                elapsed = time.time() - start_time
                speed_mbps = (total_size - last_size) / (1024**2) / 30 if elapsed > 30 else 0

                print(f"📊 Progress: {file_count} files, {size_gb:.2f} GB downloaded ({speed_mbps:.1f} MB/s)")

                last_size = total_size
            except Exception as e:
                print(f"Progress check error: {e}")

    # Start progress monitoring thread
    progress_thread = threading.Thread(target=monitor_progress, daemon=True)
    progress_thread.start()

    # Run the download
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Stop progress monitoring
    download_complete.set()
    progress_thread.join(timeout=5)

    if result.returncode != 0:
        print(f"Error downloading from S3:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Failed to download {dataset_name}")

    # Final count
    files = list(tmp_path.rglob("*"))
    file_count = len([f for f in files if f.is_file()])
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    print(f"✓ Downloaded {file_count} files ({total_size / (1024**3):.2f} GB)")

    # Copy to Modal Volume with parallel threads
    print(f"Copying to Modal Volume: {local_path}")
    print("This may take 30-60 minutes for 266 GB...")
    copy_concurrent(tmp_path, local_path, max_threads=16)  # Reduced from 32 to avoid overwhelming volume

    # Commit volume to persist
    print("Committing volume...")
    data_volume.commit()

    print(f"✓ Successfully cached {dataset_name}")


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    timeout=600,  # 10 minutes
)
def download_vocabulary(
    vocab_s3_url: str = "s3://tahoe-hackathon-data/MFM/vevo_v2_vocab.json",
    local_path: str = "/cache/vocab.json",
) -> None:
    """Download vocabulary file from S3 if not already cached."""
    local_path = pathlib.Path(local_path)

    if local_path.exists():
        print(f"✓ Vocabulary already cached at {local_path}")
        return

    print(f"Downloading vocabulary from {vocab_s3_url}...")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["aws", "s3", "cp", vocab_s3_url, str(local_path), "--no-sign-request"]
    subprocess.run(cmd, check=True)

    cache_volume.commit()
    print(f"✓ Vocabulary downloaded to {local_path}")


# =============================================================================
# VOLUME VERIFICATION
# =============================================================================


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=600,
)
def verify_dataset(dataset_path: str = "/data/tahoe-100m/train") -> dict:
    """Verify dataset integrity by counting files and checking size."""
    dataset_path = pathlib.Path(dataset_path)

    if not dataset_path.exists():
        return {"status": "missing", "path": str(dataset_path)}

    print(f"Verifying dataset at {dataset_path}...")
    files = list(dataset_path.rglob("*"))
    file_list = [f for f in files if f.is_file()]
    file_count = len(file_list)
    total_size = sum(f.stat().st_size for f in file_list)
    total_gb = total_size / (1024**3)

    # Sample some file names
    sample_files = [f.name for f in file_list[:10]]

    result = {
        "status": "present",
        "path": str(dataset_path),
        "file_count": file_count,
        "total_size_gb": round(total_gb, 2),
        "sample_files": sample_files,
    }

    print(f"✓ Verification complete:")
    print(f"  Files: {file_count:,}")
    print(f"  Size: {total_gb:.2f} GB")
    print(f"  Sample files: {sample_files[:5]}")

    return result


# =============================================================================
# TRAINING FUNCTION
# =============================================================================


@app.function(
    image=image,
    gpu="A100:2",  # 2x A100 GPUs for testing FSDP/parallel processing (cost: ~$2-3 for 20 batches)
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
        "/cache": cache_volume,
    },
    timeout=3600 * 12,  # 12 hours max (sufficient for full training runs)
    # Note: Spot instances configuration may vary by Modal version
    secrets=[
        # W&B API key for experiment tracking
        modal.Secret.from_name("wandb-api-key"),
        # Add AWS secrets if needed for private S3 buckets
        # modal.Secret.from_name("aws-credentials")
    ],
)
def train_model(config_path: str, run_name: Optional[str] = None) -> dict:
    """
    Main training function - wraps the existing scripts/train.py.

    Args:
        config_path: Path to YAML config (e.g., "configs/modal/modal_test.yaml")
        run_name: Optional run name (overrides config)

    Returns:
        dict with training results
    """
    import torch
    from omegaconf import OmegaConf

    print("=" * 80)
    print("TAHOE-X1 TRAINING ON MODAL")
    print("=" * 80)

    # GPU info
    print(f"\nGPU Info:")
    print(f"  Available GPUs: {torch.cuda.device_count()}")
    print(f"  GPU Type: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    # Load configuration
    print(f"\nLoading config: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Override paths for Modal Volumes
    if "save_folder" not in cfg or not cfg.save_folder.startswith("/checkpoints"):
        if run_name:
            cfg.save_folder = f"/checkpoints/{run_name}"
        else:
            cfg.save_folder = f"/checkpoints/{{run_name}}"

    # Update vocabulary paths
    if "vocabulary" in cfg:
        cfg.vocabulary.local = "/cache/vocab.json"

    # Update data loader paths to use cached data
    if "train_loader" in cfg and "dataset" in cfg.train_loader:
        for stream_name, stream_config in cfg.train_loader.dataset.streams.items():
            if "local" in stream_config:
                # Ensure local path points to /data volume
                original_local = stream_config.local
                if not original_local.startswith("/data"):
                    stream_config.local = f"/data/{pathlib.Path(original_local).name}"

    if "valid_loader" in cfg and "dataset" in cfg.valid_loader:
        for stream_name, stream_config in cfg.valid_loader.dataset.streams.items():
            if "local" in stream_config:
                original_local = stream_config.local
                if not original_local.startswith("/data"):
                    stream_config.local = f"/data/{pathlib.Path(original_local).name}"

    # Set run name if provided
    if run_name:
        cfg.run_name = run_name
        # Also update W&B logger name if it exists
        if "loggers" in cfg and "wandb" in cfg.loggers:
            cfg.loggers.wandb.name = run_name
            # Set entity to vevotx organization
            if "entity" not in cfg.loggers.wandb:
                cfg.loggers.wandb.entity = "vevotx"

    print("\nFinal configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Save config to temporary file for composer CLI
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        OmegaConf.save(cfg, f.name)
        temp_config_path = f.name

    # Use Composer CLI launcher for proper multi-GPU support
    # This sets up distributed training environment automatically
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")

    num_gpus = torch.cuda.device_count()
    print(f"Launching Composer with {num_gpus} GPU(s)...")

    cmd = ["composer", "--nproc", str(num_gpus), "/root/scripts/train.py", temp_config_path]
    subprocess.run(cmd, check=True)

    # Cleanup temp config
    pathlib.Path(temp_config_path).unlink()

    # Commit checkpoint volume to persist
    print("\nCommitting checkpoint volume...")
    checkpoint_volume.commit()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    return {
        "status": "success",
        "config": config_path,
        "run_name": run_name or cfg.get("run_name", "unknown"),
    }


# =============================================================================
# CLI ENTRYPOINTS
# =============================================================================


@app.local_entrypoint()
def main(
    config: str = "configs/modal/modal_test.yaml",
    run_name: Optional[str] = None,
    download_data: bool = True,
):
    """
    Main entrypoint for Modal training.

    Args:
        config: Path to config YAML file
        run_name: Optional custom run name
        download_data: Whether to download/check data before training

    Example:
        modal run modal_train.py --config configs/modal/modal_test.yaml
        modal run modal_train.py --config configs/modal/modal_test.yaml --run-name my-test-run
        modal run modal_train.py --config configs/modal/modal_test.yaml --no-download-data
    """
    print("🚀 Tahoe-x1 Modal Training Pipeline")
    print(f"   Config: {config}")
    if run_name:
        print(f"   Run Name: {run_name}")

    # Step 1: Download vocabulary if needed
    print("\n📥 Step 1/3: Checking vocabulary cache...")
    download_vocabulary.remote()

    # Step 2: Download/check training data if requested
    if download_data:
        print("\n📥 Step 2/3: Checking training data cache...")
        print("   (This may take hours on first run, but will be cached)")

        # Download tahoe-100m dataset (default for testing)
        download_dataset_from_s3.remote(
            s3_path="s3://tahoe-hackathon-data/MFM/tahoe_100m_MDS_v2/train/",
            local_path="/data/tahoe-100m/train",
            dataset_name="tahoe-100m-train",
        )

        download_dataset_from_s3.remote(
            s3_path="s3://tahoe-hackathon-data/MFM/tahoe_100m_MDS_v2/valid/",
            local_path="/data/tahoe-100m/valid",
            dataset_name="tahoe-100m-valid",
        )
    else:
        print("\n⏭️  Step 2/3: Skipping data download (--no-download-data)")

    # Step 3: Run training
    print("\n🏋️  Step 3/3: Starting training...")
    result = train_model.remote(config, run_name)

    print("\n✅ Pipeline complete!")
    print(f"   Result: {result}")


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
)
def list_checkpoints():
    """List all saved checkpoints in the checkpoint volume."""
    checkpoints = list(pathlib.Path("/checkpoints").rglob("*.pt"))

    print(f"Found {len(checkpoints)} checkpoint files:")
    for ckpt in checkpoints:
        size_mb = ckpt.stat().st_size / (1024 ** 2)
        print(f"  - {ckpt} ({size_mb:.1f} MB)")

    return checkpoints


if __name__ == "__main__":
    # For local testing/development
    print("Use 'modal run modal_train.py' to execute on Modal")
