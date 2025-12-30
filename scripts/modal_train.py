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
import modal.experimental  # Required for clustered multi-node training

# =============================================================================
# MODAL APP AND IMAGE DEFINITION
# =============================================================================

app = modal.App("tahoe-x1-training")

# Use pre-built tahoe-x1 Docker image with all dependencies
# This includes: PyTorch, flash-attn, llm-foundry, mosaicml-streaming, awscli, boto3, etc.
# Base: mosaicml/llm-foundry:2.2.1_cu121_flash2-813d596
#
# EFA (Elastic Fabric Adapter) Setup for Multi-Node RDMA:
# AWS EFA enables high-performance RDMA networking between instances for multi-node training.
# We install EFA userspace libraries and the OFI-NCCL plugin so NCCL can use EFA for communication.
EFA_VERSION = "1.44.0"
OFI_NCCL_VERSION = "1.17.2-aws"

image = (
    modal.Image.from_registry("ghcr.io/tahoebio/tahoe-x1:1.0.0")
    # Create symlink for Modal Python detection (python is at /composer-python/python)
    .run_commands("ln -sf /composer-python/python /usr/bin/python")
    # Install EFA dependencies for multi-node RDMA support
    .apt_install(
        "libibverbs-dev",
        "libibverbs1",
        "wget",
        "ca-certificates",
    )
    # Install AWS EFA userspace libraries
    .run_commands(
        f"cd /tmp && wget -q https://efa-installer.amazonaws.com/aws-efa-installer-{EFA_VERSION}.tar.gz",
        f"cd /tmp && tar -xzf aws-efa-installer-{EFA_VERSION}.tar.gz",
        f"cd /tmp/aws-efa-installer && ./efa_installer.sh -y --minimal",
        "rm -rf /tmp/aws-efa-installer*",
    )
    # Install AWS OFI-NCCL plugin (enables NCCL to use EFA)
    .run_commands(
        f"cd /opt && wget -q https://github.com/aws/aws-ofi-nccl/releases/download/v{OFI_NCCL_VERSION}/aws-ofi-nccl-{OFI_NCCL_VERSION}.tar.gz",
        f"cd /opt && tar -xzf aws-ofi-nccl-{OFI_NCCL_VERSION}.tar.gz",
        f"rm aws-ofi-nccl-{OFI_NCCL_VERSION}.tar.gz",
        # Remove EFA from default library paths to prevent conflicts on non-EFA hardware
        "rm -f /etc/ld.so.conf.d/000_efa.conf /etc/ld.so.conf.d/100_ofinccl.conf",
        "ldconfig",
    )
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
# DATA MANAGEMENT FUNCTIONS
# =============================================================================


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=60,  # Quick check, should be fast
)
def check_data_exists(
    train_path: str = "/data/tahoe-100m/train",
    val_path: str = "/data/tahoe-100m/valid",
) -> dict:
    """Check if training data exists and is valid in the Modal volume."""
    import pathlib

    train_dir = pathlib.Path(train_path)
    val_dir = pathlib.Path(val_path)

    result = {
        "train_exists": False,
        "val_exists": False,
        "train_file_count": 0,
        "val_file_count": 0,
        "ready": False,
    }

    # Check training data
    if train_dir.exists():
        train_files = list(train_dir.glob("**/*.mds*"))
        result["train_exists"] = True
        result["train_file_count"] = len(train_files)
        print(f"✓ Training data found: {len(train_files)} MDS files in {train_path}")
    else:
        print(f"✗ Training data not found at {train_path}")

    # Check validation data
    if val_dir.exists():
        val_files = list(val_dir.glob("**/*.mds*"))
        result["val_exists"] = True
        result["val_file_count"] = len(val_files)
        print(f"✓ Validation data found: {len(val_files)} MDS files in {val_path}")
    else:
        print(f"✗ Validation data not found at {val_path}")

    # Data is ready if both exist and have files
    result["ready"] = (
        result["train_exists"]
        and result["val_exists"]
        and result["train_file_count"] > 0
        and result["val_file_count"] > 0
    )

    return result


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
    cpu=32,  # CPU-only, no GPUs (much cheaper for data download)
    volumes={
        "/data": data_volume,
        "/cache": cache_volume,
    },
    timeout=3600 * 12,  # 12 hours for initial download + copy to volume (266GB takes time)
)
def download_dataset_from_s3(
    s3_path: str,
    local_path: str,
    dataset_name: str = "dataset",
) -> None:
    """
    Download dataset from S3 to Modal Volume using CPU-only container.

    This runs on a 32-core CPU container (no GPUs) to avoid paying for idle GPUs
    during data download. Only after data is ready do we allocate GPUs for training.

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

    print("\n" + "=" * 80)
    print("DATA DOWNLOAD (CPU-ONLY CONTAINER)")
    print("=" * 80)
    print(f"CPU cores: 32 (no GPUs - cost-optimized)")
    print(f"Dataset: {dataset_name}")
    print(f"Source: {s3_path}")
    print(f"Destination: {local_path}")
    print("=" * 80 + "\n")

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

    print("\n" + "=" * 80)
    print(f"✅ DATA DOWNLOAD COMPLETE: {dataset_name}")
    print("=" * 80)
    print(f"Files: {file_count}")
    print(f"Size: {total_size / (1024**3):.2f} GB")
    print("=" * 80 + "\n")


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
# TRAINING FUNCTIONS
# =============================================================================


def _train_impl(config_path: str, run_name: Optional[str] = None, gpu_type: str = "A100") -> dict:
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
        "gpu_type": gpu_type,
    }


# Thin wrappers with different GPU configurations

@app.function(
    image=image,
    gpu="A100:2",  # 2x A100 GPUs for testing
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
        "/cache": cache_volume,
    },
    timeout=3600 * 12,
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_model(config_path: str, run_name: Optional[str] = None) -> dict:
    """Training with 2x A100 GPUs (for testing)."""
    return _train_impl(config_path, run_name, gpu_type="A100")


@app.function(
    image=image,
    gpu="H100:8",  # 8x H100 GPUs for full-scale training
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
        "/cache": cache_volume,
    },
    timeout=3600 * 12,
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_model_8xh100(config_path: str, run_name: Optional[str] = None) -> dict:
    """Training with 8x H100 GPUs (single-node full-scale training)."""
    return _train_impl(config_path, run_name, gpu_type="H100")


# =============================================================================
# MULTI-NODE TRAINING WITH RDMA/EFA
# =============================================================================

# Multi-node training configuration
N_NODES = 2  # Number of nodes in cluster
N_GPU_PER_NODE = 8  # GPUs per node (8x H100 per node)
GPU_CONFIG = f"H100:{N_GPU_PER_NODE}"

@app.function(
    image=image,
    gpu=GPU_CONFIG,  # 8x H100 per node
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
        "/cache": cache_volume,
    },
    timeout=3600 * 12,
    secrets=[modal.Secret.from_name("wandb-api-key")],
    # EFA environment variables for NCCL to use AWS EFA networking
    env={
        "LD_LIBRARY_PATH": "/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib",
        "NCCL_NET_PLUGIN": "ofi",
    },
)
@modal.experimental.clustered(
    size=N_NODES,
    rdma=True,  # Enable RDMA for high-speed inter-node communication
    experimental_options={
        "efa_enabled": True,  # Enable AWS EFA hardware
    },
)
def train_model_2node_16xh100(config_path: str, run_name: Optional[str] = None) -> dict:
    """
    Training with 2 nodes × 8x H100 GPUs = 16 GPUs total (multi-node distributed training).

    Uses Modal's @clustered decorator for multi-node gang scheduling.
    Each node gets 8 H100 GPUs, and Composer handles inter-node communication.
    """
    import torch
    from omegaconf import OmegaConf

    # Get cluster information from Modal
    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank  # Which node am I? (0, 1)
    world_size = len(cluster_info.container_ips)  # Total nodes (2)
    master_addr = cluster_info.container_ips[0]  # IP of node 0 (master)
    task_id = os.environ["MODAL_TASK_ID"]

    print("=" * 80)
    print("TAHOE-X1 MULTI-NODE TRAINING ON MODAL")
    print("=" * 80)
    print(f"\nCluster Info:")
    print(f"  Node Rank: {node_rank}/{world_size}")
    print(f"  Master Address: {master_addr}")
    print(f"  Task ID: {task_id}")

    # GPU info
    num_gpus = torch.cuda.device_count()
    print(f"\nGPU Info (Node {node_rank}):")
    print(f"  Available GPUs: {num_gpus}")
    print(f"  GPU Type: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"  Total GPUs in cluster: {num_gpus * world_size}")

    # Load and configure training config
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
        if "loggers" in cfg and "wandb" in cfg.loggers:
            cfg.loggers.wandb.name = run_name
            if "entity" not in cfg.loggers.wandb:
                cfg.loggers.wandb.entity = "vevotx"

    if node_rank == 0:
        print("\nFinal configuration (from master node):")
        print(OmegaConf.to_yaml(cfg))

    # Save config to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        OmegaConf.save(cfg, f.name)
        temp_config_path = f.name

    # Launch Composer with multi-node configuration
    # Composer's launcher will handle distributed setup using these environment variables
    print("\n" + "=" * 80)
    print(f"STARTING TRAINING (Node {node_rank}/{world_size})")
    print("=" * 80 + "\n")

    # Composer CLI with multi-node parameters
    # These match what torch.distributed.run expects
    cmd = [
        "composer",
        f"--nnodes={world_size}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port=1234",
        f"--nproc_per_node={num_gpus}",
        "/root/scripts/train.py",
        temp_config_path,
    ]

    print(f"Launching command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Cleanup
    pathlib.Path(temp_config_path).unlink()

    # Only master node commits checkpoint volume
    if node_rank == 0:
        print("\nCommitting checkpoint volume (from master node)...")
        checkpoint_volume.commit()

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE (Node {node_rank})")
    print("=" * 80)

    return {
        "status": "success",
        "config": config_path,
        "run_name": run_name or cfg.get("run_name", "unknown"),
        "gpu_type": "H100-multinode",
        "node_rank": node_rank,
        "world_size": world_size,
    }


# =============================================================================
# CLI ENTRYPOINTS
# =============================================================================


@app.local_entrypoint()
def main(
    config: str = "configs/modal/modal_test.yaml",
    run_name: Optional[str] = None,
    skip_data_check: bool = False,
):
    """
    Main entrypoint for Modal training.

    This automatically:
    1. Checks if vocabulary exists in Modal volume
    2. Checks if training data exists in Modal volume
    3. If not, downloads data using CPU-only container (no GPU cost)
    4. Validates data is correct
    5. Only then allocates GPUs and starts training

    Args:
        config: Path to config YAML file
        run_name: Optional custom run name
        skip_data_check: Skip data existence check and download (default: False)

    Example:
        modal run modal_train.py --config configs/modal/modal_test.yaml
        modal run modal_train.py --config configs/modal/modal_test.yaml --run-name my-test-run
        modal run modal_train.py --config configs/modal/modal_test.yaml --skip-data-check
    """
    print("\n" + "=" * 80)
    print("🚀 LAUNCHING TAHOE-X1 TRAINING ON MODAL")
    print("=" * 80)
    print(f"   Config: {config}")
    print(f"   Run name: {run_name or 'from config'}")
    print("=" * 80 + "\n")

    # Step 1: Download vocabulary if needed
    print("📦 Step 1/4: Checking vocabulary cache...")
    download_vocabulary.remote()

    # Step 2: Check if data exists (unless skipped)
    if not skip_data_check:
        print("\n📦 Step 2/4: Checking if data exists in Modal volume...")
        data_status = check_data_exists.remote()

        if not data_status["ready"]:
            print("\n⚠️  Data not ready. Need to download.\n")

            # Step 3: Download data using CPU-only container (no GPU cost!)
            print("📦 Step 3/4: Downloading data (CPU-only container, no GPUs)...")

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

            print("\n✅ Data download complete!")
        else:
            print(f"✅ Data already exists in volume (skipping download)")
            print(f"   Training files: {data_status['train_file_count']}")
            print(f"   Validation files: {data_status['val_file_count']}")
    else:
        print("\n⚠️  Skipping data check (--skip-data-check flag)")

    # Step 4: Run training on GPUs
    print("\n📦 Step 4/4: Starting training on GPUs...")

    # Automatically choose the right training function based on config
    if "2node" in config.lower() or "16xh100" in config.lower() or "multinode" in config.lower():
        print("   Using 2 nodes × 8x H100 GPUs = 16 GPUs (multi-node training with RDMA/EFA)...")
        result = train_model_2node_16xh100.remote(config, run_name)
    elif "70m" in config or "8xh100" in config.lower() or "h100" in config.lower():
        print("   Using 8x H100 GPUs (single-node training)...")
        result = train_model_8xh100.remote(config, run_name)
    else:
        print("   Using 2x A100 GPUs for testing...")
        result = train_model.remote(config, run_name)

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"   Status: {result['status']}")
    print(f"   Run: {result['run_name']}")
    print(f"   GPU Type: {result['gpu_type']}")
    print("=" * 80 + "\n")


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
