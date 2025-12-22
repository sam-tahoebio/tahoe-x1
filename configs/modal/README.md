# Modal Configuration Files

This directory contains configuration files specifically for running Tahoe-x1 training on [Modal](https://modal.com).

## 📁 Files

| File | Description | Duration | Cost Estimate |
|------|-------------|----------|---------------|
| `modal_test.yaml` | Quick test run (20 batches) | ~5-10 min | $0.10-0.50 |

For full training, copy `modal_test.yaml` and adjust `max_duration` (see [Customizing Configurations](#-customizing-configurations) below).

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install Modal CLI
pip install modal

# Authenticate with Modal (first time only)
modal setup
```

### 2. Run Test Training

```bash
# From tahoe-x1 repository root
modal run scripts/modal_train.py --config configs/modal/modal_test.yaml
```

This will:
1. ✅ Download vocabulary from S3 (cached in Modal Volume)
2. ✅ Download training data from S3 (cached, ~hours first time)
3. ✅ Train for 20 batches (~5-10 minutes)
4. ✅ Save checkpoints to Modal Volume

### 3. Check Logs

```bash
# View Modal app logs
modal app logs tahoe-x1-training

# List saved checkpoints
modal run scripts/modal_train.py::list_checkpoints
```

### 4. Enable Weights & Biases (Optional)

Track your training runs with W&B for experiment comparison and visualization:

```bash
# 1. Set up W&B secret (first time only)
modal secret create wandb-api-key WANDB_API_KEY="your-wandb-api-key"

# 2. Run with W&B logging enabled
modal run scripts/modal_train.py --config configs/modal/modal_test.yaml --run-name my-experiment

# Your run will appear at: https://wandb.ai/vevotx/tahoe-x1/runs/...
```

**W&B Configuration** (`modal_test.yaml`):
```yaml
loggers:
  wandb:
    project: tahoe-x1      # W&B project name
    entity: vevotx         # Organization name
    name: "{run_name}"     # Run name (auto-filled)
```

The modal_train.py automatically:
- ✅ Sets the run name from `--run-name` parameter
- ✅ Logs all training metrics (loss, MFU, throughput, etc.)
- ✅ Tracks hyperparameters and system info
- ✅ Enables real-time visualization in W&B dashboard

## ⚙️ Configuration Details

### `modal_test.yaml`

**Purpose:** Cost-controlled test to verify Modal setup works correctly.

**Key Settings:**
```yaml
max_duration: "20ba"          # Only 20 batches
save_interval: "10ba"         # Checkpoint every 10 batches
device_train_batch_size: 50   # Moderate batch size
```

**Expected Output:**
- Training metrics for 20 batches
- 2 checkpoint files in `/checkpoints/`
- Total cost: ~$0.10-0.50 (with spot instances)

**GPU:** 8x A100 (spot instances enabled for 70% savings)

## 📊 Data Caching

Modal caches datasets in Volumes for reuse:

```
/data/                  # Data Volume
├── tahoe-100m/
│   ├── train/         # Training split (~96M cells)
│   └── valid/         # Validation split

/cache/                 # Cache Volume
└── vocab.json         # Gene vocabulary

/checkpoints/           # Checkpoint Volume
└── {run_name}/        # Your training runs
```

**First Run:** Downloads ~100GB of training data (takes hours)
**Subsequent Runs:** Uses cached data (starts immediately)

To skip data download check:
```bash
modal run scripts/modal_train.py --config configs/modal/modal_test.yaml --no-download-data
```

## 💰 Cost Management

### Estimated Costs

| Configuration | Duration | GPU Hours | Cost (Spot) | Cost (On-Demand) |
|---------------|----------|-----------|-------------|------------------|
| `modal_test.yaml` | 5-10 min | 0.08 | ~$0.30 | ~$1.00 |
| Full epoch | ~2 hours | 2.0 | ~$8.00 | ~$25.00 |

**Spot Instances:** Enabled by default (`spot=True` in `modal_train.py`)
- 70% cheaper than on-demand
- May be interrupted (checkpoint frequently!)

### Cost Control Tips

1. **Start Small:** Always test with `modal_test.yaml` first
2. **Limit Duration:** Set `max_duration` to control batch count
3. **Use Spot:** Keep `spot=True` (default)
4. **Set Timeouts:** Prevent runaway jobs
5. **Monitor Usage:** Check Modal dashboard regularly

## 🔧 Customizing Configurations

### Create Your Own Config

```bash
# Copy test config as template
cp configs/modal/modal_test.yaml configs/modal/my_config.yaml

# Edit your config
vim configs/modal/my_config.yaml

# Run with your config
modal run scripts/modal_train.py --config configs/modal/my_config.yaml --run-name my-experiment
```

### Important Parameters

**Training Duration:**
```yaml
max_duration: "100ba"    # 100 batches
max_duration: "1ep"      # 1 epoch
max_duration: "10000ba"  # 10,000 batches
```

**Checkpointing:**
```yaml
save_interval: "50ba"              # Save every 50 batches
save_num_checkpoints_to_keep: 3   # Keep last 3 checkpoints
save_folder: "/checkpoints/{run_name}"
```

**Model Size:**
```yaml
model:
  d_model: 512    # 70M params (default)
  d_model: 1024   # ~500M params
  d_model: 2048   # ~2B params
```

## 🐛 Troubleshooting

### Data Download Issues

```bash
# Manually trigger data download
modal run scripts/modal_train.py::download_dataset_from_s3 \
  --s3-path s3://tahoe-hackathon-data/MFM/tahoe_100m_MDS_v2/train/ \
  --local-path /data/tahoe-100m/train
```

### Checkpoint Issues

```bash
# List all checkpoints
modal run scripts/modal_train.py::list_checkpoints

# Download checkpoint to local
modal volume get tahoe-x1-checkpoints /checkpoints/my-run/latest.pt ./local-checkpoint.pt
```

### GPU Not Available

If you get "No GPUs available":
- Check Modal GPU quota
- Try different GPU type: `gpu="H100"` or `gpu="A10G"`
- Use smaller GPU count: `gpu="A100:4"`

### Out of Memory (OOM)

Reduce batch size in config:
```yaml
device_train_batch_size: 25  # Reduced from 50
```

## 📚 Additional Resources

- [Modal Documentation](https://modal.com/docs)
- [Tahoe-x1 Paper](http://www.tahoebio.ai/news/tahoe-x1)
- [MosaicML Composer](https://docs.mosaicml.com/projects/composer/)
- [FSDP Guide](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

## 🆘 Getting Help

**Modal Issues:**
- [Modal Slack Community](https://modal.com/slack)
- [Modal GitHub Issues](https://github.com/modal-labs/modal-client/issues)

**Tahoe-x1 Issues:**
- [GitHub Issues](https://github.com/tahoebio/tahoe-x1/issues)
- Email: admin@tahoebio.ai
