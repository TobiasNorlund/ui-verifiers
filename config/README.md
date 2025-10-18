# Configuration System

This directory contains YAML configuration files for training VLMs on UI tasks. Using configs makes it easy to:

- **Run experiments** with different hyperparameters
- **Compare approaches** (LoRA targeting strategies, algorithms, etc.)
- **Scale training** across multiple VMs
- **Reproduce results** by versioning configs with code

## Quick Start

```bash
# Train with default config
python scripts/train_with_config.py --config config/qwen25_vl_lora.yaml

# Override VM URL
python scripts/train_with_config.py --config config/qwen25_vl_lora.yaml --vm-url http://34.123.45.67:8000

# Override training steps
python scripts/train_with_config.py --config config/qwen25_vl_lora.yaml --num-steps 500
```

## Available Configs

### Main Configs

- **`qwen25_vl_lora.yaml`** - Recommended baseline for Qwen2.5-VL with LoRA
  - LoRA targeting: attention+mlp (recommended)
  - Rank: 32
  - Learning rate: auto-detect (2e-4 for LoRA)
  - Batch size: 4

- **`qwen25_vl_full_finetune.yaml`** - Full fine-tuning without LoRA
  - Higher GPU memory requirements
  - Learning rate: auto-detect (1e-5)
  - Smaller batch size: 2

- **`multi_vm_distributed.yaml`** - Multi-VM distributed training
  - Demonstrates load balancing across 4 VMs
  - Larger batch size: 8
  - More concurrent actors: 8

### Experimental Configs (`experiments/`)

For ablation studies and research:

- **`lora_ablation_attention_only.yaml`** - LoRA on attention layers only
- **`lora_ablation_mlp_only.yaml`** - LoRA on MLP layers only (research shows this can outperform attention-only)

## Configuration Structure

### Model Config

```yaml
model:
  name: "Qwen/Qwen2.5-VL-3B-Instruct"  # HuggingFace model identifier
  device: "cuda"  # "cuda", "cpu", "mps"
  torch_dtype: "float16"  # "float16", "bfloat16", "float32"

  # LoRA settings
  use_lora: true
  lora_target_option: "attention+mlp"  # See LoRA targeting options below
  lora_rank: 32
  lora_alpha: 32
  lora_dropout: 0.05
  lora_bias: "none"

  # Generation settings
  max_new_tokens: 128
  temperature: 0.7
  do_sample: true

  freeze_vision_encoder: false  # Set true to only train language model
```

#### LoRA Targeting Options

- **`"attention"`** - Only attention layers (q_proj, k_proj, v_proj, o_proj)
- **`"mlp"`** - Only MLP layers (gate_proj, up_proj, down_proj)
- **`"attention+mlp"`** - Both attention and MLP (**recommended**)
- **`"all-linear"`** - All linear layers (automatic detection)
- **`"custom"`** - Specify custom modules via `lora_custom_modules`

### Trainer Config

```yaml
trainer:
  batch_size: 4  # Number of trajectories per training batch
  learning_rate: null  # null = auto-detect (2e-4 for LoRA, 1e-5 for full)
  num_training_steps: 100
  save_every: 10  # Save checkpoint every N steps
  queue_timeout: 5.0  # Timeout when waiting for trajectories

  algorithm: "rejection_sampling"  # "rejection_sampling" or "ppo"
  reward_threshold: 0.0  # For rejection sampling
```

### Actor Config

```yaml
actor:
  max_steps_per_episode: 50
  action_format: "json"  # "json", "text", "coordinates"
  screenshot_size: [224, 224]  # Not currently used (VLM handles resizing)
  task_prompt: "Complete the data entry task"
  session_type: "simple_data_entry"
```

### Actor Pool Config

```yaml
actor_pool:
  target_concurrent_actors: 2  # Number of actors to run concurrently
  max_concurrent_per_vm: 2  # Memory limit per VM
  monitor_interval: 2.0  # Health check interval (seconds)
```

### Environment Config

```yaml
environment:
  vm_urls:
    - "http://localhost:8000"  # List of VM URLs for load balancing
  timeout: 30
  max_retries: 3
```

### Logging Config

```yaml
logging:
  log_level: "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
  log_dir: "experiments/my_experiment/logs"
  checkpoint_dir: "experiments/my_experiment/checkpoints"
  trajectory_dir: null  # Optional: save trajectories to disk
  verbose: false
```

## Creating Custom Configs

1. **Copy an existing config** as a starting point
2. **Modify parameters** for your experiment
3. **Save with a descriptive name** (e.g., `my_experiment.yaml`)
4. **Run training** with your config

Example:

```bash
# Copy baseline
cp config/qwen25_vl_lora.yaml config/my_experiment.yaml

# Edit config
vim config/my_experiment.yaml

# Run training
python scripts/train_with_config.py --config config/my_experiment.yaml
```

## Best Practices

### Experiment Organization

Organize configs by purpose:

```
config/
├── qwen25_vl_lora.yaml           # Baseline configs
├── qwen25_vl_full_finetune.yaml
├── multi_vm_distributed.yaml
└── experiments/                   # Research/ablation studies
    ├── lora_ablation_attention_only.yaml
    ├── lora_ablation_mlp_only.yaml
    └── hyperparameter_sweep/
        ├── rank_16.yaml
        ├── rank_32.yaml
        └── rank_64.yaml
```

### Naming Conventions

- Use descriptive names: `qwen25_vl_lora_rank64.yaml` not `config1.yaml`
- Include key parameters in name: `attention_only_lr2e-4.yaml`
- Group related experiments in subdirectories

### Version Control

- **Commit configs with code** for reproducibility
- **Tag configs** when publishing results
- **Document changes** in commit messages

### Command-Line Overrides

You can override config values via command line:

```bash
# Override VM URL
--vm-url http://34.123.45.67:8000

# Override training steps
--num-steps 500

# Override checkpoint directory
--checkpoint-dir /path/to/checkpoints

# Enable verbose logging
--verbose
```

This is useful for:
- Quick testing without modifying config files
- Using same config on different VMs
- CI/CD pipelines

## Configuration Validation

Configs are automatically validated on load. The validation checks:

- ✅ Valid LoRA targeting option
- ✅ Custom modules provided if lora_target_option="custom"
- ✅ Valid algorithm name
- ✅ Positive batch size and actor counts
- ✅ At least one VM URL provided

Invalid configs will raise a `ValueError` with a descriptive error message.

## Loading Configs Programmatically

You can also use the config system in your own scripts:

```python
from src.config import Config

# Load from YAML
config = Config.from_yaml("config/qwen25_vl_lora.yaml")

# Validate
config.validate()

# Access nested configs
print(config.model.name)
print(config.trainer.batch_size)

# Save to YAML
config.to_yaml("experiments/my_run/config.yaml")

# Print summary
print(config)  # Pretty-printed overview
```

## Troubleshooting

### Config not found

```
Error: Config file not found: config/my_config.yaml
```

**Solution:** Check the path is correct relative to project root.

### Validation error

```
Config validation error: Invalid lora_target_option: foo
```

**Solution:** Check config values match allowed options (see docs above).

### Module import error

```
ModuleNotFoundError: No module named 'src.config'
```

**Solution:** Run from project root, not from config directory.

## Examples

### Running LoRA Ablation Study

```bash
# Test attention-only
python scripts/train_with_config.py \
    --config config/experiments/lora_ablation_attention_only.yaml \
    --vm-url http://vm-1:8000

# Test MLP-only
python scripts/train_with_config.py \
    --config config/experiments/lora_ablation_mlp_only.yaml \
    --vm-url http://vm-1:8000

# Test attention+mlp (baseline)
python scripts/train_with_config.py \
    --config config/qwen25_vl_lora.yaml \
    --vm-url http://vm-1:8000
```

### Multi-VM Training

```bash
# Edit config to specify multiple VMs
vim config/multi_vm_distributed.yaml

# Or override via command line
python scripts/train_with_config.py \
    --config config/qwen25_vl_lora.yaml \
    --vm-url "http://vm-1:8000,http://vm-2:8000,http://vm-3:8000,http://vm-4:8000"
```

## See Also

- [Examples](../examples/) - Example training scripts
- [Documentation](../docs/) - Architecture and design docs
- [Source Code](../src/) - Implementation details
