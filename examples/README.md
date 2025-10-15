# Examples

This directory contains example scripts demonstrating how to use the UI VLM training framework.

## Available Examples

### 1. `train_qwen2_vl.py` - Training Qwen2-VL for UI Tasks

Complete example showing how to:
- Initialize Qwen2-VL model with LoRA
- Configure rejection sampling algorithm
- Train on UI interaction tasks
- Test inference with trained model

**Usage:**

```bash
# Training
python examples/train_qwen2_vl.py --mode train

# Testing inference
python examples/train_qwen2_vl.py --mode test
```

**What it demonstrates:**
- HuggingFace model initialization
- LoRA configuration for efficient fine-tuning
- Training orchestration with actor-learner architecture
- Action generation and parsing
- Checkpoint management

## Quick Start

### Prerequisites

1. Install dependencies:
```bash
pip install -e .
```

2. Set up a UI environment (see main README.md)

3. Configure your settings in `config/default_config.yaml`

### Running an Example

```bash
# Make sure you're in the project root
cd /path/to/ui-rl

# Run the example
python examples/train_qwen2_vl.py
```

## Creating Your Own Example

Template for a new example:

```python
#!/usr/bin/env python3
"""Your example description."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vlm_wrapper import VLMWrapper
from src.learner.algorithms.rejection_sampling import RejectionSampling
# ... other imports

def main():
    # 1. Initialize model
    vlm = VLMWrapper(
        model_name="your-model-name",
        use_lora=True
    )

    # 2. Create algorithm
    algorithm = RejectionSampling(reward_threshold=0.5)

    # 3. Train
    # ... your training code

if __name__ == "__main__":
    main()
```

## Example Configurations

### Small Model (Fast Training)

```python
vlm = VLMWrapper(
    model_name="Qwen/Qwen2-VL-2B-Instruct",  # 2B parameters
    use_lora=True,
    lora_config={"r": 8}  # Lower rank = faster
)
```

### Large Model (Better Quality)

```python
vlm = VLMWrapper(
    model_name="Qwen/Qwen2-VL-7B-Instruct",  # 7B parameters
    use_lora=True,
    lora_config={"r": 32}  # Higher rank = better
)
```

### Vision Encoder Frozen (Faster)

```python
vlm = VLMWrapper(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    use_lora=True
)
vlm.freeze_vision_encoder()  # Only train language model
```

## Tips

1. **Start small**: Use 2B model with LoRA rank 8
2. **Monitor logs**: Check `experiments/*/logs/train.log`
3. **Test quickly**: Use `--mode test` to verify model behavior
4. **Adjust batch size**: Start with 8-16, increase if memory allows
5. **Save checkpoints**: Default is every 100 steps

## Troubleshooting

**Out of memory?**
- Reduce batch size in the script
- Use smaller model (2B instead of 7B)
- Lower LoRA rank (8 instead of 16)

**Model not downloading?**
```bash
huggingface-cli login
```

**Slow training?**
- Enable LoRA: `use_lora=True`
- Freeze vision encoder
- Use mixed precision: `torch_dtype=torch.float16`

## Next Steps

After running the examples:
1. Modify for your specific UI tasks
2. Adjust hyperparameters in config files
3. Implement your custom UI environment
4. Scale up to larger models
5. Deploy trained model

## Contributing

Have a useful example? Submit a PR with:
- Clear documentation
- Minimal dependencies
- Comments explaining key concepts
