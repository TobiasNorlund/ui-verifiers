# HuggingFace VLM Setup Guide

This guide explains how to train local HuggingFace Vision-Language Models for UI interaction tasks.

## Overview

The codebase now supports training open-source VLMs from HuggingFace instead of using API-based models. This provides:

- **Full control** over model training and fine-tuning
- **Cost efficiency** - no API costs
- **Privacy** - all data stays local
- **Flexibility** - customize model architecture and training
- **LoRA support** - efficient parameter-efficient fine-tuning

## Supported Models

The `VLMWrapper` supports any HuggingFace model that follows the Vision2Seq architecture:

### Recommended Models:

1. **Qwen2-VL** (Recommended for UI tasks)
   - `Qwen/Qwen2-VL-2B-Instruct` (2B parameters)
   - `Qwen/Qwen2-VL-7B-Instruct` (7B parameters)
   - Strong vision-language understanding
   - Good instruction following

2. **LLaVA**
   - `llava-hf/llava-1.5-7b-hf` (7B parameters)
   - `llava-hf/llava-1.5-13b-hf` (13B parameters)
   - Well-tested for vision tasks

3. **Idefics2**
   - `HuggingFaceM4/idefics2-8b` (8B parameters)
   - Multimodal reasoning capabilities

4. **Florence-2**
   - `microsoft/Florence-2-large` (0.7B parameters)
   - Efficient and fast

5. **PaliGemma**
   - `google/paligemma-3b-pt-224` (3B parameters)
   - Good for visual grounding

## Installation

### 1. Install Dependencies

```bash
# Install base requirements
pip install -e .

# For GPU support (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For LoRA fine-tuning
pip install peft

# For quantization (optional, saves memory)
pip install bitsandbytes
```

### 2. Download a Model

Models will be automatically downloaded on first use, but you can pre-download:

```python
from transformers import AutoModelForVision2Seq, AutoProcessor

model_name = "Qwen/Qwen2-VL-2B-Instruct"
model = AutoModelForVision2Seq.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
```

## Configuration

### Basic Configuration

Edit `config/default_config.yaml`:

```yaml
model:
  # Choose your model
  name: "Qwen/Qwen2-VL-2B-Instruct"

  # Model loading
  torch_dtype: "float16"  # float16 for speed, float32 for accuracy
  device: "cuda"

  # LoRA for efficient fine-tuning
  use_lora: true
  lora_config:
    r: 16  # LoRA rank (8-32 typical)
    lora_alpha: 32  # LoRA alpha (usually 2*r)
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
    lora_dropout: 0.05
    bias: "none"

  # Generation settings
  max_new_tokens: 128
  temperature: 0.7
  do_sample: true

  # Training options
  freeze_vision_encoder: false

  # Action format
  action_format: "text"  # "json", "text", or "coordinates"
```

### Action Formats

The model can generate actions in different formats:

#### 1. Text Format (Recommended)
```yaml
action_format: "text"
```

Model generates: `click(100, 200)`, `type("hello")`, `scroll(down, 100)`

#### 2. JSON Format
```yaml
action_format: "json"
```

Model generates: `{"type": "click", "x": 100, "y": 200}`

#### 3. Coordinates Format
```yaml
action_format: "coordinates"
```

Model generates: `100 200 click`

## Training

### 1. Basic Training

```bash
python scripts/train.py \
  --config config/default_config.yaml \
  --experiment-name my_ui_agent
```

### 2. With LoRA (Recommended)

LoRA reduces trainable parameters from billions to millions, enabling faster training on consumer GPUs:

```yaml
model:
  use_lora: true
  lora_config:
    r: 16  # Lower for faster training, higher for better quality
    lora_alpha: 32
```

**Memory Requirements:**
- 2B model with LoRA: ~8GB VRAM
- 7B model with LoRA: ~16GB VRAM
- 13B model with LoRA: ~24GB VRAM

### 3. Freeze Vision Encoder

To only train the language model (faster, less memory):

```yaml
model:
  freeze_vision_encoder: true
```

### 4. Multi-GPU Training

```yaml
model:
  device: "auto"  # Automatically distribute across GPUs
```

## Model Architecture

### VLMWrapper API

```python
from src.models.vlm_wrapper import VLMWrapper

# Initialize
vlm = VLMWrapper(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    device="cuda",
    torch_dtype=torch.float16,
    use_lora=True
)

# Inference
action_text = vlm.predict_action(
    images=screenshot,  # numpy array or PIL Image
    prompt="Click the login button"
)

# Training (called by Trainer)
outputs = vlm.forward(
    images=batch_images,
    prompts=batch_prompts,
    labels=batch_labels
)
loss = outputs["loss"]
```

### Key Methods

- `forward()`: Training forward pass with loss computation
- `predict_action()`: Inference mode for action generation
- `save_pretrained()`: Save model checkpoint
- `load_pretrained()`: Load model checkpoint
- `freeze_vision_encoder()`: Freeze vision parameters
- `get_trainable_parameters()`: Check parameter count

### ActionDecoder

```python
from src.models.vlm_wrapper import ActionDecoder

decoder = ActionDecoder(action_format="text")
action_dict = decoder.decode("click(100, 200)")
# Returns: {"type": "click", "x": 100, "y": 200}
```

## Prompting Strategy

### System Prompt

Include clear instructions in your task prompt:

```yaml
task_prompt: |
  You are a UI automation agent. Given a screenshot and a task, generate the next action.

  Task: {task_description}

  Available actions:
  - click(x, y): Click at coordinates (x, y)
  - type(text): Type the given text
  - scroll(direction, amount): Scroll (direction: up/down, amount: pixels)
  - wait(seconds): Wait for given seconds

  Generate the next action:
```

### Few-Shot Examples

Add examples to improve performance:

```yaml
task_prompt: |
  Examples:
  Task: Click the login button
  Screenshot: [shows login button at 150, 200]
  Action: click(150, 200)

  Task: Fill username field
  Screenshot: [shows input field]
  Action: type("user@example.com")

  Now, for this task:
  Task: {task_description}
  Action:
```

## Training Tips

### 1. Start Small
- Begin with 2B model
- Use LoRA with r=8
- Freeze vision encoder initially

### 2. Monitor Training
```bash
# Watch logs
tail -f experiments/my_exp/logs/train.log

# TensorBoard
tensorboard --logdir experiments/my_exp/logs
```

### 3. Checkpoint Strategy
```yaml
training:
  checkpoint_interval: 100  # Save every N steps
```

### 4. Learning Rate
```yaml
algorithm:
  learning_rate: 0.0001  # Higher for from-scratch, lower for fine-tuning
```

### 5. Batch Size
- Start small (8-16) and increase if memory allows
- Larger batches = more stable training

## Memory Optimization

### 1. Mixed Precision Training
```yaml
model:
  torch_dtype: "float16"  # or "bfloat16" on Ampere GPUs
```

### 2. Gradient Checkpointing
Enable in model config (saves memory at cost of speed):

```python
vlm.model.gradient_checkpointing_enable()
```

### 3. Quantization
For inference only (not during training):

```python
# 8-bit quantization
vlm = VLMWrapper(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    load_in_8bit=True
)
```

### 4. Smaller LoRA Rank
```yaml
lora_config:
  r: 8  # Instead of 16
```

## Troubleshooting

### Out of Memory

**Solution 1**: Reduce batch size
```yaml
training:
  batch_size: 8  # Down from 32
```

**Solution 2**: Use gradient accumulation
```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch size: 32
```

**Solution 3**: Smaller model
```yaml
model:
  name: "Qwen/Qwen2-VL-2B-Instruct"  # Instead of 7B
```

### Model Not Loading

Check HuggingFace authentication:
```bash
huggingface-cli login
```

Some models require accepting license agreements on HuggingFace.

### Slow Training

**Solution 1**: Use LoRA
```yaml
model:
  use_lora: true
```

**Solution 2**: Freeze vision encoder
```yaml
model:
  freeze_vision_encoder: true
```

**Solution 3**: Mixed precision
```yaml
model:
  torch_dtype: "float16"
```

## Advanced: Custom Models

To use a custom architecture:

1. Ensure it's compatible with `AutoModelForVision2Seq`
2. Or extend `VLMWrapper` class:

```python
class CustomVLMWrapper(VLMWrapper):
    def __init__(self, model_name, **kwargs):
        # Custom initialization
        pass

    def forward(self, images, prompts, **kwargs):
        # Custom forward pass
        pass
```

## Evaluation

Evaluate trained model:

```bash
python scripts/eval.py \
  --checkpoint experiments/my_exp/checkpoints/checkpoint_step_1000.pt \
  --tasks data/eval_tasks.json
```

## Saving and Sharing Models

### Save Model
```python
vlm.save_pretrained("my_trained_model")
```

### Load Model
```python
vlm = VLMWrapper(model_name="my_trained_model")
```

### Share on HuggingFace
```bash
# Upload to HuggingFace Hub
huggingface-cli upload my_username/my_ui_agent my_trained_model/
```

## Next Steps

1. ✅ Set up configuration for your chosen model
2. ✅ Prepare your UI environment API
3. ✅ Start training with small batch size
4. ✅ Monitor metrics and adjust hyperparameters
5. ✅ Scale up model size and batch size as needed
6. ✅ Evaluate on held-out tasks

## Resources

- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [PEFT (LoRA) Documentation](https://huggingface.co/docs/peft)
- [Qwen2-VL Model Card](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [LLaVA Model Card](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
