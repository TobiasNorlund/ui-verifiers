# UI-RL: Reinforcement Learning for Vision-Language Model UI Agents

A scalable framework for training Vision-Language Models (VLMs) to interact with user interfaces using Reinforcement Learning.

## Overview

This project implements a distributed RL training system for teaching VLMs to perform UI tasks. It features:

- **Actor-Learner Architecture**: Separate data collection (actors) and training (learner) for efficient scaling
- **Multiple RL Algorithms**: Support for Rejection Sampling, PPO, and other RL algorithms
- **HuggingFace Integration**: Train any HuggingFace Vision2Seq model (Qwen2-VL, LLaVA, Idefics2, etc.)
- **LoRA Fine-tuning**: Parameter-efficient training with PEFT
- **Multi-Actor Support**: Scale data collection across multiple environments
- **Distributed Training**: Orchestration for multi-VM training setup

## Architecture

```
┌─────────────────┐
│   UI Env (VM)   │
│   ┌─────────┐   │
│   │ Browser │   │
│   └────┬────┘   │
│        │        │
│   ┌────▼────┐   │
│   │ FastAPI │   │
│   └────┬────┘   │
└────────┼────────┘
         │
    ┌────▼─────┐
    │  Actor   │──┐
    │ (Runner) │  │
    └──────────┘  │
                  │ Trajectories
    ┌──────────┐  │    via Queue
    │  Actor   │──┤
    │ (Runner) │  │
    └──────────┘  │
                  │
         ┌────────▼────────┐
         │     Learner     │
         │    (Trainer)    │
         │                 │
         │  ┌───────────┐  │
         │  │    VLM    │  │
         │  │  (w/ LoRA)│  │
         │  └───────────┘  │
         └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ui-rl.git
cd ui-rl

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Quick Start

### 1. Configure Your Training

Edit `config/default_config.yaml` or `config/gcp_training.yaml` for GCP deployment:

```yaml
model:
  name: "Qwen/Qwen2-VL-2B-Instruct"
  use_lora: true
  action_format: "json"

environment:
  # Use GCP VM hostnames for ui-verifiers
  ui_env_urls:
    - "http://ui-verifier-vm-1:8000"
    - "http://ui-verifier-vm-2:8000"
    - "http://ui-verifier-vm-3:8000"
    - "http://ui-verifier-vm-4:8000"

actor:
  session_type: "simple_data_entry"
  task_prompt: "Complete the data entry form"
  max_steps_per_episode: 50

training:
  batch_size: 32
  num_iterations: 1000
```

**For GCP deployment**, see the [GCP Setup Guide](docs/GCP_SETUP.md) for complete instructions.

### 2. Run Training

```bash
# Single actor training
python scripts/train.py --config config/default_config.yaml

# Multi-actor training
python examples/train_multi_actor.py
```

### 3. Evaluate Your Model

```bash
python scripts/eval.py \
  --checkpoint experiments/exp_001/checkpoints/checkpoint_step_1000.pt \
  --tasks eval_tasks.json
```

## Project Structure

```
ui-rl/
├── src/
│   ├── actor/              # Data collection components
│   │   ├── task_runner.py  # Episode runner
│   │   └── env_client.py   # UI environment interface
│   ├── learner/            # Training components
│   │   ├── trainer.py      # Training loop
│   │   └── algorithms/     # RL algorithms
│   ├── models/             # VLM wrappers
│   ├── data_utils/         # Trajectory and data utilities
│   ├── parsers/            # Action parsing
│   └── orchestration/      # Multi-VM coordination
├── scripts/                # Executable scripts
├── config/                 # Configuration files
├── tests/                  # Unit tests
└── examples/               # Example training scripts
```

## Key Components

### TaskRunner (Actor)

Runs episodes in the UI environment and collects trajectories:

```python
from src.actor.task_runner import TaskRunner
from src.models.vlm_wrapper import VLMWrapper
import queue

vlm = VLMWrapper(model_name="Qwen/Qwen2-VL-2B-Instruct")
trajectory_queue = queue.Queue()

runner = TaskRunner(
    ui_env_url="http://ui-verifier-vm-1:8000",  # GCP VM hostname
    model=vlm,
    trajectory_queue=trajectory_queue,
    task_prompt="Complete the data entry form",
    session_type="simple_data_entry",
    action_format="json"
)
trajectory = runner.run_episode()
print(f"Collected {len(trajectory)} steps with reward {trajectory.total_reward()}")
```

### Trainer (Learner)

Trains the VLM using collected trajectories:

```python
from src.learner.trainer import Trainer
from src.learner.algorithms.rejection_sampling import RejectionSampling

algorithm = RejectionSampling(reward_threshold=0.0)
trainer = Trainer(
    model=vlm,
    algorithm=algorithm,
    batch_size=32,
    learning_rate=1e-5
)
metrics = trainer.train_step(trajectories)
```

### VLM Wrapper

Unified interface for HuggingFace vision-language models:

```python
from src.models.vlm_wrapper import VLMWrapper

vlm = VLMWrapper(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    use_lora=True,
    lora_config={"r": 16, "lora_alpha": 32}
)

# Inference
action_text = vlm.predict_action(screenshot, prompt="Click login")

# Training
loss = vlm.forward(images, prompts, labels=labels)["loss"]
```

## Algorithms

### Rejection Sampling

Simplest RL algorithm - only trains on successful trajectories:

```python
from src.learner.algorithms.rejection_sampling import RejectionSampling

algorithm = RejectionSampling(reward_threshold=0.0)
```

### PPO (Coming Soon)

Proximal Policy Optimization with clipped objective:

```python
from src.learner.algorithms.ppo import PPO

algorithm = PPO(clip_epsilon=0.2, gamma=0.99, gae_lambda=0.95)
```

## Scaling

### Multi-Actor Training

Run multiple actors in parallel for faster data collection:

```python
from src.orchestration.multi_actor_runner import MultiActorRunner

runner = MultiActorRunner(
    num_actors=4,
    env_urls=["http://vm1:8000", "http://vm2:8000", ...],
    model=vlm,
    trajectory_queue=queue
)
runner.start()
```

### Distributed Setup

For multi-VM training, see `src/orchestration/distributed_runner.py` and the [Architecture Decisions](ARCHITECTURE_DECISIONS.md) document.

## Documentation

- [GCP Setup Guide](docs/GCP_SETUP.md) - **Deploy training on Google Cloud Platform**
- [HuggingFace Setup Guide](HUGGINGFACE_SETUP.md)
- [Architecture Decisions](ARCHITECTURE_DECISIONS.md)
- [Integration Summary](INTEGRATION_SUMMARY.md)

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_trainer.py
```

### Code Style

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{ui_rl,
  title = {UI-RL: Reinforcement Learning for Vision-Language Model UI Agents},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/ui-rl}
}
```

## Acknowledgments

- HuggingFace Transformers for VLM infrastructure
- PEFT library for LoRA implementation
- PyTorch team for the ML framework
