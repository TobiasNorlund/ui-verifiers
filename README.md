# UI-RL: Reinforcement Learning for Vision-Language Model UI Agents

A scalable framework for training Vision-Language Models (VLMs) to interact with user interfaces using Reinforcement Learning.

## Overview

This project implements an **Actor-Learner** RL training system for teaching VLMs to perform UI tasks. The system features:

- **Actor-Learner Architecture**: Separate data collection (actors) and training (learner) for efficient scaling
- **Config-Based Training**: YAML configuration system for easy experimentation
- **Qwen2.5-VL Integration**: Default model with LoRA fine-tuning support
- **ActorPoolManager**: Automatic actor lifecycle management with VM resource limits
- **PIL-Based Pipeline**: Efficient image handling throughout the system
- **HuggingFace Models**: Support for any `AutoModelForImageTextToText` VLM (Qwen2-VL, LLaVA, Idefics2, etc.)
- **LoRA Fine-Tuning**: Parameter-efficient training with flexible layer targeting

### Current Implementation Status

✅ **Implemented:**
- TaskRunner (actor) for UI environment interaction
- Trainer (learner) with trajectory queue management
- VLMWrapper with LoRA support (attention, MLP, attention+mlp targeting)
- ActorPoolManager for multi-actor orchestration
- Rejection Sampling algorithm
- Configuration system with YAML files
- PIL-based image pipeline

🚧 **In Progress / TODO:** (see [TODO](#todo) section)

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Configuration (YAML)                      │
│         model, trainer, actor, actor_pool, environment       │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────┐     ┌──────────────────┐
│  UI Env (VM)    │     │ ActorPoolManager │
│  ┌───────────┐  │     │                  │
│  │  Browser  │  │     │  Manages pool    │
│  └─────┬─────┘  │     │  of actors with  │
│        │        │     │  VM limits       │
│  ┌─────▼─────┐  │     └────────┬─────────┘
│  │  FastAPI  │  │              │
│  │ (ui-verif)│  │              │ spawns/monitors
│  └─────┬─────┘  │              │
└────────┼────────┘              │
         │                       │
         │ screenshots/actions   ▼
         │              ┌─────────────┐
         └──────────────┤ TaskRunner  │──┐
                        │  (Actor)    │  │
                        └─────────────┘  │
                                         │
         ┌──────────────┬ TaskRunner  │──┤ Trajectories
         │              │  (Actor)    │  │ via Queue
         │              └─────────────┘  │
         │                               │
         │                               ▼
         │                    ┌──────────────────┐
         │                    │     Trainer      │
         │                    │    (Learner)     │
         │                    │                  │
         │                    │  ┌────────────┐  │
         └────────────────────┼──│ VLMWrapper │  │
           model updates      │  │ (w/ LoRA)  │  │
                              │  └────────────┘  │
                              └──────────────────┘
```

### Component Responsibilities

#### **TaskRunner** (src/actor/task_runner.py)
- Creates UI sessions via ui-verifiers API
- Runs VLM inference to get actions from screenshots
- Executes actions in the environment
- Collects complete trajectories (observations, actions, rewards)
- Puts finished trajectories in queue for training
- **Lifecycle**: Runs ONE episode, then exits (clean session management)

#### **Trainer** (src/learner/trainer.py)
- Owns the trajectory queue
- Waits for `batch_size` trajectories from actors
- Processes trajectories with algorithm (e.g., filter by reward)
- Computes loss and runs training step
- Handles checkpointing and metrics logging
- **Auto-detects learning rate**: 2e-4 for LoRA, 1e-5 for full fine-tuning

#### **VLMWrapper** (src/models/vlm_wrapper.py)
- Loads HuggingFace VLMs (`AutoModelForImageTextToText`)
- Applies LoRA with flexible layer targeting:
  - `"attention"`: q, k, v, o projections
  - `"mlp"`: gate, up, down projections
  - `"attention+mlp"`: both (recommended)
  - `"all-linear"`: all linear layers
  - `"custom"`: user-specified modules
- Handles inference (`predict_action`) and training (`forward`)
- Accepts PIL Images directly (no unnecessary conversions)

#### **ActorPoolManager** (src/orchestration/actor_pool_manager.py)
- Maintains pool of concurrent actors (TaskRunners)
- Enforces VM resource limits (`max_concurrent_per_vm`)
- Automatically spawns replacement actors when episodes finish
- Round-robin load balancing across multiple VMs
- Monitor thread for health checks and crash recovery
- **Design**: One actor = one episode = one VM session

#### **Config System** (src/config/)
- YAML-based configuration for all components
- Dataclass-based validation
- Easy experimentation with different hyperparameters
- See `config/README.md` for details

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+

### Setup with UV (Recommended)

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone <repository-url>
cd ui-rl

# Create virtual environment and install dependencies
uv sync

# Activate environment
source .venv/bin/activate  # Unix
# or
.venv\Scripts\activate  # Windows
```

See [UV_USAGE.md](UV_USAGE.md) for more details.

### Setup with pip

```bash
pip install -e .
```

## Quick Start

### 1. Configure Your Training

Edit `config/qwen25_vl_lora.yaml` or create your own config:

```yaml
experiment_name: "my_experiment"

model:
  name: "Qwen/Qwen2.5-VL-3B-Instruct"
  use_lora: true
  lora_target_option: "attention+mlp"  # Recommended
  lora_rank: 32

trainer:
  batch_size: 4
  learning_rate: null  # Auto-detect
  num_training_steps: 100

actor:
  max_steps_per_episode: 50
  task_prompt: "Complete the data entry task"
  session_type: "simple_data_entry"

actor_pool:
  target_concurrent_actors: 2
  max_concurrent_per_vm: 2

environment:
  vm_urls:
    - "http://your-vm-ip:8000"
```

See `config/README.md` for full configuration reference.

### 2. Run Training

```bash
# Train with config file
python scripts/train_with_config.py --config config/qwen25_vl_lora.yaml

# Override VM URL
python scripts/train_with_config.py \
    --config config/qwen25_vl_lora.yaml \
    --vm-url http://34.123.45.67:8000

# Override training steps
python scripts/train_with_config.py \
    --config config/qwen25_vl_lora.yaml \
    --num-steps 500
```

### 3. Test VM Connection

Before full training, test connection to ui-verifiers VM:

```bash
python scripts/test_vm_connection.py --vm-url http://your-vm-ip:8000
```

## Project Structure

```
ui-rl/
├── src/
│   ├── actor/
│   │   ├── __init__.py
│   │   └── task_runner.py         # Episode runner, VLM inference, action execution
│   ├── learner/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training loop, queue management
│   │   └── algorithms/
│   │       ├── base.py             # Algorithm interface
│   │       └── rejection_sampling.py  # Rejection sampling implementation
│   ├── models/
│   │   ├── __init__.py
│   │   └── vlm_wrapper.py          # HuggingFace VLM wrapper with LoRA
│   ├── orchestration/
│   │   ├── __init__.py
│   │   └── actor_pool_manager.py   # Multi-actor lifecycle management
│   ├── data_utils/
│   │   ├── __init__.py
│   │   ├── trajectory.py           # Trajectory data structure
│   │   ├── collation.py            # Batching trajectories
│   │   └── actions_decoder.py      # Parse VLM outputs to actions
│   └── config/
│       ├── __init__.py
│       └── config.py               # Configuration dataclasses
├── config/                          # YAML configuration files
│   ├── README.md                   # Config documentation
│   ├── qwen25_vl_lora.yaml        # Recommended baseline
│   ├── qwen25_vl_full_finetune.yaml
│   ├── multi_vm_distributed.yaml
│   └── experiments/                # Ablation study configs
│       ├── lora_ablation_attention_only.yaml
│       └── lora_ablation_mlp_only.yaml
├── scripts/
│   ├── train_with_config.py       # Main training script
│   ├── test_vm_connection.py      # Test VM connectivity
│   └── test_actor_learner.py      # Integration test
└── docs/                           # Additional documentation
```

## Usage Examples

### Basic Training Loop

```python
from src.config import Config
from src.models.vlm_wrapper import VLMWrapper
from src.learner.trainer import Trainer
from src.learner.algorithms.rejection_sampling import RejectionSampling
from src.orchestration import ActorPoolManager
import queue

# Load config
config = Config.from_yaml("config/qwen25_vl_lora.yaml")

# Create model with LoRA
model = VLMWrapper(
    model_name=config.model.name,
    use_lora=config.model.use_lora,
    lora_target_option=config.model.lora_target_option,
    lora_rank=config.model.lora_rank
)

# Create trainer
trajectory_queue = queue.Queue()
algorithm = RejectionSampling(reward_threshold=0.0)
trainer = Trainer(
    model=model,
    algorithm=algorithm,
    trajectory_queue=trajectory_queue,
    batch_size=config.trainer.batch_size
)

# Create actor pool
actor_pool = ActorPoolManager(
    target_concurrent_actors=config.actor_pool.target_concurrent_actors,
    vm_urls=config.environment.vm_urls,
    max_concurrent_per_vm=config.actor_pool.max_concurrent_per_vm,
    model=model,
    trajectory_queue=trajectory_queue,
    task_prompt=config.actor.task_prompt,
    session_type=config.actor.session_type,
    max_steps_per_episode=config.actor.max_steps_per_episode
)

# Start training
actor_pool.start()
trainer.train(num_steps=100, save_every=10)
actor_pool.stop()
```

### Run Single Episode

```python
from src.actor.task_runner import TaskRunner
from src.models.vlm_wrapper import VLMWrapper
import queue

vlm = VLMWrapper(model_name="Qwen/Qwen2.5-VL-3B-Instruct")
trajectory_queue = queue.Queue()

runner = TaskRunner(
    ui_env_url="http://your-vm:8000",
    model=vlm,
    trajectory_queue=trajectory_queue,
    task_prompt="Complete the data entry form",
    session_type="simple_data_entry"
)

trajectory = runner.run_episode()
print(f"Steps: {len(trajectory)}, Reward: {trajectory.total_reward()}")
```

## Algorithms

### Rejection Sampling (Implemented)

Trains only on trajectories that meet a reward threshold:

```python
from src.learner.algorithms.rejection_sampling import RejectionSampling

algorithm = RejectionSampling(reward_threshold=0.0)
```

**How it works:**
1. Collect trajectories from actors
2. Filter by reward: keep only if `total_reward > threshold`
3. Train on successful trajectories using supervised learning

## Configuration

The config system allows easy experimentation. Key features:

- **Model config**: Model selection, LoRA settings, generation parameters
- **Trainer config**: Batch size, learning rate, algorithm selection
- **Actor config**: Episode length, task prompts, action format
- **Actor pool config**: Concurrent actors, VM limits
- **Environment config**: VM URLs, timeouts

### Example Configs

```bash
# Baseline LoRA training
config/qwen25_vl_lora.yaml

# Full fine-tuning (no LoRA)
config/qwen25_vl_full_finetune.yaml

# Multi-VM distributed
config/multi_vm_distributed.yaml

# Ablation studies
config/experiments/lora_ablation_attention_only.yaml
config/experiments/lora_ablation_mlp_only.yaml
```

See `config/README.md` for complete documentation.

## Documentation

- [Configuration Guide](config/README.md) - Complete config reference
- [UV Package Manager](UV_USAGE.md) - Using UV for dependency management
- [HuggingFace Setup](HUGGINGFACE_SETUP.md) - Model integration details
- [Architecture Decisions](ARCHITECTURE_DECISIONS.md) - Design rationale
- [Integration Summary](INTEGRATION_SUMMARY.md) - System integration overview

## TODO

### High Priority
- [ ] **Evaluation Process**: Implement evaluation script and metrics
  - Need to define evaluation tasks
  - Create eval.py script
  - Metrics: success rate, average steps, action accuracy

- [ ] **UI-Verifiers API Returns**: Define reward and done signals
  - Currently using dummy values (`reward=1.0`, `done=True`)
  - Need to implement `/progress` endpoint protocol
  - Define task-specific reward functions

- [ ] **Actions Decoder**: Update for model-specific outputs
  - Current decoder is generic
  - Need to match Qwen2.5-VL output format
  - May need model-specific parsing logic

### Medium Priority
- [ ] **PPO Algorithm**: Implement Proximal Policy Optimization
  - More sophisticated than rejection sampling
  - Requires advantage estimation and value function
  - See `src/learner/algorithms/base.py` for interface

### Future Enhancements
- [ ] Trajectory replay buffer for off-policy learning
- [ ] Multi-task training support
- [ ] Distributed training across multiple machines
- [ ] W&B / TensorBoard integration for metrics
- [ ] Model checkpointing strategies (best, periodic, etc.)

## Development

### Code Structure Guidelines

Current structure follows "flat is better than nested":
- Single file per directory until complexity demands splitting
- Clear separation of concerns
- Simple import paths

**When to split a file:**
- Exceeds ~600 lines
- Multiple unrelated classes used independently
- 3+ implementations of same interface

### Running Tests

```bash
# Test VM connection
python scripts/test_vm_connection.py --vm-url http://your-vm:8000

# Test full actor-learner loop
python scripts/test_actor_learner.py \
    --vm-url http://your-vm:8000 \
    --num-trajectories 5 \
    --num-actors 2
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes following the existing code style
4. Test your changes
5. Commit with clear messages
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- HuggingFace Transformers for VLM infrastructure
- PEFT library for LoRA implementation
- PyTorch team for the ML framework
- Research papers on LoRA fine-tuning strategies
