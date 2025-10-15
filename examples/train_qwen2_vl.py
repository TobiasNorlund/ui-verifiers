#!/usr/bin/env python3
"""
Example: Training Qwen2-VL for UI interaction tasks

This example shows how to train a Qwen2-VL model with LoRA
for UI automation tasks.
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vlm_wrapper import VLMWrapper, ActionDecoder
from src.learner.algorithms.rejection_sampling import RejectionSampling
from scripts.train import train_with_orchestration


def main():
    print("=" * 60)
    print("Training Qwen2-VL for UI Interaction")
    print("=" * 60)

    # 1. Initialize VLM with LoRA
    print("\n[1/4] Initializing Qwen2-VL model with LoRA...")

    vlm = VLMWrapper(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
        use_lora=True,
        lora_config={
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none"
        },
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True
    )

    print(f"   Model loaded: {vlm.get_total_parameters() / 1e6:.2f}M total parameters")
    print(f"   Trainable: {vlm.get_trainable_parameters() / 1e6:.2f}M parameters (with LoRA)")

    # 2. Create algorithm
    print("\n[2/4] Setting up Rejection Sampling algorithm...")

    algorithm = RejectionSampling(reward_threshold=0.5)
    print(f"   Reward threshold: {algorithm.reward_threshold}")

    # 3. Configure training
    print("\n[3/4] Configuring training...")

    task_prompt = """You are a UI automation agent. Given a screenshot and task, generate the next action.

Available actions:
- click(x, y): Click at coordinates
- type("text"): Type text
- scroll(direction, amount): Scroll (direction: up/down)

Task: {task}
Action:"""

    ui_env_url = "http://localhost:8000"  # Your UI environment URL
    checkpoint_dir = Path("experiments/qwen2_vl_ui/checkpoints")
    data_dir = Path("experiments/qwen2_vl_ui/trajectories")

    print(f"   UI Environment: {ui_env_url}")
    print(f"   Checkpoint dir: {checkpoint_dir}")
    print(f"   Data dir: {data_dir}")

    # 4. Start training
    print("\n[4/4] Starting training loop...")
    print("-" * 60)

    try:
        train_with_orchestration(
            model=vlm,
            ui_env_url=ui_env_url,
            task_prompt=task_prompt,
            algorithm=algorithm,
            num_training_steps=1000,
            batch_size=16,
            checkpoint_every=100,
            checkpoint_dir=checkpoint_dir,
            data_dir=data_dir
        )

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Checkpoint saved at:", checkpoint_dir)

    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        raise


def test_inference():
    """Test inference with trained model."""
    print("\n" + "=" * 60)
    print("Testing Inference")
    print("=" * 60)

    # Load model
    vlm = VLMWrapper(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16
    )

    # Or load from checkpoint
    # vlm.load_pretrained("experiments/qwen2_vl_ui/checkpoints/final_checkpoint")

    # Create dummy screenshot (replace with actual screenshot)
    import numpy as np
    screenshot = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Generate action
    prompt = "Click the login button"
    action_text = vlm.predict_action(screenshot, prompt)

    print(f"\nPrompt: {prompt}")
    print(f"Generated: {action_text}")

    # Decode action
    decoder = ActionDecoder(action_format="text")
    action_dict = decoder.decode(action_text)
    print(f"Parsed action: {action_dict}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test Qwen2-VL for UI tasks")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="Mode: train or test")

    args = parser.parse_args()

    if args.mode == "train":
        main()
    else:
        test_inference()
