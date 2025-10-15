"""VLM (Vision-Language Model) wrapper for UI interaction using HuggingFace models."""

from typing import Dict, Any, Optional, List, Union
import logging
import torch
import torch.nn as nn
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class VLMWrapper(nn.Module):
    """
    Wrapper for HuggingFace Vision-Language Models for UI interaction.

    Supports models like:
    - Qwen2-VL
    - LLaVA
    - Idefics2
    - Florence-2
    - PaliGemma
    - Any HuggingFace VLM that follows Vision2Seq architecture

    Design:
    - Inherits from nn.Module for PyTorch compatibility
    - Handles image preprocessing via HuggingFace processor
    - Supports action generation from model outputs
    - Can be fine-tuned end-to-end or with LoRA
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.float16,
        use_lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        """
        Initialize VLM wrapper with HuggingFace model.

        Args:
            model_name: HuggingFace model identifier (e.g., "Qwen/Qwen2-VL-2B-Instruct")
            device: Device to load model on
            torch_dtype: Data type for model weights (float16 for efficiency)
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_config: LoRA configuration dict (rank, alpha, dropout, etc.)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling (vs greedy decoding)
        """
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        logger.info(f"Loading VLM: {model_name}")

        # Load processor (handles image + text preprocessing)
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load AutoProcessor, trying tokenizer: {e}")
            self.processor = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

        # Load model
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )

        # Apply LoRA if requested
        if use_lora:
            self._apply_lora(lora_config)

        self.model.to(device)

        logger.info(f"VLM loaded successfully on {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.2f}M")

    def _apply_lora(self, lora_config: Optional[Dict[str, Any]] = None):
        """Apply LoRA for parameter-efficient fine-tuning."""
        try:
            from peft import LoraConfig, get_peft_model

            if lora_config is None:
                lora_config = {
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM"
                }

            config = LoraConfig(**lora_config)
            self.model = get_peft_model(self.model, config)

            logger.info("LoRA applied successfully")
            logger.info(f"LoRA config: {lora_config}")

        except ImportError:
            logger.error("peft library not installed. Install with: pip install peft")
            raise

    def forward(
        self,
        images: Union[torch.Tensor, List[Image.Image], np.ndarray],
        prompts: Union[str, List[str]],
        return_loss: bool = True,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            images: Input images [B, C, H, W] or list of PIL Images or numpy arrays
            prompts: Text prompts (single string or batch)
            return_loss: Whether to compute loss (for training)
            labels: Optional ground truth labels for supervised learning

        Returns:
            Dictionary containing:
            - logits: Model output logits
            - loss: Loss value (if return_loss=True and labels provided)
            - hidden_states: Optional hidden states
        """
        # Preprocess inputs
        if isinstance(images, torch.Tensor):
            # Convert tensor to list of PIL images for processor
            images = self._tensor_to_pil_images(images)

        if isinstance(prompts, str):
            prompts = [prompts]

        # Process inputs through HuggingFace processor
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Forward pass
        if return_loss and labels is not None:
            inputs["labels"] = labels.to(self.device)

        outputs = self.model(**inputs)

        return {
            "logits": outputs.logits,
            "loss": outputs.loss if hasattr(outputs, "loss") else None,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None
        }

    def predict_action(
        self,
        images: Union[torch.Tensor, Image.Image, np.ndarray],
        prompt: str
    ) -> str:
        """
        Generate action from image and prompt (inference mode).

        Args:
            images: Input image (single image)
            prompt: Task prompt

        Returns:
            Generated action text
        """
        self.eval()

        with torch.no_grad():
            # Preprocess
            if isinstance(images, torch.Tensor):
                # Handle batch dimension
                if images.dim() == 4:
                    images = images[0]  # Take first image from batch
                images = self._tensor_to_pil_image(images)
            elif isinstance(images, np.ndarray):
                images = Image.fromarray(images.astype(np.uint8))

            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=images,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )

            # Decode
            generated_text = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )[0]

            # Extract action from generated text
            # Remove the prompt from the generated text
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()

            return generated_text

    def generate_action_logits(
        self,
        images: Union[torch.Tensor, List[Image.Image], np.ndarray],
        prompts: Union[str, List[str]],
        action_vocab: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate action logits for training (with teacher forcing).

        Args:
            images: Input images
            prompts: Text prompts
            action_vocab: Optional vocabulary of valid actions

        Returns:
            Dictionary with action_logits and other outputs
        """
        outputs = self.forward(images, prompts, return_loss=False)

        # Extract action-relevant logits
        # This depends on your action representation
        # Placeholder implementation
        action_logits = outputs["logits"][:, -1, :]  # Last token logits

        return {
            "action_logits": action_logits,
            "logits": outputs["logits"],
            "hidden_states": outputs["hidden_states"]
        }

    def _tensor_to_pil_images(self, tensor: torch.Tensor) -> List[Image.Image]:
        """Convert batch of tensors to list of PIL images."""
        images = []

        for i in range(tensor.shape[0]):
            img = tensor[i]
            img = self._tensor_to_pil_image(img)
            images.append(img)

        return images

    def _tensor_to_pil_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert single tensor to PIL image."""
        # Tensor is [C, H, W] or [H, W, C]
        if tensor.dim() == 3:
            if tensor.shape[0] in [1, 3]:  # [C, H, W]
                tensor = tensor.permute(1, 2, 0)  # -> [H, W, C]

        # Denormalize if needed (assuming 0-1 range)
        if tensor.max() <= 1.0:
            tensor = (tensor * 255).byte()

        # Convert to numpy
        img_np = tensor.cpu().numpy().astype(np.uint8)

        # Handle grayscale
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)

        return Image.fromarray(img_np)

    def save_pretrained(self, save_path: str):
        """Save model and processor."""
        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        logger.info("Model saved successfully")

    def load_pretrained(self, load_path: str):
        """Load model and processor from path."""
        logger.info(f"Loading model from {load_path}")
        self.model = AutoModelForVision2Seq.from_pretrained(
            load_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            load_path,
            trust_remote_code=True
        )
        logger.info("Model loaded successfully")

    def freeze_vision_encoder(self):
        """Freeze vision encoder parameters (only train language model)."""
        logger.info("Freezing vision encoder")
        for name, param in self.model.named_parameters():
            if "vision" in name.lower() or "visual" in name.lower():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters after freezing: {trainable / 1e6:.2f}M")

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        logger.info("Unfreezing all parameters")
        for param in self.model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters after unfreezing: {trainable / 1e6:.2f}M")

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_total_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.model.parameters())
