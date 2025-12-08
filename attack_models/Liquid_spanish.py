# attack_models/Liquid_spanish.py
import os
import time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import requests
from io import BytesIO

class Liquid_spanish:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        model_name = "LiquidAI/LFM2-VL-3B"

        # Load model and processor
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()

    def _load_image(self, image_path):
        """Return PIL.Image or raise Exception."""
        if not image_path or not isinstance(image_path, str) or not image_path.strip():
            raise FileNotFoundError("No image path provided")
        image_path = image_path.strip()

        if image_path.startswith(("http://", "https://")):
            resp = requests.get(image_path, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            return img
        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Local image not found: {image_path}")
            img = Image.open(image_path).convert("RGB")
            return img

    def inference(self, image_path, prompt):
        """
        Run multimodal inference in Spanish using LiquidAI/LFM2-VL-3B.
        This function ALWAYS returns a string. On error it returns a
        bracketed error string like: "[Error: ...]".
        """
        try:
            # Build conversation
            conversation = [{"role": "user", "content": []}]

            # Attach image if valid
            if image_path and isinstance(image_path, str) and image_path.strip():
                try:
                    img = self._load_image(image_path)
                    conversation[0]["content"].append({"type": "image", "image": img})
                except Exception as e:
                    # Return explicit error (caller can choose to retry)
                    return f"[Error loading image: {str(e)}]"

            # Validate prompt
            if not prompt or not isinstance(prompt, str) or not prompt.strip():
                return "[Error: empty or invalid prompt]"
            conversation[0]["content"].append({"type": "text", "text": prompt.strip()})

            # Prepare inputs (processor)
            try:
                inputs = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                )
                # Move inputs to device if supported
                try:
                    inputs = inputs.to(self.model.device)
                except Exception:
                    # Some processors return dict-like moving differently; ignore if not needed
                    pass
            except Exception as e:
                return f"[Error preparing input: {str(e)}]"

            # Generate safely
            try:
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=200)
            except Exception as e:
                return f"[Error during generation: {str(e)}]"

            # Validate outputs
            if outputs is None:
                return "[Error: model.generate() returned None]"
            try:
                decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
            except Exception as e:
                return f"[Error decoding output: {str(e)}]"

            if not decoded or decoded[0] is None or not str(decoded[0]).strip():
                return "[Error: decoded output was empty]"

            return str(decoded[0]).strip()

        except Exception as e:
            # Catchall to avoid returning None in any case
            return f"[Model error: {str(e)}]"
