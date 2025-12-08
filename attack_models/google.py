import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
from io import BytesIO
import attack_model


class google(attack_model.attack_model):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)

        print(f"[INIT] Loading google/gemma-3-4b-it on {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            "google/gemma-3-4b-it",
            trust_remote_code=True
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            "google/gemma-3-4b-it",
            torch_dtype=torch.float32,  # safer dtype
            device_map="auto",
            trust_remote_code=True
        )

    def _load_image(self, path):
        """Load image from local path or URL."""
        if not path:
            return None
        try:
            if path.startswith(("http://", "https://")):
                r = requests.get(path, timeout=10)
                r.raise_for_status()
                img = Image.open(BytesIO(r.content)).convert("RGB")
            else:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Not found: {path}")
                img = Image.open(path).convert("RGB")
            return img
        except Exception as e:
            raise RuntimeError(f"Image load failed: {e}")

    def inference(self, image_path, prompt):
        """Run inference for one sample."""
        if not prompt and not image_path:
            return "Error: No image or prompt provided."

        # --- Load image as PIL only ---
        image = None
        if image_path:
            try:
                image = self._load_image(image_path)
            except Exception as e:
                return f"Error loading image '{image_path}': {e}"

        # --- Build content for chat template ---
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})  # ✅ Pass raw PIL image
        if prompt:
            content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        # --- Tokenize & prepare inputs ---
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                padding=True,        # ✅ required for batch consistency
                truncation=True,     # ✅ prevent overflow
                return_tensors="pt",
                return_dict=True
            )
        except Exception as e:
            return f"Error preparing inputs for model: {e}"

        # --- Move tensors to device ---
        try:
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
        except Exception as e:
            return f"Error moving inputs to device: {e}"

        # --- Generate safely ---
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=100)
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            return response.strip()
        except RuntimeError as re:
            err = str(re)
            print("[ERROR] RuntimeError:", err)
            if "CUDA" in err:
                torch.cuda.empty_cache()
            return f"Error during generation: {err}"
        except Exception as e:
            return f"Error during generation: {e}"


if __name__ == "__main__":
    model = google(device="cuda" if torch.cuda.is_available() else "cpu")

    print("\n[TEST] Example with image:")
    resp1 = model.inference(
        "llm_transfer_attack/SD_related_566.png",
        "Describe the image."
    )
    print("[RESULT]", resp1)

    print("\n[TEST] Example text-only:")
    resp2 = model.inference(None, "Tell me a story about a cat and a robot.")
    print("[RESULT]", resp2)
