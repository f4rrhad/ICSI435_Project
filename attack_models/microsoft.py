import os
import torch
import warnings
import requests
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import attack_model


# Silence transformers / torch deprecation noise
warnings.filterwarnings("ignore")


class microsoft(attack_model.attack_model):
    """
    Microsoft Phi-4-Multimodal-Instruct wrapper.
    Supports text-only and vision+text inference.
    Disables FlashAttention2 + LoRA/PEFT for full CUDA/Colab compatibility.
    """

    def __init__(self, device=None):
        # ---------------------------------------------------------------
        # 🔹 Device setup
        # ---------------------------------------------------------------
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)
        print(f"[INIT] Loading microsoft/Phi-4-multimodal-instruct on {self.device}")

        # ---------------------------------------------------------------
        # 🔹 Disable features that cause binary / ABI conflicts
        # ---------------------------------------------------------------
        os.environ["USE_FLASH_ATTENTION"] = "0"
        os.environ["USE_PEFT"] = "0"

        # ---------------------------------------------------------------
        # 🔹 Load processor
        # ---------------------------------------------------------------
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Phi-4-multimodal-instruct",
            trust_remote_code=True
        )

        # ---------------------------------------------------------------
        # 🔹 Load model safely
        # ---------------------------------------------------------------
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-4-multimodal-instruct",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            _attn_implementation="eager",     # disables FlashAttention2
            use_safetensors=True
        )

        # ---------------------------------------------------------------
        # 🔹 Disable LoRA / PEFT adapters if they still load
        # ---------------------------------------------------------------
        if hasattr(self.model, "disable_adapters"):
            try:
                self.model.disable_adapters()
                print("[INFO] Disabled LoRA adapters for vision/speech.")
            except Exception:
                pass

        print("[LOAD] Phi-4 multimodal model and processor ready.")

    # ---------------------------------------------------------------
    # 🔹 Helper: Load image from URL or local path
    # ---------------------------------------------------------------
    def _load_image(self, path):
        if not path:
            return None
        try:
            if path.startswith(("http://", "https://")):
                r = requests.get(path, timeout=10)
                r.raise_for_status()
                return Image.open(BytesIO(r.content)).convert("RGB")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")
            return Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Image load failed: {e}")

    # ---------------------------------------------------------------
    # 🔹 Inference (text or image+text)
    # ---------------------------------------------------------------
    def inference(self, image_path, prompt):
        if not prompt and not image_path:
            return "Error: No image or prompt provided."

        image = None
        if image_path:
            try:
                image = self._load_image(image_path)
            except Exception as e:
                return f"Error loading image '{image_path}': {e}"

        # Format according to Phi-4 multimodal prompt structure
        if image is not None:
            formatted_prompt = f"<|user|><|image_1|>{prompt or 'Describe the image.'}<|end|><|assistant|>"
            inputs = self.processor(text=formatted_prompt, images=image, return_tensors="pt").to(self.device)
        else:
            formatted_prompt = f"<|user|>{prompt}<|end|><|assistant|>"
            inputs = self.processor(text=formatted_prompt, return_tensors="pt").to(self.device)

        # ---------------------------------------------------------------
        # 🔹 Generate safely
        # ---------------------------------------------------------------
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=150)
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return decoded.strip()
        except RuntimeError as re:
            torch.cuda.empty_cache()
            return f"Error during generation: {re}"
        except Exception as e:
            return f"Error during generation: {e}"


# ---------------------------------------------------------------
# 🔹 Stand-alone test block (optional)
# ---------------------------------------------------------------
if __name__ == "__main__":
    model = microsoft(device="cuda" if torch.cuda.is_available() else "cpu")

    print("\n[TEST] Example with image:")
    resp1 = model.inference(
        "llm_transfer_attack/SD_related_566.png",
        "Describe what you see in this image."
    )
    print("[RESULT]", resp1)

    print("\n[TEST] Example text-only:")
    resp2 = model.inference(None, "Tell me a story about a scientist and a dragon.")
    print("[RESULT]", resp2)
