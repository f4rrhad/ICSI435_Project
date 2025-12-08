import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import requests
from io import BytesIO

class Liquid:
    def __init__(self, device="cuda"):
        self.device = device
        model_name = "LiquidAI/LFM2-VL-3B"

        # Load model and processor
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def inference(self, image_path, prompt):
        """
        Run multimodal inference using LiquidAI/LFM2-VL-3B (text + optional image).
        """
        conversation = [{"role": "user", "content": []}]

        # Add image if provided
        if image_path:
            try:
                if image_path.startswith(("http://", "https://")):
                    response = requests.get(image_path)
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    img = Image.open(image_path).convert("RGB")

                conversation[0]["content"].append({"type": "image", "image": img})
            except Exception as e:
                return f"Error loading image: {e}"

        # Add text
        conversation[0]["content"].append({"type": "text", "text": prompt})

        # Convert conversation to model input
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=200)

        # Decode
        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return response
