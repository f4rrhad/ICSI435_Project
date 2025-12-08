import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import requests
from io import BytesIO

class Qwen:
    def __init__(self, device="cuda"):
        self.device = device
        model_name = "Qwen/Qwen3-VL-2B-Instruct"

        # Load model and processor
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def inference(self, image_path, prompt):
        """
        Run multimodal inference on Qwen (text + optional image).
        """
        messages = [{"role": "user", "content": []}]

        # Add image if provided
        if image_path:
            try:
                if image_path.startswith(("http://", "https://")):
                    response = requests.get(image_path)
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    img = Image.open(image_path).convert("RGB")
                messages[0]["content"].append({"type": "image", "image": img})
            except Exception as e:
                return f"Error loading image: {e}"

        # Add text
        messages[0]["content"].append({"type": "text", "text": prompt})

        # Apply chat template (creates correct <image> tokens internally)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=200)

        # Trim input tokens to get generated text only
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]

        # Decode
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        return response
