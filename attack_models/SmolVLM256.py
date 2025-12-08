import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import requests
from io import BytesIO

class SmolVLM256:
    def __init__(self, device="cuda"):
        self.device = device
        model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        ).to(device)
        self.model.eval()

    def inference(self, image_paths=None, prompt=""):
        """
        Run multimodal inference on SmolVLM-256M-Instruct (text + optional images).
        image_paths: str, list of str, or None
        prompt: text prompt
        """
        messages = [{"role": "user", "content": []}]
        images = []

        # Normalize image_paths to list
        if image_paths:
            if isinstance(image_paths, str):
                image_paths = [image_paths]

            for path in image_paths:
                try:
                    if path.startswith(("http://", "https://")):
                        response = requests.get(path)
                        response.raise_for_status()
                        img = Image.open(BytesIO(response.content)).convert("RGB")
                    else:
                        img = Image.open(path).convert("RGB")
                    messages[0]["content"].append({"type": "image", "image": img})
                    images.append(img)
                except Exception as e:
                    return f"Error loading image: {e}"

        # Add text (force string)
        messages[0]["content"].append({"type": "text", "text": str(prompt)})

        # Apply chat template
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        # Prepare inputs
        inputs = self.processor(
            text=prompt_text,
            images=images if images else None,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=200)

        # Decode
        generated_text = self.processor.batch_decode(
            outputs, skip_special_tokens=True
        )[0]

        return generated_text


# Example usage
if __name__ == "__main__":
    model = SmolVLM256()
    output = model.inference(
        ["https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"],
        "Describe this image."
    )
    print("Model output:")
    print(output)
