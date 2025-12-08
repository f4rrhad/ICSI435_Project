
import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import pandas as pd
from datasets import load_dataset # Import load_dataset

class InstructBLIP:
    def __init__(self, device="cuda"):
        self.device = device
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", torch_dtype=torch.bfloat16).to(device)

    def inference(self, image_path, prompt):
        """
        Performs inference with the InstructBLIP model.

        Args:
            image_path (str): Path to the image file.
            prompt (str): The prompt text.

        Returns:
            str: The generated response.
        """
        try:
            # Handle image path from Hugging Face dataset
            if image_path.startswith("hf://"):
                # Assuming the image is in the dataset's directory structure
                # This might need adjustment based on the dataset's actual structure
                dataset_path = "/".join(image_path.split("/")[:4]) # e.g., hf://datasets/JailbreakV-28K/JailBreakV-28k
                image_relative_path = "/".join(image_path.split("/")[4:])
                local_image_path = f"/root/.cache/huggingface/datasets/{dataset_path.replace('hf://datasets/', '').replace('/', '_')}/{image_relative_path}"
                image = Image.open(local_image_path).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")

        except FileNotFoundError:
            return f"Error: Image file not found at {image_path}"
        except Exception as e:
            return f"Error loading image: {e}"


        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        # Generate response with attention_mask
        outputs = self.model.generate(
            **inputs,
            num_beams=5,
            max_new_tokens=100,
            attention_mask=inputs["attention_mask"] # Explicitly pass attention_mask
        )
        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return response

if __name__ == '__main__':
    # Example usage (replace with your actual image path and prompt)
    model = InstructBLIP()

    # Load the mini_JailBreakV_28K dataset
    dataset = load_dataset("JailbreakV-28K/JailBreakV-28k", 'JailBreakV_28K')["mini_JailBreakV_28K"]

    # Get the image path and prompt from the first example
    image_path = dataset[0]["image_path"]
    prompt = dataset[0]["jailbreak_query"]


    response = model.inference(image_path, prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
