import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import pandas as pd
import requests
from io import BytesIO

class Llama3:
    def __init__(self, device="cuda"):
        self.device = device
        # Replace with actual Llama 3 model and processor names
        # Make sure to use a model name that is compatible with AutoModelForCausalLM and AutoProcessor
        model_name = "meta-llama/Llama-3.2-1B" # Replace with an actual Llama 3 model name
        processor_name = "meta-llama/Llama-3.2-1B" # Replace with an actual Llama 3 processor name

        # Load model and processor and move them to the specified device
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
        self.processor = AutoProcessor.from_pretrained(processor_name)


    def inference(self, image_path, prompt):
        """
        Performs inference with the Llama3 model.

        Args:
            image_path (str): Path to the image file or URL.
            prompt (str): The prompt text.

        Returns:
            str: The generated response.
        """
        image = None
        if image_path:
            try:
                if image_path.startswith("http://") or image_path.startswith("https://"):
                    response = requests.get(image_path)
                    response.raise_for_status() # Raise an exception for bad status codes
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    image = Image.open(image_path).convert("RGB")
            except FileNotFoundError:
                return f"Error: Image file not found at {image_path}"
            except Exception as e:
                return f"Error loading image: {e}"

        if image is None and not prompt:
             return "Error: No image or prompt provided."

        # Prepare inputs for Llama 3 (adjust as needed for the specific model)
        # Conditionally pass the 'images' argument
        if hasattr(self.processor, 'image_processor') and image is not None:
             inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        else:
             inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)


        # Generate response (adjust parameters as needed)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            # Add other generation parameters as needed
        )
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response

if __name__ == '__main__':
    # Example usage (replace with your actual image path and prompt)
    # This example assumes a local image path for testing
    # You might need to adapt this to load from the dataset if preferred
    model = Llama3()

    # Replace with a valid local image path and prompt for testing
    test_image_path = "/content/drive/MyDrive/JailBreakV_28K/JailBreakV_28K/llm_transfer_attack/SD_related_566.png" # Example image path
    test_prompt = "Describe the image."

    response = model.inference(test_image_path, test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")

    # Example with no image
    test_prompt_no_image = "Tell me a story."
    response_no_image = model.inference(None, test_prompt_no_image)
    print(f"\nPrompt (no image): {test_prompt_no_image}")
    print(f"Response (no image): {response_no_image}")
