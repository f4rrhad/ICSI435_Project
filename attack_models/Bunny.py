import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging
hf_logging.set_verbosity_error()

class Bunny:
    def __init__(self, device="cpu", model_name="gpt2"):
        self.device = device
        self.model_name = model_name
        print(f"Loading model {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)

    def inference(self, image_path, prompt, max_new_tokens=64):
        text = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers. USER: "
            + prompt + " ASSISTANT:"
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        generated = out[0][input_ids.shape[1]:]
        resp = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return resp
