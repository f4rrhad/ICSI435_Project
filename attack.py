# attack.py
import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
import sys

# Import attack models (ensure attack_models is a package and in PYTHONPATH)
from attack_models.Bunny import Bunny
#from attack_models.InstructBLIP import InstructBLIP
#from attack_models.Llama4Scout import Llama4Scout
#from attack_models.google import google
#from attack_models.Llama3 import Llama3
from attack_models.microsoft import microsoft  
from attack_models.SmolVLM256 import SmolVLM256


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Instantiate selected model
    model = None
    name = args.model.lower()
    if name == "bunny":
        model = Bunny(device=device)
    elif name == "microsoft":  
        model = microsoft(device=device)
    elif name == "smolvlm256":  
        model = SmolVLM256(device=device)
    else:
        print(f"Error: Model '{args.model}' not supported.")
        sys.exit(1)

    print(f"Using model: {args.model}")

    # Load dataset
    try:
        df = pd.read_csv(args.data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {args.data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data file: {e}")
        sys.exit(1)

    # Quick diagnostics (optional, helpful)
    print(f"Total rows: {len(df)}")
    if "image_path" in df.columns:
        print("image_path nulls:", df["image_path"].isnull().sum())
    if "jailbreak_query" in df.columns:
        print("jailbreak_query nulls:", df["jailbreak_query"].isnull().sum())

    results = []
    print("Prompt → Response:")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        raw_image_path = row.get("image_path")
        raw_prompt = row.get("jailbreak_query")

        image_path = None
        prompt = None

        # Normalize image_path
        if isinstance(raw_image_path, str) and raw_image_path.strip() != "":
            image_path = raw_image_path.strip()

        # Normalize prompt
        if isinstance(raw_prompt, str) and raw_prompt.strip() != "":
            prompt = raw_prompt.strip()

        # If both missing, skip with clear error
        if not image_path and not prompt:
            print(f"\nSkipping example {index}: missing both image_path and jailbreak_query")
            results.append("Error: missing both image_path and jailbreak_query")
            continue

        # Run inference with robust try/except
        try:
            resp = model.inference(image_path, prompt)
            # Optional: print sample outputs for sanity check
            print(f"\n[IDX {index}] Prompt: {prompt[:120]}...")
            print(f"[Response]: {resp[:300]}")
        except Exception as e:
            print(f"\nError processing example {index}: {e}")
            resp = f"Error during inference: {e}"

        results.append(resp)

    # Append results and save
    df["response"] = results

    dataset_name = args.dataset if args.dataset else os.path.splitext(os.path.basename(args.data_path))[0]
    model_name_lower = args.model.lower()
    save_dir = os.path.join(args.save_path, dataset_name, model_name_lower)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}.csv")

    df.to_csv(save_path, index=False)
    print(f"\n✅ Saved results to: {save_path}")

    null_responses = df["response"].isnull().sum()
    print(f"Nulls: {null_responses}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attack scenarios against a specified model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset (optional).")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use (Bunny, InstructBLIP, google, microsoft, Llama3, Llama4Scout)."
    )

    args = parser.parse_args()
    main(args)
