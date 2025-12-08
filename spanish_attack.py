#!/usr/bin/env python3
# spanish_attack.py - attack runner with retries and robust handling

import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
import sys
import csv
import time

# Model imports (make sure attack_models package is on PYTHONPATH)
from attack_models.Bunny import Bunny
from attack_models.InstructBLIP import InstructBLIP
from attack_models.Llama4Scout import Llama4Scout
from attack_models.google import google
from attack_models.Llama3 import Llama3
from attack_models.microsoft import microsoft
from attack_models.Liquid import Liquid
from attack_models.Liquid_spanish import Liquid_spanish

def safe_str(x):
    if x is None:
        return "[Error: None response]"
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return "[Error: could not stringify response]"

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Instantiate selected model
    model = None
    name = args.model.lower()
    if name == "bunny":
        model = Bunny(device=device)
    elif name == "instructblip":
        model = InstructBLIP(device=device)
    elif name == "google":
        model = google(device=device)
    elif name == "llama3":
        model = Llama3(device=device)
    elif name == "llama4scout":
        model = Llama4Scout(device=device)
    elif name == "microsoft":
        model = microsoft(device=device)
    elif name == "liquid":
        model = Liquid(device=device)
    elif name in ("liquid_spanish", "liquid-spanish", "liquidspanish"):
        model = Liquid_spanish(device=device)
    else:
        print(f"Error: Model '{args.model}' not supported.")
        sys.exit(1)

    print(f"Using model: {args.model}")

    # Load dataset
    try:
        df = pd.read_csv(args.data_path, dtype=str).fillna("")
    except FileNotFoundError:
        print(f"Error: Data file not found at {args.data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data file: {e}")
        sys.exit(1)

    # Quick diagnostics
    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    if "image_path" in df.columns:
        print("image_path nulls:", df["image_path"].isnull().sum())

    # Prompt column selection
    preferred_prompt_cols = [
        "jailbreak_query_es_adapted",
        "redteam_query_es_adapted",
        "jailbreak_query_es_mt",
        "redteam_query_es_mt",
        "jailbreak_query",
        "redteam_query"
    ]
    prompt_col = None
    if args.prompt_column:
        prompt_col = args.prompt_column
        if prompt_col not in df.columns:
            print(f"Warning: requested prompt column '{prompt_col}' not found in CSV. Falling back to auto-detect.")
            prompt_col = None

    if prompt_col is None:
        for c in preferred_prompt_cols:
            if c in df.columns:
                prompt_col = c
                break

    if prompt_col is None:
        raise ValueError(f"No prompt column found. Expected one of {preferred_prompt_cols} or pass --prompt_column")

    image_col = args.image_column if args.image_column in df.columns else ("image_path" if "image_path" in df.columns else None)
    if image_col is None:
        print("Warning: no image_path column found; running text-only prompts.")

    results = []
    error_counts = 0
    print("Prompt → Response:")

    # For each row, attempt inference with retries
    for index, row in tqdm(df.iterrows(), total=len(df)):
        raw_image_path = row.get(image_col) if image_col else None
        raw_prompt = row.get(prompt_col)

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
            print(f"\nSkipping example {index}: missing both image_path and {prompt_col}")
            results.append("[Error: missing both image and prompt]")
            continue

        resp = None
        last_exception = None

        for attempt in range(1, args.retries + 1):
            try:
                # Call model inference
                raw_resp = model.inference(image_path, prompt)
                resp = safe_str(raw_resp).strip()

                # If we got an explicit error tag, decide to retry or accept
                # Acceptable responses do not start with "[Error"
                if resp and not resp.startswith("[Error"):
                    # success
                    break
                else:
                    # resp indicates error; we will retry unless it's the last attempt
                    last_exception = resp
                    if attempt < args.retries:
                        wait = args.retry_wait * attempt
                        print(f"[WARN] Attempt {attempt} failed for idx {index} with: {resp}. Retrying after {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"[ERROR] Final attempt {attempt} failed for idx {index}: {resp}")
            except Exception as e:
                last_exception = str(e)
                print(f"[EXCEPTION] Attempt {attempt} for idx {index} raised: {e}")
                if attempt < args.retries:
                    wait = args.retry_wait * attempt
                    time.sleep(wait)
                else:
                    print(f"[ERROR] Giving up on idx {index} after {attempt} attempts.")

        # If after retries resp is still an error, provide informative fallback
        if not resp:
            resp = f"[Error during inference: {last_exception or 'unknown'}]"
            error_counts += 1

        # Print a short sample for logging (limit length)
        try:
            print(f"\n[IDX {index}] Prompt({prompt_col}): {str(prompt)[:120]}...")
            print(f"[Response]: {resp[:300]}")
        except Exception:
            pass

        results.append(resp)

    # Append results and save
    df["response"] = results

    dataset_name = args.dataset if args.dataset else os.path.splitext(os.path.basename(args.data_path))[0]
    model_name_lower = args.model.lower()
    save_dir = os.path.join(args.save_path, dataset_name, model_name_lower)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}.csv")

    df.to_csv(save_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"\n✅ Saved results to: {save_path}")

    null_responses = df["response"].isnull().sum()
    print(f"Nulls: {null_responses}")
    print(f"Total inference error fallbacks: {error_counts}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attack scenarios against a specified model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset (optional).")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use (Bunny, InstructBLIP, google, microsoft, Llama3, Llama4Scout, liquid, liquid_spanish)."
    )
    parser.add_argument("--prompt_column", type=str, default=None, help="Which CSV column to use as prompt (optional)")
    parser.add_argument("--image_column", type=str, default="image_path", help="CSV image column name (default=image_path)")
    parser.add_argument("--retries", type=int, default=3, help="How many attempts per example (default=3)")
    parser.add_argument("--retry_wait", type=float, default=2.0, help="Base wait seconds between retries (multiplied by attempt index)")
    args = parser.parse_args()
    main(args)
