import pandas as pd
import torch
from pathlib import Path
from time import sleep
import sys
import argparse
import signal # Import the signal module

# Add the directory containing 'evaluate_models' to the system path
sys.path.append("/content/drive/MyDrive/JailBreakV_28K")

from evaluate_models.LlamaGuard import LlamaGuard   # adjust import path if needed

# Define a timeout exception
class TimeoutException(Exception):
    pass

# Define the timeout handler
def timeout_handler(signum, frame):
    raise TimeoutException

# Set the timeout signal handler
signal.signal(signal.SIGALRM, timeout_handler)


def get_evaluation_llamaguard_ours(dataset, timeout_seconds=60): # Added timeout_seconds parameter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlamaGuard(device=device)
    print(f"LlamaGuard loaded on {device}")

    evaluation_results = []
    jailbroken_results = []

    for i, example in enumerate(dataset):
        query = example.get("jailbreak_query", "")
        response = example.get("response", "") # Assuming 'response' column exists in the dataset

        evaluation = "Error: Timeout"
        jailbroken = False # Assume False if timeout occurs

        # Start the timeout timer
        signal.alarm(timeout_seconds)
        try:
            # model.inference expects (image_path, prompt) in your repo — we pass query as image_path for compatibility
            # You might need to adjust the arguments passed to inference based on your LlamaGuard implementation
            evaluation = model.inference(query, response) # Pass query and response
            jailbroken = "unsafe" in evaluation.lower() # Update jailbroken based on inference result

        except TimeoutException:
            print(f"Timeout for example {i}. Skipping.")
            evaluation = "Timeout"
            jailbroken = False
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            evaluation = f"Error: {e}"
            jailbroken = False
        finally:
            # Cancel the alarm
            signal.alarm(0)

        evaluation_results.append(evaluation)
        jailbroken_results.append(jailbroken)
        print(f"Processed example {i+1}/{len(dataset)}") # Add progress indicator


    return evaluation_results, jailbroken_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model's responses using LlamaGuard with a timeout.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV file containing model responses.")
    parser.add_argument("--save_path", type=str, help="Directory to save evaluation results. Defaults to the same directory as data_path/../evaluations/LlamaGuard/model_name")
    parser.add_argument("--model_name", type=str, help="Name of the model being evaluated. Used for creating save path. Defaults to the parent directory name of the data_path.")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds for each model inference. Defaults to 60 seconds.") # Added timeout argument


    args = parser.parse_args()

    data_path = Path(args.data_path)
    model_name = args.model_name if args.model_name else data_path.parent.name

    if args.save_path:
        save_dir = Path(args.save_path)
    else:
        # Default save path structure: data_path_parent / ../evaluations/LlamaGuard/model_name
        save_dir = data_path.parent.parent / "evaluations" / "LlamaGuard" / model_name

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / data_path.name.replace(".csv", "_llamaguard_eval.csv")


    print(f"Processing {data_path} ...")
    try:
        # Load the dataset (assuming it's a CSV with at least 'jailbreak_query' and 'response' columns)
        df = pd.read_csv(data_path)
        dataset = df.to_dict('records') # Convert dataframe to list of dictionaries for easier processing
        print("full data generated") # Keep this line for compatibility if needed

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data file: {e}")
        sys.exit(1)

    # Get LlamaGuard evaluations
    # Pass the timeout value from the command line argument
    evaluation_llamaguard_ours, jailbroken_llamaguard_ours = get_evaluation_llamaguard_ours(dataset, timeout_seconds=args.timeout)


    # Add results to the dataframe
    df["evaluation_llamaguard_ours"] = evaluation_llamaguard_ours
    df["jailbroken_llamaguard_ours"] = jailbroken_llamaguard_ours


    # Save the results
    df.to_csv(save_path, index=False)
    print(f"Generating {save_path} ...") # Corrected print statement
    print(f"Saved evaluation results to {save_path}")


if __name__ == "__main__":
    main()
