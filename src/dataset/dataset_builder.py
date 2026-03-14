import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from src.dataset.expression_generator import generate_expression
from src.dataset.taylor_generator import generate_taylor_pair

def generate_single_sample(_):
    while True:
        expr = generate_expression(max_operations=4) 
        try:
            model_input, taylor = generate_taylor_pair(expr)
            if taylor != "0" and len(taylor) > 1:
                model_input_clean = model_input.replace("**", "^").replace(" ", "")
                taylor_clean = taylor.replace("**", "^").replace(" ", "")
                return {"function": model_input_clean, "taylor": taylor_clean}
        except Exception:
            pass

def build_dataset(target_samples=100000):
    num_cores = max(1, os.cpu_count() - 2) 
    print(f"Starting generation: {target_samples} samples on {num_cores} cores")

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(
            executor.map(generate_single_sample, range(target_samples), chunksize=200), 
            total=target_samples,
            desc="Generating dataset"
        ))
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = build_dataset(100000)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/dataset.csv", index=False)
    print("Dataset generation complete")