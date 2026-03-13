import pandas as pd
import re
import os
import pandas as pd

def clean_math_string(text):

    text = str(text)    

    text = text.replace(" ", "")

    text = text.replace("**", "^")
    
    text = text.replace("E", "EXP_1")

    text = re.sub(r'sin\((-?\d+)\)', r'SIN_\1', text)
    text = re.sub(r'cos\((-?\d+)\)', r'COS_\1', text)
    text = re.sub(r'exp\((-?\d+)\)', r'EXP_\1', text)
    
    return text

def process_dataset(input_path, output_path):
    print(f"Loading raw dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Applying token consolidation...")
    df['function'] = df['function'].apply(clean_math_string)
    df['taylor'] = df['taylor'].apply(clean_math_string)
    
    max_len_func = df['function'].map(len).max()
    max_len_taylor = df['taylor'].map(len).max()
    
    print(f"\n--- DATASET STATS ---")
    print(f"Max Input Length (Chars): {max_len_func}")
    print(f"Max Target Length (Chars): {max_len_taylor}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Cleaned dataset saved to {output_path}")

if __name__ == "__main__":

    INPUT_FILE = "data/raw/dataset.csv" 
    OUTPUT_FILE = "data/processed/dataset_cleaned_100k.csv" 
 
    process_dataset(INPUT_FILE, OUTPUT_FILE)

    df = pd.read_csv(OUTPUT_FILE)
    print(f"Total processed samples: {len(df)}")

    df_filtered = df[df['taylor'].map(len) <= 250]

    print(f"Filtered (Safe) size: {len(df_filtered)}")

    df_filtered.to_csv("data/processed/dataset_ready_100k.csv", index=False)
    print("✅ Final Dataset Ready for Training!")
