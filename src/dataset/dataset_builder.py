import pandas as pd

from src.dataset.expression_generator import generate_expressions
from src.dataset.taylor_generator import generate_taylor_pair


def build_dataset(n_samples=1000):
    """
    Generate dataset of expressions and their Taylor expansions.
    """

    expressions = generate_expressions(n_samples)

    data = []

    for i,expr in enumerate(expressions):
        if i % 1000 == 0:
            print(f"Processed {i} samples")
        try:
            simplified_expr, taylor = generate_taylor_pair(expr)

            data.append({
                "function": simplified_expr,
                "taylor": taylor
            })

        except Exception as e:
            # Skip expressions that fail during parsing or expansion
            print(f"Skipping expression {expr}: {e}")

    df = pd.DataFrame(data)

    return df


def save_dataset(df, path="data/raw/dataset.csv"):
    """
    Save dataset to CSV.
    """

    df.to_csv(path, index=False)

    print(f"Dataset saved to {path}")


if __name__ == "__main__":

    dataset_size = 10000

    print("Generating dataset...")

    df = build_dataset(dataset_size)

    print(df.head())

    save_dataset(df)