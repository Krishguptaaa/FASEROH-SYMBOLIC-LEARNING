import random
from sympy import symbols, sympify, series, simplify, expand # <-- Added expand here

x = symbols("x")

# 50% chance of standard Maclaurin (0), 50% chance of other points
EXPANSION_POINTS = [-2, -1, 0, 1, 2] 

def simplify_expression(expr_str):
    expr = sympify(expr_str)
    return simplify(expr)

def compute_taylor(expr, point=0, order=4):
    """
    Compute Taylor expansion of expression around x=point up to given order.
    """
    # 1. Calculate the series
    taylor_series = series(expr, x, point, order + 1).removeO()
    
    # 2. THE FLATTENING FIX: Expand the polynomial
    return expand(taylor_series)

def generate_taylor_pair(expr_str):
    """
    Returns the conditional input and the target:
    Input: "simplified_expression,expansion_point"
    Target: "taylor_expansion"
    """
    point = random.choice(EXPANSION_POINTS)
    
    simplified_expr = simplify_expression(expr_str)
    taylor_expansion = compute_taylor(simplified_expr, point=point)
    
    # Format input (removed extra spaces around the comma)
    model_input = f"{simplified_expr},{point}"
    
    # Convert to strings for text manipulation
    input_str = str(model_input)
    target_str = str(taylor_expansion)
    
    # 3. THE TOKENIZER FIX: Replace ** with ^
    input_str = input_str.replace("**", "^")
    target_str = target_str.replace("**", "^")
    
    # 4. COMPRESSION FIX: Remove all whitespace
    input_str = input_str.replace(" ", "")
    target_str = target_str.replace(" ", "")
    
    return input_str, target_str

if __name__ == "__main__":
    test_expressions = ["sin(x)", "cos(2*x)", "exp(x+1)"]
    for expr in test_expressions:
        model_input, taylor = generate_taylor_pair(expr)
        print("Model Input:", model_input)
        print("Taylor Target:", taylor)
        print("-" * 40)