import random
from sympy import symbols, sympify, series, simplify, expand

x = symbols("x")
EXPANSION_POINTS = [-2, -1, 0, 1, 2] 

def simplify_expression(expr_str):
    expr = sympify(expr_str)
    return simplify(expr)

def compute_taylor(expr, point=0, order=4):
    taylor_series = series(expr, x, point, order + 1).removeO()
    return expand(taylor_series)

def generate_taylor_pair(expr_str):
    point = random.choice(EXPANSION_POINTS)
    
    simplified_expr = simplify_expression(expr_str)
    taylor_expansion = compute_taylor(simplified_expr, point=point)
    
    model_input = f"{simplified_expr},{point}"
    
    input_str = str(model_input)
    target_str = str(taylor_expansion)
    
    input_str = input_str.replace("**", "^")
    target_str = target_str.replace("**", "^")
    
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