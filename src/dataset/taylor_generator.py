from sympy import symbols, sympify, series, simplify

x = symbols("x")


def simplify_expression(expr_str):
    """
    Convert string expression into a simplified SymPy expression.
    """
    expr = sympify(expr_str)
    simplified_expr = simplify(expr)
    return simplified_expr


def compute_taylor(expr, order=4):
    """
    Compute Taylor expansion of expression around x=0 up to given order.
    """
    taylor = series(expr, x, 0, order + 1).removeO()
    simplified_taylor = simplify(taylor)
    return simplified_taylor


def generate_taylor_pair(expr_str):
    """
    Takes a string expression and returns:
    (simplified_expression, taylor_expansion)
    """

    simplified_expr = simplify_expression(expr_str)

    taylor_expansion = compute_taylor(simplified_expr)

    return str(simplified_expr), str(taylor_expansion)


if __name__ == "__main__":

    test_expressions = [
        "sin(x)",
        "cos(x)",
        "exp(x)",
        "sin(x) + 2*x",
        "x*sin(x)",
        "1*sin(x)",
        "x - x - 5"
    ]

    for expr in test_expressions:

        simplified, taylor = generate_taylor_pair(expr)

        print("Input:", expr)
        print("Simplified:", simplified)
        print("Taylor:", taylor)
        print("-" * 40)