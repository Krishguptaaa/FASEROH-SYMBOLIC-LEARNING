import random

VARIABLE = "x"

FUNCTIONS = [
    "sin",
    "cos",
    "exp"
]

OPERATORS = [
    "+",
    "-",
    "*"
]

CONSTANTS = [
    "1",
    "2",
    "3",
    "4",
    "5"
]


def random_variable():
    return VARIABLE


def random_constant():
    return random.choice(CONSTANTS)


def random_function():
    func = random.choice(FUNCTIONS)
    return f"{func}({VARIABLE})"


def random_term():
    """
    Generates a basic term:
    x
    constant
    sin(x)
    cos(x)
    exp(x)
    """
    choices = [
        random_variable(),
        random_constant(),
        random_function()
    ]
    return random.choice(choices)


def generate_expression(max_operations=3):
    """
    Generate a random expression with 1–3 operations.
    Example:
    sin(x) + 2*x
    exp(x) - 3
    """

    num_operations = random.randint(1, max_operations)

    expression = random_term()

    for _ in range(num_operations):
        operator = random.choice(OPERATORS)
        term = random_term()

        expression = f"{expression} {operator} {term}"

    return expression


def generate_expressions(n_samples):
    """
    Generate multiple expressions
    """
    expressions = []

    for _ in range(n_samples):
        expr = generate_expression()
        expressions.append(expr)

    return expressions


if __name__ == "__main__":
    samples = generate_expressions(10)

    for s in samples:
        print(s)