import re

def tokenize_expression(expr):
    expr = expr.replace(" ", "")
    pattern = r"sin|cos|exp|\*\*|\d+|x|[+\-*/(),]"
    tokens = re.findall(pattern, expr)
    return tokens

def tokenize_dataset(expressions):
    tokenized = []
    for expr in expressions:
        tokens = tokenize_expression(expr)
        tokenized.append(tokens)
    return tokenized

if __name__ == "__main__":
    test_expressions = [
        "sin(x) , 0",
        "exp(x) + x + 3 , 1",
        "x**2 - x**4/6 , -1"
    ]

    for expr in test_expressions:
        tokens = tokenize_expression(expr)
        print("Expression:", expr)
        print("Tokens:", tokens)
        print("-" * 40)