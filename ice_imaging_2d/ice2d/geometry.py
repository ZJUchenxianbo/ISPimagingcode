"""Line quadrature and restricted scalar geometry expressions."""
import ast
import numpy as np

FUNCTIONS = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "sqrt": np.sqrt,
             "exp": np.exp, "log": np.log, "abs": np.abs, "min": np.minimum, "max": np.maximum}
OPERATIONS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
              ast.Div: np.divide, ast.Pow: np.power, ast.UAdd: np.positive, ast.USub: np.negative}

def expression(text, x):
    """Evaluate only numeric arithmetic and a fixed set of functions."""
    symbols = {"x": x, "pi": np.pi, "e": np.e}

    def eval_node(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in symbols:
            return symbols[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATIONS:
            return OPERATIONS[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATIONS:
            return OPERATIONS[type(node.op)](eval_node(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords:
            argc = 2 if node.func.id in ("min", "max") else 1
            if len(node.args) == argc:
                return FUNCTIONS[node.func.id](*(eval_node(arg) for arg in node.args))
        raise ValueError("Use x, pi, real numbers, arithmetic and the listed scalar functions in f1/f2")

    try:
        with np.errstate(all="raise"):
            values = np.broadcast_to(eval_node(ast.parse(text, mode="eval").body), np.shape(x)).astype(float)
    except (SyntaxError, TypeError, OverflowError, FloatingPointError) as error:
        raise ValueError(f"Invalid interface expression {text!r}") from error
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite interface values from {text!r}")
    return values

def line_weights(coordinates):
    """Trapezoidal weights on actual nodes; a single source has unit counting weight."""
    coordinates = np.asarray(coordinates, dtype=float)
    if len(coordinates) == 1:
        return np.ones(1)
    gaps = np.diff(coordinates)
    if np.any(gaps <= 0):
        raise ValueError("Acquisition coordinates must be strictly increasing")
    weights = np.empty_like(coordinates)
    weights[0], weights[-1] = gaps[0] / 2, gaps[-1] / 2
    weights[1:-1] = (gaps[:-1] + gaps[1:]) / 2
    return weights
