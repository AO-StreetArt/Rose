import ast
import operator
from typing import Any


class SafeMathEvaluator(ast.NodeVisitor):
    """AST visitor that evaluates basic math expressions safely."""

    _binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def visit(self, node: ast.AST) -> Any:  # type: ignore[override]
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric literals are allowed.")
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._binary_ops:
                raise ValueError(f"Operator {op_type.__name__} is not allowed.")
            left = self.visit(node.left)
            right = self.visit(node.right)
            return self._binary_ops[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self._unary_ops:
                raise ValueError(f"Unary operator {op_type.__name__} is not allowed.")
            operand = self.visit(node.operand)
            return self._unary_ops[op_type](operand)
        raise ValueError(f"Unsupported expression component: {type(node).__name__}")


def evaluate_math_expression(expression: str) -> float:
    """Parse and evaluate a math expression using a restricted AST."""
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid math expression.") from exc
    evaluator = SafeMathEvaluator()
    result = evaluator.visit(parsed)
    return float(result) if isinstance(result, int) else result


def lambda_handler(event: dict, context: Any) -> dict:
    """AWS Lambda handler for Bedrock agent actions."""
    expression = event.get("MathExpression")
    if expression is None:
        raise ValueError("Event missing required 'MathExpression' parameter.")
    if not isinstance(expression, str):
        raise ValueError("'MathExpression' must be a string.")

    result = evaluate_math_expression(expression)
    return {"MathResult": result}
