SYMBOLS = {"sum", "multiply", "subtract", "divide"}

class UnsupportedOperation(ValueError):
    pass

def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def calculate(symbol: str, a: float, b: float) -> float:
    """
    异常约定（由调用方 app.py 映射为 HTTP 状态码）：
      TypeError         -> 422  operation 非字符串 / a、b 非数字
      ValueError        -> 400  未知操作
      ZeroDivisionError -> 400  除数为 0
    """
    if not isinstance(symbol, str):
        raise TypeError(f"operation must be str, got {type(symbol).__name__}")

    s = symbol.strip().lower()   # 去除多余空格并转小写

    if s not in SYMBOLS:
        raise UnsupportedOperation(f"unsupported operation: {symbol}")

    if not _is_number(a) or not _is_number(b):
        raise TypeError(f"{a!r} or {b!r} is not a number")

    if s == "sum":
        return a + b
    elif s == "subtract":
        return a - b
    elif s == "multiply":
        return a * b
    elif s == "divide":
        if b == 0:
            raise ZeroDivisionError("divisor is zero")
        return a / b
    else:
        raise ValueError(f"unsupported operation: {symbol}")