"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

# TODO: Implement for Task 0.1

def add(x: float, y: float) -> float:
    """
    Add two floating point numbers.

    Args:
        x: First number
        y: Second number

    Returns:
        Sum of x and y
    """
    return x + y


def mul(x: float, y: float) -> float:
    """
    Multiply two floating point numbers.

    Args:
        x: First number
        y: Second number

    Returns:
        Product of x and y
    """
    return x * y


def id(x: float) -> float:
    """
    Identity function.

    Args:
        x: Input value

    Returns:
        Same value as input
    """
    return x


def neg(x: float) -> float:
    """
    Negate a floating point number.

    Args:
        x: Number to negate

    Returns:
        Negative of input x
    """
    return -x


def lt(x: float, y: float) -> float:
    """
    Less than comparison of two numbers.

    Args:
        x: First number
        y: Second number

    Returns:
        1.0 if x < y, 0.0 otherwise
    """
    return float(x < y)


def eq(x: float, y: float) -> float:
    """
    Equality comparison of two numbers.

    Args:
        x: First number
        y: Second number

    Returns:
        1.0 if x == y, 0.0 otherwise
    """
    return float(x == y)


def max(x: float, y: float) -> float:
    """
    Maximum of two numbers.

    Args:
        x: First number
        y: Second number

    Returns:
        Larger value between x and y
    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """
    Check if two numbers are close in value.

    Args:
        x: First number
        y: Second number

    Returns:
        1.0 if absolute difference between x and y is less than 1e-2, 0.0 otherwise
    """
    return float(abs(x - y) < 1e-2)


def sigmoid(x: float) -> float:
    """
    Sigmoid activation function.

    Computes f(x) = 1/(1 + e^(-x)) if x >= 0 else e^x/(1 + e^x)

    Args:
        x: Input value

    Returns:
        Result of sigmoid function
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def relu(x: float) -> float:
    """
    Rectified Linear Unit (ReLU) function.

    Args:
        x: Input value

    Returns:
        x if x > 0, else 0
    """
    return max(0.0, x)


def log(x: float) -> float:
    """
    Natural logarithm.

    Args:
        x: Input value

    Returns:
        Natural log of x
    """
    return math.log(x)


def exp(x: float) -> float:
    """
    Exponential function.

    Args:
        x: Input value

    Returns:
        e raised to the power of x
    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """
    Derivative of log function.

    Args:
        x: Input value
        d: Derivative from chain rule

    Returns:
        d/x if x != 0, else 0
    """
    return d / x if x != 0 else 0.0


def inv(x: float) -> float:
    """
    Multiplicative inverse (reciprocal).

    Args:
        x: Input value

    Returns:
        1/x if x != 0, else 0
    """
    return 1.0 / x if x != 0 else 0.0


def inv_back(x: float, d: float) -> float:
    """
    Derivative of inverse function.

    Args:
        x: Input value
        d: Derivative from chain rule

    Returns:
        -d/x^2 if x != 0, else 0
    """
    return -d / (x * x) if x != 0 else 0.0


def relu_back(x: float, d: float) -> float:
    """
    Derivative of ReLU function.

    Args:
        x: Input value
        d: Derivative from chain rule

    Returns:
        d if x > 0, else 0
    """
    return d if x > 0 else 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    def apply(ls: Iterable[float]):
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret
    return apply

def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(lambda x : -x)(ls)


def zipWith(fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    ret = []
    for i in range(min(len(ls1), len(ls2))):
        ret.append(fn(ls1[i], ls2[i]))
    return ret    

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zipWith(lambda x,y: x+y, ls1, ls2)


def reduce(fn: Callable[[float, float], float], ls: Iterable[float]) -> float:
    it = iter(ls)  # Create iterator from the iterable
    try:
        acc = next(it)  # Get first element as initial accumulator
    except StopIteration:
        raise ValueError("reduce() of empty sequence")
    
    for x in it:  # Iterate over remaining elements
        acc = fn(acc, x)  # Update accumulator
    return acc 

def sum(ls: Iterable[float]) -> float:
    return reduce(lambda x, y: x+y, ls) if list(ls) else 0.0

def prod(ls: Iterable[float]) -> float:
    return reduce(lambda x, y: x*y, ls)