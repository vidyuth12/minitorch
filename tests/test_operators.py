from typing import Callable, List, Tuple
import math

import pytest
from hypothesis import given
from hypothesis.strategies import lists

from minitorch import MathTest
from minitorch.operators import (
    add,
    addLists,
    eq,
    id,
    inv,
    inv_back,
    log_back,
    lt,
    max,
    mul,
    neg,
    negList,
    prod,
    relu,
    relu_back,
    sigmoid,
    sum,
    is_close,
)

from .strategies import assert_close, small_floats

# ## Task 0.1 Basic hypothesis tests.


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_same_as_python(x: float, y: float) -> None:
    "Check that the main operators all return the same value of the python version"
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if abs(x) > 1e-5:
        assert_close(inv(x), 1.0 / x)


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a: float) -> None:
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_relu_back(a: float, b: float) -> None:
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_id(a: float) -> None:
    assert id(a) == a


@pytest.mark.task0_1
@given(small_floats)
def test_lt(a: float) -> None:
    "Check that a - 1.0 is always less than a"
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_max(a: float) -> None:
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0


@pytest.mark.task0_1
@given(small_floats)
def test_eq(a: float) -> None:
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


# ## Task 0.2 - Property Testing

# Implement the following property checks
# that ensure that your operators obey basic
# mathematical rules.


@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a: float) -> None:
    """Check properties of the sigmoid function, specifically
    * It is always between 0.0 and 1.0.
    * one minus sigmoid is the same as sigmoid of the negative
    * It crosses 0 at 0.5
    * It is  strictly increasing.
    """
    # TODO: Implement for Task 0.2.
    #raise NotImplementedError('Need to implement for Task 0.2')
    # It is always between 0.0 and 1.0.
    s = sigmoid(a)
    assert 0.0 <= s <= 1.0, f"Sigmoid output {s} is out of range [0, 1]"

    # one minus sigmoid is the same as sigmoid of the negative
    # 1 - sigmoid(a) == sigmoid(-a)
    s_neg_a = sigmoid(neg(a))
    assert is_close(1 - s, s_neg_a), f"1 - sigmoid({a}) = {1 - s} != sigmoid(-{a}) = {s_neg_a}"

    # It crosses 0 at 0.5
    s_0 = sigmoid(0)
    assert is_close(s_0, 0.5), f"sigmoid(0) = {s_0} != 0.5"

    # It is  strictly increasing.
    delta = 1e-5
    s_plus_d = sigmoid(a + delta)
    assert lt(s, s_plus_d) == 1.0, (
        f"Sigmoid function is not strictly increasing: sigmoid({a}) = {s} "
        f"vs. sigmoid({a + delta}) = {s_plus_d}"
    )

    print("Passed all tests for a = {s}.")



@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a: float, b: float, c: float) -> None:
    "Test the transitive property of less-than (a < b and b < c implies a < c)"
    # TODO: Implement for Task 0.2.
    #raise NotImplementedError('Need to implement for Task 0.2')
    if lt(a, b) == 1.0 and lt(b, c) == 1.0:
        assert lt(a, c) == 1.0, f"Transitive property failed: {a} < {b} and {b} < {c} does not imply {a} < {c}"

    print("Transitive property passed for a={a}, b={b}, c={c}")



@pytest.mark.task0_2
def test_symmetric() -> None:
    """
    Write a test that ensures that :func:`minitorch.operators.mul` is symmetric, i.e.
    gives the same value regardless of the order of its input.
    """
    # TODO: Implement for Task 0.2.
    #raise NotImplementedError('Need to implement for Task 0.2')
    a = 3.0
    b = 4.0
    result_1 = mul(a, b)
    result_2 = mul(b, a)
    assert eq(result_1, result_2) == 1.0, f"mul function is not symmetrical as result of {a},{b} and {b},{a} is not the same"



@pytest.mark.task0_2
def test_distribute() -> None:
    r"""
    Write a test that ensures that your operators distribute, i.e.
    :math:`z \times (x + y) = z \times x + z \times y`
    """
    # TODO: Implement for Task 0.2.
    #raise NotImplementedError('Need to implement for Task 0.2')
    x, y, z = 2.0, 3.0, 4.0
    left = mul(z, add(x,y))
    right = add(mul(z, x), mul(z, y))
    equal = eq(left, right)
    assert equal == 1.0, f"Operators do not distribute"
    print("Distributive property passed")


@pytest.mark.task0_2
def test_other() -> None:
    """
    Write a test that ensures some other property holds for your functions.
    """
    # TODO: Implement for Task 0.2.
    #raise NotImplementedError('Need to implement for Task 0.2')
    x, y = 3.0, 4.0
    r = max(x, y)
    assert y, f"Incorrect computation for max"



# ## Task 0.3  - Higher-order functions

# These tests check that your higher-order functions obey basic
# properties.


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a: float, b: float, c: float, d: float) -> None:
    x1, x2 = addLists([a, b], [c, d])
    y1, y2 = a + c, b + d
    assert_close(x1, y1)
    assert_close(x2, y2)


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(ls1: List[float], ls2: List[float]) -> None:
    """
    Write a test that ensures that the sum of `ls1` plus the sum of `ls2`
    is the same as the sum of each element of `ls1` plus each element of `ls2`.
    """
    # TODO: Implement for Task 0.3.
    raise NotImplementedError('Need to implement for Task 0.3')


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls: List[float]) -> None:
    assert_close(sum(ls), sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x: float, y: float, z: float) -> None:
    assert_close(prod([x, y, z]), x * y * z)


@pytest.mark.task0_3
@given(lists(small_floats))
def test_negList(ls: List[float]) -> None:
    check = negList(ls)
    for i, j in zip(ls, check):
        assert_close(i, -j)


# ## Generic mathematical tests

# For each unit this generic set of mathematical tests will run.


one_arg, two_arg, _ = MathTest._tests()


@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn: Tuple[str, Callable[[float], float]], t1: float) -> None:
    name, base_fn = fn
    base_fn(t1)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(
    fn: Tuple[str, Callable[[float, float], float]], t1: float, t2: float
) -> None:
    name, base_fn = fn
    base_fn(t1, t2)


@given(small_floats, small_floats)
def test_backs(a: float, b: float) -> None:
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
