from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Set

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    #raise NotImplementedError('Need to implement for Task 1.1')
    
    # Values with a small positive shift
    vals_positive = list(vals)
    vals_positive[arg] += epsilon

    # Values with a small negative shift
    vals_negative = list(vals)
    vals_negative[arg] -= epsilon

    return (f(*vals_positive) - f(*vals_negative)) / (2*epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    #raise NotImplementedError('Need to implement for Task 1.4')
    visited: Set[int] = set()
    topological_order: List[Variable] = []

    def DFS(node: Variable):
        if node.unique_id in visited or node.is_constant():
            return 
        
        visited.add(node.unique_id)
        for parent in node.parents:
            DFS(parent)
        topological_order.append(node)

    DFS(variable)
    return reversed(topological_order)



def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    #raise NotImplementedError('Need to implement for Task 1.4')

    # Get topological order
    topological_order = list(topological_sort(variable))

    # Dictionary to store intermediate derivatives
    derivatives = {var.unique_id: 0 for var in topological_order}
    derivatives[variable.unique_id] = deriv

    # Traverse graph
    for var in topological_order:
        d_output = derivatives[var.unique_id]

        if var.is_constant():
            continue

        # Chain rule to propagate derivatives to parents
        for parent, parent_deriv in var.chain_rule(d_output):
            if parent.unique_id in derivatives:
                derivatives[parent.unique_id] += parent_deriv
            else:
                derivatives[parent.unique_id] = parent_deriv

        # If leaf node, accumulate derivatives
        if var.is_leaf():
            var.accumulate_derivative(derivatives[var.unique_id])
            



@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
