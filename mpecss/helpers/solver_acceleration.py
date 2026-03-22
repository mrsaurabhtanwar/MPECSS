"""
Linear solver acceleration. Automatically selects the optimal linear solver based on the problem size.
"""

def select_linear_solver_oss(n_x: int) -> str:
    """
    Selects the optimal open-source linear solver for IPOPT based on problem size.
    The user strictly uses MUMPS (no HSL solvers).
    """
    return "mumps"
