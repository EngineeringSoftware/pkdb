import numpy as np
import pykokkos as pk


@pk.workunit
def work(team_member, view):
    j: int = team_member.league_rank()  # Row index
    k: int = team_member.team_size()  # Number of columns per row

    def inner(i: int):
        # Initialize each element: view[row][col] = row * cols + col
        view[j][i] = k * j + i

    pk.parallel_for(pk.TeamThreadRange(team_member, k), inner)
    view[j][0] = 0  # set first element as zero


def main():
    pk.set_default_space(pk.ExecutionSpace.DebugOpenMP)

    # Create 50x2 matrix (50 rows, 2 columns)
    matrix = np.zeros((50, 2))

    # TeamPolicy(num_teams, team_size) maps to (rows, cols)
    pk.parallel_for("work", pk.TeamPolicy(10, 2), work, view=matrix)

    print(matrix)
    print(matrix[1, 0])
    print(matrix[1, 1])
    print(matrix[30, 1])
    print(f"Shape: {matrix.shape}")


main()
