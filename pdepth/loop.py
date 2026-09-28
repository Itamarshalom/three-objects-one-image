"""Loop read-out used in the paper (corner cubes, where junctions are drawn, are not used).

Fit a line to each bar's four interior cubes (cubes 5j+1..5j+4 of bar j). The jump J_j at joint j is bar j's line minus
bar j-1's line, both evaluated at corner cube 5j. Going around the loop the jumps and the bars' changes sum to zero.
A valid object comes nearer at the same rate along every bar and jumps back by the whole mismatch T at its gap.
The excess tear e_j = (J_j - r_j) / (g_j - r_j), with r_j, g_j from a network's own possible controls, is computed in
scripts/analyze_excess.py.
"""
import numpy as np

N_BAR = 5                                   # cube steps per bar (n - 1)


def bars_and_joints(profile):
    """slopes b_j (per cube step) and joint jumps J_j (at the start of bar j) of a 15-cube loop profile"""
    t = np.arange(1, N_BAR)
    a, b = np.zeros(3), np.zeros(3)
    for j in range(3):
        y = profile[j * N_BAR + 1:(j + 1) * N_BAR]
        b[j], a[j] = np.polyfit(t, y, 1)
    J = np.array([a[j] - (a[j - 1] + N_BAR * b[j - 1]) for j in range(3)])     # line of bar j at 0 minus bar j-1 at 5
    return b, J
