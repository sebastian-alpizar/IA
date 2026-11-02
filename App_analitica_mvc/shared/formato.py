"""
formato.py (shared)
Formato numérico del curso: coma decimal.
"""


def num(x, dec=4, pct=False):
    """Formato numérico con coma decimal (estilo del curso)."""
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    if pct:
        return f"{100 * x:.2f} %".replace(".", ",")
    return f"{x:.{dec}f}".replace(".", ",")
