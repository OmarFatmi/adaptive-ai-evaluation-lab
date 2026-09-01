import math


def identity(size: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [dot(row, vector) for row in matrix]


def outer(left: list[float], right: list[float]) -> list[list[float]]:
    return [[a * b for b in right] for a in left]


def inverse(matrix: list[list[float]]) -> list[list[float]]:
    """Compatibility helper used by historical tests; LinUCB itself uses cached inverses."""
    size = len(matrix)
    augmented = [list(map(float, matrix[row])) + identity(size)[row] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Singular matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][index] - factor * augmented[column][index]
                for index in range(2 * size)
            ]
    return [row[size:] for row in augmented]


class LinUCB:
    """Disjoint full-matrix LinUCB with Sherman-Morrison inverse updates."""

    def __init__(
        self,
        arms: list[str],
        seed: int = 42,
        alpha: float = 0.8,
        dimensions: int = 13,
    ) -> None:
        del seed
        self.arms = arms
        self.alpha = alpha
        self.dimensions = dimensions
        self.a = {arm: identity(dimensions) for arm in arms}
        self.a_inv = {arm: identity(dimensions) for arm in arms}
        self.b = {arm: [0.0] * dimensions for arm in arms}
        self.counts = {arm: 0 for arm in arms}
        self.values = {arm: 0.0 for arm in arms}
        self.last_context = [0.0] * dimensions

    @property
    def A(self):
        return self.a

    def _vector(self, context: dict | None) -> list[float]:
        vector = list((context or {}).get("vector", []))
        return (vector + [0.0] * self.dimensions)[: self.dimensions]

    def score(self, arm: str, context: list[float]) -> float:
        inverse = self.a_inv[arm]
        theta = mat_vec(inverse, self.b[arm])
        projected = mat_vec(inverse, context)
        uncertainty = max(0.0, dot(context, projected))
        return dot(context, theta) + self.alpha * math.sqrt(uncertainty)

    def select(self, context: dict | None = None) -> str:
        vector = self._vector(context)
        self.last_context = vector
        untried = [arm for arm in self.arms if self.counts[arm] == 0]
        if untried:
            return untried[0]
        return max(self.arms, key=lambda arm: self.score(arm, vector))

    def update(self, arm: str, reward: float, context: dict | None = None) -> None:
        vector = self._vector(context) if context else self.last_context
        self.counts[arm] += 1
        count = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / count

        inverse = self.a_inv[arm]
        inverse_x = mat_vec(inverse, vector)
        denominator = 1.0 + dot(vector, inverse_x)
        correction = outer(inverse_x, inverse_x)

        for row in range(self.dimensions):
            self.b[arm][row] += reward * vector[row]
            for column in range(self.dimensions):
                self.a[arm][row][column] += vector[row] * vector[column]
                self.a_inv[arm][row][column] -= correction[row][column] / denominator

    def snapshot(self) -> dict:
        return {
            "counts": self.counts,
            "values": self.values,
            "A": self.a,
            "A_inv": self.a_inv,
            "b": self.b,
            "alpha": self.alpha,
            "implementation": "full_matrix",
            "inverse_update": "sherman_morrison",
        }
