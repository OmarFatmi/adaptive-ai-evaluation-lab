from collections import Counter
class CoverageTracker:
    def __init__(self, regions=None):
        self.visits = Counter(x for x in (regions or []) if x)
    def novelty(self, region: str) -> float:
        return 1.0 / (1.0 + self.visits[region])
    def visit(self, region: str) -> None:
        self.visits[region] += 1
    def ratio(self, estimated_space_size: int) -> float:
        return min(1.0, len(self.visits) / max(1, estimated_space_size))
    def unexplored(self, candidates):
        return [x for x in candidates if self.visits[x] == 0]
    def snapshot(self):
        return dict(self.visits)
