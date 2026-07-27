

class MOACBBAAgent():
    def __init__(self) -> None:
        pass

class MOACBBAAllocator():
    """Multi-objective bids (w1, w2) + re-allocation
    triggered by DetectCellChanges (Algorithm 1, lines 3-7).

    Hook: `model.changed_cells` is populated each tick by terrain edits;
    a non-empty delta should trigger a re-bid round.
    """

    name = "moa-cbba"

    def allocate(self, model) -> None:
        raise NotImplementedError
