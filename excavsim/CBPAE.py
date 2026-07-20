

class CBPAEAllocator():
    """Das et al. (2015): bidding continues during task execution."""

    name = "cbpae"

    def allocate(self, model) -> None:
        raise NotImplementedError

