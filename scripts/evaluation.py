"""Evaluation logs and metrics. Truth stays outside the feedback pipeline."""


class Logger:
    """Own simple in-memory experiment logs and parameter metadata."""

    def __init__(self):
        raise NotImplementedError("Initialise the experiment logs.")

    def record(self, timestamp, estimate, reference, command, truth):
        """Store a time-aligned sample, copying mutable values when necessary.

        Later include requested/limited commands and measured wheel motion.
        Truth is for comparison and must never feed back into control.
        """
        raise NotImplementedError("Store an evaluation sample.")

    def results(self):
        """Calculate tracking/estimation errors, stopping error and saturation."""
        raise NotImplementedError("Calculate experiment metrics.")
