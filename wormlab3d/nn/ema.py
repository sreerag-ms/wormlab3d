class EMA:
    """
    Exponential moving average tracker.
    """

    def __init__(self):
        self.vars = {}

    def register(self, name: str, val: float = None, decay: float = 0.99):
        """Register a variable to be tracked and what decay to use for it."""
        self.vars[name] = {
            'val': val,
            'decay': decay
        }

    def __call__(self, name: str, x):
        """Update the moving average for the variable."""
        assert name in self.vars
        prev_average = self.vars[name]['val']
        decay = self.vars[name]['decay']
        if prev_average is None:
            new_average = x
        else:
            new_average = (1. - decay) * x + decay * prev_average
        self.vars[name]['val'] = new_average
        return new_average

    def __getitem__(self, name: str):
        assert name in self.vars
        return self.vars[name]['val']
