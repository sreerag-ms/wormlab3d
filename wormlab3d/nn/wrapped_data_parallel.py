from torch.nn import DataParallel


class WrappedDataParallel(DataParallel):
    """
    Wrapper class for pytorch's DataParallel class. When a nn.Module is wrapped with DataParallel the attributes
    of the original module are no longer accessible. This class provides a workaround to that problem.
    """

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except Exception:
            try:
                return self.module.__getattr__(item)
            except Exception:
                if hasattr(self.module, item):
                    return getattr(self.module, item)
            raise
