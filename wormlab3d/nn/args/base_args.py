from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace, _ArgumentGroup


class BaseArgs(ABC):
    @classmethod
    @abstractmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        pass

    @classmethod
    def from_args(cls, args: Namespace) -> 'BaseArgs':
        """
        Create a BaseArgs instance from command-line arguments.
        """
        return cls(**vars(args))
