"""
data_model.py

Data model related functions.
"""


def attr_names(model,
               exclude_underscore=True,
               excludes=None):
    """Return all attribute names of a data model class."""

    excludes = [] if excludes is None else excludes
    excludes.append("STRICT")

    attrs = [attr for attr in dir(model) if
             not callable(getattr(model, attr))   # attr is not callable
             and not attr.startswith("__")        # dunder names are always excluded
             and (not attr.startswith("_") or not exclude_underscore)
             and attr not in excludes]

    return attrs
