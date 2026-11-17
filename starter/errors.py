"""Input-validation error shared by every public entry point."""


class InputError(ValueError):
    """Raised for malformed shapes, labels, probabilities, weights or masks.

    Subclasses ``ValueError`` so callers that only catch ``ValueError`` still
    see a clear, typed failure instead of an obscure NumPy traceback.
    """
