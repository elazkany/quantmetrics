# exceptions.py

class quantmetricsException(Exception):
    """Base class for all exceptions in quantmetrics"""
    pass

class InvalidParametersError(quantmetricsException):
    pass

class FeatureNotImplementedError(quantmetricsException):
    """Raised when a requested feature has not been implemented yet."""
    def __init__(self, feature):
        message = f"The '{feature}' case has not been implemented yet."
        super().__init__(message)


class UnsupportedEMMTypeError(quantmetricsException):
    """Raised when an unsupported or unknown EMM type is used."""
    def __init__(self, emm):
        message = (
            f"Unknown or unsupported EMM type: {emm}. "
            "EMM must be either 'mean-correcting' or 'Esscher'."
        )
        super().__init__(message)

class UnknownPayoffTypeError(quantmetricsException):
    """Raised when an unsupported or unknown payoff type is used."""
    def __init__(self, payoff):
        message = (
            f"Unknown payoff type: {payoff}. "
            "Payoff must be 'c' for call or 'p' for put."
        )
        super().__init__(message)
