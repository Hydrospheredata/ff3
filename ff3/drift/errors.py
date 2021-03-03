
class DriftError(Exception):
    pass


class NotEvaluatedError(DriftError):
    pass


class ValidationError(DriftError):
    pass
