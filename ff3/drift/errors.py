
class DriftError(Exception):
    pass


class NotEvaluatedError(DriftError):
    pass


class NotFoundError(DriftError):
    pass


class ValidationError(DriftError):
    pass


class EvalOnRestoredError(DriftError):
    pass