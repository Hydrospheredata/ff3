from ff3.drift.errors import NotEvaluatedError, EvalOnRestoredError


def is_evaluated(func):
    def wrapped(*args, **kwargs):
        self = args[0]
        if not self._is_evaluated:
            raise NotEvaluatedError(f"{self} is not evaluated yet.")
        return func(*args, **kwargs)
    return wrapped


def not_restored(func):
    def wrapped(*args, **kwargs):
        self = args[0]
        if self._is_restored:
            raise EvalOnRestoredError(
                f"{self} cannot be evaluated, since it was restored from a "
                "serialization format."
            )
        return func(*args, **kwargs)
    return wrapped