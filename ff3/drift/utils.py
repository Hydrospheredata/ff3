from ff3.drift.errors import NotEvaluatedError


def is_evaluated(func):
    def wrapped(*args, **kwargs):
        self = args[0]
        if not self._is_evaluated:
            raise NotEvaluatedError(f"{self} is not evaluated yet.")
        return func(*args, **kwargs)
    return wrapped