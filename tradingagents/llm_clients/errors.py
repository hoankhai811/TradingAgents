"""Safe provider-limit classification shared by the service and backtester."""


def is_provider_limit_error(exc: BaseException) -> bool:
    if any(cls.__name__ in {"RateLimitError", "ResourceExhausted"} for cls in type(exc).__mro__):
        return True
    for source in (exc, getattr(exc, "response", None)):
        for field in ("status_code", "status", "code"):
            status = getattr(source, field, None)
            if isinstance(status, int) and not isinstance(status, bool) and status == 429:
                return True
    return False
