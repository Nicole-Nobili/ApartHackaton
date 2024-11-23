def deprecated(func):
    def wrapper(*args, **kwargs):
        raise RuntimeError(f"{func.__name__} is deprecated and should not be used")
    return wrapper