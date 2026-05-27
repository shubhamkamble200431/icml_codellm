def tuple_int_str(tuple_str):
    """
    Write a function to convert a tuple of string values to a tuple of integer values.
    """
    return tuple(int(item) for item in tuple_str)