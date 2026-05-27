def binary_to_integer(test_tup):
    """
    Write a function to convert the given binary tuple to integer.
    """
    return int(''.join(map(str, test_tup)), 2)