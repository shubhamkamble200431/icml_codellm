def multiple_to_single(L):
    """
    Write a function to convert a list of multiple integers into a single integer.
    """
    return int(''.join(map(str, L)))