def add_pairwise(test_tup):
    """
    Write a function to find the pairwise addition of the elements of the given tuples.
    """
    return tuple(x + y for x, y in zip(test_tup, test_tup[1:]))