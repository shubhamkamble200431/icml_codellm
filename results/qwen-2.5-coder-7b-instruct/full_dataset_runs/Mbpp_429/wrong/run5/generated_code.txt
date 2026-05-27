def and_tuples(test_tup1, test_tup2):
    """
    Write a function to extract the elementwise and tuples from the given two tuples.
    """
    return tuple(a and b for a, b in zip(test_tup1, test_tup2))