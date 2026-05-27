def index_multiplication(test_tup1, test_tup2):
    """
    Write a function to perform index wise multiplication of tuple elements in the given two tuples.
    """
    return tuple(x * y for x, y in zip(test_tup1, test_tup2))