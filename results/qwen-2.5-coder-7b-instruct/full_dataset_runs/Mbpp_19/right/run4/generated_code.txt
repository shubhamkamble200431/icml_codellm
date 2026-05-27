def test_duplicate(arraynums):
    """
    Write a function to find whether a given array of integers contains any duplicate element.
    """
    return len(arraynums) != len(set(arraynums))