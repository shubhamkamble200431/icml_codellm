def test_distinct(data):
    """
    Write a python function to determine whether all the numbers are different from each other are not.
    """
    return len(data) == len(set(data))