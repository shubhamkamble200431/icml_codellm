def tuple_to_int(nums):
    """
    Write a function to convert a given tuple of positive integers into an integer.
    """
    return int(''.join(map(str, nums)))