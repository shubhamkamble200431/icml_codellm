def test_duplicate(arraynums):
    """
    Write a function to find whether a given array of integers contains any duplicate element.
    """
    seen = set()
    for num in arraynums:
        if num in seen:
            return True
        seen.add(num)
    return False