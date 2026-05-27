def find_substring(str1, sub_str):
    """
    Write a function to check if a substring is present in a given list of string values.
    """
    return any(sub_str in s for s in str1)