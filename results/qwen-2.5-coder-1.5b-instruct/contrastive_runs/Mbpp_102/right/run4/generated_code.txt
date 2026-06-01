def snake_to_camel(word):
    """
    Write a function to convert snake case string to camel case string.
    """
    return ''.join(x.capitalize() or '_' for x in word.split('_'))