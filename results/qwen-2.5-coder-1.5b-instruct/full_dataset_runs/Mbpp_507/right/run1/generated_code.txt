def remove_words(list1, removewords):
    """
    Write a function to remove specific words from a given list.
    """
    return [word for word in list1 if word not in removewords]