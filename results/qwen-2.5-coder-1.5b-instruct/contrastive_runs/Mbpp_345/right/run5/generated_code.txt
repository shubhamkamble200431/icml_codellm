def diff_consecutivenums(nums):
    """
    Write a function to find the difference between two consecutive numbers in a given list.
    """
    return [nums[i+1] - nums[i] for i in range(len(nums)-1)]