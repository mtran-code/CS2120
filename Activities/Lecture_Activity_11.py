def fact(n):
    """
    computes the factorial of n using recursion, where n is a non-negative integer (positive or zero)
    """
    if n < 1:
        return 1
    else:
        return n * fact(n - 1)


print(fact(5))
# Output: 120
