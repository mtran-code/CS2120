var1 = 2
var2 = 3
var3 = 4
# Input: Declare arbitrary integers

expression1 = var1 + var2
print("\nExpression 1 gives", expression1)
# Output: Expression 1 gives 5

expression2 = (var1 * var2) + var3
print("Expression 2 gives", expression2)
# Output: Expression 2 gives 10

expression3 = expression1 / expression2
print("Expression 3 gives", expression3)
# Output: Expression 3 gives 0.5

fahrenheit = int(76)
convertedF = float((fahrenheit - 32) * (5 / 9))
print(str(fahrenheit) + "°F is equivalent to approximately", str(round(convertedF, 1)) + "°C")
# Output: 76°F is equivalent to approximately 24.4°C

celsius = int(21)
convertedC = float((celsius * (9 / 5)) + 32)
print(str(celsius) + "°C is equivalent to approximately", str(round(convertedC, 1)) + "°F")
# Output: 21°C is equivalent to approximately 69.8°F
