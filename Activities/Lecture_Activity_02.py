# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:45:16 2019

@author: Michael
"""

def convert_temperature(temp, conversion='celsius_to_fahrenheit'):
    if conversion == 'celsius_to_fahrenheit':
        celsius = float(temp)
        fahrenheit = float((celsius * (9 / 5)) + 32)
        fahrenheit = round(fahrenheit, 1)
        return str(celsius) + '°C converts to ' + str(fahrenheit) + '°F.'
    elif conversion == 'fahrenheit_to_celsius':
        fahrenheit = float(temp)
        celsius = float((fahrenheit - 32) * (5 / 9))
        celsius = round(celsius, 1)
        return str(fahrenheit) + '°F converts to ' + str(celsius) + '°C.'
    else:
        return "Improper conversion type. Supported types include 'celsius_to_fahrenheit' and 'fahrenheit_to_celsius'"


print(convert_temperature(32))
# Output: 32.0°C converts to 89.6°F.

print(convert_temperature(21, 'celsius_to_fahrenheit'))
# Output: 21.0°C converts to 69.8°F.

print(convert_temperature(74, 'fahrenheit_to_celsius'))
# Output: 74.0°F converts to 23.3°C.

print(convert_temperature(25, 'kelvin_to_celsius'))
# Output: Improper conversion type. Supported types include 'celsius_to_fahrenheit' and 'fahrenheit_to_celsius'
