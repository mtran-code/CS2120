def _load_a1_data(filename='London_mean_etr_max_etr_min.csv'):
    return []

def _coldest_month(records):
    return ""

def _warmest_month(records):
    return ""

def _print_mean_annual_temperature(year, records):
    pass

import inspect
import A1

file = A1

def check_args(func,template_func):
    if inspect.getfullargspec(func) != inspect.getfullargspec(template_func):
        error_string = ''.join(["function signature for '",
                                func.__name__,
                                "' in ",
                                str(file),
                                " has been changed from its original form!"])
        raise SyntaxError(error_string)
    else:
        print(func.__name__,"has correct signature")

def check_return_type(func,template_func,args):
    if type(func(*args)) == type(template_func(*args)):
        print(func.__name__,"return type okay")
    else:
        error_string = ''.join(["return type for '",
                                func.__name__,
                                "' in ",
                                str(file),
                                " has been changed from its original form!"])
        raise TypeError(error_string)

template_functions = [_load_a1_data,
                      _coldest_month,
                      _warmest_month,
                      _print_mean_annual_temperature]

records = [[['1000','10','0','0','0']]]

template_args = [(),(records),(records),(1000,records[0])]

functions = [file.load_a1_data,
             file.coldest_month,
             file.warmest_month,
             file.print_mean_annual_temperature]

# Checking that name and student number comments have changed
with open('A1.py') as f:
    f.readline()
    if f.readline() == "## Name: PLEASE FILL THIS IN\n":
        raise SyntaxError("You forgot to enter your name!")
    if f.readline() == "## Student number: SERIOUSLY\n":
        raise SyntaxError("You forgot to enter your student number!")

# Checking that arguments and return types of functions are correct
for i in range(len(functions)):
    check_args(functions[i],template_functions[i])
    if i < len(functions)-1:
        check_return_type(functions[i],template_functions[i],template_args[i])
