import numpy

def _make_city(name,neighbours):
    return []


def _make_connections(n,density=0.35):
    return numpy.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
    
def _set_up_cities(names=['Toronto', 'New York City', 
                         'Los Angeles', 'Mexico City', 
                         'Chicago', 'Washington DC', 
                         'Arlington County', 'Langley', 
                         'Fort Meade', 'Vancouver', 
                         'Houston', 'Ottawa',
                         'Jacksonville', 'San Francisco', 
                         'Waltham', 'Bethesda']):
   return []

def _get_real_world():
    return []

def _draw_world(world):
    pass

def _print_world(world):
    pass


def _get_cityno(world,city_in):
    return 0

def _is_connected(world,city1,city2):
    return True

def _reset_world(world):
    pass

import inspect
import A2

file = A2

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

template_functions = [_make_city,
                      _make_connections,
                      _set_up_cities,
                      _get_real_world,
                      _draw_world,
                      _print_world,
                      _get_cityno,
                      _is_connected,
                      _reset_world]

name = 'Toronto'
neighbours = [0, 1]
city = [name, True, neighbours]
world = [[city,city]]

template_args = [(name,neighbours),(1,1),(),(),
                 (world),(world),(world,city),
                 (world,city,city),(world)]

functions = [file.make_city,
             file.make_connections,
             file.set_up_cities,
             file.get_real_world,
             file.draw_world,
             file.print_world,
             file.get_cityno,
             file.is_connected,
             file.reset_world]

# Checking that name and student number comments have changed
with open('A2.py') as f:
    f.readline()
    if f.readline() == "## Name: PLEASE FILL THIS IN\n":
        raise SyntaxError("You forgot to enter your name!")
    if f.readline() == "## Student number: SERIOUSLY\n":
        raise SyntaxError("You forgot to enter your student number!")

# Checking that arguments and return types of functions are correct
for i in range(len(functions)):
    check_args(functions[i],template_functions[i])
    check_return_type(functions[i],template_functions[i],template_args[i])
