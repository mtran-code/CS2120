import numpy

def _load_station_info(directory='./data/'):
    pass


def _load_temperature_data(directory='./data/'):
    pass


def _make_valid_temperature_data_dict(station_info_dict,
                                temperature_dict,
                                start_year,
                                end_year):
    pass

def _draw_map(plot_title, data_dict):
    pass

def _sort_dictionary_by_absolute_value_ascending(dictionary):
    pass

def _sort_dictionary_by_absolute_value_descending(dictionary):
    pass

def _compute_average_temp(temperatures):
    pass

def _compute_average_change(temperatures):
    pass

def _compute_average_changes_dict(station_info_dict,
                                 temperature_data_dict,
                                 start_year,
                                 end_year):
    pass

def _compute_top_average_change_dict(average_changes_dict,n):
    pass

def _make_average_change_dict_for_map(station_info_dict, average_change_dict):
    pass

def _draw_top_average_changes_map(top_average_changes_dict_for_map,
                                 start_year,
                                 end_year,
                                 num_top_values):
    pass

def _draw_maps_by_range(station_info_dict, 
                       valid_temperature_data_dict, 
                       start_year, 
                       years_per_map, 
                       num_top_values,
                       num_maps):
    pass

import inspect
import A3

file = A3

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

template_functions = [_load_station_info,
                      _load_temperature_data,
                      _make_valid_temperature_data_dict,
                      _draw_map,
                      _sort_dictionary_by_absolute_value_ascending,
                      _sort_dictionary_by_absolute_value_descending,
                      _compute_average_temp,
                      _compute_average_change,
                      _compute_average_changes_dict,
                      _compute_top_average_change_dict,
                      _make_average_change_dict_for_map,
                      _draw_top_average_changes_map,
                      _draw_maps_by_range]

functions = [file.load_station_info,
             file.load_temperature_data,
             file.make_valid_temperature_data_dict,
             file.draw_map,
             file.sort_dictionary_by_absolute_value_ascending,
             file.sort_dictionary_by_absolute_value_descending,
             file.compute_average_temp,
             file.compute_average_change,
             file.compute_average_changes_dict,
             file.compute_top_average_change_dict,
             file.make_average_change_dict_for_map,
             file.draw_top_average_changes_map,
             file.draw_maps_by_range]

# Checking that name and student number comments have changed
with open('A3.py') as f:
    if f.readline() == "## Name:\n":
        raise SyntaxError("You forgot to enter your name!")
    if f.readline() == "## Student Number:\n":
        raise SyntaxError("You forgot to enter your student number!")

# Checking that arguments of functions are correct
for i in range(len(functions)):
    check_args(functions[i],template_functions[i])