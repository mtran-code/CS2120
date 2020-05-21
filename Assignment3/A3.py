## Name: Michael Tran
## Student Number: 251033279
## Partners: None

from os import walk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
os.environ["PROJ_LIB"] = "C:\\Users\\tranm\\Anaconda3\\Library\\share"  # fix
from mpl_toolkits.basemap import Basemap
import re


def load_station_info(directory='./data/'):
    """
    Loads information about each weather station and stores it in a nested 
    dictionary.
    
    :param directory: Directory (folder) containing the station information 
    file.
    
    :return: A nested dictionary STATION ID -> dict of STATION INFO
    """
    with open(directory + 'Temperature_Stations.csv', 'r') as temp_station_file:
        temp_station_lines = temp_station_file.readlines()[4:]
        temp_station_dict = dict()
        for line in temp_station_lines:
            line = re.sub(r'[^a-zA-Z0-9.,-]+', '', line)
            line = line.split(',')
            line = [str(x) for x in line]
            prov = line[0]
            station_name = line[1]
            station_id = line[2]
            begin_year = int(line[3])
            begin_month = int(line[4])
            end_year = int(line[5])
            end_month = int(line[6])
            lat = float(line[7])
            lon = float(line[8])
            elev = int(line[9])
            joined = line[10]

            temp_station_dict[station_id] = {'prov': prov,
                                             'station_name': station_name,
                                             'station_id': station_id,
                                             'begin_year': begin_year,
                                             'begin_month': begin_month,
                                             'end_year': end_year,
                                             'end_month': end_month,
                                             'lat': lat,
                                             'lon': lon,
                                             'elev': elev,
                                             'joined': joined}

    return temp_station_dict


def load_temperature_data(directory='./data/'):
    """
    Loads temperature data from all files into a nested dict with station_id as
    top level keys. Data for each station is then stored as a dict: 
                  YEAR -> list of monthly mean temperatures.
    NOTE: Missing data is not handled gracefully - it is simply ignored.
    
    :param directory: Directory containing the temperature data files.
    
    :return: A nested dictionary STATION ID -> YEAR -> LIST OF TEMPERATURES
    """
    all_stations_temp_dict = dict()
    for _, _, files in walk(directory):
        for file_name in files:
            if file_name.startswith('mm'):
                station_temp_dict = dict()
                file = open(directory + file_name, 'r')
                station_id = file.readline().strip().split(',')[0]
                file.seek(0)
                file_lines = file.readlines()[4:]
                for line in file_lines:
                    line = re.sub(r'[^a-zA-Z0-9.,-]+', '', line)
                    line = line.strip().split(',')
                    year = int(line[0])
                    monthly_temperatures = []
                    for i in range(1, 24, 2):
                        value = float(line[i])
                        if value > -100:
                            monthly_temperatures.append(value)
                    station_temp_dict[year] = monthly_temperatures
                all_stations_temp_dict[station_id] = station_temp_dict
    return all_stations_temp_dict


def make_valid_temperature_data_dict(station_info_dict,
                                     temperature_dict,
                                     start_year,
                                     end_year):
    """
    Processes the input temperature data dictionary to remove data from 
    stations that do not have valid data over the period from the input start 
    year to the input end year. This routine will change the input temperature
    dictionary.
    
    :param station_info_dict: Dictionary mapping 
     STATION ID -> dict of STATION INFO
    :param temperature_dict: Dictionary mapping
     STATION NAME -> YEAR -> LIST OF TEMPERATURES
    :param start_year: starting year of the valid data to retain.
    :param end_year: ending year of the valid data to retain.
    
    :return: A reduced temperature data dictionary containing data only from 
     stations with valid data over the indicated range of years and only the 
     temperature data over that same range of years.
    """
    td = temperature_dict
    sdd = station_info_dict
    ids_to_remove = []
    for sid in temperature_dict:
        if sdd[sid]['begin_year'] > start_year or sdd[sid]['end_year'] < end_year:
            ids_to_remove.append(sid)
            continue
        is_valid = True
        years_to_remove = []
        for year in td[sid]:
            if year < start_year or year > end_year:
                years_to_remove.append(year)
            elif len(td[sid][year]) != 12:
                is_valid = False
                break
        if not is_valid:
            ids_to_remove.append(sid)
        else:
            for year in years_to_remove:
                td[sid].pop(year)
    for idn in ids_to_remove:
        td.pop(idn)
    return temperature_dict


def draw_map(plot_title, data_dict):
    """
    Draws a map of North America with temperature station names and values. 
    Positive values are drawn next to red dots and negative values next to 
    blue dots. The location of values are determined by the latitude and 
    longitude. A dictionary (data_dict) is used to provide a map from 
    station_name names to a tuple containing the (latitude, longitude, value)
    used for drawing.
    
    :param plot_title: Title for the plot.
    :param data_dict: A dictionary mapping 
     STATION NAME -> tuple(LATITUDE, LONGITUDE, VALUE)
    """
    fig = plt.figure(figsize=(9, 9), dpi=100)
    map1 = Basemap(projection='ortho', resolution=None, lat_0=53, lon_0=-97, )
    map1.etopo(scale=0.5, alpha=0.5)

    for station_name_name in data_dict:
        data = data_dict[station_name_name]
        print(station_name_name, data)
        x, y = map1(data[1], data[0])
        value = data[2]
        color = 'black'
        if value < 0:
            color = 'blue'
        elif value > 0:
            color = 'red'
        plt.plot(x, y, 'ok', markersize=3, color=color)
        plt.text(x, y, '{}\n {:.2f}°C'.format(station_name_name, value), fontsize=8)
    plt.title(plot_title)
    plt.show()
    fig.savefig(plot_title + ".png")


def sort_dictionary_by_absolute_value_ascending(dictionary):
    """
    Sort a dictionary so that the items appear in ascending order according 
    to the values.
    
    :param dictionary: A dictionary.
    
    :return: A sorted list of tupes of key-value pairs
    """

    return sorted(dictionary.items(), key=lambda x: abs(x[1]))


def sort_dictionary_by_absolute_value_descending(dictionary):
    """
    Sort a dictionary so that the items appear in descending order according 
    to the values.
    
    :param dictionary: A dictionary.
    
    :return: A sorted list of tupes of key-value pairs
    """

    return sorted(dictionary.items(), key=lambda x: abs(x[1]), reverse=True)


# CODE BEGINS HERE


def compute_average_temp(temperatures):
    """
    Compute the average of a list of temperatures.
    
    :param temperatures: A list of temperature values.
    
    :return: Their average.
    """
    return np.sum(temperatures) / len(temperatures)
    # straightforward mean formula, could've just used np.mean as well


def compute_average_change(temperatures):
    """
    Compute the average CHANGE over a list of temperatures. For example, if 
    temperatures is [0, 1, 2, 3, 4], this function should return 1.0. If 
    annual_temperatures is [2, 1.5, 1, 0.5, 0], this function should return 
    -0.5
    
    :param temperatures: A list of temperature values.
    
    :return: The average change of these values.
    """
    temp_changes = []
    # create empty list to hold temperature changes
    for index in range(1, (len(temperatures))):
        change = temperatures[index] - temperatures[index - 1]
        temp_changes.append(change)
    # for all temperatures in list (except first one), subtract previous temperature and append differences to list
    avg_change = compute_average_temp(temp_changes)
    # average the list to get overall average change
    return avg_change


def compute_average_changes_dict(station_info_dict,
                                 temperature_data_dict,
                                 start_year,
                                 end_year):
    """
    Create a dictionary mapping STATION IDS to the AVERAGE TEMP CHANGE over the
    range from start_year (inclusive) to end_year (exclusive).
    
    :param station_info_dict: Dictionary mapping 
     STATION ID -> dict of STATION INFO
    :param temperature_data_dict: Dictionary mapping 
     STATION NAME -> YEAR -> LIST OF TEMPERATURES
    :param start_year: The first year to take into account (inclusive)
    :param end_year: The last year to take into account (exclusive)
    
    :return: A dictionary mapping STATION ID -> AVERAGE TEMP CHANGE
    """
    dict_avg_changes = {}
    # create new empty dictionary for output
    for station in temperature_data_dict:
        list_station_avg_annual_temps = []
        # create an empty list to hold mean annual temperatures for a single station
        for year in range(start_year, end_year):
            mean_annual_temp = compute_average_temp(temperature_data_dict[station][year])
            list_station_avg_annual_temps.append(mean_annual_temp)
        # for all relevant years, take the mean annual temperature and add it to the station's list
        dict_avg_changes[station] = compute_average_change(list_station_avg_annual_temps)
    # calculate the average temperature change for the station and add it to dictionary for every station
    return dict_avg_changes


def compute_top_average_change_dict(average_changes_dict, n):
    """
    Create a reduced dictionary that maps STATION IDS to the AVERAGE TEMP 
    CHANGES that contains only the top n AVERAGE TEMP CHANGES (in absolute 
    value).

    :param average_changes_dict: Dictionary mapping 
     STATION ID -> AVERAGE TEMP CHANGE
    :param n: The number of changes to include in the output dictionary.
    
    :return: For convenience, a reference to the reduced input dictionary 
     containing the items in the input with the n largest (in absolute value) 
     temperature changes.
    """
    sorted_values = sort_dictionary_by_absolute_value_descending(average_changes_dict)
    # sort dictionary values so that highest absolute values come first in list
    top_average_change_dict = {}
    # create new empty dictionary for output
    for i in range(n):
        entry_added = sorted_values[0]
        sorted_values.remove(entry_added)
        top_average_change_dict[entry_added[0]] = entry_added[1]
    # for n number of top values, take the first value in sorted list (highest absolute value), remove it,
    # and add it to the new dictionary
    return top_average_change_dict


def make_average_change_dict_for_map(station_info_dict, average_changes_dict):
    """
    Create a dictionary mapping STATION NAMES to 
    tuples(LATITUDE, LONGITUDE, AVERAGE TEMP CHANGE).

    :param station_info_dict: Dictionary mapping 
     STATION ID -> dict of STATION INFO
    :param average_changes_dict: Dictionary mapping
     STATION ID -> AVERAGE TEMP CHANGE
    
    :return: A dictionary mapping 
     STATION NAME -> (LATITUDE, LONGITUDE, AVERAGE TEMP CHANGE)
    """
    new_dict_for_map = {}
    # create new empty dictionary for output
    for station in average_changes_dict:
        station_dict_entry = station_info_dict[station]
        # set station_dict_entry as current station information list
        new_dict_for_map[station_dict_entry['station_name']] = (station_dict_entry['lat'],
                                                                station_dict_entry['lon'],
                                                                average_changes_dict[station])
    # for every station with annual temp change data, find matching station info,
    # and add new dictionary entry mapping station name to latitude, longitude, and average temp change
    return new_dict_for_map


def draw_top_average_changes_map(top_average_changes_dict_for_map,
                                 start_year,
                                 end_year,
                                 num_top_values):
    """
    Given the a dictionary mapping station names to mapping data, together with
    a start_year (inclusive) and end_year (exclusive) and the number of top
    average changes computed, draw a map with this data by calling the draw_map
    function.
    
    :param top_average_changes_dict_for_map: A dictionary mapping 
     STATION NAME -> (LATITUDE, LONGITUDE, AVERAGE TEMP CHANGE)
     containing the num_top_values largest (in absolute value) changes in 
     temperature over the analysis period.
    :param start_year: Start year, as integer, inclusive, for years in 
     analysis.
    :param end_year: End year, as integer, exclusive, for years in 
     analysis.
    :param num_top_values: The number of largest average annual temperature
     changes.
    
    :return: No return statement.
    """
    title = "Top " + str(num_top_values) + " Average Annual Temperature Changes between " \
            + str(start_year) + " and " + str(end_year - 1) + "."
    draw_map(title, top_average_changes_dict_for_map)


def draw_maps_by_range(station_info_dict,
                       valid_temperature_data_dict,
                       start_year,
                       years_per_map,
                       num_top_values,
                       num_maps):
    """
    Given the station data dictionary, a dictionary of valid temperature 
    data over the years to be plotted, a start_year (inclusive, 
    integer), the number of years_per_map (integer), the num_top_values to 
    compute, and the num_maps (integer), draw num_maps maps, each 
    showing the top num_top_values average changes in temperature over 
    years_per_map.

    :param station_info_dict: Dictionary mapping 
     STATION ID -> dict of STATION INFO
    :param valid_temperature_data_dict: Dictionary mapping 
     STATION NAME -> YEAR -> LIST OF TEMPERATURES
     containing valid temperature data over the period from start_year to 
     end_year.
    :param start_year: Start year, as integer, inclusive, for years in 
     analysis.
    :param years_per_map: Number of years in range of each
     map.
    :param num_top_values: The number of largest average annual temperature
     changes.
    :param num_maps: The number of maps to draw.
        
    :return: No return statement.
    """
    map_start_year = start_year
    # set starting year for first map
    for current_map in range(num_maps):
        map_end_year = map_start_year + years_per_map
        # calculate end year for map
        average_change_dict = compute_average_changes_dict(station_info_dict,
                                                           valid_temperature_data_dict,
                                                           map_start_year,
                                                           map_end_year)
        # create dict containing average annual temperature changes over specified years
        top_average_change_dict = compute_top_average_change_dict(average_change_dict,
                                                                  num_top_values)
        # create new dict containing only the top n absolute values from previous dict
        top_average_change_dict_for_map = make_average_change_dict_for_map(station_info_dict,
                                                                           top_average_change_dict)
        # create new dict taking previous dict and adding relevant station data for map (name, latitude, longitude)
        draw_top_average_changes_map(top_average_change_dict_for_map,
                                     map_start_year,
                                     map_end_year,
                                     num_top_values)
        # draw the current map according to the previous dict
        map_start_year = map_end_year
        # reset next start year to the current end year for the next map
    # repeat process for however many maps needed


# Testing completed program

station_data = load_station_info()
temperature_data = load_temperature_data()
start_year = 1980
end_year = 2000
temperature_data = make_valid_temperature_data_dict(station_data,
                                                    temperature_data,
                                                    start_year,
                                                    end_year)

# Maps
start_year = 1980
years_per_map = 10
num_top_values = 5
num_maps = 2

draw_maps_by_range(station_data,
                   temperature_data,
                   start_year,
                   years_per_map,
                   num_top_values,
                   num_maps)
