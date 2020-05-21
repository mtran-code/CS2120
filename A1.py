## CS 2120 Assignment #1
## Name: Michael Tran
## Student number: 251033279


def load_a1_data(filename='London_mean_etr_max_etr_min.csv'):
    import re

    with open(filename, 'r') as file:
        records = []

        for r in file:
            r = re.sub(r'[^a-zA-Z0-9.,-]+', '', r)
            s = r.split(',')
            s = [str(x) for x in s]
            records.append(s)

    return records


Run = load_a1_data(filename='London_mean_etr_max_etr_min.csv')


def convert_month(month_number):
    month_list = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
    return month_list[month_number - 1]


def coldest_month(records):
    coldest_year_so_far = None
    coldest_month_so_far = None
    coldest_temp_so_far = 100

    for record in records:

        cold_year = int(record[0])
        cold_month = int(record[1])
        cold_temp = float(record[4])

        if cold_temp < coldest_temp_so_far:
            coldest_temp_so_far = cold_temp
            coldest_month_so_far = cold_month
            coldest_year_so_far = cold_year

    return convert_month(coldest_month_so_far) + ' ' \
        + str(coldest_year_so_far) + " was the coldest month on record with an extreme minimum temperature of " \
        + str(coldest_temp_so_far) + " degrees Celsius."


print(coldest_month(Run))


def warmest_month(records):
    warmest_year_so_far = None
    warmest_month_so_far = None
    warmest_temp_so_far = -100

    for record in records:

        warm_year = int(record[0])
        warm_month = int(record[1])
        warm_temp = float(record[3])

        if warm_temp > warmest_temp_so_far:
            warmest_temp_so_far = warm_temp
            warmest_month_so_far = warm_month
            warmest_year_so_far = warm_year

    return convert_month(warmest_month_so_far) + ' ' \
        + str(warmest_year_so_far) + ' was the hottest month on record with an extreme maximum of ' \
        + str(warmest_temp_so_far) + ' degrees Celsius.'


print(warmest_month(Run))


def print_mean_annual_temperature(year, records):
    import math

    month_count = 0
    mean_temp = 0
    success = True

    for record in records:

        if int(record[0]) != int(year):
            pass
        elif int(record[0]) == int(year) and not math.isnan(float(record[2])):
            mean_temp += float(record[2])
            month_count += 1
        else:
            success = False

    if month_count < 12:
        success = False
    mean_temp = round(mean_temp / 12, 2)

    if not success:
        print('The temperature data for that year is unavailable, and the annual mean could not be calculated.')
    else:
        print('The mean annual temperature for ' + str(year) + ' is ' + str(mean_temp) + ' degrees Celsius.')


print_mean_annual_temperature(2000, Run)

import math

mean_annual_temperatures = []
mean_years = []
month_count = 1
year_count = 0
mean_temp = 0

for record in Run:

    if int(record[0]) == int(year_count):
        if not math.isnan(float(record[2])):
            mean_temp += float(record[2])
            month_count += 1
        else:
            pass
    else:
        if month_count < 12:
            mean_temp = float(record[2])
            year_count = record[0]
            month_count = 1
        else:
            mean_temp = mean_temp / month_count
            mean_annual_temperatures.append(float(mean_temp))
            mean_years.append(int(year_count))
            year_count = record[0]
            mean_temp = float(record[2])
            month_count = 1

import matplotlib.pyplot as plt

plt.title('Average Annual Temperature in London, Ontario')
plt.plot(mean_years, mean_annual_temperatures)
plt.ylabel('Average Temperature')
plt.xlabel('Year')
plt.grid(True)
plt.show()
