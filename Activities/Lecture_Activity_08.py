import csv
import matplotlib.pyplot as plt
import numpy as np


csv_reader = csv.reader(open('./sports_data.csv', 'r'))
data = []
for line in csv_reader:
    data.append(line)


def compute_annual_search_sums(desired_year):
    """
    For a given year, determine the sum of all searches for each sports league.

    :param desired_year: The year for which the data should be considered.

    :return: A list of total sums of sports searches for the desired year. (In order NHL, MLB, NBA, NFL)
    """
    start_slice = (desired_year - 2003) + (11*(desired_year - 2004))
    sliced_data = data[start_slice:(start_slice + 12)]
    nhl_sum = np.sum([int(row[1]) for row in sliced_data])
    mlb_sum = np.sum([int(row[2]) for row in sliced_data])
    nba_sum = np.sum([int(row[3]) for row in sliced_data])
    nfl_sum = np.sum([int(row[4]) for row in sliced_data])
    return [nhl_sum, mlb_sum, nba_sum, nfl_sum]


input_year = 2017
calculated_values = compute_annual_search_sums(2017)
plt.pie(calculated_values, labels=['NHL', 'MLB', 'NBA', 'NFL'])
plt.title('Relative search volume of professional Canadian sports leagues in ' + str(input_year))
plt.show()
