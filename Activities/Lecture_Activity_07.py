import numpy
from pylab import *


def load_data(filename='sports_data.csv'):
    import csv
    reader = csv.reader(open(filename, 'r'))
    data_list = []
    for row in reader:
        data_list.append(row)
    return data_list


data_list = load_data()

column_titles = data_list[0]
month_info = []
data = []
for i in range(1, len(data_list)):
    month_info.append(data_list[i][0])
    data.append(data_list[i][1:5])

data_array = numpy.array(data).astype(numpy.float).transpose()

plot(data_array[0])