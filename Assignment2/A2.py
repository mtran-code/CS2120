## CS 2120 Assignment #2 -- Take Back Our World!
## Name: Michael Tran
## Student number: 251033279


import matplotlib.pyplot as plt
import numpy


def make_city(name, neighbours):
    """
    Create a city (implemented as a list).

    :param name: String containing the city name
    :param neighbours: The city's row from an adjacency matrix.
    :return: [name, Infection status, List of neighbours]
    """
    return [name, False, list(numpy.where(neighbours == 1)[0])]


def make_connections(n, density=0.35):
    """
    This function will return a random adjacency matrix of size
    n x n. You read the matrix like this:
    if matrix[2,7] = 1, then cities '2' and '7' are connected.
    if matrix[2,7] = 0, then the cities are _not_ connected.

    :param n: number of cities
    :param density: controls the ratio of 1s to 0s in the matrix
    :returns: an n x n adjacency matrix
    """
    import networkx

    a = numpy.int32(numpy.triu((numpy.random.random_sample(size=(n, n)) < density)))
    G = networkx.from_numpy_matrix(a)
    while not networkx.is_connected(G):
        a = numpy.int32(numpy.triu((numpy.random.random_sample(size=(n, n)) < density)))
        G = networkx.from_numpy_matrix(a)
    numpy.fill_diagonal(a, 1)
    return a + numpy.triu(a, 1).T


def set_up_cities(names=['Toronto', 'New York City',
                         'Los Angeles', 'Mexico City',
                         'Chicago', 'Washington DC',
                         'Arlington County', 'Langley',
                         'Fort Meade', 'Vancouver',
                         'Houston', 'Ottawa',
                         'Jacksonville', 'San Francisco',
                         'Waltham', 'Bethesda']):
    """
    Set up a collection of cities (world) for our simulator.
    Each city is a 3 element list, and our world will be a list of cities.
    
    :param names: A list with the names of the cities in the world.
    :return: a list of cities
    """
    con = make_connections(len(names))
    city_list = []
    for n in enumerate(names):
        city_list += [make_city(n[1], con[n[0]])]
    return city_list


def get_real_world():
    """
    Set up a particular collection of cities (world) for our simulator so that
    all of us have a common model of the real world to work with.
    Each city is a 3 element list, and our world will be a list of cities.

    :return: a list of cities
    """
    return [['Toronto', True, [0, 6, 9, 11, 14]],
            ['New York City', False, [1, 4, 7, 9, 11, 14]],
            ['Los Angeles', False, [2, 5, 7, 9, 10, 11, 12]],
            ['Mexico City', False, [3, 7, 8, 10, 14, 15]],
            ['Chicago', False, [1, 4, 8, 11, 14]],
            ['Washington DC', False, [2, 5, 8, 13, 14, 15]],
            ['Arlington County', False, [0, 6, 7, 10, 12, 14, 15]],
            ['Langley', False, [1, 2, 3, 6, 7, 14, 15]],
            ['Fort Meade', False, [3, 4, 5, 8, 9, 13]],
            ['Vancouver', False, [0, 1, 2, 8, 9, 11, 15]],
            ['Houston', False, [2, 3, 6, 10, 15]],
            ['Ottawa', False, [0, 1, 2, 4, 9, 11, 12, 14]],
            ['Jacksonville', False, [2, 6, 11, 12, 14, 15]],
            ['San Francisco', False, [5, 8, 13, 15]],
            ['Waltham', False, [0, 1, 3, 4, 5, 6, 7, 11, 12, 14, 15]],
            ['Bethesda', False, [3, 5, 6, 7, 9, 10, 12, 13, 14, 15]]]


def draw_world(world):
    """
    Given a list of cities, produces a nice graph visualization. Infected
    cities are drawn as red nodes, clean cities as blue. Edges are drawn
    between neighbouring cities.

    :param world: a list of cities
    """
    import networkx
    import matplotlib.pyplot as plt
    G = networkx.Graph()
    redlist = []
    greenlist = []
    plt.clf()
    # plt.figure(num=None, figsize=(8, 6), dpi=180, facecolor='w', edgecolor='k')
    for city in enumerate(world):
        if city[1][1] == False:
            G.add_node(city[0])
            redlist.append(city[0])
        else:
            G.add_node(city[0], node_color='g')
            greenlist.append(city[0])

        for neighbour in city[1][2]:
            G.add_edge(city[0], neighbour)
    position = networkx.circular_layout(G)
    networkx.draw_networkx_nodes(G, position, nodelist=redlist, node_color="r")
    networkx.draw_networkx_nodes(G, position, nodelist=greenlist, node_color="g")
    networkx.draw_networkx_edges(G, position)
    networkx.draw_networkx_labels(G, position)
    plt.legend(['Lost', 'Regained'])
    plt.show()
    plt.draw()


def print_world(world):
    """
    In case the graphics don't work for you, this function will print
    out the current state of the world as text.

    :param world: a list of cities
    """
    print('{:19}{}'.format('City', 'Regained?'))
    print('----------------------------')
    for city in world:
        print('{:19}{}'.format(city[0], city[1]))


def get_cityno(world, city_in):
    """
    Given a world and a city within it, find the numerical index of 
    that city in the world.

    :param world: a list of cities
    :param city_in: a city
    """
    cityno = 0
    for i, city in enumerate(world):
        if city_in[0] == city[0]:
            cityno = i
    return cityno


def is_connected(world, city1, city2):
    """
    Given a world and two cities within it, determines whether the
    two cities are directly connected in the network.

    :param world: a list of cities
    :param city1: a city
    :param city2: another city
    """
    return get_cityno(world, city1) in city2[2]


def reset_world(world):
    """
    Resets a given world to the state where all cities are lost.
    :param world: a list of cities
    """
    for city in world:
        city[1] = False
    world[0][1] = True


# CODE STARTS HERE


def regain(world, city_no):
    """
    Regains a specified city within a world by setting status to True.

    :param world: The world in which the city resides that will be regained. (list)
    :param city_no: The city number of the city being regained. (int)
    :return: none
    """
    world[city_no][1] = True


def lose(world, city_no):
    """
    Loses a specified city within a world by setting status to False.

    :param world: tTe world in which the city resides that will be lost. (list)
    :param city_no: The city number of the city being regained. (int)
    :return: none
    """
    world[city_no][1] = False


def sim_step(world, p_regain, p_lose):
    """
    Moves forward one simulation step for a given world, and specified probabilities of city gain and city loss.

    :param world: The world for which the simulation progresses one step. (list)
    :param p_regain: The probability of a city being regained in the step. (float)
    :param p_lose: The probability of a city being lost in the step. (float)
    :return: none
    """
    for city in world:
        # Run through all the cities in the world

        if city[1] is False and numpy.random.rand() < p_regain:
            regain(world, get_cityno(world, city))
            # If the city is not in possession, roll within regain probability. If successful, gain that city.

        if city[1] is False and numpy.random.rand() < p_lose:
            lost_city = city[2][numpy.random.randint(0, len(city[2]))]
            lose(world, lost_city)
            # If the city is not in possession, roll within loss probability. If successful, lose a neighboring city.

        regain(world, 0)
        # If the city is the home city (i.e. city number 0), do not lose it and ensure it is in possession.


def sim_multi_step(world, p_regain, p_lose, steps, draw=True):
    """
    Exact same function as sim_step(), however allows for the function to be called multiple times consecutively.

    :param world: The world for which the simulation progresses. (list)
    :param p_regain: The probability of a city being regained in one step. (float)
    :param p_lose: The probability of a city being lost in one step. (float)
    :param steps: Number of steps to be iterated. (int)
    :param draw: Decides if the world should be drawn after every iteration or only after all steps have been made. (bool)
    :return: none
    """
    for i in range(steps):
        sim_step(world, p_regain, p_lose)
        if draw is True:
            draw_world(world)
        elif draw is False and i == steps:
            draw_world(world)


def is_world_saved(world):
    """
    Returns a bool value on whether or not all cities in the specified world are in possession.

    :param world: The world for which to check saved status. (list)
    :return: True if all cities are saved, False otherwise. (bool)
    """
    status = None
    for city in world:
        if city[1] is True:
            status = True
            continue
        elif city[1] is False:
            status = False
            break
    return status


def time_to_save_world(world, p_regain, p_lose):
    """
    Simulates a new world and counts how many simulation steps are needed before the world is saved.

    :param world: The world to simulate. (list)
    :param p_regain: The probability of a city being regained in one step. (float)
    :param p_lose: The probability of a city being lost in one step. (float)
    :return: Number of simulation steps needed to save world. (int)
    """
    reset_world(world)
    regained_world_steps = 0
    while is_world_saved(world) is False:
        sim_step(world, p_regain, p_lose)
        regained_world_steps += 1
    return regained_world_steps


def save_world_many_times(world, n, p_regain, p_lose):
    """
    Simulates multiple worlds and collects data on the simulation steps needed to save each world.

    :param world: The world to simulate. (list)
    :param n: The number of worlds to simulate, and therefore the number of data entries in the output list. (int)
    :param p_regain: The probability of a city being regained in one step. (float)
    :param p_lose: The probability of a city being lost in one step. (float)
    :return: A list containing number of simulation steps needed to save the world each time. (list)
    """
    times_list_data = []
    for i in range(n):
        entry = time_to_save_world(world, p_regain, p_lose)
        times_list_data.append(entry)
    return times_list_data


# SIMULATION SETUP, DEFINE SIMULATION VALUES HERE

number_of_simulations = 5000
# Self-explanatory

regain_probability = 0.5
# Self-explanatory

loss_probability = 0.1
# Self-explanatory

steps_to_save_max = 20
# Set max x-axis value.

simulation_frequency_max = 1500
# Set max y-axis value.

with_invgauss_distribution_fit = True
# If true, plots an inverse gaussian distribution curve of best fit over histogram.

with_normal_distribution_fit = True
# If true, plots a normal distribution curve of best fit over histogram.
# Ineffective as a curve fit, invgauss has much better fit to data in all cases.


# SIMULATION HISTOGRAM GENERATION

earth = get_real_world()
data = save_world_many_times(earth, number_of_simulations, regain_probability, loss_probability)
# Sets up world 'earth' and runs simulations using given parameters to generate data to be plotted.

fig, ax = plt.subplots()
values, bins, args = ax.hist(data, bins=range(0, steps_to_save_max),
                             color='green', align='mid', rwidth=0.8, alpha=0.8)
# Generates histogram plt using given data with number of columns (bins) set to a previously defined max.
# Histogram columns are green, middle aligned, and set to 80% width to add spacing between columns.

if steps_to_save_max <= 20:
    plt.xticks(numpy.arange(-0.5, steps_to_save_max + 0.5, 1), range(0, steps_to_save_max + 1))
elif 20 < steps_to_save_max <= 40:
    plt.xticks(numpy.arange(-0.5, steps_to_save_max + 0.5, 2), range(0, steps_to_save_max + 1, 2))
else:
    plt.xticks(numpy.arange(-0.5, steps_to_save_max + 0.5, 5), range(0, steps_to_save_max + 1, 5))
# Sets an appropriate x-axis scale from the previously defined limit, steps_to_save_max.
# Also shifts all tick marks down by 0.5 while keeping integer labels, to center on histogram bars

if simulation_frequency_max <= 100:
    plt.yticks(range(0, simulation_frequency_max + 1, 10))
elif 100 < simulation_frequency_max <= 200:
    plt.yticks(range(0, simulation_frequency_max + 1, 20))
elif 200 < simulation_frequency_max <= 400:
    plt.yticks(range(0, simulation_frequency_max + 1, 50))
elif 400 < simulation_frequency_max <= 1000:
    plt.yticks(range(0, simulation_frequency_max + 1, 100))
else:
    plt.yticks(range(0, simulation_frequency_max + 1, 500))
# Sets an appropriate y-axis scale from the previously defined limit, simulation_frequency_max.

plt.ylim(0, simulation_frequency_max)
# Set y-axis limit to defined value.

plt.grid(True, 'major', 'y', alpha=0.6)
# Adds major horizontal grid-lines only.

plt.title('Probability Spread of Steps Needed to Save the World')
plt.xlabel('Steps to Save the World')
plt.ylabel('Frequency of Simulations')
# Define title and axis titles

plt.text(steps_to_save_max * 0.95, simulation_frequency_max * 0.65,
         'number of simulations run = ' + str(number_of_simulations)
         + '\np_regain = ' + str(regain_probability)
         + '\np_lose = ' + str(loss_probability),
         ha='right', linespacing=1.8)
# Adds a legend indicating the number of simulations, p_regain value, and p_lose value.
# Places this legend right justified near the right edge, and at 65% of the height of the graph.

if with_normal_distribution_fit:
    from scipy.stats import norm

    norm_values = numpy.linspace(0, bins.max(), steps_to_save_max) - 0.5
    parameter_estimates = norm.fit(data)
    normal_fit = number_of_simulations * norm.pdf(norm_values, *parameter_estimates)
    ax.plot(normal_fit, '--', color='red', label='normal distribution curve fit')
    ax.legend(loc='upper right')
# If with_normal_distribution_fit is True, generate a line space with same dimensions as histogram.
# Use these values with scipy.stats.norm to generate a normal curve of best fit as a red dashed line.

if with_invgauss_distribution_fit:
    from scipy.stats import invgauss

    invgauss_values = numpy.linspace(0, bins.max(), steps_to_save_max) - 0.5
    parameter_estimates = invgauss.fit(data)
    invgauss_fit = number_of_simulations * invgauss.pdf(invgauss_values, *parameter_estimates)
    ax.plot(invgauss_fit, '--', color='blue', label='inverse gaussian distribution curve fit')
    ax.legend(loc='upper right')
# If with_invgauss_distribution_fit is True, generate a line space with same dimensions as histogram.
# Use these values with scipy.stats.invgauss to generate an inverse gaussian curve of best fit as a blue dashed line.

plt.show()
