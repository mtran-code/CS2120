import numpy


def make_list_of_lists(n, m):
    output_list = []
    for i in range(n):
        entry = list(numpy.random.randint(1, 10, m))
        output_list.append(entry)
    return output_list


def deep_copy_list_of_lists(input_list):
    output_copy = []
    for i in range(len(input_list)):
        output_copy.append(input_list[i][:])
    return output_copy


test_make_list = make_list_of_lists(3, 5)

test_deep_copy = deep_copy_list_of_lists(test_make_list)

test_deep_copy[0][0] = 2120

print(test_make_list)  # Output [[4, 1, 4, 2, 7], [4, 7, 9, 6, 5], [7, 1, 9, 6, 6]]
print(test_deep_copy)  # Output [[2120, 1, 4, 2, 7], [4, 7, 9, 6, 5], [7, 1, 9, 6, 6]]
