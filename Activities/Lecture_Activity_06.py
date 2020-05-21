def remove_item(dictionary, key):
    """
    Removes the item with the supplied key from the dictionary, if it exists.
    If the input key does not exist in the dictionary, prints out a message
    saying that the new item has not been removed because there is no matching
    key in the dictionary.
    :param dictionary: the dictionary containing the item being removed
    :param key: the key for the item to be removed
    :return:
    """
    if key not in dictionary:
        print("This item has not been removed because there is no matching key in the dictionary.")
    else:
        dictionary.pop(key)


def add_new_item(dictionary, key, value):
    """
    Adds a new item to the dictionary (using the input key and value)
    if there is no existing item in the dictionary with the supplied key.
    If the input key already exists in the dictionary, print out a message
    saying that the new item has not been added because there is already a
    matching key in the dictionary.
    :param dictionary: the dictionary to add an item to
    :param key: the key to be added to the dictionary
    :param value: value to be associated with key in dictionary
    :return:
    """
    if key in dictionary:
        print("This item has not been added because there is already a matching key in the dictionary.")
    else:
        dictionary[key] = value


dictionnaire = {'Basketballs': 51,
                'Volleyballs': 24,
                'Footballs': 33}
print(dictionnaire)
# Creates dictionary with name 'dictionnaire' and declares three items.
# Output: {'Basketballs': 51, 'Volleyballs': 24, 'Footballs': 33}

remove_item(dictionnaire, 'Basketballs')
print(dictionnaire)
# Removes the 'Basketballs' entry from the dictionary
# Prints the new dictionary with only two items.
# Output: {'Volleyballs': 24, 'Footballs': 33}

remove_item(dictionnaire, 'Golf balls')
print(dictionnaire)
# Unsuccessfully attempts to remove 'Golf balls' from the dictionary.
# Prints current dictionary again with no changes.
# Output: This item has not been removed because there is no matching key in the dictionary.
# Output: {'Volleyballs': 24, 'Footballs': 33}

add_new_item(dictionnaire, 'Baseballs', 15)
print(dictionnaire)
# Adds the 'Baseballs' entry to the dictionary
# Prints the new dictionary with the new item.
# Output: {'Volleyballs': 24, 'Footballs': 33, 'Baseballs': 15}

add_new_item(dictionnaire, 'Footballs', 30)
print(dictionnaire)
# Unsuccessfully attempts to add 'Footballs' to the dictionary.
# Prints current dictionary with no changes.
# Output: This item has not been added because there is already a matching key in the dictionary.
# Output: {'Volleyballs': 24, 'Footballs': 33, 'Baseballs': 15}



