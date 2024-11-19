def compare_params(big_dict, small_dict):
    # Initialize a list to hold keys with different values
    different_keys = []

    # Iterate over the keys in the smaller dictionary
    for key in small_dict:
        # Check if the key exists in the bigger dictionary and if the values are different
        if key in big_dict and big_dict[key] != small_dict[key]:
            different_keys.append(key)

    return different_keys
