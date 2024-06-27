
def str_to_bool(value):
    value = value.lower()
    if value == 'true':
        return True
    elif value == 'false':
        return False
    else:
        raise ValueError(f"Invalid value: {value}")
