def print_function_name_before_execution(f):
    def body(*args, **kwargs):
        print('-ran-> function {}'.format(f.__name__))
        out = f(*args, **kwargs)
        return out
    return body
