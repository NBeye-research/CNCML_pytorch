import os


DEFAULT_ROOT = './materials'


datasets = {}
def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    if kwargs.get('root_path') is None:
        kwargs['root_path'] = os.path.join(DEFAULT_ROOT, name)
    if kwargs.get('test_data_path_arr') is not None:
        dataset = []
        for tmp in kwargs.get('test_data_path_arr'):
            kwargs['root_path'] = tmp
            tmp_data = datasets[name](**kwargs)
            dataset.append(tmp_data)
    else:
        dataset = datasets[name](**kwargs)
    return dataset

