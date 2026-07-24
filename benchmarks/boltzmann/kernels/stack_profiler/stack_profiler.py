import inspect
import time

# import numba

region_name: str = ""
t_start: float = 0.0


def profile(func, *args):
    caller_frame_info = inspect.stack()[1]

    name: str
    is_numba: bool = hasattr(func, "dispatcher")
    if is_numba:
        name = func.dispatcher.py_func.__name__
    else:
        name = func.__name__

    if name != "main":
        parent_name: str = caller_frame_info.function
        identifier = (parent_name, name, caller_frame_info.lineno)
    else:
        identifier = (name, caller_frame_info.lineno)

    if identifier not in profile.calls:
        profile.calls[identifier] = (0, 0)  # time and number of calls

    start = time.time()
    ret = func(*args)
    if is_numba:
        numba.cuda.synchronize()
    stop = time.time()

    previous_data = profile.calls[identifier]
    profile.calls[identifier] = (previous_data[0] + stop - start, previous_data[1] + 1)

    return ret


profile.calls = {}


def region_start(name: str):
    global region_name
    region_name = name
    global t_start
    t_start = time.time()


def region_stop():
    t_stop = time.time()

    caller_frame_info = inspect.stack()[1]
    parent_name: str = caller_frame_info.function
    global region_name
    identifier = (parent_name, region_name, caller_frame_info.lineno)

    if identifier not in profile.calls:
        profile.calls[identifier] = (0, 0)

    previous_data = profile.calls[identifier]
    profile.calls[identifier] = (
        previous_data[0] + t_stop - t_start,
        previous_data[1] + 1,
    )


def get_profile():
    return profile.calls


def reset_profile():
    profile.calls = {}


def print_profile():
    for key in sorted(
        profile.calls.keys(), key=lambda key: key[1] if len(key) == 2 else key[2]
    ):  # sort based on lineno
        print(f"{key}: {profile.calls[key]}")
