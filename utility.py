import numpy as np


def get_overlap_length(intervals):
    temp_tuple = intervals
    temp_tuple.sort(key=lambda interval: interval[0])
    merged = [temp_tuple[0]]
    for current in temp_tuple:
        previous = merged[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
    l=0
    for x in merged:
        l+= x[1]-x[0]
    return l 