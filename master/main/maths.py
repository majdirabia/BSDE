import numpy as np


def interval_slicing(A, delta):
    return np.arange(min(A), max(A) + 1, delta)


def is_x_in_interval(x, I):
    try:
        return True if ((x - I[0] >= 0) and (x - I[1] < 0)) else False
    except KeyError:
        print("cannot access index of this interval")


def bucketing_by_interval(points, I):
    return sorted([(i, x) for i, x in enumerate(points) if is_x_in_interval(x, I)])


def mean_by_bucket(bucket, Y):
    if len(bucket) > 0:
        Y_int = np.array([Y[x[0]] for x in bucket])
        mean = np.mean(Y_int)
        return mean
    else:
        return None


def get_all_buckets(points, sliced_interval):
    buckets = []
    for i in range(len(sliced_interval) - 1):
        bucket = bucketing_by_interval(points, [sliced_interval[i], sliced_interval[i + 1]])
        buckets.append(bucket)
    return sliced_interval, buckets


def locate_point_in_sliced_interval(x, sliced_interval):
    i = 0
    I = [sliced_interval[i], sliced_interval[i + 1]]
    while i < len(sliced_interval) - 2 and is_x_in_interval(x, I) == False:
        i = i + 1
        I = [sliced_interval[i], sliced_interval[i + 1]]
    if i <= len(sliced_interval) - 2:
        return i
    else:
        return None


def mean_buckets(Y, buckets):
    means = []
    for i, bucket in enumerate(buckets[1]):
        try:
            mean = mean_by_bucket(bucket, Y)
            if mean is not None:
                means.append(mean_by_bucket(bucket, Y))
            elif mean_by_bucket(buckets[1][i - 1], Y) is not None:
                means.append(mean_by_bucket(buckets[1][i - 1], Y))
        except:
            raise Exception("here")

    return buckets[0], means


def hc_one_point_prediction(A, delta, X, Y, x):
    sliced_interval = interval_slicing(A, delta)
    buckets = get_all_buckets(X, sliced_interval)
    regression = mean_buckets(Y, buckets)
    position = locate_point_in_sliced_interval(x, sliced_interval)
    piecewise = regression[1]
    try:
        if position is not None:
            return piecewise[position]
    except:
        raise IndexError("Somehow")
    else:
        if x < min(sliced_interval):
            return piecewise[0]
        elif x >= max(sliced_interval):
            return piecewise[-1]


def hc_regression(A, delta, X, Y):
    return [hc_one_point_prediction(A, delta, X, Y, x) for x in X]
