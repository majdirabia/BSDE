import numpy as np
import maths
import unittest


class TestHypercubes(unittest.TestCase):

    def test_slice_interval(self):
        reference = np.array([1, 3, 5, 7, 9])

        solution = maths.interval_slicing([1, 10],  2)
        print(maths.interval_slicing([1, 10],  2))
        self.assertListEqual(solution.tolist(), reference.tolist())

    def test_point_is_in_interval(self):
        x = 4
        I = [1, 5]
        self.assertTrue(maths.is_x_in_interval(x, I))

    def test_point_is_not_in_interval(self):
        x = 0
        I = [1, 5]
        self.assertFalse(maths.is_x_in_interval(x, I))

    def test_bucketing_by_interval(self):
        I = [1, 9]
        points = [0, 2, 3, 6, 10, 12, 9]
        reference = [(1, 2), (2, 3), (3, 6)]

        bucket = maths.bucketing_by_interval(points, I)
        self.assertListEqual(reference, bucket)

    def test_mean_by_bucket(self):
        I = [1, 9]
        points = [0, 2, 3, 6, 10, 12, 9]
        Y = [x ** 2 for x in points]  # [0, 4, 9, 36, 100, 144, 81]
        reference = 49 / 3
        bucket = maths.bucketing_by_interval(points, I)

        solution = maths.mean_by_bucket(bucket, Y)
        self.assertAlmostEqual(solution, reference)

    def test_mean_by_empty_bucket(self):
        I = [1, 9]
        points = [0, 10, 12]
        Y = [x ** 2 for x in points]  # [0, 4, 9, 36, 100, 144, 81]
        reference = None
        bucket = maths.bucketing_by_interval(points, I)

        solution = maths.mean_by_bucket(bucket, Y)
        self.assertEqual(solution, reference)

    def test_get_all_buckets(self):
        A = [0, 100]
        slicing_size = 20
        points = [1, 2, 3, 19, 40, 100, 120, 49, 29, 78, 79, 89]

        reference = ([0, 20, 40, 60, 80, 100],
                     [[(0, 1), (1, 2), (2, 3), (3, 19)], [(8, 29)], [(4, 40), (7, 49)], [(9, 78), (10, 79)],
                      [(11, 89)]])

        solution = maths.get_all_buckets(points, sliced_interval=maths.interval_slicing(A, slicing_size))
        self.assertListEqual(reference[0], solution[0].tolist())
        self.assertListEqual(reference[1], solution[1])

    def test_locate_point_in_sliced_interval(self):
        A = [0, 100]
        slicing_size = 20
        point = 18
        reference = 0
        sliced_interval = maths.interval_slicing(A, slicing_size)

        solution = maths.locate_point_in_sliced_interval(point, sliced_interval)
        self.assertEqual(reference, solution)

    def test_mean_buckets(self):
        A = [0, 100]
        slicing_size = 20
        points = [1, 2, 3, 19, 40, 100, 120, 49, 29, 78, 79, 89]
        Y = [point ** 2 for point in points]
        reference = ([0, 20, 40, 60, 80, 100], [93.75, 841.0, 2000.5, 6162.5, 7921.0])

        solution = maths.mean_buckets(Y, maths.get_all_buckets(points, maths.interval_slicing(A, slicing_size)))
        self.assertListEqual(reference[1], solution[1])

    def test_mean_with_empty_bucket(self):
        A = [0, 100]
        slicing_size = 20
        points = [1, 2, 3, 19, 40, 49, 29, 78, 79]
        Y = [point ** 2 for point in points]
        reference = [93.75, 841.0, 2000.5, 6162.5, 6162.5]

        solution = maths.mean_buckets(Y, maths.get_all_buckets(points, maths.interval_slicing(A, slicing_size)))
        self.assertListEqual(solution[1], reference)

    def test_hc_one_point_regression_(self):
        A = [0, 100]
        slicing_size = 20
        X = [1, 2, 3, 19, 40, 100, 120, 49, 29, 78, 79, 89]
        Y = [x ** 2 for x in X]
        x = 50
        reference = 2000.5

        solution = maths.hc_one_point_prediction(A, slicing_size, X, Y, x)
        self.assertEqual(reference, solution)

    def test_one_point_regression_left(self):
        A = [0, 100]
        slicing_size = 20
        X = [1, 2, 3, 19, 40, 100, 120, 49, 29, 78, 79, 89]
        Y = [x ** 2 for x in X]
        x = -1
        reference = 93.75

        solution = maths.hc_one_point_prediction(A, slicing_size, X, Y, x)
        self.assertEqual(reference, solution)

    def test_hc_regression(self):
        A = [0, 100]
        slicing_size = 20
        X = [1, 2, 3, 19, 40, 100, 120, 49, 29, 78, 79, 89]
        Y = [x ** 2 for x in X]
        reference = [93.75, 93.75, 93.75, 93.75, 2000.5, 7921.0, 7921.0, 2000.5, 841.0, 6162.5, 6162.5, 7921.0]

        solution = maths.hc_regression(A, slicing_size, X, Y)
        self.assertListEqual(reference, solution)

