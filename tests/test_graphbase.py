from numpy import ndarray, array, zeros, ones, random, count_nonzero, abs, min, sum, log2, all, any, where
from unittest import TestCase

from sbw import approximate_capacity


class TestOutDegree(TestCase):

    def setUp(self):
        self.accessors = {
            "1+2": array([[-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1],
                          [-1, -1, 2, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, 14, -1],
                          [-1, 1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, 13, -1, -1],
                          [-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1]]),
            "2": array([[-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1],
                        [-1, 1, 2, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, 13, 14, -1],
                        [-1, 1, 2, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, 13, 14, -1],
                        [-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1]]),
            "4": array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
        }

    def test(self):
        for label, accessor in self.accessors.items():
            expected_result = []
            for vertex in accessor:
                expected_result.append(len(where(vertex >= 0)[0]))
            expected_result = array(expected_result)
            obtained_result = count_nonzero(accessor + 1, axis=1)  # used in the core code.
            self.assertEqual(all(expected_result == obtained_result), True)


class TestInDegree(TestCase):

    def setUp(self):
        self.accessors = {
            "1+2": array([[-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1],
                          [-1, -1, 2, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, 14, -1],
                          [-1, 1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, 13, -1, -1],
                          [-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1]]),
            "2": array([[-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1],
                        [-1, 1, 2, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, 13, 14, -1],
                        [-1, 1, 2, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, 13, 14, -1],
                        [-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1]]),
            "4": array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
        }

    def test(self):
        for label, accessor in self.accessors.items():
            expected_result = zeros(shape=(len(accessor),), dtype=int)
            for latters in accessor:
                for latter in latters:
                    if latter >= 0:
                        expected_result[latter] += 1
            obtained_result = zeros(shape=(len(accessor),), dtype=int)  # used in the core code.
            for piece in accessor.reshape(len(accessor) // 4, 16):
                obtained_result[piece[piece >= 0]] += 1
            self.assertEqual(all(expected_result == obtained_result), True)


class TestVertexDetection(TestCase):

    def setUp(self):
        self.accessors = {
            "1+2": array([[-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1],
                          [-1, -1, 2, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, 14, -1],
                          [-1, 1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, 13, -1, -1],
                          [-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1]]),
            "2": array([[-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1],
                        [-1, 1, 2, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, 13, 14, -1],
                        [-1, 1, 2, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, 13, 14, -1],
                        [-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1]]),
            "3": array([[-1, -1, -1, -1], [4, -1, 6, 7], [8, 9, -1, 11], [12, 13, 14, -1],
                        [-1, 1, 2, 3], [-1, -1, -1, -1], [8, 9, -1, 11], [12, 13, 14, -1],
                        [-1, 1, 2, 3], [4, -1, 6, 7], [-1, -1, -1, -1], [12, 13, 14, -1],
                        [-1, 1, 2, 3], [4, -1, 6, 7], [8, 9, -1, 11], [-1, -1, -1, -1]]),
            "4": array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
        }

    def test(self):
        for name, accessor in self.accessors.items():
            expected_result = self.normal_detect(accessor=accessor)
            obtained_result = where(sum(((accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0]
            self.assertEqual(all(expected_result == obtained_result), True)

    @staticmethod
    def normal_detect(accessor: ndarray) \
            -> ndarray:
        vertices = []
        for index, vertex in enumerate(accessor):
            # noinspection PyTypeChecker
            if any(vertex >= 0):
                vertices.append(index)

        return array(vertices)


class TestDuplicateSearch(TestCase):

    def setUp(self):
        self.lengths = [10, 20, 30, 40, 50]
        self.numbers = [100, 200, 300, 400, 500]
        self.ranges = [(0, 100), (0, 1000), (0, 10000), (0, 100000)]
        self.repeat = 10
        self.seed = 2023

    def test(self):
        random.seed(self.seed)
        for length in self.lengths:
            for number in self.numbers:
                for low, high in self.ranges:
                    for _ in range(self.repeat):
                        paths = random.randint(low, high, size=(length, number))
                        # noinspection PyTypeChecker
                        expected_result = self.normal_detect(paths=paths)
                        # noinspection PyTypeChecker
                        obtained_result = self.parallel_detect(paths=paths)
                        self.assertEqual(all(expected_result == obtained_result), True)

    @staticmethod
    def normal_detect(paths: ndarray) \
            -> ndarray:
        duplicate_flags = zeros(shape=(paths.shape[1],), dtype=bool)

        for index, path in enumerate(paths.T):
            if len(path) > len(set(path)):
                duplicate_flags[index] = 1

        return duplicate_flags

    @staticmethod
    def parallel_detect(paths: ndarray) \
            -> ndarray:
        duplicate_flags = zeros(shape=(paths.shape[1],), dtype=bool)

        for index, level in enumerate(paths[1:]):
            action = where(level >= 0)[0]
            detections = min(abs(paths[:index + 1, action] - level[action]), axis=0)
            duplicate_flags[action[where(detections == 0)[0]]] = 1

        return duplicate_flags


class TestPalindromableSearch(TestCase):

    def setUp(self):
        self.orders = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]

        self.accessors = {
            "1+2": array([[-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1],
                          [-1, -1, 2, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, 14, -1],
                          [-1, 1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, 13, -1, -1],
                          [-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1]]),
            "2": array([[-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1],
                        [-1, 1, 2, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, 13, 14, -1],
                        [-1, 1, 2, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, 13, 14, -1],
                        [-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1]]),
            "3": array([[-1, -1, -1, -1], [4, -1, 6, 7], [8, 9, -1, 11], [12, 13, 14, -1],
                        [-1, 1, 2, 3], [-1, -1, -1, -1], [8, 9, -1, 11], [12, 13, 14, -1],
                        [-1, 1, 2, 3], [4, -1, 6, 7], [-1, -1, -1, -1], [12, 13, 14, -1],
                        [-1, 1, 2, 3], [4, -1, 6, 7], [8, 9, -1, 11], [-1, -1, -1, -1]]),
            "4": array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
        }

        self.normal_maps = {}
        for name, accessor in self.accessors.items():
            self.normal_maps[name] = {}
            for former, vertex in enumerate(accessor):
                for latter in vertex[vertex >= 0]:
                    # noinspection PyUnresolvedReferences
                    if self.orders[former] == self.orders[latter].translate(str.maketrans("ACGT", "TGCA"))[::-1]:
                        self.normal_maps[name][former] = latter

        self.parallel_maps = {}
        for name, normal_map in self.normal_maps.items():
            matrix = -ones(shape=(len(self.orders,)), dtype=int)
            for former, latter in normal_map.items():
                matrix[former] = latter
            self.parallel_maps[name] = matrix

        self.lengths = [2, 4, 6, 8, 10]
        self.numbers = [100, 200, 300, 400, 500]
        self.repeat = 10
        self.seed = 2023

    def test(self):
        random.seed(self.seed)

        for _ in range(self.repeat):
            for number in self.numbers:
                for length in self.lengths:
                    for name, accessor in self.accessors.items():
                        vertices, paths = where(sum(((accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0], []
                        for index in range(number):
                            vertex, path = random.choice(vertices), []
                            for step in range(length):
                                available = accessor[vertex]
                                available = available[available >= 0]
                                path.append(random.choice(available))
                                vertex = path[-1]
                            paths.append(path)
                        paths = array(paths).T

                        expected_result = self.normal_detect(name=name, paths=paths)
                        obtained_result = self.parallel_detect(name=name, paths=paths)
                        self.assertEqual(all(expected_result == obtained_result), True)

    def normal_detect(self,
                      name: str,
                      paths: ndarray) \
            -> ndarray:
        palindrome_flags = zeros(shape=(paths.shape[1],), dtype=bool)
        for index, path in enumerate(paths.T):
            for step, value in enumerate(path[1:]):
                if value in self.normal_maps[name] and self.normal_maps[name][value] in path[:step + 1]:
                    palindrome_flags[index] = 1
                    break

        return palindrome_flags

    def parallel_detect(self,
                        name: str,
                        paths: ndarray) \
            -> ndarray:
        palindrome_flags = zeros(shape=(paths.shape[1],), dtype=bool)

        for index, level in enumerate(paths[1:]):
            action_1 = where(level >= 0)[0]
            palindromable_level = self.parallel_maps[name][level[action_1]]
            action_2 = where(palindromable_level >= 0)[0]
            detections = min(abs(paths[:index + 1, action_1[action_2]] - level[action_1[action_2]]), axis=0)
            palindrome_flags[action_1[action_2[where(detections == 0)[0]]]] = 1

        return palindrome_flags


class TestCapacityApproximation(TestCase):

    def setUp(self):
        self.accessors = {
            "2": array([[-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1],
                        [-1, 1, 2, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, 13, 14, -1],
                        [-1, 1, 2, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, 13, 14, -1],
                        [-1, -1, -1, -1], [4, -1, -1, 7], [8, -1, -1, 11], [-1, -1, -1, -1]]),
            "3": array([[-1, -1, -1, -1], [4, -1, 6, 7], [8, 9, -1, 11], [12, 13, 14, -1],
                        [-1, 1, 2, 3], [-1, -1, -1, -1], [8, 9, -1, 11], [12, 13, 14, -1],
                        [-1, 1, 2, 3], [4, -1, 6, 7], [-1, -1, -1, -1], [12, 13, 14, -1],
                        [-1, 1, 2, 3], [4, -1, 6, 7], [8, 9, -1, 11], [-1, -1, -1, -1]]),
            "4": array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
        }

    def test(self):
        for accessor_label, accessor in self.accessors.items():
            value = abs(log2(float(accessor_label)) - approximate_capacity(accessor=accessor))
            self.assertEqual(value < 1e-10, True)
