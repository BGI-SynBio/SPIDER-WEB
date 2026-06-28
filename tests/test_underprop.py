from numpy import ndarray, copy, arange, array, zeros, random, min, abs, sum, all, uint8
from unittest import TestCase

from sbw.underprop import m2s, s2m, i2m, m2i, hamming, levenshtein


class TestM2S(TestCase):

    def setUp(self):
        self.length = 200
        self.number = 100
        self.repeat = 100
        self.mapping = {0: "A", 1: "C", 2: "G", 3: "T", 4: ""}
        self.seed = 2023

    def test(self):
        random.seed(self.seed)

        for _ in range(self.repeat):
            # noinspection PyArgumentList,PyTypeChecker
            matrix = random.randint(low=0, high=3, size=(self.number, self.length), dtype=uint8)
            obtained_strings = m2s(matrix=copy(matrix))
            # noinspection PyTypeChecker
            for index, (values, obtained_string) in enumerate(zip(matrix, obtained_strings)):
                self.assertEqual(self.integers_2_string(values) == obtained_string.replace("-", ""), True)

    def integers_2_string(self,
                          nucleotide_values: ndarray) -> str:
        string = ""
        for nucleotide_value in nucleotide_values:
            string += self.mapping[nucleotide_value]
        return string


class TestS2M(TestCase):

    def setUp(self):
        self.length = 200
        self.number = 100
        self.repeat = 100
        self.seed = 2023

    def test(self):
        random.seed(self.seed)

        for _ in range(self.repeat):
            # noinspection PyArgumentList,PyTypeChecker
            expected_matrix = random.randint(low=0, high=3, size=(self.number, self.length))
            obtained_strings = m2s(matrix=copy(expected_matrix))
            obtained_matrix = s2m(dna_sequences=obtained_strings)
            self.assertEqual(all(obtained_matrix == expected_matrix), True)


class TestO3(TestCase):

    def setUp(self):
        self.lengths = arange(10, 14, dtype=int)

    def test(self):
        for length in self.lengths:
            integers = arange(2 ** length, dtype=int)
            for integer in integers:
                binary_message = i2m(integers=array([integer]), length=length, base_size=2)[0]

                expected_quotient, expected_reminder = self.origin1(dividend=binary_message)
                obtained_quotient, obtained_reminder = self.design1(dividend=binary_message)
                self.assertEqual(all(expected_quotient == obtained_quotient), True)
                self.assertEqual(expected_reminder, obtained_reminder)

                # noinspection PyTypeChecker
                expected_product = self.origin2(multiplicand=expected_quotient, reminder=expected_reminder)
                # noinspection PyTypeChecker
                obtained_product = self.design2(multiplicand=expected_quotient, reminder=expected_reminder)
                self.assertEqual(all(expected_product == obtained_product), True)

    @staticmethod
    def origin1(dividend: ndarray):
        decimal_value = m2i(matrix=dividend, base_size=2)
        quotient = i2m(integers=array([decimal_value // 3]), length=len(dividend), base_size=2)[0]
        reminder = decimal_value % 3
        return quotient, reminder

    @staticmethod
    def design1(dividend: ndarray):
        quotient = zeros(shape=(len(dividend),), dtype=int)
        reminder = dividend[0]
        for location in arange(1, len(dividend)):
            value = reminder * 2 + dividend[location]
            if value >= 3:
                quotient[location], reminder = 1, value - 3
            else:
                reminder = value
        return quotient, reminder

    @staticmethod
    def origin2(multiplicand: ndarray,
                reminder: int):
        decimal_value = m2i(matrix=multiplicand, base_size=2)
        product = i2m(integers=array([decimal_value * 3 + reminder]), length=len(multiplicand), base_size=2)[0]
        return product

    @staticmethod
    def design2(multiplicand: ndarray,
                reminder: int):
        product = zeros(shape=(len(multiplicand),), dtype=int)
        product[1:] += multiplicand[1:]
        product[:-1] += multiplicand[1:]
        product[-2:] += [reminder // 2, reminder % 2]
        for location in arange(1, len(multiplicand))[::-1]:
            if product[location] > 1:
                product[location - 1] += product[location] // 2
                product[location] %= 2
        return product


class TestPalindromable(TestCase):

    def setUp(self):
        self.length = 10
        self.number = 10000
        self.repeat = 100
        self.seed = 2023

    def test(self):
        random.seed(seed=self.seed)

        for _ in range(self.repeat):
            values = random.randint(0, 4, size=(self.number, self.length))
            # noinspection PyTypeChecker
            expected_results_1 = self.original_design_1(values=values)
            # noinspection PyTypeChecker
            expected_results_2 = self.original_design_2(values=values)
            # noinspection PyUnresolvedReferences
            obtained_results = 3 - values[:, ::-1]  # rapid method.
            self.assertEqual(all(obtained_results == expected_results_1), True)
            self.assertEqual(all(obtained_results == expected_results_2), True)

    @staticmethod
    def original_design_1(values: ndarray):
        segments, palindromable_segments = m2s(matrix=values), []
        for segment in segments:
            palindromable_segment = segment.replace("A", "t").replace("T", "a").replace("C", "g").replace("G", "c")
            palindromable_segment = palindromable_segment.upper()[::-1]  # normal method.
            palindromable_segments.append(palindromable_segment)

        return s2m(dna_sequences=palindromable_segments, assume_equal=True)

    @staticmethod
    def original_design_2(values: ndarray):
        segments, palindromable_segments = m2s(matrix=values), []
        for segment in segments:
            palindromable_segment = segment.translate(str.maketrans("ACGT", "TGCA"))[::-1]  # normal method.
            palindromable_segments.append(palindromable_segment)

        return s2m(dna_sequences=palindromable_segments, assume_equal=True)


class TestHamming(TestCase):

    def setUp(self):
        self.length = 100
        self.numbers = [100, 50]
        self.seed = 2023

    def test(self):
        random.seed(seed=self.seed)

        # noinspection PyTypeChecker
        group = random.randint(0, 4, size=(self.numbers[0], self.length))
        expected_results = zeros(shape=(self.numbers[0], self.numbers[0]), dtype=int)
        for index_1 in range(self.numbers[0] - 1):
            for index_2 in range(index_1 + 1, self.numbers[0]):
                # noinspection PyUnresolvedReferences
                distance = self.origin(values_1=group[index_1], values_2=group[index_2])
                expected_results[index_1, index_2] = distance
                expected_results[index_2, index_1] = distance
        # noinspection PyTypeChecker
        obtained_results = hamming(group=group)
        self.assertEqual(all(obtained_results == expected_results), True)

        # noinspection PyTypeChecker
        another_group = random.randint(0, 4, size=(self.numbers[1], self.length))
        expected_results = zeros(shape=(self.numbers[0], self.numbers[1]), dtype=int)
        for index_1 in range(self.numbers[0]):
            for index_2 in range(self.numbers[1]):
                # noinspection PyUnresolvedReferences
                distance = self.origin(values_1=group[index_1], values_2=another_group[index_2])
                expected_results[index_1, index_2] = distance
        # noinspection PyTypeChecker
        obtained_results = hamming(group=group, another_group=another_group)
        self.assertEqual(all(obtained_results == expected_results), True)

        # noinspection PyTypeChecker
        for sample in another_group:
            expected_results = zeros(shape=(self.numbers[0],), dtype=int)
            for index in range(self.numbers[0]):
                # noinspection PyUnresolvedReferences
                distance = self.origin(values_1=group[index], values_2=sample)
                expected_results[index] = distance
            # noinspection PyTypeChecker
            obtained_results = hamming(group=group, sample=sample)
            self.assertEqual(all(obtained_results == expected_results), True)

    @staticmethod
    def origin(values_1: ndarray,
               values_2: ndarray) \
            -> int:
        return sum(abs(values_1 - values_2).astype(bool))


class TestLevenshtein(TestCase):

    def setUp(self):
        self.length = 100
        self.numbers = [20, 10]
        self.seed = 2023

    def test(self):
        random.seed(seed=self.seed)

        group = random.randint(0, 4, size=(self.numbers[0], self.length))
        expected_results = zeros(shape=(self.numbers[0], self.numbers[0]), dtype=int)
        for index_1 in range(self.numbers[0] - 1):
            for index_2 in range(index_1 + 1, self.numbers[0]):
                # noinspection PyUnresolvedReferences
                distance = self.origin(values_1=group[index_1], values_2=group[index_2])
                expected_results[index_1, index_2] = distance
                expected_results[index_2, index_1] = distance
        # noinspection PyTypeChecker
        obtained_results = levenshtein(group=group)
        self.assertEqual(all(obtained_results == expected_results), True)

        # noinspection PyTypeChecker
        another_group = random.randint(0, 4, size=(self.numbers[1], self.length))
        expected_results = zeros(shape=(self.numbers[0], self.numbers[1]), dtype=int)
        for index_1 in range(self.numbers[0]):
            for index_2 in range(self.numbers[1]):
                # noinspection PyUnresolvedReferences
                distance = self.origin(values_1=group[index_1], values_2=another_group[index_2])
                expected_results[index_1, index_2] = distance
        # noinspection PyTypeChecker
        obtained_results = levenshtein(group=group, another_group=another_group)
        self.assertEqual(all(obtained_results == expected_results), True)

        # noinspection PyTypeChecker
        for sample in another_group:
            expected_results = zeros(shape=(self.numbers[0],), dtype=int)
            for index in range(self.numbers[0]):
                # noinspection PyUnresolvedReferences
                distance = self.origin(values_1=group[index], values_2=sample)
                expected_results[index] = distance
            # noinspection PyTypeChecker
            obtained_results = levenshtein(group=group, sample=sample)
            self.assertEqual(all(obtained_results == expected_results), True)

    @staticmethod
    def origin(values_1: ndarray,
               values_2: ndarray) \
            -> ndarray:
        distances = zeros((len(values_1) + 1, len(values_2) + 1), dtype=int)

        for location in range(len(values_1) + 1):
            distances[location][0] = location

        for location in range(len(values_2) + 1):
            distances[0][location] = location

        for location_1 in range(1, len(values_1) + 1):
            for location_2 in range(1, len(values_2) + 1):
                if values_1[location_1 - 1] == values_2[location_2 - 1]:
                    distances[location_1, location_2] = distances[location_1 - 1, location_2 - 1]
                else:
                    a = distances[location_1, location_2 - 1]
                    b = distances[location_1 - 1, location_2]
                    c = distances[location_1 - 1, location_2 - 1]
                    distances[location_1][location_2] = min([a, b, c]) + 1

        return distances[len(values_1), len(values_2)]
