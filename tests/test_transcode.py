from numpy import ndarray, arange, array, zeros, random, sum, all, where, copy
from unittest import TestCase

from sbw.spiderweb import SpiderWeb, FastTranscoder, create_coding_digraph, create_coexisting_map
from sbw.underprop import adjust_motif_format, i2m, m2s


class TestNormalTranscoder(TestCase):

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
        self.lengths = [100, 160, 200, 240]
        self.repeat = 100
        self.seed = 2023

    def test(self):
        for _, accessor in self.accessors.items():
            method1, method2 = SpiderWeb(accessor=accessor), FastTranscoder(accessor=accessor)
            virtualities = arange(len(where(sum(((accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0]))
            for length in self.lengths:
                random.seed(self.seed)
                for virtuality in virtualities:
                    binary_messages = random.randint(0, 2, size=(self.repeat, length), dtype=int)
                    # noinspection PyTypeChecker
                    obtained_sequences = method2.encode(messages=binary_messages, virtuality=virtuality)[0]
                    for index in range(self.repeat):
                        # noinspection PyUnresolvedReferences
                        expected_sequence = method1.encode(message=binary_messages[index], virtuality=virtuality)
                        self.assertEqual(obtained_sequences[index] == expected_sequence, True)
                    indices = zeros(shape=(self.repeat,), dtype=int) + virtuality
                    obtained_messages = method2.decode(sequences=obtained_sequences, virtualities=indices,
                                                       length=length)
                    self.assertEqual(all(binary_messages == obtained_messages), True)
                    # noinspection PyTypeChecker
                    obtained_sequences = method2.encode(messages=binary_messages, virtuality=virtuality,
                                                        length_fixed=True)[0]
                    lengths = set([len(sequence) for sequence in obtained_sequences])
                    self.assertEqual(len(lengths), 1)
                    obtained_messages = method2.decode(sequences=obtained_sequences, virtualities=indices,
                                                       length=length)
                    self.assertEqual(all(binary_messages == obtained_messages), True)


class TestAllO3Transcode(TestCase):

    def setUp(self):
        self.method1 = SpiderWeb(accessor=array([[-1, -1, -1, -1], [4, -1, 6, 7], [8, 9, -1, 11], [12, 13, 14, -1],
                                                 [-1, 1, 2, 3], [-1, -1, -1, -1], [8, 9, -1, 11], [12, 13, 14, -1],
                                                 [-1, 1, 2, 3], [4, -1, 6, 7], [-1, -1, -1, -1], [12, 13, 14, -1],
                                                 [-1, 1, 2, 3], [4, -1, 6, 7], [8, 9, -1, 11], [-1, -1, -1, -1]]))

        self.method2 = FastTranscoder(accessor=array([[-1, -1, -1, -1], [4, -1, 6, 7], [8, 9, -1, 11], [12, 13, 14, -1],
                                                      [-1, 1, 2, 3], [-1, -1, -1, -1], [8, 9, -1, 11], [12, 13, 14, -1],
                                                      [-1, 1, 2, 3], [4, -1, 6, 7], [-1, -1, -1, -1], [12, 13, 14, -1],
                                                      [-1, 1, 2, 3], [4, -1, 6, 7], [8, 9, -1, 11], [-1, -1, -1, -1]]))

        self.virtualities = arange(12)
        self.lengths = [100, 160, 200, 240]
        self.repeat = 100
        self.seed = 2023

    def test(self):
        for length in self.lengths:
            random.seed(self.seed)
            for virtuality in self.virtualities:
                for _ in range(self.repeat):
                    # noinspection PyTypeChecker
                    expected_message = random.randint(0, 2, size=(length,), dtype=int)
                    # noinspection PyTypeChecker
                    sequence = self.method1.encode(message=expected_message, virtuality=virtuality)
                    obtained_message = self.method1.decode(sequence=sequence, length=length, virtuality=virtuality)
                    self.assertEqual(all(expected_message == obtained_message), True)

        for length in self.lengths:
            random.seed(self.seed)
            for virtuality in self.virtualities:
                binary_messages = random.randint(0, 2, size=(self.repeat, length), dtype=int)
                # noinspection PyTypeChecker
                obtained_sequences = self.method2.encode(messages=binary_messages, virtuality=virtuality)[0]
                for index in range(self.repeat):
                    # noinspection PyUnresolvedReferences
                    expected_sequence = self.method1.encode(message=binary_messages[index], virtuality=virtuality)
                    self.assertEqual(obtained_sequences[index] == expected_sequence, True)
                indices = zeros(shape=(self.repeat,), dtype=int) + virtuality
                obtained_messages = self.method2.decode(sequences=obtained_sequences, virtualities=indices,
                                                        length=length)
                self.assertEqual(all(binary_messages == obtained_messages), True)


class TestPracticeTranscode(TestCase):

    def setUp(self):
        window_length = 10
        homopolymer = 2
        gc_range = [0.4, 0.6]
        motifs = adjust_motif_format(motif_set=[["GGC", "GGT"], ["AG", "GA", "CT", "TC"]])
        accessor, vertices = create_coding_digraph(window_length=window_length,
                                                   run_length=homopolymer, gc_range=gc_range, undesired_motifs=motifs)
        self.method1 = SpiderWeb(accessor=accessor)
        self.method2 = FastTranscoder(accessor=accessor)
        self.repeat = 10
        self.virtualities = arange(len(vertices))
        self.diversities = [10, 100, 1000, 10000]
        self.lengths = [100, 200, 300, 400]
        self.seed = 2023

    def test(self):
        random.seed(self.seed)
        for length in self.lengths:
            for diversity in self.diversities:
                virtualities = copy(self.virtualities)
                random.shuffle(virtualities)
                for virtuality in self.virtualities[:self.repeat]:
                    e_messages = random.randint(2, size=(diversity, length))
                    # noinspection PyTypeChecker
                    for e_message in e_messages:
                        sequence = self.method1.encode(message=e_message, virtuality=virtuality)
                        o_message = self.method1.decode(sequence=sequence, virtuality=virtuality, length=length)
                        self.assertEqual(all(e_message == o_message), True)
                    # noinspection PyTypeChecker
                    sequences = self.method2.encode(messages=e_messages, virtuality=virtuality)[0]
                    indices = zeros(shape=(diversity,), dtype=int) + virtuality
                    o_messages = self.method2.decode(sequences=sequences, virtualities=indices, length=length)
                    self.assertEqual(all(e_messages == o_messages), True)


class TestCoexistingMap(TestCase):

    def setUp(self):
        window_length = 10
        homopolymer = 2
        gc_range = [0.4, 0.6]
        motifs = adjust_motif_format(motif_set=[["GGC", "GGT"], ["AG", "GA", "CT", "TC"]])

        self.minimum_difference = 4

        accessor, vertices = create_coding_digraph(window_length=window_length,
                                                   run_length=homopolymer, gc_range=gc_range, undesired_motifs=motifs)
        self.coexisting_map = create_coexisting_map(window_length=window_length,
                                                    minimum_difference=self.minimum_difference, vertices=vertices)
        self.vertices = vertices
        self.segments = m2s(matrix=i2m(integers=vertices, length=window_length))

        self.repeat = 1000
        self.length = 10
        self.seed = 2023

    def test(self):
        random.seed(self.seed)

        for current in range(self.repeat):
            coexisting_vertices = [random.choice(self.vertices)]
            for step in range(self.length - 1):
                available_vertices = self.coexisting_map[coexisting_vertices[-1]]
                for saved_vertex in coexisting_vertices:
                    available_vertices &= self.coexisting_map[saved_vertex]
                if len(available_vertices) == 0:
                    break
                coexisting_vertices.append(random.choice(list(available_vertices)))

            used_segments = []
            for vertex in coexisting_vertices:
                used_segments.append(self.segments[int(where(self.vertices == vertex)[0][0])])

            for index_1 in range(len(used_segments) - 1):
                for index_2 in range(index_1 + 1, len(used_segments)):
                    difference = self.levenshtein(list(used_segments[index_1]), list(used_segments[index_2]))
                    self.assertEqual(difference >= self.minimum_difference, True)

    @staticmethod
    def levenshtein(values_1: list,
                    values_2: list) \
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
