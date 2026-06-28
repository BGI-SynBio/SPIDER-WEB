from collections import Counter
from datetime import datetime
from numpy import ndarray, array, zeros, ones, arange, in1d, all, vstack, expand_dims, repeat, swapaxes, add, dot
from numpy import abs, sum, min, max, log, log2, where, uint8, copy, load as n_load, save
from pickle import load as p_load, dump
from typing import Tuple, Optional


class Monitor(object):

    def __init__(self):
        """
        Initialize the monitor to identify the task progress.
        """
        self.last_time = None

    def __call__(self,
                 moment: int,
                 finish: int,
                 extra: Optional[dict] = None):
        """
        Output the current state of process.

        :param moment: current state of process.
        :type moment: int

        :param finish: total state of process.
        :type finish: int

        :param extra: extra vision information if required.
        :type extra: dict
        """
        if self.last_time is None:
            self.last_time = datetime.now()

        position = int(moment / finish * 100)

        string = "|"

        for index in range(0, 100, 5):
            if position >= index:
                string += "█"
            else:
                string += " "

        string += "|"

        string += " " * (3 - len(str(position))) + str(position) + "% ("
        string += " " * (len(str(finish)) - len(str(moment))) + str(moment) + "/" + str(finish)

        pass_time = (datetime.now() - self.last_time).total_seconds()
        if moment > 0:
            wait_time = int(pass_time * (finish - moment) / moment)
            if moment < finish:
                minute, second = divmod(wait_time, 60)
                hour, minute = divmod(minute, 60)
                string += ") wait " + "%04d:%02d:%02d" % (hour, minute, second)
            else:
                minute, second = divmod(pass_time, 60)
                hour, minute = divmod(minute, 60)
                string += ") used " + "%04d:%02d:%02d" % (hour, minute, second)
                self.runtime = pass_time
        else:
            string += ") wait unknown time"

        if extra is not None:
            string += " " + str(extra).replace("\'", "").replace("{", "(").replace("}", ")") + "."
        else:
            string += "."

        print("\r" + string, end="", flush=True)

        if moment >= finish:
            self.last_time = None
            print()

    def reset(self):
        """
        Reset the monitor.
        """
        self.last_time = None


def save_data(file_path: str,
              information: object):
    """
    Save data to the file.

    :param file_path: file path.
    :type file_path: str

    :param information: information for saving.
    :type information: object
    """
    if ".pkl" in file_path:
        with open(file_path, "wb") as file:
            dump(obj=information, file=file)
    elif ".npy" in file_path:
        # noinspection PyTypeChecker
        save(file=file_path, arr=information)
    else:
        raise ValueError("No such type of file path.")


def load_data(file_path: str):
    """
    Load data from the file path.

    :param file_path: file path.
    :type file_path: str

    :return: data.
    """
    if ".pkl" in file_path:
        with open(file_path, "rb") as file:
            return p_load(file=file)
    elif ".npy" in file_path:
        return n_load(file_path)
    else:
        raise ValueError("No such type of file path.")


def adjust_motif_format(motif_set: list,
                        bases: Optional[tuple] = None) \
        -> list:
    """
    Adjust motif format from sequence list set to numerical matrix.

    :param motif_set: motif set.
    :type motif_set: list

    :param bases: base system, i.e. a tuple: ("A", "C", "G", "T").
    :type bases: tuple

    :return: motif list.
    :rtype: list
    """
    new_motifs, used_bases = [], ("A", "C", "G", "T") if bases is None else bases

    for motifs in motif_set:
        for motif in motifs:
            new_motifs.append(array([used_bases.index(base) for base in motif]))

    return new_motifs


def get_latters(vertices: ndarray,
                base: int,
                window_length: int,
                base_size: int = 4) \
        -> ndarray:
    """
    Obtain latter vertices based on the current vertices.

    :param vertices: current vertices, formed as integers.
    :type vertices: numpy.ndarray

    :param base: base of the latter value, e.g. 0, 1, 2, 3 for 4-ary digits.
    :type base: int

    :param window_length: window length of vertices.
    :type window_length: int

    :param base_size: size of base, e.g. the size of nucleotide is 4.
    :type base_size: int

    :return: latter vertices.
    :rtype: numpy.ndarray
    """
    return (vertices * base_size + base) % (base_size ** window_length)


def get_formers(vertices: ndarray,
                base: int,
                window_length: int,
                base_size: int = 4) \
        -> ndarray:
    """
    Obtain former vertices based on the current vertices.

    :param vertices: current vertices, formed as integers.
    :type vertices: numpy.ndarray

    :param base: base of the latter value, e.g. 0, 1, 2, 3 for 4-ary digits.
    :type base: int

    :param window_length: window length of vertices.
    :type window_length: int

    :param base_size: size of base, e.g. the size of nucleotide is 4.
    :type base_size: int

    :return: former vertices.
    :rtype: numpy.ndarray
    """
    # noinspection PyTypeChecker
    return vertices // base_size + base * int(base_size ** (window_length - 1))


def hamming(group: ndarray,
            another_group: Optional[ndarray] = None,
            sample: Optional[ndarray] = None) \
        -> ndarray:
    """
    Rapidly calculate Hamming distances of inner-group, two groups, group versus sample.

    :param group: sample group.
    :type group: numpy.ndarray

    :param another_group: another sample group if required.
    :type another_group: numpy.ndarray or None

    :param sample: sample if required.
    :type sample: numpy.ndarray or None.

    :return: Hamming distances.
    :rtype: numpy.ndarray
    """
    if another_group is None and sample is None:
        former = repeat(expand_dims(group, 0), len(group), 0)
        latter = swapaxes(copy(former), 0, 1)
        distances = sum(abs(former - latter).astype(bool), axis=2, dtype=int)

    elif another_group is not None and sample is None:
        former = repeat(expand_dims(another_group, 0), len(group), 0)
        latter = swapaxes(repeat(expand_dims(group, 0), len(another_group), 0), 0, 1)
        distances = sum(abs(former - latter).astype(bool), axis=2, dtype=int)

    elif another_group is None and sample is not None:
        matrix = abs(group - expand_dims(sample, axis=0))
        distances = sum(matrix.astype(bool), axis=1, dtype=int)

    else:
        raise ValueError("Illegal parameter input (two mutually exclusive parameters), "
                         + "the program does not know which process to enter!")

    return distances


def levenshtein(group: ndarray,
                another_group: Optional[ndarray] = None,
                sample: Optional[ndarray] = None) \
        -> ndarray:
    """
    Rapidly calculate Levenshtein distances of inner-group, two groups, group versus sample.

    :param group: sample group.
    :type group: numpy.ndarray

    :param another_group: another sample group if required.
    :type another_group: numpy.ndarray or None

    :param sample: sample if required.
    :type sample: numpy.ndarray or None.

    :return: Levenshtein distances.
    :rtype: numpy.ndarray
    """
    if another_group is None and sample is None:
        distances = zeros(shape=(group.shape[0], group.shape[0], group.shape[1] + 1, group.shape[1] + 1), dtype=int)
        for location in range(group.shape[1] + 1):
            distances[:, :, location, 0] = location
            distances[:, :, 0, location] = location

        for location_1 in range(1, group.shape[1] + 1):
            for location_2 in range(1, group.shape[1] + 1):
                matrix = hamming(group=expand_dims(group[:, location_1 - 1], axis=1),
                                 another_group=expand_dims(group[:, location_2 - 1], axis=1))
                choices = where(matrix == 0)
                if len(choices[0]) > 0:
                    used = distances[choices[0], choices[1], location_1 - 1, location_2 - 1]
                    distances[choices[0], choices[1], location_1, location_2] = used

                choices = where(matrix > 0)
                if len(choices[0]) > 0:
                    used = min(vstack((distances[choices[0], choices[1], location_1 - 0, location_2 - 1],
                                       distances[choices[0], choices[1], location_1 - 1, location_2 - 0],
                                       distances[choices[0], choices[1], location_1 - 1, location_2 - 1])), axis=0) + 1
                    distances[choices[0], choices[1], location_1, location_2] = used

        return distances[:, :, group.shape[1], group.shape[1]]

    elif another_group is not None and sample is None:
        distances = zeros(shape=(group.shape[0], another_group.shape[0],
                                 group.shape[1] + 1, another_group.shape[1] + 1), dtype=int)
        for location in range(group.shape[1] + 1):
            distances[:, :, location, 0] = location
        for location in range(another_group.shape[1] + 1):
            distances[:, :, 0, location] = location

        for location_1 in range(1, group.shape[1] + 1):
            for location_2 in range(1, another_group.shape[1] + 1):
                matrix = hamming(group=expand_dims(group[:, location_1 - 1], axis=1),
                                 another_group=expand_dims(another_group[:, location_2 - 1], axis=1))
                choices = where(matrix == 0)
                if len(choices[0]) > 0:
                    used = distances[choices[0], choices[1], location_1 - 1, location_2 - 1]
                    distances[choices[0], choices[1], location_1, location_2] = used

                choices = where(matrix > 0)
                if len(choices[0]) > 0:
                    used = min(vstack((distances[choices[0], choices[1], location_1 - 0, location_2 - 1],
                                       distances[choices[0], choices[1], location_1 - 1, location_2 - 0],
                                       distances[choices[0], choices[1], location_1 - 1, location_2 - 1])), axis=0) + 1
                    distances[choices[0], choices[1], location_1, location_2] = used

        return distances[:, :, group.shape[1], another_group.shape[1]]

    elif another_group is None and sample is not None:
        distances = zeros(shape=(group.shape[0], group.shape[1] + 1, sample.shape[0] + 1), dtype=int)

        for location in range(group.shape[1] + 1):
            distances[:, location, 0] = location
        for location in range(sample.shape[0] + 1):
            distances[:, 0, location] = location

        for location_1 in range(1, group.shape[1] + 1):
            for location_2 in range(1, sample.shape[0] + 1):
                choices = where(group[:, location_1 - 1] == sample[location_2 - 1])[0]
                distances[choices, location_1, location_2] = distances[choices, location_1 - 1, location_2 - 1]
                choices = where(group[:, location_1 - 1] != sample[location_2 - 1])[0]
                a = distances[choices, location_1 - 0, location_2 - 1]
                b = distances[choices, location_1 - 1, location_2 - 0]
                c = distances[choices, location_1 - 1, location_2 - 1]
                distances[choices, location_1, location_2] = min(vstack((a, b, c)), axis=0) + 1

        return distances[:, group.shape[1], sample.shape[0]]

    else:
        raise ValueError("Illegal parameter input (two mutually exclusive parameters), "
                         + "the program does not know which process to enter!")


def calculate_practical_density(accessor: ndarray,
                                length: int,
                                verbose: bool = False) \
        -> ndarray:
    """
    Calculate the practical density under the specific length

    :param accessor: coding digraph.
    :type accessor: numpy.ndarray

    :param length: expected encoding length.
    :type length: int

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: practical density of each vertex.
    :rtype: numpy.ndarray
    """
    vertices = where(sum(((accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0]
    combinations = zeros(shape=(len(accessor),), dtype=float)
    combinations[vertices] = 1.0
    window_length = int(log(len(accessor)) / log(4))

    monitor, scale_times = Monitor(), 0
    for step in range(length):
        new_combinations = zeros(shape=(len(accessor),), dtype=float)
        for nucleotide in range(4):
            latters = get_latters(vertices=vertices, base=nucleotide, window_length=window_length)
            usable = where(in1d(latters, vertices) == 1)
            add.at(new_combinations, latters[usable], combinations[vertices[usable]])

        # noinspection PyUnresolvedReferences
        if max(new_combinations) > 1e20:  # may cause upper bound overflow
            new_combinations /= 1048576.0  # 2^20
            scale_times += 1

        combinations = new_combinations

        if verbose:
            monitor(step + 1, length)

    # screen vertices without in-degrees.
    in_degrees = zeros(shape=(len(vertices),), dtype=int)
    for nucleotide in range(4):
        formers = get_formers(vertices=vertices, base=nucleotide, window_length=window_length)
        in_degrees += in1d(formers, vertices)
    used_indices = where(in_degrees > 0)[0]

    records = combinations[vertices][used_indices]
    records = records[records > 0]

    # noinspection PyTypeChecker
    return (log2(records) + scale_times * 20) / length


def approximate_capacity(accessor: ndarray,
                         iteration=100) \
        -> float:
    """
    Approximate the capacity of the specific graph through Perron–Frobenius theorem.

    :param accessor: coding digraph.
    :type accessor: numpy.ndarray

    :param iteration: required iteration in the power method.
    :type iteration: int

    :return: capacity.
    :rtype: float
    """
    if all(accessor == -1):
        return 0.0

    vertices = where(sum(((accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0]
    counter, out_statistics = Counter(sum(where(accessor[vertices] >= 0, 1, 0), axis=1)), zeros(shape=(5,), dtype=int)
    for out_degree in [1, 2, 3, 4]:
        if out_degree in counter:
            # noinspection PyTypeChecker
            out_statistics[out_degree] += counter[out_degree]
    if sum(out_statistics[2:]) == 0:
        return 0.0

    ignore_positions = where(sum(accessor, axis=1) == -len(accessor[0]))[0]
    last_eigenvector, last_eigenvalue = ones(shape=(len(accessor),)), None
    last_eigenvector[ignore_positions] = 0.0  # refers to the vertex without follow-up vertices.

    for _ in range(iteration):
        eigenvector = zeros(shape=(len(accessor),))
        for positions in accessor.T:
            available = where(positions >= 0)
            eigenvector[available] += last_eigenvector[positions[available]]

        eigenvalue = dot(last_eigenvector, eigenvector) / dot(last_eigenvector, last_eigenvector)

        if eigenvalue > 0:
            last_eigenvalue, last_eigenvector = eigenvalue, eigenvector / eigenvalue
        else:
            raise ValueError("The capacity cannot be approximated.")

    # noinspection PyTypeChecker
    return log2(last_eigenvalue)


def i2m(integers: ndarray,
        length: int,
        base_size: int = 4) \
        -> ndarray:
    """
    Convert integers to digit matrix.

    :param integers: integers.
    :type integers: numpy.ndarray

    :param length: length of matrix.
    :type length: int

    :param base_size: size of base, e.g. the size of nucleotide is 4.
    :type base_size: int

    :return: digit matrix.
    :rtype: numpy.ndarray
    """
    if base_size ** length <= max(integers):
        raise ValueError("The maximum integer is larger than the upper bound of digit matrix!")

    matrix, values = zeros(shape=(len(integers), length), dtype=int), copy(integers)

    for location in arange(length)[::-1]:
        matrix[:, location] = values % base_size
        values //= base_size

    return matrix


def m2i(matrix: ndarray,
        base_size: int = 4) \
        -> ndarray:
    """
    Convert digit matrix to integers.

    :param matrix: digit matrix.
    :type matrix: numpy.ndarray

    :param base_size: size of base, e.g. the size of nucleotide is 4.
    :type base_size: int

    :return: integers.
    :rtype: numpy.ndarray
    """
    if len(matrix.shape) == 2:
        return sum(matrix * (base_size ** arange(matrix.shape[1])[::-1]), axis=1)
    elif len(matrix.shape) == 1:
        # noinspection PyTypeChecker
        return sum(matrix * (base_size ** arange(len(matrix))[::-1]))
    else:
        raise ValueError("No such matrix dimension, only support 1 or 2.")


def m2s(matrix: ndarray) \
        -> list:
    """
    Convert the integer matrix to DNA sequences.

    :param matrix: integer matrix, containing 5 digits: 0, 1, 2, 3 and 4 (none value).
    :type matrix: numpy.ndarray

    :return: one or more DNA sequence(s).
    :rtype: list
    """
    new_matrix = copy(matrix)
    for value, nucleotide_value in zip([0, 1, 2, 3, 4], [ord("A"), ord("C"), ord("G"), ord("T"), ord("-")]):
        new_matrix[new_matrix == value] = nucleotide_value

    if len(new_matrix.shape) == 2 and new_matrix.shape[0] > 1:
        sequences = new_matrix.astype(uint8).view("S" + str(new_matrix.shape[1])).squeeze().astype("str").tolist()
        return [sequence.replace("-", "") for sequence in sequences]

    elif len(new_matrix.shape) == 2 and new_matrix.shape[0] == 1:
        sequence = str(new_matrix.astype(uint8).view("S" + str(new_matrix.shape[1])).squeeze().astype("str"))
        return [sequence.replace("-", "")]

    elif len(new_matrix.shape) == 1:
        sequence = str(new_matrix.astype(uint8).view("S" + str(len(new_matrix))).squeeze().astype("str"))
        return [sequence.replace("-", "")]

    else:
        raise ValueError("No such matrix dimension, only support 1 or 2.")


def s2m(dna_sequences: list,
        assume_equal: bool = False) \
        -> ndarray:
    """
    Convert DNA sequences to the integer matrix.

    :param dna_sequences: inputted DNA strings.
    :type dna_sequences: list

    :param assume_equal: the length of inputted DNA strings are both assumed to be same, default setting is False.
    :type assume_equal: bool

    :return: integer matrix.
    :rtype: numpy.ndarray
    """
    if len(dna_sequences) > 1:
        if not assume_equal:
            lengths = array([len(dna_sequence) for dna_sequence in dna_sequences])
            additions = max(lengths) - lengths
            dna_sequences = [dna_sequence + "-" * addition for dna_sequence, addition in zip(dna_sequences, additions)]

        try:
            matrix = array(dna_sequences, dtype="c").view(uint8)
        except ValueError:
            raise ValueError("Inputted DNA sequences are not of equal length, please set \"assume_equal\" to False!")

        for value, nucleotide_value in zip([0, 1, 2, 3, 4], [ord("A"), ord("C"), ord("G"), ord("T"), ord("-")]):
            matrix[matrix == nucleotide_value] = value
    else:
        dna_sequence = dna_sequences[0].replace("-", "")  # ignore "-"

        matrix = array(dna_sequence, dtype="c").view(uint8)
        for value, nucleotide_value in zip([0, 1, 2, 3], [ord("A"), ord("C"), ord("G"), ord("T")]):
            matrix[matrix == nucleotide_value] = value

    return matrix


def detect_duplicates(sequence: str,
                      lower_bound: int) \
        -> dict:
    """
    Detect duplicates in the sequence.

    :param sequence: DNA sequence.
    :type sequence: str

    :param lower_bound: lower detection bound.
    :type lower_bound: int

    :return: detection records.
    :rtype: dict
    """
    records, values = {}, s2m([sequence])
    for bound in range(lower_bound, len(sequence)):
        group = zeros(shape=(len(sequence) - bound, bound), dtype=int)
        for location in range(bound, len(sequence)):
            group[location - bound] = values[location - bound: location]

        similarities, collection = hamming(group=group), []
        for case_1, case_2 in zip(*where(similarities == 0)):
            if case_1 != case_2:
                saved_index = -1
                for index, indices in enumerate(collection):
                    if case_1 in indices or case_2 in indices:
                        saved_index = index
                        break
                if saved_index >= 0:
                    collection[saved_index].add(case_1)
                    collection[saved_index].add(case_2)
                else:
                    collection.append({case_1, case_2})

        if len(collection) > 0:
            records[bound] = {}
            for start_indices in collection:
                start_indices = sorted(list(start_indices))
                records[bound][sequence[start_indices[0]: start_indices[0] + bound]] = start_indices
        else:
            break

    return records


def detect_palindromes(sequence: str,
                       lower_bound: int) \
        -> dict:
    """
    Detect palindromes in the sequence.

    :param sequence: DNA sequence.
    :type sequence: str

    :param lower_bound: lower detection bound.
    :type lower_bound: int

    :return: detection records.
    :rtype: dict
    """
    records, forward_values = {}, s2m([sequence])
    reverse_values = 3 - forward_values[::-1]
    for bound in range(lower_bound, len(sequence) + 1):
        forward_group = zeros(shape=(len(sequence) - bound, bound), dtype=int)
        for location in range(bound, len(sequence)):
            forward_group[location - bound] = forward_values[location - bound: location]

        reverse_group = zeros(shape=(len(sequence) - bound, bound), dtype=int)
        for location in range(bound, len(sequence)):
            reverse_group[location - bound] = reverse_values[location - bound: location]

        similarities, collection = hamming(group=forward_group, another_group=reverse_group), []
        for case_1, case_2 in zip(*where(similarities == 0)):
            case_2 = len(sequence) - case_2 - bound
            saved_index = -1
            for index, indices in enumerate(collection):
                if case_1 in indices or case_2 in indices:
                    saved_index = index
                    break
            if saved_index >= 0:
                collection[saved_index].add(case_1)
                collection[saved_index].add(case_2)
            else:
                collection.append({case_1, case_2})

        if len(collection) > 0:
            data = {}
            for start_indices in collection:
                for start_index in start_indices:
                    if sequence[start_index: start_index + bound] in data:
                        data[sequence[start_index: start_index + bound]].add(start_index)
                    else:
                        data[sequence[start_index: start_index + bound]] = {start_index}
            data = {key: "(" + str(sorted(list(values)))[1:-1].replace(", ", "/") + ")" for key, values in data.items()}

            pairs = []
            for segment in data.keys():
                temp = segment.translate(str.maketrans("ACGT", "TGCA"))[::-1]
                index = -1
                for pair_index, pair in enumerate(pairs):
                    if temp in pair:
                        index = pair_index
                        break
                if index >= 0:
                    pairs[index].append(segment)
                else:
                    pairs.append([segment])

            records[bound] = []
            for pair in pairs:
                if len(pair) == 2:
                    records[bound].append(data[pair[0]] + pair[0] + " - " + data[pair[1]] + pair[1])
                else:  # self palindrome
                    records[bound].append(data[pair[0]] + pair[0])

        else:
            break

    return records


def calculus_addition(number: str,
                      base: str) \
        -> str:
    """
    Do huge number addition calculus with a small base value, as number + base.

    :param number: huge number.
    :type number: str

    :param base: base value:
    :type base: str

    :return: number + base.
    :rtype: str

    .. note::
        The integer of parameter "base" must less be than 10 in decimal system.
    """
    # noinspection PyTypeChecker
    number, base = list(number), list(base.zfill(len(number)))

    result = [0 for _ in range(len(number) + 1)]
    for index in range(len(number) - 1, -1, -1):
        sum_value = int(number[index]) + int(base[index]) + int(result[index + 1])
        if sum_value < 10:
            result[index + 1] = sum_value
        else:
            flag = 0
            while sum_value > 0:
                result[index + 1 - flag] = sum_value % 10
                sum_value //= 10
                flag += 1

    result = "".join(list(map(str, result)))

    return result if result[0] != "0" else result[1:]


def calculus_subtraction(number: str,
                         base: str) \
        -> str:
    """
    Do huge number subtraction calculus with a small base value, as number - base.

    :param number: huge number.
    :type number: str

    :param base: base value:
    :type base: str

    :return: number - base.
    :rtype: str

    .. note::
        The integer of parameter "base" must less be than 10 in decimal system.
    """
    # noinspection PyTypeChecker
    number, base = [int(item) for item in number], [int(item) for item in base]

    residue = ""
    for index in range(len(base)):
        flag_a, flag_b = len(number) - 1 - index, len(base) - 1 - index
        if int(number[flag_a]) >= int(base[flag_b]):
            residue = str(int(number[flag_a]) - int(base[flag_b])) + residue
        else:
            residue = str(10 + int(number[flag_a]) - int(base[flag_b])) + residue
            while number[flag_a - 1] == 0:
                # noinspection PyUnresolvedReferences
                number[flag_a - 1] = 9
                flag_a -= 1
            number[flag_a - 1] -= 1

        for flag in range(len(number) - 1 - index - 1, -1, -1):
            residue = str(number[flag]) + residue

    for index in range(len(residue)):
        if residue[index] != "0":
            return residue[index:]

    return "0"


def calculus_multiplication(number: str,
                            base: str) \
        -> str:
    """
    Do huge number multiplication calculus with a small base value, as number * base.

    :param number: huge number.
    :type number: str

    :param base: base value:
    :type base: str

    :return: number * base.
    :rtype: str

    .. note::
        The integer of parameter "base" must less be than 10 in decimal system.
    """
    if base == "0":
        return "0"

    if base == "1":
        return number

    # noinspection PyTypeChecker
    number = [int(item) for item in number]

    remainder = 0
    for index in range(len(number))[::-1]:
        # noinspection PyTypeChecker
        current = number[index] * int(base) + remainder
        if current >= 10:
            # noinspection PyUnresolvedReferences
            number[index] = current % 10
            # noinspection PyUnresolvedReferences
            remainder = current // 10
        else:
            # noinspection PyUnresolvedReferences
            number[index] = current
            remainder = 0

    while remainder > 0:
        # noinspection PyUnresolvedReferences
        number.insert(0, remainder % 10)
        remainder //= 10

    result = "".join(list(map(str, number)))

    return result


def calculus_division(number: str,
                      base: str) \
        -> Tuple[str, str]:
    """
    Do huge number division calculus with a small base value, as number / base and number % base.

    :param number: huge number.
    :type number: str

    :param base: base value.
    :type base: str

    :return: number // base and number % base.
    :rtype: str, str

    .. note::
        The integer of parameter "base" must less be than 10 in decimal system.

        When the divisor (base) is "0", the program will return "0", "0".
    """
    if base == "0":
        return "0", "0"

    if base == "1":
        return number, "0"

    if len(number) == 1 and number[0] < base:
        return "0", number[0]

    # noinspection PyTypeChecker
    number, new_number, remainder = [int(item) for item in number], [], 0
    for index, quotient in enumerate(number):
        # noinspection PyTypeChecker
        current = quotient + remainder * 10
        if current >= int(base):
            # noinspection PyUnresolvedReferences
            new_number.append(current // int(base))
            remainder = current - new_number[-1] * int(base)
        else:
            new_number.append(0)
            remainder = current

    quotient = "".join(list(map(str, new_number)))

    for index in range(len(quotient)):
        if quotient[index] != "0":
            return quotient[index:], str(remainder)

    return "0", str(remainder)


def bits_to_number(bit_array: ndarray,
                   verbose=False) \
        -> str:
    """
    Transform a bit array to the equivalent decimal number.

    :param bit_array: bit array.
    :type bit_array: numpy.ndarray

    :param verbose: need to print log.
    :type verbose: bool

    :return: equivalent decimal number (may huge) of the inputted bit array.
    :rtype: str
    """
    monitor, decimal_number = Monitor(), "0"

    for index, a_bit in enumerate(bit_array):
        # multiply by 2
        decimal_number = calculus_multiplication(number=decimal_number, base="2")
        # add current bit
        decimal_number = calculus_addition(number=decimal_number, base=str(a_bit))

        if verbose:
            monitor(index + 1, len(bit_array))

    return decimal_number


def number_to_bits(decimal_number: str,
                   bit_length: int) \
        -> ndarray:
    """
    Transform a decimal number to the equivalent bit array with specific length.

    :param decimal_number: decimal number (may huge) of the bit array.
    :type decimal_number: str

    :param bit_length: default length of the bit array.
    :type bit_length: int

    :return: bit array.
    :rtype: list
    """
    one_array = []

    while decimal_number != "0":  # decimal number > 0
        decimal_number, remainder = calculus_division(number=decimal_number, base="2")
        one_array.insert(0, int(remainder))

    if len(one_array) == bit_length:
        return array(one_array)
    elif len(one_array) < bit_length:
        return array([0] * (bit_length - len(one_array)) + one_array, dtype=int)
    else:
        return array(one_array[:bit_length])


def dna_to_number(dna_sequence: str) \
        -> str:
    """
    Transform a DNA string to the equivalent decimal number.

    :param dna_sequence: required DNA string.
    :type dna_sequence: str

    :return: equivalent decimal number (may huge) of the inputted DNA string.
    :rtype: str
    """
    nucleotides = "ACGT"
    nucleotide_values = list(map(nucleotides.index, dna_sequence))
    decimal_number = "0"
    for nucleotide_value in nucleotide_values:
        # multiply by length of usage of nucleotides.
        decimal_number = calculus_multiplication(number=decimal_number, base=str(len(nucleotides)))
        # add current nucleotide value.
        decimal_number = calculus_addition(number=decimal_number, base=str(nucleotide_value))

    return decimal_number


def number_to_dna(decimal_number: str,
                  dna_length: int) \
        -> str:
    """
    Transform a decimal number to the equivalent DNA string with specific length.

    :param decimal_number: decimal number (may huge) of the DNA string.
    :type decimal_number: str or int

    :param dna_length: default length of the DNA string.
    :type dna_length: int

    :return: equivalent DNA string of the decimal number.
    :rtype: str
    """
    nucleotides = "ACGT"
    one_array = []

    while decimal_number != "0":  # decimal number > 0
        decimal_number, remainder = calculus_division(number=decimal_number, base=str(len(nucleotides)))
        one_array.insert(0, nucleotides[int(remainder)])

    one_array = "".join(one_array)

    return nucleotides[0] * (dna_length - len(one_array)) + one_array
