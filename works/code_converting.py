from numpy import ndarray, array, arange, zeros, ones, concatenate, where, longlong
from numpy import linalg, real, max, abs, sum, log, floor, sqrt, cumsum
from random import random, seed, sample
from sys import setrecursionlimit
from typing import Tuple, Generator, Optional

from sbw import Monitor, m2s, bits_to_number


# noinspection PyTypeChecker
class DNAFountain:

    def __init__(self,
                 failure: float = 0.001,                         # default delta in DNA Fountain.
                 redundancy: float = 0.025,                      # default cdist in DNA Fountain.
                 recursion_depth: int = 10000000):
        """
        Initialise the DNA fountain algorithm.

        :param failure: decoding failure probability (known as delta).
        :type failure: float

        :param redundancy: redundant control parameter (known as cdist).
        :type redundancy: float

        :param recursion_depth: recursive call depth limit (default is 1000, not applicable to the current scenario).
        :type recursion_depth: int
        """
        self.__failure, self.__redundancy = failure, redundancy  # delta and cdist in DNA Fountain.
        self.__cdf, self.__success, self.__droplets = None, None, []
        setrecursionlimit(recursion_depth)

    def evaluate(self,
                 binary_messages: ndarray,
                 window_size: int,
                 homopolymer: int,
                 gc_bias: float,
                 false_threshold: Optional[int] = None,
                 verbose: bool = False) \
            -> Tuple[float, int, int, list]:
        """
        Evaluate the DNA Fountain algorithm based on the given constraints.

        :param binary_messages: binary messages.
        :type binary_messages: ndarray

        :param window_size: window size.
        :type window_size: int

        :param homopolymer: maximum single-nucleotide length in the produced DNA sequences.
        :type homopolymer: int

        :param gc_bias: degree of deviation of regionalised GC content from 50%.
        :type gc_bias: float

        :param false_threshold: upper limit of continuous inability to obtain valid DNA sequences.
        :type false_threshold: int or None

        :param verbose: verbosity mode.
        :type verbose: bool

        :return: code rate, screened count, sequence number, and used degrees.
        :rtype: float, int, int, list
        """
        if len(binary_messages[0]) % 2 != 0:
            raise ValueError("Binary messages must have even length!")

        self.__cdf = self.__rsd(number=len(binary_messages), failure=self.__failure, redundancy=self.__redundancy)
        self.__success, droplets = zeros(shape=(len(binary_messages),), dtype=int), []

        undesired_motifs, monitor = [nucleotide * (homopolymer + 1) for nucleotide in ["A", "C", "G", "T"]], Monitor()
        generator, used_seeds, sequence_number, screened_count, false_times, degrees = self.__lfsr(), set(), 0, 0, 0, []
        while True:
            used_seed = next(generator)
            if used_seed in used_seeds:
                continue

            used_seeds.add(used_seed)

            selected_indices = self.__combine(number=len(binary_messages), used_seed=used_seed)
            combined_message = sum(binary_messages[selected_indices], axis=0) % 2
            sequence = m2s(matrix=combined_message[0::2] * 2 + combined_message[1::2])[0]

            flag = True
            if homopolymer < window_size:
                for undesired_motif in undesired_motifs:
                    if undesired_motif in sequence:
                        flag = False
                        break
            if flag:
                for location in range(len(sequence) - window_size + 1):
                    k_mer = sequence[location: location + window_size]
                    gc_content = (k_mer.count("G") + k_mer.count("C")) / float(window_size)
                    if abs(gc_content - 0.5) > gc_bias:
                        flag = False
                        break

            if not flag:
                screened_count += 1

            if verbose:
                monitor(sum(self.__success), len(binary_messages))

            if flag:
                false_times, sequence_number, indices = 0, sequence_number + 1, set(selected_indices)
                degrees.append(len(indices))
                self.__droplets.append(indices)
                self.__divide()
                if sum(self.__success, dtype=int) == len(binary_messages):
                    if verbose:
                        monitor(sum(self.__success), len(binary_messages))
                    code_rate = float(len(binary_messages)) / float(sequence_number) * 2.0
                    return code_rate, screened_count, sequence_number, degrees
                if sum(self.__success, dtype=int) == 0 and sequence_number == false_threshold:
                    if verbose:
                        print()
                    return -1.0, screened_count, sequence_number, degrees
            else:
                false_times, screened_count = false_times + 1, screened_count + 1
                if false_threshold is not None and false_times == false_threshold:
                    if verbose:
                        print()
                    return -1.0, screened_count, sequence_number, degrees

    def __combine(self,
                  number: int,
                  used_seed: int) \
            -> list:
        """
        Obtain the combined indices in the binary messages.

        :param number: number of binary messages.
        :type number: int

        :param used_seed: used random seed.
        :type used_seed: int

        :return: list of indices for the XOR operation.
        :rtype: list
        """
        seed(used_seed)
        probability = random()
        for degree, value in enumerate(self.__cdf):
            if value > probability:
                return sample(range(number), degree + 1)
        return sample(range(number), len(self.__cdf))

    def __divide(self):
        """
        Divide the indices in the obtained droplets.
        """
        if len(self.__droplets) > 0:
            for droplet in self.__droplets:
                unsolved_blocks = {index for index in droplet if self.__success[index] == 0}

                if len(unsolved_blocks) == 1:
                    message_index = unsolved_blocks.pop()
                    self.__success[message_index] = 1

                    updated_droplets = []
                    for previous_one in self.__droplets:
                        updated_droplet = previous_one - {message_index}
                        if updated_droplet:
                            updated_droplets.append(updated_droplet)

                    self.__droplets = updated_droplets
                    self.__divide()

    @staticmethod
    def __rsd(number: int,
              failure: float,
              redundancy: float) \
            -> ndarray:
        """
        Generate the Robust Soliton Distribution (RSD) based on the given number and

        :param number: number of binary messages.
        :type number: int

        :param failure: decoding failure probability (known as delta).
        :type failure: float

        :param redundancy: redundant control parameter (known as c-dist).
        :type redundancy: float

        :return: Robust Soliton Distribution array.
        :rtype: ndarray
        """
        value = redundancy * log(number / failure) * sqrt(number)

        # concatenate three arrays to form the tau array
        pivot = floor(number / value).astype(int)
        array_1 = value / number * 1 / arange(1, pivot)
        array_2 = array([value / number * log(value / failure)])
        array_3 = zeros(shape=(number - pivot,))
        tau = concatenate((array_1, array_2, array_3))

        # generate the rho array
        array_4 = array([1.0 / number])
        array_5 = 1.0 / (arange(2, number + 1, dtype=float) * arange(1, number, dtype=float))
        rho = concatenate((array_4, array_5))

        # calculate the CDF (Cumulative Distribution Function)
        # noinspection PyUnresolvedReferences
        return cumsum((rho + tau) / (sum(rho) + sum(tau)))

    @staticmethod
    def __lfsr(state: int = 0b001010101,
               mask: int = 0b100000000000000000000000011000101) \
            -> Generator[int, None, None]:
        """
        Create the Galois linear feedback shift register (LFSR) with given parameters.

        :param state: hard coded state.
        :type state: int

        :param mask: hard coded polynomial (mask).
        :type mask: int

        :return: LFSR generator.
        :rtype: Generator[int, None, None]
        """
        result, nbits = state, mask.bit_length() - 1
        while True:
            result <<= 1
            xor = result >> nbits

            if xor != 0:
                result ^= mask

            yield result


class HEDGES:

    def __init__(self,
                 pattern: list,
                 salt_number: int = 46,
                 previous_number: int = 8,
                 low_order_number: int = 10):
        """
        Initialise the HEDGES algorithm.

        :param pattern: encode pattern in HEDGES algorithm.
        :type pattern: list

        :param salt_number: number of salt bits.
        :type salt_number: int

        :param previous_number: number of previous bits.
        :type previous_number: int

        :param low_order_number: number of low-order bits.
        :type low_order_number: int
        """
        self.pattern = pattern
        self.salt_number, self.previous_number, self.low_order_number = salt_number, previous_number, low_order_number

    def evaluate(self,
                 binary_messages: ndarray,
                 window_size: int,
                 homopolymer: int,
                 gc_bias: float,
                 false_threshold: Optional[int] = None,
                 verbose: bool = False) \
            -> Tuple[float, int]:
        """
        Evaluate the DNA Fountain algorithm based on the given constraints.

        :param binary_messages: binary messages.
        :type binary_messages: ndarray

        :param window_size: window size.
        :type window_size: int

        :param homopolymer: maximum single-nucleotide length in the produced DNA sequences.
        :type homopolymer: int

        :param gc_bias: degree of deviation of regionalised GC content from 50%.
        :type gc_bias: float

        :param false_threshold: upper limit of continuous inability to obtain valid DNA sequences.
        :type false_threshold: int or None

        :param verbose: verbosity mode.
        :type verbose: bool

        :return: code rate and false count.
        :rtype: float, int
        """
        message_number, nucleotide_number, false_count, monitor = 0, 0, 0, Monitor()
        for strand_index, binary_message in enumerate(binary_messages):
            # noinspection PyTypeChecker
            dna_sequence = self.__encode(binary_message=binary_message, strand_index=strand_index,
                                         window_size=window_size, homopolymer=homopolymer, gc_bias=gc_bias,
                                         false_threshold=false_threshold)
            if dna_sequence is not None:
                nucleotide_number += len(dna_sequence)
                message_number += len(binary_message)
            else:
                false_count += 1
            if verbose:
                monitor(strand_index + 1, len(binary_messages))

        if nucleotide_number > 0:
            return message_number / nucleotide_number, false_count
        else:
            return -1, false_count

    def __encode(self,
                 binary_message: ndarray,
                 strand_index: int,
                 window_size: int,
                 homopolymer: int,
                 gc_bias: float,
                 false_threshold: Optional[int] = None) \
            -> Optional[str]:
        """
        Encode the given binary message into a DNA sequence using HEDGES algorithm.

        :param binary_message: binary messages.
        :type binary_message: ndarray

        :param strand_index: strand index of this binary message.
        :type strand_index: int

        :param window_size: window size.
        :type window_size: int

        :param homopolymer: maximum single-nucleotide length in the produced DNA sequences.
        :type homopolymer: int

        :param gc_bias: degree of deviation of regionalised GC content from 50%.
        :type gc_bias: float

        :param false_threshold: upper limit of continuous inability to obtain valid DNA sequences.
        :type false_threshold: int or None

        :return: DNA sequence if encoded successfully.
        :rtype: str or None
        """
        false_count = 0

        while True:
            dna_sequence, available_nucleotides, bit_location, pattern_flag = "", ["A", "C", "G", "T"], 0, 0
            salt_value = strand_index % (2 ** self.salt_number)  # s bits of salt (strand ID).
            try:
                while bit_location < len(binary_message):
                    bit_number = self.pattern[pattern_flag]
                    # low-order q bits of the bit position index i.
                    location_value = bit_location % (2 ** self.low_order_number)

                    if bit_location - self.previous_number >= 0:
                        previous_info = binary_message[bit_location - self.previous_number: bit_location]
                        previous_value = int(bits_to_number(previous_info))
                    else:
                        previous_value = 0

                    hash_value = self.__hash_function(location_value | previous_value | salt_value)
                    if len(available_nucleotides) >= 2 ** bit_number:
                        message_bit = binary_message[bit_location: bit_location + bit_number]
                        bit_value = int(bits_to_number(message_bit)) if len(message_bit) > 0 else 0
                        nucleotide = available_nucleotides[(hash_value + bit_value) % len(available_nucleotides)]
                        bit_location += bit_number
                        pattern_flag = (pattern_flag + 1) % len(self.pattern)
                    else:
                        raise ValueError("DNA sequence (index = " + str(strand_index) + ") "
                                         + "cannot be encoded because of the established constraints!")

                    dna_sequence += nucleotide

                    available_nucleotides = []
                    for potential_nucleotide in ["A", "C", "G", "T"]:
                        if self.__screen(segment=dna_sequence + potential_nucleotide, window_size=window_size,
                                         homopolymer=homopolymer, gc_bias=gc_bias):
                            available_nucleotides.append(potential_nucleotide)

                return dna_sequence

            except ValueError:
                pass

            false_count += 1
            if false_threshold is not None and false_count == false_threshold:
                return None

    @staticmethod
    def __screen(segment: str,
                 window_size: int,
                 homopolymer: int,
                 gc_bias: float) \
            -> bool:
        """
        bio filtering function for the produced DNA sequence.

        :param segment: DNA sequence.
        :type segment: str

        :param window_size: window size of location constraints.
        :type window_size: int

        :param homopolymer: maximum single-nucleotide length in the produced DNA sequences.
        :type homopolymer: int

        :param gc_bias: bias of local GC content.
        :type gc_bias: float

        :return: screening flag (True means meeting the given constraints).
        :rtype: bool
        """
        for motif in ["A" * (homopolymer + 1), "C" * (homopolymer + 1),
                      "G" * (homopolymer + 1), "T" * (homopolymer + 1)]:
            if motif in segment:
                return False

        minimum_count, maximum_count = (0.5 - gc_bias) * window_size, (0.5 + gc_bias) * window_size
        if len(segment) >= window_size:
            for part_index in range(len(segment) - window_size + 1):
                part = segment[part_index: part_index + window_size]
                gc_count = part.count("G") + part.count("C")
                if (gc_count < minimum_count) or (gc_count > maximum_count):
                    return False
        else:
            gc_count = segment.count("G") + segment.count("C")
            if gc_count > maximum_count:
                return False
            at_count = segment.count("A") + segment.count("T")
            if at_count > maximum_count:
                return False

        return True

    @staticmethod
    def __hash_function(source_value: int) \
            -> int:
        """
        Calculate the hash value from the source value.

        :param source_value: source value.
        :type source_value: int

        :return: hash value.
        :rtype: int
        """
        target_value = array(source_value, dtype=longlong)

        target_value *= array(3935559000370003845, dtype=longlong)
        target_value += array(2691343689449507681, dtype=longlong)
        target_value ^= target_value >> 21
        target_value ^= target_value << 37
        target_value ^= target_value >> 4

        target_value *= array(4768777513237032717, dtype=longlong)
        target_value ^= target_value << 20
        target_value ^= target_value >> 41
        target_value ^= target_value << 5

        # noinspection PyTypeChecker
        return target_value


def accessor_to_matrix(accessor: ndarray,
                       verbose: bool = False) \
        -> ndarray:
    """
    Convert the accessor (compressed matrix) to its equivalent adjacency matrix.

    :param accessor: accessor (compressed matrix).
    :type: numpy.ndarray

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: adjacency matrix of the uncompressed graph.
    :rtype: numpy.ndarray
    """
    matrix, monitor = zeros(shape=(len(accessor), len(accessor)), dtype=int), Monitor()
    for vertex_index, vertex in enumerate(accessor):
        # noinspection PyTypeChecker
        follow_ups = vertex[vertex >= 0]
        matrix[vertex_index][follow_ups] = 1

        if verbose:
            monitor(vertex_index + 1, len(accessor))

    return matrix


def matrix_to_accessor(matrix: ndarray,
                       verbose: bool = False) \
        -> ndarray:
    """
    Convert the adjacency matrix to its equivalent accessor (compressed matrix).

    :param matrix: adjacency matrix.
    :type matrix: numpy.ndarray

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: accessor (compressed matrix).
    :rtype: numpy.ndarray
    """
    nucleotides = "ACGT"

    accessor, monitor = -ones(shape=(len(matrix), len(nucleotides)), dtype=int), Monitor()
    observed_length = int(log(len(accessor)) / log(len(nucleotides)))

    for vertex_index, vertex in enumerate(matrix):
        next_indices = where(vertex == 1)[0].tolist()
        reference_latters = []
        for latter_value in range(len(nucleotides)):
            latter = int((vertex_index * len(nucleotides) + latter_value) % (len(nucleotides) ** observed_length))
            reference_latters.append(latter)
        if list(set(next_indices) | set(reference_latters)) != reference_latters:
            raise ValueError("Wrong format in the adjacency matrix, "
                             + "which cannot be converted to equivalent compressed accessor!")
        saved_information = [index if index in next_indices else -1 for index in reference_latters]
        accessor[vertex_index] = saved_information

        if verbose:
            monitor(vertex_index + 1, len(matrix))

    return accessor


def calculate_largest_eigenvalue(accessor: ndarray):
    """
    Calculate the largest eigenvalue of a compressed matrix.

    :param accessor: accessor (compressed matrix).
    :type: numpy.ndarray

    :return: largest eigenvalue.
    :rtype: float
    """
    matrix = accessor_to_matrix(accessor)

    # QR transformation (Q is an orthogonal matrix and R is a right triangular matrix).
    return real(max(linalg.eig(matrix)[0]))


def get_complete_accessor(observed_length: int,
                          verbose: bool = False) \
        -> ndarray:
    """
    Get a complete accessor with the required observed length.

    :param observed_length: length of the DNA sequence in a vertex.
    :type observed_length: int

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: complete accessor.
    :rtype: numpy.ndarray

    .. note::
        The size of accessor is 4 ^ l * 4 and that of corresponding adjacency matrix is 4 ^ l * 4 ^ l.
    """
    accessor, nucleotides, monitor = -ones(shape=(int(4 ** observed_length), 4), dtype=int), "ACGT", Monitor()

    for vertex_index in range(int(4 ** observed_length)):
        latters = []
        for latter_value in range(len(nucleotides)):
            latter = int((vertex_index * len(nucleotides) + latter_value) % (len(nucleotides) ** observed_length))
            latters.append(latter)
        for position, latter_vertex_index in enumerate(latters):
            accessor[vertex_index][position] = latter_vertex_index

        if verbose:
            monitor(vertex_index + 1, int(4 ** observed_length))

    return accessor
