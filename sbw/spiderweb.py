from collections import Counter
from copy import deepcopy
from itertools import product
from multiprocessing import Process
from networkx import Graph, connected_components
from numpy import ndarray, arange, array, zeros, zeros_like, ones, random, expand_dims, hstack, vstack, in1d, setdiff1d
from numpy import count_nonzero, nonzero, log, abs, sum, min, mean, max, argmax, argwhere, where, any, all, copy, uint8
from os import name, path, fdopen, remove, system
from prettytable import PrettyTable
from psutil import process_iter
from socket import socket, AF_INET, SOCK_STREAM
from subprocess import Popen, check_output, TimeoutExpired
from sys import getsizeof
from tempfile import mkstemp
from time import sleep
from typing import Tuple, Union, Optional

from sbw.underprop import Monitor, m2s, s2m, m2i, i2m
from sbw.underprop import bits_to_number, number_to_bits, calculus_division, calculus_multiplication, calculus_addition
from sbw.underprop import get_latters, get_formers, calculate_practical_density, hamming, levenshtein


def basic_encode(binary_message: ndarray,
                 accessor: ndarray,
                 virtuality: int,
                 verbose: bool = False) \
        -> str:
    """
    Encode a bit array by the specific accessor.

    :param binary_message: binary message.
    :type binary_message: numpy.ndarray

    :param accessor: accessor of the coding algorithm.
    :type accessor: numpy.ndarray

    :param virtuality: virtual vertex to start encoding.
    :type virtuality: int

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: DNA sequence encoded by this graph (and VT check sequence if required).
    :rtype: str
    """
    quotient, dna_sequence, nucleotides = bits_to_number(binary_message, verbose=verbose), "", "ACGT"
    total_state, monitor = len(quotient), Monitor()

    vertex = where(sum(((accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0][virtuality]

    while quotient != "0":
        used_indices = where(accessor[vertex] >= 0)[0]

        if len(used_indices) > 1:  # current vertex contains information.
            quotient, remainder = calculus_division(number=quotient, base=str(len(used_indices)))
            remainder = int(remainder)
            value = used_indices[remainder]

        elif len(used_indices) == 1:  # current vertex does not contain information.
            value = used_indices[0]

        else:  # current vertex is wrong.
            raise ValueError("Current vertex doesn't have an out-degree, "
                             + "the accessor or the start vertex is wrong!")

        nucleotide, vertex = nucleotides[value], accessor[vertex, value]

        dna_sequence += nucleotide

        if verbose:
            if quotient != "0":
                monitor(total_state - len(quotient), total_state)
            else:
                monitor(total_state, total_state)

    return dna_sequence


def basic_decode(dna_sequence: str,
                 accessor: ndarray,
                 virtuality: int,
                 bit_length: int,
                 verbose: bool = False) \
        -> ndarray:
    """
    Decode a DNA sequence by the specific accessor.

    :param dna_sequence: DNA sequence encoded by this graph.
    :type dna_sequence: str

    :param accessor: accessor of the coding algorithm.
    :type accessor: numpy.ndarray

    :param virtuality: virtual vertex to start encoding.
    :type virtuality: int

    :param bit_length: length of the bit array.
    :type bit_length: int

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: binary message decoded by this graph.
    :rtype: numpy.ndarray
    """
    nucleotides, monitor, quotient, saved_values = "ACGT", Monitor(), "0", []

    vertex = where(sum(((accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0][virtuality]

    for location, nucleotide in enumerate(dna_sequence):
        used_indices = where(accessor[vertex] >= 0)[0]

        if len(used_indices) > 1:  # current vertex contains information.
            used_nucleotides = [nucleotides[used_index] for used_index in used_indices]

            if nucleotide in used_nucleotides:  # check whether the DNA sequence is right currently.
                remainder = used_nucleotides.index(nucleotide)
            else:
                raise ValueError("At least one error is found in this DNA sequence!")

            saved_values.append((len(used_indices), remainder))
            vertex = accessor[vertex, nucleotides.index(nucleotide)]

        elif len(used_indices) == 1:  # current vertex does not contain information.
            used_nucleotide = nucleotides[used_indices[0]]
            if nucleotide == used_nucleotide:
                vertex = accessor[vertex, nucleotides.index(nucleotide)]
            else:
                raise ValueError("At least one error is found in this DNA sequence!")

        else:  # current vertex is wrong.
            raise ValueError("Current vertex doesn't have an out-degree, "
                             + "the accessor, the start vertex, or DNA sequence is wrong!")

        if verbose:
            monitor(location + 1, len(dna_sequence))

    for location, (out_degree, number) in enumerate(saved_values[::-1]):
        quotient = calculus_multiplication(number=quotient, base=str(out_degree))
        quotient = calculus_addition(number=quotient, base=str(number))

    return number_to_bits(decimal_number=quotient, bit_length=bit_length)


# noinspection PyTypeChecker,PyUnresolvedReferences
class SpiderWeb:

    def __init__(self,
                 accessor: ndarray,
                 check_3: bool = True):
        """
        Initialise the SPIDER-WEB architecture.

        :param accessor: the established coding digraph.
        :type accessor: numpy.ndarray

        :param check_3: check out-degree 3 in the digraph.
        :type check_3: bool
        """
        self.__accessor = accessor
        self.__vertices = where(sum(((self.__accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0]
        self.__windowed = int(log(len(self.__accessor)) / log(4))

        out_statistics = count_out_degrees(accessor=accessor)
        if check_3 and out_statistics[3] > 0 and out_statistics[3] != sum(out_statistics):
            raise ValueError("If a graph contains nodes with out-degree of 3 and others (i.e., 1, 2, and 4), "
                             "it will cause encoding ambiguity")
        self.__all_3 = out_statistics[3] == sum(out_statistics)

        self.__bases = ("A", "C", "G", "T")
        self.__monitor = Monitor()

    def encode(self,
               message: ndarray,
               virtuality: int,
               need_virtuality: bool = False,
               length_fixed: Optional[int] = None,
               duplicatable: bool = False,
               palindromable: bool = False,
               seed: Optional[int] = None,
               retry: Optional[int] = None,
               verbose: bool = False) \
            -> str:
        """
        Encode binary messages to DNA sequences based on the established coding digraph.

        :param message: binary message.
        :type message: numpy.ndarray

        :param virtuality: virtual vertex to start encoding.
        :type virtuality: int

        :param need_virtuality: need to include the virtual vertex.
        :type need_virtuality: bool

        :param duplicatable: non-duplicate mode for DNA sequences if exists.
        :type duplicatable: bool

        :param palindromable: non-palindrome mode for DNA sequences if exists.
        :type palindromable: bool

        :param length_fixed: fixing the length of the produced DNA sequence.
        :type length_fixed: bool

        :param seed: seed for selecting new virtual vertex if required (usable if palindrome mode).
        :type seed: int or None

        :param retry: retry for expanding DNA sequence with the given length.
        :type retry: int or None

        :param verbose: verbosity mode.
        :type verbose: bool

        :return: DNA sequence.
        :rtype: str
        """
        if self.__accessor is None:
            raise ValueError("Need to load or generate the coding digraph!")

        if virtuality is None:
            raise ValueError("Virtual vertex (i.e., \"virtuality\")  used in this process needs to be specified!")

        if virtuality >= len(self.__vertices):
            raise ValueError("\"virtuality\" should less than the number of vertices!")

        vertex, length, marker = self.__vertices[virtuality], len(message), 0

        if need_virtuality:
            sequence = m2s(i2m(array([self.__vertices[virtuality]]), length=self.__windowed))[0]
        else:
            sequence = ""

        if not self.__all_3:
            while marker < length:
                available_values = []
                for value in self.__accessor[vertex]:
                    if value >= 0:
                        nucleotide, available = self.__bases[value % 4], True
                        if duplicatable and self.__check_duplicatable(sequence=sequence + nucleotide):
                            available = False
                        if palindromable and self.__check_palindromable(sequence=sequence + nucleotide):
                            available = False
                        if available:
                            available_values.append(value)

                if len(available_values) == 3:
                    available_values = available_values[:2]

                out_degree = len(available_values)

                if out_degree == 0:
                    if seed is not None:
                        random.seed()
                    raise ValueError("out-degree = 0!")
                if out_degree == 1:
                    vertex = available_values[0]
                    sequence += self.__bases[vertex % 4]
                elif out_degree == 2:
                    index_choice = message[marker]
                    vertex = available_values[int(index_choice)]
                    sequence += self.__bases[vertex % 4]
                    marker += 1
                else:  # out-degree = 4.
                    index_choice = message[marker] * 2
                    if marker + 1 < length:
                        index_choice += message[marker + 1]
                    vertex = available_values[int(index_choice)]
                    sequence += self.__bases[vertex % 4]
                    marker += 2

                if verbose:
                    if length_fixed is None:
                        self.__monitor(marker, length)
                    else:
                        self.__monitor(len(sequence), length_fixed)
        else:
            if duplicatable:
                raise ValueError("Non-duplicatable mode is not available for all-3 coding digraph!")
            if palindromable:
                raise ValueError("Non-palindromable mode is not available for all-3 coding digraph!")
            if length_fixed:
                raise ValueError("Length-fixed requirement is not available for all-3 coding digraph!")

            backup, sequence, vertex = copy(message), "", self.__vertices[virtuality]

            while sum(backup) > 0:
                quotient, reminder = zeros(shape=(len(backup),), dtype=int), backup[0]
                for position in arange(1, len(backup)):
                    value = reminder * 2 + backup[position]
                    if value >= 3:
                        quotient[position], reminder = 1, value - 3
                    else:
                        reminder = value

                available_indices = where(self.__accessor[vertex] >= 0)
                next_vertices = self.__accessor[vertex][available_indices]
                vertex = next_vertices[reminder]
                sequence += self.__bases[vertex % 4]
                backup = quotient

        if seed is not None:
            random.seed(seed)

        if length_fixed is not None:
            if self.__all_3:
                if duplicatable:
                    raise ValueError("Non-duplicatable mode is not available for all-3 coding digraph!")
                if palindromable:
                    raise ValueError("Non-palindromable mode is not available for all-3 coding digraph!")
                if length_fixed:
                    raise ValueError("Length-fixed requirement is not available for all-3 coding digraph!")

            if len(sequence) > length_fixed:
                raise ValueError("The length setting of the fixed sequence length is unreasonable!")
            else:
                saved_sequence, times = sequence, 0
                while True:
                    sequence = saved_sequence
                    while len(sequence) < length_fixed:
                        available_values = []
                        if not self.__all_3:
                            for value in self.__accessor[vertex]:
                                if value >= 0:
                                    nucleotide, available = self.__bases[value % 4], True
                                    if duplicatable and self.__check_duplicatable(sequence=sequence + nucleotide):
                                        available = False
                                    if palindromable and self.__check_palindromable(sequence=sequence + nucleotide):
                                        available = False
                                    if available:
                                        available_values.append(value)
                        else:
                            for value in self.__accessor[vertex]:
                                if value >= 0:
                                    available_values.append(value)

                        if len(available_values) == 0:
                            break
                        else:
                            vertex = random.choice(available_values)
                            sequence += self.__bases[vertex % 4]

                    if len(sequence) == length_fixed:
                        if seed is not None:
                            random.seed()
                        return sequence

                    times += 1

                    if retry is not None and times == retry:
                        raise ValueError("Retry limit exceeded!")
        else:
            return sequence

    def decode(self,
               sequence: str,
               length: int,
               virtuality: Optional[int] = None,
               duplicatable: bool = False,
               palindromable: bool = False,
               verbose: bool = False) \
            -> ndarray:
        """
        Decode a DNA sequence to a binary message based on the established coding digraph.

        :param sequence: DNA sequence.
        :type sequence: str

        :param length: length of binary message.
        :type length: int

        :param virtuality: index of virtual vertex for each sequence; if not, take first k-mer as virtual index.
        :type virtuality: int or None

        :param duplicatable: non-duplicate mode for DNA sequences if needed.
        :type duplicatable: bool

        :param palindromable: non-palindrome mode for DNA sequences if needed.
        :type palindromable: bool

        :param verbose: verbosity mode.
        :type verbose: bool

        :return: binary message.
        :rtype: numpy.ndarray
        """
        if self.__accessor is None:
            raise ValueError("Need to load or generate the coding digraph!")

        message, number, marker = zeros(shape=(length,), dtype=int), len(sequence), 0
        if virtuality is not None:
            vertex = self.__vertices[virtuality]
            base_values = s2m(dna_sequences=[sequence])
        else:
            vertex = m2i(s2m([sequence[:self.__windowed]]))
            base_values = s2m(dna_sequences=[sequence[self.__windowed:]])

        if not self.__all_3:
            for process, base_value in enumerate(base_values):
                available_values = []
                for value in self.__accessor[vertex]:
                    if value >= 0:
                        nucleotide, available = self.__bases[value % 4], True
                        if duplicatable and \
                                self.__check_duplicatable(sequence=sequence[:process + self.__windowed] + nucleotide):
                            available = False
                        if palindromable and \
                                self.__check_palindromable(sequence=sequence[:process + self.__windowed] + nucleotide):
                            available = False
                        if available:
                            available_values.append(value)

                if len(available_values) == 3:
                    available_values = available_values[:2]

                out_degree = len(available_values)

                if out_degree == 0:
                    raise ValueError("out-degree = 0!")
                if out_degree == 1:
                    vertex = available_values[0]
                elif out_degree == 2:
                    for index, value in enumerate(available_values):
                        if value % 4 == base_value:
                            message[marker] = index
                            vertex = value
                            break
                    marker += 1
                else:
                    for index, value in enumerate(available_values):
                        if value % 4 == base_value:
                            # noinspection PyTypeChecker
                            message[marker] = base_value // 2
                            if marker + 1 < length:
                                # noinspection PyTypeChecker
                                message[marker + 1] = base_value % 2
                            vertex = value
                            break
                    marker += 2

                if verbose:
                    self.__monitor(process + 1, number)

                if marker > len(message) - 1:
                    if verbose:
                        self.__monitor(number, number)
                    break
        else:
            if duplicatable:
                raise ValueError("Non-duplicatable mode is not available for all-3 coding digraph!")
            if palindromable:
                raise ValueError("Non-palindromable mode is not available for all-3 coding digraph!")

            whole_sequence = (self.__bases[vertex % 4] + sequence)[::-1]
            for support_base, current_base in zip(whole_sequence[1:], whole_sequence[:-1]):
                reminder = 0
                for base in self.__bases:
                    if base != support_base:
                        if base != current_base:
                            reminder += 1
                        else:
                            break
                products = zeros(shape=(len(message),), dtype=int)
                products[1:] += message[1:]
                products[:-1] += message[1:]
                products[-2:] += [reminder // 2, reminder % 2]
                for location in arange(1, length)[::-1]:
                    if products[location] > 1:
                        products[location - 1] += products[location] // 2
                        products[location] %= 2
                message = products

        return message

    def repair(self,
               sequence: str,
               virtuality: int,
               length: Optional[int] = None,
               duplicatable: bool = False,
               palindromable: bool = False,
               verbose: bool = False) \
            -> Tuple[set, dict]:
        """
        Repair one DNA sequence based on the walk-base error-correcting (and other requirements).

        :param sequence: DNA sequence waiting for repair.
        :type sequence: str

        :param virtuality: index of virtual vertex to start decoding.
        :type virtuality: int

        :param length: expected length of DNA sequence if exists.
        :type length: int or None

        :param duplicatable: non-duplicate mode for DNA sequences if exists.
        :type duplicatable: bool

        :param palindromable: non-palindrome mode for DNA sequences if exists.
        :type palindromable: bool

        :param verbose: verbosity mode.
        :type verbose: bool

        :return: repaired candidates and the corresponding repair record if required.
        :rtype: set and None or set and dict
        """
        if self.__accessor is None:
            raise ValueError("Need to load or generate the coding digraph!")

        if virtuality >= len(self.__vertices):
            raise ValueError("\"virtuality\" should less than the number of vertices!")

        if len(sequence) <= 2 * self.__windowed + 1:
            raise ValueError("Sequence length must be at least " + str(2 * self.__windowed + 1) + "!")

        base_values, walk, records = s2m([sequence]), -ones(shape=(len(sequence),), dtype=int), {"f": [], "r": []}
        step, moment, access, split_pieces, chuck_pieces, markers = 0, self.__vertices[virtuality], 0, [""], [], []

        while step < len(sequence):
            nucleotide, base_value = sequence[step], base_values[step]
            if self.__accessor[moment, base_value] >= 0:  # satisfy the given constraints.
                split_pieces[-1] += nucleotide
                moment = self.__accessor[moment, base_value]
                walk[step] = moment

            elif len(split_pieces[-1]) > 0:  # find an error based on the given constraints.
                records["f"].append(step)

                # calculate the former and latter locations of the current chuck piece.
                former, latter = step - self.__windowed + 1, step + self.__windowed

                # save the current chuck piece.
                chuck_pieces.append(sequence[max([former, 0]): min([latter, len(sequence)])])

                # save the recent split piece and the path marker for the current chuck piece.
                if former <= 0:  # determine whether the error occurred within the first window.
                    marker = array([self.__vertices[virtuality]] + walk[: step].tolist())
                    if min(marker) > -1:
                        markers.append(marker)
                        split_pieces = [""]
                    else:  # two error merged, exceed the limit of error tolerance.
                        if verbose:
                            print("Two error merged, exceed the limit of error tolerance.")
                        return set(), {"detect": len(chuck_pieces), "access": access, "count": 0}
                else:
                    marker = walk[former - 1: step]
                    if min(marker) > -1:
                        markers.append(marker)
                        split_pieces[-1] = split_pieces[-1][: - self.__windowed + 1]
                    else:  # two error merged, exceed the limit of error tolerance.
                        if verbose:
                            print("Two error merged, exceed the limit of error tolerance.")
                        return set(), {"detect": len(chuck_pieces), "access": access, "count": 0}

                # calculate the next split piece if applicable.
                if latter + 1 <= len(sequence):
                    moment = m2i(matrix=base_values[step + 1: step + self.__windowed + 1])
                    split_pieces.append(self.__bases[moment % 4])
                    walk[step + self.__windowed] = moment
                else:
                    split_pieces.append("")  # determine whether the error occurred within the last window.

                step = latter

            step, access = step + 1, access + 1

        if verbose:
            if len(chuck_pieces) > 0:
                print("Found " + str(len(chuck_pieces)) + " errors:")
                table = PrettyTable(["error index", "location", "nucleotide"])
                for index, step in enumerate(records["f"]):
                    table.add_row([index + 1, step + 1, sequence[step]])
                print(table)
            else:
                print("Found " + str(len(chuck_pieces)) + " errors.")

        if len(chuck_pieces) == 0:
            if length is not None and length != len(sequence):
                return set(), {"detect": 0, "access": access, "count": 0}

            if duplicatable:
                duplicatable_box = set()
                for location in range(len(sequence) - self.__windowed + 1):
                    access += 1
                    duplicatable_box.add(sequence[location: location + self.__windowed])

                if len(duplicatable_box) != len(sequence) - self.__windowed + 1:
                    if verbose:
                        print("Find k-mer duplicate(s) in the repaired DNA sequence with non-duplicate mode!")
                    return set(), {"detect": 0, "access": access, "count": 0}

            if palindromable:
                palindromable_box = set()
                for location in range(len(sequence) - self.__windowed):
                    access += 1
                    palindromable_box.add(sequence[location: location + self.__windowed + 1])

                reverse_sequence, flag = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1], True
                for location in range(len(reverse_sequence) - self.__windowed):
                    access += 1
                    if reverse_sequence[location: location + self.__windowed + 1] in palindromable_box:
                        if verbose:
                            print("Find k-mer palindrome(s) in the repaired DNA sequence with non-palindrome mode!")
                        flag = False
                        break

                if not flag:
                    return set(), {"detect": 0, "access": access, "count": 0}

            return {sequence}, {"detect": 0, "access": access, "count": 0}

        repaired_sequences, repaired_set, repaired_count = set(), [], 1
        for index, (chuck_piece, marker) in enumerate(zip(chuck_pieces, markers)):
            repaired_set.append(set())
            for recall, previous in enumerate(marker[::-1]):
                sub_candidates, sub_access, occur = set(), 0, len(marker) - recall - 1
                for base_value in where(self.__accessor[previous] >= 0)[0]:
                    base = self.__bases[base_value]

                    # check as a substitution.
                    if base != chuck_piece[occur]:
                        moment, reliable = self.__accessor[previous, base_value], True
                        for chuck_base in chuck_piece[occur + 1:]:
                            found = False
                            for latter in self.__accessor[moment]:
                                if latter >= 0 and self.__bases[latter % 4] == chuck_base:
                                    moment, found = latter, True
                                    sub_access += 1
                            if not found:
                                reliable = False
                                break
                        if reliable:
                            candidate_sequence = list(chuck_piece)
                            candidate_sequence[occur] = base
                            # noinspection PyTypeChecker
                            sub_candidates.add("".join(candidate_sequence))

                    # check as an insertion.
                    moment, reliable = self.__accessor[previous, base_value], True
                    for chuck_base in chuck_piece[occur:]:
                        found = False
                        for latter in self.__accessor[moment]:
                            if latter >= 0 and self.__bases[latter % 4] == chuck_base:
                                moment, found = latter, True
                                sub_access += 1
                        if not found:
                            reliable = False
                            break
                    if reliable:
                        candidate_sequence = list(chuck_piece)
                        candidate_sequence.insert(occur, base)
                        # noinspection PyTypeChecker
                        sub_candidates.add("".join(candidate_sequence))

                # check as a deletion.
                moment, reliable = previous, True
                for chuck_base in chuck_piece[occur + 1:]:
                    found = False
                    for latter in self.__accessor[moment]:
                        if latter >= 0 and self.__bases[latter % 4] == chuck_base:
                            moment, found = latter, True
                            sub_access += 1
                    if not found:
                        reliable = False
                        break
                if reliable:
                    candidate_sequence = list(chuck_piece)
                    del candidate_sequence[occur]
                    # noinspection PyTypeChecker
                    sub_candidates.add("".join(candidate_sequence))

                repaired_set[-1].update(sub_candidates)
                access += sub_access

            repaired_count *= len(repaired_set[-1])
            records["r"].append(len(repaired_set[-1]))

        if verbose:
            print("Obtain " + str(repaired_count) + " repaired candidate(s) are obtained:")
            table = PrettyTable(["error index", "candidate number"])
            for index, count in enumerate(records["r"]):
                table.add_row([index + 1, count])
            print(table)

        for fragments in product(*repaired_set):
            repaired_sequence = ""
            for index in range(len(split_pieces) - 1):
                repaired_sequence += split_pieces[index] + fragments[index]
            repaired_sequence += split_pieces[-1]

            if length is not None and length != len(repaired_sequence):
                continue

            if duplicatable:
                duplicatable_box = set()
                for location in range(len(repaired_sequence) - self.__windowed + 1):
                    access += 1
                    duplicatable_box.add(repaired_sequence[location: location + self.__windowed])

                if len(duplicatable_box) != len(repaired_sequence) - self.__windowed + 1:
                    continue

            if palindromable:
                palindromable_box = set()
                for location in range(len(repaired_sequence) - self.__windowed):
                    access += 1
                    palindromable_box.add(repaired_sequence[location: location + self.__windowed + 1])

                reverse_sequence, flag = repaired_sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1], True
                for location in range(len(reverse_sequence) - self.__windowed):
                    access += 1
                    if reverse_sequence[location: location + self.__windowed + 1] in palindromable_box:
                        flag = False
                        break

                if not flag:
                    continue

            repaired_sequences.add(repaired_sequence)

        return repaired_sequences, {"detect": len(chuck_pieces), "access": access, "count": repaired_count}

    def produce(self,
                length: int,
                diversity: int,
                virtuality: int,
                duplicatable: bool = False,
                palindromable: bool = False,
                need_preorder: bool = False,
                seed: Optional[int] = None,
                analysis: Optional[int] = None,
                similarity: Optional[int] = None,
                fail_repeats: Optional[int] = None,
                verbose: bool = False) \
            -> Tuple[list, ndarray, int]:
        """
        Produce DNA sequences randomly.

        :param length: DNA sequence length.
        :type length: int

        :param diversity: production number of DNA sequences.
        :type diversity: int

        :param virtuality: index of virtual vertex to start decoding.
        :type virtuality: int

        :param duplicatable: non-duplicate mode for produced DNA sequences if required.
        :type duplicatable: bool

        :param palindromable: non-palindrome mode for produced DNA sequences if required.
        :type palindromable: bool

        :param need_preorder: put the sequence of virtual node at the start  of the produced DNA sequence if required.
        :type need_preorder: bool

        :param seed: random seed if required.
        :type seed: int or None

        :param analysis: method to analyse the produced sequence difference if required, i.e. hamming or levenshtein.
        :type analysis: str or None

        :param similarity: minimum and/or maximum similarity between produced DNA sequences.
        :type similarity: dict or None

        :param fail_repeats: maximum number of repeated failures if requited.
        :type fail_repeats: int or None

        :param verbose: verbosity mode.
        :type verbose: bool

        :return: produced DNA sequences, the corresponding similarity, and miss count.
        :rtype: list, numpy.ndarray, int
        """
        if self.__accessor is None:
            raise ValueError("Need to load or generate the coding digraph!")

        if seed is not None:
            random.seed(seed)

        sequences, values, counts = set(), None, zeros(shape=(length + 1,), dtype=int)

        if virtuality is None:
            raise ValueError("Virtual vertex (i.e., \"start\")  used in the producing process needs to be specified!")

        if virtuality >= len(self.__vertices):
            raise ValueError("\"virtuality\" should less than the number of vertices!")

        if need_preorder:
            preorder = m2s(i2m(integers=array([self.__vertices[virtuality]]), length=self.__windowed))[0]
        else:
            preorder = ""

        miss = 0
        while len(sequences) < diversity:
            miss_flag, fail = False, 0

            while True:
                value = -ones(shape=(length - len(preorder) + 1,), dtype=int)
                sequence, value[0] = preorder, self.__vertices[virtuality]
                for step in range(length - len(preorder)):
                    available = []
                    for follow_up in self.__accessor[value[step]]:
                        if follow_up == -1:
                            continue
                        if duplicatable and self.__check_duplicatable(sequence=sequence):
                            continue
                        if palindromable and self.__check_palindromable(sequence=sequence):
                            continue

                        available.append(follow_up)

                    if len(available) == 3 and not self.__all_3:
                        available = available[:2]

                    if len(available) == 0:
                        miss_flag = True
                        break

                    # noinspection PyArgumentList
                    moment = random.choice(available)
                    sequence += self.__bases[moment % 4]
                    value[step + 1] = moment

                if len(sequence) == length:
                    break
                else:
                    fail += 1

                if fail == fail_repeats:
                    return [], counts, fail_repeats

            if sequence in sequences:
                continue

            else:
                if analysis is not None:
                    if values is not None:
                        value = array(value, dtype=int)
                        if analysis == "hamming":
                            # noinspection PyTypeChecker
                            distances = hamming(group=values % 4, sample=value % 4)
                            if similarity is not None:
                                if "min" in similarity and similarity["min"] >= min(distances):
                                    continue
                                if "max" in similarity and similarity["max"] <= max(distances):
                                    continue
                            for distance in distances:
                                counts[distance] += 1

                        elif analysis == "levenshtein":
                            # noinspection PyTypeChecker
                            distances = levenshtein(group=values % 4, sample=value % 4)
                            if similarity is not None:
                                if "min" in similarity and similarity["min"] >= min(distances):
                                    continue
                                if "max" in similarity and similarity["max"] <= max(distances):
                                    continue
                            for distance in distances:
                                counts[distance] += 1

                        else:
                            raise ValueError("No such distance metric!")

                        values = vstack((values, expand_dims(value, axis=0)))
                    else:
                        values = array([value], dtype=int)

            sequences.add(sequence)
            miss += miss_flag

            if verbose:
                if sum(counts) > 0:
                    order = where(counts > 0)[0]
                    self.__monitor(len(sequences), diversity, extra={"min": order[0], "max": order[-1], "miss": miss})
                else:
                    self.__monitor(len(sequences), diversity, extra={"miss": miss})

        if seed is not None:
            random.seed()

        return list(sequences), counts, miss

    def retrieve(self,
                 sequences: list,
                 virtuality: int,
                 diversity: int,
                 length: Optional[int] = None,
                 duplicatable: bool = False,
                 palindromable: bool = False,
                 need_raw: bool = False,
                 verbose: bool = False) \
            -> Union[set, Tuple[set, dict]]:
        """
        Retrieve expected DNA sequences from raw DNA sequences based on ranking.

        :param sequences: retrieved raw DNA sequences.
        :type sequences: list

        :param virtuality: virtual vertex to start decoding (consistent with the encoding log).
        :type virtuality: int

        :param diversity: expected retrieval diversity (or sequence number).
        :type diversity: int

        :param duplicatable: non-duplicate mode for retrieved DNA sequences if required.
        :type duplicatable: bool

        :param palindromable: non-palindrome mode for retrieved DNA sequences if required.
        :type palindromable: bool

        :param length: expected length of retrieved DNA sequences if required.
        :type length: int or None

        :param need_raw: raw heap data if needed.
        :type need_raw: bool

        :param verbose: verbosity mode.
        :type verbose: bool

        :return: retrieved DNA sequences and raw heap data if required.
        :rtype: set or (set, dict)
        """
        collection = RankingHeap()
        for index, sequence in enumerate(sequences):
            candidates = self.repair(sequence=sequence, virtuality=virtuality, length=length,
                                     duplicatable=duplicatable, palindromable=palindromable)[0]
            if len(candidates) > 0:
                for candidate in candidates:
                    collection.push(candidate)

            if verbose:
                self.__monitor(index + 1, len(sequences))

        if need_raw:
            return collection.pop_all(top=diversity), collection.get_raw()
        else:
            return collection.pop_all(top=diversity)

    def __check_duplicatable(self,
                             sequence: str) \
            -> bool:
        """
        Check the duplicates in DNA sequence.

        :param sequence: produced DNA sequence.
        :type sequence: str

        :return: duplicatable flag.
        :rtype: bool
        """
        k_mers = []

        for location in range(len(sequence) - self.__windowed + 1):
            k_mers.append(sequence[location: location + self.__windowed])

        return len(k_mers) > len(set(k_mers))

    def __check_palindromable(self,
                              sequence: str) \
            -> bool:
        """
        Check the palindrome in DNA sequence.

        :param sequence: produced DNA sequence.
        :type sequence: str

        :return: palindromable flag.
        :rtype: bool
        """
        chuck_sequence = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]

        for location in range(len(sequence) - self.__windowed):
            if sequence[location: location + self.__windowed + 1] in chuck_sequence:
                return True

        return False

    def get_params(self) \
            -> dict:
        """
        Get the parameters of the SPIDER-WEB architecture.

        :return: window size, vertices, and accessor.
        :rtype: dict
        """
        return {"window size": self.__windowed,
                "vertices": deepcopy(self.__vertices),
                "accessor": deepcopy(self.__accessor)}


# noinspection PyTypeChecker,PyUnresolvedReferences
class FastTranscoder:

    def __init__(self,
                 accessor: ndarray,
                 coexisting_map: Optional[dict] = None,
                 palindrome_map: Optional[ndarray] = None):
        """
        Initialise the SPIDER-WEB architecture.

        :param accessor: the established coding digraph.
        :type accessor: numpy.ndarray

        :param coexisting_map: coexisting virtual vertex map if required.
        :type coexisting_map: dict or None

        :param palindrome_map: palindrome map for non-palindrome mode if required.
        :type palindrome_map: ndarray or None
        """
        self.__accessor = accessor
        self.__coexisting_map, self.__palindrome_map = coexisting_map, palindrome_map
        self.__vertices = where(sum(((self.__accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0]
        self.__windowed = int(log(len(self.__accessor)) / log(4))

        out_statistics = count_out_degrees(accessor=accessor)
        if out_statistics[3] > 0 and out_statistics[3] != sum(out_statistics):
            raise ValueError("If a graph contains nodes with out-degree of 3 and others (i.e., 1, 2, and 4), "
                             "it will cause encoding ambiguity")
        self.__all_3 = out_statistics[3] == sum(out_statistics)

        self.__bases = ("A", "C", "G", "T")
        self.__monitor = Monitor()

    def encode(self,
               messages: ndarray,
               virtuality: int,
               duplicatable: bool = False,
               palindromable: bool = False,
               length_fixed: bool = False,
               seed: int = None,
               verbose: bool = False) \
            -> Tuple[list, ndarray]:
        """
        Encode binary messages to DNA sequences based on the established coding digraph.

        :param messages: binary messages.
        :type messages: numpy.ndarray

        :param virtuality: virtual vertex to start encoding.
        :type virtuality: int

        :param duplicatable: non-duplicate mode for DNA sequences if exists.
        :type duplicatable: bool

        :param palindromable: non-palindrome mode for DNA sequences if exists.
        :type palindromable: bool

        :param length_fixed: fixing the length of the produced DNA sequence.
        :type length_fixed: bool

        :param seed: seed for selecting new virtual vertex if required (usable if palindrome mode).
        :type seed: int or None

        :param verbose: verbosity mode.
        :type verbose: bool

        :return: DNA sequences and their corresponding virtual vertices.
        :rtype: list, numpy.ndarray
        """
        if self.__accessor is None:
            raise ValueError("Need to load or generate the coding digraph!")

        if virtuality is None:
            raise ValueError("Virtual vertex (i.e., \"virtuality\")  used in this process needs to be specified!")

        if virtuality >= len(self.__vertices):
            raise ValueError("\"virtuality\" should less than the number of vertices!")

        if (duplicatable or palindromable) and self.__coexisting_map is None:
            raise ValueError("Non-duplicatable or non-palindromable mode is not available"
                             + " if \"coexisting_map\" is not initialized!")

        if palindromable and self.__palindrome_map is None:
            raise ValueError("Non-palindromable mode is not available"
                             + " if \"palindrome_map\" is not initialized!")

        if (duplicatable or palindromable or length_fixed) and seed is not None:
            random.seed(seed)

        backup, number, length, additional_round = deepcopy(messages), messages.shape[0], messages.shape[1], 0
        start_vertices = ones(shape=(number,), dtype=int) * self.__vertices[virtuality]
        saved_sequences, saved_vertices, used_virtualities, total = [], [], set(), number

        while True:  # design for non-duplicate mode and non-palindrome mode.
            sequences, paths = None, expand_dims(start_vertices, axis=0)

            if verbose:
                if additional_round > 0:
                    info = "additional round = " + str(additional_round) + " for "
                    if duplicatable and palindromable:
                        info += "non-duplicatable and non-palindromable mode, "
                    elif duplicatable:
                        info += "non-duplicatable mode, "
                    elif palindromable:
                        info += "non-palindromable mode, "
                    info += "remaining %d message(s)." % len(start_vertices)
                    print(info)

            vertices = deepcopy(start_vertices)
            if not self.__all_3:
                markers = zeros(shape=(total,), dtype=int)

                while True:
                    if length_fixed:
                        # even if a sequence has completed encoding, it can still be continuously encoded with 0.
                        action = arange(number, dtype=int)
                    else:
                        action = where(markers < length)[0]

                    if all(markers >= length):
                        break

                    paths = vstack((paths, -ones(shape=(1, total), dtype=int)))

                    out_degrees = count_nonzero(self.__accessor[vertices[action]] + 1, axis=1)
                    addition = ones(shape=(total,), dtype=int) * 4

                    radix_choices = where(out_degrees == 0)[0]
                    if len(radix_choices) > 0:
                        raise ValueError("Vertex " + str(vertices[action[radix_choices]])
                                         + " have no outgoing vertices,"
                                         + " indicating that the coding digraph or the start vertex is wrong!")

                    # check vertices with out-degree of 1.
                    radix_choices = where(out_degrees == 1)[0]
                    if len(radix_choices) > 0:
                        action_choices = action[radix_choices]
                        vertices[action_choices] = sum(self.__accessor[vertices[action_choices]], axis=1) + 3
                        addition[action_choices] = vertices[action_choices] % 4

                    # check vertices with out-degree of 2.
                    radix_choices = where(out_degrees == 2)[0]
                    if len(radix_choices) > 0:
                        action_choices = action[radix_choices]
                        location_choices = markers[action_choices]
                        if length_fixed:
                            remained = where(location_choices < length)[0]
                            index_choices = random.randint(0, 2, size=(len(radix_choices),), dtype=int)
                            index_choices[remained] = backup[action_choices[remained], location_choices[remained]]
                        else:
                            index_choices = backup[action_choices, location_choices]
                        candidates = nonzero(self.__accessor[vertices[action_choices]] + 1)[1]
                        candidates = candidates.reshape(len(radix_choices), 2)
                        follow_ups = candidates[arange(len(candidates)), index_choices]
                        addition[action_choices] = follow_ups
                        vertices[action_choices] = self.__accessor[vertices[action_choices], follow_ups]
                        markers[action_choices] += 1

                    # check vertices with out-degree of 3.
                    radix_choices = where(out_degrees == 3)[0]
                    if len(radix_choices) > 0:
                        raise ValueError("Vertex " + str(vertices[action[radix_choices]]) + " have 3 outgoing vertices,"
                                         + " which are removed when creating the coding digraph!")

                    # check vertices with out-degree of 4.
                    radix_choices = where(out_degrees == 4)[0]
                    if len(radix_choices) > 0:
                        action_choices = action[radix_choices]
                        location_choices = markers[action_choices]
                        if length_fixed:
                            remained_1 = where(location_choices + 0 < length)[0]
                            remained_2 = where(location_choices + 1 < length)[0]
                            index_choices = random.randint(0, 2, size=(len(radix_choices),), dtype=int)
                            index_choices[remained_1] = backup[action_choices, location_choices[remained_1]] * 2
                            if len(remained_2) > 0:
                                x, y = action_choices[remained_2], location_choices[remained_2] + 1
                                index_choices[remained_2] += backup[x, y]
                        else:
                            remained_2 = where(location_choices + 1 < length)[0]
                            index_choices = backup[action_choices, location_choices] * 2
                            if len(remained_2) > 0:
                                x, y = action_choices[remained_2], location_choices[remained_2] + 1
                                index_choices[remained_2] += backup[x, y]
                        candidates = nonzero(self.__accessor[vertices[action_choices]] + 1)[1]
                        candidates = candidates.reshape(len(radix_choices), 4)
                        follow_ups = candidates[arange(len(candidates)), index_choices]
                        addition[action_choices] = follow_ups
                        vertices[action_choices] = self.__accessor[vertices[action_choices], follow_ups]
                        markers[action_choices] += 2

                    if sequences is not None:
                        sequences = hstack((sequences, expand_dims(addition, axis=1)))
                    else:
                        sequences = expand_dims(addition, axis=1)

                    paths[-1, action] = vertices[action]

                    if verbose:
                        self.__monitor(min(markers), length)

            else:
                while True:
                    processes = length - sum(backup, axis=1)
                    if length_fixed:
                        # even if a sequence has completed encoding, it can still be continuously encoded with 0.
                        action = arange(number, dtype=int)
                    else:
                        action = where(processes != length)[0]

                    if all(processes == length):
                        break

                    paths = vstack((paths, -ones(shape=(1, total), dtype=int)))

                    quotients, reminders = zeros_like(backup), backup[:, 0]
                    for location in range(1, length):
                        values = reminders * 2 + backup[:, location]
                        group_1, group_2 = where(values >= 3)[0], where(values < 3)[0]
                        quotients[group_1, location], reminders[group_1] = 1, values[group_1] - 3
                        reminders[group_2] = values[group_2]

                    addition = ones(shape=(total,), dtype=int) * 4
                    candidates = nonzero(self.__accessor[vertices[action]] + 1)[1]
                    candidates = candidates.reshape(len(action), 3)
                    follow_ups = candidates[arange(len(candidates)), reminders[action]]
                    addition[action] = follow_ups

                    vertices[action] = self.__accessor[vertices[action], follow_ups]
                    backup[action] = quotients[action]

                    if sequences is not None:
                        sequences = hstack((sequences, expand_dims(addition, axis=1)))
                    else:
                        sequences = expand_dims(addition, axis=1)

                    paths[-1, action] = vertices[action]

                    if verbose:
                        self.__monitor(length - max(processes), length)

            duplicate_flags = zeros(shape=(total,), dtype=int)
            if duplicatable:
                for index, level in enumerate(paths[1:]):
                    action = where(level >= 0)[0]
                    detections = min(abs(paths[:index + 1, action] - level[action]), axis=0)
                    duplicate_flags[action[where(detections == 0)[0]]] = 1  # mark same vertices (becoming a cycle).

            palindrome_flags = zeros(shape=(total,), dtype=int)
            if palindromable:
                for index, level in enumerate(paths[1:]):
                    action_1 = where(level >= 0)[0]
                    palindromable_level = self.__palindrome_map[level[action_1]]
                    action_2 = where(palindromable_level >= 0)[0]
                    detections = min(abs(paths[:index + 1, action_1[action_2]] - level[action_1[action_2]]), axis=0)
                    palindrome_flags[action_1[action_2[where(detections == 0)[0]]]] = 1

            if all(duplicate_flags == 0) and all(palindrome_flags == 0):
                saved_sequences += m2s(matrix=sequences)
                saved_vertices += list(start_vertices)
            else:  # screen sequences with duplicate and/or palindrome if required.
                usable_indices = where(duplicate_flags + palindrome_flags == 0)[0]
                if len(usable_indices) > 0:
                    saved_sequences += m2s(matrix=sequences[usable_indices])
                    saved_vertices += list(start_vertices[usable_indices])
                    # noinspection PyUnhashable
                    used_virtualities.add(start_vertices[0])

            if len(saved_vertices) == number and len(saved_vertices) == number:
                break

            # choose a new virtual vertex.
            available_virtual_vertices = set(self.__vertices)
            for used_virtuality in list(used_virtualities):
                available_virtual_vertices &= self.__coexisting_map[used_virtuality]

            if len(available_virtual_vertices) == 0:
                raise RuntimeError("No available virtualities, remaining " + str(total) + " message(s) unencoded. "
                                   + "Selecting a new \"virtuality\" or a new random \"seed\" may solve this problem.")
            new_virtual_vertex = random.choice(list(available_virtual_vertices))

            # encode remaining binary messages.
            screened_indices = where(duplicate_flags + palindrome_flags > 0)[0]  # duplicate and/or palindrome
            backup, total = deepcopy(messages[screened_indices]), len(screened_indices)
            start_vertices = ones(shape=(total,), dtype=int) * new_virtual_vertex
            additional_round += 1

        if (duplicatable or palindromable or length_fixed) and seed is not None:
            random.seed()

        return saved_sequences, array(saved_vertices)

    def decode(self,
               sequences: list,
               virtualities: ndarray,
               length: int,
               verbose: bool = False) \
            -> ndarray:
        """
        Decode DNA sequences to binary messages based on the established coding digraph.

        :param sequences: DNA sequences
        :type sequences: list

        :param virtualities: index of virtual vertex for each sequence.
        :type virtualities: numpy.ndarray

        :param length: length of binary message.
        :type length: int

        :param verbose: verbosity mode.
        :type verbose: bool

        :return: binary messages.
        :rtype: numpy.ndarray
        """
        if self.__accessor is None:
            raise ValueError("Need to load or generate the coding digraph!")

        if len(virtualities) != len(sequences):
            raise ValueError("\"sequences\" and \"virtualities\" are one-to-one correspondence!")

        sequences, number = s2m(dna_sequences=sequences), len(sequences)

        messages, vertices = zeros(shape=(number, length), dtype=int), self.__vertices[virtualities]

        if not self.__all_3:
            markers = zeros(shape=(number,), dtype=int)

            for index, base_values in enumerate(sequences.T):
                # noinspection PyTypeChecker
                action = where((base_values >= 0) & (base_values <= 3))[0]

                if len(action) == 0:
                    continue

                out_degrees = count_nonzero(self.__accessor[vertices[action]] + 1, axis=1)

                radix_choices = where(out_degrees == 0)[0]
                if len(radix_choices) > 0:
                    raise ValueError("Vertex " + str(vertices[action[radix_choices]]) + " have no outgoing vertices, "
                                     + "indicating that the coding digraph, the start vertex, "
                                     + "or the corresponding sequence is wrong!")

                radix_choices = where(out_degrees == 1)[0]
                if len(radix_choices) > 0:
                    action_choices = action[radix_choices]
                    vertices[action_choices] = sum(self.__accessor[vertices[action_choices]], axis=1) + 3

                radix_choices = where(out_degrees == 2)[0]
                if len(radix_choices) > 0:
                    action_choices = action[radix_choices]
                    location_choices, base_choices = markers[action_choices], base_values[action_choices]
                    candidates = nonzero(self.__accessor[vertices[action_choices]] + 1)[1]
                    candidates = candidates.reshape(len(radix_choices), 2)
                    binaries = argwhere(candidates == expand_dims(base_choices, axis=1))[:, 1]
                    used_indices = where(location_choices < length)[0]
                    messages[action_choices[used_indices], location_choices[used_indices]] = binaries[used_indices]
                    vertices[action_choices] = self.__accessor[vertices[action_choices], base_choices]
                    markers[action_choices] += 1

                radix_choices = where(out_degrees == 3)[0]
                if len(radix_choices) > 0:
                    raise ValueError("Vertex " + str(vertices[action[radix_choices]]) + " have 3 outgoing vertices,"
                                     + " which are removed when creating the coding digraph!")

                radix_choices = where(out_degrees == 4)[0]
                if len(radix_choices) > 0:
                    action_choices = action[radix_choices]
                    location_choices, base_choices = markers[action_choices], base_values[action_choices]
                    remained_1 = where(location_choices + 0 < length)[0]
                    remained_2 = where(location_choices + 1 < length)[0]
                    messages[action_choices, location_choices[remained_1] + 0] = base_choices[remained_1] // 2
                    messages[action_choices, location_choices[remained_2] + 1] = base_choices[remained_2] % 2
                    vertices[action_choices] = self.__accessor[vertices[action_choices], base_choices]
                    markers[action_choices] += 2

                if verbose:
                    self.__monitor(index + 1, sequences.shape[1])

        else:
            reminder_groups, markers = zeros_like(sequences, dtype=int) - 1, zeros(shape=(number,), dtype=int)
            for location, base_values in enumerate(sequences.T):
                # noinspection PyTypeChecker
                action = where((base_values >= 0) & (base_values <= 3))[0]
                base_choices = base_values[action]
                candidates = nonzero(self.__accessor[vertices[action]] + 1)[1].reshape(len(action), 3)
                reminders = argwhere(candidates == expand_dims(base_choices, axis=1))[:, 1]
                reminder_groups[action, markers[action]] = reminders
                vertices[action] = self.__accessor[vertices[action], base_choices]
                markers[action] += 1

                if verbose:
                    self.__monitor(location + 1, length * 2)

            temporary = zeros(shape=(number, length), dtype=int)
            for index, reminders in enumerate(reminder_groups.T[::-1]):
                action = where(reminders >= 0)[0]

                if len(action) == 0:
                    continue

                temporary *= 0  # refresh the temporary matrix
                temporary[action, 1:] += messages[action, 1:]
                temporary[action, :-1] += messages[action, 1:]
                temporary[action, -2] += reminders[action] // 2
                temporary[action, -1] += reminders[action] % 2
                for location in arange(1, length)[::-1]:
                    moved_indices = where(temporary[action, location] > 1)[0]
                    if len(moved_indices) > 0:
                        moved_values = temporary[action[moved_indices], location] // 2
                        temporary[action[moved_indices], location - 1] += moved_values
                        temporary[action[moved_indices], location] %= 2

                messages[action] = temporary[action]

                if verbose:
                    self.__monitor(index + length + 1, length * 2)

        return messages


# noinspection PyUnresolvedReferences
class FastGenerator:

    def __init__(self,
                 accessor: ndarray,
                 palindrome_map: Optional[ndarray] = None):
        """
        Initialise the fast sequence generator based on the SPIDER-WEB architecture.

        :param accessor: the established coding digraph.
        :type accessor: numpy.ndarray

        :param palindrome_map: palindrome map for non-palindrome mode if required.
        :type palindrome_map: ndarray or None
        """
        self.__accessor = accessor
        self.__palindrome_map = palindrome_map
        self.__vertices = where(sum(((self.__accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0]
        self.__windowed = int(log(len(self.__accessor)) / log(4))

        out_statistics = count_out_degrees(accessor=accessor)
        if out_statistics[3] > 0 and out_statistics[3] != sum(out_statistics):
            raise ValueError("If a graph contains nodes with out-degree of 3 and others (i.e., 1, 2, and 4), "
                             "it will cause encoding ambiguity")
        self.__all_3 = out_statistics[3] == sum(out_statistics)

        self.__bases = ("A", "C", "G", "T")
        self.__monitor = Monitor()

    def __call__(self,
                 diversity: int,
                 batch_size: int,
                 length: int,
                 virtuality: int,
                 duplicatable: bool = False,
                 palindromable: bool = False,
                 seed: Optional[int] = None,
                 verbose: bool = False) \
            -> list:
        """
        Generate DNA sequences.

        :param diversity: production number of DNA sequences.
        :type diversity: int

        :param batch_size: batch size of DNA sequences in each round.
        :type batch_size: int

        :param length: DNA sequence length.
        :type length: int

        :param virtuality: index of virtual vertex to start decoding.
        :type virtuality: int

        :param duplicatable: non-duplicate mode for produced DNA sequences if required.
        :type duplicatable: bool

        :param palindromable: non-palindrome mode for produced DNA sequences if required.
        :type palindromable: bool

        :param seed: random seed if required.
        :type seed: int or None

        :param verbose: verbosity mode.
        :type verbose: bool

        :return: produced DNA sequences.
        :rtype: list
        """
        if self.__accessor is None:
            raise ValueError("Need to load or generate the coding digraph!")

        if palindromable and self.__palindrome_map is None:
            raise ValueError("Non-palindrome mode is not available if \"palindrome_map\" is not initialized!")

        if seed is not None:
            random.seed(seed)

        sequences = set()
        while len(sequences) < diversity:
            batch_sequences = ones(shape=(batch_size, length), dtype=int) * 4
            paths = -ones(shape=(length + 1, batch_size), dtype=int)
            paths[0] = self.__vertices[virtuality]
            if not self.__all_3:
                for index in range(length):
                    out_degrees = count_nonzero(self.__accessor[paths[index]] + 1, axis=1)

                    # check vertices with out-degree of 0.
                    radix_choices = where(out_degrees == 0)[0]
                    if len(radix_choices) > 0:
                        raise ValueError("Vertex " + str(paths[index, radix_choices])
                                         + " have no outgoing vertices,"
                                         + " indicating that the coding digraph or the start vertex is wrong!")

                    # check vertices with out-degree of 1.
                    radix_choices = where(out_degrees == 1)[0]
                    if len(radix_choices) > 0:
                        paths[index + 1, radix_choices] = sum(self.__accessor[paths[index, radix_choices]], axis=1) + 3
                        batch_sequences[radix_choices, index] = paths[index + 1, radix_choices] % 4

                    # check vertices with out-degree of 2.
                    radix_choices = where(out_degrees == 2)[0]
                    if len(radix_choices) > 0:
                        candidates = nonzero(self.__accessor[paths[index, radix_choices]] + 1)[1]
                        candidates = candidates.reshape(len(radix_choices), 2)
                        follow_ups = candidates[arange(len(candidates)), random.randint(0, 2, size=(len(candidates),))]
                        paths[index + 1, radix_choices] = self.__accessor[paths[index, radix_choices], follow_ups]
                        batch_sequences[radix_choices, index] = follow_ups

                    # check vertices with out-degree of 3.
                    radix_choices = where(out_degrees == 3)[0]
                    if len(radix_choices) > 0:
                        raise ValueError("Vertex " + str(paths[index, radix_choices]) + " have 3 outgoing vertices,"
                                         + " which are removed when creating the coding digraph!")

                    # check vertices with out-degree of 4.
                    radix_choices = where(out_degrees == 4)[0]
                    if len(radix_choices) > 0:
                        candidates = nonzero(self.__accessor[paths[index, radix_choices]] + 1)[1]
                        candidates = candidates.reshape(len(radix_choices), 4)
                        follow_ups = candidates[arange(len(candidates)), random.randint(0, 4, size=(len(candidates),))]
                        batch_sequences[radix_choices, index] = follow_ups
                        paths[index + 1, radix_choices] = self.__accessor[paths[index, radix_choices], follow_ups]

            else:
                for index in range(length):
                    candidates = nonzero(self.__accessor[paths[index]] + 1)[1]
                    candidates = candidates.reshape(batch_size, 3)
                    follow_ups = candidates[arange(len(candidates)), random.randint(0, 3, size=(len(candidates),))]
                    batch_sequences[index] = follow_ups
                    paths[index + 1] = self.__accessor[paths[index], follow_ups]

            duplicate_flags = zeros(shape=(batch_size,), dtype=int)
            if duplicatable:
                for index, level in enumerate(paths[1:]):
                    action = where(level >= 0)[0]
                    detections = min(abs(paths[:index + 1, action] - level[action]), axis=0)
                    duplicate_flags[action[where(detections == 0)[0]]] = 1  # mark same vertices (becoming a cycle).

            palindrome_flags = zeros(shape=(batch_size,), dtype=int)
            if palindromable:
                for index, level in enumerate(paths[1:]):
                    action_1 = where(level >= 0)[0]
                    palindromable_level = self.__palindrome_map[level[action_1]]
                    action_2 = where(palindromable_level >= 0)[0]
                    detections = min(abs(paths[:index + 1, action_1[action_2]] - level[action_1[action_2]]), axis=0)
                    palindrome_flags[action_1[action_2[where(detections == 0)[0]]]] = 1

            usable_indices = where(duplicate_flags + palindrome_flags == 0)[0]
            if len(usable_indices) > 0:
                sequences |= set(m2s(matrix=batch_sequences[usable_indices]))

            if verbose:
                self.__monitor(len(sequences), diversity)

        if seed is not None:
            random.seed()

        return list(sequences)


# noinspection PyTypeChecker
class FastRepair:

    def __init__(self,
                 accessor: ndarray,
                 virtuality: int,
                 duplicatable: bool,
                 palindromable: bool,
                 length: int,
                 thread_number: int = 1,
                 diversity: int = None):
        """
        Initialise the fast sequence retriever based on the SPIDER-WEB architecture.

        :param accessor: the established coding digraph.
        :type accessor: numpy.ndarray

        :param virtuality: index of virtual vertex to start decoding.
        :type virtuality: int

        :param duplicatable: non-duplicate mode for produced DNA sequences if required.
        :type duplicatable: bool

        :param palindromable: non-palindrome mode for produced DNA sequences if required.
        :type palindromable: bool

        :param length: length of repaired DNA sequence.
        :type length: int

        :param thread_number: number of threads used in the accelerator.
        :type thread_number: int

        :param diversity: sequence number of the predetermined digital data.
        :type diversity: int or None
        """
        parent_path = path.dirname(path.abspath(__file__)).replace("\\", "/")
        if name == "nt":  # Windows system
            self.__acc_path = parent_path + "/accelerator.exe"

        else:
            self.__acc_path = parent_path + "/accelerator"

        if not path.exists(self.__acc_path):
            raise RuntimeError("Due to the lack of the necessary executable file in "
                               + parent_path + "/, acceleration is not supported!")

        self.__accessor = accessor
        self.__vertices = where(sum(((self.__accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0]
        self.__windowed = int(log(len(self.__accessor)) / log(4))

        accessor_file_descriptor, accessor_path = mkstemp(suffix=".csv")
        with fdopen(accessor_file_descriptor, "w") as temp_file:
            for row in accessor:
                temp_file.write(",".join(map(str, row)) + "\n")

        self.__params = {"-a": accessor_path, "-w": self.__windowed, "-v": int(self.__vertices[virtuality]),
                         "-d": int(duplicatable), "-p": int(palindromable), "-l": length, "-t": thread_number}

        # -a: file path with CSV format of the data structure "accessor" (coding digraph) (str).
        # -v: virtual vertex to start decoding (int).
        # -s: window size of the coding digraph (int).
        # -d: non-duplicatable mode (int).
        # -p: non-palindromable mode (int).
        # -t: thread number, only used in the communication-port method (int).
        # -l: maximum length of repaired DNA sequence (int).
        # -i: file path of receiving the raw sequencing data (str).
        # -o: file path of saving the repaired candidate results (str).
        # -r: port (regarding accelerator) to receive the raw sequencing data (int).
        # -s: port (regarding accelerator) to send the repaired candidate results, sequence-by-sequence (int).

        self.__accelerator, self.__sender, self.__receiver, self.__collection = None, None, None, RankingHeap()
        self.__diversity, self.__length, self.__runtime = diversity, length, None

    def start(self,
              sequence_paths: Optional[tuple] = None,
              communication_ports: Optional[tuple] = None,
              latency: int = 0):
        """
        Start the pipeline.

        :param sequence_paths: Tuple of file paths (raw sequencing file and repaired file).
        :type sequence_paths: tuple or None

        :param communication_ports: Port to receive and send DNA sequences.
        :type communication_ports: tuple or None

        :param latency: latency seconds for initialising the accelerator.
        :type latency: int
        """
        if sequence_paths is not None:
            self.__params["-i"] = sequence_paths[0]  # file path of receiving the raw sequencing data.
            self.__params["-o"] = sequence_paths[1]  # file path of saving the repaired candidate results.

            param_shell = str(self.__params)[1:-1].translate(str.maketrans('', '', '\',:'))
            command_line = self.__acc_path + " repair " + param_shell
            runtime_info = check_output(command_line, shell=True)
            self.__runtime = float(runtime_info.decode())

        elif communication_ports is not None:
            self.__params["-r"] = communication_ports[0]  # port to receive the raw sequencing data.
            self.__params["-s"] = communication_ports[1]  # port to send the repaired candidate results.

            param_shell = str(self.__params)[1:-1].translate(str.maketrans('', '', '\',:'))
            self.__accelerator = Popen(self.__acc_path + " " + param_shell, shell=True)

            if latency > 0:
                sleep(latency)

            self.__receiver = Process(target=self.retrieve_from_port)
            self.__receiver.daemon = True
            self.__receiver.start()

            self.__sender = Process(target=self.send_by_port)
            self.__sender.daemon = True
            self.__sender.start()
        else:
            raise RuntimeError("There is no way for accelerators to obtain raw sequencing data!")

    def stop(self,
             latency: int = 5):
        """
        Stop the pipeline.

        :param latency: latency seconds before closing the pipeline.
        :type latency: int
        """
        sleep(latency)

        if self.__sender is not None:
            self.__sender.terminate()
            self.__sender.join()

        if self.__receiver is not None:
            self.__receiver.terminate()
            self.__receiver.join()

        if self.__accelerator is not None:
            self.__accelerator.terminate()
            try:
                self.__accelerator.wait(timeout=latency)
            except TimeoutExpired:
                self.__accelerator.kill()

            for process in process_iter(["pid", "name"]):
                # noinspection PyUnresolvedReferences
                if process.info["name"] == "accelerator":
                    if name == "nt":
                        # noinspection PyUnresolvedReferences
                        system("taskkill /PID " + str(process.info["pid"]) + "/F")
                    else:
                        # noinspection PyUnresolvedReferences
                        system("kill -9 " + str(process.info["pid"]))

        if path.exists(self.__params["-a"]):
            remove(self.__params["-a"])

        if "-m" in self.__params and path.exists(self.__params["-m"]):
            remove(self.__params["-m"])

    def retrieve_from_file(self,
                           diversity: int,
                           need_rank: bool = False,
                           need_order: bool = False) \
            -> Tuple[list, dict]:
        """
        Retrieve the repaired candidates from the file.

        :param diversity: retrieved number when the ranking results are needed.
        :type diversity: int

        :param need_rank: need rank results.
        :type need_rank: bool

        :param need_order: need order of retrieved data.
        :type need_order: bool

        :return: retrieved sequences and the corresponding results.
        :rtype: list, dict
        """
        with open(self.__params["-o"]) as file:
            retrieved_data = []
            for line in file.readlines():
                samples = []
                for sample in line[:-1].split(","):
                    if len(sample) > 0:
                        samples.append(sample)
                retrieved_data.append(samples)

        collection = RankingHeap()
        for repaired_sequences in retrieved_data:
            for repaired_sequence in repaired_sequences:
                collection.push(sequence=repaired_sequence)
        top_data = list(collection.pop_all(top=diversity))

        results = {"runtime": self.__runtime}

        if need_rank:
            results["rank"] = collection.get_raw()

        if need_order:
            results["order"] = retrieved_data

        return top_data, results

    def retrieve_from_port(self):
        """
        Retrieve the required candidates from the port.
        """
        with socket(AF_INET, SOCK_STREAM) as retrieved_socket:
            retrieved_socket.bind(("127.0.0.1", self.__params["-s"]))
            retrieved_socket.listen()

            while True:
                connection, _ = retrieved_socket.accept()
                with connection:
                    sequence = connection.recv(self.__length).decode()
                    self.__collection.push(sequence=sequence)

    def send_by_port(self,
                     sequence: str = None):
        with socket(AF_INET, SOCK_STREAM) as send_socket:
            send_socket.connect(("127.0.0.1", self.__params["-r"]))
            if sequence is not None:
                send_socket.sendall(sequence.encode())

    def get_collection(self):
        """
        Get the sequence collection.

        :return: sequence collection.
        :rtype: RankingHeap
        """
        return deepcopy(self.__collection)

    def clear(self):
        """
        Clear the sequence collection.
        """
        self.__collection = RankingHeap()


class FastDecoder:

    def __init__(self,
                 accessor: ndarray,
                 virtuality: int,
                 duplicatable: bool,
                 palindromable: bool,
                 length: int):
        parent_path = path.dirname(path.abspath(__file__)).replace("\\", "/")
        if name == "nt":  # Windows system
            self.__acc_path = parent_path + "/accelerator.exe"

        else:
            self.__acc_path = parent_path + "/accelerator"

        if not path.exists(self.__acc_path):
            raise RuntimeError("Due to the lack of the necessary executable file in "
                               + parent_path + "/, acceleration is not supported!")

        if duplicatable or palindromable:
            raise NotImplementedError("Non-duplicate/palindrome mode is not implemented!")

        self.__accessor = accessor
        self.__vertices = where(sum(((self.__accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0]
        self.__windowed = int(log(len(self.__accessor)) / log(4))
        self.__virtuality = int(self.__vertices[virtuality])

        accessor_file_descriptor, accessor_path = mkstemp(suffix=".csv")
        with fdopen(accessor_file_descriptor, "w") as temp_file:
            for row in accessor:
                temp_file.write(",".join(map(str, row)) + "\n")

        self.__params = {"-a": accessor_path, "-w": self.__windowed, "-v": self.__virtuality, "-l": length}

        # -a: file path with CSV format of the data structure "accessor" (coding digraph) (str).
        # -v: virtual vertex to start decoding (int).
        # -s: window size of the coding digraph (int).
        # -l: length of binary message sequence (int).
        # -i: file path of receiving the raw sequencing data (str).
        # -o: file path of saving the repaired candidate results (str).

        self.__accelerator, self.__sender, self.__receiver, self.__collection = None, None, None, RankingHeap()
        self.__length, self.__runtime = length, None

    def handle(self,
               sequence_paths: tuple) \
            -> Tuple[ndarray, float]:
        self.__params["-i"] = sequence_paths[0]  # file path of receiving the raw sequencing data.
        self.__params["-o"] = sequence_paths[1]  # file path of saving the repaired candidate results.

        param_shell = str(self.__params)[1:-1].translate(str.maketrans('', '', '\',:'))
        command_line = self.__acc_path + " decode " + param_shell
        runtime_info = check_output(command_line, shell=True)
        self.__runtime = float(runtime_info.decode())

        matrix = []
        with open(self.__params["-o"]) as file:
            for line in file.readlines():
                matrix.append([int(v) for v in line[:-1]])

        return array(matrix, dtype=uint8), self.__runtime


class Heap:

    def __init__(self,
                 ranking_type: str):
        """
        Heap to save sequence ranking information.

        :param ranking_type: data structure for ranking.
        :type ranking_type: str
        """
        self.__ranking_type = ranking_type

    def push(self,
             sequence: str):
        """
        Push raw sequence onto the heap.

        :param sequence: repaired candidate sequence.
        :type sequence: str
        """
        raise NotImplementedError

    def pop_all(self,
                top: Optional[int] = None) \
            -> list:
        """
        Pop sequences with frequency ranking.

        :param top: top number of the sequences.
        :type top: int or None

        :return: sequences.
        :rtype: list
        """
        raise NotImplementedError

    def get_raw(self) \
            -> dict:
        """
        Get raw heap data.

        :return: raw heap data.
        :rtype: dict
        """
        raise NotImplementedError

    def get_memory(self) \
            -> int:
        """
        Get memory occupy of the heap data.

        :return: memory occupy (unit is byte).
        :rtype: int
        """
        raise NotImplementedError

    def get_type(self) \
            -> str:
        """
        Get ranking type (brief information of the heap data).

        :return: brief information of the heap data.
        :rtype: str
        """
        return self.__ranking_type


class RankingHeap(Heap):

    def __init__(self):
        """
        Heap to save sequence ranking information.
        """
        ranking_type = "key(sequence)-value(frequency) & key(frequency)-value(sequence set)"
        super(RankingHeap, self).__init__(ranking_type=ranking_type)
        self.__sequence_set = {}
        self.__count_set = {}

    def push(self,
             sequence: str):
        """
        Push raw sequence onto the heap.

        :param sequence: repaired candidate sequence.
        :type sequence: str
        """
        if sequence in self.__sequence_set:
            self.__sequence_set[sequence] += 1
            count = self.__sequence_set[sequence]
        else:
            self.__sequence_set[sequence] = 1
            count = 1

        if count in self.__count_set:
            self.__count_set[count].add(sequence)
        else:
            self.__count_set[count] = {sequence}

    def pop_all(self,
                top: Optional[int] = None) \
            -> set:
        """
        Pop sequences with frequency ranking.

        :param top: top number of the sequences.
        :type top: int or None

        :return: sequences.
        :rtype: set
        """
        retrieved_sequences = set()
        if top is not None:
            for count in sorted(self.__count_set.keys(), reverse=True):
                addition = self.__count_set[count].difference(retrieved_sequences)
                if len(addition) + len(retrieved_sequences) < top:
                    retrieved_sequences.update(addition)
                else:
                    retrieved_sequences.update(list(addition)[:top - len(retrieved_sequences)])
                    break
        else:
            for count, sequences in self.__count_set.items():
                retrieved_sequences.update(sequences)

        return retrieved_sequences

    def get_raw(self) \
            -> dict:
        """
        Get raw heap data.

        :return: raw heap data.
        :rtype: dict
        """
        return deepcopy(self.__count_set)

    def get_memory(self) \
            -> int:
        """
        Get memory occupy of the heap data.

        :return: memory occupy (unit is byte).
        :rtype: int
        """
        return getsizeof(self.__sequence_set) + getsizeof(self.__count_set)


# noinspection PyTypeChecker,PyUnresolvedReferences
def create_coding_digraph(window_length: int,
                          usable_os: Optional[ndarray] = None,
                          usable_is: Optional[ndarray] = None,
                          run_length: Optional[int] = None,
                          gc_range: Optional[list] = None,
                          undesired_motifs: Optional[list] = None,
                          strongly_connected: bool = False,
                          expected_length: Optional[int] = 200,
                          graph_only: bool = False,
                          verbose: bool = False) \
        -> Union[ndarray, Tuple[ndarray, ndarray]]:
    """
    Create the coding digraph based on the predetermined constraints.

    :param window_length: window length.
    :type window_length: int

    :param usable_os: usable out-degrees for vertices.
    :type usable_os: numpy.ndarray or None

    :param usable_is: usable in-degrees for vertices.
    :type usable_is: numpy.ndarray or None

    :param run_length: maximum single-nucleotide length in the produced DNA sequences.
    :type run_length: int or None

    :param gc_range: regionalised GC content per window length in the produced DNA sequences.
    :type gc_range: list or None

    :param undesired_motifs: undesired DNA motifs in the produced DNA sequences.
    :type undesired_motifs: list or None

    :param strongly_connected: make sure the coding digraph is strongly-connected.
    :type strongly_connected: bool

    :param expected_length: expected length of produced DNA sequences, usable when strongly_connected is True.
    :type expected_length: int

    :param graph_only: return coding digraph only if required.
    :type graph_only: bool

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: coding digraph and corresponding vertices.
    :rtype: numpy.ndarray, numpy.ndarray
    """
    # initialize k-mers (as known as vertices) based on the established constraints.
    vertices = find_vertices(window_length, run_length, gc_range, undesired_motifs, verbose)

    # construct the coding digraph.
    usable_os = array([1, 2, 3, 4]) if usable_os is None else usable_os
    usable_is = array([1, 2, 3, 4]) if usable_is is None else usable_is
    accessor, vertices = connect_digraph(window_length, vertices, usable_os, usable_is, verbose)

    if strongly_connected:
        # find the best component in the established coding digraph.
        accessor, vertices = select_component(accessor, expected_length, verbose)

    out_statistics = count_out_degrees(accessor)
    if out_statistics[3] > 0 and out_statistics[3] != sum(out_statistics):
        accessor, vertices = reduce_3_to_2(accessor, 0 in usable_is, verbose)

    if graph_only:
        return accessor
    else:
        return accessor, vertices


# noinspection PyUnresolvedReferences
def find_vertices(window_length: int,
                  run_length: Optional[int] = None,
                  gc_range: Optional[list] = None,
                  undesired_motifs: Optional[list] = None,
                  verbose: bool = False) \
        -> ndarray:
    """
    Find vertices of the given window length that meet the given constraints.

    :param window_length: window length of coding digraph.
    :type window_length: int

    :param run_length: maximum single-nucleotide length in the produced DNA sequences.
    :type run_length: int or None

    :param gc_range: regionalised GC content per window length in the produced DNA sequences.
    :type gc_range: list or None

    :param undesired_motifs: undesired DNA undesired_motifs in the produced DNA sequences.
    :type undesired_motifs: list or None

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: usable vertices.
    :rtype: numpy.ndarray
    """
    # initialise the k-mers (a.k.a vertices).
    k_mers = i2m(integers=arange(4 ** window_length, dtype=int), length=window_length)

    usable, screen_info = ones(shape=(4 ** window_length,), dtype=bool), []

    # screen k-mers that do not satisfy the homopolymer run-length constraint.
    remain = sum(usable)
    if run_length is not None:
        for location in range(window_length - (run_length + 1) + 1):
            differences = k_mers[:, location: location + run_length + 1] - expand_dims(k_mers[:, location], axis=1)
            usable[where(sum(abs(differences), axis=1) == 0)] = 0
    screen_info.append(remain - sum(usable))

    # screen k-mers that do not satisfy the guanine-cytosine content constraint.
    remain = sum(usable)
    if gc_range is not None:
        gc_contents = sum(where((k_mers == 1) | (k_mers == 2), 1, 0), axis=1) / window_length
        usable[where((gc_contents < gc_range[0]) | (gc_contents > gc_range[1]))] = 0
    screen_info.append(remain - sum(usable))

    # screen k-mers that contain the undesired motif constraint.
    remain = sum(usable)
    if undesired_motifs is not None:
        for motif in undesired_motifs:
            for location in range(window_length - len(motif) + 1):
                for check in [motif, 3 - motif[::-1]]:  # check the selected motif and its complementary palindrome.
                    differences = k_mers[:, location: location + len(motif)] - expand_dims(check, axis=0)
                    usable[where(sum(abs(differences), axis=1) == 0)] = 0
    screen_info.append(remain - sum(usable))

    vertices = where(usable == 1)[0]

    if len(vertices) == 0:
        raise RuntimeError("The constraints are too strict, resulting in no vertices being generated!")

    if verbose:
        print("In the vertex screening process:")
        table = PrettyTable(["step", "+", "-"])
        table.add_row(["initialization", str(len(k_mers)), ""])
        table.add_row(["homopolymer run-length", "", str(screen_info[0])])
        table.add_row(["guanine-cytosine content", "", str(screen_info[1])])
        table.add_row(["undesired motif", "", str(screen_info[2])])
        print(table)
        print(str(len(vertices)) + " k-mers are obtained!")
        print()

    return vertices


def connect_digraph(window_length: int,
                    vertices: ndarray,
                    usable_outs: ndarray,
                    usable_ins: ndarray,
                    verbose: bool = False) \
        -> Tuple[ndarray, ndarray]:
    """
    Construct the primary coding digraph based on the de Bruijn graph mechanism.

    :param window_length: window length of coding digraph.
    :type window_length: int

    :param vertices: usable vertices.
    :type vertices: numpy.ndarray

    :param usable_outs: usable out-degrees of the saved vertices.
    :type usable_outs: numpy.ndarray

    :param usable_ins: usable in-degrees of the saved vertices.
    :type usable_ins: numpy.ndarray

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: coding digraph and its corresponding vertices.
    :rtype: numpy.ndarray, numpy.ndarray
    """
    remain = len(vertices)
    while True:
        if not any(usable_ins == 0):
            # screen vertices without in-degrees.
            in_degrees = zeros(shape=(len(vertices),), dtype=int)
            for base in range(4):
                formers = get_formers(vertices=vertices, base=base, window_length=window_length)
                in_degrees += in1d(formers, vertices)
            saved_indices = where(in_degrees > 0)[0]
            if len(saved_indices) < len(vertices):
                vertices = vertices[saved_indices]
                continue

        # screen based on the usable out-degrees.
        out_degrees = zeros(shape=(len(vertices),), dtype=int)
        for base in range(4):
            latters = get_latters(vertices=vertices, base=base, window_length=window_length)
            out_degrees += in1d(latters, vertices)
        saved_indices = where(in1d(out_degrees, usable_outs) == 1)[0]

        if len(saved_indices) < len(vertices):  # trim arcs and their corresponding vertices.
            vertices = vertices[saved_indices]
            continue

        if any(usable_outs == 1):  # detect and screen cycles in the coding digraph.
            out_1_vertices = vertices[where(out_degrees == 1)]
            coding_paths = -ones(shape=(len(out_1_vertices), window_length + 1), dtype=int)
            coding_paths[:, 0] = out_1_vertices
            for step in range(window_length):
                temp, used = -ones(shape=(len(out_1_vertices), 4), dtype=int), where(coding_paths[:, step] != -1)[0]
                for base in range(4):
                    latters = get_latters(vertices=coding_paths[used, step], base=base, window_length=window_length)
                    latters[where(in1d(latters, vertices) == 0)] = -1
                    temp[used, base] = latters
                locations = where(sum(where(temp >= 0, 1, 0), axis=1) == 1)[0]
                coding_paths[locations, step + 1] = sum(temp[locations], axis=1) + 3
            removed_vertices = []
            for coding_path in coding_paths:
                for step in range(1, window_length + 1):
                    if coding_path[step] == coding_path[0]:
                        removed_vertices += list(coding_path[: step - 1])
                        break
                    if coding_path[step] == -1:
                        break
            removed_vertices = array(list(set(removed_vertices)))
            if len(removed_vertices) > 0:
                vertices = setdiff1d(vertices, removed_vertices)
                continue

        break

    # connect vertices as a coding digraph.
    accessor = -ones(shape=(4 ** window_length, 4), dtype=int)
    for base in range(4):
        latters = get_latters(vertices=vertices, base=base, window_length=window_length)
        latters[where(in1d(latters, vertices) == 0)] = -1
        accessor[vertices, base] = latters

    if all(accessor == -1):
        raise RuntimeError("The too stringent digraph-building conditions "
                           + "resulted in the coding digraph not being generated!")

    if verbose:
        print("In the arc trimming process:")
        table = PrettyTable(["step", "+", "-"])
        table.add_row(["out-degree & cycle requirement", "", str(remain - len(vertices))])
        print(table)
        print("Now we generate a coding digraph with " + str(len(vertices)) + " vertices.")
        table = PrettyTable(["out-degree", "vertex number"])
        counter = Counter(sum(where(accessor[vertices] >= 0, 1, 0), axis=1))
        for out_degree in range(1, 4):
            if out_degree in counter:
                # noinspection PyTypeChecker
                table.add_row([str(out_degree), str(counter[out_degree])])
        print(table)
        print()

    return accessor, vertices


# noinspection PyTypeChecker
def select_component(entire_accessor: ndarray,
                     expected_length: int,
                     verbose: bool = False) \
        -> Tuple[ndarray, ndarray]:
    """
    Split the established coding digraph.

    :param entire_accessor: entire coding digraph.
    :type entire_accessor: numpy.ndarray

    :param expected_length: expected expected_length of output DNA sequences.
    :type expected_length: int

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: best component in the established coding digraph and its corresponding vertices.
    :rtype: numpy.ndarray, numpy.ndarray
    """
    if all(entire_accessor == -1):
        raise RuntimeError("No valid coding digraph was generated!")

    graph, window_length = Graph(), int(log(len(entire_accessor)) / log(4))
    for vertex in where(sum(((entire_accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0]:
        for used_index in where(entire_accessor[vertex] >= 0)[0]:
            graph.add_edge(vertex, entire_accessor[vertex, used_index])

    components, records = [], []
    for component_index, new_vertices in enumerate(connected_components(graph)):
        new_accessor, new_vertices = -ones(shape=(4 ** window_length, 4), dtype=int), array(list(new_vertices))
        for nucleotide in range(4):
            latters = get_latters(vertices=new_vertices, base=nucleotide, window_length=window_length)
            latters[where(in1d(latters, new_vertices) == 0)] = -1
            new_accessor[new_vertices, nucleotide] = latters
        practical_density = mean(calculate_practical_density(accessor=new_accessor, length=expected_length))
        components.append((new_accessor, new_vertices))
        records.append(practical_density)

    choice = argmax(records)
    if verbose:
        if len(records) == 1:
            print("We have 1 component in the coding digraph, its information density is:")
        else:
            print("We have " + str(len(records)) + " components in the coding digraph, their information density are:")
        table = PrettyTable(["component index", "average information density"])
        for index, value in enumerate(records):
            table.add_row([str(index + 1), "%.3f" % value])
        print(table)
        if len(records) > 1:
            print("And we select the " + str(choice + 1) + "-th component according to the information density.")
        print()

    return components[choice]


def count_out_degrees(accessor: ndarray) \
        -> ndarray:
    """
    Count out-degree of each vertex.

    :param accessor: coding digraph.
    :type accessor: numpy.ndarray

    :return: out-degree statistics.
    :rtype: numpy.ndarray
    """
    vertices = where(sum(((accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0]

    counter, out_statistics = Counter(sum(where(accessor[vertices] >= 0, 1, 0), axis=1)), zeros(shape=(5,), dtype=int)
    for out_degree in [1, 2, 3, 4]:
        if out_degree in counter:
            # noinspection PyTypeChecker
            out_statistics[out_degree] += counter[out_degree]

    return out_statistics


def reduce_3_to_2(accessor: ndarray,
                  maintain_zero: bool,
                  verbose: bool = False) \
        -> Tuple[ndarray, ndarray]:
    """
    Reduce redundant arc from 3-out-degree to 2-out-degree in the coding digraph.

    :param accessor: coding digraph.
    :type accessor: numpy.ndarray

    :param maintain_zero: maintain in-degree is 0.
    :type maintain_zero: bool

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: trimmed coding digraph and vertices.
    :rtype: numpy.ndarray, numpy.ndarray
    """
    if verbose:
        print("Due to the inclusion of vertices with the out-degree of 3 and 1/2/4 in the coding digraph.")
        print("So vertices with the out-degree of 3 will be used as points with a degree of 2.\n")

    out_degrees = count_nonzero(accessor + 1, axis=1)
    out_3_locations = where(out_degrees == 3)[0]
    candidates = nonzero(accessor[out_3_locations] + 1)[1].reshape(len(out_3_locations), 3)

    g_locations = where(candidates == 2)[0]  # remove G-out firstly.
    accessor[out_3_locations[g_locations], 2] = -1
    c_locations = setdiff1d(arange(len(out_3_locations)), g_locations)  # remove C-out firstly.
    accessor[out_3_locations[c_locations], 1] = -1

    if verbose:
        print(str(len(g_locations)) + " G-out arcs and " + str(len(c_locations)) + " C-out arcs are trimmed!")

    if not maintain_zero:
        total = 0
        while True:
            in_degrees = zeros(shape=(len(accessor),), dtype=int)  # used in the core code.
            for piece in accessor.reshape(len(accessor) // 4, 16):
                in_degrees[piece[piece >= 0]] += 1

            vertices = where(sum(((accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0]
            remove_indices = where(in_degrees[vertices] == 0)[0]
            if len(remove_indices) > 0:
                accessor[vertices[remove_indices]] = -1
            else:
                break

            total += len(remove_indices)

        if verbose:
            print(str(total) + " vertices in the coding digraph are removed to satisfy the in-degree requirement.")
            print()
            out_statistics = count_out_degrees(accessor=accessor)
            print("Finally, we have a coding digraph with " + str(sum(out_statistics)) + " vertices and")
            table = PrettyTable(["out-degree", "vertex number"])
            for out_degree, number in zip([1, 2, 3, 4], out_statistics[1:]):
                if number > 0:
                    table.add_row([out_degree, number])
            print(table)
            print()

    return accessor, where(sum(((accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0]


def create_start_finder(accessor: ndarray,
                        vertices: ndarray,
                        window_length: int,
                        contain_error: bool,
                        verbose: bool = False) \
        -> dict:
    """
    Create start position finder (can accommodate up to 1 edit error) for data retrieval.

    :param accessor: coding digraph.
    :type accessor: numpy.ndarray

    :param vertices: usable vertices.
    :type vertices: numpy.ndarray

    :param window_length: window length of coding digraph.
    :type window_length: int

    :param contain_error: consider one edit error in the window length if required.
    :type contain_error: bool

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: start position finder.
    :rtype: dict
    """
    start_finder, bases, monitor = {}, ("A", "C", "G", "T"), Monitor()
    segments = m2s(matrix=i2m(integers=vertices, length=window_length))

    if verbose:
        if contain_error:
            print("Create start position finder containing 1 edit error.")
        else:
            print("Create a error-free start position finder.")

    for process, (virtual, segment) in enumerate(zip(vertices, segments)):
        start_finder[(0, segment)] = segment

        if contain_error:
            # one substitution.
            for index in range(len(segment)):
                for base in bases:
                    if base != segment[index]:
                        new_segment = list(segment)
                        new_segment[index] = base
                        new_segment = "".join(new_segment)
                        if (1, new_segment) in start_finder:
                            start_finder[(1, new_segment)].add((segment, 0))
                        else:
                            start_finder[(1, new_segment)] = set()
                            start_finder[(1, new_segment)].add((segment, 0))

            # one insertion.
            for index in range(len(segment)):
                for base in bases:
                    new_segment = list(segment)
                    new_segment.insert(index, base)
                    new_segment = "".join(new_segment)
                    if (1, new_segment) in start_finder:
                        start_finder[(1, new_segment[:window_length])].add((segment, +1))
                    else:
                        start_finder[(1, new_segment[:window_length])] = set()
                        start_finder[(1, new_segment[:window_length])].add((segment, +1))

            # one deletion.
            for latter in accessor[virtual]:
                if latter >= 0:
                    candidate = deepcopy(segment) + bases[latter % 4]
                    for index in range(len(candidate)):
                        new_segment = list(candidate)
                        del new_segment[index]
                        new_segment = "".join(new_segment)
                        if (1, new_segment) in start_finder:
                            start_finder[(1, new_segment)].add((segment, -1))
                        else:
                            start_finder[(1, new_segment)] = set()
                            start_finder[(1, new_segment)].add((segment, -1))

        if verbose:
            monitor(process + 1, len(vertices), extra={"items": len(start_finder)})

    if verbose:
        print()

    return start_finder


def create_coexisting_map(window_length: int,
                          minimum_difference: int,
                          vertices: ndarray,
                          verbose: bool = False) \
        -> dict:
    """
    Collect coexisting vertices.

    :param window_length: window length of coding digraph.
    :type window_length: int

    :param minimum_difference: minimum difference requirement between virtual vertices.
    :type minimum_difference: int

    :param vertices: usable vertices.
    :type vertices: numpy.ndarray

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: coexisting vertex collection.
    :rtype: dict
    """
    if verbose:
        print("Collect coexisting vertices.")

    segments = i2m(integers=vertices, length=window_length)

    collections, monitor = {}, Monitor()
    for index, vertex in enumerate(vertices):
        distances = levenshtein(group=segments, sample=segments[index])
        collections[vertex] = set(vertices[where(distances >= minimum_difference)[0]])

        if verbose:
            monitor(index + 1, len(vertices), extra={"count": len(collections[vertex])})

    if verbose:
        print()

    return collections


def create_palindrome_map(window_length: int,
                          vertices: ndarray,
                          verbose: bool = False) \
        -> ndarray:
    """
    Create the palindrome map for non-palindrome production mode.

    :param window_length: window length of coding digraph.
    :type window_length: int

    :param vertices: usable vertices.
    :type vertices: numpy.ndarray

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: palindrome map.
    :rtype: numpy.ndarray
    """
    if verbose:
        print("Create the palindrome map for non-palindrome mode for produced DNA sequences.")

    palindrome_map, monitor = -ones(shape=(4 ** window_length,), dtype=int), Monitor()
    reverse_vertices = m2i(3 - i2m(integers=vertices, length=window_length)[:, ::-1])

    for index, (forward_vertex, reverse_vertex) in enumerate(zip(vertices, reverse_vertices)):
        if reverse_vertex in vertices:
            palindrome_map[forward_vertex] = reverse_vertex

        if verbose:
            monitor(index + 1, len(vertices))

    if verbose:
        print()

    return palindrome_map
