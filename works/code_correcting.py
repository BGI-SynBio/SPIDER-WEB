from datetime import datetime
from numpy import ndarray, random, array, arange, zeros, array_split, stack, concatenate, frombuffer
from numpy import packbits, unpackbits, cumsum, sqrt, log, floor, abs, min, max, sum, log2, any, ceil, where, uint8
from reedsolo import RSCodec, ReedSolomonError
from typing import Tuple, List, Generator, Optional

from sbw import Monitor, SpiderWeb, i2m, m2i


def introduce_errors(right_sequence: str,
                     errors: int) \
        -> str:
    """
    Introduce random errors to the inputted DNA sequence.

    :param right_sequence: right DNA sequence.
    :type right_sequence: str

    :param errors: error number.
    :type errors: int

    :return: wrong DNA sequence.
    :rtype: str
    """
    if errors == 0:
        return right_sequence

    total_length, nucleotides = len(right_sequence), ["A", "C", "G", "T"]

    while True:
        wrong_sequence, locations = list(right_sequence), []  # introduce error under most dense.
        for _ in range(errors):
            # noinspection PyTypeChecker
            wrong_type, location, success = random.randint(0, 3), None, False
            candidates = arange(1, len(wrong_sequence) - 1, dtype=int)
            # noinspection PyArgumentList
            random.shuffle(candidates)
            for location in candidates:
                # uniform distribution detection
                # noinspection PyUnresolvedReferences
                if len(locations) == 0 or min(abs(array(locations) - location)) > total_length / (errors + 2):
                    success = True
                    locations.append(location)
                    break

            if not success:
                wrong_sequence, locations, wrong_type = list(right_sequence), [], -1
                break

            if wrong_type == 0:  # substitution
                # noinspection PyTypeChecker
                nucleotide = wrong_sequence[location]
                # noinspection PyArgumentList,PyTypeChecker
                wrong_sequence[location] = random.choice(list(filter(lambda n: n != nucleotide, nucleotides)))
            elif wrong_type == 1:  # insertion
                # noinspection PyArgumentList,PyTypeChecker
                wrong_sequence.insert(location, random.choice(nucleotides))
            else:  # deletion
                # noinspection PyTypeChecker
                del wrong_sequence[location]

        wrong_sequence = "".join(wrong_sequence)

        if wrong_sequence != right_sequence:
            return wrong_sequence


def practice(right_sequences: list,
             errors: int,
             reads: int = 1,
             seed: Optional[int] = None,
             verbose: bool = False) \
        -> List[str]:
    """
    Introduce random errors to all right DNA sequences.

    :param right_sequences: right DNA sequences.
    :type right_sequences: list

    :param errors: error number.
    :type errors: int

    :param reads: error introduction times.
    :type reads: int

    :param seed: random seed for replication.
    :type seed: int or None

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: wrong DNA sequences.
    :rtype: list
    """
    if seed is not None:
        random.seed(seed)

    monitor, wrong_sequences, process = Monitor(), [], 0

    for _ in range(reads):
        for index, right_sequence in enumerate(right_sequences):
            wrong_sequences.append(introduce_errors(right_sequence=right_sequence, errors=errors))
            if verbose:
                monitor(process + 1, len(right_sequences) * reads)
            process += 1

    if seed is not None:
        random.seed()

    return wrong_sequences


def evaluate_single_repair(method: SpiderWeb,
                           virtuality: int,
                           duplicatable: bool,
                           palindromable: bool,
                           right_sequences: list,
                           wrong_sequences: list,
                           verbose: bool = False) \
        -> Tuple[list, float]:
    """
    Evaluate repair performance of single DNA sequence.

    :param method: SPIDER-WEB coding digraph.
    :type method: sbw.sbw.SpiderWeb

    :param virtuality: virtual vertex to start decoding (consistent with the encoding process).
    :type virtuality: int

    :param duplicatable: non-duplicate mode for DNA sequences if exists.
    :type duplicatable: bool

    :param palindromable: non-palindrome mode for DNA sequences if exists.
    :type palindromable: bool

    :param right_sequences: right DNA sequences.
    :type right_sequences: list

    :param wrong_sequences: wrong DNA sequences.
    :type wrong_sequences: list

    :param verbose: verbosity mode (runtime calculation is not applicable when verbose = True).
    :type verbose: bool

    :return: repaired data, and runtime if available.
    :rtype: list, float
    """
    if verbose:
        records, length, monitor = [], len(right_sequences[0]), Monitor()
        for process, wrong_sequence in enumerate(wrong_sequences):
            records.append(method.repair(sequence=wrong_sequence, virtuality=virtuality, length=length,
                                         duplicatable=duplicatable, palindromable=palindromable))
            monitor(process + 1, len(wrong_sequences))
        runtime = -1.0
    else:
        records, length, last_time = [], len(right_sequences[0]), datetime.now()
        for process, wrong_sequence in enumerate(wrong_sequences):
            records.append(method.repair(sequence=wrong_sequence, virtuality=virtuality, length=length,
                                         duplicatable=duplicatable, palindromable=palindromable))
        runtime = (datetime.now() - last_time).total_seconds()

    return records, runtime


def evaluate_single_repeats(method: SpiderWeb,
                            virtuality: int,
                            duplicatable: bool,
                            palindromable: bool,
                            right_sequences: list,
                            errors: int,
                            seed: Optional[int] = None,
                            verbose: bool = False) \
        -> ndarray:
    """
    Evaluate the minimum reads to make highest-frequency sequence and right sequence equivalent.

    :param method: SPIDER-WEB coding digraph.
    :type method: sbw.sbw.SpiderWeb

    :param virtuality: virtual vertex to start decoding (consistent with the encoding log).
    :type virtuality: int

    :param right_sequences: right DNA sequences.
    :type right_sequences: list

    :param duplicatable: non-duplicate mode for DNA sequences if exists.
    :type duplicatable: bool

    :param palindromable: non-palindrome mode for DNA sequences if exists.
    :type palindromable: bool

    :param errors: error number.
    :type errors: int

    :param seed: random seed for replication.
    :type seed: int or None

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: minimum reads number results.
    :rtype: numpy.ndarray
    """
    if seed is not None:
        random.seed(seed)

    records, monitor = [], Monitor()
    for process, right_sequence in enumerate(right_sequences):
        length, reads, result_set = len(right_sequence), 0, {}

        while True:
            reads += 1
            wrong_sequence = introduce_errors(right_sequence=right_sequence, errors=errors)
            repaired_sequences = method.repair(sequence=wrong_sequence, virtuality=virtuality, length=length,
                                               duplicatable=duplicatable, palindromable=palindromable)[0]

            for repaired_sequence in repaired_sequences:
                if repaired_sequence in result_set:
                    result_set[repaired_sequence] += 1
                else:
                    result_set[repaired_sequence] = 1

            saved_sequences, counts = [], []
            for sequence, count in result_set.items():
                saved_sequences.append(sequence)
                counts.append(count)
            counts = array(counts)

            if len(counts) > 0:
                maximum_count = max(counts)
                used_indices = where(counts == maximum_count)[0]
                # only one candidate has the highest frequency.
                if len(used_indices) > 1:
                    continue
                # has a gap in the obtained number between the best sequence and other candidates.
                if any(counts[counts != maximum_count]) >= maximum_count - 1:
                    continue
                if saved_sequences[used_indices[0]] != right_sequence:
                    continue
                records.append(reads)
                break

        if verbose:
            monitor(process + 1, len(right_sequences))

    if seed is not None:
        random.seed()

    return array(records)


def evaluate_astar_search(accessor: ndarray,
                          dna_sequences: list,
                          virtuality: int,
                          maximum_error: int,
                          initial_score: float = 0.0,
                          correct_penalty: float = -0.035,
                          insert_penalty: float = 1.0,
                          delete_penalty: float = 1.0,
                          mutate_penalty: float = 1.0,
                          seed: Optional[int] = None) \
        -> dict:
    """
    Evaluate A-star search for repairing DNA sequences.

    :param accessor:
    :type accessor: list

    :param dna_sequences: expected or correct DNA sequences.
    :type dna_sequences: list

    :param virtuality: virtual vertex to start encoding.
    :type virtuality: int

    :param maximum_error: maximum number of edit error can be introduced.
    :type maximum_error: int

    :param initial_score: initial score starting the A* search.
    :type initial_score: float

    :param correct_penalty: penalty when correction.
    :type correct_penalty: float

    :param insert_penalty: penalty when insertion.
    :type insert_penalty: float

    :param delete_penalty: penalty when deletion.
    :type delete_penalty: float

    :param mutate_penalty: penalty when mutation.
    :type mutate_penalty: float

    :param seed: random seed if needed.
    :type seed: int or None

    :return: vertex access records.
    :rtype: dict
    """
    bases = ("A", "C", "G", "T")

    class HypothesisNode:

        def __init__(self,
                     vertex: int,
                     segment: str,
                     location: int,
                     score: float):
            """
            Initialize a hypothesis node.

            :param vertex: vertex index in the coding digraph.
            :type vertex: int

            :param segment: executed DNA segment.
            :type segment: str

            :param location: location in the raw DNA sequence.
            :type location: int

            :param score: score of the hypothesis.
            :type score: float
            """
            self.vertex, self.segment, self.location, self.score = vertex, segment, location, score

        def next(self,
                 sequence: str) \
                -> Tuple[list, int]:
            """
            Generate follow-up hypothesis nodes.

            :param sequence: raw DNA sequence.
            :type sequence: str

            :return: follow-up hypothesis nodes and local vertex access time.
            :rtype: list, int
            """
            available_pairs = []
            for value in accessor[self.vertex]:
                if value >= 0:
                    available_pairs.append((bases[value % 4], value))

            try:
                nucleotide, follow_nodes, vertex_count = sequence[self.location], [], 0
                for available_nucleotide, available_value in available_pairs:
                    if available_nucleotide == nucleotide:
                        return [HypothesisNode(vertex=available_value, segment=self.segment + nucleotide,
                                               location=self.location + 1, score=self.score + correct_penalty)], 1

                # assume that current nucleotide is mutated.
                for available_nucleotide, available_value in available_pairs:
                    node = HypothesisNode(vertex=available_value, segment=self.segment + available_nucleotide,
                                          location=self.location + 1, score=self.score + mutate_penalty)
                    follow_nodes.append(node)
                vertex_count += len(available_pairs)

                # assume that current nucleotide is inserted, the (i + 1)-th nucleotide is i-th nucleotide.
                node = HypothesisNode(vertex=self.vertex, segment=self.segment,
                                      location=self.location + 1, score=self.score + insert_penalty)
                follow_nodes.append(node)
                vertex_count += 1

                # assume that current nucleotide is deleted.
                for available_nucleotide, available_value in available_pairs:
                    node = HypothesisNode(vertex=available_value, segment=self.segment + available_nucleotide,
                                          location=self.location, score=self.score + delete_penalty)
                    follow_nodes.append(node)
                vertex_count += len(available_pairs)

                return follow_nodes, vertex_count

            except IndexError:
                return [], 0

    def search(sequence: str,
               expected_length: int) \
            -> Tuple[str, int]:
        """
        Search "correct" DNA sequence by A star search (score priority).

        :param sequence: raw DNA sequence.
        :type sequence: str

        :param expected_length: expected length of DNA sequence.
        :type expected_length: int

        :return: retrieved DNA sequence and vertex access time.
        :rtype: str, int
        """
        access_time, vertex = 0, where(sum(((accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0][virtuality]
        heap = {"n": [HypothesisNode(vertex=vertex, segment="", location=0, score=initial_score)], "s": [initial_score]}

        while True:  # repair by A star search (score priority).
            # noinspection PyTypeChecker
            chuck_indices, chuck_score = where(array(heap["s"]) == min(heap["s"]))[0], min(heap["s"])
            for chuck_index in chuck_indices:
                # noinspection PyTypeChecker
                heap["s"][chuck_index] = len(sequence) + 1  # set the chuck vertex to inaccessible.
                follow_nodes, count = heap["n"][chuck_index].next(sequence=sequence)

                access_time += count
                for node in follow_nodes:
                    # the first segment of hypotheses to access the whole DNA sequence wins.
                    if len(node.segment) == expected_length:
                        return node.segment, access_time
                    heap["n"].append(node)
                    heap["s"].append(node.score)

    records = {}
    for errors in range(0, maximum_error + 1):
        wrong_sequences, record = practice(right_sequences=dna_sequences, errors=errors, seed=seed), []
        monitor, counts = Monitor(), []
        for sequence_index, wrong_sequence in enumerate(wrong_sequences):
            wrong_sequence += "AAA"  # avoid the tail issue.
            expected_sequence = dna_sequences[sequence_index % len(dna_sequences)]
            obtained_sequence, vertex_access = search(sequence=wrong_sequence, expected_length=len(expected_sequence))
            counts.append(vertex_access)
            record.append((obtained_sequence == expected_sequence, vertex_access))
        records[errors] = record

    return records


class OuterCode:

    def __init__(self,
                 method: str):
        if method not in ["xor", "reed-solomon", "fountain"]:
            raise ValueError("Invalid method!")

        self.method, self.redundancy, self.parameters = method, None, None

    def set_params(self,
                   redundancy: float,
                   parameters: Optional[dict] = None):
        if self.method == "fountain":
            if parameters is None or "delta" not in parameters or "cdist" not in parameters:
                raise ValueError("Invalid parameters!")

        self.redundancy, self.parameters = redundancy, parameters

    def insert(self,
               binary_messages: ndarray,
               address_length: int,
               skipping_range: int = 0,
               block_size: Optional[int] = None) \
            -> ndarray:
        redundant_number = int(len(binary_messages) * self.redundancy + 0.5)
        if self.method == "xor":
            total_number = len(binary_messages) + redundant_number + skipping_range
            if total_number > 2 ** address_length:
                raise ValueError("Address length needs to be larger than %d!" % ceil(log2(total_number)))

            redundant_messages = []
            for group in array_split(binary_messages, redundant_number):
                redundant_messages.append((sum(group, axis=0) % 2))

            payloads = concatenate((binary_messages, stack(redundant_messages)), axis=0)

            address_indices = arange(len(payloads))
            address_indices[len(binary_messages):] += skipping_range
            addresses = i2m(integers=address_indices, length=address_length, base_size=2)

        elif self.method == "reed-solomon":
            total_number = len(binary_messages) + redundant_number + skipping_range
            if total_number > 2 ** address_length:
                raise ValueError("Address length needs to be larger than %d!" % ceil(log2(total_number)))

            payloads = []
            if total_number < 255:
                rs_code = RSCodec(nsym=redundant_number)
                for byte_array in packbits(binary_messages.astype(uint8), axis=1).T:
                    byte_array = frombuffer(rs_code.encode(byte_array), dtype=uint8)
                    # noinspection PyTypeChecker
                    payloads.append(unpackbits(byte_array))

                payloads = stack(payloads).T
            else:
                if block_size is None:
                    raise ValueError("The parameter \"block_size\" needs to be specified!")
                if len(binary_messages) // block_size * block_size < len(binary_messages):
                    raise ValueError("The parameter \"block_size\"  must be a divisor of the message number!")

                block_number = len(binary_messages) // block_size

                if redundant_number // block_number * block_number < redundant_number:
                    raise ValueError("The block number (%d) must be a divisor of the redundant number (%d)!" %
                                     (block_number, redundant_number))

                sub_redundant_number = redundant_number // block_number
                rs_code, redundant_messages = RSCodec(nsym=sub_redundant_number), []
                for group in array_split(binary_messages, block_number):
                    sub_redundant_messages = []
                    for byte_array in packbits(group.astype(uint8), axis=1).T:
                        byte_array = frombuffer(rs_code.encode(byte_array), dtype=uint8)
                        sub_redundant_messages.append(byte_array[len(group):])
                    # noinspection PyTypeChecker
                    redundant_messages.append(unpackbits(stack(sub_redundant_messages).T, axis=1))
                payloads = concatenate((binary_messages, concatenate(redundant_messages, axis=0)), axis=0)

            address_indices = arange(len(payloads))
            address_indices[len(binary_messages):] += skipping_range
            addresses = i2m(integers=address_indices, length=address_length, base_size=2)

        else:  # self.method == "fountain":
            cdf = self.__rsd(len(binary_messages), self.parameters["delta"], self.parameters["cdist"])
            generator, used_seeds, scale = self.__lfsr(), [], 2 ** address_length - 1

            payloads = []
            while len(payloads) < len(binary_messages) + redundant_number:
                used_seed = next(generator)
                used_seed %= scale

                if used_seed in used_seeds:
                    continue

                used_seeds.append(used_seed)

                selected_indices = self.__combine(cdf=cdf, number=len(binary_messages), used_seed=used_seed)
                combined_message = sum(binary_messages[selected_indices], axis=0) % 2

                payloads.append(combined_message)

            addresses, payloads = i2m(integers=array(used_seeds), length=address_length, base_size=2), stack(payloads)

        return concatenate((addresses, payloads), axis=1)

    def repair(self,
               binary_messages: ndarray,
               address_length: int,
               original_number: int,
               skipping_range: int = 0,
               block_size: Optional[int] = None) \
            -> ndarray:
        addresses, payloads = binary_messages[:, :address_length], binary_messages[:, address_length:]
        address_indices, message_length = m2i(matrix=addresses, base_size=2), len(payloads[0])
        redundant_number = int(original_number * self.redundancy + 0.5)

        if self.method == "xor":
            retrieved_messages = [None for _ in range(original_number)]
            expected_indices, group_number = arange(original_number), int(original_number * self.redundancy)
            for index, grouped_indices in enumerate(array_split(expected_indices, group_number)):
                redundant_index = original_number + skipping_range + index
                if redundant_index in address_indices:
                    redundant_message, loss = payloads[where(address_indices == redundant_index)[0]][0], []
                    for message_index in grouped_indices:
                        if message_index not in address_indices:
                            loss.append(message_index)
                        else:
                            retrieved_messages[message_index] = payloads[where(address_indices == message_index)[0]][0]
                    if len(loss) == 0:
                        continue
                    elif len(loss) == 1:  # can be repaired by XOR operation.
                        loss_index, message = loss[0], redundant_message
                        for message_index in grouped_indices:
                            if retrieved_messages[message_index] is not None:
                                message += retrieved_messages[message_index]
                        retrieved_messages[loss_index] = message % 2
                    else:
                        for loss_index in loss:
                            retrieved_messages[loss_index] = zeros(shape=(message_length,), dtype=int)

                else:
                    for message_index in grouped_indices:
                        if message_index in address_indices:
                            retrieved_messages[message_index] = payloads[where(address_indices == message_index)[0]][0]
                        else:
                            retrieved_messages[message_index] = zeros(shape=(message_length,), dtype=int)

            # noinspection PyTypeChecker
            retrieved_messages = stack(retrieved_messages)

        elif self.method == "reed-solomon":
            if block_size is None:
                raise ValueError("The parameter \"block_size\" needs to be specified!")
            if original_number // block_size * block_size < original_number:
                raise ValueError("The parameter \"block_size\"  must be a divisor of the message number!")

            block_number = original_number // block_size

            if redundant_number // block_number * block_number < redundant_number:
                raise ValueError("The block number (%d) must be a divisor of the redundant number (%d)!" %
                                 (block_number, redundant_number))

            retrieved_messages, rs_code = [], RSCodec(nsym=redundant_number // block_number)
            for o_indices, r_indices in zip(array_split(arange(original_number), block_number),
                                            array_split(arange(redundant_number) + original_number + skipping_range,
                                                        block_number)):
                sub_reordered_messages, positions, current = [], [], 0
                for o_index in o_indices:
                    if o_index in address_indices:
                        sub_reordered_messages.append(payloads[where(address_indices == o_index)[0]][0])
                    else:
                        sub_reordered_messages.append(zeros(shape=(message_length,), dtype=int))
                        positions.append(current)
                    current += 1

                for r_index in r_indices:
                    if r_index in address_indices:
                        sub_reordered_messages.append(payloads[where(address_indices == r_index)[0]][0])
                    else:
                        sub_reordered_messages.append(zeros(shape=(message_length,), dtype=int))
                        positions.append(current)
                    current += 1

                if 0 < len(positions) <= len(r_indices):
                    sub_retrieved_messages = []
                    for byte_array in packbits(stack(sub_reordered_messages).astype(uint8), axis=1).T:
                        try:
                            byte_array = frombuffer(rs_code.decode(byte_array, erase_pos=positions)[0], dtype=uint8)
                        except ReedSolomonError:
                            byte_array = byte_array[:block_size]

                        sub_retrieved_messages.append(byte_array)

                    # noinspection PyTypeChecker
                    sub_retrieved_messages = unpackbits(stack(sub_retrieved_messages).T, axis=1)
                else:
                    sub_retrieved_messages = stack(sub_reordered_messages[:block_size])

                retrieved_messages.append(sub_retrieved_messages)

            retrieved_messages = concatenate(retrieved_messages, axis=0)

        else:
            cdf, droplets = self.__rsd(original_number, self.parameters["delta"], self.parameters["cdist"]), []
            for used_seed, payload in zip(address_indices, payloads):
                selected_indices = self.__combine(cdf=cdf, number=original_number, used_seed=used_seed)
                droplets.append([set(selected_indices), payload])

            messages, updated = [None for _ in range(original_number)], True
            while updated:
                new_droplets, updated = [], False
                for indices_set, payload in droplets:
                    unsolved = [index for index in indices_set if messages[index] is None]

                    if len(unsolved) == 0:  # nothing to do
                        continue

                    elif len(unsolved) == 1:  # Only one unknown: can be solved.
                        # noinspection PyTypeChecker
                        known_payload = sum(messages[index] for index in indices_set if messages[index] is not None) % 2
                        solved_payload = (payload + known_payload) % 2
                        messages[unsolved[0]] = solved_payload
                        updated = True

                    else:  # still multiple unknowns, keep it.
                        new_droplets.append((indices_set, payload))

                droplets = new_droplets

            retrieved_messages = []
            for message in messages:
                if message is not None:
                    retrieved_messages.append(message)
                else:
                    retrieved_messages.append(zeros(shape=(message_length,), dtype=int))
            retrieved_messages = stack(retrieved_messages)

        return retrieved_messages

    @staticmethod
    def __combine(cdf,
                  number: int,
                  used_seed: int) \
            -> ndarray:
        """
        Obtain the combined indices in the binary messages.

        :param number: number of binary messages.
        :type number: int

        :param used_seed: used random seed.
        :type used_seed: int

        :return: list of indices for the XOR operation.
        :rtype: numpy.ndarray
        """
        rng = random.default_rng(used_seed)
        probability = rng.random()

        for degree, value in enumerate(cdf):
            if value > probability:
                return rng.choice(number, degree + 1, replace=False)

        return rng.choice(number, len(cdf), replace=False)

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
