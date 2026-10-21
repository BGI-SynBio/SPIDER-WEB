from numpy import ndarray, array, zeros, ones, zeros_like, arange, random, concatenate, expand_dims, hstack
from numpy import count_nonzero, nonzero, sum, min, max, log2, ceil, all, where, uint8
from typing import List, Tuple, Optional

from sbw.spiderweb import count_out_degrees
from sbw.underprop import Monitor, m2s, i2m


def merge(virtual_vertices: ndarray,
          payloads: list,
          windowed: int,
          former_primer: str,
          middle_primer: str,
          latter_primer: str,
          verbose: bool = False) \
        -> List[str]:
    """
    Merge primers, segments of virtual vertices, and payloads.

    :param virtual_vertices: virtual vertices used for each payload.
    :type virtual_vertices: numpy.ndarray

    :param payloads: payload segments (one payload has one corresponding virtual vertex).
    :type payloads: list

    :param windowed: window size for payloads.
    :type windowed: int

    :param former_primer: forward primer before virtual segments if required.
    :type former_primer: str

    :param middle_primer: identification marker between virtual segments and payloads if required.
    :type middle_primer: str

    :param latter_primer: reverse primer after payload segments if required.
    :type latter_primer: str

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: complete synthesised DNA sequences.
    :rtype: List[str]

    .. note::
        To ensure decoding efficiency, we suggest that
        (1) the middle primer can set as 10;
        (2) the last base of former primer is inconsistent with the first base of virtual segment;
        (3) the first base of latter primer is inconsistent with the last base of payload segment;
        (4) there are no sub-strings of virtual segment and middle primer in former primer;
        (4) here are no 10-mer of payload segment in latter primer.
    """
    sequences, virtual_segments, monitor = [], m2s(matrix=i2m(integers=virtual_vertices, length=windowed)), Monitor()
    for index, (virtual_segment, payload_segment) in enumerate(zip(virtual_segments, payloads)):
        sequences.append(former_primer + virtual_segment + middle_primer + payload_segment + latter_primer)

        if verbose:
            monitor(index + 1, len(payloads))

    return sequences


def convert_messages(binaries: ndarray,
                     payload_length: int,
                     index_length: Optional[int] = None) \
        -> ndarray:
    """
    Convert one binary array to binary messages.

    :param binaries: the given binaries.
    :type binaries: numpy.ndarray

    :param payload_length: length of payload.
    :type payload_length: int

    :param index_length: length of index region.
    :type index_length: int or None

    :return: binary messages.
    """
    if len(binaries) % payload_length != 0:
        binaries = concatenate((binaries, zeros(shape=(payload_length - len(binaries) % payload_length,))))
    if payload_length == len(binaries):
        matrix = expand_dims(binaries, axis=0)
    else:
        matrix = binaries.reshape(len(binaries) // payload_length, payload_length)

    message_number = len(matrix)

    minimum_index_length = int(ceil(log2(message_number)))
    if index_length is not None:
        if index_length < minimum_index_length:
            raise ValueError("The given index length cannot be less than the minimum index length!")
        indices = i2m(integers=arange(message_number), length=index_length, base_size=2)
    else:
        indices = i2m(integers=arange(message_number), length=minimum_index_length, base_size=2)

    return hstack((indices, matrix)).astype(uint8)


# noinspection PyTypeChecker,PyUnresolvedReferences
def estimate_payload_length(binaries: ndarray,
                            accessor: ndarray,
                            virtuality: int,
                            sequence_length: Optional[int] = None,
                            start_length: Optional[int] = None,
                            index_length: Optional[int] = None,
                            seed: Optional[int] = None,
                            verbose: bool = False) \
        -> Tuple[int, int]:
    """
    Estimate the payload length and its corresponding message number.

    :param binaries: the given binaries.
    :type binaries: numpy.ndarray

    :param accessor: the established coding digraph.
    :type accessor: numpy.ndarray

    :param virtuality: virtual vertex to start encoding.
    :type virtuality: int

    :param sequence_length: the expected sequence length.
    :type sequence_length: int or None

    :param start_length: starting detection length of the payload.
    :type start_length: int or None

    :param index_length: the given index length.
    :type index_length: int or None

    :param seed: seed for selecting remaining segments.
    :type seed: int or None

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: the payload length and the message number.
    :rtype: tuple
    """
    vertices = where(sum(((accessor + 1).astype(bool)), axis=1).astype(bool) == 1)[0]
    payload_length = sequence_length * 2 if start_length is None else start_length

    out_statistics = count_out_degrees(accessor=accessor)
    if out_statistics[3] > 0 and out_statistics[3] != sum(out_statistics):
        raise ValueError("If a graph contains nodes with out-degree of 3 and others (i.e., 1, 2, and 4), "
                         + "it will cause encoding ambiguity")
    all_3 = (out_statistics[3] == sum(out_statistics))

    monitor = Monitor()
    while True:
        binary_messages = convert_messages(binaries=binaries, payload_length=payload_length, index_length=index_length)
        message_length, message_number = len(binary_messages[0]), len(binary_messages)
        used_vertices = ones(shape=(message_number,), dtype=int) * vertices[virtuality]

        if seed is not None:
            random.seed(seed)

        practical_sequence_length = 0
        if not all_3:
            markers = zeros(shape=(message_number,), dtype=int)
            while True:
                if all(markers >= message_length):
                    break

                action = arange(message_number, dtype=int)
                out_degrees = count_nonzero(accessor[used_vertices[action]] + 1, axis=1)

                radix_choices = where(out_degrees == 0)[0]
                if len(radix_choices) > 0:
                    raise ValueError("Vertex " + str(used_vertices[action[radix_choices]])
                                     + " have no outgoing vertices,"
                                     + " indicating that the coding digraph or the start vertex is wrong!")

                # check vertices with out-degree of 1.
                radix_choices = where(out_degrees == 1)[0]
                if len(radix_choices) > 0:
                    action_choices = action[radix_choices]
                    used_vertices[action_choices] = sum(accessor[used_vertices[action_choices]], axis=1) + 3

                # check vertices with out-degree of 2.
                radix_choices = where(out_degrees == 2)[0]
                if len(radix_choices) > 0:
                    action_choices = action[radix_choices]
                    location_choices = markers[action_choices]
                    remained = where(location_choices < message_length)[0]
                    index_choices = random.randint(0, 2, size=(len(radix_choices),), dtype=int)
                    index_choices[remained] = binary_messages[action_choices[remained], location_choices[remained]]
                    candidates = nonzero(accessor[used_vertices[action_choices]] + 1)[1]
                    candidates = candidates.reshape(len(radix_choices), 2)
                    follow_ups = candidates[arange(len(candidates)), index_choices]
                    used_vertices[action_choices] = accessor[used_vertices[action_choices], follow_ups]
                    action_choices = action[radix_choices]
                    markers[action_choices] += 1

                # check vertices with out-degree of 3.
                radix_choices = where(out_degrees == 3)[0]
                if len(radix_choices) > 0:
                    raise ValueError("Vertex " + str(used_vertices[action[radix_choices]])
                                     + " have 3 outgoing vertices,"
                                     + " which are removed when creating the coding digraph!")

                # check vertices with out-degree of 4.
                radix_choices = where(out_degrees == 4)[0]
                if len(radix_choices) > 0:
                    action_choices = action[radix_choices]
                    location_choices = markers[action_choices]
                    remained_1 = where(location_choices + 0 < message_length)[0]
                    remained_2 = where(location_choices + 1 < message_length)[0]
                    index_choices = random.randint(0, 2, size=(len(radix_choices),), dtype=int)
                    index_choices[remained_1] = binary_messages[action_choices, location_choices[remained_1]] * 2
                    if len(remained_2) > 0:
                        index_choices[remained_2] += binary_messages[action_choices, location_choices[remained_2] + 1]
                    candidates = nonzero(accessor[used_vertices[action_choices]] + 1)[1]
                    candidates = candidates.reshape(len(radix_choices), 4)
                    follow_ups = candidates[arange(len(candidates)), index_choices]
                    used_vertices[action_choices] = accessor[used_vertices[action_choices], follow_ups]
                    markers[action_choices] += 2

                practical_sequence_length += 1

                if verbose:
                    monitor(min(markers), message_length,
                            extra={"sequence length": practical_sequence_length})

        else:
            while True:
                processes = message_length - sum(binary_messages, axis=1)
                if all(processes == message_number):
                    break

                action = arange(message_number, dtype=int)
                quotients, reminders = zeros_like(binary_messages), binary_messages[:, 0]
                for location in range(1, message_length):
                    values = reminders * 2 + binary_messages[:, location]
                    group_1, group_2 = where(values >= 3)[0], where(values < 3)[0]
                    quotients[group_1, location], reminders[group_1] = 1, values[group_1] - 3
                    reminders[group_2] = values[group_2]

                candidates = nonzero(accessor[used_vertices[action]] + 1)[1]
                candidates = candidates.reshape(len(action), 3)
                follow_ups = candidates[arange(len(candidates)), reminders[action]]
                used_vertices[action] = accessor[used_vertices[action], follow_ups]

                practical_sequence_length += 1

                if verbose:
                    monitor(message_length - max(processes), message_length,
                            extra={"sequence length": practical_sequence_length})

        if seed is not None:
            random.seed()

        if practical_sequence_length <= sequence_length:
            return payload_length, message_number

        payload_length -= 1


def detect_head(segment: str,
                nucleotide: str,
                length: int) \
        -> bool:
    """
    Check whether the specified nucleotide appears within the first `length` nucleotides of the given segment.

    :param segment: given DNA segment.
    :type segment: str

    :param nucleotide: specified nucleotide, i.e., "A", "C", "G", or "T".
    :type nucleotide: str

    :param length: number of nucleotides from the beginning of the segment to check.
    :type length: int

    :return: True if the nucleotide is found within the first `length` nucleotides, False otherwise.
    :rtype: bool
    """
    if nucleotide in segment[:length]:
        return True

    return False


def detect_tail(segment: str,
                nucleotide: str,
                length: int) \
        -> bool:
    """
    Check whether the specified nucleotide appears within the last `length` nucleotides of the input segment.

    :param segment: given DNA segment.
    :type segment: str

    :param nucleotide: specified nucleotide, i.e., "A", "C", "G", or "T".
    :type nucleotide: str

    :param length: number of nucleotides from the end of the segment to check.
    :type length: int

    :return: True if the nucleotide is found within the last `length` nucleotides, False otherwise.
    :rtype: bool
    """
    if nucleotide in segment[-length:]:
        return True

    return False


def detect_duplicatable(segment: str,
                        window_size: int) \
        -> bool:
    """
    Detect whether there are any duplicated k-mers of a given window size within the given segment.

    :param segment: given DNA segment.
    :type segment: str

    :param window_size: observation window size used to extract k-mers.
    :type window_size: int

    :return: True if there are any duplicated k-mers of a given window size, False otherwise.
    :rtype: bool
    """
    k_mers = []
    for location in range(len(segment) - (window_size - 1)):
        k_mers.append(segment[location: location + window_size])

    if len(k_mers) > len(set(k_mers)):  # find one or more duplicates.
        return True

    return False


def detect_palindromable(segment: str,
                         window_size: int) \
        -> bool:
    """
    Detect whether there are any palindromic k-mers of a given window size within the given segment.

    :param segment: given DNA segment.
    :type segment: str

    :param window_size: observation window size used to extract k-mers.
    :type window_size: int

    :return: True if any k-mer of the given window size appears in the reverse complement, False otherwise.
    :rtype: bool
    """
    chuck_segment = segment.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    for location in range(len(segment) - (window_size - 1)):
        if segment[location: location + window_size] in chuck_segment:  # find a palindrome.
            return True

    return True


def compute_minimum_duplicate_spacing(segment: str,
                                      window_size: int) \
        -> int:
    """
    Compute the minimum spacing between any two duplicated k-mers of a given window size in the segment.

    :param segment: given DNA segment.
    :type segment: str

    :param window_size: observation window size used to extract k-mers.
    :type window_size: int

    :return: minimum spacing between any two duplicate k-mers.
    :rtype: int

    .. note::
        The spacing is defined as the distance between two occurrences of the same k-mer.
    """
    k_mer_set = {}
    for index in range(len(segment) - (window_size - 1)):
        k_mer = segment[index: index + window_size]
        if k_mer in k_mer_set:
            k_mer_set[k_mer].append(index)
        else:
            k_mer_set[k_mer] = [index]

    minimum_value = len(segment)
    for k_mer, locations in {k: v for k, v in k_mer_set.items() if len(v) > 1}.items():
        locations = array(sorted(locations))

        for distance in (locations[1:] - locations[:-1]):
            if minimum_value >= distance - window_size:
                minimum_value = distance - window_size

    return minimum_value


# noinspection PyTypeChecker
def compute_minimum_palindrome_spacing(segment: str,
                                       window_size: int) \
        -> int:
    """
    Compute the minimum spacing between any k-mer and the reverse complement.

    :param segment: given DNA segment.
    :type segment: str

    :param window_size: observation window size used to extract k-mers.
    :type window_size: int

    :return: minimum spacing between any k-mer and its reverse complement.
    :rtype: int

    .. note::
        The spacing is defined as the gap between two distinct occurrences of a k-mer and its reverse complement,
        excluding self-matches at the same position.
    """
    k_mer_set = {}
    for index in range(len(segment) - (window_size - 1)):
        k_mer = segment[index: index + window_size]

        if k_mer in k_mer_set:
            k_mer_set[k_mer].append(index)
        else:
            k_mer_set[k_mer] = [index]

    chuck_segment, minimum_value = segment.translate(str.maketrans("ACGT", "TGCA")), len(segment)
    for index in range(len(chuck_segment) - (window_size - 1)):
        motif = chuck_segment[index: index + window_size][::-1]

        if motif in k_mer_set:
            locations = array(sorted(k_mer_set[motif]))
            formers = where(locations < index - window_size)[0]  # valid upstream matches.
            latters = where(locations > index + window_size)[0]  # valid downstream matches.
            if len(formers) > 0:
                minimum_value = min([minimum_value, min(index - (locations[formers] + window_size))])
            if len(latters) > 0:
                minimum_value = min([minimum_value, min((locations[latters] - (index + window_size)))])

    return minimum_value
