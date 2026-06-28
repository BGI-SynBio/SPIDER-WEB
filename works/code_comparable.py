"""
This script consolidates the entire pipelines from the Composite-Hedges-Nanopores repository
(https://github.com/ysfhtxn/Composite-Hedges-Nanopores/tree/main/CHN_invitro)
into a single, self-contained Python implementation.
These modifications are introduced to address practical limitations in the original implementation,
including undocumented behaviors and instability in certain preprocessing steps,
which hinder direct and reproducible comparison under controlled conditions.
To mitigate these issues, parts of the pipeline are restructured to ensure deterministic and consistent behavior.
These changes are made solely for evaluation purposes
and do not constitute a production-ready or strictly equivalent reimplementation of the original system.
"""
from __future__ import annotations
from Bio import SeqIO
from collections import Counter
from copy import deepcopy
from datetime import datetime
from Levenshtein import distance
from numpy import ndarray, array, zeros, eye, frombuffer, random, vstack
from numpy import unpackbits, packbits, argsort, argmin, sum, min, max, ceil, all, uint8
from os import makedirs, system, path
from reedsolo import RSCodec, ReedSolomonError
from scipy.spatial.distance import jensenshannon
from shutil import rmtree
from typing import List, Dict, Set, Tuple, Optional

from sbw import Monitor


def construct_barcode_pairs(payload_groups: List[List[str]],
                            former_barcode_length: int = 0,
                            latter_barcode_length: int = 0,
                            minimum_difference: int = 1,
                            maximum_attempts: int = 10000,
                            window_size: Optional[int] = 20,
                            homopolymer: Optional[int] = 4,
                            gc_counts: Optional[Tuple[int, int]] = (9, 11),
                            verbose: bool = False) \
        -> List[Tuple[str, str]]:
    """
    Construct barcode pairs for each composite sequence (composited by multiple DNA sequences).

    :param payload_groups: grouped payload sequences, where each element is a list of DNA sequences
                           assumed to originate from the same composite sequence.
    :type payload_groups: list

    :param former_barcode_length: length of the former barcode.
    :type former_barcode_length: int

    :param latter_barcode_length: length of the latter barcode.
    :type latter_barcode_length: int

    :param minimum_difference: minimum edit distance required between the generated barcode and
                               any k-mer derived from the payload sequences.
    :type minimum_difference: int

    :param maximum_attempts: maximum number of random trials used to generate a valid barcode candidate.
    :type maximum_attempts: int

    :param window_size: observation window size.
    :type window_size: int or None

    :param homopolymer: maximum homopolymer length.
    :type homopolymer: int or None

    :param gc_counts: GC count in any window size, formed as (minimum count, maximum count).
    :type gc_counts: tuple or None

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: barcode pair group.
    :rtype: list
    """
    if former_barcode_length == 0 and latter_barcode_length == 0:
        raise ValueError("Due to the chosen barcode length setting, no barcodes are created.")

    if homopolymer is not None or gc_counts is not None:
        assert window_size is not None
    if window_size is not None and gc_counts is not None:
        assert 0 <= gc_counts[0] <= gc_counts[1] <= window_size
    if window_size is not None and homopolymer is not None:
        assert 1 < homopolymer < window_size
    if window_size is not None and former_barcode_length > 0:
        assert former_barcode_length >= window_size
    if window_size is not None and latter_barcode_length > 0:
        assert latter_barcode_length >= window_size

    def detect(candidate: str,
               k_mers: List[str]) \
            -> bool:
        """
        Check whether the barcode candidate satisfies the predefined requirements.

        :param candidate: barcode candidate.
        :type candidate: str

        :param k_mers: a set of k-mers to be kept distant in terms of edit distance.
        :type k_mers: list

        :return: valid flag.
        :rtype: bool
        """
        if homopolymer is not None:
            for nucleotide in bases:
                if nucleotide * (homopolymer + 1) in candidate:
                    return False

        if gc_counts is not None:
            for index in range(len(candidate) - window_size + 1):
                segment = candidate[index:index + window_size]
                gc_count = segment.count("G") + segment.count("C")
                if gc_count < gc_counts[0] or gc_count > gc_counts[1]:
                    return False

        for k_mer in k_mers:
            if distance(k_mer, candidate) < minimum_difference:
                return False

        return True

    barcode_pairs, bases, monitor = [], ["A", "C", "G", "T"], Monitor()
    for group_index, payloads in enumerate(payload_groups):
        if former_barcode_length > 0:
            former_k_mers, former_barcode = set(), None
            for payload in payloads:
                for location in range(len(payload) - former_barcode_length + 1):
                    former_k_mers.add(payload[location:location + former_barcode_length])
            former_k_mers = list(former_k_mers)
            for barcode, _ in barcode_pairs:
                former_k_mers.append(barcode)

            for _ in range(maximum_attempts):
                # noinspection PyTypeChecker
                barcode_candidate = "".join([random.choice(bases) for _ in range(former_barcode_length)])
                if detect(barcode_candidate, former_k_mers):
                    former_barcode = barcode_candidate
                    break

            if former_barcode is None:
                raise ValueError("The former barcode for the " + str(group_index + 1)
                                 + "-th group of sequences could not be generated!")
        else:
            former_barcode = ""

        if latter_barcode_length > 0:
            latter_k_mers, latter_barcode = set(), None
            for payload in payloads:
                for location in range(len(payload) - latter_barcode_length + 1):
                    latter_k_mers.add(payload[location:location + latter_barcode_length])
            latter_k_mers = list(latter_k_mers)
            for _, barcode in barcode_pairs:
                latter_k_mers.append(barcode)

            for _ in range(maximum_attempts):
                # noinspection PyTypeChecker
                barcode_candidate = "".join([random.choice(bases) for _ in range(latter_barcode_length)])
                if detect(barcode_candidate, latter_k_mers):
                    latter_barcode = barcode_candidate
                    break

            if latter_barcode is None:
                raise ValueError("The latter barcode for the " + str(group_index + 1)
                                 + "-th group of sequences could not be generated!")
        else:
            latter_barcode = ""

        barcode_pairs.append((former_barcode, latter_barcode))

        if verbose:
            monitor(group_index + 1, len(payload_groups))

    return barcode_pairs


def group_composite_sequences(sequences: List[str],
                              barcode_pairs: List[Tuple[str, str]],
                              former_start: int = -1,
                              latter_start: int = -1,
                              verbose: bool = False) \
        -> Dict[int, List[str]]:
    """
    Group the composite sequences from the raw sequencing data.

    :param sequences: raw sequence data.
    :type sequences: list

    :param barcode_pairs: barcode pairs.
    :type barcode_pairs: list

    :param former_start: former location start.
    :type former_start: int

    :param latter_start: latter location start.
    :type latter_start: int

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: composite sequence groups.
    :rtype: dict
    """
    barcode_group, monitor = [], Monitor()

    def generate_candidates(barcode: str) \
            -> Set[str]:
        """
        Construct the set of possible barcode candidates under the one-edit-error scenario.

        :param barcode: barcode in the sequence.
        :type barcode: str

        :return: barcode candidates.
        :rtype: set
        """
        candidates = {barcode}

        for changed_index in range(len(barcode)):
            for mutation in "ACGT":
                candidates.add(barcode[:changed_index] + mutation + barcode[changed_index + 1:])
        for changed_index in range(len(barcode)):
            for insertion in "ACGT":
                candidates.add((barcode[:changed_index] + insertion + barcode[changed_index:])[:len(barcode)])
        for changed_index in range(len(barcode)):
            for supplement in "ACGT":
                candidates.add(barcode[:changed_index] + barcode[changed_index + 1:] + supplement)

        return candidates

    for index, (former_barcode, latter_barcode) in enumerate(barcode_pairs):
        former_candidates = generate_candidates(former_barcode) if len(former_barcode) > 0 else None
        latter_candidates = generate_candidates(latter_barcode) if len(latter_barcode) > 0 else None
        barcode_group.append((former_candidates, latter_candidates, len(former_barcode), len(latter_barcode)))

    search_order, sequence_groups = [v for i in range(0, 10 + 1) for v in ((0,) if i == 0 else (i, -i))], {}
    for sequence_index, sequence in enumerate(sequences):
        finished = False
        for oriented_sequence in [sequence, sequence[::-1].translate(str.maketrans("ACGT", "TGCA"))]:
            for index, (former_barcodes, latter_barcodes, former_length, latter_length) in enumerate(barcode_group):
                former_bound, latter_bound = -1, -1
                if former_length >= 0:
                    for search_offset in search_order:
                        start, end = former_start + search_offset, former_start + search_offset + former_length
                        if start >= 0 and end <= len(oriented_sequence):
                            detected_barcode = oriented_sequence[start:end]
                            if detected_barcode in former_barcodes:
                                former_bound = end
                                break
                        if former_bound >= 0:
                            break
                if latter_length > 0:
                    for search_offset in search_order:
                        start, end = latter_start + search_offset, latter_start + search_offset + latter_length
                        if start >= 0 and end <= len(oriented_sequence):
                            detected_barcode = oriented_sequence[start:end]
                            if detected_barcode in latter_barcodes:
                                latter_bound = start
                                break
                        if latter_bound >= 0:
                            break

                if former_length > 0 and latter_length > 0:
                    if 0 <= former_bound < latter_bound:
                        sequence_groups.setdefault(index, []).append(oriented_sequence)
                        finished = True
                        break
                elif former_length > 0 and latter_length == 0:  # former barcode only.
                    if 0 <= former_bound:
                        sequence_groups.setdefault(index, []).append(oriented_sequence)
                        finished = True
                        break
                elif former_length == 0 and latter_length > 0:
                    if 0 < latter_bound:
                        sequence_groups.setdefault(index, []).append(oriented_sequence)
                        finished = True
                        break
                else:
                    raise ValueError("No barcode!")

            if finished:
                break

        if verbose:
            monitor(sequence_index + 1, len(sequences))

    return sequence_groups


def find_consensus_sequences(raw_sequences: List[str],
                             designed_length: int,
                             temp_folder: str,
                             threshold: int,
                             similarity: float = 0.85,
                             coverage: float = 0.90) \
        -> List[str]:
    """
    Find consensus sequences from the grouped raw sequences.

    :param raw_sequences: grouped raw sequences.
    :type raw_sequences: list

    :param designed_length: length designed for raw sequences.
    :type designed_length: int

    :param temp_folder: temporary folder for storing clustering and alignment data.
    :type temp_folder: str

    :param threshold: minimum cluster size.
    :type threshold: int

    :param similarity: nucleotide-level consistency within the aligned region.
    :type similarity: float

    :param coverage: extent of sequence coverage in the alignment.
    :type coverage: float

    :return: consensus sequences.
    :rtype: list
    """
    o_folder, t_folder = temp_folder + "o/", temp_folder + "t/"

    if not path.exists(temp_folder):
        makedirs(temp_folder)
    if not path.exists(o_folder):
        makedirs(o_folder)
    if not path.exists(t_folder):
        makedirs(t_folder)

    # use the MMSeqs2 tool to cluster raw sequences.
    with open(file=o_folder + "origin.fasta", mode="w") as file:
        for index, sequence in enumerate(raw_sequences):
            file.write(">" + str(index) + "\n")
            file.write(sequence + "\n")
    system("mmseqs easy-cluster " + o_folder + "origin.fasta" + " " + o_folder + " " + t_folder + " "
           + "--min-seq-id " + str(similarity) + " -c " + str(coverage)
           + " --cov-mode 0 --rescore-mode 3 --threads 1 > /dev/null 2>&1")

    clusters = {}
    with open(file=o_folder + "_cluster.tsv") as file:
        for line in file.readlines():
            cluster_index, sequence_index = [int(value) for value in line[:-1].split("\t")]
            clusters.setdefault(cluster_index, []).append(sequence_index)

    consensus_sequences = []
    for cluster_index, sequence_indices in clusters.items():
        if len(sequence_indices) >= threshold:
            clustered_path = o_folder + "clustered-" + str(cluster_index) + ".fasta"
            aligned_path = o_folder + "aligned-" + str(cluster_index) + ".fasta"
            with open(file=clustered_path, mode="w") as file:
                for sequence_index in sequence_indices:
                    file.write(">" + str(sequence_index) + "\n")
                    file.write(raw_sequences[sequence_index] + "\n")
            # use MUSCLE5 to do the alignment of sub-clusters.
            system("muscle -align " + clustered_path + " -output " + aligned_path + " --threads 1 > /dev/null 2>&1")
            assembled_sequences = [str(info.seq) for info in SeqIO.parse(aligned_path, "fasta")]

            # obtain the consensus sequence.
            lengths = [len(sequence) for sequence in assembled_sequences]
            if len(set(lengths)) != 1:
                used_length = max(lengths)
                used_sequences = [sequence + "-" * (used_length - len(sequence)) for sequence in assembled_sequences]
            else:
                used_length, used_sequences = lengths[0], assembled_sequences
            # contain the count of "A", "C", "G", "T", and "-".
            radio_group = zeros(shape=(used_length, 5))
            for location in range(used_length):
                counter = Counter([sequence[location] for sequence in used_sequences])
                radio_group[location] = [counter.get(base, 0) for base in "ACGT-"]
            drop_mask = zeros(shape=(used_length,), dtype=bool)
            if used_length > designed_length:
                order = argsort(radio_group[:, 4])[::-1]
                drop_mask[order[:used_length - designed_length]] = True
            elif used_length < designed_length:
                raise ValueError("systematic loss of " + str(designed_length - used_length) + " nucleotide(s)!")
            candidates, mapping, ideal_ratios = [], ["A", "C", "G", "T"], eye(4)
            for location, flag in enumerate(drop_mask):
                if not flag:
                    radios = radio_group[location][:4]
                    total = float(sum(radios))
                    if total > 0.0:
                        radios /= total
                    else:
                        raise ValueError("systematic loss of a nucleotide!")
                    candidates.append(mapping[int(argmin([jensenshannon(ideal_ratios[i], radios) for i in range(4)]))])
            consensus_sequences.append("".join(candidates))

    rmtree(temp_folder)

    return consensus_sequences


def trim_barcodes(sequences: List[str],
                  former_barcode: str = "",
                  latter_barcode: str = "",
                  former_start: int = -1,
                  latter_start: int = -1,
                  verbose: bool = False) \
        -> List[str]:
    """
    Trim barcodes from the original sequences.

    :param sequences: original sequences.
    :type sequences: list

    :param former_barcode: former barcode in the original sequences (left barcode in 5->3 sequences).
    :type former_barcode: str

    :param latter_barcode: latter barcode in the original sequences (right barcode in 5->3 sequences).
    :type latter_barcode: str

    :param former_start: former location start.
    :type former_start: int

    :param latter_start: latter location start.
    :type latter_start: int

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: trimmed sequences.
    :rtype: list
    """
    trimmed_sequences, monitor = [], Monitor()

    if len(former_barcode) == 0 and len(latter_barcode) == 0:
        return sequences

    if len(former_barcode) > 0 > former_start:
        raise ValueError("The parameter \"former_start\" has not been initialized.")
    if len(latter_barcode) > 0 > latter_start:
        raise ValueError("The parameter \"latter_start\" has not been initialized.")

    if (len(former_barcode) > 0) and (len(latter_barcode) > 0) and (former_start + len(former_barcode) > latter_start):
        raise ValueError("The former barcode overlaps with or exceeds the latter barcode.")

    for index, sequence in enumerate(sequences):
        if len(former_barcode) > 0 and len(latter_barcode) > 0:
            trimmed_sequences.append(sequence[former_start + len(former_barcode): latter_start])
        elif len(former_barcode) > 0 and len(latter_barcode) == 0:
            trimmed_sequences.append(sequence[former_start + len(former_barcode):])
        else:  # former_barcode is None and latter_barcode is not None:
            trimmed_sequences.append(sequence[:latter_start])

        if verbose:
            monitor(index + 1, len(sequences))

    return trimmed_sequences


class CompositeHEDGESNanopores:

    def __init__(self,
                 resolution: int,
                 sigma: int,
                 step: int,
                 pattern: int = 8,
                 maximum_instance: int = 250000,
                 penalty_threshold: int = 10,
                 node_threshold: int = 50,
                 bucket_number: int = 10,
                 window_size: Optional[int] = 20,
                 homopolymer: Optional[int] = 4,
                 gc_counts: Optional[Tuple[int, int]] = (9, 11)):
        """
        Initialize the Composite HEDGES Nanopores coding algorithm.

        :param resolution: generated sequence number.
        :type resolution: int

        :param sigma: modulus parameter for valid composite letters.
        :type sigma: int

        :param step: the number of bits per sliding-window step.
        :type step: int

        :param pattern: the number of bits per letter.
        :type pattern: int

        :param maximum_instance: the maximum number of allowed HypothesisNode instances.
        :type maximum_instance: int

        :param penalty_threshold: maximum penalty, used to control early termination of the decoding process.
        :type penalty_threshold: int

        :param node_threshold: maximum number of nodes per layer in the hypothesis tree.
        :type node_threshold: int

        :param bucket_number: number of buckets used for beam pruning of excess nodes in a layer of the hypothesis tree.
        :type bucket_number: int

        :param window_size: observation window size.
        :type window_size: int or None

        :param homopolymer: maximum homopolymer length.
        :type homopolymer: int or None

        :param gc_counts: GC count in any window size, formed as (minimum count, maximum count).
        :type gc_counts: tuple or None
        """
        if sigma > 4 and resolution % 2 != 0:
            raise ValueError("The resolution must be divisible by 2.")

        if homopolymer is not None or gc_counts is not None:
            assert window_size is not None
        if window_size is not None and gc_counts is not None:
            assert 0 <= gc_counts[0] <= gc_counts[1] <= window_size
        if window_size is not None and homopolymer is not None:
            assert 1 < homopolymer < window_size

        self.resolution, self.sigma, self.step, self.pattern = resolution, sigma, step, pattern
        self.window_size, self.homopolymer, self.gc_counts = window_size, homopolymer, gc_counts
        self.maximum_instance = maximum_instance
        self.pattern_mask = (1 << 7 * self.pattern) - 1
        self.alphabet = [("A", ("A",)), ("C", ("C",)), ("G", ("G",)), ("T", ("T",)),
                         ("M", ("A", "C")), ("K", ("G", "T")), ("R", ("A", "G")), ("Y", ("C", "T"))][:sigma]
        self.penalty_threshold, self.node_threshold = penalty_threshold, node_threshold
        self.bucket_number = bucket_number

    def encode(self,
               binary_messages: ndarray,
               verbose: bool = False) \
            -> Tuple[List[str], List[List[str]]]:
        """
        Encode binary messages using CHN.

        :param binary_messages: binary messages
        :type binary_messages: numpy.ndarray

        :param verbose: verbosity mode.
        :type verbose: bool

        :return: composite DNA sequences and their corresponding sequence composition.
        :rtype: list
        """
        if ceil(binary_messages.shape[1] / self.step) > self.maximum_instance:
            raise ValueError("The maximum number of allowed HypothesisNode instances ("
                             + str(self.maximum_instance) + ") is insufficient to support the encoding process.")

        composite_sequences, sequence_groups, monitor = [], [], Monitor()
        for message_index, binary_message in enumerate(binary_messages):
            sequences, composite_sequence, state_info, location = [[] for _ in range(self.resolution)], "", 0, 0
            while location < len(binary_message):
                symbol_info = 0
                for _ in range(self.step):
                    if location < len(binary_message):
                        symbol_info = (symbol_info << 1) + int(binary_message[location])
                    else:
                        symbol_info <<= 1
                    location += 1

                state_info = ((state_info << self.step) + symbol_info) & self.pattern_mask

                used_alphabet, modulus = self.produce_alphabet(composite_sequence=composite_sequence)
                if modulus > 0:
                    digest_index = self.digest(state_info=state_info, bit_index=location - 1, modulus=modulus)
                    composite_nucleotide, individual_nucleotides = used_alphabet[digest_index]
                    composite_sequence += composite_nucleotide
                else:
                    raise ZeroDivisionError("modulus = 0!")

                if len(individual_nucleotides) == 1:
                    supplements = individual_nucleotides[0] * self.resolution
                elif len(individual_nucleotides) == 2:
                    supplements = individual_nucleotides[0] * (self.resolution // 2)
                    supplements += individual_nucleotides[1] * (self.resolution // 2)
                else:
                    raise ValueError("Unsupported composite alphabet definition.")

                for sequence_index, supplement in enumerate(supplements):
                    sequences[sequence_index].append(supplement)

            composite_sequences.append(composite_sequence)
            sequence_groups.append(["".join(composition) for composition in sequences])

            if verbose:
                monitor(message_index + 1, len(binary_messages))

        return composite_sequences, sequence_groups

    def decode(self,
               sequence_groups: Dict[int, List[str]],
               verbose: bool = False) \
            -> Dict[int, ndarray]:
        """
        Decode grouped sequences using CHN.

        :param sequence_groups: sequence groups, each representing a set of sequences aligned into a composite sequence.
        :type sequence_groups: dict

        :param verbose: verbosity mode.
        :type verbose: bool

        :return: decoded binary message groups.
        :rtype: numpy.ndarray
        """
        results, monitor = {}, Monitor()
        ideal_ratios = array([[1.0, 0.0, 0.0, 0.0],  # A
                              [0.0, 1.0, 0.0, 0.0],  # C
                              [0.0, 0.0, 1.0, 0.0],  # G
                              [0.0, 0.0, 0.0, 1.0],  # T
                              [0.5, 0.5, 0.0, 0.0],  # M = A/C
                              [0.0, 0.0, 0.5, 0.5],  # K = G/T
                              [0.5, 0.0, 0.5, 0.0],  # R = A/G
                              [0.0, 0.5, 0.0, 0.5]])[:self.sigma]  # Y = C/T
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "M": 4, "K": 5, "R": 6, "Y": 7}

        for process, (group_index, sequences) in enumerate(sequence_groups.items()):
            has_issue = False
            if len(sequences) == 0:
                if verbose:
                    raise ValueError("Each sequence group must contain at least one sequence.")
                else:
                    has_issue = True

            if len(set([len(sequence) for sequence in sequences])) != 1:
                if verbose:
                    raise ValueError("The length of all sequences must be same!")
                else:
                    has_issue = True

            if has_issue:
                continue

            sequence_length = len(sequences[0])
            tree = CompositeHEDGESNanopores.HypothesisTree(step=self.step, pattern_mask=self.pattern_mask, maximum_instance=sequence_length)
            current_nodes, bit_index = [tree.root], 0

            for position in range(sequence_length):
                next_nodes, bit_index = [], bit_index + self.step
                counter = Counter([sequence[position] for sequence in sequences])
                radios = array([counter.get(nucleotide, 0) for nucleotide in "ACGT"], dtype=float)
                total = float(sum(radios))
                if total > 0.0:
                    radios /= total
                else:
                    if verbose:
                        raise ValueError("All sequences contain gaps at position " + str(position) + ".")
                    else:
                        has_issue = True
                if has_issue:
                    break

                penalties = array([jensenshannon(ideal_ratios[i], radios) for i in range(self.sigma)])

                for parent_node in current_nodes:
                    if verbose:
                        established_part = tree.backtrack_established_sequence(node=parent_node)
                    else:
                        try:
                            established_part = tree.backtrack_established_sequence(node=parent_node)
                        except ValueError:
                            has_issue = True
                    if has_issue:
                        break

                    # noinspection PyUnboundLocalVariable
                    used_alphabet, modulus = self.produce_alphabet(composite_sequence=established_part)

                    if modulus == 0:
                        if verbose:
                            raise ValueError("The modulus cannot be zero!")
                        else:
                            has_issue = True
                            break

                    for symbol_info in range(1 << self.step):
                        predicted_state = ((parent_node.state_info << self.step) + symbol_info) & self.pattern_mask
                        predicted_letter, _ = used_alphabet[self.digest(predicted_state, bit_index - 1, modulus)]
                        penalty = penalties[mapping[predicted_letter]]
                        child = parent_node.expand_child(symbol_info, penalty, self.pattern_mask, sequence_length,
                                                         predicted_letter)

                        if child.score < self.penalty_threshold:
                            next_nodes.append(child)
                        else:
                            child.detach()

                if len(next_nodes) > self.node_threshold:
                    # We do not adopt bucket-based coarse pruning here,
                    # as it may occasionally discard the correct sequence.
                    # We directly rank candidates by score and remove only the highest-scoring cases.
                    # In empirical evaluation under small-batch settings
                    # (consistent with the scale used in the original experiments),
                    # this strategy did not introduce a negative impact on runtime;
                    # on the contrary, the improved search quality led to faster overall completion of the function.
                    next_nodes = sorted(next_nodes, key=lambda node: node.score)[:self.node_threshold]

                current_nodes = next_nodes

            if has_issue:
                continue

            if len(current_nodes) == 0:
                if verbose:
                    raise ValueError("No surviving hypothesis!")
                else:
                    has_issue = True

            if has_issue:
                continue

            if verbose:
                results[group_index] = tree.backtrack_best_messages(current_nodes)
            else:
                try:
                    results[group_index] = tree.backtrack_best_messages(current_nodes)
                except ValueError:
                    pass

            if verbose:
                monitor(process + 1, len(sequence_groups))

        return results

    def produce_alphabet(self,
                         composite_sequence: str) \
            -> Tuple[List[Tuple[str, Tuple[str, ...]]], int]:
        """
        Produce current alphabet based on the established constraints.

        :param composite_sequence: composite sequence.
        :type composite_sequence: str

        :return: current alphabet and its size.
        :rtype: tuple
        """
        if self.window_size is None:
            return deepcopy(self.alphabet), len(self.alphabet)

        info = []
        if len(composite_sequence) > (self.window_size - 1):
            for composite_nucleotide in composite_sequence[-(self.window_size - 1):]:
                for composite, individuals in self.alphabet:
                    if composite_nucleotide == composite:
                        info.append(individuals)
        else:
            for composite_nucleotide in composite_sequence:
                for composite, individuals in self.alphabet:
                    if composite_nucleotide == composite:
                        info.append(individuals)
        situations = []
        for _, individuals in self.alphabet:
            situations.append(deepcopy(info) + [individuals])

        if self.homopolymer is not None:
            new_situations = []
            for situation in situations:
                available = True
                if len(situation) >= self.homopolymer + 1:
                    for nucleotide in ["A", "C", "G", "T"]:
                        if all([nucleotide in individuals for individuals in situation[-(self.homopolymer + 1):]]):
                            available = False
                if available:
                    new_situations.append(situation)
            situations = new_situations

        if self.gc_counts is not None:
            new_situations = []
            # no matter how it is chosen, the GC content is either insufficient or excessive.
            for situation in situations:
                available = True
                if len(situation) >= self.window_size:
                    bound = [0, 0]
                    for individuals in situation[-self.window_size:]:
                        if all([nucleotide in "AT" for nucleotide in individuals]):
                            bound[0] += 1
                        if all([nucleotide in "GC" for nucleotide in individuals]):
                            bound[1] += 1
                    bound[0] = self.window_size - bound[0]
                    if bound[0] < self.gc_counts[0] or bound[1] > self.gc_counts[1]:
                        available = False
                else:
                    bound = 0
                    for individuals in situation:
                        if all([nucleotide in "GC" for nucleotide in individuals]):
                            bound += 1
                    if bound > self.gc_counts[1]:
                        available = False
                if available:
                    new_situations.append(situation)
            situations = new_situations

        available_next = []
        for situation in situations:
            available_next.append(situation[-1])

        current_alphabet = []
        for composite, individuals in self.alphabet:
            if individuals in available_next:
                current_alphabet.append((composite, individuals))

        return current_alphabet, len(current_alphabet)

    def digest(self,
               state_info: int,
               bit_index: int,
               modulus: int) \
            -> int:
        """
        Compute the hash-based alphabet index for the current hypothesis state.

        :param state_info: masked historical bit state of the current hypothesis node.
        :type state_info: int

        :param bit_index: bit index of the current hypothesis state.
        :type bit_index: int

        :param modulus: current alphabet size.
        :type modulus: int

        :return: alphabet index produced by the hash function.
        :rtype: int
        """
        original_value = state_info + (bit_index << self.pattern)

        hashed_value = original_value * 3935559000370003845 + 2691343689449507681
        hashed_value ^= hashed_value >> 21
        hashed_value ^= hashed_value << 37
        hashed_value ^= hashed_value >> 4
        hashed_value *= 4768777513237032717
        hashed_value ^= hashed_value << 20
        hashed_value ^= hashed_value >> 41
        hashed_value ^= hashed_value << 5

        return int(hashed_value % modulus)

    class HypothesisNode:

        __slots__ = ("step", "state_info", "index", "symbol_info", "score", "parent", "children", "composite_symbol")

        def __init__(self,
                     step: int,
                     state_info: int,
                     index: int = -1,
                     symbol_info: Optional[int] = None,
                     score: float = 0.0,
                     parent: Optional[CompositeHEDGESNanopores.HypothesisNode] = None,
                     composite_symbol: Optional[str] = None):
            """
            Initialize a hypothesis node.

            :param step: number of bits represented by each transition.
            :type step: int

            :param state_info: masked historical message state of this node.
            :type state_info: int

            :param index: node index in the decoding path.
            :type index: int

            :param symbol_info: current step-bit value used to reach this node.
            :type symbol_info: int or None

            :param score: accumulated path score or penalty.
            :type score: float

            :param parent: parent hypothesis node.
            :type parent: HypothesisNode or None

            :param composite_symbol: composite symbol.
            :type composite_symbol: str
            """
            if not isinstance(step, int) or step <= 0:
                raise ValueError("step must be positive integer")

            self.step, self.index, self.parent, self.children = step, index, parent, {}
            self.state_info, self.symbol_info, self.score = state_info, symbol_info, score
            self.composite_symbol = composite_symbol

        def expand_child(self,
                         symbol_info: int,
                         penalty: float,
                         pattern_mask: int,
                         maximum_instance: int,
                         composite_symbol: str) \
                -> CompositeHEDGESNanopores.HypothesisNode:
            """
            Expand the current node by appending one step-bit message.

            :param symbol_info: current step-bit message.
            :type symbol_info: int

            :param penalty: incremental score or penalty added to the current node.
            :type penalty: float

            :param pattern_mask: bit mask used to retain the finite historical state.
            :type pattern_mask: int

            :param maximum_instance: maximum allowed node index.
            :type maximum_instance: int

            :param composite_symbol: composite symbol.
            :type composite_symbol: str

            :return: newly expanded child node.
            :rtype: HypothesisNode
            """
            if self.index + 1 > maximum_instance:
                raise ValueError("Maximum node too small!")

            new_state = ((self.state_info << self.step) + symbol_info) & pattern_mask

            child = CompositeHEDGESNanopores.HypothesisNode(self.step, new_state, self.index + 1, symbol_info, self.score + penalty,
                                                            parent=self, composite_symbol=composite_symbol)

            self.children[symbol_info] = child

            return child

        def detach(self):
            """
            Detach this node from its parent.

            After detachment, the parent no longer references this node through its children dictionary.
            """
            if self.parent is None:
                return

            self.parent.children.pop(self.symbol_info, None)
            self.parent = None

    class HypothesisTree:

        __slots__ = ("step", "pattern_mask", "maximum_instance", "root", "current_node")

        def __init__(self,
                     step: int,
                     pattern_mask: int,
                     maximum_instance: int):
            self.step, self.pattern_mask, self.maximum_instance = step, pattern_mask, maximum_instance
            self.root = CompositeHEDGESNanopores.HypothesisNode(step=step, state_info=0)
            self.current_node = self.root

        # noinspection PyUnresolvedReferences
        def backtrack_symbols(self,
                              node: Optional[CompositeHEDGESNanopores.HypothesisNode] = None) \
                -> List[int]:
            """
            Backtrack the symbol sequence from the given hypothesis node to the root.

            :param node: terminal hypothesis node. If None, use current_node.
            :type node: HypothesisNode or None

            :return: list of step-wise symbol indices in forward order.
            :rtype: list
            """
            if node is None:
                node = self.current_node

            symbols = []
            while node is not None:
                parent = node.parent
                if parent is None:
                    break
                if node.symbol_info is None:
                    raise RuntimeError("Invalid hypothesis node, symbol_info is None!")

                symbols.append(node.symbol_info)
                node = parent

            symbols.reverse()

            return symbols

        # noinspection PyUnresolvedReferences
        def backtrack_established_sequence(self,
                                           node: Optional[CompositeHEDGESNanopores.HypothesisNode] = None) \
                -> str:
            """
            Backtrack established composite sequence.

            :param node: terminal hypothesis node. If None, use current_node.
            :type node: CompositeHEDGESNanopores.HypothesisNode or None

            :return: composite sequence.
            :rtype: str
            """
            if node is None:
                node = self.current_node

            symbols = []

            while node is not None and node.parent is not None:
                if node.composite_symbol is None:
                    raise RuntimeError("Invalid hypothesis node: composite_symbol is None.")

                symbols.append(node.composite_symbol)
                node = node.parent

            symbols.reverse()

            return "".join(symbols)

        def backtrack_best_messages(self,
                                    node_list: List[CompositeHEDGESNanopores.HypothesisNode]) \
                -> List[ndarray]:
            """
            Backtrack the binary messages corresponding to the minimum-score hypotheses.

            :param node_list: list of terminal hypothesis nodes.
            :type node_list: list

            :return: a list of decoded binary messages.
            :rtype: list
            """
            if len(node_list) == 0:
                raise ValueError("The node list is empty, so backtracking is unavailable.")

            best_score, candidates = min([node.score for node in node_list]), []

            for node in node_list:
                if node.score == best_score:
                    symbols = self.backtrack_symbols(node)

                    binary_message = []
                    for symbol in symbols:
                        for shift in reversed(range(self.step)):
                            binary_message.append((symbol >> shift) & 1)

                    candidates.append(array(binary_message))

            return candidates


def test_chn(bit_size: int,
             seed: Optional[int] = None):
    """
    Test the correctness of the Composite HEDGES Nanopores coding algorithm.

    :param bit_size: the bit size to be stored.
    :type bit_size: int

    :param seed: random seed for replication.
    :type seed: int or None
    """
    if seed is not None:
        random.seed(seed)

    # 36-byte message is encoded into a 40-byte Reed–Solomon codeword with 4 parity symbols (original usage).
    bit_length, rs_byte_length = 8 * 36, 4

    rs_code = RSCodec(rs_byte_length, bit_length // 8 + rs_byte_length)

    monitor = Monitor()

    sequence_number = int(ceil(bit_size // bit_length))
    print("Step 1: prepare binary messages with shape (" + str(sequence_number) + "," + str(bit_length) + ").")
    # prepare binary messages.
    binary_messages = random.randint(0, 2, size=(sequence_number, bit_length), dtype=uint8)
    monitor(1, 1)

    print("Step 2: add fault tolerance mechanism.")
    # add fault tolerance mechanism
    byte_messages = []
    # noinspection PyTypeChecker
    for process, byte_message in enumerate(packbits(binary_messages, axis=1)):
        byte_messages.append(frombuffer(rs_code.encode(bytes(byte_message.tolist())), dtype=uint8))
        # noinspection PyTypeChecker
        monitor(process + 1, len(binary_messages))
    full_binary_messages = unpackbits(vstack(byte_messages), axis=1)

    print("Step 3: encode binary messages.")
    # encode binary messages with the original usage.
    code_method = CompositeHEDGESNanopores(resolution=2, sigma=8, step=2)
    _, sequence_groups = code_method.encode(full_binary_messages, verbose=True)

    new_sequence_groups = {}
    for group_index, group in enumerate(sequence_groups):
        new_sequence_groups[group_index] = group

    print("Step 4: decode DNA sequences.")
    message_groups = code_method.decode(sequence_groups=new_sequence_groups, verbose=True)

    print("Step 5: correct binary messages and confirm the correctness of the implementation.")
    # noinspection PyTypeChecker
    correctness = zeros(shape=(len(binary_messages),), dtype=bool)
    for process, (group_index, message_group) in enumerate(message_groups.items()):
        candidates = []
        for message in message_group:
            if len(message) == 8 * 40:
                byte_message = packbits(message).astype(uint8)
                try:
                    decoded_result = rs_code.decode(bytes(byte_message.tolist()))
                    # handle version- or platform-dependent return formats of reedsolo.decode().
                    if isinstance(decoded_result, tuple):
                        decoded_result = decoded_result[0]
                    candidates.append((True, frombuffer(decoded_result, dtype=uint8)))
                except (ReedSolomonError, IndexError):
                    candidates.append((False, byte_message[:bit_length // 8]))
        count = sum([flag for flag, _ in candidates])
        if count >= 1:  # select the only one or the first one.
            for flag, candidate in candidates:
                if flag:
                    # noinspection PyUnresolvedReferences
                    correctness[group_index] = all(unpackbits(candidate) == binary_messages[group_index])
                    break
        else:  # select the first one.
            # noinspection PyUnresolvedReferences
            correctness[group_index] = all(unpackbits(candidates[0][1]) == binary_messages[group_index])
        monitor(process + 1, len(message_groups))

    print("correctness: " + str(sum(correctness)) + "/" + str(len(correctness)) + ".")
    print()

    if seed is not None:
        random.seed()


def test_symmetry(bit_size: int,
                  seed: Optional[int] = None):
    """
    Infer the functionality correctness through symmetry equivalence testing.

    :param bit_size: the bit size to be stored.
    :type bit_size: int

    :param seed: random seed for replication.
    :type seed: int or None
    """
    monitor = Monitor()

    if seed is not None:
        random.seed(seed)

    print("Equivalence testing of adding, correcting, and removing RS code segments without error rate:")
    # 36-byte message is encoded into a 40-byte Reed–Solomon codeword with 4 parity symbols (original usage).
    bit_length, rs_byte_length = 8 * 36, 4
    rs_code = RSCodec(rs_byte_length, bit_length // 8 + rs_byte_length)

    # prepare binary messages.
    binary_messages = random.randint(0, 2, size=(bit_size // bit_length, bit_length), dtype=uint8)

    # add fault tolerance mechanism
    byte_messages = []
    # noinspection PyTypeChecker
    for byte_message in packbits(binary_messages, axis=1):
        byte_messages.append(frombuffer(rs_code.encode(bytes(byte_message.tolist())), dtype=uint8))
    full_binary_messages = unpackbits(vstack(byte_messages), axis=1)

    correctness = zeros(shape=(len(full_binary_messages),), dtype=bool)
    for index, full_binary_message in enumerate(full_binary_messages):
        if len(full_binary_message) == 8 * 40:
            byte_message = packbits(full_binary_message).astype(uint8)
            try:
                decoded_result = rs_code.decode(bytes(byte_message.tolist()))
                # handle version- or platform-dependent return formats of reedsolo.decode().
                if isinstance(decoded_result, tuple):
                    decoded_result = decoded_result[0]
                obtained_byte_message = frombuffer(decoded_result, dtype=uint8)
            except (ReedSolomonError, IndexError):
                obtained_byte_message = byte_message[:bit_length // 8]

            # noinspection PyTypeChecker
            obtained_binary_message = unpackbits(obtained_byte_message)

            # noinspection PyUnresolvedReferences
            if all(obtained_binary_message == binary_messages[index]):
                correctness[index] = True

        monitor(index + 1, len(full_binary_messages))

    print("correctness: " + str(sum(correctness)) + "/" + str(len(correctness)) + ".")
    print()

    if seed is not None:
        random.seed(seed)

    print("Equivalence testing of constructing, grouping, and detecting barcodes without error rate:")
    # 36-byte message is encoded into a 40-byte Reed–Solomon codeword with 4 parity symbols (original usage).
    bit_length, rs_byte_length = 8 * 36, 4
    barcode_length = 34
    rs_code = RSCodec(rs_byte_length, bit_length // 8 + rs_byte_length)

    # prepare binary messages.
    binary_messages = random.randint(0, 2, size=(bit_size // bit_length, bit_length), dtype=uint8)

    monitor(1, 5)

    # add fault tolerance mechanism
    byte_messages = []
    # noinspection PyTypeChecker
    for byte_message in packbits(binary_messages, axis=1):
        byte_messages.append(frombuffer(rs_code.encode(bytes(byte_message.tolist())), dtype=uint8))
    full_binary_messages = unpackbits(vstack(byte_messages), axis=1)

    monitor(2, 5)

    # encode binary messages with the original usage.
    method = CompositeHEDGESNanopores(resolution=2, sigma=8, step=2)
    composite_sequences, sequence_compositions = method.encode(full_binary_messages)
    payload_length = len(composite_sequences[0])

    monitor(3, 5)

    # generate barcodes as addresses.
    barcode_pairs = construct_barcode_pairs(payload_groups=sequence_compositions,
                                            former_barcode_length=barcode_length,
                                            latter_barcode_length=barcode_length)
    former_start, latter_start = 0, barcode_length + payload_length

    original_sequences = []
    for (former_barcode, latter_barcode), sequence_composition in zip(barcode_pairs, sequence_compositions):
        for payload in sequence_composition:
            original_sequences.append(former_barcode + payload + latter_barcode)

    monitor(4, 5)

    # group
    sequence_groups = group_composite_sequences(sequences=original_sequences, barcode_pairs=barcode_pairs,
                                                former_start=former_start, latter_start=latter_start)

    success = True
    for group_index, group in sequence_groups.items():
        former_barcode, latter_barcode = barcode_pairs[group_index]
        trimmed_sequences = trim_barcodes(sequences=group,
                                          former_barcode=former_barcode, latter_barcode=latter_barcode,
                                          former_start=former_start, latter_start=latter_start)
        if set(trimmed_sequences) != set(sequence_compositions[group_index]):
            success = False
            break

    monitor(5, 5)

    print("correctness: " + str(success) + ".")

    if seed is not None:
        random.seed()


def produce(bit_size: int,
            seed: int) \
        -> Tuple[ndarray, List[str], List[Tuple[str, str]], int]:
    """
    Produce DNA sequences by Composite HEDGES Nanopores coding algorithm.

    :param bit_size: bit size of digital data.
    :type bit_size: int

    :param seed: the seed for random number generator.
    :type seed: int

    :return: expected binary messages, produced DNA sequences, their barcodes, and failure number.
    :rtype: tuple
    """
    method, rs_code = CompositeHEDGESNanopores(resolution=8, sigma=8, step=2), RSCodec(4, 40)
    sequence_number, barcode_length, bit_length, process = int(ceil(bit_size / (8 * 36))), 34, 8 * 36, 0

    random.seed(seed)

    messages, payload_groups, failure_number = [], [], 0
    while process < sequence_number:
        expected_message = random.randint(0, 2, size=(bit_length,), dtype=uint8)
        byte_message = frombuffer(rs_code.encode(bytes(packbits(expected_message).tolist())), dtype=uint8)
        full_binary_message = unpackbits(vstack(byte_message))
        try:
            _, sequence_compositions = method.encode(full_binary_message[None, :])
            message_group = method.decode(sequence_groups={0: sequence_compositions[0]})[0]
        except ZeroDivisionError:
            failure_number += 1
            continue
        except ValueError:
            failure_number += 1
            continue
        except KeyError:
            failure_number += 1
            continue

        candidates = []
        for message in message_group:
            # noinspection PyTypeChecker
            if len(message) == 8 * 40:
                byte_message = packbits(message).astype(uint8)
                try:
                    decoded_result = rs_code.decode(bytes(byte_message.tolist()))
                    # handle version- or platform-dependent return formats of reedsolo.decode().
                    if isinstance(decoded_result, tuple):
                        decoded_result = decoded_result[0]
                    candidates.append((True, frombuffer(decoded_result, dtype=uint8)))
                except (ReedSolomonError, IndexError):
                    candidates.append((False, byte_message[:36]))
        count = sum([flag for flag, _ in candidates])
        if count > 1:
            failure_number += 1
        elif count == 0:
            failure_number += 1
        else:  # count == 1
            expected = True
            for flag, candidate in candidates:
                if flag:
                    obtained_message = unpackbits(candidate)
                    if not all(obtained_message == expected_message):
                        failure_number += 1
                        expected = False
                    break
            if expected:
                messages.append(expected_message)
                payload_groups.append(sequence_compositions[0])
                process += 1

    barcode_pairs = construct_barcode_pairs(payload_groups=payload_groups,
                                            former_barcode_length=barcode_length)

    sequences = []
    for (former_barcode, latter_barcode), sequence_composition in zip(barcode_pairs, payload_groups):
        for payload in sequence_composition:
            sequences.append(former_barcode + payload + latter_barcode)

    return vstack(messages), sequences, barcode_pairs, failure_number


def retrieve(sequences: List[str],
             barcode_pairs: List[Tuple[str, str]],
             data_shape: Tuple[int, ...],
             temp_folder: str,
             cluster_threshold: int = 10) \
        -> Tuple[float, ndarray, int]:
    """
    Retrieve binary messages by Composite HEDGES Nanopores coding algorithm.

    :param sequences: raw DNA sequences.
    :type sequences: list

    :param barcode_pairs: barcode pairs.
    :type barcode_pairs: list

    :param data_shape: data shape of expected binary messages.
    :type data_shape: tuple

    :param temp_folder: temporary folder for storing clustering and alignment data.
    :type temp_folder: str

    :param cluster_threshold: minimum sample number in a cluster.
    :type cluster_threshold: int

    :return: runtime, retrieved messages, and miss sequence count.
    :rtype: tuple
    """
    payload_length, barcode_length = 160, 34

    start_time = datetime.now()

    # group
    # noinspection PyTypeChecker
    sequence_groups_1 = group_composite_sequences(sequences=sequences, barcode_pairs=barcode_pairs,
                                                  former_start=0)
    # cluster and align
    sequence_groups_2 = {}
    for group_index, group in sequence_groups_1.items():
        consensus_sequences = find_consensus_sequences(raw_sequences=group,
                                                       designed_length=payload_length + barcode_length,
                                                       temp_folder=temp_folder, threshold=cluster_threshold)
        sequence_groups_2[group_index] = consensus_sequences

    # trim
    sequence_groups_3 = {}
    for group_index, group in sequence_groups_2.items():
        former_barcode, latter_barcode = barcode_pairs[group_index]
        trimmed_sequences = trim_barcodes(sequences=group,
                                          former_barcode=former_barcode, latter_barcode=latter_barcode,
                                          former_start=0)
        sequence_groups_3[group_index] = trimmed_sequences

    # decode
    method = CompositeHEDGESNanopores(resolution=8, sigma=8, step=2)
    message_groups_1 = method.decode(sequence_groups=sequence_groups_3)

    # correct
    message_groups_2 = {}
    rs_code = RSCodec(4, 40)
    for group_index, message_group in message_groups_1.items():
        candidates = []
        for message in message_group:
            if len(message) == 8 * 40:
                byte_message = packbits(message).astype(uint8)
                try:
                    decoded_result = rs_code.decode(bytes(byte_message.tolist()))
                    # handle version- or platform-dependent return formats of reedsolo.decode().
                    if isinstance(decoded_result, tuple):
                        decoded_result = decoded_result[0]
                    candidates.append((True, frombuffer(decoded_result, dtype=uint8)))
                except (ReedSolomonError, IndexError):
                    candidates.append((False, byte_message[:36]))

        count = sum([flag for flag, _ in candidates])
        if count == 1:  # select the only one.
            for flag, candidate in candidates:
                if flag:
                    # noinspection PyTypeChecker
                    message_groups_2[group_index] = unpackbits(candidate)
                    break

    # noinspection PyUnresolvedReferences
    retrieved_messages, miss_count = zeros(shape=data_shape, dtype=int), 0
    # noinspection PyTypeChecker
    for index in range(len(barcode_pairs)):
        if index in message_groups_2:
            retrieved_messages[index] = message_groups_2[index]
        else:
            miss_count += 1

    runtime = (datetime.now() - start_time).total_seconds()

    return runtime, retrieved_messages, miss_count
