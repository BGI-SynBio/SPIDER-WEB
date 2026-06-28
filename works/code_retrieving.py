from Bio import SeqIO
from collections import Counter
from copy import deepcopy
from datetime import datetime
from gzip import GzipFile
from numpy import ndarray, array, arange, random, zeros, min, sum, argsort, where
from os import system, path, makedirs, remove
from shutil import rmtree
from sys import getsizeof
from typing import Tuple, Optional

from sbw import SpiderWeb, FastGenerator, Monitor, Heap, save_data
from works.code_correcting import practice


class CountHeap(Heap):

    def __init__(self):
        """
        Heap to save sequence ranking information.
        """
        super(CountHeap, self).__init__(ranking_type="key(frequency)-value(sequence set)")
        self.__heap = {}

    def push(self,
             sequence: str):
        """
        Push raw sequence onto the heap.

        :param sequence: repaired candidate sequence.
        :type sequence: str
        """
        if sequence in self.__heap:
            self.__heap[sequence] += 1
        else:
            self.__heap[sequence] = 1

    def pop_all(self,
                top: Optional[int] = None) \
            -> set:
        """
        Pop sequences with frequency ranking.

        :param top: top number of the sequences.
        :type top: int or None

        :return: sequences.
        :rtype: list
        """
        ranked_records = sorted(self.__heap.items(), key=lambda sample: sample[1], reverse=True)
        if top is not None:
            return set([sequence[0] for sequence in ranked_records[:top]])
        else:
            return set([sequence[0] for sequence in ranked_records])

    def get_raw(self) \
            -> dict:
        """
        Get raw heap data.

        :return: raw heap data.
        :rtype: dict
        """
        return deepcopy(self.__heap)

    def get_memory(self) \
            -> int:
        """
        Get memory occupy of the heap data.

        :return: memory occupy (unit is byte).
        :rtype: int
        """
        return getsizeof(self.__heap)


class SequenceHeap(Heap):

    def __init__(self):
        """
        Heap to save sequence ranking information.
        """
        super(SequenceHeap, self).__init__(ranking_type="key(sequence)-value(frequency)")
        self.__heap = {}

    def push(self,
             sequence: str):
        """
        Push raw sequence onto the heap.

        :param sequence: repaired candidate sequence.
        :type sequence: str
        """
        used_count = 1
        for count, sequences in self.__heap.items():
            if sequence in sequences:
                # noinspection PyTestUnpassedFixture
                sequences.remove(sequence)
                used_count = count + 1
                break

        if used_count in self.__heap:
            self.__heap[used_count].add(sequence)
        else:
            self.__heap[used_count] = {sequence}

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

        for count in sorted(self.__heap.keys(), reverse=True):
            if len(self.__heap[count]) + len(retrieved_sequences) < top:
                retrieved_sequences += list(self.__heap[count])
            else:
                # noinspection PyUnresolvedReferences
                retrieved_sequences += list(self.__heap[count])[:top - len(retrieved_sequences)]
                break

        return retrieved_sequences

    def get_raw(self) \
            -> dict:
        """
        Get raw heap data.

        :return: raw heap data.
        :rtype: dict
        """
        raw_data = {}
        for sequence, count in self.__heap.items():
            if count not in raw_data:
                raw_data[count].add(sequence)
            else:
                raw_data[count] = {sequence}

        return deepcopy(raw_data)

    def get_memory(self) \
            -> int:
        """
        Get memory occupy of the heap data.

        :return: memory occupy (unit is byte).
        :rtype: int
        """
        return getsizeof(self.__heap)


def batch_generate(accessor: ndarray,
                   palindrome_map: ndarray,
                   virtuality: int,
                   length: int,
                   diversity: int,
                   batch_size: int,
                   duplicatable: bool,
                   palindromable: bool,
                   used_seed: int,
                   parent_path: str):
    """
    Generate DNA sequences in a batched way.

    :param accessor: the established coding digraph.
    :type accessor: numpy.ndarray

    :param palindrome_map: palindrome map for non-palindrome mode if required.
    :type palindrome_map: ndarray or None

    :param virtuality: virtual vertex to start encoding.
    :type virtuality: int

    :param length: length of produced DNA sequences.
    :type length: int

    :param diversity: number of produced DNA sequences.
    :type diversity: int

    :param batch_size: batch size of each search.
    :type batch_size: int

    :param duplicatable: non-duplicate mode for DNA sequences if exists.
    :type duplicatable: bool

    :param palindromable: non-palindrome mode for DNA sequences if exists.
    :type palindromable: bool

    :param used_seed: random seed used in this task.
    :type used_seed: int

    :param parent_path: path of the parent directory of output data.
    :type parent_path: str
    """
    generator = FastGenerator(accessor=accessor, palindrome_map=palindrome_map)
    sequences = generator(diversity=diversity, batch_size=batch_size, length=length, virtuality=virtuality,
                          seed=used_seed, duplicatable=duplicatable, palindromable=palindromable)
    save_data(file_path=parent_path + str(used_seed) + ".pkl", information=sequences)


def generate_retrieval_data(right_data: list,
                            error_range: ndarray,
                            maximum_reads: int,
                            seed: Optional[int] = None,
                            verbose: bool = False) \
        -> dict:
    """
    Generate retrieval data (with the uniform distribution).

    :param right_data: right sequence data.
    :type right_data: list

    :param error_range: error number range.
    :type error_range: numpy.ndarray

    :param maximum_reads: maximum reads number.
    :type maximum_reads: int

    :param seed: random seed if required.
    :type seed: int or None

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: practical retrieval data.
    :rtype: dict
    """
    records, monitor, process = {}, Monitor(), 0
    for errors in error_range:
        wrong_data = practice(right_sequences=right_data, errors=errors, reads=maximum_reads, seed=seed)
        records[errors] = wrong_data
        if verbose:
            monitor(process + 1, len(error_range))
        process += 1

    return records


def conventional_retrieval_process(raw_sequences: list,
                                   temp_folder: str,
                                   similarity: float = 0.9,
                                   default: bool = True) \
        -> Tuple[list, float]:
    # noinspection SpellCheckingInspection
    """
        Retrieve digital data based on de novo sampling retrieval.

        :param raw_sequences: raw sequence data.
        :type raw_sequences: str

        :param temp_folder: temporary folder for storing clustering and alignment data.
        :type temp_folder: str

        :param similarity: similarity threshold.
        :type similarity: float

        :param default: using muscle to align clustered sequences; otherwise using clustal-omega.
        :type default: bool

        :return: high-confidence DNA sequences and runtime of the retrieval process.
        :rtype: list, float
        """
    raw_path, o_folder, t_folder = temp_folder + "raw.fasta", temp_folder + "o/", temp_folder + "t/"

    if not path.exists(temp_folder):
        makedirs(temp_folder)
    if not path.exists(o_folder):
        makedirs(o_folder)
    if not path.exists(t_folder):
        makedirs(t_folder)

    with open(file=raw_path, mode="w") as file:
        for index, payload in enumerate(raw_sequences):
            file.write(">" + str(index) + "\n")
            file.write(payload + "\n")

    last_time = datetime.now()

    params = raw_path + " " + o_folder + " " + t_folder + " --min-seq-id " + str(similarity) + " --threads 1 -v 0"
    system("mmseqs easy-cluster " + params)  # used by Composite-Hedges-Nanopores

    clusters = {}
    with open(file=o_folder + "_cluster.tsv") as file:
        for line in file.readlines():
            cluster_index, sequence_index = [int(value) for value in line[:-1].split("\t")]
            clusters.setdefault(cluster_index, []).append(sequence_index)

    high_confidence_sequences, available_elements = set(), {"A", "C", "G", "T"}
    for cluster_index, sequence_indices in clusters.items():
        # align sequences if required.
        if len(sequence_indices) >= 3:  # at least 3 sequences in a cluster.
            clustered_path = o_folder + "clustered-" + str(cluster_index) + ".fasta"
            aligned_path = o_folder + "aligned-" + str(cluster_index) + ".fasta"

            with open(file=clustered_path, mode="w") as file:
                for sequence_index in sequence_indices:
                    file.write(">" + str(sequence_index) + "\n")
                    # noinspection PyTypeChecker
                    file.write(raw_sequences[sequence_index] + "\n")

            if default:  # more accurate but too slow (used by Composite-Hedges-Nanopores)
                params = "-align " + clustered_path + " -output " + aligned_path + " --threads 1 -quiet"
                system("muscle " + params)

            else:  # faster but less accurate.
                params = "-i " + clustered_path + " -o " + aligned_path + " --threads 1"
                system("clustalo " + params)

            candidate_set = []
            with open(file=aligned_path) as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    sequence = str(record.seq)
                    for index, symbol in enumerate(sequence):
                        if symbol in available_elements:
                            if len(candidate_set) > index + 1:
                                candidate_set[index].append(symbol)
                            else:
                                candidate_set.append([symbol])

            sequence = ""
            for candidates in candidate_set:
                element, count = Counter(candidates).most_common(1)[0]
                if count > len(sequence_indices) - len(candidates):
                    sequence += element

            high_confidence_sequences.add(sequence)

        else:
            # noinspection PyUnhashable
            high_confidence_sequences.add(raw_sequences[sequence_indices[0]])

    runtime = (datetime.now() - last_time).total_seconds()

    rmtree(temp_folder)

    return list(high_confidence_sequences), runtime


def get_candidates(raw_data: list,
                   method: SpiderWeb,
                   start: int,
                   length: Optional[int] = None) \
        -> list:
    """
    Get candidate sequences.

    :param raw_data: raw sequence data.
    :type raw_data: list

    :param method: coding digraph.
    :type method: rtr.SpiderWeb

    :param start: virtual vertex to start decoding (consistent with the encoding log).
    :type start: int

    :param length: expected sequence length if required.
    :type length: int or None

    :return: candidate sequences.
    :rtype: list
    """
    candidate_data = []
    for sequence in raw_data:
        candidates = method.repair(sequence=sequence, virtuality=start, length=length)[0]
        candidate_data += list(candidates)

    return candidate_data


def design_pcr_primer(payloads: list,
                      length: int,
                      left_mark: str = "",
                      right_mark: str = "",
                      seed: Optional[int] = None) \
        -> Tuple[str, str]:
    """
    Design primer for error-prone PCR.

    :param payloads: DNA sequences, regarded as payloads.
    :type payloads: list

    :param length: length of primer.
    :type length: int

    :param left_mark: mark between left primer and payload.
    :type left_mark: str

    :param right_mark: mark between right primer and payload.
    :type right_mark: str

    :param seed: random seed if required.
    :type seed: int or None

    :return: left primer segment and right primer segment.
    :rtype: str, str
    """
    bases = ["A", "C", "G", "T"]
    # noinspection PyTypeChecker
    random.seed(seed)
    while True:
        # noinspection PyArgumentList,PyTypeChecker
        l_segment = "".join([random.choice(bases) for _ in range(length - len(left_mark))]) + left_mark
        primer = l_segment

        if not homopolymer_run(primer=primer, maximum_run_length=3):
            continue
        if not melting_temperature(primer=primer, requirement=75):
            continue
        if not gc_content(primer=primer, lower_bound=0.45, upper_bound=0.60):
            continue
        if not gc_content(primer=primer[-8:], lower_bound=0.00, upper_bound=0.40):
            continue
        if not hairpin(primer=primer, maximum_stem_length=3):
            continue
        if not primer_difference(primer=primer, payloads=payloads, minimum_difference=11):
            continue
        if not check_duplicate(primer=primer, payloads=payloads, length=10):
            continue
        break

    # noinspection PyTypeChecker
    random.seed(seed)
    while True:
        # noinspection PyArgumentList,PyTypeChecker
        r_segment = right_mark + "".join([random.choice(bases) for _ in range(length - len(right_mark))])

        primer = r_segment.translate(str.maketrans("ACGT", "TGCA"))[::-1]

        if not homopolymer_run(primer=primer, maximum_run_length=2):
            continue
        if not melting_temperature(primer=primer, requirement=75):
            continue
        if not gc_content(primer=primer, lower_bound=0.45, upper_bound=0.60):
            continue
        if not gc_content(primer=primer[-8:], lower_bound=0.00, upper_bound=0.40):
            continue
        if not hairpin(primer=primer, maximum_stem_length=3):
            continue
        if not primer_difference(primer=primer, payloads=payloads, minimum_difference=11):
            continue
        if not check_duplicate(primer=primer, payloads=payloads, length=10):
            continue
        if levenshtein(l_segment, primer) < 11:
            continue
        break
    return l_segment, r_segment


def hairpin(primer: str,
            maximum_stem_length: int = 4) \
        -> bool:
    """
    Detect the specific hairpin structure of the primer.

    :param primer: primer sequence.
    :type primer: str

    :param maximum_stem_length: maximum acceptable stem length.
    :type maximum_stem_length: int

    :return: detection flag.
    :rtype: bool
    """
    for location_1 in range(len(primer) - (maximum_stem_length + 1) + 1):
        for location_2 in range(location_1 + (maximum_stem_length + 1), len(primer) - (maximum_stem_length + 1) + 1):
            part_1 = primer[location_1: location_1 + (maximum_stem_length + 1)]
            part_2 = primer[location_2: location_2 + (maximum_stem_length + 1)][::-1]

            found = True
            for a, b in zip(part_1, part_2):
                if a + b not in ["AT", "TA", "CG", "GC"]:  # hairpin-like matching.
                    found = False
                    break

            if found:
                return False

    return True


def melting_temperature(primer: str,
                        requirement: float) \
        -> bool:
    """
    Detect the melting temperature of the primer.

    :param primer: primer sequence.
    :type primer: str

    :param requirement: required temperature.
    :type requirement: float

    :return: detection flag.
    :rtype: bool
    """
    temperature = 2 * (primer.count("A") + primer.count("T")) + 4 * (primer.count("C") + primer.count("G"))

    return temperature >= requirement and temperature - requirement < 5


def homopolymer_run(primer: str,
                    maximum_run_length: int) \
        -> bool:
    """
    Detect the maximum homopolymer run-length of the primer.

    :param primer: primer sequence.
    :type primer: str

    :param maximum_run_length: maximum acceptable homopolymer length.
    :type maximum_run_length: int

    :return: detection flag.
    :rtype: bool
    """
    for nucleotide in ["A", "C", "G", "T"]:
        if nucleotide * (maximum_run_length + 1) in primer:
            return False

    return True


def gc_content(primer: str,
               lower_bound: float,
               upper_bound: float) \
        -> bool:
    """
    Detect the GC content limitation.

    :param primer: primer sequence.
    :type primer: str

    :param lower_bound: lower bound of GC content.
    :type lower_bound: float

    :param upper_bound: upper bound of GC content.
    :type upper_bound: float

    :return: detection flag.
    :rtype: bool
    """
    gc_rate = (primer.count("G") + primer.count("C")) / len(primer)

    if lower_bound <= gc_rate <= upper_bound:
        return True

    return False


def check_duplicate(primer: str,
                    payloads: list,
                    length: int) \
        -> bool:
    """
    Check PCR duplicates.

    :param primer: primer of error-prone PCR.
    :type primer: str

    :param payloads: data payloads.
    :type payloads: list

    :param length: duplicate length.
    :type length: int

    :return: detection flag.
    :rtype: bool
    """
    for payload in payloads:
        for location_1 in range(len(payload) - length + 1):
            segment_1 = payload[location_1: location_1 + length]
            for location_2 in range(len(primer) - length + 1):
                segment_2 = primer[location_2: location_2 + length]
                if segment_1 == segment_2:
                    return False
    return True


def primer_difference(primer: str,
                      payloads: list,
                      minimum_difference: int) \
        -> bool:
    """
    Detect the usable difference between the primer and payloads.

    :param primer: primer of error-prone PCR.
    :type primer: str

    :param payloads: data payloads.
    :type payloads: list

    :param minimum_difference: minimum Levenshtein difference between the primer and data payloads.
    :type minimum_difference: int

    :return: detection flag.
    :rtype: bool
    """
    for process, payload in enumerate(payloads):
        for position in range(len(payload) - len(primer) + 1):
            distance = levenshtein(payload[position: position + len(primer)], primer)
            if distance < minimum_difference:
                return False
    return True


def levenshtein(sequence_1: str,
                sequence_2: str) \
        -> int:
    """
    Calculate the Levenshtein distance between two sequences.

    :param sequence_1: one sequence.
    :type sequence_1: str

    :param sequence_2: another sequence.
    :type sequence_2: str

    :return: Levenshtein distance.
    :rtype: int
    """
    distances = zeros((len(sequence_1) + 1, len(sequence_2) + 1), dtype=int)

    for location in range(len(sequence_1) + 1):
        distances[location][0] = location

    for location in range(len(sequence_2) + 1):
        distances[0][location] = location

    for location_1 in range(1, len(sequence_1) + 1):
        for location_2 in range(1, len(sequence_2) + 1):
            if sequence_1[location_1 - 1] == sequence_2[location_2 - 1]:
                distances[location_1][location_2] = distances[location_1 - 1][location_2 - 1]
            else:
                a = distances[location_1][location_2 - 1]
                b = distances[location_1 - 1][location_2]
                c = distances[location_1 - 1][location_2 - 1]

                if a <= b and a <= c:
                    distances[location_1][location_2] = a + 1
                elif b <= a and b <= c:
                    distances[location_1][location_2] = b + 1
                else:
                    distances[location_1][location_2] = c + 1

    return int(distances[len(sequence_1), len(sequence_2)])


def obtain_raw_sequences(compressed_file_paths: dict,
                         tmp_path: str,
                         forward_primer: str,
                         reverse_primer: str,
                         expected_length: int,
                         tolerable_deviation: int,
                         batch: int,
                         verbose: bool = False) \
        -> list:
    """
    Retrieve raw DNA sequences from the compressed files (high-throughput sequencing).

    :param compressed_file_paths: dictionary of two complementary file path, keys are "1.fq.gz" and "2.fq.gz".
    :type compressed_file_paths: dict

    :param tmp_path: path of the temporary folder.
    :type tmp_path: str

    :param forward_primer: forward primer for amplification in DNA sequences.
    :type forward_primer: str

    :param reverse_primer: reverse primer for amplification in DNA sequences.
    :type reverse_primer: str

    :param expected_length: expected average length of retrieved DNA sequences.
    :type expected_length: int

    :param tolerable_deviation: tolerable length deviation of retrieved DNA sequences.
    :type tolerable_deviation: int

    :param batch: batch size for assembly (adjustments to the community software version).
    :type batch: int

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: retrieved DNA sequences.
    :rtype: list
    """
    monitor = Monitor()
    if verbose:
        monitor(0, 7)

    compressed_file = GzipFile(compressed_file_paths["1.fq.gz"])
    open(tmp_path + "source-1.fastq", "wb+").write(compressed_file.read())
    compressed_file.close()
    if verbose:
        monitor(1, 7)
    compressed_file = GzipFile(compressed_file_paths["2.fq.gz"])
    open(tmp_path + "source-2.fastq", "wb+").write(compressed_file.read())
    compressed_file.close()
    if verbose:
        monitor(2, 7)

    cutadapt_params = ["-g", forward_primer, "-G", reverse_primer, "--discard-untrimmed", "--quiet",
                       "-o", tmp_path + "trimmed-sub-1.f.fastq", "-p", tmp_path + "trimmed-sub-2.r.fastq",
                       tmp_path + "source-1.fastq", tmp_path + "source-2.fastq"]
    system("cutadapt " + " ".join(cutadapt_params))
    if verbose:
        monitor(3, 7)
    cutadapt_params = ["-g", reverse_primer, "-G", forward_primer, "--discard-untrimmed", "--quiet",
                       "-o", tmp_path + "trimmed-sub-1.r.fastq", "-p", tmp_path + "trimmed-sub-2.f.fastq",
                       tmp_path + "source-1.fastq", tmp_path + "source-2.fastq"]
    system("cutadapt " + " ".join(cutadapt_params))
    if verbose:
        monitor(4, 7)
    remove(tmp_path + "source-1.fastq")
    remove(tmp_path + "source-2.fastq")

    for index, flag in enumerate(["f", "r"]):
        path_1 = tmp_path + "trimmed-sub-1." + flag + ".fastq"
        path_2 = tmp_path + "trimmed-sub-2." + flag + ".fastq"
        path_3 = tmp_path + "trimmed-" + flag + ".fastq"
        with open(path_1) as in_file_1, open(path_2) as in_file_2, open(path_3, "w") as out_file:
            for line in in_file_1:
                out_file.write(line)
            for line in in_file_2:
                out_file.write(line)
        remove(path_1)
        remove(path_2)
        if verbose:
            monitor(5 + index, 7)

    length_range = (expected_length - tolerable_deviation, expected_length + tolerable_deviation)
    retrieved_sequences, simple_bases, forward_set, used_samples = [], set("ACGT"), {}, []
    with open(tmp_path + "trimmed-f.fastq") as handle:
        for index, forward_sample in enumerate(SeqIO.parse(handle, "fastq")):
            forward_set[forward_sample.id[:-2]] = forward_sample
    with open(tmp_path + "trimmed-r.fastq") as handle:
        for index, reverse_sample in enumerate(SeqIO.parse(handle, "fastq")):
            if reverse_sample.id[:-2].replace(" ", "-") in forward_set:
                lengths = [len(forward_set[reverse_sample.id[:-2]].seq), len(reverse_sample.seq)]
                if lengths[0] >= expected_length // 2 and lengths[1] >= expected_length // 2:  # has overlap
                    used_samples.append((forward_set[reverse_sample.id[:-2]], reverse_sample))
                elif lengths[0] == expected_length // 2 and lengths[1] == expected_length // 2:  # no overlap
                    former = str(forward_set[reverse_sample.id[:-2]].seq)
                    latter = str(reverse_sample.seq).translate(str.maketrans("ACGT", "TGCA"))[::-1]
                    retrieved_sequences.append(former + latter)
                del forward_set[reverse_sample.id[:-2]]  # remove for reducing the memory.
    remove(tmp_path + "trimmed-f.fastq")
    remove(tmp_path + "trimmed-r.fastq")
    if verbose:
        monitor(7, 7)

    locations = list(range(0, len(used_samples), batch))
    for index, location in enumerate(locations):
        batched_samples = used_samples[location: min([location + batch, len(used_samples)])]
        with open(tmp_path + "seq_R1.fastq", "w") as handle_f:
            for sample, _ in batched_samples:
                sample.id = sample.id[:-2].replace(" ", "-")
                SeqIO.write(sample, handle_f, "fastq")
        with open(tmp_path + "seq_R2.fastq", "w") as handle_r:
            for _, sample in batched_samples:
                sample.id = sample.id[:-2].replace(" ", "-")
                SeqIO.write(sample, handle_r, "fastq")
        # noinspection SpellCheckingInspection
        usearch_params = ["-fastq_mergepairs", tmp_path + "seq_R1.fastq",
                          "-reverse", tmp_path + "seq_R2.fastq",
                          "-fastqout", tmp_path + "merged_seq.fastq",
                          "-fastq_maxns 0", "> NUL 2>&1"]
        system("usearch " + " ".join(usearch_params))
        try:
            if path.exists(tmp_path + "merged_seq.fastq"):
                with open(tmp_path + "merged_seq.fastq") as handle_m:
                    for merged_sample in SeqIO.parse(handle_m, "fastq"):
                        sequence = str(merged_sample.seq)
                        if len(set(sequence) - simple_bases) > 0:
                            continue
                        if length_range[0] <= len(sequence) <= length_range[1]:
                            retrieved_sequences.append(sequence)
                remove(tmp_path + "merged_seq.fastq")
        except ValueError:
            pass  # unknown error
        remove(tmp_path + "seq_R1.fastq")
        remove(tmp_path + "seq_R2.fastq")

        if verbose:
            monitor(index + 1, len(locations), extra={"retrieved": len(retrieved_sequences)})

    return retrieved_sequences


def calculate_retrieval_quality(expected_sequences: list,
                                obtained_sequences: list,
                                verbose: bool = False) \
        -> list:
    """
    Calculates similarity of retrieved DNA sequences to expected DNA sequences.

    :param expected_sequences: expected DNA sequences.
    :type expected_sequences: list

    :param obtained_sequences: obtained DNA sequences.
    :type obtained_sequences: list

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: retrieval quality.
    :rtype: list
    """
    monitor, records = Monitor(), []
    for process, obtained_sequence in enumerate(obtained_sequences):
        similarities, flag = [], False
        for index, expected_sequence in enumerate(expected_sequences):
            if obtained_sequence != expected_sequence:
                similarities.append(levenshtein(sequence_1=obtained_sequence, sequence_2=expected_sequence))
            else:
                flag = True
                records.append((0, array([index])))  # find same.
                break

        if not flag:
            similarities = array(similarities)
            minimum_similarity = min(similarities)
            records.append((minimum_similarity, where(minimum_similarity == similarities)[0].astype(int)))

        if verbose:
            monitor(process + 1, len(obtained_sequences))

    return records


def obtain_samples(raw_sequences: list,
                   quality_records: list,
                   copies: ndarray,
                   batch_size: int,
                   batch_number: int,
                   seed: int,
                   verbose: bool = False) \
        -> dict:
    """
    Obtain samples from raw DNA sequences.

    :param raw_sequences: raw DNA sequences from base calling.
    :type raw_sequences: list

    :param quality_records: corresponding quality records of raw sequences.
    :type quality_records: list

    :param copies: sequence copies of the expected sequences in raw DNA sequences.
    :type copies: ndarray

    :param batch_size: expected sequence number per batch (sequence diversity x sequencing depth).
    :type batch_size: int

    :param batch_number: number of batch for retrieval experiments.
    :type batch_number: int

    :param seed: random seed.
    :type seed: int

    :param verbose: verbosity mode.
    :type verbose: bool

    :return: sampling results.
    :rtype: dict
    """
    counts = (copies / sum(copies) * batch_size + 0.5).astype(int)
    for index, (count, copy) in enumerate(zip(counts, copies)):
        if count > copy / batch_number:
            counts[index] -= int(copy / batch_number)
    order = argsort(copies)[::-1]
    for index in range(batch_size - sum(counts)):
        counts[order[index]] += 1

    monitor, groups, saved = Monitor(), {}, zeros(shape=(len(counts)), dtype=int)
    for index, (raw_sequence, quality_record) in enumerate(zip(raw_sequences, quality_records)):
        sequence_index = quality_record[1][0]
        if sequence_index in groups:
            if saved[sequence_index] < counts[sequence_index] * batch_number:
                groups[sequence_index].append(index)
                saved[sequence_index] += 1
        else:
            if saved[sequence_index] < counts[sequence_index] * batch_number:
                groups[sequence_index] = [index]
                saved[sequence_index] += 1

        # noinspection PyUnresolvedReferences
        remain = sum(counts) * batch_number - sum(saved)

        if verbose:
            monitor(index + 1, len(raw_sequences))

        if remain == 0:
            monitor(len(raw_sequences), len(raw_sequences), extra={"used": index + 1})
            break

    random.seed(seed)
    temp_records = [[[], [], []] for _ in range(batch_number)]
    for sequence_index, indices in groups.items():
        if len(indices) < counts[sequence_index] * batch_number:
            raise RuntimeError("Sequence " + str(sequence_index + 1) + " does not have enough copies.")
        for sample_index, index in enumerate(indices):
            temp_records[sample_index % batch_number][0].append(raw_sequences[index])
            temp_records[sample_index % batch_number][1].append(sequence_index)
            temp_records[sample_index % batch_number][2].append(quality_records[index][0])

    records = {}
    for batch_index in range(batch_number):
        records[batch_index + 1] = {"seq": [], "loc": [], "err": []}
        used_sequences, used_locations, errors = temp_records[batch_index]
        order = arange(batch_size, dtype=int)
        random.shuffle(order)

        for index in order:
            records[batch_index + 1]["seq"].append(used_sequences[index])
            records[batch_index + 1]["loc"].append(used_locations[index])
            records[batch_index + 1]["err"].append(errors[index])

        records[batch_index + 1]["loc"] = array(records[batch_index + 1]["loc"])
        records[batch_index + 1]["err"] = array(records[batch_index + 1]["err"])

    random.seed()

    return records
