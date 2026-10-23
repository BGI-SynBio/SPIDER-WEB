from copy import deepcopy
from datetime import datetime
from numpy import array, arange, zeros, ones, linspace, random, delete, stack, concatenate, pad
from numpy import ceil, log2, abs, sum, mean, all, where
from os import path, remove
from prettytable import PrettyTable
from pysam import AlignmentFile  # noqa  used in LINUX
from zipfile import ZipFile

from sbw import SpiderWeb, FastTranscoder, FastRepair, FastDecoder, basic_encode, basic_decode, approximate_capacity
from sbw import find_vertices, connect_digraph, select_component, count_out_degrees, reduce_3_to_2, m2i, i2m
from sbw import Monitor, load_data, save_data

from works import raw_path, tmp_path, exp_path, DNAFountain
from works import OuterCode, practice, evaluate_single_repair, evaluate_astar_search, conventional_retrieval_process
from works import produce, retrieve, test_chn, test_symmetry


def evaluate_conversion(seed: int,
                        virtuality: int):
    if not path.exists(raw_path + "supplement-simulation.conversion-1.fountain.pkl"):
        random.seed(seed)
        matrix = random.randint(0, 2, size=(20, 1000, 400))
        random.seed()

        print("Calculate information density of DNA fountain under different constraints:")
        cases, table = {}, PrettyTable(["homopolymer run-length", "GC content", "success rate", "information density"])
        monitor, process = Monitor(), 0
        for homopolymer in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for gc_bias in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                record, gc_range, success_count, information_density = [], [0.5 - gc_bias, 0.5 + gc_bias], 0, 0
                # noinspection PyTypeChecker
                for data_index, binary_messages in enumerate(matrix):
                    method = DNAFountain()
                    result = method.evaluate(binary_messages=binary_messages, window_size=10,
                                             homopolymer=homopolymer, gc_bias=gc_bias, false_threshold=1000)
                    if result[0] > 0:
                        success_count += 1
                        information_density += result[0]

                    record.append(result)
                    process += 1
                    monitor(process, 600)
                if success_count > 0:
                    information_density /= float(success_count)
                cases[(homopolymer, gc_bias)] = record
                label_1, label_2 = str(homopolymer), "[" + ("%.1f" % gc_range[0]) + "," + ("%.1f" % gc_range[1]) + "]"
                table.add_row([label_1, label_2, "%d/20" % success_count, "%.2f" % information_density])

        save_data(file_path=raw_path + "supplement-simulation.conversion-1.fountain.pkl", information=cases)

        print()
        print(table)
        print()

    if not path.exists(raw_path + "supplement-simulation.conversion-2.basic.pkl"):
        random.seed(seed)
        matrix = random.randint(0, 2, size=(20, 2000, 200))
        random.seed()

        print("Calculate information density of SPIDER-WEB under different constraints:")
        table = PrettyTable(["homopolymer run-length", "GC content", "information density 1", "information density 2"])
        cases, window_length, degrees, monitor, process = {}, 10, array([1, 2, 3, 4]), Monitor(), 0
        for homopolymer in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for gc_bias in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                gc_range = [0.5 - gc_bias, 0.5 + gc_bias]

                vertices = find_vertices(window_length=window_length, run_length=homopolymer, gc_range=gc_range)
                accessor = connect_digraph(window_length=window_length, vertices=vertices,
                                           usable_outs=degrees, usable_ins=degrees)[0]
                accessor_1, _ = select_component(entire_accessor=accessor, expected_length=200)
                code_rates_1 = []
                # noinspection PyTypeChecker
                for data_index, binary_messages in enumerate(matrix):
                    nucleotides = 0
                    for binary_message in binary_messages:
                        dna_sequence = basic_encode(binary_message=binary_message,
                                                    accessor=accessor_1, virtuality=virtuality)
                        nucleotides += len(dna_sequence)
                    code_rates_1.append(float(2000 * 200) / float(nucleotides))

                accessor_2 = accessor_1.copy()
                out_statistics = count_out_degrees(accessor_2)
                if out_statistics[3] > 0 and out_statistics[3] != sum(out_statistics):
                    accessor_2, _ = reduce_3_to_2(accessor=accessor_2, maintain_zero=False)
                code_rates_2, method = [], SpiderWeb(accessor=accessor_2)
                # noinspection PyTypeChecker
                for data_index, binary_messages in enumerate(matrix):
                    nucleotides = 0
                    for binary_message in binary_messages:
                        dna_sequence = method.encode(message=binary_message, virtuality=virtuality)
                        nucleotides += len(dna_sequence)
                    code_rates_2.append(float(2000 * 200) / float(nucleotides))

                cases[(homopolymer, gc_bias)] = (array(code_rates_1), array(code_rates_2))

                label_1, label_2 = str(homopolymer), "[" + ("%.1f" % gc_range[0]) + "," + ("%.1f" % gc_range[1]) + "]"
                value_1, value_2 = "%.2f" % mean(code_rates_1), "%.2f" % mean(code_rates_2)
                table.add_row([label_1, label_2, value_1, value_2])

                process += 1
                monitor(process, 60)

        save_data(file_path=raw_path + "supplement-simulation.conversion-2.basic.pkl", information=cases)

        print()
        print(table)
        print()

    if not path.exists(raw_path + "supplement-simulation.conversion-3.fast-tradeoffs.1.pkl"):
        with ZipFile(raw_path + "simulation.conversion-2.graphs.zip") as file:
            file.extractall(tmp_path)

        print("Quantify the impact of eliminating ambiguity on information density:")
        monitor, process, required_number = Monitor(), 0, 100

        records, window_length, degrees = {"random": [], "basic": []}, 10, array([1, 2, 3, 4])
        for index in range(10):
            for number in range(1, 101):
                accessor_path = tmp_path + str(index) + "." + str(number).zfill(len(str(required_number))) + ".npy"
                accessor = load_data(file_path=accessor_path)

                accessor_1, _ = select_component(entire_accessor=accessor, expected_length=200)
                bound_1 = approximate_capacity(accessor=accessor_1)
                accessor_2 = accessor_1.copy()
                out_statistics = count_out_degrees(accessor_1)
                if out_statistics[3] > 0 and out_statistics[3] != sum(out_statistics):
                    accessor_2, _ = reduce_3_to_2(accessor=accessor_2, maintain_zero=False)
                bound_2 = approximate_capacity(accessor=accessor_2)

                if bound_1 - bound_2 > 0.01:
                    out_statistics = count_out_degrees(accessor)
                    records["random"].append([out_statistics[3] / sum(out_statistics), bound_1 - bound_2])
                remove(accessor_path)
                monitor(process + 1, 1060)
                process += 1

        for homopolymer in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for gc_bias in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                gc_range = [0.5 - gc_bias, 0.5 + gc_bias]
                vertices = find_vertices(window_length=window_length, run_length=homopolymer, gc_range=gc_range)
                accessor = connect_digraph(window_length=window_length, vertices=vertices,
                                           usable_outs=degrees, usable_ins=degrees)[0]
                accessor_1, _ = select_component(entire_accessor=accessor, expected_length=200)
                bound_1 = approximate_capacity(accessor=accessor_1)
                accessor_2 = accessor_1.copy()
                out_statistics = count_out_degrees(accessor_1)
                if out_statistics[3] > 0 and out_statistics[3] != sum(out_statistics):
                    accessor_2, _ = reduce_3_to_2(accessor=accessor_2, maintain_zero=False)
                bound_2 = approximate_capacity(accessor=accessor_2)
                if bound_1 - bound_2 > 0.01:
                    out_statistics = count_out_degrees(accessor)
                    records["basic"].append([out_statistics[3] / sum(out_statistics), bound_1 - bound_2])
                monitor(process + 1, 1060)
                process += 1

        save_data(raw_path + "supplement-simulation.conversion-3.fast-tradeoffs.1.pkl", records)

    if not path.exists(raw_path + "supplement-simulation.conversion-4.fast-tradeoffs.2.pkl"):
        with ZipFile(raw_path + "simulation.conversion-2.graphs.zip") as file:
            file.extractall(tmp_path)

        print("Quantify the impact of eliminating ambiguity on decoding speed:")
        monitor, process, sample_number, required_number = Monitor(), 0, 10000, 100

        records, window_length, degrees = {"random": [], "basic": []}, 10, array([1, 2, 3, 4])
        for index in range(10):
            for number in range(1, 101):
                accessor_path = tmp_path + str(index) + "." + str(number).zfill(len(str(required_number))) + ".npy"
                accessor = load_data(file_path=accessor_path)
                accessor_1, _ = select_component(entire_accessor=accessor, expected_length=200)
                out_statistics = count_out_degrees(accessor_1)
                if out_statistics[3] > 0 and out_statistics[3] != sum(out_statistics):
                    bound_1 = approximate_capacity(accessor=accessor_1)
                    accessor_2, _ = reduce_3_to_2(accessor=accessor_1, maintain_zero=False)
                    method = SpiderWeb(accessor=accessor_2)
                    bound_2 = approximate_capacity(accessor=accessor_2)
                    if bound_1 == 0 or bound_2 == 0:
                        continue
                    density_reduction, runtime_improvements = bound_2 / bound_1, []
                    for message_length in [100, 200, 300, 400, 500]:
                        random.seed(seed)
                        binary_messages = random.randint(0, 2, size=(sample_number, message_length))
                        dna_sequences = []
                        # noinspection PyTypeChecker
                        for binary_message in binary_messages:
                            dna_sequences.append(basic_encode(binary_message=binary_message,
                                                              accessor=accessor_1, virtuality=0))
                        start_time = datetime.now()
                        for dna_sequence in dna_sequences:
                            basic_decode(dna_sequence=dna_sequence, accessor=accessor_1, virtuality=0,
                                         bit_length=message_length)
                        runtime_1 = (datetime.now() - start_time).total_seconds()

                        dna_sequences = []
                        # noinspection PyTypeChecker
                        for binary_message in binary_messages:
                            dna_sequences.append(method.encode(message=binary_message, virtuality=0))
                        start_time = datetime.now()
                        for dna_sequence in dna_sequences:
                            method.decode(sequence=dna_sequence, length=message_length, virtuality=0)
                        runtime_2 = (datetime.now() - start_time).total_seconds()
                        runtime_improvements.append(runtime_1 / runtime_2)
                    records["random"].append((density_reduction, runtime_improvements))
                monitor(process + 1, 1060)
                process += 1

        for homopolymer in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for gc_bias in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                gc_range = [0.5 - gc_bias, 0.5 + gc_bias]
                vertices = find_vertices(window_length=window_length, run_length=homopolymer, gc_range=gc_range)
                accessor = connect_digraph(window_length=window_length, vertices=vertices,
                                           usable_outs=degrees, usable_ins=degrees)[0]
                accessor_1, _ = select_component(entire_accessor=accessor, expected_length=200)
                out_statistics = count_out_degrees(accessor_1)
                if out_statistics[3] > 0 and out_statistics[3] != sum(out_statistics):
                    bound_1 = approximate_capacity(accessor=accessor_1)
                    accessor_2, _ = reduce_3_to_2(accessor=accessor_1, maintain_zero=False)
                    method = SpiderWeb(accessor=accessor_2)
                    bound_2 = approximate_capacity(accessor=accessor_2)
                    if bound_1 == 0 or bound_2 == 0:
                        continue
                    density_reduction, runtime_improvements = bound_2 / bound_1, []
                    for message_length in [100, 200, 300, 400, 500]:
                        random.seed(seed)
                        binary_messages = random.randint(0, 2, size=(sample_number, message_length))
                        dna_sequences = []
                        # noinspection PyTypeChecker
                        for binary_message in binary_messages:
                            dna_sequences.append(basic_encode(binary_message=binary_message,
                                                              accessor=accessor_1, virtuality=0))
                        start_time = datetime.now()
                        for dna_sequence in dna_sequences:
                            basic_decode(dna_sequence=dna_sequence, accessor=accessor_1, virtuality=0,
                                         bit_length=message_length)
                        runtime_1 = (datetime.now() - start_time).total_seconds()

                        dna_sequences = []
                        # noinspection PyTypeChecker
                        for binary_message in binary_messages:
                            dna_sequences.append(method.encode(message=binary_message, virtuality=0))
                        start_time = datetime.now()
                        for dna_sequence in dna_sequences:
                            method.decode(sequence=dna_sequence, length=message_length, virtuality=0)
                        runtime_2 = (datetime.now() - start_time).total_seconds()
                        runtime_improvements.append(runtime_1 / runtime_2)
                    records["basic"].append((density_reduction, runtime_improvements))
                monitor(process + 1, 1060)
                process += 1

        save_data(raw_path + "supplement-simulation.conversion-4.fast-tradeoffs.2.pkl", records)

    if not path.exists(raw_path + "supplement-simulation.conversion-5.selection.pkl"):
        print("Calculate information density of the special case start from different vertices:")
        accessor, records, repeats = load_data(raw_path + "generation-2.special-case.pkl")[0], {}, 100
        method, monitor = SpiderWeb(accessor=accessor), Monitor()
        vertex_number = len(method.get_params()["vertices"])
        for length in [100, 200, 300, 400, 500, 600]:
            print("message length = " + str(length))
            random.seed(seed)
            matrix = random.randint(0, 2, size=(repeats, length))
            random.seed()
            code_rates = []
            for virtuality in range(vertex_number):
                nucleotides = 0
                # noinspection PyTypeChecker
                for binary_message in matrix:
                    dna_sequence = method.encode(message=binary_message, virtuality=virtuality)
                    nucleotides += len(dna_sequence)
                code_rates.append(float(length * repeats) / float(nucleotides))
                monitor(virtuality + 1, vertex_number)
            code_rates = array(code_rates)
            average_rate = mean(code_rates)
            cv_values = abs(code_rates - average_rate) / average_rate
            records[length] = cv_values

        save_data(raw_path + "supplement-simulation.conversion-5.selection.pkl", records)


def evaluate_correction(seed: int,
                        virtuality: int):
    accessor = load_data(raw_path + "generation-2.special-case.pkl")[0]
    method, monitor, length, maximum_error = SpiderWeb(accessor=accessor), Monitor(), 200, 8

    if not path.exists(path=raw_path + "supplement-simulation.correction-1.single.pkl"):
        right_sequences, records = load_data(file_path=raw_path + "simulation.correction-1.sequences.pkl"), {}
        table = PrettyTable(["errors",
                             "containing count 1", "screening count 1", "runtime 1",
                             "containing count 2", "screening count 2", "runtime 2"])

        for errors in range(maximum_error + 1):
            source_path = tmp_path + "s" + str(errors) + ".csv"
            target_path = tmp_path + "t" + str(errors) + ".csv"
            wrong_sequences = practice(right_sequences=right_sequences, errors=errors, seed=seed)
            with open(source_path, "w") as file:
                for wrong_sequence in wrong_sequences:
                    file.write(wrong_sequence + "\n")

            fast_retriever_1 = FastRepair(accessor=accessor, virtuality=virtuality,
                                          duplicatable=True, palindromable=True, length=length)
            fast_retriever_1.start(sequence_paths=(source_path, target_path))
            retrieved_data, record = fast_retriever_1.retrieve_from_file(diversity=1000000, need_order=True)
            runtime_1 = record["runtime"]
            statistics = zeros(shape=(len(right_sequences), 2), dtype=int)
            for index, (right_sequence, repaired_sequences) in enumerate(zip(right_sequences, retrieved_data)):
                statistics[index, 0] = len(repaired_sequences)
                if right_sequence in repaired_sequences:
                    statistics[index, 1] = 1
            containing_count_1, screening_count_1 = sum(statistics[:, 1]), len(where(statistics[:, 0] == 0)[0])
            remove(target_path)

            fast_retriever_2 = FastRepair(accessor=method.get_params()["accessor"], virtuality=virtuality,
                                          duplicatable=False, palindromable=False, length=length)
            fast_retriever_2.start(sequence_paths=(source_path, target_path))
            retrieved_data, record = fast_retriever_2.retrieve_from_file(diversity=1000000, need_order=True)
            runtime_2 = record["runtime"]
            statistics = zeros(shape=(len(right_sequences), 2), dtype=int)
            for index, (right_sequence, repaired_sequences) in enumerate(zip(right_sequences, retrieved_data)):
                statistics[index, 0] = len(repaired_sequences)
                if right_sequence in repaired_sequences:
                    statistics[index, 1] = 1
            containing_count_2, screening_count_2 = sum(statistics[:, 1]), len(where(statistics[:, 0] == 0)[0])
            remove(target_path)

            remove(source_path)
            records[errors] = ([containing_count_1, screening_count_1, runtime_1],
                               [containing_count_2, screening_count_2, runtime_2])

            table.add_row([errors,
                           containing_count_1, screening_count_1, "%.4f" % runtime_1,
                           containing_count_2, screening_count_2, "%.4f" % runtime_2])

        save_data(file_path=raw_path + "supplement-simulation.correction-1.single.pkl", information=records)

        print("Correct 1,000,000 DNA sequences under different error rates (using C++):")
        print(table)
        print()

    if not path.exists(path=raw_path + "supplement-simulation.correction-2.sequences.pkl"):
        print("Produce 1,000 DNA sequences in each vertex for additional correction evaluation:")
        vertex_number, right_data = len(method.get_params()["vertices"]), {}
        for new_virtuality in range(vertex_number):
            right_data[new_virtuality] = method.produce(virtuality=new_virtuality, length=length, diversity=1000,
                                                        duplicatable=True, palindromable=True, seed=seed)[0]
            monitor(new_virtuality + 1, vertex_number)
        print()

        save_data(file_path=raw_path + "supplement-simulation.correction-2.sequences.pkl", information=right_data)

    if not path.exists(path=raw_path + "supplement-simulation.correction-3.uniformity.pkl"):
        right_data, records = load_data(file_path=raw_path + "supplement-simulation.correction-2.sequences.pkl"), {}
        table = PrettyTable(["virtual vertex", "errors",
                             "containing count 1", "screening count 1", "containing count 2", "screening count 2"])
        print("Correct 1,000 DNA sequences started from each vertex under different error rates:")
        for process, (used_virtuality, right_sequences) in enumerate(right_data.items()):
            record = []
            if len(right_sequences) > 0:
                for errors in arange(maximum_error + 1):
                    wrong_sequences = practice(right_sequences=right_sequences, errors=errors, seed=seed)
                    repaired_data_1, _ = evaluate_single_repair(method=method, virtuality=used_virtuality,
                                                                duplicatable=True, palindromable=True,
                                                                right_sequences=right_sequences,
                                                                wrong_sequences=wrong_sequences)
                    repaired_data_2, _ = evaluate_single_repair(method=method, virtuality=used_virtuality,
                                                                duplicatable=False, palindromable=False,
                                                                right_sequences=right_sequences,
                                                                wrong_sequences=wrong_sequences)

                    statistics_1 = zeros(shape=(len(right_sequences), 2), dtype=int)
                    for index, (right_sequence, (sequences, _)) in enumerate(zip(right_sequences, repaired_data_1)):
                        statistics_1[index, 0] = len(sequences)
                        if right_sequence in sequences:
                            statistics_1[index, 1] = 1
                    statistics_2 = zeros(shape=(len(right_sequences), 2), dtype=int)
                    for index, (right_sequence, (sequences, _)) in enumerate(zip(right_sequences, repaired_data_2)):
                        statistics_2[index, 0] = len(sequences)
                        if right_sequence in sequences:
                            statistics_2[index, 1] = 1

                    containing_count_1 = sum(statistics_1[:, 1])
                    screening_count_1 = len(where(statistics_1[:, 0] == 0)[0])
                    containing_count_2 = sum(statistics_2[:, 1])
                    screening_count_2 = len(where(statistics_2[:, 0] == 0)[0])

                    record.append([errors,
                                   containing_count_1, screening_count_1, containing_count_2, screening_count_2])
                    table.add_row([used_virtuality, errors,
                                   containing_count_1, screening_count_1, containing_count_2, screening_count_2])

                records[used_virtuality] = array(record)
            monitor(process + 1, len(right_data))

        save_data(file_path=raw_path + "supplement-simulation.correction-3.uniformity.pkl", information=records)

        print()
        print(table)
        print()

    if not path.exists(path=raw_path + "supplement-simulation.correction-4.astar.pkl"):
        right_sequences = load_data(file_path=raw_path + "simulation.correction-1.sequences.pkl")[:400]

        records = evaluate_astar_search(accessor=accessor, dna_sequences=right_sequences, virtuality=virtuality,
                                        maximum_error=maximum_error, seed=seed)

        save_data(file_path=raw_path + "supplement-simulation.correction-4.astar.pkl", information=records)


def evaluate_retrieving(seed: int,
                        virtuality: int):
    if not path.exists(path=raw_path + "supplement-experiment.retrieving-1.default.pkl"):
        print("Evaluate default setting of the conventional data retrieval pipeline:")
        expected_data, records = load_data(file_path=raw_path + "experiment.retrieval-1.sequences.pkl")[0], {}
        sequence_set = {"0001-1000": set(expected_data[:1000]),
                        "1001-2000": set(expected_data[1000:2000]),
                        "2001-3000": set(expected_data[2000:])}

        sample_data, sample_number, batch_number = load_data(raw_path + "experiment.retrieval-6.samples.pkl"), 1000, 3
        monitor, process, total = Monitor(), 0, 108
        for index_scope in ["0001-1000", "1001-2000", "2001-3000"]:
            for pcr_number in [0, 1, 2]:
                for batch_index in range(batch_number):
                    temp_folder = tmp_path + index_scope + "." + str(pcr_number) + "." + str(batch_index)
                    raw_sequence_data = sample_data[(index_scope, "#" + str(pcr_number))][batch_index + 1]["seq"]
                    for depth in arange(10, 41, 10, dtype=int):
                        number = depth * 1000
                        if number < len(raw_sequence_data):
                            raw_sequences = raw_sequence_data[:number]
                        else:
                            raw_sequences = raw_sequence_data
                        retrieved_data, runtime = conventional_retrieval_process(raw_sequences=raw_sequences,
                                                                                 temp_folder=temp_folder)
                        success_count = len(set(sequence_set[index_scope]).intersection(set(retrieved_data[:1000])))
                        contain_count = len(set(sequence_set[index_scope]).intersection(set(retrieved_data)))
                        total_count = len(retrieved_data)
                        records[(index_scope, pcr_number, depth, batch_index + 1)] = (success_count, runtime,
                                                                                      contain_count, total_count)
                        monitor(process + 1, total)
                        process += 1

        save_data(file_path=raw_path + "supplement-experiment.retrieving-1.default.pkl", information=records)

    if not path.exists(path=raw_path + "supplement-experiment.retrieving-2.error-profile.pkl"):
        print("Calculate high-throughput sequencing error:")
        error_counts, location_counts, monitor = {}, {}, Monitor()
        with AlignmentFile(exp_path + "2/select.align.sort.bam", "rb") as bam_file:
            total, ignore = bam_file.count(), 0
            for index, read in enumerate(bam_file.fetch()):
                monitor(index + 1, total, extra={"ignore": ignore})
                try:
                    if read.is_unmapped or read.is_secondary or read.is_supplementary:
                        ignore += 1
                        continue

                    reference_length = bam_file.get_reference_length(read.reference_name)
                    label, start = read.reference_name, read.reference_start

                    # construct pairwise alignment with qualities.
                    expected_string, obtained_string = read.get_reference_sequence(), read.query_sequence
                    expected_alignment, obtained_alignment, obtained_qualities = [], [], []
                    for obtained_location, expected_location in read.get_aligned_pairs():
                        if obtained_location is None:  # deletion
                            expected_alignment.append(expected_string[expected_location - start].upper())
                            obtained_alignment.append("-")
                            obtained_qualities.append(60)
                        elif expected_location is None:  # insertion
                            expected_alignment.append("-")
                            obtained_alignment.append(obtained_string[obtained_location].upper())
                            obtained_qualities.append(read.query_qualities[obtained_location])
                        else:
                            expected_alignment.append(expected_string[expected_location - start].upper())
                            obtained_alignment.append(obtained_string[obtained_location].upper())
                            obtained_qualities.append(read.query_qualities[obtained_location])

                    # handle soft/hard clipping in left terminal.
                    if read.cigartuples[0][0] in (4, 5):
                        length, remove_count = read.cigartuples[0][1], 0
                        while remove_count < length:
                            if expected_alignment[0] in "ACGT":
                                start += 1
                            if obtained_alignment[0] in "ACGT":
                                remove_count += 1
                            expected_alignment = expected_alignment[1:]
                            obtained_alignment = obtained_alignment[1:]
                            obtained_qualities = obtained_qualities[1:]

                    # handle soft/hard clipping in right terminal.
                    if read.cigartuples[-1][0] in (4, 5):
                        length, remove_count = read.cigartuples[-1][1], 0
                        while remove_count < length:
                            if obtained_alignment[-1] in "ACGT":
                                remove_count += 1
                            expected_alignment = expected_alignment[:-1]
                            obtained_alignment = obtained_alignment[:-1]
                            obtained_qualities = obtained_qualities[:-1]

                    expected_alignment, obtained_alignment = "".join(expected_alignment), "".join(obtained_alignment)

                    counts, locations, offset = zeros(shape=(4,)), zeros(shape=(2, reference_length)), 0
                    for expected, obtained, quality in zip(expected_alignment, obtained_alignment, obtained_qualities):
                        if quality > 10 or start + offset >= reference_length:
                            if expected != obtained:
                                if (expected == "-") and (obtained in "ACGT"):  # insertion
                                    counts[1] += 1
                                    counts[3] += 1
                                    locations[0, start + offset] += 1
                                    locations[1, start + offset] += 1
                                elif (expected in "ACGT") and (obtained == "-"):  # deletion
                                    counts[2] += 1
                                    counts[3] += 1
                                    locations[0, start + offset] += 1
                                    locations[1, start + offset] += 1
                                elif (expected in "ACGT") and (obtained in "ACGT"):  # substitution
                                    counts[0] += 1
                                    counts[3] += 1
                                    locations[0, start + offset] += 1
                                    locations[1, start + offset] += 1
                            else:
                                counts[3] += 1
                                locations[1, start + offset] += 1

                        if expected != "-":
                            offset += 1

                    if read.reference_name in error_counts:
                        error_counts[read.reference_name].append(counts.tolist())
                    else:
                        error_counts[read.reference_name] = [counts.tolist()]

                    if read.reference_name in location_counts:
                        location_counts[read.reference_name] += locations
                    else:
                        location_counts[read.reference_name] = locations

                except ValueError:
                    pass

        save_data(file_path=raw_path + "supplement-experiment.retrieving-2.error-profile.pkl",
                  information=(error_counts, location_counts))

    if not path.exists(path=raw_path + "supplement-simulation.retrieving-1.outer-baseline.pkl"):
        diversity, address_length, bit_length, sample_number, skipping_range = 1000, 20, 80, 100, 100
        loss_rates, redundant_rates = linspace(0.00, 0.10, 11), linspace(0.01, 0.50)

        data_groups, monitor = [], Monitor()
        random.seed(seed=seed)
        for _ in range(sample_number):
            data_groups.append(random.randint(0, 2, size=(diversity, bit_length)))

        records_1, method, current = {}, OuterCode(method="xor"), 0
        print("Evaluate the basic performance of exclusive-or parity without any inner code:")
        for redundant_rate in redundant_rates:
            method.set_params(redundancy=redundant_rate)
            for loss_rate in loss_rates:
                cases = zeros(shape=(sample_number,))
                loss_number = int(loss_rate * (diversity * (1 + redundant_rate)))
                for data_index, expected_messages in enumerate(data_groups):
                    stored_messages = method.insert(binary_messages=expected_messages.copy(),
                                                    address_length=address_length, skipping_range=skipping_range)
                    if loss_number > 0:
                        removed_indices = random.choice(len(expected_messages), size=loss_number, replace=False)
                        practical_messages = delete(stored_messages, removed_indices, axis=0)
                    else:
                        practical_messages = stored_messages.copy()
                    retrieved_data = method.repair(binary_messages=practical_messages, address_length=address_length,
                                                   original_number=diversity, skipping_range=skipping_range)
                    # noinspection PyUnresolvedReferences
                    cases[data_index] = sum(all(retrieved_data == expected_messages, axis=1)) / float(diversity)
                monitor(current + 1, 550,
                        {"set": "%.2f" % redundant_rate, "loss": "%.2f" % loss_rate, "obtain": "%.3f" % mean(cases)})
                current += 1
                records_1[("%.2f" % redundant_rate, "%.2f" % loss_rate)] = cases
        print()

        records_2, method, current = {}, OuterCode(method="reed-solomon"), 0
        block_size = 100
        print("Evaluate the basic performance of reed-solomon code without any inner code:")
        for redundant_rate in redundant_rates:
            method.set_params(redundancy=redundant_rate)
            for loss_rate in loss_rates:
                cases = zeros(shape=(sample_number,))
                loss_number = int(loss_rate * (diversity * (1 + redundant_rate)))
                for data_index, expected_messages in enumerate(data_groups):
                    stored_messages = method.insert(binary_messages=expected_messages.copy(),
                                                    address_length=address_length, skipping_range=skipping_range,
                                                    block_size=block_size)
                    if loss_number > 0:
                        removed_indices = random.choice(len(expected_messages), size=loss_number, replace=False)
                        practical_messages = delete(stored_messages, removed_indices, axis=0)
                    else:
                        practical_messages = stored_messages.copy()
                    retrieved_data = method.repair(binary_messages=practical_messages, address_length=address_length,
                                                   original_number=diversity, skipping_range=skipping_range,
                                                   block_size=block_size)
                    # noinspection PyUnresolvedReferences
                    cases[data_index] = sum(all(retrieved_data == expected_messages, axis=1)) / float(diversity)
                monitor(current + 1, 550,
                        {"set": "%.2f" % redundant_rate, "loss": "%.2f" % loss_rate, "obtain": "%.3f" % mean(cases)})
                current += 1
                records_2[("%.2f" % redundant_rate, "%.2f" % loss_rate)] = cases
        print()

        records_3, method, current = {}, OuterCode(method="fountain"), 0
        fountain_params = {"delta": 0.001, "cdist": 0.025}
        print("Evaluate the basic performance of fountain code without any inner code:")
        for redundant_rate in redundant_rates:
            method.set_params(redundancy=redundant_rate, parameters=fountain_params)
            for loss_rate in loss_rates:
                cases = zeros(shape=(sample_number,))
                loss_number = int(loss_rate * (diversity * (1 + redundant_rate)))
                for data_index, expected_messages in enumerate(data_groups):
                    stored_messages = method.insert(binary_messages=expected_messages.copy(),
                                                    address_length=address_length, skipping_range=skipping_range)
                    if loss_number > 0:
                        removed_indices = random.choice(len(expected_messages), size=loss_number, replace=False)
                        practical_messages = delete(stored_messages, removed_indices, axis=0)
                    else:
                        practical_messages = stored_messages.copy()
                    retrieved_data = method.repair(binary_messages=practical_messages, address_length=address_length,
                                                   original_number=diversity, skipping_range=skipping_range)
                    # noinspection PyUnresolvedReferences
                    cases[data_index] = sum(all(retrieved_data == expected_messages, axis=1)) / float(diversity)
                monitor(current + 1, 550,
                        {"set": "%.2f" % redundant_rate, "loss": "%.2f" % loss_rate, "obtain": "%.3f" % mean(cases)})
                current += 1
                records_3[("%.2f" % redundant_rate, "%.2f" % loss_rate)] = cases
        print()

        save_data(file_path=raw_path + "supplement-simulation.retrieving-1.outer-baseline.pkl",
                  information={"xor": records_1, "reed-solomon": records_2, "fountain": records_3})

    if not path.exists(path=raw_path + "supplement-simulation.retrieving-2.outer-spider.pkl"):
        diversity, address_length, bit_length, sample_number, skipping_range = 1000, 20, 80, 100, 100
        loss_rates, redundant_rates = linspace(0.00, 0.10, 11), linspace(0.01, 0.50)
        inner_code = FastTranscoder(accessor=load_data(raw_path + "generation-2.special-case.pkl")[0])

        data_groups, monitor = [], Monitor()
        random.seed(seed=seed)
        for _ in range(sample_number):
            data_groups.append(random.randint(0, 2, size=(diversity, bit_length)))

        records_1, outer_code, current = {}, OuterCode(method="xor"), 0
        print("Evaluate the performance of exclusive-or parity with SPIDER-WEB:")
        for redundant_rate in redundant_rates:
            outer_code.set_params(redundancy=redundant_rate)
            for loss_rate in loss_rates:
                cases = zeros(shape=(sample_number,))
                loss_number = int(loss_rate * (diversity * (1 + redundant_rate)))
                for data_index, expected_messages in enumerate(data_groups):
                    stored_messages = outer_code.insert(binary_messages=expected_messages,
                                                        address_length=address_length, skipping_range=skipping_range)
                    sequences, _ = inner_code.encode(messages=stored_messages, virtuality=virtuality, seed=seed)

                    if loss_number > 0:
                        removed_indices = random.choice(len(expected_messages), size=loss_number, replace=False)
                        practical_sequences = [seq for i, seq in enumerate(sequences) if i not in removed_indices]
                    else:
                        practical_sequences = deepcopy(sequences)

                    virtualities = ones(shape=(len(practical_sequences),), dtype=int) * virtuality
                    used_messages = inner_code.decode(sequences=practical_sequences, virtualities=virtualities,
                                                      length=address_length + bit_length)

                    retrieved_data = outer_code.repair(binary_messages=used_messages, address_length=address_length,
                                                       original_number=diversity, skipping_range=skipping_range)
                    cases[data_index] = sum(all(retrieved_data == expected_messages, axis=1)) / float(diversity)
                monitor(current + 1, 550,
                        {"set": "%.2f" % redundant_rate, "loss": "%.2f" % loss_rate, "obtain": "%.3f" % mean(cases)})
                current += 1
                records_1[("%.2f" % redundant_rate, "%.2f" % loss_rate)] = cases
        print()

        records_2, outer_code, current = {}, OuterCode(method="reed-solomon"), 0
        block_size = 100
        print("Evaluate the performance of reed-solomon code with SPIDER-WEB:")
        for redundant_rate in redundant_rates:
            outer_code.set_params(redundancy=redundant_rate)
            for loss_rate in loss_rates:
                cases = zeros(shape=(sample_number,))
                loss_number = int(loss_rate * (diversity * (1 + redundant_rate)))
                for data_index, expected_messages in enumerate(data_groups):
                    stored_messages = outer_code.insert(binary_messages=expected_messages,
                                                        address_length=address_length, skipping_range=skipping_range,
                                                        block_size=block_size)
                    sequences, _ = inner_code.encode(messages=stored_messages, virtuality=virtuality, seed=seed)

                    if loss_number > 0:
                        removed_indices = random.choice(len(expected_messages), size=loss_number, replace=False)
                        practical_sequences = [seq for i, seq in enumerate(sequences) if i not in removed_indices]
                    else:
                        practical_sequences = deepcopy(sequences)

                    virtualities = ones(shape=(len(practical_sequences),), dtype=int) * virtuality
                    used_messages = inner_code.decode(sequences=practical_sequences, virtualities=virtualities,
                                                      length=address_length + bit_length)

                    retrieved_data = outer_code.repair(binary_messages=used_messages, address_length=address_length,
                                                       original_number=diversity, skipping_range=skipping_range,
                                                       block_size=block_size)
                    cases[data_index] = sum(all(retrieved_data == expected_messages, axis=1)) / float(diversity)
                monitor(current + 1, 550,
                        {"set": "%.2f" % redundant_rate, "loss": "%.2f" % loss_rate, "obtain": "%.3f" % mean(cases)})
                current += 1
                records_2[("%.2f" % redundant_rate, "%.2f" % loss_rate)] = cases
        print()

        records_3, outer_code, current = {}, OuterCode(method="fountain"), 0
        fountain_params = {"delta": 0.001, "cdist": 0.025}
        print("Evaluate the performance of fountain code with SPIDER-WEB:")
        for redundant_rate in redundant_rates:
            outer_code.set_params(redundancy=redundant_rate, parameters=fountain_params)
            for loss_rate in loss_rates:
                cases = zeros(shape=(sample_number,))
                loss_number = int(loss_rate * (diversity * (1 + redundant_rate)))
                for data_index, expected_messages in enumerate(data_groups):
                    stored_messages = outer_code.insert(binary_messages=expected_messages,
                                                        address_length=address_length, skipping_range=skipping_range)
                    sequences, _ = inner_code.encode(messages=stored_messages, virtuality=virtuality, seed=seed)

                    if loss_number > 0:
                        removed_indices = random.choice(len(expected_messages), size=loss_number, replace=False)
                        practical_sequences = [seq for i, seq in enumerate(sequences) if i not in removed_indices]
                    else:
                        practical_sequences = deepcopy(sequences)

                    virtualities = ones(shape=(len(practical_sequences),), dtype=int) * virtuality
                    used_messages = inner_code.decode(sequences=practical_sequences, virtualities=virtualities,
                                                      length=address_length + bit_length)

                    retrieved_data = outer_code.repair(binary_messages=used_messages, address_length=address_length,
                                                       original_number=diversity, skipping_range=skipping_range)
                    cases[data_index] = sum(all(retrieved_data == expected_messages, axis=1)) / float(diversity)
                monitor(current + 1, 550,
                        {"set": "%.2f" % redundant_rate, "loss": "%.2f" % loss_rate, "obtain": "%.3f" % mean(cases)})
                current += 1
                records_3[("%.2f" % redundant_rate, "%.2f" % loss_rate)] = cases
        print()

        save_data(file_path=raw_path + "supplement-simulation.retrieving-2.outer-spider.pkl",
                  information={"xor": records_1, "reed-solomon": records_2, "fountain": records_3})

    if not path.exists(path=raw_path + "supplement-simulation.retrieving-3.outer-preparation.pkl"):
        diversity, address_length, bit_length, sample_number, sequence_length = 1000, 20, 80, 10, 200
        skipping_range, redundant_rates = 100, linspace(0.01, 0.50)

        inner_code, monitor = SpiderWeb(accessor=load_data(raw_path + "generation-2.special-case.pkl")[0]), Monitor()

        data_groups, monitor = [], Monitor()
        random.seed(seed=seed)
        for _ in range(sample_number):
            original_messages = random.randint(0, 2, size=(diversity, bit_length))
            data_groups.append(original_messages)

        records_0, current, addresses = {}, 0, i2m(integers=arange(diversity), length=address_length, base_size=2)
        print("Generate SPIDER-WEB:")
        for data_index, payloads in enumerate(data_groups):
            sequences = []
            for binary_message in concatenate((addresses, payloads), axis=1):
                sequences.append(inner_code.encode(message=binary_message, virtuality=virtuality,
                                                   length_fixed=sequence_length))
            records_0[data_index] = sequences
            monitor(current + 1, sample_number)
            current += 1

        records_1, outer_code, current = {}, OuterCode(method="xor"), 0
        print("Generate exclusive-or parity for SPIDER-WEB:")
        for redundant_rate in redundant_rates:
            outer_code.set_params(redundancy=redundant_rate)
            for data_index, payloads in enumerate(data_groups):
                stored_messages = outer_code.insert(binary_messages=payloads, address_length=address_length,
                                                    skipping_range=skipping_range)
                sequences = []
                for binary_message in stored_messages:
                    sequences.append(inner_code.encode(message=binary_message, virtuality=virtuality,
                                                       length_fixed=sequence_length))
                records_1[(data_index, str(int(redundant_rate * 100 + 0.5)) + "%")] = sequences
                monitor(current + 1, sample_number * len(redundant_rates))
                current += 1
        print()

        records_2, outer_code, current = {}, OuterCode(method="reed-solomon"), 0
        block_size = 100
        print("Generate reed-solomon code for SPIDER-WEB:")
        for redundant_rate in redundant_rates:
            outer_code.set_params(redundancy=redundant_rate)
            for data_index, payloads in enumerate(data_groups):
                stored_messages = outer_code.insert(binary_messages=payloads, address_length=address_length,
                                                    skipping_range=skipping_range, block_size=block_size)
                sequences = []
                for binary_message in stored_messages:
                    sequences.append(inner_code.encode(message=binary_message, virtuality=virtuality,
                                                       length_fixed=sequence_length))
                records_2[(data_index, str(int(redundant_rate * 100 + 0.5)) + "%")] = sequences
                monitor(current + 1, sample_number * len(redundant_rates))
                current += 1
        print()

        records_3, outer_code, current = {}, OuterCode(method="fountain"), 0
        fountain_params = {"delta": 0.001, "cdist": 0.025}
        print("Generate fountain code for SPIDER-WEB:")
        for redundant_rate in redundant_rates:
            outer_code.set_params(redundancy=redundant_rate, parameters=fountain_params)
            for data_index, payloads in enumerate(data_groups):
                stored_messages = outer_code.insert(binary_messages=payloads, address_length=address_length,
                                                    skipping_range=skipping_range)
                sequences = []
                for binary_message in stored_messages:
                    sequences.append(inner_code.encode(message=binary_message, virtuality=virtuality,
                                                       length_fixed=sequence_length))
                records_3[(data_index, str(int(redundant_rate * 100 + 0.5)) + "%")] = sequences
                monitor(current + 1, sample_number * len(redundant_rates))
                current += 1
        print()

        save_data(file_path=raw_path + "supplement-simulation.retrieving-3.outer-preparation.pkl",
                  information={"bit": data_groups, "basic": records_0,
                               "xor": records_1, "reed-solomon": records_2, "fountain": records_3})

    if not path.exists(path=raw_path + "supplement-simulation.retrieving-4.redundant-retrieval.pkl"):
        address_length, bit_length, sample_number, sequence_length, skipping_range = 20, 80, 10, 200, 100
        redundant_rates, error_params, reads_params = linspace(0.01, 0.50), arange(0, 9), arange(1, 51)
        prepared_data = load_data(file_path=raw_path + "supplement-simulation.retrieving-3.outer-preparation.pkl")
        original_sample_number = 1000

        accessor, monitor = load_data(raw_path + "generation-2.special-case.pkl")[0], Monitor()
        retriever = FastRepair(accessor, virtuality, False, False, sequence_length)
        decoder = FastDecoder(accessor, virtuality, False, False, address_length + bit_length)

        print("Test baseline:")
        records_0, current = {}, 0
        for errors in error_params:
            cases = zeros(shape=(len(reads_params), sample_number, 2))
            for reads in reads_params:
                for data_index in range(sample_number):
                    expected_messages = prepared_data["bit"][data_index]
                    expected_sequences = prepared_data["basic"][data_index]
                    obtained_sequences = practice(right_sequences=deepcopy(expected_sequences),
                                                  errors=errors, reads=reads, seed=seed + data_index)
                    source_path = tmp_path + "sr.csv"
                    target_path = tmp_path + "tr.csv"
                    with open(source_path, "w") as source_file:
                        for sequence in obtained_sequences:
                            source_file.write(sequence + "\n")
                    retriever.start(sequence_paths=(source_path, target_path))
                    retrieved, results = retriever.retrieve_from_file(diversity=len(expected_sequences))
                    runtime_1 = results["runtime"]
                    remove(source_path)
                    remove(target_path)
                    source_path = tmp_path + "sd.csv"
                    target_path = tmp_path + "td.csv"
                    with open(source_path, "w") as source_file:
                        for sequence in retrieved:
                            source_file.write(sequence + "\n")
                    matrix, runtime_2 = decoder.handle(sequence_paths=(source_path, target_path))
                    remove(source_path)
                    remove(target_path)
                    start_time, obtained_messages = datetime.now(), []
                    address_indices = m2i(matrix=matrix[:, :address_length], base_size=2)
                    payloads = matrix[:, address_length:]
                    for index in range(original_sample_number):
                        if index in address_indices:
                            obtained_messages.append(payloads[where(address_indices == index)[0]][0])
                        else:
                            obtained_messages.append(zeros(shape=(bit_length,), dtype=int))
                    obtained_messages = stack(obtained_messages)
                    runtime_3 = (datetime.now() - start_time).total_seconds()

                    time = runtime_1 + runtime_2 + runtime_3
                    # noinspection PyUnresolvedReferences
                    rate = sum(all(obtained_messages == expected_messages, axis=1)) / float(original_sample_number)
                    cases[reads - 1, data_index] = [time, rate]

                    monitor(current + 1, 4500)
                    current += 1

            records_0[("%.1f" % (errors / 2)) + "%"] = cases
        print()

        print("Test exclusive-or parity:")
        records_1, outer_code, current = {}, OuterCode(method="xor"), 0
        for redundant_index, redundant_rate in enumerate(redundant_rates):
            outer_code.set_params(redundancy=redundant_rate)
            records_1[str(redundant_index + 1) + "%"] = {}
            for errors in error_params:
                cases = zeros(shape=(len(reads_params), sample_number, 2))
                for reads in reads_params:
                    for data_index in range(sample_number):
                        expected_messages = prepared_data["bit"][data_index]
                        label = (data_index, str(int(redundant_rate * 100 + 0.5)) + "%")
                        expected_sequences = prepared_data["xor"][label]
                        obtained_sequences = practice(right_sequences=deepcopy(expected_sequences),
                                                      errors=errors, reads=reads, seed=seed + data_index)
                        source_path = tmp_path + "sr.csv"
                        target_path = tmp_path + "tr.csv"
                        with open(source_path, "w") as source_file:
                            for sequence in obtained_sequences:
                                source_file.write(sequence + "\n")
                        retriever.start(sequence_paths=(source_path, target_path))
                        retrieved, results = retriever.retrieve_from_file(diversity=len(expected_sequences))
                        runtime_1 = results["runtime"]
                        remove(source_path)
                        remove(target_path)
                        source_path = tmp_path + "sd.csv"
                        target_path = tmp_path + "td.csv"
                        with open(source_path, "w") as source_file:
                            for sequence in retrieved:
                                source_file.write(sequence + "\n")
                        matrix, runtime_2 = decoder.handle(sequence_paths=(source_path, target_path))
                        remove(source_path)
                        remove(target_path)
                        start_time = datetime.now()
                        obtained_messages = outer_code.repair(binary_messages=matrix, address_length=address_length,
                                                              original_number=original_sample_number,
                                                              skipping_range=skipping_range)
                        runtime_3 = (datetime.now() - start_time).total_seconds()
                        time = runtime_1 + runtime_2 + runtime_3
                        # noinspection PyUnresolvedReferences
                        rate = sum(all(obtained_messages == expected_messages, axis=1)) / float(original_sample_number)
                        cases[reads - 1, data_index] = [time, rate]

                        monitor(current + 1, 225000)
                        current += 1

                records_1[str(redundant_index + 1) + "%"][("%.1f" % (errors / 2)) + "%"] = cases
        print()

        print("Test reed-solomon code:")
        records_2, outer_code, current = {}, OuterCode(method="reed-solomon"), 0
        block_size = 100
        for redundant_index, redundant_rate in enumerate(redundant_rates):
            outer_code.set_params(redundancy=redundant_rate)
            records_2[str(redundant_index + 1) + "%"] = {}
            for errors in error_params:
                cases = zeros(shape=(len(reads_params), sample_number, 2))
                for reads in reads_params:
                    for data_index in range(sample_number):
                        expected_messages = prepared_data["bit"][data_index]
                        label = (data_index, str(int(redundant_rate * 100 + 0.5)) + "%")
                        expected_sequences = prepared_data["reed-solomon"][label]
                        obtained_sequences = practice(right_sequences=deepcopy(expected_sequences),
                                                      errors=errors, reads=reads, seed=seed + data_index)
                        source_path = tmp_path + "sr.csv"
                        target_path = tmp_path + "tr.csv"
                        with open(source_path, "w") as source_file:
                            for sequence in obtained_sequences:
                                source_file.write(sequence + "\n")
                        retriever.start(sequence_paths=(source_path, target_path))
                        retrieved, results = retriever.retrieve_from_file(diversity=len(expected_sequences))
                        runtime_1 = results["runtime"]
                        remove(source_path)
                        remove(target_path)
                        source_path = tmp_path + "sd.csv"
                        target_path = tmp_path + "td.csv"
                        with open(source_path, "w") as source_file:
                            for sequence in retrieved:
                                source_file.write(sequence + "\n")
                        matrix, runtime_2 = decoder.handle(sequence_paths=(source_path, target_path))
                        remove(source_path)
                        remove(target_path)
                        start_time = datetime.now()
                        obtained_messages = outer_code.repair(binary_messages=matrix, address_length=address_length,
                                                              original_number=original_sample_number,
                                                              skipping_range=skipping_range, block_size=block_size)
                        runtime_3 = (datetime.now() - start_time).total_seconds()
                        time = runtime_1 + runtime_2 + runtime_3
                        # noinspection PyUnresolvedReferences
                        rate = sum(all(obtained_messages == expected_messages, axis=1)) / float(original_sample_number)
                        cases[reads - 1, data_index] = [time, rate]

                        monitor(current + 1, 225000)
                        current += 1

                records_2[str(redundant_index + 1) + "%"][("%.1f" % (errors / 2)) + "%"] = cases
        print()

        print("Test fountain code:")
        records_3, outer_code, current = {}, OuterCode(method="fountain"), 0
        fountain_params = {"delta": 0.001, "cdist": 0.025}
        for redundant_index, redundant_rate in enumerate(redundant_rates):
            outer_code.set_params(redundancy=redundant_rate, parameters=fountain_params)
            records_3[str(redundant_index + 1) + "%"] = {}
            for errors in error_params:
                cases = zeros(shape=(len(reads_params), sample_number, 2))
                for reads in reads_params:
                    for data_index in range(sample_number):
                        expected_messages = prepared_data["bit"][data_index]
                        label = (data_index, str(int(redundant_rate * 100 + 0.5)) + "%")
                        expected_sequences = prepared_data["fountain"][label]
                        obtained_sequences = practice(right_sequences=deepcopy(expected_sequences),
                                                      errors=errors, reads=reads, seed=seed + data_index)
                        source_path = tmp_path + "sr.csv"
                        target_path = tmp_path + "tr.csv"
                        with open(source_path, "w") as source_file:
                            for sequence in obtained_sequences:
                                source_file.write(sequence + "\n")
                        retriever.start(sequence_paths=(source_path, target_path))
                        retrieved, results = retriever.retrieve_from_file(diversity=len(expected_sequences))
                        runtime_1 = results["runtime"]
                        remove(source_path)
                        remove(target_path)
                        source_path = tmp_path + "sd.csv"
                        target_path = tmp_path + "td.csv"
                        with open(source_path, "w") as source_file:
                            for sequence in retrieved:
                                source_file.write(sequence + "\n")
                        matrix, runtime_2 = decoder.handle(sequence_paths=(source_path, target_path))
                        remove(source_path)
                        remove(target_path)
                        start_time = datetime.now()
                        obtained_messages = outer_code.repair(binary_messages=matrix, address_length=address_length,
                                                              original_number=original_sample_number,
                                                              skipping_range=skipping_range)
                        runtime_3 = (datetime.now() - start_time).total_seconds()
                        time = runtime_1 + runtime_2 + runtime_3
                        # noinspection PyUnresolvedReferences
                        rate = sum(all(obtained_messages == expected_messages, axis=1)) / float(original_sample_number)
                        cases[reads - 1, data_index] = [time, rate]

                        monitor(current + 1, 225000)
                        current += 1

                records_3[str(redundant_index + 1) + "%"][("%.1f" % (errors / 2)) + "%"] = cases
        print()

        save_data(file_path=raw_path + "supplement-simulation.retrieving-4.redundant-retrieval.pkl",
                  information={"basic": records_0, "xor": records_1, "reed-solomon": records_2, "fountain": records_3})

    if not path.exists(path=raw_path + "supplement-simulation.retrieving-5.chn-raw-data.pkl"):
        test_chn(bit_size=8 * 1024, seed=seed)
        test_symmetry(bit_size=8 * 1024, seed=seed)

    if not path.exists(path=raw_path + "supplement-simulation.retrieving-5.chn-raw-data.pkl"):
        monitor, sample_number, bit_length, bit_sizes = Monitor(), 5, 8 * 36, (2 ** arange(5)) * 8 * 1024

        print("Produce CHN data:")
        datasets, process, all_failure_number = {}, 0, 0
        # noinspection PyTypeChecker
        for bit_size in bit_sizes:
            for data_index in range(sample_number):
                used_seed = seed * data_index + bit_size
                expected_messages, dna_sequences, barcode_pairs, failure_number = produce(bit_size=bit_size,
                                                                                          seed=used_seed)
                info = (expected_messages, dna_sequences, barcode_pairs, failure_number)
                datasets[(str(bit_size // (8 * 1024)) + "KB", data_index)] = info
                all_failure_number += failure_number

                # noinspection PyTypeChecker
                monitor(process + 1, len(bit_sizes) * sample_number, extra={"fail": all_failure_number})
                process += 1

        save_data(file_path=raw_path + "supplement-simulation.retrieving-5.chn-raw-data.pkl", information=datasets)

    if not path.exists(path=raw_path + "supplement-simulation.retrieving-6.chn-evaluation.pkl"):
        records, monitor, process, total, bit_length, wait, success = [], Monitor(), 0, 225, 8 * 36, 3, 0

        print("Evaluate CHN performance:")
        raw_dataset = load_data(file_path=raw_path + "supplement-simulation.retrieving-5.chn-raw-data.pkl")
        for (size_key, data_key), (expected_messages, dna_sequences, barcode_pairs, _) in raw_dataset.items():
            for error_rate in linspace(0.00, 0.04, 9):
                error_number = int(len(dna_sequences[0]) * error_rate + 0.5)
                raw_sequences, shift, copy_number, saved_error_bits, same = [], 0, 1, None, 0
                while True:
                    if copy_number * len(dna_sequences) > len(raw_sequences):
                        found_batch = practice(right_sequences=dna_sequences, errors=error_number, reads=wait,
                                               seed=seed + shift)
                        raw_sequences += found_batch
                        shift += 1

                    # noinspection PyTypeChecker
                    runtime, obtained_messages, miss_count = retrieve(sequences=raw_sequences,
                                                                      barcode_pairs=barcode_pairs,
                                                                      data_shape=expected_messages.shape,
                                                                      temp_folder=tmp_path)
                    error_bits = int(sum(abs(obtained_messages - expected_messages)))

                    if miss_count == 0 and error_bits == 0:
                        info = [size_key, data_key, ("%.1f" % (error_rate * 100)) + "%", runtime, copy_number]
                        records.append(info)
                        success += 1
                        monitor(process + 1, total, {"success": success})
                        process += 1
                        break
                    else:
                        if saved_error_bits is not None:
                            # noinspection PyTypeChecker,PyUnresolvedReferences
                            if saved_error_bits > error_bits:
                                same = 0
                            elif saved_error_bits == error_bits:
                                same += 1
                            else:  # saved_error_bits < error_bits
                                monitor(process + 1, total, {"success": success})
                                process += 1
                                break
                        if same == wait:
                            monitor(process + 1, total, {"success": success})
                            process += 1
                            break

                    saved_error_bits, copy_number = error_bits, copy_number + 1

        save_data(file_path=raw_path + "supplement-simulation.retrieving-6.chn-evaluation.pkl", information=records)

    if not path.exists(path=raw_path + "supplement-simulation.retrieving-7.web-raw-data.pkl"):
        raw_dataset = load_data(file_path=raw_path + "supplement-simulation.retrieving-5.chn-raw-data.pkl")
        method = FastTranscoder(accessor=load_data(raw_path + "generation-2.special-case.pkl")[0])
        record, process, total, monitor = {}, 0, 25, Monitor()

        print("Produce equivalent SPIDER-WEB from CHN data:")
        for (size_key, data_key), (expected_messages, chn_sequences, _, _) in raw_dataset.items():
            bits = expected_messages.flatten(order="C")[:int(size_key[:-2]) * 1024 * 8]
            expected_dna_length = len(chn_sequences[0])
            address_length, bit_length = None, int(0.724 * expected_dna_length)
            while True:
                if len(bits) % bit_length != 0:
                    bits = pad(bits, pad_width=(0, bit_length - len(bits) % bit_length), constant_values=0)
                expected_messages = bits.reshape(len(bits) // bit_length, bit_length)
                address_length = int(ceil(log2(len(expected_messages))))
                addresses = i2m(arange(len(expected_messages)), length=address_length, base_size=2)
                stored_messages = concatenate([addresses, expected_messages], axis=1)
                dna_sequences = method.encode(messages=stored_messages, virtuality=virtuality,
                                              seed=seed, length_fixed=True)[0]
                if len(dna_sequences[0]) > expected_dna_length:
                    bit_length -= 1
                else:
                    break

            record[(size_key, data_key)] = (expected_messages, dna_sequences, address_length, bit_length)

            monitor(process + 1, total)
            process += 1

        save_data(raw_path + "supplement-simulation.retrieving-7.web-raw-data.pkl", record)

    if not path.exists(path=raw_path + "supplement-simulation.retrieving-8.web-evaluation.pkl"):
        records, monitor, process, total, wait = [], Monitor(), 0, 225, 3
        accessor = load_data(raw_path + "generation-2.special-case.pkl")[0]

        print("Evaluate SPIDER-WEB performance:")
        raw_dataset = load_data(file_path=raw_path + "supplement-simulation.retrieving-7.web-raw-data.pkl")
        evaluated_data = load_data(file_path=raw_path + "supplement-simulation.retrieving-6.chn-evaluation.pkl")
        for (size_key, data_key), (expected_messages, dna_sequences, address_length, bit_length) in raw_dataset.items():
            for error_rate in linspace(0.00, 0.04, 9):
                chuck, success = [size_key, data_key, ("%.1f" % (error_rate * 100)) + "%"], False
                for chn_record in evaluated_data:
                    if chuck == chn_record[:3]:
                        success = True
                        break

                if not success:
                    monitor(process + 1, total)
                    process += 1

                sample_number, sequence_length = len(expected_messages), len(dna_sequences[0])
                error_number = int(sequence_length * error_rate + 0.5)
                raw_sequences, shift, copy_number, saved_error_bits, same = [], 0, 1, None, 0
                virtualities = virtuality * ones(shape=(sample_number,), dtype=int)
                while True:
                    if copy_number * len(dna_sequences) > len(raw_sequences):
                        found_batch = practice(right_sequences=dna_sequences, errors=error_number, reads=wait,
                                               seed=seed + shift)
                        raw_sequences += found_batch
                        shift += 1

                    # C++ implementation
                    c_retriever = FastRepair(accessor, virtuality, False, False, sequence_length)
                    # Python implementation
                    p_transcoder = FastTranscoder(accessor=accessor)

                    start_time = datetime.now()

                    source_path = tmp_path + "sr.csv"
                    target_path = tmp_path + "tr.csv"
                    with open(source_path, "w") as source_file:
                        for sequence in raw_sequences:
                            source_file.write(sequence + "\n")
                    c_retriever.start(sequence_paths=(source_path, target_path))
                    retrieved_sequences, _ = c_retriever.retrieve_from_file(diversity=len(dna_sequences))
                    # noinspection PyTypeChecker
                    matrix = p_transcoder.decode(sequences=retrieved_sequences, virtualities=virtualities,
                                                 length=address_length + bit_length)
                    obtained_messages, miss_count = zeros(shape=(sample_number, bit_length), dtype=int), 0
                    addresses, payloads = m2i(matrix[:, :address_length], base_size=2), matrix[:, address_length:]
                    for address, payload in zip(addresses, payloads):
                        if 0 <= address < sample_number:
                            obtained_messages[address] = payload
                        else:
                            miss_count += 1

                    runtime = (datetime.now() - start_time).total_seconds()

                    error_bits = int(sum(abs(obtained_messages - expected_messages)))

                    remove(source_path)
                    remove(target_path)

                    if miss_count == 0 and error_bits == 0:
                        info = [size_key, data_key, ("%.1f" % (error_rate * 100)) + "%", runtime, copy_number]
                        records.append(info)
                        monitor(process + 1, total)
                        process += 1
                        break
                    else:
                        if saved_error_bits is not None:
                            # noinspection PyTypeChecker,PyUnresolvedReferences
                            if saved_error_bits > error_bits:
                                same = 0
                            elif saved_error_bits == error_bits:
                                same += 1
                            else:  # saved_error_bits < error_bits
                                monitor(process + 1, total)
                                process += 1
                                break
                        if same == wait:
                            monitor(process + 1, total)
                            process += 1
                            break

                    saved_error_bits, copy_number = error_bits, copy_number + 1

        save_data(file_path=raw_path + "supplement-simulation.retrieving-8.web-evaluation.pkl", information=records)


if __name__ == "__main__":
    evaluate_conversion(seed=2023, virtuality=2023)
    evaluate_correction(seed=2023, virtuality=2023)
    evaluate_retrieving(seed=2023, virtuality=2023)
