from multiprocessing.pool import Pool
from numpy import array, arange, linspace, zeros, random, sort, where
from numpy import min, mean, max, sum, std, ceil, log10, all
from os import path, remove
from prettytable import PrettyTable
from zipfile import ZipFile, ZIP_DEFLATED

from sbw import Monitor, SpiderWeb, FastDecoder, FastRepair, load_data, save_data
from sbw import find_vertices, connect_digraph, select_component, reduce_3_to_2
from sbw import approximate_capacity, calculate_practical_density, count_out_degrees

from works import raw_path, tmp_path, practice, evaluate_single_repair, evaluate_single_repeats
from works import batch_generate, conventional_retrieval_process


def evaluate_conversion(seed: int,
                        virtuality: int):
    monitor, sequence_length, window_length = Monitor(), 200, 10

    if not path.exists(path=raw_path + "simulation.conversion-1.theory.pkl"):
        print("Calculate theoretical information density under different constraints:")
        cases, table = {}, PrettyTable(["homopolymer run-length", "GC content", "theoretical information density"])
        process, miss, degrees = 0, 0, array([1, 2, 3, 4])
        for homopolymer in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for gc_bias in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                gc_range = [0.5 - gc_bias, 0.5 + gc_bias]
                label_1, label_2 = str(homopolymer), "[" + ("%.1f" % gc_range[0]) + "," + ("%.1f" % gc_range[1]) + "]"
                try:
                    vertices = find_vertices(window_length=window_length, run_length=homopolymer, gc_range=gc_range)
                    accessor, vertices = connect_digraph(window_length=window_length, vertices=vertices,
                                                         usable_outs=degrees, usable_ins=degrees)
                    densities = calculate_practical_density(accessor=accessor, length=sequence_length)
                    cases[(homopolymer, gc_bias)] = densities
                    # noinspection PyStringFormat
                    table.add_row([label_1, label_2, "%.3f ± %.3f" % (mean(densities), std(densities))])
                    monitor(process + 1, 60, extra={"miss": miss, "average density": mean(densities)})
                except RuntimeError:
                    miss += 1
                    monitor(process + 1, 60, extra={"miss": miss})
                process += 1

        save_data(file_path=raw_path + "simulation.conversion-1.theory.pkl", information=cases)

        print()
        print(table)
        print()

    if not path.exists(path=raw_path + "simulation.conversion-2.graphs.zip"):
        section, required_number, degrees = 10, 100, array([1, 2, 3, 4])
        collection_map, records, process = {}, {}, 0
        for index, (former, latter) in enumerate(zip(linspace(0, 2, section + 1)[:-1],
                                                     linspace(0, 2, section + 1)[1:])):
            collection_map[index] = (former, latter)
            records[index] = []

        referenced_vertex_group = []
        for homopolymer in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for gc_bias in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                gc_range = [0.5 - gc_bias, 0.5 + gc_bias]
                vertices = find_vertices(window_length=window_length, run_length=homopolymer, gc_range=gc_range)
                referenced_vertex_group.append(vertices)

        random.seed(seed)
        while True:
            # noinspection PyTypeChecker
            total_vertices = referenced_vertex_group[random.randint(0, len(referenced_vertex_group))].copy()
            random.shuffle(total_vertices)
            removed_number = random.randint(1, len(total_vertices) // 2)
            used_vertices = sort(total_vertices[:-removed_number])
            try:
                accessor, _ = connect_digraph(window_length=window_length, vertices=used_vertices,
                                              usable_outs=degrees, usable_ins=degrees)
                out_degrees = count_out_degrees(accessor=accessor)
                if sum(out_degrees[2:]) > 1:
                    density = mean(calculate_practical_density(accessor=accessor, length=sequence_length))
                    for index in range(section):
                        a, b = collection_map[index]
                        if a <= density <= b and len(records[index]) < required_number:
                            records[index].append(density)
                            identity = str(index) + "." + str(len(records[index])).zfill(len(str(required_number)))
                            save_data(file_path=tmp_path + identity + ".npy", information=accessor)
                            process += 1
                            break
            except RuntimeError:
                pass

            monitor(process, section * required_number)

            if process == section * required_number:
                break

        random.seed()

        with ZipFile(file=raw_path + "simulation.conversion-2.graphs.zip", mode="w",
                     compression=ZIP_DEFLATED) as file:
            for index in range(section):
                for number in range(1, required_number + 1):
                    identity = str(index) + "." + str(number).zfill(len(str(required_number)))
                    accessor_path = tmp_path + identity + ".npy"
                    # noinspection bad-argument-type
                    file.write(accessor_path, path.basename(accessor_path))
                    remove(accessor_path)

    if not path.exists(path=raw_path + "simulation.conversion-3.tradeoffs.pkl"):
        with ZipFile(raw_path + "simulation.conversion-2.graphs.zip") as file:
            file.extractall(tmp_path)

        records, section, required_number, degrees, current = {}, 10, 100, array([1, 2, 3, 4]), 0

        print("Calculate the trade-offs under different constraints:")

        records["random"] = []
        for index in range(section):
            for number in range(1, required_number + 1):
                accessor_path = tmp_path + str(index) + "." + str(number).zfill(len(str(required_number))) + ".npy"
                accessor_1 = load_data(file_path=accessor_path)
                bound_1 = approximate_capacity(accessor=accessor_1)
                vertex_number_1 = len(where(sum(((accessor_1 + 1).astype(bool)), axis=1).astype(bool) == 1)[0])

                accessor_2, _ = select_component(entire_accessor=accessor_1, expected_length=200)
                bound_2 = approximate_capacity(accessor=accessor_2)
                vertex_number_2 = len(where(sum(((accessor_2 + 1).astype(bool)), axis=1).astype(bool) == 1)[0])

                accessor_3 = accessor_2.copy()
                out_statistics = count_out_degrees(accessor_2)
                if out_statistics[3] > 0 and out_statistics[3] != sum(out_statistics):
                    accessor_3, _ = reduce_3_to_2(accessor=accessor_3, maintain_zero=False)
                bound_3 = approximate_capacity(accessor=accessor_3)
                vertex_number_3 = len(where(sum(((accessor_3 + 1).astype(bool)), axis=1).astype(bool) == 1)[0])
                records["random"].append((bound_1, bound_2, bound_3, vertex_number_1, vertex_number_2, vertex_number_3))
                remove(accessor_path)

                current += 1
                monitor(current, section * required_number + 60)

        records["basic"] = []
        for homopolymer in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for gc_bias in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                gc_range = [0.5 - gc_bias, 0.5 + gc_bias]
                vertices = find_vertices(window_length=window_length, run_length=homopolymer, gc_range=gc_range)
                accessor_1 = connect_digraph(window_length=window_length, vertices=vertices,
                                             usable_outs=degrees, usable_ins=degrees)[0]
                bound_1 = approximate_capacity(accessor=accessor_1)
                vertex_number_1 = len(where(sum(((accessor_1 + 1).astype(bool)), axis=1).astype(bool) == 1)[0])

                accessor_2, _ = select_component(entire_accessor=accessor_1, expected_length=200)
                bound_2 = approximate_capacity(accessor=accessor_2)
                vertex_number_2 = len(where(sum(((accessor_2 + 1).astype(bool)), axis=1).astype(bool) == 1)[0])

                accessor_3 = accessor_2.copy()
                out_statistics = count_out_degrees(accessor_2)
                if out_statistics[3] > 0 and out_statistics[3] != sum(out_statistics):
                    accessor_3, _ = reduce_3_to_2(accessor=accessor_3, maintain_zero=False)
                bound_3 = approximate_capacity(accessor=accessor_3)
                vertex_number_3 = len(where(sum(((accessor_3 + 1).astype(bool)), axis=1).astype(bool) == 1)[0])
                records["basic"].append((bound_1, bound_2, bound_3, vertex_number_1, vertex_number_2, vertex_number_3))

                current += 1
                monitor(current, section * required_number + 60)

        save_data(file_path=raw_path + "simulation.conversion-3.tradeoffs.pkl", information=records)

    if not path.exists(path=raw_path + "simulation.conversion-4.speeds.pkl"):
        accessor, records, length, repeats = load_data(raw_path + "generation-2.special-case.pkl")[0], {}, 200, 100
        method = SpiderWeb(accessor=accessor)
        # In the early implementation (before 2024),
        # we used a set to collect randomly generated sequences and remove duplicates in the method.produce function.
        # We noticed that the sequence order resulting from converting a set into a list was not controlled
        # by the random seed we provided and could therefore vary across executions.
        # However, such differences in sequence order do not affect the subsequent measurements.
        sequences = method.produce(virtuality=virtuality, length=length, diversity=1000000,
                                   allow_self_palindrome=False, duplicatable=True, palindromable=True,
                                   fail_repeats=1000, seed=seed)[0]

        decoder = FastDecoder(accessor=accessor, virtuality=2023, duplicatable=False, palindromable=False, length=200)

        print("Calculate the decoding speeds under different file scale:")
        records, repeats, target_path, current = {}, 100, tmp_path + "t.csv", 0
        for scale in [10, 100, 1000, 10000, 100000, 1000000]:
            source_path = tmp_path + "s" + str(scale) + ".csv"
            target_path = tmp_path + "t.csv"
            if not path.exists(path=source_path):
                with open(source_path, "w") as file:
                    if len(sequences) > scale:
                        for right_sequence in sequences[:scale]:
                            file.write(right_sequence + "\n")
                    else:
                        for right_sequence in sequences:
                            file.write(right_sequence + "\n")
            records[str(scale)] = []
            for repeat in range(repeats):
                _, runtime = decoder.handle(sequence_paths=(source_path, target_path))
                remove(target_path)
                records[str(scale)].append(runtime)
                current += 1
                monitor(current, 6 * repeats)
            remove(source_path)

        save_data(file_path=raw_path + "simulation.conversion-4.speeds.pkl", information=records)


def evaluate_correction(seed: int,
                        virtuality: int):
    accessor, _, _, _ = load_data(raw_path + "generation-2.special-case.pkl")
    length, maximum_error = 200, 8
    monitor = Monitor()

    if not path.exists(path=raw_path + "simulation.correction-1.sequences.pkl"):
        print("Produce 1,000,000 DNA sequences for additional correction evaluation:")
        method = SpiderWeb(accessor=accessor)
        # In the early implementation (before 2024),
        # we used a set to collect randomly generated sequences and remove duplicates in the method.produce function.
        # We noticed that the sequence order resulting from converting a set into a list was not controlled
        # by the random seed we provided and could therefore vary across executions.
        # However, such differences in sequence order do not affect the subsequent measurements.
        right_sequences = method.produce(virtuality=virtuality, length=length, diversity=1000000,
                                         allow_self_palindrome=False, duplicatable=True, palindromable=True,
                                         fail_repeats=1000, seed=seed, verbose=True)[0]
        print()
        save_data(file_path=raw_path + "simulation.correction-1.sequences.pkl", information=right_sequences)

    if not path.exists(path=raw_path + "simulation.correction-2.single.pkl"):
        method = SpiderWeb(accessor=accessor)
        right_sequences, records = load_data(file_path=raw_path + "simulation.correction-1.sequences.pkl"), {}
        table = PrettyTable(["errors",
                             "containing count 1", "screening count 1", "runtime 1",
                             "containing count 2", "screening count 2", "runtime 2"])
        for errors in range(maximum_error + 1):
            wrong_sequences = practice(right_sequences=right_sequences, errors=errors, seed=seed)
            repaired_data_1, runtime_1 = evaluate_single_repair(method=method, virtuality=virtuality,
                                                                duplicatable=True, palindromable=True,
                                                                right_sequences=right_sequences,
                                                                wrong_sequences=wrong_sequences)
            statistics_1 = zeros(shape=(len(right_sequences), 2), dtype=int)
            for index, (right_sequence, (repaired_sequences, _)) in enumerate(zip(right_sequences, repaired_data_1)):
                statistics_1[index, 0] = len(repaired_sequences)
                if right_sequence in repaired_sequences:
                    statistics_1[index, 1] = 1

            monitor(errors * 2 + 1, (maximum_error + 1) * 2)

            repaired_data_2, runtime_2 = evaluate_single_repair(method=method, virtuality=virtuality,
                                                                duplicatable=False, palindromable=False,
                                                                right_sequences=right_sequences,
                                                                wrong_sequences=wrong_sequences)
            statistics_2 = zeros(shape=(len(right_sequences), 2), dtype=int)
            for index, (right_sequence, (repaired_sequences, _)) in enumerate(zip(right_sequences, repaired_data_2)):
                statistics_2[index, 0] = len(repaired_sequences)
                if right_sequence in repaired_sequences:
                    statistics_2[index, 1] = 1

            monitor(errors * 2 + 2, (maximum_error + 1) * 2)

            records[errors] = (wrong_sequences, repaired_data_1, repaired_data_2,
                               statistics_1, statistics_2, runtime_1, runtime_2)

            containing_count_1, screening_count_1 = sum(statistics_1[:, 1]), len(where(statistics_1[:, 0] == 0)[0])
            containing_count_2, screening_count_2 = sum(statistics_2[:, 1]), len(where(statistics_2[:, 0] == 0)[0])

            table.add_row([errors,
                           containing_count_1, screening_count_1, "%.4f" % runtime_1,
                           containing_count_2, screening_count_2, "%.4f" % runtime_2])

        save_data(file_path=raw_path + "simulation.correction-2.single.pkl", information=records)

        print()
        print("Correct 1,000,000 DNA sequences under different error rates (using Python):")
        print(table)
        print()

    if not path.exists(path=raw_path + "simulation.correction-3.multiple.pkl"):
        method = SpiderWeb(accessor=accessor)
        right_data, records = load_data(file_path=raw_path + "simulation.correction-1.sequences.pkl"), {}
        table = PrettyTable(["errors",
                             "average required number (with global screening)",
                             "average required number (without global screening)"])
        for errors in arange(1, maximum_error + 1):
            print("Evaluate the retrieval of single sequences when error(s) = %d:" % errors)
            record_1 = evaluate_single_repeats(method=method, virtuality=virtuality,
                                               duplicatable=True, palindromable=True,
                                               right_sequences=right_data, errors=errors, seed=seed,
                                               verbose=True)
            table.add_row([errors, True, "%d" % min(record_1), "%.1f" % mean(record_1), "%d" % max(record_1)])
            record_2 = evaluate_single_repeats(method=method, virtuality=virtuality,
                                               duplicatable=False, palindromable=False,
                                               right_sequences=right_data, errors=errors, seed=seed,
                                               verbose=True)
            table.add_row([errors, False, "%d" % min(record_2), "%.1f" % mean(record_2), "%d" % max(record_2)])
            records[errors] = (record_1, record_2)
            print()

        print("Estimate required copy number from 1,000,000 DNA sequences under different error rates:")
        print(table)
        print()

        save_data(file_path=raw_path + "simulation.correction-3.multiple.pkl", information=records)

    if not path.exists(path=raw_path + "simulation.correction-4.readouts.pkl"):
        records, retrievers, repeats, maximum_thread = {}, {}, 100, 9
        for threads in arange(1, maximum_thread + 1):
            retrievers[threads] = FastRepair(accessor=accessor, virtuality=virtuality, thread_number=threads,
                                             duplicatable=False, palindromable=False, length=length)
        right_sequences = load_data(file_path=raw_path + "simulation.correction-1.sequences.pkl")
        for errors in range(maximum_error + 1):
            source_path = tmp_path + "s" + str(errors) + ".csv"
            target_path = tmp_path + "t" + str(errors) + ".csv"
            if not path.exists(path=source_path):
                wrong_sequences = practice(right_sequences=right_sequences, errors=errors, seed=seed)
                with open(source_path, "w") as file:
                    for wrong_sequence in wrong_sequences:
                        file.write(wrong_sequence + "\n")
            records[errors] = {}
            for threads in arange(1, 10):
                for repeat in range(repeats):
                    retrievers[threads].start(sequence_paths=(source_path, target_path))
                    runtime = retrievers[threads].retrieve_from_file(diversity=len(right_sequences))[1]["runtime"]
                    records[errors][threads].append(runtime)
            remove(target_path)
        for errors in range(maximum_error + 1):
            remove(tmp_path + "s" + str(errors) + ".csv")

        save_data(file_path=raw_path + "simulation.correction-4.readouts.pkl", information=records)


def evaluate_retrieving(seed: int,
                        virtuality: int):
    accessor, _, _, palindrome_map = load_data(raw_path + "generation-2.special-case.pkl")
    length, maximum_error = 200, 8

    if not path.exists(path=raw_path + "simulation.retrieving-1.sequences.pkl"):
        # Here we used a set to collect randomly generated sequences and remove duplicates.
        # Therefore, the sequence order resulting from converting a set into a list was not controlled
        # by the random seed we provided and could therefore vary across executions.
        # However, such differences in sequence order do not affect the subsequent measurements.
        diversity, batch_size = 10 ** ceil(log10(8 * (1024 ** 3) / (0.724 * length))).astype(int), 1000000
        pool = Pool(processes=10)
        for used_seed in arange(10, dtype=int) + seed:
            params = (accessor, palindrome_map, virtuality, length,
                      int(diversity / 10 * 1.2),  # To avoid the inability to meet the expected quantity after merging.
                      batch_size, True, True, used_seed, tmp_path)
            pool.apply_async(batch_generate, args=params)
        pool.close()
        pool.join()

        right_data = set()
        for used_seed in arange(10, dtype=int) + seed:
            batch_data = load_data(file_path=tmp_path + str(used_seed) + ".pkl")
            right_data |= set(batch_data)
            if len(right_data) >= diversity:
                break
        right_data = list(right_data)[:diversity]
        save_data(file_path=raw_path + "simulation.retrieving-1.sequences.pkl", information=right_data)

    if not path.exists(path=raw_path + "simulation.retrieving-2.rates.pkl"):
        total_data, records = load_data(file_path=raw_path + "simulation.retrieving-1.sequences.pkl"), []
        repeats, error_params, reads_params, diversity = 20, arange(1, 9, dtype=int), arange(1, 51, dtype=int), 10
        retriever = FastRepair(accessor, virtuality, False, False, length)

        while diversity <= len(total_data):
            expected, chuck = total_data[:diversity], set(total_data[:diversity])
            for reads in reads_params:
                for errors in error_params:
                    values_1, values_2 = [], []
                    for repeat in range(repeats):
                        obtained = practice(right_sequences=expected, errors=errors, reads=reads, seed=seed + repeat)
                        source_path = tmp_path + "s.csv"
                        target_path = tmp_path + "t.csv"
                        with open(source_path, "w") as source_file:
                            for sequence in obtained:
                                source_file.write(sequence + "\n")

                        # During supercomputing operations, the source files are processed through slicing.
                        retriever.start(sequence_paths=(source_path, target_path))
                        retrieved, results = retriever.retrieve_from_file(need_rank=True, diversity=diversity)
                        remove(source_path)
                        remove(target_path)

                        success_count = len(chuck.intersection(retrieved))
                        maximum_wrong, count = 0, max(list(results["rank"].keys()))
                        while count >= 1:
                            if count in results["rank"] and len(results["rank"][count] - chuck) > 0:
                                maximum_wrong = count
                                break
                            count -= 1

                        values_1.append(success_count)
                        values_2.append(maximum_wrong)
                    records.append([diversity, reads, errors, mean(values_1), mean(values_2)])
            diversity *= 10

        save_data(file_path=raw_path + "simulation.retrieving-2.rates.pkl", information=records)

    if not path.exists(path=raw_path + "simulation.retrieving-3.comparable.pkl"):
        diversity, length = 1000, 150
        total_data = load_data(file_path=raw_path + "simulation.retrieving-1.sequences.pkl")[:diversity]
        used_parts, records = [sequence[:length] for sequence in total_data], {}

        retriever = FastRepair(accessor, virtuality, False, False, length)
        repeats, error_params, reads_params = 4, arange(1, 7, dtype=int), arange(10, 101, 10, dtype=int)
        expected, chuck = used_parts[:diversity], set(used_parts[:diversity])
        for reads in reads_params:
            for errors in error_params:
                cov_values, our_values = [[], []], [[], []]
                for repeat in range(repeats):
                    obtained = practice(right_sequences=expected, errors=errors, reads=reads, seed=seed + repeat)

                    retrieved, runtime = conventional_retrieval_process(raw_sequences=obtained,
                                                                        temp_folder=tmp_path, default=False)
                    cov_values[0].append(len(chuck.intersection(set(retrieved))))
                    cov_values[1].append(runtime)

                    source_path = tmp_path + "s.csv"
                    target_path = tmp_path + "t.csv"
                    with open(source_path, "w") as source_file:
                        for sequence in obtained:
                            source_file.write(sequence + "\n")

                    retriever.start(sequence_paths=(source_path, target_path))
                    retrieved, results = retriever.retrieve_from_file(diversity=diversity)
                    runtime = results["runtime"]
                    remove(source_path)
                    remove(target_path)
                    our_values[0].append(len(chuck.intersection(set(retrieved))))
                    our_values[1].append(runtime)
                records[(reads, errors)] = (tuple(cov_values), tuple(our_values))

        save_data(file_path=raw_path + "simulation.retrieving-3.comparable.pkl", information=records)

    if not path.exists(path=raw_path + "simulation.retrieving-4.scalable.pkl"):
        maximum_diversity, length = 100000, 150
        total_data = load_data(file_path=raw_path + "simulation.retrieving-1.sequences.pkl")
        used_parts, records = [sequence[:length] for sequence in total_data], {}

        retriever = FastRepair(accessor, virtuality, False, False, length)
        repeats, error_params, reads_params = 4, [1, 6], arange(10, 101, 10, dtype=int)
        for diversity in [10, 100, 1000, 10000, 100000]:
            expected, chuck = used_parts[:diversity], set(used_parts[:diversity])
            for errors in error_params:
                saved_reads, our_runtimes, cov_runtimes, cov_retrieval = 0, [], [], []
                for reads in reads_params:
                    our_values = [[], []]
                    for repeat in range(repeats):
                        obtained = practice(right_sequences=expected, errors=errors, reads=reads, seed=seed + repeat)
                        source_path = tmp_path + "s.csv"
                        target_path = tmp_path + "t.csv"
                        with open(source_path, "w") as source_file:
                            for sequence in obtained:
                                source_file.write(sequence + "\n")

                        retriever.start(sequence_paths=(source_path, target_path))
                        retrieved, results = retriever.retrieve_from_file(diversity=diversity)
                        runtime = results["runtime"]
                        remove(source_path)
                        remove(target_path)
                        our_values[0].append(len(chuck.intersection(set(retrieved))))
                        our_values[1].append(runtime)
                    if all(array(our_values[0]) == diversity):
                        saved_reads = reads
                        our_runtimes.append(our_values[1])
                        break

                for repeat in range(repeats):
                    obtained = practice(right_sequences=expected, errors=errors, reads=saved_reads, seed=seed + repeat)
                    retrieved, runtime = conventional_retrieval_process(raw_sequences=obtained, temp_folder=tmp_path,
                                                                        default=False)
                    cov_retrieval.append(len(chuck.intersection(set(retrieved))))
                    cov_runtimes.append(runtime)
                records[(diversity, errors)] = (saved_reads, our_runtimes, cov_runtimes, cov_retrieval)

        save_data(file_path=raw_path + "simulation.retrieving-4.scalable.pkl", information=records)


if __name__ == "__main__":
    evaluate_conversion(seed=2023, virtuality=2023)
    evaluate_correction(seed=2023, virtuality=2023)
    evaluate_retrieving(seed=2023, virtuality=2023)
