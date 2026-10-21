# noinspection PyProtectedMember
from cv2 import imread, resize, cvtColor, COLOR_BGR2GRAY
from datetime import datetime
from imageio import mimsave
from matplotlib import pyplot, use
from numpy import array, zeros, ones, arange, random, expand_dims, unpackbits, packbits, all
from numpy import sum, abs, min, inf, uint8, where
from os import listdir, path, remove
from re import split, compile
from shutil import copyfileobj
# noinspection PyCompatibility
from urllib.request import Request, urlopen
from zipfile import ZipFile, ZIP_DEFLATED

from sbw import SpiderWeb, FastRepair, Monitor, adjust_motif_format, connect_digraph, m2s, i2m, load_data, save_data
from sbw import detect_head, detect_tail, detect_duplicatable, detect_palindromable
from sbw import compute_minimum_duplicate_spacing, compute_minimum_palindrome_spacing

from works import raw_path, exp_path, tmp_path
from works import design_pcr_primer, levenshtein
from works import obtain_raw_sequences, calculate_retrieval_quality, obtain_samples, conventional_retrieval_process


def experiment_1(seed: int,
                 virtuality: int):
    monitor, lengths, number, maximum_depth = Monitor(), [150, 25], 3000, 1000

    if not path.exists(raw_path + "experiment.retrieval-1.sequences.pkl"):
        print("Produce 3000 DNA sequences for comparable evaluation of different high-quality real-time retrievals:")
        method = SpiderWeb(accessor=load_data(raw_path + "generation-2.special-case.pkl")[0])
        payloads, counts, miss = method.produce(virtuality=virtuality, length=lengths[0],
                                                duplicatable=True, palindromable=True, need_preorder=True,
                                                diversity=number, seed=seed, verbose=True)
        print()
        primers = design_pcr_primer(payloads=payloads, length=lengths[1], seed=seed)
        save_data(file_path=raw_path + "experiment.retrieval-1.sequences.pkl", information=(payloads, primers))

    if not path.exists(raw_path + "experiment.retrieval-2.similarities.npy"):
        print("Analyze the similarity among 3,000 produced DNA sequences:")
        payloads = load_data(file_path=raw_path + "experiment.retrieval-1.sequences.pkl")[0]
        matrix, process = zeros(shape=(len(payloads), len(payloads)), dtype=int), 0
        for index_1 in range(0, len(payloads) - 1):
            for index_2 in range(index_1 + 1, len(payloads)):
                similarity = levenshtein(sequence_1=payloads[index_1], sequence_2=payloads[index_2])
                matrix[index_1, index_2], matrix[index_2, index_1] = similarity, similarity
                monitor(process + 1, int((len(payloads) - 1) * len(payloads) / 2))
                process += 1
        print()
        save_data(file_path=raw_path + "experiment.retrieval-2.similarities.npy", information=matrix)

    if not path.exists(raw_path + "experiment.retrieval-3.collections.zip"):
        mapping = {("0001-1000", 0): ("L02", "423"), ("1001-2000", 0): ("L03", "400"), ("2001-3000", 0): ("L03", "402"),
                   ("0001-1000", 1): ("L02", "595"), ("1001-2000", 1): ("L02", "417"), ("2001-3000", 1): ("L02", "420"),
                   ("0001-1000", 2): ("L02", "598"), ("1001-2000", 2): ("L02", "418"), ("2001-3000", 2): ("L02", "421"),
                   ("0001-1000", 3): ("L02", "601"), ("1001-2000", 3): ("L02", "602"), ("2001-3000", 3): ("L03", "603"),
                   ("0001-1000", 4): ("L03", "604"), ("1001-2000", 4): ("L03", "605"), ("2001-3000", 4): ("L03", "606"),
                   ("0001-1000", 5): ("L03", "607"), ("1001-2000", 5): ("L03", "608"), ("2001-3000", 5): ("L03", "609")}
        # noinspection PyTypeChecker
        child_paths, used_paths = list(listdir(exp_path + "1/")), []
        primers = load_data(raw_path + "experiment.retrieval-1.sequences.pkl")[-1]
        forward_primer, reverse_primer = primers[0], primers[1].translate(str.maketrans("ACGT", "TGCA"))[::-1]
        for (index_scope, pcr_number), (lane_index, chip_index) in mapping.items():
            file_paths = {}
            for child_path in child_paths:
                if "_" + lane_index + "_" in child_path and "-" + chip_index + "_" in child_path:
                    file_paths[child_path[-7:]] = exp_path + "1/" + child_path
            assert "1.fq.gz" in file_paths and "2.fq.gz" in file_paths and len(file_paths) == 2

            print("Retrieve " + str(number * maximum_depth) + " DNA sequences of IDs from "
                  + index_scope + " with (#" + str(pcr_number) + ")-round error-prone PCR amplification:")
            raw_sequences = obtain_raw_sequences(compressed_file_paths=file_paths, tmp_path=tmp_path,
                                                 forward_primer=forward_primer, reverse_primer=reverse_primer,
                                                 expected_length=lengths[0], tolerable_deviation=10,
                                                 batch=1000000, verbose=True)
            save_data(file_path=tmp_path + index_scope + "." + str(pcr_number) + ".pkl", information=raw_sequences)
            used_paths.append(tmp_path + index_scope + "." + str(pcr_number) + ".pkl")

        with ZipFile(file=raw_path + "experiment.retrieval-3.collections.zip", mode="w",
                     compression=ZIP_DEFLATED) as file:
            for used_path in used_paths:
                # noinspection PyTypeChecker
                file.write(used_path, path.basename(used_path))
                remove(used_path)

    if not path.exists(raw_path + "experiment.retrieval-4.quality.zip"):
        with ZipFile(raw_path + "experiment.retrieval-3.collections.zip") as file:
            file.extractall(tmp_path)
        expected_data, used_paths = load_data(file_path=raw_path + "experiment.retrieval-1.sequences.pkl")[0], []
        sequence_set = {"0001-1000": expected_data[:1000],
                        "1001-2000": expected_data[1000:2000],
                        "2001-3000": expected_data[2000:]}
        for index_scope in ["0001-1000", "1001-2000", "2001-3000"]:
            for pcr_number in range(6):
                print("Evaluation quality of DNA sequences of IDs from "
                      + index_scope + " with (#" + str(pcr_number) + ")-round error-prone PCR amplification:")
                obtained_sequences = load_data(file_path=tmp_path + index_scope + "." + str(pcr_number) + ".pkl")
                record = calculate_retrieval_quality(expected_sequences=sequence_set[index_scope],
                                                     obtained_sequences=obtained_sequences, verbose=True)
                save_data(file_path=tmp_path + index_scope + "." + str(pcr_number) + ".q.pkl", information=record)
                used_paths.append(tmp_path + index_scope + "." + str(pcr_number) + ".q.pkl")

        with ZipFile(file=raw_path + "experiment.retrieval-4.quality.zip", mode="w",
                     compression=ZIP_DEFLATED) as file:
            for used_path in used_paths:
                # noinspection PyTypeChecker
                file.write(used_path, path.basename(used_path))
                remove(used_path)

    if not path.exists(raw_path + "experiment.retrieval-5.statistics.pkl"):
        with ZipFile(raw_path + "experiment.retrieval-4.quality.zip") as file:
            file.extractall(tmp_path)

        similarity_matrix, threshold = load_data(raw_path + "experiment.retrieval-2.similarities.npy"), inf
        for index_1 in range(number - 1):
            for index_2 in range(index_1 + 1, number):
                threshold = min([threshold, similarity_matrix[index_1, index_2]])

        records = {}
        for index_scope in ["0001-1000", "1001-2000", "2001-3000"]:
            for pcr_number in [0, 1, 2, 3, 4, 5]:
                numbers, errors = zeros(shape=(1000,), dtype=int), zeros(shape=(151,), dtype=int)
                for error, sequence_indices in load_data(tmp_path + index_scope + "." + str(pcr_number) + ".q.pkl"):
                    if error < threshold:
                        numbers[sequence_indices[0]] += 1
                        errors[error] += 1
                    records[(index_scope, "#" + str(pcr_number))] = (errors, numbers)
                remove(tmp_path + index_scope + "." + str(pcr_number) + ".q.pkl")
        save_data(file_path=raw_path + "experiment.retrieval-5.statistics.pkl", information=records)

    if not path.exists(raw_path + "experiment.retrieval-6.samples.pkl"):
        with ZipFile(raw_path + "experiment.retrieval-3.collections.zip") as file:
            file.extractall(tmp_path)
        with ZipFile(raw_path + "experiment.retrieval-4.quality.zip") as file:
            file.extractall(tmp_path)

        similarity_matrix, threshold = load_data(raw_path + "experiment.retrieval-2.similarities.npy"), inf
        for index_1 in range(number - 1):
            for index_2 in range(index_1 + 1, number):
                threshold = min([threshold, similarity_matrix[index_1, index_2]])
        statistical_data = load_data(file_path=raw_path + "experiment.retrieval-5.statistics.pkl")

        records, batch_number, maximum_depth, sample_number = {}, 3, 100, 1000
        for index_scope in ["0001-1000", "1001-2000", "2001-3000"]:
            for pcr_number in [0, 1, 2, 3, 4, 5]:
                raw_sequences = load_data(file_path=tmp_path + index_scope + "." + str(pcr_number) + ".pkl")
                quality_records = load_data(file_path=tmp_path + index_scope + "." + str(pcr_number) + ".q.pkl")
                pairs = [[], []]
                for raw_sequence, quality_record in zip(raw_sequences, quality_records):
                    if quality_record[0] < threshold:
                        pairs[0].append(raw_sequence)
                        pairs[1].append(quality_record)
                copies = statistical_data[(index_scope, "#" + str(pcr_number))][1]
                sample_data = obtain_samples(raw_sequences=pairs[0], quality_records=pairs[1], copies=copies,
                                             batch_size=sample_number * maximum_depth, batch_number=batch_number,
                                             seed=seed, verbose=True)
                records[(index_scope, "#" + str(pcr_number))] = sample_data

        save_data(file_path=raw_path + "experiment.retrieval-6.samples.pkl", information=records)

        for index_scope in ["0001-1000", "1001-2000", "2001-3000"]:
            for pcr_number in range(6):
                remove(tmp_path + index_scope + "." + str(pcr_number) + ".pkl")
                remove(tmp_path + index_scope + "." + str(pcr_number) + ".q.pkl")

    if not path.exists(raw_path + "experiment.retrieval-7.performances.pkl"):
        expected_data, records = load_data(file_path=raw_path + "experiment.retrieval-1.sequences.pkl")[0], {}
        sequence_set = {"0001-1000": set(expected_data[:1000]),
                        "1001-2000": set(expected_data[1000:2000]),
                        "2001-3000": set(expected_data[2000:])}
        barcode = "CGTGTATGCG"
        fast_repair = FastRepair(accessor=load_data(raw_path + "generation-2.special-case.pkl")[0],
                                 virtuality=virtuality, duplicatable=False, palindromable=False, length=lengths[0] - 10)
        sample_data, sample_number, batch_number = load_data(raw_path + "experiment.retrieval-6.samples.pkl"), 1000, 3
        for index_scope in ["0001-1000", "1001-2000", "2001-3000"]:
            for pcr_number in [0, 1, 2, 3, 4, 5]:
                for batch_index in range(batch_number):
                    temp_folder = tmp_path + index_scope + "." + str(pcr_number) + "." + str(batch_index)
                    raw_sequence_data = sample_data[(index_scope, "#" + str(pcr_number))][batch_index + 1]["seq"]
                    for depth in arange(10, 101, 10, dtype=int):
                        number = depth * 1000
                        if number < len(raw_sequence_data):
                            raw_sequences = raw_sequence_data[:number]
                        else:
                            raw_sequences = raw_sequence_data

                        retrieved_data, runtime = conventional_retrieval_process(raw_sequences=raw_sequences,
                                                                                 temp_folder=temp_folder,
                                                                                 default=False)
                        success_count = len(set(sequence_set[index_scope]).intersection(set(retrieved_data[:1000])))
                        contain_count = len(set(sequence_set[index_scope]).intersection(set(retrieved_data)))
                        total_count = len(retrieved_data)
                        result_1 = (success_count, runtime, contain_count, total_count)

                        # noinspection PyUnresolvedReferences
                        source_path = tmp_path + index_scope + "." + pcr_number[1:] + "." + str(batch_index) + ".csv"
                        target_path = tmp_path + "result.csv"

                        with open(file=source_path, mode="w") as file:
                            for index, payload in enumerate(raw_sequences):
                                file.write(payload[10:] + "\n")

                        fast_repair.start(sequence_paths=(source_path, target_path))
                        retrieved_data, results = fast_repair.retrieve_from_file(diversity=1000, need_rank=True)
                        top_retrieved_sequences = []
                        for sequence in retrieved_data:
                            top_retrieved_sequences.append(barcode + sequence)
                        success_count = len(set(top_retrieved_sequences).intersection(set(sequence_set[index_scope])))
                        runtime = results["runtime"]
                        total_retrieved_sequences = []
                        for count, sequences in results["rank"].items():
                            for sequence in list(sequences):
                                total_retrieved_sequences.append(barcode + sequence)
                        contain_count = len(set(total_retrieved_sequences).intersection(set(sequence_set[index_scope])))
                        total_count = len(total_retrieved_sequences)
                        result_2 = (success_count, runtime, contain_count, total_count)

                        records[(index_scope, pcr_number, depth, batch_index + 1)] = (result_1, result_2)

        save_data(file_path=raw_path + "experiment.retrieval-7.performances.pkl", information=records)


def experiment_2(seed: int):
    method, monitor = SpiderWeb(accessor=load_data(raw_path + "generation-2.special-case.pkl")[0]), Monitor()
    window_size, payload_length, address_length, buffer_length = 10, 110, 60, 200
    near_threshold_1, near_threshold_2, anchor = 60, 18, "T" * window_size

    if (not path.exists(raw_path + "experiment.case-1.video.gif")) or \
       (not path.exists(raw_path + "experiment.case-2.figures.npy")):
        if not path.exists(tmp_path + "total-horse.jpg"):
            url = "https://upload.wikimedia.org/wikipedia/commons/4/4a/Muybridge_race_horse_gallop.jpg"
            req = Request(url=url, headers={"User-Agent": "SPIDER-WEB"})
            with urlopen(req, timeout=30) as response:
                with open(tmp_path + "total-horse.jpg", "wb") as file:
                    copyfileobj(response, file)
        image = imread(tmp_path + "total-horse.jpg")
        gray_image = cvtColor(image, COLOR_BGR2GRAY)
        resized_image = resize(gray_image, (160, 120))
        blocks, block_h, block_w = [], resized_image.shape[0] // 4, resized_image.shape[1] // 4
        for index_1 in range(4):
            for index_2 in range(4):
                blocks.append(resized_image[index_1 * block_h: (index_1 + 1) * block_h,
                                            index_2 * block_w: (index_2 + 1) * block_w].tolist())
        blocks = array(blocks, dtype=uint8)
        save_data(file_path=raw_path + "experiment.case-2.figures.npy", information=array(blocks))
        # noinspection PyTypeChecker
        mimsave(uri=raw_path + "experiment.case-1.video.gif", ims=blocks, duration=2.0 / 16.0, loop=0)
        remove(tmp_path + "total-horse.jpg")

    if not path.exists(raw_path + "experiment.case-3.raw-payloads.pkl"):
        print("Encode high-compatible information segments for the real-time validation experiment:")
        frames, segment_info = load_data(file_path=raw_path + "experiment.case-2.figures.npy"), {}
        chuck_matrix, fixed_length, current = ones(shape=(16, 30, 40), dtype=uint8) * 255, payload_length, 0
        vertex_segments, retried = m2s(i2m(method.get_params()["vertices"], length=window_size)), 1000

        for frame_index in range(16):
            for index_1 in range(10):
                for index_2 in range(20):
                    block = frames[frame_index,
                                   index_1 * 3: (index_1 + 1) * 3,
                                   index_2 * 2: (index_2 + 1) * 2]
                    payload = unpackbits(block.reshape(-1))
                    # noinspection PyUnresolvedReferences
                    chuck_matrix[frame_index,
                                 index_1 * 3: (index_1 + 1) * 3,
                                 index_2 * 2: (index_2 + 1) * 2] = packbits(payload).reshape(3, 2)

                    produced_set = []
                    for virtuality, vertex_segment in enumerate(vertex_segments):
                        vertex_segment = vertex_segments[virtuality]
                        if not detect_head(segment=vertex_segment, nucleotide="T", length=window_size // 2):
                            try:
                                sequence = method.encode(message=payload, virtuality=virtuality,
                                                         duplicatable=True, palindromable=True, need_virtuality=True)
                                if len(sequence) > payload_length - window_size:
                                    continue

                                # sequence <= 100, + redundant >= 10
                                remain_vertex = vertex_segments.index(sequence[-window_size:])
                                for retried_index in range(retried // 10):
                                    remained_segments = method.produce(length=fixed_length - len(sequence),
                                                                       virtuality=remain_vertex, diversity=10,
                                                                       seed=virtuality + retried_index)[0]
                                    success = False
                                    for remained_segment in remained_segments:
                                        if detect_tail(segment=remained_segment, nucleotide="T",
                                                       length=window_size // 2):
                                            continue

                                        # previous predetermined maximum homopolymer run-length is 2.
                                        temp_sequence = "T" * 3 + sequence + remained_segment + "T" * 3

                                        if detect_duplicatable(segment=temp_sequence, window_size=window_size):
                                            continue

                                        if detect_palindromable(segment=temp_sequence, window_size=window_size + 1):
                                            continue

                                        sequence, success = sequence + remained_segment, True
                                        break

                                    if success:
                                        break

                                decoded_message = method.decode(sequence=sequence, length=len(payload),
                                                                duplicatable=True, palindromable=True)
                                assert all(decoded_message == payload)
                                produced_set.append((virtuality, sequence))
                            except ValueError:
                                pass  # out-degree = 0

                    segment_info[(frame_index, index_1, index_2)] = produced_set

                    monitor(current + 1, 3200)

        assert all(frames == chuck_matrix)
        save_data(file_path=raw_path + "experiment.case-3.raw-payloads.pkl", information=segment_info)
        print()

    if not path.exists(raw_path + "experiment.case-4.raw-combined-sequences.pkl"):
        print("Find suitable combination of DNA segments for global constraints:")
        segment_info = load_data(file_path=raw_path + "experiment.case-3.raw-payloads.pkl")
        combinations, current, total = {}, 0, 160
        for frame_index in range(16):
            for raw_index in range(10):
                candidate_set = []
                for column_index in range(20):
                    candidate_set.append(segment_info[(frame_index, raw_index, column_index)])
                # no random seed, utilizing supercomputing.
                while True:
                    combination = [candidates[random.choice(len(candidates))][1] for candidates in candidate_set]

                    temp_sequence = anchor
                    for segment in combination:
                        temp_sequence += segment + anchor

                    # check the minimum spacing between the same 10-mer.
                    minimum_value_1 = compute_minimum_duplicate_spacing(segment=temp_sequence, window_size=window_size)
                    if minimum_value_1 < near_threshold_1:
                        continue

                    # check the minimum spacing between the 10 mer and its complementary palindrome.
                    minimum_value_2 = compute_minimum_palindrome_spacing(segment=temp_sequence, window_size=window_size)
                    if minimum_value_2 < near_threshold_2:
                        continue

                    # all global constraints are met.
                    sequence = anchor
                    for segment in combination:
                        sequence += segment + anchor
                    # noinspection PyTypeChecker
                    combinations[(frame_index, raw_index)] = (sequence, combination)

            monitor(current + 1, total)
            current += 1

        save_data(file_path=raw_path + "experiment.case-4.raw-combined-sequences.pkl", information=combinations)
        print()

    if not path.exists(raw_path + "experiment.case-5.raw-addresses.pkl"):
        print("Generate high-compatible addresses for the real-time validation experiment:")
        selected_sequences = load_data(file_path=raw_path + "experiment.case-4.raw-combined-sequences.pkl")
        address_group, candidate_number, current, total = {}, 1000, 0, 160
        for frame_index in range(16):
            for raw_index in range(10):
                sequence, combination = selected_sequences[(frame_index, raw_index)]
                address_candidates = set()
                random.seed(seed)
                while len(address_candidates) < candidate_number:
                    virtuality = random.choice(len(method.get_params()["vertices"]))
                    # noinspection PyTypeChecker
                    segments = method.produce(length=address_length, virtuality=virtuality,
                                              duplicatable=True, palindromable=True, diversity=10)[0]
                    for segment in segments:
                        if detect_head(segment=segment, nucleotide="T", length=window_size // 2):
                            continue

                        if detect_tail(segment=segment, nucleotide="T", length=window_size // 2):
                            continue

                        # analyze forward combination.
                        segment_1 = anchor + segment + sequence

                        # check the minimum spacing between the same 10-mer.
                        minimum_value_1 = compute_minimum_duplicate_spacing(segment=segment_1, window_size=window_size)
                        if minimum_value_1 < near_threshold_1:
                            continue

                        # check the minimum spacing between the 10 mer and its complementary palindrome.
                        minimum_value_2 = compute_minimum_palindrome_spacing(segment=segment_1, window_size=window_size)
                        if minimum_value_2 < near_threshold_2:
                            continue

                        # analyze reverse combination.
                        segment_2 = sequence + segment + anchor

                        # check the minimum spacing between the same 10-mer.
                        minimum_value_1 = compute_minimum_duplicate_spacing(segment=segment_2, window_size=window_size)
                        if minimum_value_1 < near_threshold_1:
                            continue

                        # check the minimum spacing between the 10 mer and its complementary palindrome.
                        minimum_value_2 = compute_minimum_palindrome_spacing(segment=segment_2, window_size=window_size)
                        if minimum_value_2 < near_threshold_2:
                            continue

                        address_candidates.add(segment)

                    if len(address_candidates) > candidate_number:
                        address_candidates = set(list(address_candidates)[:candidate_number])

                address_group[(frame_index, raw_index)] = address_candidates

                monitor(current + 1, total)
                current += 1

        save_data(file_path=raw_path + "experiment.case-5.raw-addresses.pkl", information=address_group)
        print()

    if not path.exists(raw_path + "experiment.case-6.raw-selected-addresses.pkl"):
        print("Find the best address for each sequence:")
        retried, difference_threshold = 1000, 18  # >30% of the address segment.
        address_group = load_data(file_path=raw_path + "experiment.case-5.raw-addresses.pkl")
        # no random seed, utilizing supercomputing.
        selected_addresses, current, success = [], 0, True
        for frame_index in range(16):
            for raw_index in range(10):
                # sorted for repeatability.
                collection, address_candidates = [], address_group[(frame_index, raw_index)]
                for _ in range(retried):
                    forward_candidate, reverse_candidate = random.choice(address_candidates), None
                    forward_flag, reverse_flag = True, True
                    for selected_address in selected_addresses:
                        # noinspection PyTypeChecker
                        if levenshtein(selected_address, forward_candidate) <= difference_threshold:
                            forward_flag = False
                            break
                    if forward_flag:
                        for reverse_one in address_candidates:
                            reverse_flag = True
                            # noinspection PyTypeChecker
                            if levenshtein(forward_candidate, reverse_one) > difference_threshold:
                                for selected_identity in selected_addresses:
                                    if levenshtein(selected_identity, reverse_one) <= difference_threshold:
                                        reverse_flag = False
                                        break
                            else:
                                reverse_flag = False

                            if reverse_flag:
                                reverse_candidate = reverse_one
                                break

                    if forward_flag and reverse_flag:
                        collection = [forward_candidate, reverse_candidate]
                        break

                if len(collection) == 2:
                    selected_addresses += collection
                    monitor(current + 1, 160)
                else:
                    success = False
                    print()
                    break

                current += 1

                if not success:
                    break

            if not success:
                break

        if success:
            used_addresses, current = {}, 0
            for frame_index in range(16):
                for column_index in range(10):
                    used_addresses[(frame_index, column_index)] = (selected_addresses[current + 0],
                                                                   selected_addresses[current + 1])
                    current += 2

            save_data(file_path=raw_path + "experiment.case-6.raw-selected-addresses.pkl",
                      information=used_addresses)
        else:
            raise ValueError("Cannot generated successfully!")
        print()

    if not path.exists(raw_path + "experiment.case-7.buffer-graph.pkl"):
        print("Create redundant DNA segments for Nanopore sequencing:")
        selected_addresses = load_data(file_path=raw_path + "experiment.case-6.raw-selected-addresses.pkl")
        selected_sequences = load_data(file_path=raw_path + "experiment.case-4.raw-combined-sequences.pkl")
        near_k_mers, observed_threshold = set(), 200
        for frame_index in range(16):
            for raw_index in range(10):
                _, combination = selected_sequences[(frame_index, raw_index)]
                for payload in combination:
                    near_k_mers.add(payload[:window_size])

                forward_address, reverse_address = selected_addresses[(frame_index, raw_index)]
                information_sequence = selected_sequences[(frame_index, raw_index)][0]
                sequence = anchor + forward_address + information_sequence + reverse_address + anchor

                for location in range(observed_threshold - (window_size - 1)):
                    near_k_mers.add(sequence[location: location + window_size])
                for location in range(len(sequence) - (observed_threshold + 1), len(sequence) - (window_size - 1)):
                    near_k_mers.add(sequence[location: location + window_size])

        near_k_mers = list(near_k_mers)
        undesired_motifs = adjust_motif_format(motif_set=[list(near_k_mers)])
        run_length = 3

        k_mers = i2m(integers=arange(4 ** window_size, dtype=int), length=window_size)
        usable = ones(shape=(4 ** window_size,), dtype=bool)
        current, total = 0, window_size - (run_length + 1) + 1 + len(undesired_motifs)

        for location in range(window_size - (run_length + 1) + 1):
            differences = k_mers[:, location: location + run_length + 1] - expand_dims(k_mers[:, location], axis=1)
            usable[where(sum(abs(differences), axis=1) == 0)] = 0
            monitor(current + 1, total)
            current += 1

        for index, motif in enumerate(undesired_motifs):
            differences = k_mers - expand_dims(motif, axis=0)
            usable[where(sum(abs(differences), axis=1) == 0)] = 0
            monitor(current + 1, total)
            current += 1

        accessor, _ = connect_digraph(window_length=10, vertices=where(usable == 1)[0],
                                      usable_outs=arange(1, 5), usable_ins=arange(1, 5))

        save_data(file_path=raw_path + "experiment.case-7.buffer-graph.pkl", information=accessor)

    if not path.exists(raw_path + "experiment.case-8.raw-nanopore.pkl"):
        print("Link buffer segments for Nanopore sequencing:")
        selected_addresses = load_data(file_path=raw_path + "experiment.case-6.raw-selected-addresses.pkl")
        selected_sequences = load_data(file_path=raw_path + "experiment.case-4.raw-combined-sequences.pkl")
        core_sequences, buffer_segments = [], []
        for frame_index in range(16):
            for raw_index in range(10):
                forward_address, reverse_address = selected_addresses[(frame_index, raw_index)]
                information_sequence = selected_sequences[(frame_index, raw_index)][0]
                sequence = anchor + forward_address + information_sequence + reverse_address + anchor
                core_sequences.append(sequence)

        buffer_method = SpiderWeb(accessor=load_data(raw_path + "experiment.case-7.buffer-graph.pkl"),
                                  check_3=False)
        random.seed(seed)
        while len(buffer_segments) < 2:  # only 2 buffer segment: forward and reverse.
            virtuality = random.choice(len(buffer_method.get_params()["vertices"]))
            # noinspection PyTypeChecker
            segments = buffer_method.produce(length=buffer_length, virtuality=virtuality,
                                             duplicatable=True, palindromable=True, diversity=10)[0]
            for segment in segments:
                if detect_head(segment=segment, nucleotide="T", length=5):
                    continue
                if detect_tail(segment=segment, nucleotide="T", length=5):
                    continue

                flag = True
                for core_sequence in core_sequences:
                    if len(buffer_segments) == 0:
                        temp_sequence = segment + core_sequence
                    else:  # len(buffer_segments) == 1:
                        temp_sequence = buffer_segments[0] + core_sequence + segment

                    # check the minimum spacing between the same 10-mer.
                    minimum_value_1 = compute_minimum_duplicate_spacing(segment=temp_sequence, window_size=window_size)
                    if minimum_value_1 < near_threshold_1:
                        flag = False
                        break

                    # check the minimum spacing between the 10 mer and its complementary palindrome.
                    minimum_value_2 = compute_minimum_palindrome_spacing(segment=temp_sequence, window_size=window_size)
                    if minimum_value_2 < near_threshold_2:
                        flag = False
                        break

                if not flag:
                    continue

                if len(buffer_segments) > 0 and levenshtein(buffer_segments[0], segment) <= buffer_length // 2:
                    continue

                buffer_segments.append(segment)

        whole_sequences, current, total = [], 0, 160
        for frame_index in range(16):
            for raw_index in range(10):
                forward_address, reverse_address = selected_addresses[(frame_index, raw_index)]
                information_sequence = selected_sequences[(frame_index, raw_index)][0]
                core_sequence = anchor + forward_address + information_sequence + reverse_address + anchor
                marker = "seq." + str(frame_index + 1).zfill(2) + "-" + str(raw_index + 1).zfill(2)
                whole_sequence = buffer_segments[0] + core_sequence + buffer_segments[1]
                whole_sequences.append((marker, whole_sequence))
                monitor(current + 1, total)
                current += 1

        save_data(file_path=raw_path + "experiment.case-8.raw-nanopore.pkl", information=whole_sequences)

    if not path.exists(raw_path + "experiment.case-9.raw-per-check.pkl"):
        print("Verify recoverability under the lossless condition:")
        frames, detections = load_data(file_path=raw_path + "experiment.case-2.figures.npy"), []
        addresses = load_data(file_path=raw_path + "experiment.case-6.raw-selected-addresses.pkl")
        nanopore_sequences = load_data(file_path=raw_path + "experiment.case-8.raw-nanopore.pkl")
        for current, (marker, sequence) in enumerate(nanopore_sequences):
            if "A" * (window_size // 2) in sequence:
                sequence = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
            parts = split(compile("T{5,}"), sequence)

            obtained_forward_address, obtained_reverse_address = parts[1], parts[-2]
            index_0, index_1 = None, None
            for (frame_index, raw_index), (forward_address, reverse_address) in addresses.items():
                forward_distance = levenshtein(forward_address, obtained_forward_address)
                reverse_distance = levenshtein(reverse_address, obtained_reverse_address)
                if forward_distance == 0 and reverse_distance == 0:
                    index_0, index_1 = frame_index, raw_index
                    break

            for index_2, part in enumerate(parts[2:-2]):
                obtained_data = packbits(method.decode(sequence=part, length=3 * 2 * 8, duplicatable=True,
                                                       palindromable=True)).reshape(3, 2)
                # noinspection PyTypeChecker
                expected_data = frames[index_0,
                                       index_1 * 3: (index_1 + 1) * 3,
                                       index_2 * 2: (index_2 + 1) * 2]

                if all(obtained_data == expected_data):
                    detections.append(True)
                else:
                    detections.append(False)

            monitor(current + 1, len(nanopore_sequences))

        count = 0
        for detection in enumerate(detections):
            if detection:
                count += 1
        print("%d out of 3200 blocks were successfully retrieved." % count)

        save_data(file_path=raw_path + "experiment.case-9.raw-per-check.pkl", information=detections)

    if not path.exists(raw_path + "experiment.case-10.retrieval-process.pkl"):
        reference = load_data(file_path=raw_path + "experiment.case-2.figures.npy")
        default_frame = ones(shape=(30, 40), dtype=uint8) * 255
        retrieval_folder = ""

        use("TkAgg")

        pyplot.ion()
        fig, ax = pyplot.subplots()
        image = ax.imshow(default_frame, cmap="gray", vmin=0, vmax=255)
        title = ax.set_title("No data")
        label = ax.set_xlabel("Initializing")
        ax.set_xticks([])
        ax.set_yticks([])

        update, process, last_time = 0, 0, datetime.now()

        while True:
            if not pyplot.fignum_exists(fig.number):
                break

            used_time = "%.2f" % (datetime.now() - last_time).total_seconds()

            if path.exists(retrieval_folder + "update-" + str(update + 1) + ".npy"):
                try:
                    frames = load_data(file_path="./case/update-" + str(update + 1) + ".npy")

                    count = 0
                    for frame_index in range(16):
                        for index_1 in range(10):
                            for index_2 in range(8):
                                expected_block = reference[frame_index,
                                                           index_1 * 3: (index_1 + 1) * 3,
                                                           index_2 * 2: (index_2 + 1) * 2]
                                obtained_block = frames[frame_index,
                                                        index_1 * 3: (index_1 + 1) * 3,
                                                        index_2 * 2: (index_2 + 1) * 2]
                                if all(expected_block == obtained_block):
                                    count += 1
                    process = count

                    title.set_text("updated round " + str(update + 1))
                    for frame_index in range(16):
                        image.set_data(frames[frame_index])
                        label.set_text(used_time + " seconds, " + str(process) + "/3200 correctly retrieved")
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                    update += 1

                    if process == 3200:
                        break

                except PermissionError:
                    pass

            else:
                used_time = "%.2f" % (datetime.now() - last_time).total_seconds()
                title.set_text("Wait for next round")
                if process > 0:
                    label.set_text(used_time + " seconds, " + str(process) + "/3200 correctly retrieved")
                else:
                    label.set_text(used_time + " seconds, 0/3200 correctly retrieved")
                image.set_data(default_frame)
                fig.canvas.draw()
                fig.canvas.flush_events()


if __name__ == "__main__":
    # The synthetic order submitted on July 31, 2023, sets both the virtual vertex and the random number seed to 2023.
    experiment_1(seed=2023, virtuality=2023)
    experiment_2(seed=2023)
