from datetime import datetime
from numpy import array, zeros, ones, linspace, arange, vstack, mgrid, mean, sqrt, std, min, sum, max, log10, where
from os import path
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, wasserstein_distance_nd
from warnings import filterwarnings

from sbw import load_data, save_data

from works import raw_path, exp_path, pkg_path
from works import estimate_sequence_copies

filterwarnings("ignore")


def supp_f01():
    """
    Create plotting data for Figure S1 in the supplementary file.

    TITLE: Complete pipeline of coding algorithm generation.
    """
    pass  # illustration


def supp_f02():
    """
    Create plotting data for Figure S2 in the supplementary file.

    TITLE: Algorithm generation runtime under commonly used constraints.
    """
    if not path.exists(pkg_path + "supp-02.pkl"):
        records = load_data(file_path=raw_path + "generation-1.general-cases.pkl")

        runtimes = -ones(shape=(10, 6))
        for key, (accessor, densities, runtime) in records.items():
            runtimes[key[0] - 1, int(key[1] * 10)] = runtime

        save_data(file_path=pkg_path + "supp-02.pkl", information=runtimes)


def supp_f03():
    """
    Create plotting data for Figure S3 in the supplementary file.

    TITLE: Message encoding and sequence decoding process using the basic mode.
    """
    pass  # illustration


def supp_f04():
    """
    Create plotting data for Figure S4 in the supplementary file.

    TITLE: Information density comparison of DNA Fountain, SPIDER-WEB (basic mode), and SPIDER-WEB (fast mode).
    """
    if not path.exists(pkg_path + "supp-04.pkl"):
        records_1 = load_data(file_path=raw_path + "supplement-simulation.conversion-1.fountain.pkl")
        records_2 = load_data(file_path=raw_path + "supplement-simulation.conversion-2.basic.pkl")
        panel_a_data, panel_c_data, panel_e_data = -ones(shape=(10, 6)), -ones(shape=(10, 6)), -ones(shape=(10, 6))
        panel_b_data, panel_d_data, panel_f_data = -ones(shape=(10, 6)), -ones(shape=(10, 6)), -ones(shape=(10, 6))
        for homopolymer in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for gc_bias in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                record_1 = []
                for data in records_1[(homopolymer, gc_bias)]:
                    if data[0] > 0:
                        record_1.append(data[0])
                if len(record_1) > 0:
                    panel_a_data[homopolymer - 1, int(gc_bias * 10)] = mean(record_1)
                    panel_b_data[homopolymer - 1, int(gc_bias * 10)] = std(record_1)
                panel_c_data[homopolymer - 1, int(gc_bias * 10)] = mean(records_2[(homopolymer, gc_bias)][0])
                panel_d_data[homopolymer - 1, int(gc_bias * 10)] = std(records_2[(homopolymer, gc_bias)][0])
                panel_e_data[homopolymer - 1, int(gc_bias * 10)] = mean(records_2[(homopolymer, gc_bias)][1])
                panel_f_data[homopolymer - 1, int(gc_bias * 10)] = std(records_2[(homopolymer, gc_bias)][1])
        results = {"a": panel_a_data, "b": panel_b_data, "c": panel_c_data,
                   "d": panel_d_data, "e": panel_e_data, "f": panel_f_data}
        save_data(file_path=pkg_path + "supp-04.pkl", information=results)


def supp_f05():
    """
    Create plotting data for Figure S5 in the supplementary file.

    TITLE: Error-correcting performance of HEDGES.
    """
    if not path.exists(pkg_path + "supp-05.pkl"):
        results = {}

        panel_a_data, panel_b_data = [], zeros(shape=(9,))

        records = load_data(file_path=raw_path + "supplement-simulation.correction-4.astar.pkl")
        for key, values in records.items():
            counts = []
            for flag, count in values:
                counts.append(count)
                if flag:
                    panel_b_data[key] += 1
            panel_a_data.append(counts)
        results["a"] = panel_a_data
        results["b"] = panel_b_data
        save_data(file_path=pkg_path + "supp-05.pkl", information=results)


def supp_f06():
    """
    Create plotting data for Figure S6 in the supplementary file.

    TITLE: Differences in data retrieval between Composite HEDGES Nanopores and SPIDER-WEB.
    """
    if not path.exists(pkg_path + "supp-06.pkl"):
        speeds, runtimes, reads = [], {}, [[], []]

        raw_data_1 = load_data(file_path=raw_path + "supplement-simulation.retrieving-5.chn-raw-data.pkl")
        raw_data_2 = load_data(file_path=raw_path + "supplement-simulation.retrieving-7.web-raw-data.pkl")
        eva_data_1 = load_data(file_path=raw_path + "supplement-simulation.retrieving-6.chn-evaluation.pkl")
        eva_data_2 = load_data(file_path=raw_path + "supplement-simulation.retrieving-8.web-evaluation.pkl")
        for chn_record, web_record in zip(eva_data_1, eva_data_2):
            if chn_record[0] not in runtimes:
                runtimes[chn_record[0]] = [[], []]

            sub_data_1 = raw_data_1[(chn_record[0], chn_record[1])]
            sub_data_2 = raw_data_2[(web_record[0], web_record[1])]

            value_1_1 = chn_record[3]
            value_1_2 = len(sub_data_1[1]) * chn_record[4]
            speeds.append((int(chn_record[0][:-2]) * 8 * 1024) / value_1_1)

            value_2_1 = web_record[3]
            value_2_2 = len(sub_data_2[1]) * web_record[4]
            runtimes[chn_record[0]][0].append(value_1_1)
            runtimes[chn_record[0]][1].append(value_2_1)
            reads[0].append(value_1_2)
            reads[1].append(value_2_2)

        save_data(file_path=pkg_path + "supp-06.pkl", information={"a": speeds, "b": runtimes, "c": reads})


def supp_f07():
    """
    Create plotting data for Figure S7 in the supplementary file.

    TITLE: Effects of sequence diversity and error rate on the average unintended proportion
           across different sequence copy numbers, grouped by sequence diversity.
    """
    if not path.exists(pkg_path + "supp-07.pkl"):
        matrix = zeros(shape=(8, 50, 8))
        for diversity, reads, errors, value, _ in load_data(file_path=raw_path + "simulation.retrieving-2.rates.pkl"):
            matrix[int(log10(diversity)) - 1, reads - 1, errors - 1] = value

        results = {}
        for error_index in range(8):
            results[chr(ord("a") + error_index)] = []
            for diversity_index in range(8):
                values = matrix[diversity_index, :, error_index] / (10 ** (diversity_index + 1))
                if max(values) == 1.0:
                    stop = where(values == 1.0)[0][0]
                    points = [0] + log10(1 - values[:stop]).tolist()
                else:
                    points = [0] + log10(1 - values).tolist()
                results[chr(ord("a") + error_index)].append(array(points, dtype=float))

        save_data(file_path=pkg_path + "supp-07.pkl", information=results)


def supp_f08():
    """
    Create plotting data for Figure S8 in the supplementary file.

    TITLE: Effects of sequence diversity and error rate on the average unintended proportion
           across different sequence copy numbers, grouped by error rate.
    """
    if not path.exists(pkg_path + "supp-08.pkl"):
        matrix = zeros(shape=(8, 50, 8))
        for diversity, reads, errors, value, _ in load_data(file_path=raw_path + "simulation.retrieving-2.rates.pkl"):
            matrix[int(log10(diversity)) - 1, reads - 1, errors - 1] = value

        results = {}
        for diversity_index in range(8):
            results[chr(ord("a") + diversity_index)] = []
            for error_index in range(8):
                values = matrix[diversity_index, :, error_index] / (10 ** (diversity_index + 1))
                if max(values) == 1.0:
                    stop = where(values == 1.0)[0][0]
                    points = [0] + log10(1 - values[:stop]).tolist()
                else:
                    points = [0] + log10(1 - values).tolist()
                results[chr(ord("a") + diversity_index)].append(array(points, dtype=float))

        save_data(file_path=pkg_path + "supp-08.pkl", information=results)


def supp_f09():
    """
    Create plotting data for Figure S9 in the supplementary file.

    TITLE: Inner-outer code integration and its loss tolerance.
    """
    if not path.exists(pkg_path + "supp-09.pkl"):
        data_1 = load_data(raw_path + "supplement-simulation.retrieving-1.outer-baseline.pkl")
        loss_rates, redundant_rates = linspace(0.00, 0.10, 11), linspace(0.01, 0.50)
        results = {}

        matrix_1, values_1 = zeros(shape=(51, 11)), data_1["xor"]
        matrix_1[0] = 1 - loss_rates
        for index_1, redundant_rate in enumerate(redundant_rates):
            for index_2, loss_rate in enumerate(loss_rates):
                info = ("%.2f" % redundant_rate, "%.2f" % loss_rate)
                matrix_1[index_1 + 1, index_2] = mean(values_1[info])
        matrix_2, values_2 = zeros(shape=(51, 11)), data_1["reed-solomon"]
        matrix_2[0] = 1 - loss_rates
        for index_1, redundant_rate in enumerate(redundant_rates):
            for index_2, loss_rate in enumerate(loss_rates):
                info = ("%.2f" % redundant_rate, "%.2f" % loss_rate)
                matrix_2[index_1 + 1, index_2] = mean(values_2[info])
        matrix_3, values_3 = zeros(shape=(51, 11)), data_1["fountain"]
        for index_1, redundant_rate in enumerate(redundant_rates):
            for index_2, loss_rate in enumerate(loss_rates):
                info = ("%.2f" % redundant_rate, "%.2f" % loss_rate)
                matrix_3[index_1 + 1, index_2] = mean(values_3[info])
        matrix_3[0] = matrix_3[1]
        results["b"] = (matrix_1, matrix_2, matrix_3)

        data_2, values = load_data(raw_path + "supplement-simulation.retrieving-2.outer-spider.pkl"), {}
        for index, label in enumerate(["xor", "reed-solomon", "fountain"]):
            pairs = [[], []]
            for redundant_rate in redundant_rates:
                for loss_rate in loss_rates:
                    info = ("%.2f" % redundant_rate, "%.2f" % loss_rate)
                    pairs[0].append(data_2[label][info])
                    pairs[1].append(data_1[label][info])
            x, y = [min(pairs[0]), max(pairs[0])], [min(pairs[1]), max(pairs[1])]
            values[label] = (x, y)
        results["c"] = values

        save_data(file_path=pkg_path + "supp-09.pkl", information=results)


def supp_f10():
    """
    Create plotting data for Figure S10 in the supplementary file.

    TITLE: Evaluation methods and result analysis for handling runtime differences.
    """
    if not path.exists(pkg_path + "supp-10.pkl"):
        data = load_data(raw_path + "supplement-simulation.retrieving-4.redundant-retrieval.pkl")
        redundant_rates, error_params, reads_params = linspace(0.01, 0.50), arange(0, 9), arange(1, 51)
        results = {}

        matrix = [[] for _ in range(50)]
        for errors in error_params:
            values_1 = data["basic"][("%.1f" % (errors / 2)) + "%"][:, :, 0]
            for redundant_index, redundant_rate in enumerate(redundant_rates):
                values_2 = data["xor"][str(redundant_index + 1) + "%"][("%.1f" % (errors / 2)) + "%"][:, :, 0]
                differences = (values_2 / values_1).reshape(-1)
                matrix[redundant_index] += differences.tolist()
        centers_1 = []
        for index in range(50):
            centers_1.append(mean(matrix[index]))

        matrix = [[] for _ in range(50)]
        for errors in error_params:
            values_1 = data["basic"][("%.1f" % (errors / 2)) + "%"][:, :, 0]
            for redundant_index, redundant_rate in enumerate(redundant_rates):
                values_2 = data["reed-solomon"][str(redundant_index + 1) + "%"][("%.1f" % (errors / 2)) + "%"][:, :, 0]
                differences = (values_2 / values_1).reshape(-1)
                matrix[redundant_index] += differences.tolist()
        centers_2 = []
        for index in range(50):
            centers_2.append(mean(matrix[index]))

        matrix = [[] for _ in range(50)]
        for errors in error_params:
            values_1 = data["basic"][("%.1f" % (errors / 2)) + "%"][:, :, 0]
            for redundant_index, redundant_rate in enumerate(redundant_rates):
                values_2 = data["fountain"][str(redundant_index + 1) + "%"][("%.1f" % (errors / 2)) + "%"][:, :, 0]
                differences = (values_2 / values_1).reshape(-1)
                matrix[redundant_index] += differences.tolist()

        centers_3 = []
        for index in range(50):
            centers_3.append(mean(matrix[index]))

        results["b"] = (centers_1, centers_2, centers_3)

        save_data(file_path=pkg_path + "supp-10.pkl", information=results)


def supp_f11():
    """
    Create plotting data for Figure S11 in the supplementary file.

    TITLE: The performance difference between "MMSeqs2+MUSCLE5" and "MMSeqs2+USEARCH".
    """
    if not path.exists(pkg_path + "supp-11.pkl"):
        records_1 = load_data(file_path=raw_path + "supplement-experiment.retrieving-1.default.pkl")
        records_2 = load_data(file_path=raw_path + "experiment.retrieval-7.performances.pkl")

        success_data_pair, runtime_data_pair = [], []
        for key, value_1 in records_1.items():
            value_2 = records_2[key][0]
            success_data_pair.append([value_1[0], value_2[0]])
            runtime_data_pair.append([value_1[1], value_2[1]])
        success_data_pair, runtime_data_pair = array(success_data_pair), array(runtime_data_pair)

        save_data(file_path=pkg_path + "supp-11.pkl", information={"a": success_data_pair, "b": runtime_data_pair})


def supp_f12():
    """
    Create plotting data for Figure S12 in the supplementary file.

    TITLE: Different data retrieval approaches in DNA-based data storage.
    """
    pass  # illustration


def supp_f13():
    """
    Create plotting data for Figure S13 in the supplementary file.

    TITLE: Effects of sequence diversity and error rate on the average of maximum frequency of
           a false-positive sequence candidate in dictionary across different sequence copies,
           grouped by error rate.
    """
    if not path.exists(pkg_path + "supp-13.pkl"):
        matrix = zeros(shape=(8, 50, 8))
        for diversity, reads, errors, _, value in load_data(file_path=raw_path + "simulation.retrieving-2.rates.pkl"):
            matrix[int(log10(diversity)) - 1, reads - 1, errors - 1] = value

        results = {}
        for error_index in range(8):
            results[chr(ord("a") + error_index)] = []
            for diversity_index in range(8):
                results[chr(ord("a") + error_index)].append(matrix[diversity_index, :, error_index])

        save_data(file_path=pkg_path + "supp-13.pkl", information=results)


def supp_f14():
    """
    Create plotting data for Figure S14 in the supplementary file.

    TITLE: Effects of sequence diversity and error rate on the average of maximum frequency of
           a false-positive sequence candidate in dictionary across different sequence copies,
           grouped by sequence diversity.
    """
    if not path.exists(pkg_path + "supp-14.pkl"):
        matrix = zeros(shape=(8, 50, 8))
        for diversity, reads, errors, _, value in load_data(file_path=raw_path + "simulation.retrieving-2.rates.pkl"):
            matrix[int(log10(diversity)) - 1, reads - 1, errors - 1] = value

        results = {}
        for diversity_index in range(8):
            results[chr(ord("a") + diversity_index)] = []
            for error_index in range(8):
                results[chr(ord("a") + diversity_index)].append(matrix[diversity_index, :, error_index])

        save_data(file_path=pkg_path + "supp-14.pkl", information=results)


def supp_f15():
    """
    Create plotting data for Figure S15 in the supplementary file.

    TITLE: Illustration of global constraints.
    """
    pass  # illustration


def supp_f16():
    """
    Create plotting data for Figure S16 in the supplementary file.

    TITLE: Valid number of encoded segments in each frame.
    """
    if not path.exists(pkg_path + "supp-16.pkl"):
        segment_info = load_data(file_path=raw_path + "experiment.case-3.raw-payloads.pkl")
        count_data, minimum_value, maximum_value = [], 1000, 0
        for frame_index in range(16):
            count_data.append([])
            for raw_index in range(10):
                values = [len(segment_info[(frame_index, raw_index, column_index)]) for column_index in range(20)]
                count_data[-1].append(values)
            minimum_value = min([minimum_value, min(array(count_data[-1]))])
            maximum_value = max([maximum_value, max(array(count_data[-1]))])
            count_data[-1] = log10(array(count_data[-1]))

        save_data(file_path=pkg_path + "supp-16.pkl", information=(count_data, minimum_value, maximum_value))


def supp_f17():
    """
    Create plotting data for Figure S17 in the supplementary file.

    TITLE: The distribution of minimum spacing between duplicated 10-mers.
    """
    if not path.exists(pkg_path + "supp-17.pkl"):
        window_size, payload_length, address_length, buffer_length = 10, 110, 60, 200
        sequence_data = load_data(file_path=raw_path + "experiment.case-8.raw-nanopore.pkl")
        panel_data = [[[] for _ in range(10)] for _ in range(16)]
        for marker, sequence in sequence_data:
            sequence = sequence[280:-280]
            index_1, index_2 = [int(v) - 1 for v in marker[4:].split("-")]
            k_mer_set = {}
            for index in range(len(sequence) - (window_size - 1)):
                k_mer = sequence[index: index + window_size]
                if k_mer in k_mer_set:
                    k_mer_set[k_mer].append(index)
                else:
                    k_mer_set[k_mer] = [index]

            # noinspection PyRedeclaration
            for k_mer, locations in k_mer_set.items():
                locations = array(sorted(locations))
                distances = locations[1:] - locations[:-1]
                if len(distances) > 0:
                    panel_data[index_1][index_2].append(min(distances) - 10)

        save_data(file_path=pkg_path + "supp-17.pkl", information=panel_data)


def supp_f18():
    """
    Create plotting data for Figure S18 in the supplementary file.

    TITLE: The distribution of minimum spacing between a 10-mer and its reverse complement segment.
    """
    if not path.exists(pkg_path + "supp-18.pkl"):
        window_size, payload_length, address_length, buffer_length = 10, 110, 60, 200
        sequence_data = load_data(file_path=raw_path + "experiment.case-8.raw-nanopore.pkl")
        panel_data = [[[] for _ in range(10)] for _ in range(16)]
        for marker, sequence in sequence_data:
            sequence = sequence[280:-280]
            index_1, index_2 = [int(v) - 1 for v in marker[4:].split("-")]
            k_mer_set = {}
            for index in range(len(sequence) - (window_size - 1)):
                k_mer = sequence[index: index + window_size]
                if k_mer in k_mer_set:
                    k_mer_set[k_mer].append(index)
                else:
                    k_mer_set[k_mer] = [index]
            chuck_sequence = sequence.translate(str.maketrans("ACGT", "TGCA"))
            distances = {}
            for index in range(len(chuck_sequence) - (window_size - 1)):
                motif = chuck_sequence[index: index + window_size][::-1]
                if motif in k_mer_set:
                    minimum_value, locations = 1000, array(sorted(k_mer_set[motif]))
                    formers = where(locations < index - window_size)[0]  # valid upstream matches.
                    latters = where(locations > index + window_size)[0]  # valid downstream matches.
                    if len(formers) > 0:
                        minimum_value = min([minimum_value, min(index - (locations[formers] + window_size))])
                    if len(latters) > 0:
                        # noinspection PyTypeChecker
                        minimum_value = min([minimum_value, min((locations[latters] - (index + window_size)))])
                    if motif in distances:
                        # noinspection PyTypeChecker
                        distances[motif] = min([minimum_value, distances[motif]])
                    else:
                        distances[motif] = minimum_value
            values = [value for _, value in distances.items()]
            panel_data[index_1][index_2] = values

        save_data(file_path=pkg_path + "supp-18.pkl", information=panel_data)


def supp_f19():
    """
    Create plotting data for Figure S19 in the supplementary file.

    TITLE: Observation and modelling of instantaneous efficiency.
    """

    def function(values, parameter_1, parameter_2):
        return (parameter_1 * (1 - values)) / (values ** parameter_2)

    if not path.exists(pkg_path + "supp-19.pkl"):
        child_paths = ["250F501329017/statistics.csv", "250F501329018/statistics.csv",
                       "250F501329019/statistics.csv", "250F50132901A/statistics.csv",
                       "250F50132901B/statistics.csv"]

        points, sub_groups, starts = [], [], []
        for tried_index, child_path in enumerate(child_paths):
            with open(exp_path + "3/" + child_path) as file:
                # noinspection PyRedeclaration
                last, former_time, former_blocks, sub_points = None, 0.0, 0.0, []
                for index, line in enumerate(file.readlines()):
                    if index > 0:
                        info = line.strip().split(",")
                        timestamp = info[1][len("2025-09-12T"): -len("+08:00")]
                        value = datetime.strptime(timestamp, "%H:%M:%S.%f")
                        seconds = value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
                        if last is None:
                            last = seconds
                        runtime = seconds - last
                        processing_time = int(info[2]) / 1e6
                        used_time = runtime + processing_time
                        retrieval_blocks = int(info[3])
                        if retrieval_blocks > former_blocks:
                            change_x = used_time - former_time
                            change_y = retrieval_blocks - former_blocks
                            if index == 1:
                                starts.append(change_y / change_x * 48.0)
                            sub_points.append([retrieval_blocks / 3200.0, change_y / change_x * 48.0])
                            former_blocks = retrieval_blocks
                            former_time = used_time
                points += sub_points
                sub_groups.append(array(sub_points))

        points = array(points)

        # noinspection PyTupleAssignmentBalance
        params, _ = curve_fit(function, points[:, 0], points[:, 1])

        save_data(file_path=pkg_path + "supp-19.pkl", information=(sub_groups, points, params))


def supp_f20():
    """
    Create plotting data for Figure S20 in the supplementary file.

    TITLE: Operating factor of SPIDER-WEB in the real-world data retrieval demonstration.
    """
    if not path.exists(pkg_path + "supp-20.pkl"):
        panel_data = {}
        child_paths = ["250F501329017/statistics.csv", "250F501329018/statistics.csv",
                       "250F501329019/statistics.csv", "250F50132901A/statistics.csv",
                       "250F50132901B/statistics.csv"]
        values = []
        for tried_index, child_path in enumerate(child_paths):
            with open(exp_path + "3/" + child_path) as file:
                last, values_1, values_2 = None, [], []
                for index, line in enumerate(file.readlines()):
                    if index > 0:
                        info = line.strip().split(",")
                        timestamp = info[1][len("2025-09-12T"): -len("+08:00")]
                        value = datetime.strptime(timestamp, "%H:%M:%S.%f")
                        seconds = value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
                        if last is None:
                            last = seconds
                        runtime = seconds - last
                        processing_time = int(info[2]) / 1e6
                        values_1.append(runtime)
                        values_2.append(processing_time)
                values_1 = array(values_1)
                values_2 = array(values_2)
                time_shots = values_1[1:] - values_1[:-1]
                values_2 = values_2[:-1]
                collection = values_2 / time_shots
                for runtime, factor in zip(values_1[:-1], collection):
                    values.append([runtime / values_1[-1], factor])

        values = array(values).T
        x_values, y_values = linspace(0, 1, 100), linspace(0.0, max(values[1]), 100)
        x_indices, y_indices = mgrid[x_values[0]:x_values[-1]:100j, y_values[0]:y_values[-1]:100j]
        positions = vstack([x_indices.ravel(), y_indices.ravel()])
        z_values = gaussian_kde(values)(positions)
        z_values = z_values.reshape(len(x_values), len(y_values))
        markers = sum(z_values, axis=1, keepdims=True)
        markers[markers == 0] = 1
        z_values /= markers

        panel_data["a"] = (x_values, y_values, z_values)

        values, markers = [[] for _ in range(10)], linspace(0.1, 1.0, 10)
        for tried_index, child_path in enumerate(child_paths):
            with open(exp_path + "3/" + child_path) as file:
                last, values_1, values_2 = None, [], []
                for index, line in enumerate(file.readlines()):
                    if index > 0:
                        info = line.strip().split(",")
                        timestamp = info[1][len("2025-09-12T"): -len("+08:00")]
                        value = datetime.strptime(timestamp, "%H:%M:%S.%f")
                        seconds = value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
                        if last is None:
                            last = seconds
                        runtime = seconds - last
                        processing_time = int(info[2]) / 1e6
                        values_1.append(runtime)
                        values_2.append(processing_time)
                values_1 = array(values_1)
                values_2 = array(values_2)
                time_shots = values_1[1:] - values_1[:-1]
                values_2 = values_2[:-1]
                collection = values_2 / time_shots
                for runtime, factor in zip(values_1[:-1], collection):
                    for index, value in enumerate(markers):
                        if runtime / values_1[-1] <= value:
                            values[index].append(factor)
                            break

        distributions, x_values = [], linspace(0.0, 0.1, 101)
        for index, points in enumerate(values):
            y_values = gaussian_kde(points)(x_values)
            y_values /= sum(y_values)
            distributions.append(y_values)

        matrix = zeros(shape=(10, 10))
        for index_1 in range(10):
            for index_2 in range(10):
                distance = wasserstein_distance_nd(distributions[index_1], distributions[index_2])
                matrix[index_1, index_2] = distance

        panel_data["b"] = matrix

        save_data(file_path=pkg_path + "supp-20.pkl", information=panel_data)


def supp_t01():
    """
    Create summarising data for Table S1 in the supplementary file.

    TITLE: The count of CHN encoding failures across random trials
    """
    if not path.exists(pkg_path + "supp-21.pkl"):
        records = {}
        data = load_data(file_path=raw_path + "supplement-simulation.retrieving-5.chn-raw-data.pkl")
        for file_size in ["1KB", "2KB", "4KB", "8KB", "16KB"]:
            records[file_size] = [None, []]
            for data_index in range(5):
                binary_messages, _, _, failure_number = data[(file_size, data_index)]
                if data_index == 0:
                    records[file_size][0] = len(binary_messages)
                records[file_size][1].append(failure_number)

        save_data(file_path=pkg_path + "supp-21.pkl", information=records)


def supp_t02():
    """
    Create summarising data for Table S2 in the supplementary file.

    TITLE: The fitted constant theta from non-linear least-squares fitting of the empirical scaling law.
    """
    if not path.exists(pkg_path + "supp-22.pkl"):
        fitted_matrix, fitted_param_group = zeros(shape=(8, 50, 8)), {}
        for diversity, reads, errors, value, _ in load_data(file_path=raw_path + "simulation.retrieving-2.rates.pkl"):
            fitted_matrix[int(log10(diversity)) - 1, reads - 1, errors - 1] = value
        records = {"a": [], "b": [], "c": [], "d": []}
        for error_index in range(8):
            points_x, points_y = [], []
            # noinspection PyTypeChecker
            for diversity in 10 ** arange(1, 9):
                if max(fitted_matrix[int(log10(diversity)) - 1, :, error_index] / diversity) == 1.0:
                    index = where(fitted_matrix[int(log10(diversity)) - 1, :, error_index] / diversity == 1.0)[0][0]
                    points_x += (1 - fitted_matrix[int(log10(diversity)) - 1, :index, error_index] / diversity).tolist()
                    points_y += arange(1, index + 1).tolist()
                else:
                    points_x += (1 - fitted_matrix[int(log10(diversity)) - 1, :, error_index] / diversity).tolist()
                    # noinspection PyUnresolvedReferences
                    points_y += (arange(50) + 1).tolist()
            points_x, points_y = array(points_x, dtype=float), array(points_y, dtype=float)
            # noinspection PyTupleAssignmentBalance
            fitted_params, _ = curve_fit(f=estimate_sequence_copies, xdata=points_x, ydata=points_y, p0=[1.0, 1.0])
            # noinspection PyTypeChecker
            new_points_y = array([estimate_sequence_copies(x, *fitted_params) for x in points_x])

            ss_res = sum((points_y - new_points_y) ** 2)
            ss_tot = sum((points_y - mean(points_y)) ** 2)
            r2 = 1 - ss_res / ss_tot

            # noinspection SpellCheckingInspection
            rmse = sqrt(mean((new_points_y - points_y) ** 2))

            records["a"].append(fitted_params[0])
            records["b"].append(fitted_params[1])
            records["c"].append(r2)
            records["d"].append(rmse)

        save_data(file_path=pkg_path + "supp-22.pkl", information=records)


def supp_t03():
    """
    Create summarising data for Table S3 in the supplementary file.

    TITLE: First observed instantaneous efficiency across replicate trials.
    """
    if not path.exists(pkg_path + "supp-23.pkl"):
        child_paths = ["250F501329017/statistics.csv", "250F501329018/statistics.csv",
                       "250F501329019/statistics.csv", "250F50132901A/statistics.csv",
                       "250F50132901B/statistics.csv"]

        initial_times, initial_speeds = [], []
        for tried_index, child_path in enumerate(child_paths):
            with open(exp_path + "3/" + child_path) as file:
                # noinspection PyRedeclaration
                last, former_time, former_blocks, sub_points = None, 0.0, 0.0, []
                for index, line in enumerate(file.readlines()):
                    if index > 0:
                        info = line.strip().split(",")
                        timestamp = info[1][len("2025-09-12T"): -len("+08:00")]
                        value = datetime.strptime(timestamp, "%H:%M:%S.%f")
                        seconds = value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
                        if last is None:
                            last = seconds
                        runtime = seconds - last
                        processing_time = int(info[2]) / 1e6
                        used_time = runtime + processing_time
                        retrieval_blocks = int(info[3])
                        if retrieval_blocks > former_blocks:
                            change_x = used_time - former_time
                            change_y = retrieval_blocks - former_blocks
                            if index == 1:
                                initial_times.append(used_time)
                                initial_speeds.append(change_y / change_x * 48.0 / (8 * 1024))
                                break

        save_data(file_path=pkg_path + "supp-23.pkl", information=(initial_speeds, initial_times))


def supp_t04():
    """
    Create summarising data for Table S4 in the supplementary file.

    TITLE: Nanopore basecalling performances.
    """
    pass  # The relevant data are derived from Tables 1 and 2 of the AWS HPC Blog.


if __name__ == "__main__":
    supp_f01()
    supp_f02()
    supp_f03()
    supp_f04()
    supp_f05()
    supp_f06()
    supp_f07()
    supp_f08()
    supp_f09()
    supp_f10()
    supp_f11()
    supp_f12()
    supp_f13()
    supp_f14()
    supp_f15()
    supp_f16()
    supp_f17()
    supp_f18()
    supp_f19()
    supp_f20()
    supp_t01()
    supp_t02()
    supp_t03()
    supp_t04()
