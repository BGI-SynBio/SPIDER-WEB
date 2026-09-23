from datetime import datetime
from Levenshtein import distance
from numpy import array, zeros, ones, arange, linspace, polyfit, meshgrid, vstack
from numpy import mean, min, max, sum, ceil, log, log10, where
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, wasserstein_distance_nd
from os import path
from warnings import filterwarnings

from sbw import load_data, save_data

from works import raw_path, exp_path, pkg_path
from works import estimate_sequence_copies, estimate_runtime_1, estimate_runtime_2

filterwarnings("ignore")


def main_f01():
    """
    Create plotting data for Figure 1 in the main text.

    TITLE: Overview of this work.
    """
    if not path.exists(pkg_path + "main-01.pkl"):
        results = {}

        records_1 = load_data(file_path=raw_path + "simulation.correction-4.readouts.pkl")
        records_2 = load_data(file_path=raw_path + "simulation.conversion-4.speeds.pkl")
        labels, values = zeros(shape=(6, 100)), zeros(shape=(6, 100))
        for index, key in enumerate(["10", "100", "1000", "10000", "100000", "1000000"]):
            labels[index] = int(key)
            values[index] = records_2[key]
        d_param_a, d_param_b = polyfit(log10(labels.reshape(-1)), log10(values.reshape(-1)), deg=1)
        fitted_matrix, fitted_param_group = zeros(shape=(8, 50, 8)), {}
        for diversity, reads, errors, value, _ in load_data(file_path=raw_path + "simulation.retrieving-2.rates.pkl"):
            fitted_matrix[int(log10(diversity)) - 1, reads - 1, errors - 1] = value
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
            fitted_param_group[error_index] = fitted_params
        # based on the speed information collected above.
        correcting_speeds = array([1e6 / mean(records_1[1][errors]) for errors in arange(1, 9)])
        discussed_error_index = 0
        fitted_params = fitted_param_group[discussed_error_index]
        correcting_speed = correcting_speeds[discussed_error_index]
        x_values = linspace(1, 3, 201)
        y_values = linspace(1, 6, 501)
        matrix = ones(shape=(len(x_values), len(y_values)))
        for x_index, x_value in enumerate(x_values):
            for y_index, y_value in enumerate(y_values):
                required_throughput = 8 * (1024 ** x_value)
                bit_number = 8 * (1024 ** y_value)
                # 0.724 is the practical information density, calculating when the special coding digraph is generated.
                # it calculated when sequence length is 200 nucleotides.
                # check https://github.com/HaolingZHANG/RealTimeRetrieval/blob/main/works/log/step-1.2.txt for details.
                nucleotide_number = bit_number / 0.724
                sequence_number = ceil(nucleotide_number / 200.0)
                error_rate = 1.0 / sequence_number
                required_copy_number = estimate_sequence_copies(error_rate, *fitted_params)
                reads_number = sequence_number * required_copy_number
                expected_time = bit_number / required_throughput
                decoding_speed = sequence_number / (10 ** (d_param_a * log10(sequence_number) + d_param_b))
                if reads_number / correcting_speed + sequence_number / decoding_speed < expected_time * 0.95:
                    matrix[x_index, y_index] = -1  # will mask as none later.
                else:
                    threads = 1
                    while True:
                        correcting_time = reads_number / (threads * correcting_speed)
                        decoding_time = sequence_number / (threads * decoding_speed)
                        if correcting_time + decoding_time <= expected_time:
                            break
                        threads += 1
                    matrix[x_index, y_index] = threads
        ranges = []
        for discussed_error_index in range(8):
            fitted_params = fitted_param_group[discussed_error_index]
            correcting_speed = correcting_speeds[discussed_error_index]
            # maximum 1-terabyte hard drive
            storage_capacity, bound_y_value = 8 * (1024 ** 4), None
            for y_index, y_value in enumerate(y_values):
                bit_number = 8 * (1024 ** y_value)
                # 0.724 is the practical information density, calculating when the special coding digraph is generated.
                # it calculated when sequence length is 200 nucleotides.
                # check https://github.com/HaolingZHANG/RealTimeRetrieval/blob/main/works/log/step-1.2.txt for details.
                nucleotide_number = bit_number / 0.724
                sequence_number = ceil(nucleotide_number / 200.0)
                error_rate = 1.0 / sequence_number
                required_copy_number = estimate_sequence_copies(error_rate, *fitted_params)
                if required_copy_number * sequence_number * 200 * 8 <= storage_capacity:
                    bound_y_value = y_value
                else:
                    break
            # use 16 threads
            bound_x_values = []
            for y_value in [bound_y_value, 1]:
                bit_number = 8 * (1024 ** y_value)
                # 0.724 is the practical information density, calculating when the special coding digraph is generated.
                # it calculated when sequence length is 200 nucleotides.
                # check https://github.com/HaolingZHANG/RealTimeRetrieval/blob/main/works/log/step-1.2.txt for details.
                nucleotide_number = bit_number / 0.724
                sequence_number = ceil(nucleotide_number / 200.0)
                error_rate = 1.0 / sequence_number
                required_copy_number = estimate_sequence_copies(error_rate, *fitted_params)
                reads_number = sequence_number * required_copy_number
                decoding_speed = sequence_number / (10 ** (d_param_a * log10(sequence_number) + d_param_b))
                runtime = reads_number / (16.0 * correcting_speed) + sequence_number / (16.0 * decoding_speed)
                bound_x_values.append(bit_number / runtime)
            ranges.append(bound_x_values)
        results["f"] = [min(ranges), max(ranges)]

        save_data(file_path=pkg_path + "main-01.pkl", information=results)


def main_f02():
    """
    Create plotting data for Figure 2 in the main text.

    TITLE: Error tolerance performance of SPIDER-WEB.
    """
    if not path.exists(pkg_path + "main-02.pkl"):
        results = {}

        panel_data_b, records = [], load_data(file_path=raw_path + "simulation.correction-2.single.pkl")
        for errors, (_, _, repaired_data_2, _, _, _, _) in records.items():
            values = []
            for _, info in repaired_data_2:
                values.append(info["access"])
            values = array(values)
            # noinspection PyUnresolvedReferences
            marker = linspace(min(values), max(values), max(values) - min(values) + 1)
            counts = zeros(shape=(len(marker),))
            for value in values:
                counts[where(marker == value)[0]] += 1
            panel_data_b.append((marker, counts / 1000000.0))
        results["b"] = panel_data_b

        panel_data_c = []
        for errors, (_, _, _, _, statistics_2, _, _) in records.items():
            contained, screened = sum(statistics_2[:, 1]), len(where(statistics_2[:, 0] == 0)[0])
            # noinspection PyUnresolvedReferences
            panel_data_c.append([contained, screened, 1000000 - (contained + screened)])
        results["c"] = array(panel_data_c) / 1000000.0

        panel_data_d, records = ones(shape=(9,)), load_data(file_path=raw_path + "simulation.correction-3.multiple.pkl")
        for errors, (_, record_2) in records.items():
            panel_data_d[errors] = mean(record_2)
        results["d"] = panel_data_d

        matrix = zeros(shape=(8, 50, 8))
        for diversity, reads, errors, value, _ in load_data(file_path=raw_path + "simulation.retrieving-2.rates.pkl"):
            matrix[int(log10(diversity)) - 1, reads - 1, errors - 1] = value

        panel_data_e = []
        for error_index in range(8):
            points_x, points_y = [], []
            for diversity_index in range(8):
                values = matrix[diversity_index, :, error_index] / (10 ** (diversity_index + 1))
                if max(values) == 1.0:
                    stop = where(values == 1.0)[0][0]
                    points_x += [0] + (-log10(1 - values[:stop])).tolist()
                    points_y += arange(0, stop + 1).tolist()
                else:
                    points_x += [0] + (-log10(1 - values)).tolist()
                    # noinspection PyUnresolvedReferences
                    points_y += [0] + (arange(50) + 1).tolist()
            panel_data_e.append([array(points_x, dtype=float), array(points_y, dtype=float)])
        results["e"] = panel_data_e

        panel_data_f = []
        for error_index in range(8):
            points_x, points_y = [], []
            # noinspection PyTypeChecker
            for diversity in 10 ** arange(1, 9):
                if max(matrix[int(log10(diversity)) - 1, :, error_index] / diversity) == 1.0:
                    index = where(matrix[int(log10(diversity)) - 1, :, error_index] / diversity == 1.0)[0][0]
                    points_x += (1 - matrix[int(log10(diversity)) - 1, :index, error_index] / diversity).tolist()
                    points_y += arange(1, index + 1).tolist()
                else:
                    points_x += (1 - matrix[int(log10(diversity)) - 1, :, error_index] / diversity).tolist()
                    # noinspection PyUnresolvedReferences
                    points_y += (arange(50) + 1).tolist()
            points_x, points_y = array(points_x, dtype=float), array(points_y, dtype=float)

            # noinspection PyTupleAssignmentBalance
            fitted_params, _ = curve_fit(f=estimate_sequence_copies, xdata=points_x, ydata=points_y, p0=[1.0, 1.0])
            sizes = linspace(1, 6, 101)
            # 0.724 is the practical information density, calculating when the special coding digraph is generated.
            # it calculated when sequence length is 200 nucleotides.
            # check https://github.com/HaolingZHANG/RealTimeRetrieval/blob/main/works/log/step-1.2.txt for details.
            sequence_numbers = (8 * (1024 ** sizes)) / (200.0 * 0.724)
            error_rates = 1.0 / sequence_numbers
            # noinspection PyTypeChecker
            copies = estimate_sequence_copies(error_rates, *fitted_params)
            panel_data_f.append([sizes, copies])
        results["f"] = panel_data_f

        save_data(file_path=pkg_path + "main-02.pkl", information=results)


def main_f03():
    """
    Create plotting data for Figure 3 in the main text.

    TITLE: Direct data retrieval capability.
    """
    if not path.exists(pkg_path + "main-03.pkl"):
        results = {}
        records = load_data(file_path=raw_path + "experiment.retrieval-2.similarities.npy")
        differences = [[] for _ in range(6)]
        for index_1 in range(0, 1000 - 1):
            for index_2 in range(index_1 + 1, 1000):
                differences[0].append(records[index_1, index_2])
        for index_1 in range(1000, 2000 - 1):
            for index_2 in range(index_1 + 1, 2000):
                differences[1].append(records[index_1, index_2])
        for index_1 in range(2000, 3000 - 1):
            for index_2 in range(index_1 + 1, 3000):
                differences[2].append(records[index_1, index_2])
        for index_1 in range(0, 1000):
            for index_2 in range(1000, 2000):
                differences[3].append(records[index_1, index_2])
        for index_1 in range(0, 1000):
            for index_2 in range(2000, 3000):
                differences[4].append(records[index_1, index_2])
        for index_1 in range(1000, 2000):
            for index_2 in range(2000, 3000):
                differences[5].append(records[index_1, index_2])
        panel_data_b = []
        for index in range(6):
            x = arange(min(differences[index]), max(differences[index]) + 1)
            y = zeros(shape=(len(x),), dtype=int)
            for difference in differences[index]:
                y[where(difference == x)[0]] += 1
            panel_data_b.append((x, y))
        results["b"] = panel_data_b

        records = load_data(file_path=raw_path + "experiment.retrieval-5.statistics.pkl")
        mapping, panel_data_c = {"0001-1000": 0, "1001-2000": 1, "2001-3000": 2}, {}
        for (index_scope, pcr_number), (errors, _) in records.items():
            values = []
            for index, number in enumerate(errors):
                if number > 0:
                    values += [index] * number
            panel_data_c[(mapping[index_scope], int(pcr_number[1:]))] = mean(values)
        results["c"] = panel_data_c

        panel_data_d = {}
        for (index_scope, pcr_number), (_, numbers) in records.items():
            if pcr_number == "#0":
                values = -ones(shape=(len(numbers),), dtype=float)
                indices = where(numbers > 0)[0]
                values[indices] = log10(numbers[indices].astype(float))
                # noinspection PyUnresolvedReferences
                average = sum(numbers) / 1000.0
                norm_indices = where((numbers <= average * 2) & (numbers >= average / 2))[0]
                more_indices = where(numbers > average * 2)[0]
                less_indices = where(numbers < average / 2)[0]
                panel_data_d[mapping[index_scope]] = (values, norm_indices, more_indices, less_indices)
        results["d"] = panel_data_d

        records = load_data(file_path=raw_path + "experiment.retrieval-7.performances.pkl")
        panel_data_1, panel_data_2 = zeros(shape=(3, 6, 11)), zeros(shape=(3, 6, 11))
        for index, index_scope in enumerate(["0001-1000", "1001-2000", "2001-3000"]):
            for pcr_number in [0, 1, 2, 3, 4, 5]:
                for depth in arange(10, 101, 10, dtype=int):
                    average_1, average_2 = 0, 0
                    for batch_index in range(3):
                        results_1, results_2 = records[(index_scope, pcr_number, depth, batch_index + 1)]
                        average_1, average_2 = average_1 + results_1[0], average_2 + results_2[0]
                    panel_data_1[index, pcr_number, depth // 10] = average_1 / 3.0 / 1000.0
                    panel_data_2[index, pcr_number, depth // 10] = average_2 / 3.0 / 1000.0
        results["e"], results["f"], results["g"] = panel_data_1[0], panel_data_1[1], panel_data_1[2]
        results["h"], results["i"], results["j"] = panel_data_2[0], panel_data_2[1], panel_data_2[2]

        save_data(file_path=pkg_path + "main-03.pkl", information=results)


def main_f04():
    """
    Create plotting data for Figure 4 in the main text.

    TITLE: Comparable evaluation of retrieval runtime.
    """
    if not path.exists(pkg_path + "main-04.pkl"):
        results = {}

        records_1 = load_data(file_path=raw_path + "experiment.retrieval-7.performances.pkl")
        panel_data_a = zeros(shape=(3, 2, 180))
        for index, index_scope in enumerate(["0001-1000", "1001-2000", "2001-3000"]):
            current = 0
            for pcr_number in [0, 1, 2, 3, 4, 5]:
                for depth in arange(10, 101, 10, dtype=int):
                    for batch_index in range(3):
                        results_1, results_2 = records_1[(index_scope, pcr_number, depth, batch_index + 1)]
                        panel_data_a[index, 0, current] = results_1[1]
                        panel_data_a[index, 1, current] = results_2[1]
                        current += 1
        results["a"] = panel_data_a

        records_2 = load_data(file_path=raw_path + "simulation.retrieving-3.comparable.pkl")

        pairs_1, pairs_2 = [[], []], [[], []]
        for index, index_scope in enumerate(["0001-1000", "1001-2000", "2001-3000"]):
            for pcr_number in [0, 1, 2, 3, 4, 5]:
                for depth in arange(10, 101, 10, dtype=int):
                    for batch_index in range(3):
                        results_1, results_2 = records_1[(index_scope, pcr_number, depth, batch_index + 1)]
                        pairs_1[0].append(results_1[1])
                        pairs_1[1].append(results_2[1])
        for reads in arange(10, 101, 10, dtype=int):
            for errors in arange(1, 7, dtype=int):
                cov_data, our_data = records_2[(reads, errors)]
                pairs_2[0] += cov_data[1]
                pairs_2[1] += our_data[1]
        pairs_1, pairs_2 = array(pairs_1), array(pairs_2)
        grid_x, grid_y = linspace(0, 2000, 100), linspace(0, 2, 100)
        mesh_x, mesh_y = meshgrid(grid_x, grid_y)
        distribution_1 = gaussian_kde(pairs_1)(vstack([mesh_x.ravel(), mesh_y.ravel()])).reshape(100, 100)
        distribution_1 /= sum(distribution_1)
        distribution_2 = gaussian_kde(pairs_2)(vstack([mesh_x.ravel(), mesh_y.ravel()])).reshape(100, 100)
        distribution_2 /= sum(distribution_2)
        distance_1 = wasserstein_distance_nd(distribution_1, distribution_2)
        if min(distribution_1) == 0.0:
            distribution_1 += 1e-10
            distribution_1 /= sum(distribution_1)
        if min(distribution_2) == 0.0:
            distribution_2 += 1e-10
            distribution_2 /= sum(distribution_2)
        distance_2 = sum(distribution_1 * log(distribution_1 / distribution_2))

        grid_1_x = linspace(0, max(pairs_1[0]), 100)
        grid_1_y = linspace(0, max(pairs_1[1]), 100)
        mesh_1_x, mesh_1_y = meshgrid(grid_1_x, grid_1_y)
        distribution_1 = gaussian_kde(pairs_1)(vstack([mesh_1_x.ravel(), mesh_1_y.ravel()])).reshape(mesh_1_y.shape)
        distribution_1 /= sum(distribution_1)
        distribution_data_1 = (mesh_1_x, mesh_1_y, distribution_1)

        grid_2_x = linspace(0, max(pairs_2[0]), 100)
        grid_2_y = linspace(0, max(pairs_2[1]), 100)
        mesh_2_x, mesh_2_y = meshgrid(grid_2_x, grid_2_y)
        distribution_2 = gaussian_kde(pairs_2)(vstack([mesh_2_x.ravel(), mesh_2_y.ravel()])).reshape(mesh_2_y.shape)
        distribution_2 /= sum(distribution_2)
        distribution_data_2 = (mesh_2_x, mesh_2_y, distribution_2)

        results["c"] = (distance_1, distance_2, distribution_data_1, distribution_data_2)

        records_3 = load_data(file_path=raw_path + "simulation.retrieving-4.scalable.pkl")

        our_data_1, cov_data_1 = [], []
        for diversity in [10, 100, 1000, 10000, 100000]:
            our_data_1.append(mean(records_3[(diversity, 1)][1]))
            cov_data_1.append(mean(records_3[(diversity, 1)][2]))

        values_1 = (log10([10, 100, 1000, 10000, 100000]), log10(our_data_1), log10(cov_data_1))

        # noinspection PyTupleAssignmentBalance
        fitted_params_our_1, _ = curve_fit(f=estimate_runtime_1, xdata=values_1[0], ydata=values_1[1])
        obtained_values = estimate_runtime_1(values_1[0], *fitted_params_our_1)
        residual_sum_of_squares = sum((values_1[1] - obtained_values) ** 2)
        total_sum_of_squares = sum((values_1[1] - mean(values_1[1])) ** 2)
        fit_info_our_1 = 1 - (residual_sum_of_squares / total_sum_of_squares)

        # noinspection PyTupleAssignmentBalance
        fitted_params_cov_1, _ = curve_fit(f=estimate_runtime_2, xdata=values_1[0], ydata=values_1[2])
        obtained_values = estimate_runtime_2(values_1[0], *fitted_params_cov_1)
        residual_sum_of_squares = sum((values_1[2] - obtained_values) ** 2)
        total_sum_of_squares = sum((values_1[2] - mean(values_1[2])) ** 2)
        fit_info_cov_1 = 1 - (residual_sum_of_squares / total_sum_of_squares)

        point_x = linspace(1.0, 5.0, 101)
        point_our_y = estimate_runtime_1(point_x, *fitted_params_our_1)
        point_cov_y = estimate_runtime_2(point_x, *fitted_params_cov_1)

        results["d"] = (values_1, (point_x, point_our_y, point_cov_y), fit_info_our_1, fit_info_cov_1)

        our_data_6, cov_data_6 = [], []
        for diversity in [10, 100, 1000, 10000, 100000]:
            our_data_6.append(mean(records_3[(diversity, 6)][1]))
            cov_data_6.append(mean(records_3[(diversity, 6)][2]))

        values_6 = (log10([10, 100, 1000, 10000, 100000]), log10(our_data_6), log10(cov_data_6))

        # noinspection PyTupleAssignmentBalance
        fitted_params_our_6, _ = curve_fit(f=estimate_runtime_1, xdata=values_6[0], ydata=values_6[1])
        obtained_values = estimate_runtime_1(values_6[0], *fitted_params_our_6)
        residual_sum_of_squares = sum((values_6[1] - obtained_values) ** 2)
        total_sum_of_squares = sum((values_6[1] - mean(values_6[1])) ** 2)
        fit_info_our_6 = 1 - (residual_sum_of_squares / total_sum_of_squares)

        # noinspection PyTupleAssignmentBalance
        fitted_params_cov_6, _ = curve_fit(f=estimate_runtime_2, xdata=values_6[0], ydata=values_6[2])
        obtained_values = estimate_runtime_2(values_6[0], *fitted_params_cov_6)
        residual_sum_of_squares = sum((values_6[2] - obtained_values) ** 2)
        total_sum_of_squares = sum((values_6[2] - mean(values_6[2])) ** 2)
        fit_info_cov_6 = 1 - (residual_sum_of_squares / total_sum_of_squares)

        point_x = linspace(1.0, 5.0, 101)
        point_our_y = estimate_runtime_1(point_x, *fitted_params_our_6)
        point_cov_y = estimate_runtime_2(point_x, *fitted_params_cov_6)
        results["e"] = (values_6, (point_x, point_our_y, point_cov_y), fit_info_our_6, fit_info_cov_6)

        amount_data, fitted_data_1, fitted_data_6 = linspace(2, 6, 101), [], []
        for value in amount_data:
            bit_number = 8.0 * (1024.0 ** value)
            nucleotide_number = bit_number / 0.724
            sequence_number = ceil(nucleotide_number / 200.0)
            our_value_1 = estimate_runtime_1(log10(sequence_number), *fitted_params_our_1)
            cov_value_1 = estimate_runtime_2(log10(sequence_number), *fitted_params_cov_1)
            fitted_data_1.append(cov_value_1 - our_value_1)
            our_value_6 = estimate_runtime_1(log10(sequence_number), *fitted_params_our_6)
            cov_value_6 = estimate_runtime_2(log10(sequence_number), *fitted_params_cov_6)
            fitted_data_6.append(cov_value_6 - our_value_6)
        fitted_data_1, fitted_data_6 = array(fitted_data_1), array(fitted_data_6)

        results["f"] = (amount_data, fitted_data_1, fitted_data_6)

        save_data(file_path=pkg_path + "main-04.pkl", information=results)


def main_f05():
    """
    Create plotting data for Figure 5 in the main text.

    TITLE: Approximation of data retrieval efficiency.
    """
    if not path.exists(pkg_path + "main-05.pkl"):
        results = {}

        records_1 = load_data(file_path=raw_path + "simulation.correction-4.readouts.pkl")
        panel_data_b = [[1.0 / (4.0 * 60.0), 3.0, 450.0],
                        [1e6 * 200 / mean(records_1[1][errors]) for errors in arange(1, 9)]]
        results["b"] = panel_data_b

        records_2 = load_data(file_path=raw_path + "simulation.conversion-4.speeds.pkl")
        speed_groups = []
        for index, key in enumerate(["10", "100", "1000", "10000", "100000", "1000000"]):
            speeds = (150.0 * int(key) / array(records_2[key]))
            speed_groups.append(speeds)

        results["c"] = speed_groups

        labels, values = zeros(shape=(6, 100)), zeros(shape=(6, 100))
        for index, key in enumerate(["10", "100", "1000", "10000", "100000", "1000000"]):
            labels[index] = int(key)
            values[index] = records_2[key]
        d_param_a, d_param_b = polyfit(log10(labels.reshape(-1)), log10(values.reshape(-1)), deg=1)

        fitted_matrix, fitted_param_group = zeros(shape=(8, 50, 8)), {}
        for diversity, reads, errors, value, _ in load_data(file_path=raw_path + "simulation.retrieving-2.rates.pkl"):
            fitted_matrix[int(log10(diversity)) - 1, reads - 1, errors - 1] = value
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
            fitted_param_group[error_index] = fitted_params

        # based on the speed information collected above.
        correcting_speeds = array([1e6 / mean(records_1[1][errors]) for errors in arange(1, 9)])

        discussed_error_index = 0
        fitted_params = fitted_param_group[discussed_error_index]
        correcting_speed = correcting_speeds[discussed_error_index]

        x_values = linspace(1, 3, 201)
        y_values = linspace(1, 6, 501)
        matrix = ones(shape=(len(x_values), len(y_values)))
        for x_index, x_value in enumerate(x_values):
            for y_index, y_value in enumerate(y_values):
                required_throughput = 8 * (1024 ** x_value)
                bit_number = 8 * (1024 ** y_value)
                # 0.724 is the practical information density, calculating when the special coding digraph is generated.
                # it calculated when sequence length is 200 nucleotides.
                # check https://github.com/HaolingZHANG/RealTimeRetrieval/blob/main/works/log/step-1.2.txt for details.
                nucleotide_number = bit_number / 0.724
                sequence_number = ceil(nucleotide_number / 200.0)
                error_rate = 1.0 / sequence_number
                required_copy_number = estimate_sequence_copies(error_rate, *fitted_params)
                reads_number = sequence_number * required_copy_number
                expected_time = bit_number / required_throughput
                decoding_speed = sequence_number / (10 ** (d_param_a * log10(sequence_number) + d_param_b))

                if reads_number / correcting_speed + sequence_number / decoding_speed < expected_time * 0.95:
                    matrix[x_index, y_index] = -1  # will mask as none later.
                else:
                    threads = 1
                    while True:
                        correcting_time = reads_number / (threads * correcting_speed)
                        decoding_time = sequence_number / (threads * decoding_speed)
                        if correcting_time + decoding_time <= expected_time:
                            break
                        threads += 1
                    matrix[x_index, y_index] = threads

        # maximum 1-terabyte hard drive
        storage_capacity, bound_y_value = 8 * (1024 ** 4), None
        for y_index, y_value in enumerate(y_values):
            bit_number = 8 * (1024 ** y_value)
            # 0.724 is the practical information density, calculating when the special coding digraph is generated.
            # it calculated when sequence length is 200 nucleotides.
            # check https://github.com/HaolingZHANG/RealTimeRetrieval/blob/main/works/log/step-1.2.txt for details.
            nucleotide_number = bit_number / 0.724
            sequence_number = ceil(nucleotide_number / 200.0)
            error_rate = 1.0 / sequence_number
            required_copy_number = estimate_sequence_copies(error_rate, *fitted_params)
            if required_copy_number * sequence_number * 200 * 8 <= storage_capacity:
                bound_y_value = y_value
            else:
                break

        # maximum 16 threads
        # noinspection PyTypeChecker
        new_y_values = linspace(1, bound_y_value, 101)
        bound_x_01_values, bound_x_16_values = [], []
        # noinspection PyTypeChecker
        for y_index, y_value in enumerate(new_y_values):
            bit_number = 8 * (1024 ** y_value)
            # 0.724 is the practical information density, calculating when the special coding digraph is generated.
            # it calculated when sequence length is 200 nucleotides.
            # check https://github.com/HaolingZHANG/RealTimeRetrieval/blob/main/works/log/step-1.2.txt for details.
            nucleotide_number = bit_number / 0.724
            sequence_number = ceil(nucleotide_number / 200.0)
            error_rate = 1.0 / sequence_number
            required_copy_number = estimate_sequence_copies(error_rate, *fitted_params)
            reads_number = sequence_number * required_copy_number
            decoding_speed = sequence_number / (10 ** (d_param_a * log10(sequence_number) + d_param_b))
            runtime_01 = reads_number / correcting_speed + sequence_number / decoding_speed
            runtime_16 = reads_number / (16.0 * correcting_speed) + sequence_number / (16.0 * decoding_speed)
            throughput_01, throughput_16 = bit_number / runtime_01, bit_number / runtime_16
            bound_x_01_values.append(throughput_01)
            bound_x_16_values.append(throughput_16)
        bound_x_01_values, bound_x_16_values = array(bound_x_01_values), array(bound_x_16_values)
        plotted_x_values = (bound_x_01_values.tolist()
                            + [bound_x_01_values[-1], bound_x_16_values[-1]]
                            + bound_x_16_values[::-1].tolist())
        plotted_x_values = log10(array(plotted_x_values) / 8.0) / log10(1024.0)
        # noinspection PyUnresolvedReferences
        plotted_y_values = array(new_y_values.tolist() + [bound_y_value, bound_y_value] + new_y_values[::-1].tolist())
        range_info = (bound_x_01_values[0], bound_x_01_values[-1], bound_x_16_values[-1], bound_x_16_values[0])
        points = [(log10(bound_x_01_values[0] / 8.0) / log10(1024.0), 1.0),
                  (log10(bound_x_01_values[-1] / 8.0) / log10(1024.0), bound_y_value),
                  (log10(bound_x_16_values[-1] / 8.0) / log10(1024.0), bound_y_value),
                  (log10(bound_x_16_values[0] / 8.0) / log10(1024.0), 1.0)]

        results["d"] = ((x_values, y_values, matrix), (plotted_x_values, plotted_y_values), (range_info, points))

        ranges = []
        for discussed_error_index in range(8):
            fitted_params = fitted_param_group[discussed_error_index]
            correcting_speed = correcting_speeds[discussed_error_index]
            # maximum 1-terabyte hard drive
            storage_capacity, bound_y_value = 8 * (1024 ** 4), None
            for y_index, y_value in enumerate(y_values):
                bit_number = 8 * (1024 ** y_value)
                # 0.724 is the practical information density, calculating when the special coding digraph is generated.
                # it calculated when sequence length is 200 nucleotides.
                # check https://github.com/HaolingZHANG/RealTimeRetrieval/blob/main/works/log/step-1.2.txt for details.
                nucleotide_number = bit_number / 0.724
                sequence_number = ceil(nucleotide_number / 200.0)
                error_rate = 1.0 / sequence_number
                required_copy_number = estimate_sequence_copies(error_rate, *fitted_params)
                if required_copy_number * sequence_number * 200 * 8 <= storage_capacity:
                    bound_y_value = y_value
                else:
                    break

            # use 16 threads
            bound_x_values = []
            for y_value in [bound_y_value, 1]:
                bit_number = 8 * (1024 ** y_value)
                # 0.724 is the practical information density, calculating when the special coding digraph is generated.
                # it calculated when sequence length is 200 nucleotides.
                # check https://github.com/HaolingZHANG/RealTimeRetrieval/blob/main/works/log/step-1.2.txt for details.
                nucleotide_number = bit_number / 0.724
                sequence_number = ceil(nucleotide_number / 200.0)
                error_rate = 1.0 / sequence_number
                required_copy_number = estimate_sequence_copies(error_rate, *fitted_params)
                reads_number = sequence_number * required_copy_number
                decoding_speed = sequence_number / (10 ** (d_param_a * log10(sequence_number) + d_param_b))
                runtime = reads_number / (16.0 * correcting_speed) + sequence_number / (16.0 * decoding_speed)
                bound_x_values.append(bit_number / runtime)

            ranges.append(bound_x_values)
        results["e"] = array(ranges)

        save_data(file_path=pkg_path + "main-05.pkl", information=results)


def main_f06():
    """
    Create plotting data for Figure 6 in the main text.

    TITLE: Demonstration of real-time non-blocking data retrieval.
    """
    if not path.exists(pkg_path + "main-06.pkl"):
        results = {"a": load_data(raw_path + "experiment.case-2.figures.npy")}

        address_data, collection = load_data(file_path=raw_path + "experiment.case-6.raw-selected-addresses.pkl"), {}
        for frame_index in range(16):
            for raw_index in range(10):
                forward_address, reverse_address = address_data[(frame_index, raw_index)]
                difference = distance(forward_address, reverse_address)
                collection[forward_address] = [[], [difference] * 159 * 2]
                collection[forward_address] = [[], [difference] * 159 * 2]
        for frame_index in range(16):
            for raw_index in range(10):
                forward_address, reverse_address = address_data[(frame_index, raw_index)]
                for observed_address in collection.keys():
                    if observed_address not in [forward_address, reverse_address]:
                        difference_1 = distance(observed_address, forward_address)
                        difference_2 = distance(observed_address, reverse_address)
                        collection[observed_address][0] += [difference_1, difference_2]
        x_labels, y_counts = arange(61), zeros(shape=(2, 61), dtype=float)
        for _, values in collection.items():
            for value in values:
                y_counts[0, value[0]] += 1
                y_counts[1, value[1]] += 1
        y_counts[0] /= sum(y_counts[0])
        y_counts[1] /= sum(y_counts[1])
        results["b"] = (x_labels, y_counts)

        records = load_data(file_path=raw_path + "supplement-experiment.retrieving-2.error-profile.pkl")
        error_counts, location_counts = records

        rates = []
        for key, values in error_counts.items():
            values = array(values)
            rates.append(mean(sum(values[:, :3], axis=1) / values[:, 3]))
        x_values = linspace(min(rates), max(rates), 100)
        y_values = gaussian_kde(rates)(x_values)
        y_values /= sum(y_values)
        average_value = mean(rates)

        results["c"] = (x_values, y_values, average_value)

        rates = []
        for label, values in error_counts.items():
            values = array(values)
            # noinspection PyTypeChecker
            rates.append([mean(values[:, 0] / values[:, 3]),
                          mean(values[:, 1] / values[:, 3]),
                          mean(values[:, 2] / values[:, 3])])
        rates = array(rates)
        # noinspection PyUnresolvedReferences
        results["d"] = mean(rates, axis=0) / sum(mean(rates, axis=0))

        s_ranges, a_ranges, p_ranges, b_ranges, start = [], [], [], [], 0
        lengths = {"r": 200, "s": 10, "a": 60, "p": 110}
        for segment_type in ["r", "s", "a"] + (["s", "p"] * 20) + ["s", "a", "s", "r"]:
            stop = start + lengths[segment_type]
            if segment_type == "s":
                s_ranges.append([start, stop])
            elif segment_type == "a":
                a_ranges.append([start, stop])
            elif segment_type == "p":
                p_ranges.append([start, stop])
            else:
                b_ranges.append([start, stop])
            start = stop

        rates = []
        for label, values in location_counts.items():
            temp_1 = [mean(values[0, start:stop] / values[1, start:stop]) for start, stop in s_ranges]
            temp_2 = [mean(values[0, start:stop] / values[1, start:stop]) for start, stop in a_ranges]
            temp_3 = [mean(values[0, start:stop] / values[1, start:stop]) for start, stop in p_ranges]
            # noinspection PyTypeChecker
            rates.append([mean(temp_1), mean(temp_2), mean(temp_3)])
        results["e"] = array(rates)

        child_paths = ["250F501329017/statistics.csv", "250F501329018/statistics.csv",
                       "250F501329019/statistics.csv", "250F50132901A/statistics.csv",
                       "250F50132901B/statistics.csv"]

        processes = {}
        for tried_index, child_path in enumerate(child_paths):
            with open(exp_path + "3/" + child_path) as file:
                last, collection = None, []
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
                        retrieval_rate = int(info[3]) / 3200.0
                        collection.append([runtime + processing_time, retrieval_rate])
                processes[tried_index + 1] = array(collection)

        bounds = {"50%": [[], []], "90%": [[], []], "99%": [[], []], "100%": [[], []]}
        for tried_index, child_path in enumerate(child_paths):
            with open(exp_path + "3/" + child_path) as file:
                last, collection = None, []
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
                        retrieval_rate = int(info[3]) / 3200.0
                        depth = (100 * index) / 160.0
                        if retrieval_rate >= 0.500 and len(bounds["50%"][0]) != (tried_index + 1):
                            bounds["50%"][0].append(runtime + processing_time)
                            bounds["50%"][1].append(depth)
                        if retrieval_rate >= 0.900 and len(bounds["90%"][0]) != (tried_index + 1):
                            bounds["90%"][0].append(runtime + processing_time)
                            bounds["90%"][1].append(depth)
                        if retrieval_rate >= 0.990 and len(bounds["99%"][0]) != (tried_index + 1):
                            bounds["99%"][0].append(runtime + processing_time)
                            bounds["99%"][1].append(depth)
                        if retrieval_rate == 1.000:
                            bounds["100%"][0].append(runtime + processing_time)
                            bounds["100%"][1].append(depth)
        results["f"] = (processes, bounds)

        save_data(file_path=pkg_path + "main-06.pkl", information=results)


if __name__ == "__main__":
    main_f01()
    main_f02()
    main_f03()
    main_f04()
    main_f05()
    main_f06()
