from numpy import ndarray, log, log10, ceil
from typing import Union, Optional


def estimate_sequence_copies(error_rate: float,
                             param_1: float,
                             param_2: float) \
        -> float:
    """
    Estimate the number of copies needed for the expected error rate.

    :param error_rate: expected file-scale error rate.
    :type error_rate: float

    :param param_1: hyperparameter 1.
    :type param_1: float

    :param param_2: hyperparameter 2.
    :type param_2: float

    :return: estimated sequence copies.
    :rtype: float
    """
    return (-log10(error_rate) / param_1) ** (1.0 / param_2)


def estimate_file_size(diversity: int,
                       param_1: float,
                       param_2: float) \
        -> float:
    """
    Estimate the produced file size based on the sequence diversity (sequence length = 200nt).

    :param diversity: sequence diversity.
    :type diversity: int

    :param param_1: hyperparameter 1.
    :type param_1: float

    :param param_2: hyperparameter 2.
    :type param_2: float

    :return: estimated file size.
    :rtype: float
    """
    return (diversity / param_1) ** (1.0 / param_2)


def estimate_throughput(reads_number: float,
                        copy_params: list,
                        size_params: list,
                        correcting_speed: float,
                        decoding_speed: float,
                        thread_number: int,
                        referenced_diversity: Optional[int] = None) \
        -> float:
    """
    Estimate the throughput based on the given reads number and other hyperparameters.

    :param reads_number: given sequence copies.
    :type reads_number: int

    :param copy_params: parameters of the fitted curve in the copy issue.
    :type copy_params: list

    :param size_params: parameters of the fitted curve in the size issue.
    :type size_params: list

    :param correcting_speed: average sequence correcting speed.
    :type correcting_speed: float

    :param decoding_speed: average sequence decoding speed.
    :type decoding_speed: float

    :param thread_number: given thread number.
    :type thread_number: int

    :param referenced_diversity: referenced diversity if available (for accelerating iteration time).
    :type referenced_diversity: int

    :return: throughput.
    :rtype: float
    """
    diversity, success_flag = referenced_diversity if referenced_diversity is not None else 1, False
    while True:
        copies = estimate_sequence_copies(1.0 / diversity, *copy_params)
        if diversity * copies > reads_number:
            if success_flag:
                correcting_time = ceil(reads_number / thread_number) / correcting_speed
                decoding_time = ceil(diversity / thread_number) / decoding_speed
                file_size = estimate_file_size(diversity=diversity, *size_params)
                return file_size / (correcting_time + decoding_time)
            else:
                raise ValueError("Referenced diversity is set too large!")

        success_flag = True
        diversity += 1


def estimate_runtime_1(diversity: Union[ndarray, float],
                       param_1: float,
                       param_2: float,
                       param_3: float)\
        -> Union[ndarray, float]:
    """
    Estimate the runtime based on the sequence diversity for the SPIDER-WEB framework.

    :param diversity: sequence diversity.
    :type diversity: Union[ndarray, float]

    :param param_1: fitted parameter 1.
    :type param_1: float

    :param param_2: fitted parameter 2.
    :type param_2: float

    :param param_3: fitted parameter 3.
    :type param_3: float

    :return: estimated runtime.
    :rtype: Union[ndarray, float]
    """
    return param_1 * diversity * (log(diversity) ** param_2) + param_3


def estimate_runtime_2(diversity: Union[ndarray, float],
                       param_1: float,
                       param_2: float,
                       param_3: float)\
        -> Union[ndarray, float]:
    """
    Estimate the runtime based on the sequence diversity for the conventional pipeline.

    :param diversity: sequence diversity.
    :type diversity: Union[ndarray, float]

    :param param_1: fitted parameter 1.
    :type param_1: float

    :param param_2: fitted parameter 2.
    :type param_2: float

    :param param_3: fitted parameter 3.
    :type param_3: float

    :return: estimated runtime.
    :rtype: Union[ndarray, float]
    """
    return param_1 * (diversity ** 2) + param_2 * diversity + param_3
