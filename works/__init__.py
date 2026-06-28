from os import path

from works.code_converting import DNAFountain, HEDGES, calculate_largest_eigenvalue
from works.code_converting import get_complete_accessor, accessor_to_matrix, matrix_to_accessor
from works.code_correcting import practice, evaluate_single_repair, evaluate_single_repeats
from works.code_correcting import OuterCode, evaluate_astar_search
from works.code_retrieving import batch_generate, generate_retrieval_data, conventional_retrieval_process, levenshtein
from works.code_retrieving import design_pcr_primer, obtain_raw_sequences, calculate_retrieval_quality, obtain_samples
from works.code_comparable import produce, retrieve, test_chn, test_symmetry
from works.code_estimating import estimate_sequence_copies, estimate_file_size, estimate_throughput
from works.code_estimating import estimate_runtime_1, estimate_runtime_2

work_path = path.dirname(path.abspath(__file__)).replace("\\", "/")

raw_path = work_path + "/raw/"
exp_path = work_path + "/exp/"
tmp_path = work_path + "/tmp/"
pkg_path = work_path + "/pkg/"
fig_path = work_path + "/fig/"
tab_path = work_path + "/tab/"
