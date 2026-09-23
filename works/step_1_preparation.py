from datetime import datetime
from numpy import mean, std
from os.path import exists
from prettytable import PrettyTable

from sbw import adjust_motif_format, create_coding_digraph
from sbw import create_start_finder, create_palindrome_map, create_coexisting_map, calculate_practical_density
from sbw import save_data, Monitor

from works import raw_path


def generate_general_cases(window_length: int):
    if not exists(raw_path + "generation-1.general-cases.pkl"):
        print("Generate general coding digraphs:")
        cases, table = {}, PrettyTable(["homopolymer run-length", "GC content", "information density", "runtime"])
        monitor, process, miss = Monitor(), 0, 0
        for homopolymer in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for gc_bias in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                gc_range = [0.5 - gc_bias, 0.5 + gc_bias]
                label_1, label_2 = str(homopolymer), "[" + ("%.1f" % gc_range[0]) + "," + ("%.1f" % gc_range[1]) + "]"
                try:
                    last_time = datetime.now()
                    accessor, vertices = create_coding_digraph(window_length=window_length,
                                                               run_length=homopolymer, gc_range=gc_range)
                    runtime = (datetime.now() - last_time).total_seconds()
                    densities = calculate_practical_density(accessor=accessor, length=200)
                    cases[(homopolymer, gc_bias)] = (accessor, densities, runtime)
                    # noinspection PyStringFormat
                    table.add_row([label_1, label_2,
                                   "%.3f ± %.3f" % (mean(densities), std(densities)), "%.3f" % runtime])
                except RuntimeError:
                    miss += 1

                monitor(process + 1, 60, extra={"miss": miss})
                process += 1

        print()
        save_data(file_path=raw_path + "generation-1.general-cases.pkl", information=cases)
        print("The practical information density of each generated coding digraph are:")
        print(table)
        print()


def generate_special_case(window_length: int):
    if not exists(raw_path + "generation-2.special-case.pkl"):
        # extremely-short single-nucleotide repeat permission for both DNA synthesis and sequencing.
        # |cite| Roy Shafir et al. (2021) IEEE Information Theory Workshop
        # |cite| André E Minoche et al. (2011) Genome biology
        homopolymer = 2
        # balanced GC content for higher PCR success rate.
        # |cite| Yair Benita et al. (2003) Nucleic Acids Research
        gc_range = [0.4, 0.6]
        # potential high-error motifs in high-throughput sequencers: GGC and GGT.
        # |cite| Kensuke Nakamura et al. (2021) Nucleic Acids Research
        # |cite| Frazer Meacham et al. (2021) BMC Bioinformatics
        # potential high-error motifs in single-molecule sequencers: AG, GA, CT and TC.
        # |cite| Ian M. Derrington et al. (2010) Proceedings of the National Academy of Sciences
        # |cite| Ryan R. Wick et al. (2019) Genome biology
        motifs = adjust_motif_format(motif_set=[["GGC", "GGT"], ["AG", "GA", "CT", "TC"]])
        # minimum Levenshtein distance between virtual vertices.
        minimum_difference = 4

        accessor, vertices = create_coding_digraph(window_length=window_length,
                                                   run_length=homopolymer, gc_range=gc_range, undesired_motifs=motifs,
                                                   strongly_connected=True, verbose=True)
        start_finder = create_start_finder(accessor=accessor, vertices=vertices, window_length=window_length,
                                           contain_error=True, verbose=True)
        coexisting_map = create_coexisting_map(window_length=window_length, minimum_difference=minimum_difference,
                                               vertices=vertices, verbose=True)
        palindrome_map = create_palindrome_map(window_length=window_length,
                                               vertices=vertices, verbose=True)

        densities = calculate_practical_density(accessor=accessor, length=200)

        # noinspection PyStringFormat
        print("Information density: %.3f ± %.3f.\n" % (mean(densities), std(densities)))

        information = (accessor, start_finder, coexisting_map, palindrome_map)

        save_data(file_path=raw_path + "generation-2.special-case.pkl", information=information)


if __name__ == "__main__":
    generate_general_cases(window_length=10)
    generate_special_case(window_length=10)
