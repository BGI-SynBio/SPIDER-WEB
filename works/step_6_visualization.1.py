from logging import getLogger, CRITICAL
from matplotlib import pyplot, patches, rcParams
from numpy import array, linspace, arange, random, deg2rad, mean, min, max, sin, cos, log10, nan, where
from warnings import filterwarnings

from sbw import load_data

from works import pkg_path

filterwarnings("ignore")

getLogger("matplotlib").setLevel(CRITICAL)

rcParams["font.family"] = "Arial"
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "Linux Libertine"
rcParams["mathtext.cal"] = "Lucida Calligraphy"
rcParams["mathtext.it"] = "Linux Libertine:italic"
rcParams["mathtext.bf"] = "Linux Libertine:bold"


def main_f01():
    """
    Create Figure 1 in the main text.

    TITLE: Overview of this work.
    """
    shown_data = load_data(file_path=pkg_path + "main-01.pkl")

    figure = pyplot.figure(figsize=(10, 11), tight_layout=True)
    pyplot.subplots_adjust(left=0.10, right=0.90, top=1.00, bottom=0.10)
    grid = pyplot.GridSpec(8, 1)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[:6, 0])

    # panel a -> build by BioRender

    # panel b
    center_b = 3.15
    ranges, width = [], (0.50 - 2.5 * 0.04) / 3.0
    for index in range(3):
        ranges.append([0.52 + (width + 0.04) * index, 0.52 + (width + 0.04) * index + width])
    pyplot.fill_between([ranges[0][0], ranges[2][1]], center_b - 0.37, center_b - 0.34, fc="#BAD5D8", lw=0, zorder=2)
    labels = ["compatible k-mer selecting", "de Bruijn graph connecting", "bit-nucleotide binding"]
    for one_range, label in zip(ranges, labels):
        pyplot.text((one_range[0] + one_range[1]) / 2.0, center_b + 0.35, label, va="center", ha="center", fontsize=8)
    random.seed(0)
    x_locations = linspace(ranges[0][0], ranges[0][1], 7)
    y_locations = linspace(center_b - 0.295, center_b + 0.295, 10)
    for x_former, x_latter in zip(x_locations[:-1], x_locations[1:]):
        for y_former, y_latter in zip(y_locations[:-1], y_locations[1:]):
            if random.random() < 0.6:
                # noinspection bad-argument-type
                pyplot.fill_between([x_former + 0.02, x_latter - 0.02], y_former + 0.05, y_latter - 0.05,
                                    ec="k", fc="w", lw=0.5)
            else:
                # noinspection bad-argument-type
                pyplot.fill_between([x_former + 0.02, x_latter - 0.02], y_former + 0.05, y_latter - 0.05,
                                    ec="k", fc="gray", lw=0.5)
    random.seed()
    pyplot.annotate("", xy=(ranges[0][1], center_b), xytext=(ranges[1][0], center_b), zorder=1,
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=3, shrinkB=3, lw=0.75))
    color_groups = [["#FBE5D6", "#F4B183", "#ED7D31", "#A24A0E"], ["#E2F0D9", "#A9D18E", "#70AD47", "#4D7731"]]
    # noinspection SpellCheckingInspection
    segment, x_locations = "ACATATGAGTC", linspace(ranges[1][0], ranges[1][1], 12)
    interval = x_locations[1] - x_locations[0]
    for index, location in enumerate(x_locations[:-1]):
        current = segment[index]
        if 0 < index < 10:
            pyplot.vlines(location + interval / 2, center_b, center_b + 0.19, lw=0.75, ls="--", color="k", zorder=1)
            if current != "T":
                # noinspection bad-argument-type
                pyplot.text(location + interval / 2, center_b + 0.19 - 0.006, current,
                            color="k", va="center", ha="center", fontsize=7, zorder=3)
                # noinspection bad-argument-type
                pyplot.text(location + interval / 2, center_b - 0.006, current,
                            color="k", va="center", ha="center", fontsize=7, zorder=3)
            else:
                # noinspection bad-argument-type
                pyplot.text(location + interval / 2, center_b + 0.19 - 0.006, current,
                            color="w", va="center", ha="center", fontsize=7, zorder=3)
                # noinspection bad-argument-type
                pyplot.text(location + interval / 2, center_b - 0.006, current,
                            color="w", va="center", ha="center", fontsize=7, zorder=3)
            pyplot.fill_between([location, location + interval], center_b + 0.19 - 0.03, center_b + 0.19 + 0.03,
                                fc=color_groups[0][{"A": 0, "C": 1, "G": 2, "T": 3}[current]], lw=0, zorder=2)
            pyplot.fill_between([location, location + interval], center_b - 0.03, center_b + 0.03,
                                fc=color_groups[0][{"A": 0, "C": 1, "G": 2, "T": 3}[current]], lw=0, zorder=2)
        elif index == 0:
            if current != "T":
                # noinspection bad-argument-type
                pyplot.text(location + interval / 2, center_b + 0.19 - 0.006, current,
                            color="k", va="center", ha="center", fontsize=7, zorder=3)
            else:
                # noinspection bad-argument-type
                pyplot.text(location + interval / 2, center_b + 0.19 - 0.006, current,
                            color="w", va="center", ha="center", fontsize=7, zorder=3)
            pyplot.fill_between([location, location + interval], center_b + 0.19 - 0.03, center_b + 0.19 + 0.03,
                                fc=color_groups[0][{"A": 0, "C": 1, "G": 2, "T": 3}[current]], lw=0, zorder=2)
        else:
            if current != "T":
                # noinspection bad-argument-type
                pyplot.text(location + interval / 2, center_b - 0.006, current,
                            color="k", va="center", ha="center", fontsize=7, zorder=3)
            else:
                # noinspection bad-argument-type
                pyplot.text(location + interval / 2, center_b - 0.006, current,
                            color="w", va="center", ha="center", fontsize=7, zorder=3)
            pyplot.fill_between([location, location + interval], center_b - 0.03, center_b + 0.03,
                                fc=color_groups[0][{"A": 0, "C": 1, "G": 2, "T": 3}[current]], lw=0, zorder=2)
    pyplot.text(ranges[1][0] + interval * 5, center_b + 0.26, "vertex x", va="center", ha="center", fontsize=8)
    pyplot.fill_between([ranges[1][0], ranges[1][0] + interval * 10], center_b + 0.19 - 0.03, center_b + 0.19 + 0.03,
                        ec="k", fc="none", lw=0.75, zorder=4)
    pyplot.text(ranges[1][0] + interval * 6, center_b - 0.07, "vertex y", va="center", ha="center", fontsize=8)
    pyplot.fill_between([ranges[1][0] + interval, ranges[1][0] + interval * 11], center_b - 0.03, center_b + 0.03,
                        ec="k", fc="none", lw=0.75, zorder=4)
    pyplot.plot([x_locations[-1] - interval / 2.0, x_locations[-1] - interval / 2.0,
                 (ranges[1][0] + ranges[1][1]) / 2.0, (ranges[1][0] + ranges[1][1]) / 2.0],
                [center_b, center_b - 0.12, center_b - 0.12, center_b - 0.19], lw=0.75, ls="--", color="k", zorder=1)
    pyplot.scatter([ranges[1][0] + interval / 2.0, ranges[1][1] - interval / 2.0],
                   [center_b - 0.19, center_b - 0.19], ec="k", fc="w", lw=0.75, s=20, zorder=2)
    pyplot.annotate("", xy=(ranges[1][0] + interval / 2.0, center_b - 0.19),
                    xytext=(ranges[1][1] - interval / 2.0, center_b - 0.19), zorder=1,
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=2, shrinkB=2, lw=0.75))
    pyplot.text((ranges[1][0] + ranges[1][1]) / 2.0, center_b - 0.26, "used nucleotide",
                va="center", ha="center", fontsize=8)
    pyplot.text(ranges[1][0] + interval / 2.0, center_b - 0.26, "x", va="center", ha="center", fontsize=8)
    pyplot.text(ranges[1][1] - interval / 2.0, center_b - 0.26, "y", va="center", ha="center", fontsize=8)
    pyplot.text((ranges[1][0] + ranges[1][1]) / 2.0, center_b - 0.19, "C", color="k",
                va="center", ha="center", fontsize=7, zorder=3)
    pyplot.fill_between([(ranges[1][0] + ranges[1][1]) / 2.0 - interval / 2.0,
                         (ranges[1][0] + ranges[1][1]) / 2.0 + interval / 2.0],
                        center_b - 0.19 - 0.03, center_b - 0.19 + 0.03,
                        ec="k", fc=color_groups[0][1], lw=0.75, zorder=2)
    pyplot.annotate("", xy=(ranges[1][1], center_b), xytext=(ranges[2][0], center_b), zorder=1,
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=3, shrinkB=3, lw=0.75))
    x_locations = linspace(ranges[2][0], ranges[2][1], 12)
    y_locations = linspace(center_b - 0.295, center_b + 0.295, 9)
    pyplot.vlines(x_locations, center_b - 0.295, center_b + 0.295, lw=0.75, color="k", zorder=4)
    pyplot.hlines(y_locations[4], ranges[2][0], ranges[2][1], lw=0.75, color="k", zorder=4)
    for index, bases in enumerate(["ACXX", "AXGX", "AXXT", "XCGX", "XCXT", "XXGT",
                                   "ACGX", "ACXT", "AXGT", "XCGT", "ACGT"]):
        digit = 0
        for location, base in enumerate(bases):
            if base != "X":
                pyplot.fill_between([x_locations[index], x_locations[index + 1]],
                                    y_locations[-location - 1], y_locations[-location - 2],
                                    color=color_groups[0][{"A": 0, "C": 1, "G": 2, "T": 3}[base]], zorder=0, lw=0)
                if base != "T":
                    # noinspection PyTypeChecker
                    pyplot.text((x_locations[index] + x_locations[index + 1]) / 2,
                                (y_locations[-location - 1] + y_locations[-location - 2]) / 2 - 0.005,
                                base, va="center", ha="center", fontsize=7)
                else:
                    # noinspection PyTypeChecker
                    pyplot.text((x_locations[index] + x_locations[index + 1]) / 2,
                                (y_locations[-location - 1] + y_locations[-location - 2]) / 2 - 0.005,
                                base, va="center", ha="center", fontsize=7, color="white")
                pyplot.fill_between([x_locations[index], x_locations[index + 1]],
                                    y_locations[4 - location], y_locations[3 - location],
                                    color=color_groups[1][digit], zorder=0, lw=0)
                if digit < 3:
                    # noinspection PyTypeChecker
                    pyplot.text((x_locations[index] + x_locations[index + 1]) / 2,
                                (y_locations[4 - location] + y_locations[3 - location]) / 2.0 - 0.005,
                                str(digit), va="center", ha="center", fontsize=7, zorder=1)
                else:
                    # noinspection PyTypeChecker
                    pyplot.text((x_locations[index] + x_locations[index + 1]) / 2,
                                (y_locations[4 - location] + y_locations[3 - location]) / 2.0 - 0.005,
                                str(digit), va="center", ha="center", fontsize=7, color="white", zorder=1)
                digit += 1

    # link
    pyplot.plot([0.76, 0.76, 0.59], [2.78, 2.65, 2.65], lw=1.5, color="#BAD5D8", zorder=0)
    pyplot.annotate("", xy=(0.59, 2.65), xytext=(0.59, 2.50),
                    arrowprops=dict(arrowstyle="<|-", color="#BAD5D8", shrinkA=0, shrinkB=0, lw=1.5), zorder=1)
    pyplot.text(0.675, 2.65, "coding digraph", va="center", ha="center",
                bbox=dict(fc="w", ec="#BAD5D8", lw=1.5, boxstyle="round,pad=0.4"), color="#6FA9AF",
                weight="bold", fontsize=9)

    # panel c
    center_c = 2.10
    ranges = [0.52, 1.00]
    pyplot.fill_between([ranges[1] - 0.12, ranges[1]], center_c - 0.38, center_c - 0.35, fc="#BAD5D8", lw=0, zorder=3)
    pyplot.text(ranges[0] + 0.070, center_c + 0.35, "walk on a coding digraph", va="center", ha="center", fontsize=8)
    points = array([[+0.070, +0.010, +0.050, +0.090, +0.130, +0.090, +0.070, +0.110, +0.090, +0.130],
                    [+0.290, +0.145, +0.145, +0.145, +0.145, +0.000, -0.145, -0.145, -0.290, -0.290]])
    width, height = 0.02, 0.05
    pyplot.scatter(points[0] + ranges[0], points[1] + center_c, ec="k", fc="w", lw=0.75, s=20, zorder=2)
    for former, latter, range_info, mark in [(0, 1, "{A|0}", 0), (0, 2, "{C|1}", 0), (0, 3, "{G|2}", 1),
                                             (0, 4, "{T|3}", 0), (3, 5, "{A|0}", 1), (5, 6, "{A|0}", 0),
                                             (5, 7, "{T|1}", 1), (7, 8, "{C|0}", 1), (7, 9, "{G|1}", 0)]:
        point_1 = (points[0, former] + ranges[0], points[1, former] + center_c)
        point_2 = (points[0, latter] + ranges[0], points[1, latter] + center_c)
        connection = None
        if points[0][former] < points[0][latter]:
            connection = "arc3,rad=0.2"
        elif points[0][former] > points[0][latter]:
            connection = "arc3,rad=-0.2"
        if mark:
            pyplot.annotate("", xy=point_1, xytext=point_2, zorder=1,
                            arrowprops=dict(arrowstyle="<|-", color="k", connectionstyle=connection,
                                            shrinkA=2, shrinkB=0, lw=0.75))
            x, y = (point_1[0] + point_2[0]) / 2, (point_1[1] + point_2[1]) / 2 + 0.015
            pyplot.fill_between([x - width / 2, x], y - height / 2.0, y + height / 2.0,
                                fc=color_groups[0][{"A": 0, "C": 1, "G": 2, "T": 3}[range_info[1]]], lw=0, zorder=2)
            if range_info[1] != "T":
                pyplot.text(x - width / 4, y - 0.005, range_info[1], color="k", va="center", ha="center", fontsize=6,
                            zorder=3)
            else:
                pyplot.text(x - width / 4, y - 0.005, range_info[1], color="w", va="center", ha="center", fontsize=6,
                            zorder=3)
            pyplot.fill_between([x, x + width / 2], y - height / 2.0, y + height / 2.0,
                                fc=color_groups[1][int(range_info[3])], lw=0, zorder=2)
            if range_info[3] != "3":
                pyplot.text(x + width / 4, y - 0.005, range_info[3], color="k", va="center", ha="center", fontsize=6,
                            zorder=3)
            else:
                pyplot.text(x + width / 4, y - 0.005, range_info[3], color="w", va="center", ha="center", fontsize=6,
                            zorder=3)
            pyplot.fill_between([x - width / 2, x + width / 2], y - height / 2.0, y + height / 2.0,
                                ec="k", fc="none", lw=0.75, zorder=4)
        else:
            pyplot.annotate("", xy=point_1, xytext=point_2, zorder=1,
                            arrowprops=dict(arrowstyle="<|-", color="#888888", connectionstyle=connection,
                                            shrinkA=2, shrinkB=0, lw=0.75))

    pyplot.text(ranges[0] + 0.20, center_c + 0.35, "out-degree", va="center", ha="center", fontsize=8)
    for location, node_index, label in zip([+0.145, +0.000, -0.145, -0.290], [4, 5, 7, 9], ["4", "1", "2", "2"]):
        pyplot.hlines(location + center_c, points[0, node_index] + ranges[0] + 0.01, ranges[0] + 0.20,
                      color="#666666", lw=0.75, ls="--", zorder=0)
        pyplot.text(ranges[0] + 0.20, location + center_c, label, bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.4"),
                    va="center", ha="center", fontsize=8, zorder=2)

    pyplot.text(ranges[0] + 0.30, center_c + 0.35, "bit number", va="center", ha="center", fontsize=8)
    # noinspection PyTypeChecker
    former_points, average_value = [], mean([+0.145, +0.000, -0.145, -0.290]) + center_c
    for location, node_index, rad, label in zip(average_value + array([0.12, 0.04, -0.04, -0.12]),
                                                [4, 5, 7, 9], [0.10, 0.05, -0.05, -0.10], ["2", "0", "1", "1"]):
        former_points.append([ranges[0] + 0.30, location])
        pyplot.annotate("",
                        xy=(ranges[0] + 0.20, points[1, node_index] + center_c), xytext=(ranges[0] + 0.30, location),
                        arrowprops=dict(arrowstyle="-", color="#666666", connectionstyle="arc3,rad=" + str(rad),
                                        shrinkA=5, shrinkB=5, lw=0.75, ls="--"), zorder=1)
        pyplot.text(ranges[0] + 0.30, location, label, bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.4"),
                    va="center", ha="center", fontsize=8, zorder=2)

    x_locations = linspace(ranges[0] + 0.42 - 0.05, ranges[0] + 0.42 + 0.05, 6)
    y_locations = linspace(center_c - 0.22, center_c + 0.27, 4)[array([3, 1, 0])]
    pyplot.text(ranges[0] + 0.42, center_c + 0.35, "binary message", va="center", ha="center", fontsize=8)
    for y_location, range_info in zip(y_locations, ["[1010]", "[2010]", "[GATC]"]):
        for x_location, data in zip(x_locations, range_info):
            pyplot.text(x_location, y_location, data, va="center", ha="center", fontsize=8)
    pyplot.hlines(center_c + 0.23, x_locations[1] - 0.004, x_locations[2] + 0.004, lw=1.5, color="k")
    pyplot.scatter([(x_locations[2] + x_locations[3]) / 2.0], center_c + 0.23, s=3, color="k", zorder=2)
    pyplot.hlines(center_c + 0.23, x_locations[3] - 0.004, x_locations[3] + 0.004, lw=1.5, color="k")
    pyplot.hlines(center_c + 0.23, x_locations[4] - 0.004, x_locations[4] + 0.004, lw=1.5, color="k")
    for x in x_locations[1:-1]:
        pyplot.annotate("", xy=(x, y_locations[1] - 0.05), xytext=(x, y_locations[2] + 0.06), zorder=1,
                        arrowprops=dict(arrowstyle="-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(ranges[0] + 0.42, center_c - 0.30, "data coding DNA sequence", va="center", ha="center", fontsize=8)
    sub_x_locations = array([(x_locations[1] + x_locations[2]) / 2.0, (x_locations[2] + x_locations[3]) / 2.0,
                             x_locations[3], x_locations[4]])
    sub_y_locations = linspace(center_c + 0.23, center_c, 6)[1:-1]
    latter_points = []
    for x, y in zip(sub_x_locations, sub_y_locations):
        latter_points.append([x, y])
        pyplot.vlines(x, y, center_c + 0.23, lw=0.75, color="k")
    pyplot.annotate("", xy=(sub_x_locations[0], sub_y_locations[0]), xytext=(x_locations[1], center_c),
                    arrowprops=dict(arrowstyle="-", color="k", connectionstyle="arc3,rad=-0.10",
                                    shrinkA=0, shrinkB=1, lw=0.75), zorder=2)
    pyplot.annotate("", xy=(sub_x_locations[1], sub_y_locations[1]), xytext=(x_locations[2], center_c),
                    arrowprops=dict(arrowstyle="-", color="k", connectionstyle="arc3,rad=-0.06",
                                    shrinkA=0, shrinkB=1, lw=0.75), zorder=2)
    pyplot.annotate("", xy=(sub_x_locations[2], sub_y_locations[2]), xytext=(sub_x_locations[2], center_c),
                    arrowprops=dict(arrowstyle="-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=2)
    pyplot.annotate("", xy=(sub_x_locations[3], sub_y_locations[3]), xytext=(sub_x_locations[3], center_c),
                    arrowprops=dict(arrowstyle="-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=2)
    pyplot.scatter(sub_x_locations, sub_y_locations, s=3, color="k", zorder=2)
    for former_point, latter_point, radius in zip(former_points, latter_points, ["0.08", "0.11", "0.16", "0.22"]):
        pyplot.annotate("", xy=former_point, xytext=latter_point, zorder=0,
                        arrowprops=dict(arrowstyle="-", color="#666666", connectionstyle="arc3,rad=" + radius,
                                        shrinkA=0, shrinkB=5, lw=0.75, ls="--"))

    # link
    pyplot.plot([ranges[1] - 0.06, ranges[1] - 0.06, 0.76], [1.72, 1.59, 1.59], lw=1.5, color="#BAD5D8", zorder=0)
    pyplot.annotate("", xy=(0.76, 1.59), xytext=(0.76, 1.44),
                    arrowprops=dict(arrowstyle="<|-", color="#BAD5D8", shrinkA=0, shrinkB=0, lw=1.5), zorder=1)
    pyplot.text((ranges[1] + 0.7) / 2.0, 1.59, "for synthesis\nand sequencing", va="center", ha="center",
                bbox=dict(fc="w", ec="#BAD5D8", lw=1.5, boxstyle="round,pad=0.4"), color="#6FA9AF",
                weight="bold", fontsize=9)

    # panel d
    # noinspection SpellCheckingInspection
    information = "[GATC]  [GTC]  [GACTC]  [GATG]"
    locations = linspace(0, len(information) * 0.015, len(information))
    # noinspection PyTypeChecker
    locations = locations - 0.5 * max(locations) + 0.76
    # noinspection PyTypeChecker
    for location, range_info in zip(locations, information):
        pyplot.text(location, 1.29, range_info, va="center", ha="center", fontsize=8)
    # noinspection PyTypeChecker,PyUnresolvedReferences
    number_locations = [(locations[2] + locations[3]) / 2.0, locations[10],
                        locations[18], (locations[26] + locations[27]) / 2.0]
    for index, location in enumerate(number_locations):
        pyplot.scatter(location, 1.39, s=50, color="#AAAAAA", zorder=2)
        pyplot.text(location, 1.385, str(index + 1), color="w", va="center", ha="center", fontsize=8)
    pyplot.fill_between([0.52, 1.00], 1.20, 1.23, fc="#BAD5D8", lw=0, zorder=3)

    # link
    pyplot.plot([0.76, 0.76, 0.59], [1.20, 1.07, 1.07], lw=1.5, color="#BAD5D8", zorder=0)
    pyplot.annotate("", xy=(0.59, 1.07), xytext=(0.59, 0.92),
                    arrowprops=dict(arrowstyle="<|-", color="#BAD5D8", shrinkA=0, shrinkB=0, lw=1.5), zorder=1)
    pyplot.text(0.675, 1.07, "sequencing reads", va="center", ha="center",
                bbox=dict(fc="w", ec="#BAD5D8", lw=1.5, boxstyle="round,pad=0.4"), color="#6FA9AF",
                weight="bold", fontsize=9)

    # panel e
    center_e = 0.40
    width, height, interval, ranges = 0.14, 0.34, 0.03, []
    for index in range(3):
        ranges.append([0.52 + (width + interval) * index, 0.52 + (width + interval) * index + width])
    labels = [["sequence correcting", "( detailed in Fig. 2a )"],
              ["sequence ranking", "using corrected candidates"],
              ["sequence decoding", "( detailed in Fig. 1c )"]]
    for one_range, label in zip(ranges, labels):
        center = (one_range[0] + one_range[1]) / 2.0
        pyplot.fill_between(one_range, center_e - height, center_e + height, fc="#EFEFEF", lw=0, zorder=0)
        pyplot.text(center, center_e + 0.47, label[0], va="center", ha="center", fontsize=8)
        pyplot.text(center, center_e + 0.40, label[1], va="center", ha="center", fontsize=8)
    for former, latter in zip([ranges[0][1], ranges[1][1]], [ranges[1][0], ranges[2][0]]):
        pyplot.annotate("", xy=(former, center_e), xytext=(latter, center_e), zorder=1,
                        arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=3, shrinkB=3, lw=0.75))
    center_1 = (ranges[0][0] + ranges[0][1]) / 2.0

    center_2 = (ranges[1][0] + ranges[1][1]) / 2.0
    biases = (linspace(-0.06, 0.06, 5)[:-1] + linspace(-0.06, 0.06, 5)[1:]) / 2.0
    # noinspection PyTypeChecker
    for index, bias in enumerate(biases):
        pyplot.scatter(center_1 + bias, center_e + 0.270, s=50, color="#AAAAAA", zorder=2)
        pyplot.text(center_1 + bias, center_e + 0.265, str(index + 1), color="w",
                    va="center", ha="center", fontsize=8, zorder=3)
    point_x_bias_groups = array([[+0.000, +0.010, +0.010, +0.020, +0.010],
                                 [+0.000, +0.010, +0.010, +0.000, +0.010],
                                 [+0.000, +0.010, +0.010, +0.020, +0.030]])
    point_y_biases, walk_biases = linspace(+0.090, -0.270, 5), [-0.055, -0.015, +0.025]
    for walk_bias, point_x_biases in zip(walk_biases, point_x_bias_groups):
        pyplot.scatter(center_1 + walk_bias + point_x_biases, center_e + point_y_biases,
                       ec="k", fc="w", lw=0.75, s=10, zorder=4)
        # noinspection PyUnresolvedReferences
        points = array([(center_1 + walk_bias + point_x_biases).tolist(), (center_e + point_y_biases).tolist()]).T
        for point_1, point_2 in zip(points[:-1], points[1:]):
            connection = None
            if point_1[0] < point_2[0]:
                connection = "arc3,rad=0.2"
            elif point_1[0] > point_2[0]:
                connection = "arc3,rad=-0.2"
            pyplot.annotate("", xy=point_1, xytext=point_2, zorder=3,
                            arrowprops=dict(arrowstyle="<|-", connectionstyle=connection,
                                            color="k", shrinkA=1, shrinkB=1, lw=0.75))
    for left, right in [[-0.060, -0.030], [-0.020, +0.010], [+0.020, +0.060]]:
        pyplot.fill_between([center_1 + left, center_1 + right], center_e - 0.290, center_e + 0.110,
                            color="#CCCCCC", zorder=2)
    # noinspection PyUnresolvedReferences
    points_1 = array([biases.tolist(), [+0.27, +0.27, +0.27, +0.27]])
    points_2 = array([[-0.045, -0.005, 0.040], [+0.11, +0.11, +0.11]])
    for former, latter in [[0, 0], [1, 0], [2, 0], [2, 1], [3, 2]]:
        pyplot.plot([center_1 + points_1[0][former], center_1 + points_2[0][latter]],
                    [center_e + points_1[1][former], center_e + points_2[1][latter]],
                    lw=1, color="#CCCCCC", zorder=1)
    for index, (left, right, count) in enumerate([[-0.06, -0.03, 3], [-0.02, +0.01, 1], [+0.02, +0.06, 1]]):
        if index > 0:
            pyplot.fill_between([center_2 + left, center_2 + right], center_e - 0.11, center_e + 0.29,
                                color="#CCCCCC", zorder=2)
            pyplot.vlines(center_2 + (left + right) / 2.0, center_e - 0.21, center_e - 0.11, lw=1, color="#CCCCCC")
        else:
            pyplot.fill_between([center_2 + left, center_2 + right], center_e - 0.11, center_e + 0.29,
                                color="#E8C3E2", zorder=2)
            pyplot.vlines(center_2 + (left + right) / 2.0, center_e - 0.21, center_e - 0.11, lw=1, color="#E8C3E2")
        pyplot.text(center_2 + (left + right) / 2.0, center_e - 0.27, "x" + str(count),
                    va="center", ha="center", fontsize=8, zorder=3)
    for walk_bias, point_x_biases in zip(walk_biases, point_x_bias_groups):
        pyplot.scatter(center_2 + walk_bias + point_x_biases, center_e + 0.18 + point_y_biases,
                       ec="k", fc="w", lw=0.75, s=10, zorder=4)
        # noinspection PyUnresolvedReferences
        points = array([(center_2 + walk_bias + point_x_biases).tolist(),
                        (center_e + 0.18 + point_y_biases).tolist()]).T
        for point_1, point_2 in zip(points[:-1], points[1:]):
            connection = None
            if point_1[0] < point_2[0]:
                connection = "arc3,rad=0.2"
            elif point_1[0] > point_2[0]:
                connection = "arc3,rad=-0.2"
            pyplot.annotate("", xy=point_1, xytext=point_2, zorder=3,
                            arrowprops=dict(arrowstyle="<|-", connectionstyle=connection,
                                            color="k", shrinkA=1, shrinkB=1, lw=0.75))

    center_3 = (ranges[2][0] + ranges[2][1]) / 2.0
    biases = array([[+0.00, +0.015, +0.015, +0.030, +0.015], [+0.24, +0.12, +0.00, -0.12, -0.24]])
    pyplot.scatter(center_3 - 0.055 + biases[0], center_e + biases[1], ec="k", fc="w", lw=0.75, s=10, zorder=3)
    for bias_1, bias_2 in zip(biases.T[:-1], biases.T[1:]):
        point_1 = (center_3 - 0.055 + bias_1[0], center_e + bias_1[1])
        point_2 = (center_3 - 0.055 + bias_2[0], center_e + bias_2[1])
        connection = None
        if point_1[0] < point_2[0]:
            connection = "arc3,rad=0.2"
        elif point_1[0] > point_2[0]:
            connection = "arc3,rad=-0.2"
        pyplot.annotate("", xy=point_1, xytext=point_2, zorder=2,
                        arrowprops=dict(arrowstyle="<|-", connectionstyle=connection,
                                        color="k", shrinkA=1, shrinkB=2, lw=0.75))
    for bias, info_1, info_2 in zip([-0.025, -0.015, -0.005, +0.005, +0.015, +0.025], "[GATC]", "[1010]"):
        pyplot.text(center_3 + 0.025 + bias, center_e + 0.180, info_1, va="center", ha="center", fontsize=8)
        pyplot.text(center_3 + 0.025 + bias, center_e - 0.180, info_2, va="center", ha="center", fontsize=8)
    pyplot.text(center_3 + 0.025, center_e + 0.050, "reconstruct", va="center", ha="center", fontsize=8)
    pyplot.text(center_3 + 0.025, center_e - 0.050, "the walk", va="center", ha="center", fontsize=8)

    pyplot.xlim(+0.10, 1.01)
    pyplot.ylim(-0.03, 3.53)
    pyplot.axis("off")

    # noinspection PyTypeChecker
    pyplot.subplot(grid[6:, 0])

    speeds = array([[0.5, 9.1], shown_data["f"],
                    [150 * 1024 * 8, 4.5 * (1024 ** 2) * 8],
                    [20 * (1024 ** 2) * 8, 250 * (1024 ** 2) * 8],
                    [550 * (1024 ** 2) * 8, 7 * (1024 ** 3) * 8]])
    indices = [0.8, 0.8, 2.3, 3.1, 4.0]
    labels = ["DNA-based", "DNA-based", "optical", "magnetic", "semiconductor"]
    ranges = ["0.5 bit/s~9.1 bit/s", "96 KB/s~13.9 MB/s", "150 KB/s~4.5 MB/s", "20 MB/s~250 MB/s", "550 MB/s~7 GB/s"]
    additions = ["Zhao, et al. 2024", "SPIDER-WEB", "", "", ""]
    colors = ["#9899C0", "#7AB5BB", "#F3DAA3", "#F3DAA3", "#F3DAA3"]
    for location, bounds, label, addition, range_info, color in zip(indices, speeds, labels, additions, ranges, colors):
        former_value, latter_value = log10(bounds * 10.0) / (log10(32 * 8 * (1024 ** 3) * 10.0)) * 4.00
        if former_value > 0.7:
            former_value -= 0.7
        if latter_value > 0.7:
            latter_value -= 0.7
        if len(addition) > 0:
            pyplot.text((former_value + latter_value) / 2.0, location - 0.08, label + "\n   data storage   \n",
                        bbox=dict(color=color, boxstyle="round,pad=0.2"), ha="center", va="center", fontsize=9)
            pyplot.text((former_value + latter_value) / 2.0, location - 0.43, addition, color="w",
                        ha="center", va="center", fontsize=8)
        else:
            pyplot.text((former_value + latter_value) / 2.0, location - 0.20, label + "\ndata storage",
                        bbox=dict(color=color, boxstyle="round,pad=0.2"), ha="center", va="center", fontsize=9)
        pyplot.hlines(location - 0.2, former_value, latter_value, lw=0.75, color="k")
        pyplot.vlines([former_value, latter_value], location - 0.4, location, lw=1.5, color="k")
        former_label, latter_label = range_info.split("~")
        pyplot.text(former_value - 0.02, location - 0.21, former_label, va="center", ha="right", fontsize=8)
        pyplot.text(latter_value + 0.02, location - 0.21, latter_label, va="center", ha="left", fontsize=8)

    pyplot.plot([0.00, 0.00, 3.30, 3.30], [-0.15, 0.00, 0.00, -0.15], lw=0.75, color="k", zorder=2)
    for value, label in zip([0.1, 1.0, 8.0, 8 * 1024, 8 * (1024 ** 2), 8 * (1024 ** 3), 32 * 8 * (1024 ** 3)],
                            ["0.1 bit/s", "1 bit/s", "1 byte/s", "1 KB/s", "1 MB/s", "1 GB/s", "32 GB/s"]):
        location = log10(value * 10.0) / (log10(32 * 8 * (1024 ** 3) * 10.0)) * 4.00
        if location > 0.70:
            location -= 0.70
        if 0.00 < location < 3.30:
            pyplot.vlines(location, -0.15, 0.00, color="k", lw=0.75, zorder=1)
        pyplot.text(location, -0.45, label, va="center", ha="center", fontsize=9, zorder=2)
    former_value = log10(8.0 * 10.0) / (log10(32 * 8 * (1024 ** 3) * 10.0)) * 4.00
    latter_value = log10(8 * 1024 * 10.0) / (log10(32 * 8 * (1024 ** 3) * 10.0)) * 4.00 - 0.70
    center = (former_value + latter_value) / 2.0
    pyplot.fill_between([center - 0.02, center + 0.02], -0.05, +0.05, color="w", zorder=3)
    pyplot.vlines([center - 0.02, center + 0.02], -0.10, +0.10, lw=0.75, color="k", zorder=4)

    pyplot.text(1.70, -0.80, "data retrieval efficiency", va="center", ha="center", fontsize=9)
    pyplot.xlim(-0.10, +3.40)
    pyplot.ylim(-1.00, +4.20)
    pyplot.axis("off")

    figure.align_labels()
    figure.text(0.020, 0.995, "a", va="center", ha="center", fontsize=12)
    figure.text(0.435, 0.995, "b", va="center", ha="center", fontsize=12)
    figure.text(0.435, 0.785, "c", va="center", ha="center", fontsize=12)
    figure.text(0.435, 0.560, "d", va="center", ha="center", fontsize=12)
    figure.text(0.435, 0.460, "e", va="center", ha="center", fontsize=12)
    figure.text(0.020, 0.255, "f", va="center", ha="center", fontsize=12)

    pyplot.savefig("./fig/main-01.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main_f02():
    """
    Create Figure 2 in the main text.

    TITLE: Error tolerance performance of SPIDER-WEB.
    """
    shown_data = load_data(file_path=pkg_path + "main-02.pkl")

    figure = pyplot.figure(figsize=(10, 9), tight_layout=True)
    grid = pyplot.GridSpec(3, 3)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[0, :])

    def draw_graph(ps, s, w, se=None, h=None):
        for i, (xi, yi) in enumerate(ps):
            pyplot.scatter(xi, yi, color="white", edgecolor="k", s=s, lw=w)
        for f, l in [(1, 2), (2, 4), (2, 5), (3, 1), (3, 8), (4, 3), (5, 6), (6, 1), (6, 8), (7, 4), (7, 5), (8, 7)]:
            if se is None:
                # noinspection PyTypeChecker
                pyplot.annotate("", xy=tuple(ps[l - 1]), xytext=tuple(ps[f - 1]),
                                arrowprops=dict(arrowstyle="-|>", color="k",
                                                shrinkA=s / 5, shrinkB=s / 5, lw=w), zorder=1)
            elif (f, l) in se:
                # noinspection PyTypeChecker
                pyplot.annotate("", xy=tuple(ps[l - 1]), xytext=tuple(ps[f - 1]),
                                arrowprops=dict(arrowstyle="-|>", color=("black" if (f, l) not in h else "red"),
                                                shrinkA=s / 5, shrinkB=s / 5, lw=w), zorder=1)
        for f, l in [(1, 3), (5, 7)]:
            if se is None:
                # noinspection PyTypeChecker
                pyplot.annotate("", xy=tuple(ps[f - 1]), xytext=tuple(ps[l - 1]), zorder=1,
                                arrowprops=dict(arrowstyle="<|-", color="k",
                                                connectionstyle="arc3,rad=0.9", shrinkA=s / 5, shrinkB=s / 5, lw=w))
            elif (f, l) in se:
                # noinspection PyTypeChecker
                pyplot.annotate("", xy=tuple(ps[f - 1]), xytext=tuple(ps[l - 1]), zorder=1,
                                arrowprops=dict(arrowstyle="<|-", color="k" if (f, l) not in h else "red",
                                                connectionstyle="arc3,rad=0.9", shrinkA=s / 5, shrinkB=s / 5, lw=w))
        for f, l in [(4, 2), (8, 6)]:
            if se is None:
                # noinspection PyTypeChecker
                pyplot.annotate("", xy=tuple(ps[l - 1]), xytext=tuple(ps[f - 1]), zorder=1,
                                arrowprops=dict(arrowstyle="-|>", color="k",
                                                connectionstyle="arc3,rad=0.9", shrinkA=s / 5, shrinkB=s / 5, lw=w))
            elif (f, l) in se:
                # noinspection PyTypeChecker
                pyplot.annotate("", xy=tuple(ps[l - 1]), xytext=tuple(ps[f - 1]), zorder=1,
                                arrowprops=dict(arrowstyle="-|>", color="k" if (f, l) not in h else "red",
                                                connectionstyle="arc3,rad=0.9", shrinkA=s / 5, shrinkB=s / 5, lw=w))

    length, bias, points = 0.3, 22.5, []
    for angle in arange(0, 360, 45):
        radio = deg2rad(angle + bias)
        points.append([sin(radio) * length + 0.45, cos(radio) * length + 0.5])
    points = array(points)
    pyplot.text(0.45, 0.85, "GC-balanced coding digraph", va="center", ha="center", fontsize=9)
    draw_graph(ps=points, s=20, w=0.75)
    color_groups = [["#FBE5D6", "#F4B183", "#ED7D31", "#A24A0E"], ["#E2F0D9", "#A9D18E", "#70AD47", "#4D7731"]]
    x, y = 0.45, 0.5
    for location, info in zip([(x + 0.38, y + 0.15), (x - 0.38, y + 0.15), (x + 0.38, y - 0.15), (x - 0.38, y - 0.15),
                               (x + 0.10, y + 0.07), (x + 0.10, y - 0.07), (x - 0.10, y + 0.07), (x - 0.10, y - 0.07),
                               (x + 0.22, y + 0.21), (x - 0.22, y + 0.21), (x + 0.22, y - 0.21), (x - 0.22, y - 0.21),
                               (x + 0.15, y + 0.14), (x + 0.15, y - 0.14), (x - 0.15, y + 0.14), (x - 0.15, y - 0.14)],
                              ["{G|1}", "{G|1}", "{C|0}", "{C|0}", "{T|1}", "{T|1}", "{A|0}", "{A|0}",
                               "{C|0}", "{C|0}", "{G|1}", "{G|1}", "{A|0}", "{A|0}", "{T|1}", "{T|1}"]):
        xs = [location[0] - 0.050, location[0] - 0.050, location[0] + 0.050, location[0] + 0.050, location[0] - 0.050]
        ys = [location[1] - 0.025, location[1] + 0.025, location[1] + 0.025, location[1] - 0.025, location[1] - 0.025]
        pyplot.plot(xs, ys, color="k", lw=0.75, zorder=4)
        pyplot.vlines(location[0], location[1] - 0.025, location[1] + 0.025, color="k", lw=0.75, zorder=3)
        pyplot.fill_between([location[0] - 0.05, location[0]], location[1] - 0.025, location[1] + 0.025,
                            fc=color_groups[0][{"A": 0, "C": 1, "G": 2, "T": 3}[info[1]]], lw=0, zorder=2)
        if info[1] != "T":
            pyplot.text(location[0] - 0.025, location[1] - 0.005, info[1],
                        color="k", va="center", ha="center", fontsize=7.5, zorder=3)
        else:
            pyplot.text(location[0] - 0.025, location[1] - 0.005, info[1],
                        color="w", va="center", ha="center", fontsize=7.5, zorder=3)
        pyplot.fill_between([location[0], location[0] + 0.05], location[1] - 0.025, location[1] + 0.025,
                            fc=color_groups[1][int(info[3])], lw=0, zorder=2)
        if info[3] != "3":
            pyplot.text(location[0] + 0.025, location[1] - 0.005, info[3],
                        color="k", va="center", ha="center", fontsize=7.5, zorder=3)
        else:
            pyplot.text(location[0] + 0.025, location[1] - 0.005, info[3],
                        color="w", va="center", ha="center", fontsize=7.5, zorder=3)

    pyplot.scatter([0.02], [0.07], fc="w", ec="k", s=20, lw=0.75)
    pyplot.text(0.05, 0.07, ": vertex", va="center", ha="left", fontsize=8)
    pyplot.annotate("", xy=(0.22, 0.11), xytext=(0.32, 0.11), zorder=1,
                    arrowprops=dict(arrowstyle="<|-", color="k", lw=1))
    pyplot.annotate("", xy=(0.22, 0.03), xytext=(0.32, 0.03), zorder=1,
                    arrowprops=dict(arrowstyle="<|-", color="red", lw=1))
    pyplot.text(0.33, 0.11, ": normal arc", va="center", ha="left", fontsize=8)
    pyplot.text(0.33, 0.03, ": accessible arc (when error detected)", va="center", ha="left", fontsize=8)
    pyplot.fill_between([0.62 - 0.017, 0.62 + 0.017], 0.11 - 0.016, 0.11 + 0.025, lw=0, color="lightblue")
    pyplot.text(0.62, 0.11, "N", va="center", ha="center", fontsize=8)
    pyplot.text(0.65, 0.11, ": adjustment using N", va="center", ha="left", fontsize=8)
    pyplot.text(1.12, 0.03, "N", va="center", ha="center", color="r", fontsize=8)
    pyplot.text(1.15, 0.03, ": inaccessible N (error)", va="center", ha="left", fontsize=8)
    pyplot.scatter([1.12], [0.11], marker=".", s=8, color="k")
    pyplot.text(1.15, 0.11, ": no need for detection", va="center", ha="left", fontsize=8)

    pyplot.text(0.93, 0.545, "encode", va="center", ha="center", color="#0070C0", fontsize=9)
    pyplot.annotate("", xy=(1.05, 0.5), xytext=(0.83, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="#0070C0", lw=2))
    pyplot.text(1.37, 0.545, "wet lab", va="center", ha="center", color="#0070C0", fontsize=9)
    pyplot.annotate("", xy=(1.49, 0.5), xytext=(1.28, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="#0070C0", lw=2))
    length = 0.1
    # noinspection SpellCheckingInspection
    for center, select, highlight, sequence in zip([(1.15, 0.35), (1.60, 0.35)],
                                                   [[(3, 8), (8, 7), (7, 5), (5, 6), (6, 1), (1, 2)],
                                                    [(3, 8), (8, 7), (7, 5), (5, 6), (5, 7)]],
                                                   [[], [(5, 6), (5, 7)]],
                                                   ["TCTGAC", "TCTAAC"]):
        points = []
        for angle in arange(0, 360, 45):
            radio = deg2rad(angle + bias)
            points.append([sin(radio) * length + center[0], cos(radio) * length + center[1]])
        points = array(points)
        draw_graph(ps=points, s=10, w=0.75, se=select, h=highlight)
        start = center[0] - 0.03 * 2.5
        for index, nucleotide in enumerate(sequence):
            if index == 3 and nucleotide == "A":
                pyplot.text(start + index * 0.03, 0.5, nucleotide, color="r", va="center", ha="center", fontsize=8)
            else:
                pyplot.text(start + index * 0.03, 0.5, nucleotide, va="center", ha="center", fontsize=8)
    pyplot.text(1.58, 0.67, "correct", va="center", ha="center", color="#0070C0", fontsize=9)
    pyplot.annotate("", xy=(1.75, 0.675), xytext=(1.6, 0.55),
                    arrowprops=dict(arrowstyle="-|>", color="#0070C0", lw=2, connectionstyle="arc3,rad=-0.2"))
    pyplot.fill_between([1.75, 4.00], 0, 1.0, color="#EEEEEE", zorder=0)
    pyplot.text(1.99, 0.91, "tracing", va="bottom", ha="right", fontsize=9)
    pyplot.annotate("", xy=(2.0, 0.9), xytext=(2.2, 0.9),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(2.0, 0.9), xytext=(2.0, 0.73),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.hlines(0.86, 2.15, 2.65, color="k", lw=0.75)
    pyplot.hlines(0.86, 2.75, 3.25, color="k", lw=0.75)
    pyplot.hlines(0.86, 3.35, 3.55, color="k", lw=0.75)
    pyplot.text(2.40, 0.90, "substitute", va="center", ha="center", fontsize=8.5)
    pyplot.text(3.00, 0.90, "insert", va="center", ha="center", fontsize=8.5)
    pyplot.text(3.45, 0.90, "delete", va="center", ha="center", fontsize=8.5)
    pyplot.vlines(2.10, 0.575, 0.775, color="k", lw=0.75)
    pyplot.vlines(2.10, 0.310, 0.510, color="k", lw=0.75)
    pyplot.vlines(2.10, 0.025, 0.225, color="k", lw=0.75)
    pyplot.text(2.08, 0.675, "(+0) position", va="center", ha="right", fontsize=8.5)
    pyplot.text(2.08, 0.410, "(\N{MINUS SIGN}1) position", va="center", ha="right", fontsize=8.5)
    pyplot.text(2.08, 0.125, "(\N{MINUS SIGN}2) position", va="center", ha="right", fontsize=8.5)
    # noinspection SpellCheckingInspection
    for center, select, highlight, sequence, change in zip([(2.25, 0.675), (2.25, 0.410), (2.25, 0.125), (2.55, 0.675),
                                                            (2.85, 0.675), (2.85, 0.410), (2.85, 0.125),
                                                            (3.15, 0.675), (3.15, 0.410), (3.15, 0.125),
                                                            (3.45, 0.675), (3.45, 0.410), (3.45, 0.125)],
                                                           [[(3, 8), (8, 7), (7, 5), (5, 7), (7, 4), (4, 2)],
                                                            [(3, 8), (8, 7), (7, 4), (4, 3), (4, 2)],
                                                            [(3, 8), (8, 6), (6, 8), (8, 7)],
                                                            [(3, 8), (8, 7), (7, 5), (5, 6), (6, 1), (1, 2)],
                                                            [(3, 8), (8, 7), (7, 5), (5, 7), (7, 4), (4, 3), (4, 2)],
                                                            [(3, 8), (8, 7), (7, 4), (4, 3), (4, 2)],
                                                            [(3, 8), (8, 7), (7, 5), (7, 4)],
                                                            [(3, 8), (8, 7), (7, 5), (5, 6), (6, 1), (1, 2), (1, 3)],
                                                            [(3, 8), (8, 7), (7, 5), (5, 6), (5, 7)],
                                                            [(3, 8), (8, 6), (6, 1), (6, 8)],
                                                            [(3, 8), (8, 7), (7, 5), (5, 6), (5, 7)],
                                                            [(3, 8), (8, 7), (7, 4), (4, 3), (4, 2)],
                                                            [(3, 8), (8, 7), (8, 6)]],
                                                           [[], [(4, 3), (4, 2)], [(6, 8), (8, 7)], [],
                                                            [(4, 3), (4, 2)], [(4, 3), (4, 2)], [(7, 5), (7, 4)],
                                                            [(1, 2), (1, 3)], [(5, 6), (5, 7)], [(6, 1), (6, 8)],
                                                            [(5, 6), (5, 7)], [(4, 2), (4, 3)], [(8, 7), (8, 6)]],
                                                           ["TCTCAC", "TCAAAC", "TGTAAC", "TCTGAC",
                                                            "TCTCAAC", "TCATAAC", "TCCTAAC",
                                                            "TCTGAAC", "TCTTAAC", "TGCTAAC",
                                                            "TCTAC", "TCAAC", "TTAAC"],
                                                           [4, 3, 2, 4, 4, 3, 2, 4, 3, 2, 4, 3, 2]):
        points = []
        for angle in arange(0, 360, 45):
            radio = deg2rad(angle + bias)
            points.append([sin(radio) * length + center[0], cos(radio) * length + center[1]])
        points = array(points)
        draw_graph(ps=points, s=10, w=0.75, se=select, h=highlight)
        # noinspection SpellCheckingInspection
        if sequence not in ["TCTCAC", "TCTGAC"]:
            for location, nucleotide in enumerate(sequence):
                if location == change - 1:
                    pyplot.fill_between([center[0] - 0.08 + location * 0.03 - 0.016,
                                         center[0] - 0.08 + location * 0.03 + 0.016],
                                        center[1] + 0.155, center[1] + 0.155 - 0.05, lw=0., color="lightblue")
                if location == len(select) - 3:
                    pyplot.text(center[0] - 0.08 + location * 0.03, center[1] + 0.125,
                                nucleotide, color="red", va="center", ha="center", fontsize=8)
                elif location <= change - 1:
                    pyplot.text(center[0] - 0.08 + location * 0.03, center[1] + 0.125,
                                nucleotide, va="center", ha="center", fontsize=8)
                else:
                    pyplot.scatter([center[0] - 0.08 + location * 0.03], [center[1] + 0.125], marker=".",
                                   s=8, color="k")
        else:
            for location, nucleotide in enumerate(sequence):
                pyplot.text(center[0] - 0.08 + location * 0.03, center[1] + 0.125,
                            nucleotide, va="center", ha="center", fontsize=8)
                if location == change - 1:
                    pyplot.fill_between([center[0] - 0.08 + location * 0.03 - 0.016,
                                         center[0] - 0.08 + location * 0.03 + 0.016],
                                        center[1] + 0.155, center[1] + 0.155 - 0.05, lw=0., color="lightblue")
    pyplot.xlabel("\nillustration of walk tracing process", fontsize=9)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.xlim(0.0, 3.6)
    pyplot.ylim(0.0, 1.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[1, 0])
    results = shown_data["b"]
    colors = pyplot.get_cmap("RdYlBu_r")(linspace(0.2, 0.8, 8))

    for index, (x, y) in enumerate(results[1:]):
        pyplot.plot(x, y, color=colors[index], lw=1.5, label="%.1f" % ((index + 1) / 2))
    pyplot.text(200, 0.0325, "sequence\nlength", va="center", ha="center", fontsize=9)
    pyplot.hlines(0.03, 120, 280, lw=0.75, color="k")
    pyplot.vlines(200, 0.0, 0.03, lw=0.75, ls="--", colors="k")
    legend = pyplot.legend(loc="upper right", ncol=2, fontsize=8, framealpha=1,
                           title="% of edit errors\n(samples = 1E6)", title_fontsize=8)
    # noinspection PyUnresolvedReferences
    legend.get_title().set_ha("center")
    pyplot.xlabel("vertex access times", fontsize=9)
    pyplot.ylabel("proportion", fontsize=9)
    pyplot.xticks(arange(0, 801, 200), arange(0, 801, 200), fontsize=9)
    pyplot.yticks(linspace(0, 0.04, 9), [("%.1f" % v) + "%" for v in linspace(0, 4, 9)], fontsize=9)
    pyplot.xlim(0, 800)
    pyplot.ylim(0, 0.04)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[1, 1])
    results = shown_data["c"]

    access, labels = [False, False, False], ["contained", "screened", "undetected"]
    for index in range(9):
        location = 0
        for type_index, color in enumerate(["#A9D18E", "#57AFFF", "#FA8D65"]):
            if results[index, type_index] > 0:
                if not access[type_index]:
                    pyplot.fill_between([location, location + results[index, type_index]], index - 0.4, index + 0.4,
                                        fc=color, lw=0, zorder=0, label=labels[type_index])
                else:
                    pyplot.fill_between([location, location + results[index, type_index]], index - 0.4, index + 0.4,
                                        fc=color, lw=0, zorder=0)
                access[type_index] = True
                location += results[index, type_index]
            if results[index, type_index] >= 0.2:
                if results[index, type_index] < 1:
                    info = ("%.2f" % (results[index, type_index] * 100)) + "%"
                else:
                    info = "100%"
                pyplot.text(location - 0.01, index - 0.08, info, va="center", ha="right", fontsize=9)

    pyplot.legend(loc="lower left", fontsize=8, framealpha=1, title="correcting situation", title_fontsize=8)
    pyplot.xlabel("situation proportion in sequence candidate set", fontsize=9)
    pyplot.ylabel("% of edit errors in raw sequence", fontsize=9)
    pyplot.xticks(linspace(0, 1, 5), [str(int(v * 100)) + "%" for v in linspace(0, 1, 5)], fontsize=9)
    pyplot.yticks(arange(0, 10), ["%.1f" % (v / 200.0 * 100.0) for v in arange(0, 10)], fontsize=9)
    pyplot.xlim(0, 1)
    pyplot.ylim(-0.6, 8.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[1, 2])

    results = shown_data["d"]
    colors = pyplot.get_cmap("RdYlBu_r")(linspace(0.2, 0.8, 8))
    for index, value in enumerate(results):
        if index > 0:
            pyplot.fill_between([0, value], index - 0.4, index + 0.4, fc=colors[index - 1], ec="k", lw=0.75)
        else:
            pyplot.fill_between([0, value], index - 0.4, index + 0.4, fc="w", ec="k", lw=0.75)
        pyplot.text(value + 0.1, index - 0.08, "%.2f" % value, va="center", ha="left", fontsize=9)

    pyplot.xlabel("average required sequence copies", fontsize=9)
    pyplot.ylabel("% of edit errors in raw sequence", fontsize=9)
    pyplot.xticks(arange(7), arange(7), fontsize=9)
    pyplot.yticks(arange(0, 10), ["%.1f" % (v / 200.0 * 100.0) for v in arange(0, 10)], fontsize=9)
    pyplot.xlim(0, 6)
    pyplot.ylim(-0.6, 8.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2, :2])

    results = shown_data["e"]
    colors = pyplot.get_cmap("RdYlBu_r")(linspace(0.2, 0.8, 8))
    pyplot.hlines(arange(0, 51, 10)[1:-1], 0, 10, color="silver", lw=0.75, ls="--", zorder=0)
    pyplot.vlines(arange(11)[1:-1], 0, 50, color="silver", lw=0.75, ls="--", zorder=0)
    for error_index in range(8):
        points_x, points_y = results[error_index]
        pyplot.scatter(points_x, points_y, s=12, ec="k", fc=colors[error_index], lw=0.75, zorder=1,
                       label="%.1f" % ((error_index + 1) / 2))

    legend = pyplot.legend(loc="upper left", ncol=2, fontsize=8, framealpha=1,
                           title="% of edit errors\n(samples: 1E1 - 1E8)", title_fontsize=8)
    # noinspection PyUnresolvedReferences
    legend.get_title().set_ha("center")
    pyplot.xlabel("average unintended proportion (false positive sequences / retrieved sequences)", fontsize=9)
    pyplot.ylabel("sequence copies", fontsize=9)
    pyplot.xticks(arange(10), ["1E+0"] + ["1E\N{MINUS SIGN}" + str(v) for v in arange(1, 10)], fontsize=9)
    pyplot.yticks(arange(0, 51, 10), arange(0, 51, 10), fontsize=9)
    pyplot.xlim(0, 9)
    pyplot.ylim(0, 50)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2, 2])

    results = shown_data["f"]
    colors = pyplot.get_cmap("RdYlBu_r")(linspace(0.2, 0.8, 8))
    pyplot.vlines(arange(1, 7), 0, 400, color="silver", lw=0.75, ls="--")
    pyplot.hlines(arange(0, 401, 80)[1:-1], 0, 6, color="silver", lw=0.75, ls="--")
    for error_index in range(8):
        sizes, copies = results[error_index]
        pyplot.plot(sizes, copies, color=colors[error_index], lw=1.5, label="%.1f" % ((error_index + 1) / 2))

    legend = pyplot.legend(loc="upper left", ncol=2, fontsize=8, framealpha=1,
                           title="% of edit errors\n(fitted by Fig. 2e)", title_fontsize=8)
    # noinspection PyUnresolvedReferences
    legend.get_title().set_ha("center")
    pyplot.xlabel("amount of data", fontsize=9)
    pyplot.ylabel("required sequence copies", fontsize=9)
    pyplot.xticks(arange(6) + 1, ["KB", "MB", "GB", "TB", "PB", "EB"], fontsize=9)
    pyplot.yticks(arange(0, 401, 80), arange(0, 401, 80), fontsize=9)
    pyplot.xlim(1, 6)
    pyplot.ylim(0, 400)

    figure.align_labels()
    figure.text(0.022, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.022, 0.675, "b", va="center", ha="center", fontsize=12)
    figure.text(0.360, 0.675, "c", va="center", ha="center", fontsize=12)
    figure.text(0.685, 0.675, "d", va="center", ha="center", fontsize=12)
    figure.text(0.022, 0.335, "e", va="center", ha="center", fontsize=12)
    figure.text(0.685, 0.335, "f", va="center", ha="center", fontsize=12)

    pyplot.savefig("./fig/main-02.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main_f03():
    """
    Create Figure 3 in the main text.

    TITLE: Direct data retrieval capability.
    """
    shown_data = load_data(file_path=pkg_path + "main-03.pkl")

    figure = pyplot.figure(figsize=(10, 10), tight_layout=True)
    grid = pyplot.GridSpec(11, 12)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[:3, :8])

    gradient_colors = pyplot.get_cmap("binary")(linspace(0.05, 0.75, 6))
    pyplot.plot([0.02, 0.02, 0.34, 0.34], [0.880, 0.905, 0.905, 0.880], lw=0.75, color="k", zorder=0)
    pyplot.vlines(0.06, 0.880, 0.905, lw=0.75, color="k", zorder=0)
    pyplot.vlines(0.30, 0.880, 0.905, lw=0.75, color="k", zorder=0)
    pyplot.text(0.51, 0.99, "experimental replicates\n(3 library pools)", va="center", ha="center", fontsize=9)
    pyplot.text(0.82, 0.99, "introducing error\nby error-prone PCR",
                va="center", ha="center", fontsize=9)
    for idx, info in zip([0.04, 0.18, 0.32], ["primer\n25 nt", "payload\n150 nt", "primer\n25 nt"]):
        pyplot.vlines(idx, 0.905, 0.930, lw=0.75, color="k", zorder=0)
        pyplot.text(idx, 0.99, info, va="center", ha="center", fontsize=9)
    for index, (refs_y, sample_color) in enumerate(zip([0.87, 0.57, 0.27], ["#E0A698", "#98B7E6", "#7CBBC6"])):
        for bias_y in [0.02, 0.09, 0.22]:
            pyplot.fill_between([0.02, 0.34], refs_y - bias_y, refs_y - bias_y - 0.05,
                                fc="w", ec="k", lw=0.75, zorder=0)
            pyplot.fill_between([0.06, 0.30], refs_y - bias_y, refs_y - bias_y - 0.05,
                                fc=sample_color, ec="k", lw=0.75, zorder=0)
        pyplot.plot([0.35, 0.36, 0.36, 0.35], [refs_y - 0.02, refs_y - 0.02, refs_y - 0.27, refs_y - 0.27],
                    lw=0.75, color="k", zorder=0)
        pyplot.hlines(refs_y - 0.15, 0.36, 0.37, lw=0.75, color="k", zorder=0)
        pyplot.text(0.41, refs_y - 0.15, r"× 1000", va="center", ha="center", fontsize=9)
        # noinspection SpellCheckingInspection
        pyplot.text(0.18, refs_y - 0.19, r"$\vdots$", va="center", ha="center", fontsize=9)
        pyplot.hlines(refs_y - 0.15, 0.45, 0.46, lw=0.75, color="k", zorder=0)
        if index == 0:
            pyplot.annotate("", xy=(0.65, refs_y - 0.15), xytext=(0.56, refs_y - 0.15),
                            arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=0)
        if index == 1:
            pyplot.hlines(refs_y - 0.15, 0.56, 0.57, lw=0.75, color="k", zorder=0)
        if index == 2:
            pyplot.plot([0.56, 0.57, 0.57], [refs_y - 0.15, refs_y - 0.15, 0.72], lw=0.75, color="k", zorder=0)
        pyplot.fill_between([0.46, 0.56], refs_y - 0.02, refs_y - 0.27, fc=gradient_colors[0], lw=0, zorder=-1)
        pyplot.text(0.51, refs_y - 0.24, "pool " + str(index + 1), va="center", ha="center", fontsize=8)
        pyplot.fill_between([0.50, 0.52], refs_y - 0.15, refs_y - 0.08, lw=0, fc="w")
        values = array([[0.50, 0.15], [0.50, 0.17], [0.51, 0.20], [0.52, 0.17], [0.52, 0.15]])
        values[:, 1] = refs_y - values[:, 1]
        # noinspection PyTypeChecker
        polygon = patches.Polygon(values, lw=0.75, ec="k", fc=sample_color, joinstyle="round")
        ax.add_patch(polygon)
        values = array([[0.50, 0.08], [0.50, 0.17], [0.51, 0.20], [0.52, 0.17], [0.52, 0.08]])
        values[:, 1] = refs_y - values[:, 1]
        # noinspection PyTypeChecker
        polygon = patches.Polygon(values, lw=0.75, ec="k", fc="none", joinstyle="round")
        ax.add_patch(polygon)
        pyplot.fill_between([0.495, 0.525], refs_y - 0.08, refs_y - 0.07, lw=0.75, ec="k", fc="w", zorder=2)
        pyplot.fill_between([0.495, 0.525], refs_y - 0.07, refs_y - 0.06, lw=0.75, ec="k", fc="w", zorder=2)
        pyplot.hlines(refs_y - 0.10, 0.50, 0.52, lw=0.75, color="k", zorder=3)
        pyplot.plot([0.495, 0.490, 0.490, 0.495], [refs_y - 0.075, refs_y - 0.075, refs_y - 0.065, refs_y - 0.065],
                    lw=0.75, color="k", zorder=2)
        pyplot.hlines([refs_y - 0.12, refs_y - 0.14, refs_y - 0.16], 0.500, 0.505, lw=0.75, color="k", zorder=3)

    pyplot.text(0.60, 0.78, "used\nas #0", va="center", ha="center", fontsize=8)
    indices_1 = linspace(0.64, 1.00, 7)[:-1]
    indices_2 = array([0.87, 0.77, 0.77, 0.77, 0.77, 0.77])
    for round_index, (index_1, index_2, gradient_color) in enumerate(zip(indices_1, indices_2, gradient_colors)):
        pyplot.fill_between([index_1 + 0.0075, index_1 + 0.0125], index_2 - 0.06, index_2,
                            lw=0.75, ec="k", fc="w", zorder=2)
        pyplot.fill_between([index_1 + 0.0125, index_1 + 0.0200], index_2 - 0.05, index_2 - 0.01,
                            lw=0.75, ec="k", fc="w", zorder=2)
        pyplot.plot([index_1 + 0.010, index_1 + 0.010, index_1 + 0.015],
                    [index_2 - 0.06, index_2 - 0.075, index_2 - 0.075],
                    lw=0.75, color="k")
        values = array([[0.02, 0.15], [0.02, 0.17], [0.03, 0.20], [0.04, 0.17], [0.04, 0.15]])
        values[:, 0] += index_1
        values[:, 1] = index_2 - values[:, 1]
        # noinspection PyTypeChecker
        polygon = patches.Polygon(values, lw=0.75, ec="k", fc=gradient_color, joinstyle="round")
        ax.add_patch(polygon)
        values = array([[0.02, 0.08], [0.02, 0.17], [0.03, 0.20], [0.04, 0.17], [0.04, 0.08]])
        values[:, 0] += index_1
        values[:, 1] = index_2 - values[:, 1]
        # noinspection PyTypeChecker
        polygon = patches.Polygon(values, lw=0.75, ec="k", fc="none", joinstyle="round")
        ax.add_patch(polygon)
        pyplot.fill_between([index_1 + 0.015, index_1 + 0.045], index_2 - 0.08, index_2 - 0.07,
                            lw=0.75, ec="k", fc="w", zorder=2)
        pyplot.hlines([index_2 - 0.12, index_2 - 0.14, index_2 - 0.16], index_1 + 0.020, index_1 + 0.025,
                      lw=0.75, color="k", zorder=3)
        pyplot.annotate("", xy=(index_1 + 0.03, 0.43), xytext=(index_1 + 0.03, index_2 - 0.28),
                        arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
        pyplot.scatter([index_1 + 0.03], [index_2 - 0.28], s=10, fc=gradient_color, ec="k", lw=0.75, zorder=4)
        if round_index > 0:
            pyplot.text(index_1 + 0.03, index_2 - 0.24, "#" + str(round_index),
                        va="center", ha="center", fontsize=8)
            if round_index != 5:
                pyplot.scatter([index_1 + 0.04], [index_2], s=10, fc=gradient_color, ec="k", lw=0.75, zorder=4)
                values = array([[0.04, 0.05, 0.08], [0.00, -0.03, -0.03]])
                values[0] += index_1
                values[1] = index_2 - values[1]
                pyplot.plot(values[0], values[1], lw=0.75, color="k", zorder=3)
                pyplot.annotate("", xy=(index_1 + 0.09, index_2 - 0.01), xytext=(index_1 + 0.08, index_2 + 0.03),
                                arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
                pyplot.text(index_1 + 0.065, index_2 + 0.060, "1uL", va="center", ha="center", fontsize=8)
        else:
            pyplot.text(index_1 + 0.03, index_2 - 0.24, "#" + str(round_index),
                        va="center", ha="center", fontsize=8)
            pyplot.scatter([index_1 + 0.04], [index_2 - 0.05], s=10, fc=gradient_color, ec="k", lw=0.75, zorder=4)
            values = array([[0.04, 0.05, 0.08, 0.09], [0.05, 0.02, 0.02, 0.05]])
            values[0] += index_1
            values[1] = index_2 - values[1]
            pyplot.plot(values[0], values[1], lw=0.75, color="k", zorder=3)
            pyplot.annotate("", xy=(index_1 + 0.09, index_2 - 0.10), xytext=(index_1 + 0.09, index_2 - 0.05),
                            arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
            pyplot.text(index_1 + 0.065, index_2 + 0.010, "1uL", va="center", ha="center", fontsize=8)

    pyplot.hlines(0.420, 0.660, 0.980, lw=2.0, color="k")
    pyplot.text(0.820, 0.350, "obtain library pools by\nhigh-throughput DNA sequencing",
                va="center", ha="center", fontsize=8)

    pyplot.annotate("", xy=(0.820, 0.240), xytext=(0.820, 0.290),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)

    pyplot.text(0.820, 0.170, "retrieve DNA sequences\n(3 technical replicates)",
                va="center", ha="center", fontsize=8)

    pyplot.annotate("", xy=(0.820, 0.050), xytext=(0.820, 0.100),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)

    pyplot.text(0.820, 0.015, "conventional pipeline and SPIDER-WEB", va="center", ha="center", fontsize=8)

    pyplot.xlabel("experimental validation", fontsize=9)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.xlim(0.01, 1.00)
    pyplot.ylim(-0.01, 1.00)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["bottom"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["left"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[:3, 8:])
    results = shown_data["b"]
    for index, (x_value, y_value) in enumerate(results):
        # noinspection PyUnresolvedReferences
        y_value = y_value / max(y_value) * 0.7
        for x, y in zip(x_value, y_value):
            if index < 3:
                pyplot.fill_between([x - 0.5, x + 0.5], index + 0.1, index + 0.1 + y, fc="linen", ec="k",
                                    lw=0.75, zorder=2)
            else:
                pyplot.fill_between([x - 0.5, x + 0.5], index + 0.1, index + 0.1 + y, fc="azure", ec="k",
                                    lw=0.75, zorder=2)

    # noinspection PyTypeChecker
    for index in arange(5) + 1:
        pyplot.hlines(index, 30, 810, lw=0.75, color="k")
        pyplot.vlines(arange(40, 81, 10), index, index - 0.1, lw=0.75, color="k")

    pyplot.xlabel("sequence similarity by edit distance", fontsize=9)
    pyplot.ylabel("comparison between replicates", fontsize=9)
    pyplot.xticks(arange(30, 81, 10), arange(30, 81, 10), fontsize=9)
    pyplot.yticks(arange(6) + 0.5, ["1-1", "2-2", "3-3", "1-2", "1-3", "2-3"], fontsize=9)
    pyplot.xlim(30, 80)
    pyplot.ylim(0, 6)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[3:6, :4])
    results = shown_data["c"]

    pyplot.text(0.00, 6.20, "threshold of SPIDER-WEB", va="bottom", ha="left", fontsize=8)
    pyplot.hlines(6, -0.1, 6.1, lw=0.75, ls="--", color="grey", zorder=0)
    mapping = {0: "0001-1000", 1: "1001-2000", 2: "2001-3000"}
    for index, (values, color, bias) in enumerate(zip(results, ["#E0A698", "#98B7E6", "#7CBBC6"], [-0.3, 0.0, 0.3])):
        for location in range(6):
            value = results[(index, location)]
            if location > 0:
                pyplot.fill_between([location + 0.5 + bias - 0.1, location + 0.5 + bias + 0.1],
                                    0, value, fc=color, lw=0, zorder=2)
            else:
                label = "pool " + str(index + 1) + ": " + mapping[index]
                pyplot.fill_between([location + 0.5 + bias - 0.1, location + 0.5 + bias + 0.1],
                                    0, value, fc=color, lw=0, zorder=2, label=label)

    pyplot.legend(loc="upper left", fontsize=8, framealpha=1, title="replicates", title_fontsize=8)
    pyplot.xlabel("error-prone round in the experiment", fontsize=9)
    pyplot.ylabel("average number of edit errors", fontsize=9)
    pyplot.xticks(arange(6) + 0.5, ["#" + str(v) for v in arange(6)], fontsize=9)
    pyplot.yticks(arange(0, 15, 2), ["0", "2", "4", "6", "8", "10", "12", "14"], fontsize=9)
    pyplot.xlim(-0.1, 6.1)
    pyplot.ylim(-0.4, 14.4)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[3:6, 4:])
    results = shown_data["d"]

    pyplot.hlines(arange(8), -0.4, 34.4, lw=0.75, ls="--", color="gray", zorder=1)
    for index, (color, left, right) in enumerate(zip(["#E0A698", "#98B7E6", "#7CBBC6"], [0, 12, 24], [10, 22, 34])):
        pyplot.fill_between([left, right], -0.4, 7.4, fc=color, lw=0, alpha=0.3, zorder=2)
        locations = linspace(left, right, 1000)
        copies, norm_indices, more_indices, less_indices = results[index]
        if index > 0:
            pyplot.scatter(locations[norm_indices], copies[norm_indices],
                           ec="k", color="#EFEFEF", lw=0.1, s=5, zorder=3)
            if len(more_indices) > 0:
                pyplot.scatter(locations[more_indices], copies[more_indices],
                               ec="k", fc="#71C6D4", lw=0.1, s=10, zorder=4)
            if len(less_indices) > 0:
                pyplot.scatter(locations[less_indices], copies[less_indices],
                               ec="k", color="#EEA1B4", lw=0.1, s=10, zorder=5)
        else:
            pyplot.scatter(locations[norm_indices], copies[norm_indices],
                           ec="k", color="#EFEFEF", lw=0.1, s=5, zorder=3, label="normal")
            if len(more_indices) > 0:
                pyplot.scatter(locations[more_indices], copies[more_indices],
                               ec="k", fc="#71C6D4", lw=0.1, s=10, zorder=4, label="overloaded (> 2.0 average)")
            if len(less_indices) > 0:
                pyplot.scatter(locations[less_indices], copies[less_indices],
                               ec="k", color="#EEA1B4", lw=0.1, s=10, zorder=5, label="underloaded (< 0.5 average)")

    pyplot.legend(loc="lower left", fontsize=8, framealpha=1, title="copy type", title_fontsize=8)
    pyplot.xlabel("sequence index within different experimental replicates (pools)", fontsize=9)
    pyplot.ylabel("copy number of DNA sequences (#0)", fontsize=9)
    labels = [r"pool 1 (0001$\rightarrow$1000)", r"pool 2 (1001$\rightarrow$2000)", r"pool 3 (2001$\rightarrow$3000)"]
    pyplot.xticks([5, 17, 29], labels, fontsize=9)
    pyplot.yticks(arange(8), ["1E+" + str(v) for v in arange(8)], fontsize=9)
    pyplot.xlim(-0.4, 34.4)
    pyplot.ylim(-0.2, 7.2)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    mesh = None
    for index, label in enumerate(["e", "f", "g"]):
        # noinspection PyTypeChecker
        pyplot.subplot(grid[6:8, index * 4: (index + 1) * 4])
        results = shown_data[label]
        pyplot.title("pool " + str(index + 1) + " (conventional pipeline)", fontsize=9)

        mesh = pyplot.pcolormesh(arange(12), arange(7), results, cmap="plasma", vmin=0, vmax=1)

        pyplot.xlabel("sequencing depth (X)", fontsize=9)
        pyplot.ylabel("error-prone round", fontsize=9)
        pyplot.xticks(arange(0, 11) + 0.5, arange(0, 101, 10), fontsize=9)
        pyplot.yticks(arange(6) + 0.5, ["#" + str(v) for v in range(6)], fontsize=9)
        pyplot.xlim(0, 11)
        pyplot.ylim(0, 6)

    for index, label in enumerate(["h", "i", "j"]):
        # noinspection PyTypeChecker
        pyplot.subplot(grid[8:10, index * 4: (index + 1) * 4])
        results = shown_data[label]
        pyplot.title("pool " + str(index + 1) + " (SPIDER-WEB)", fontsize=9)

        mesh = pyplot.pcolormesh(arange(12), arange(7), results, cmap="plasma", vmin=0, vmax=1)

        pyplot.xlabel("sequencing depth (X)", fontsize=9)
        pyplot.ylabel("error-prone round", fontsize=9)
        pyplot.xticks(arange(0, 11) + 0.5, arange(0, 101, 10), fontsize=9)
        pyplot.yticks(arange(6) + 0.5, ["#" + str(v) for v in range(6)], fontsize=9)
        pyplot.xlim(0, 11)
        pyplot.ylim(0, 6)

    figure.align_labels()

    # noinspection PyTypeChecker
    cax = figure.add_axes([0.056, 0.050, 0.922, 0.010])
    # noinspection PyTypeChecker
    bar = figure.colorbar(mesh, cax=cax, location="bottom")
    bar.set_label("direct retrieval rate (average value of 3 technical replicates)", fontsize=9)
    bar.set_ticks(linspace(0, 1, 21))
    bar.set_ticklabels([str(int(v * 100)) + "%" for v in linspace(0, 1, 21)], fontsize=9)

    figure.text(0.021, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.679, 0.990, "b", va="center", ha="center", fontsize=12)
    figure.text(0.021, 0.705, "c", va="center", ha="center", fontsize=12)
    figure.text(0.336, 0.705, "d", va="center", ha="center", fontsize=12)
    figure.text(0.021, 0.430, "e", va="center", ha="center", fontsize=12)
    figure.text(0.336, 0.430, "f", va="center", ha="center", fontsize=12)
    figure.text(0.679, 0.430, "g", va="center", ha="center", fontsize=12)
    figure.text(0.021, 0.240, "h", va="center", ha="center", fontsize=12)
    figure.text(0.336, 0.240, "i", va="center", ha="center", fontsize=12)
    figure.text(0.679, 0.240, "j", va="center", ha="center", fontsize=12)

    pyplot.savefig("./fig/main-03.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main_f04():
    """
    Create Figure 4 in the main text.

    TITLE: Comparable evaluation of retrieval runtime.
    """
    shown_data = load_data(file_path=pkg_path + "main-04.pkl")

    figure = pyplot.figure(figsize=(10, 10), tight_layout=True)
    grid = pyplot.GridSpec(10, 3)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[:4, :])
    results = shown_data["a"]

    pyplot.scatter(results[0][0], results[0][1], ec="k", fc="#E0A698", lw=0.75, label="1", zorder=3)
    pyplot.scatter(results[1][0], results[1][1], ec="k", fc="#98B7E6", lw=0.75, label="2", zorder=2)
    pyplot.scatter(results[2][0], results[2][1], ec="k", fc="#7CBBC6", lw=0.75, label="3", zorder=1)

    pyplot.plot([0, 80.0], [0, 0.8], color="k", lw=0.75, ls="--", zorder=1)
    pyplot.plot([0, 800.0], [0, 0.8], color="k", lw=0.75, ls="--", zorder=1)
    pyplot.plot([0, 2000.0], [0, 0.2], color="k", lw=0.75, ls="--", zorder=1)

    pyplot.text(95.0, 0.78, "100-fold\nspeed-up", va="center", ha="center",
                bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.2"), fontsize=8)
    pyplot.text(750.0, 0.78, "1,000-fold\nspeed-up", va="center", ha="center",
                bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.3"), fontsize=8)
    pyplot.text(1750.0, 0.18, "10,000-fold\nspeed-up", va="center", ha="center",
                bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.3"), fontsize=8)

    pyplot.legend(loc="upper right", fontsize=8, framealpha=1, title="pool", title_fontsize=8)
    pyplot.xlabel("retrieval runtime of conventional pipeline (seconds)", fontsize=9)
    pyplot.ylabel("retrieval runtime of SPIDER-WEB (seconds)", fontsize=9)
    pyplot.xticks(arange(0, 2001, 100), arange(0, 2001, 100), fontsize=9)
    pyplot.yticks(linspace(0.0, 0.8, 9), ["%.2f" % v for v in linspace(0.0, 0.8, 9)], fontsize=9)
    pyplot.xlim(-24, 2024)
    pyplot.ylim(-0.024, 0.824)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[4:7, :2])

    pyplot.text(0.40, 0.95, "expected sequences (length = 150 nucleotides)", va="center", ha="center", fontsize=8)
    pyplot.vlines(0.40, 0.88, 0.91, lw=0.75, color="k")
    pyplot.plot([0.20, 0.20, 0.60, 0.60], [0.85, 0.88, 0.88, 0.85], lw=0.75, color="k")
    locations = linspace(0.60, 0.80, 5)
    for index, location in enumerate(locations):
        if index != 1:
            pyplot.fill_between([0.12, 0.28], location - 0.015, location + 0.015, ec="k", fc="#EFEFEF", lw=0.75)
            pyplot.fill_between([0.52, 0.68], location - 0.015, location + 0.015, ec="k", fc="#EFEFEF", lw=0.75)
        else:
            pyplot.scatter([0.20, 0.20, 0.20], [location - 0.015, location, location + 0.015], s=1, color="k")
            pyplot.scatter([0.60, 0.60, 0.60], [location - 0.015, location, location + 0.015], s=1, color="k")
    pyplot.plot([0.29, 0.30, 0.30, 0.29], [0.585, 0.585, 0.815, 0.815], lw=0.75, color="k")
    pyplot.plot([0.51, 0.50, 0.50, 0.51], [0.585, 0.585, 0.815, 0.815], lw=0.75, color="k")
    pyplot.hlines(0.70, 0.30, 0.50, lw=0.75, ls="--", color="k", zorder=1)
    pyplot.text(0.40, 0.70, "diversity = 1E3", va="center", ha="center",
                bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.2"), fontsize=8)
    pyplot.annotate("", xy=(0.20, 0.55), xytext=(0.20, 0.49),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.annotate("", xy=(0.60, 0.55), xytext=(0.60, 0.49),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.fill_between([0.02, 0.38], 0.33, 0.47, color="#D6E9E6")
    pyplot.text(0.20, 0.40, "error rates and copy distribution\ndriven by error-prone PCR technology",
                va="center", ha="center", fontsize=8)
    pyplot.fill_between([0.42, 0.78], 0.33, 0.47, color="#dae5ff")
    pyplot.text(0.60, 0.40, "given error rates (randomly introduced)\ngiven copies (same to all sequences)",
                va="center", ha="center", fontsize=8)
    pyplot.annotate("", xy=(0.20, 0.31), xytext=(0.20, 0.25),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.annotate("", xy=(0.60, 0.31), xytext=(0.60, 0.25),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.text(0.20, 0.19, "runtimes of two methods\n(wet lab)",
                va="center", ha="center", fontsize=8)
    pyplot.text(0.60, 0.19, "runtimes of two methods\n(simulation)",
                va="center", ha="center", fontsize=8)
    pyplot.vlines([0.20, 0.60], 0.07, 0.12, lw=0.75, color="k")
    pyplot.text(0.400, 0.07, "equivalence comparison\n(Fig. 4c)",
                va="center", ha="center", color="#C00000", fontsize=8)
    pyplot.annotate("", xy=(0.20, 0.07), xytext=(0.28, 0.07),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.annotate("", xy=(0.60, 0.07), xytext=(0.52, 0.07),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.annotate("", xy=(0.69, 0.70), xytext=(0.79, 0.70),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.text(0.74, 0.77, "intervene\ndiversity", va="center", ha="center", fontsize=8)
    pyplot.hlines(0.60, 0.80, 1.00, color="k", lw=1.5, zorder=1)
    pyplot.text(0.90, 0.55, "diversity", va="center", ha="center", fontsize=8)
    x_locations, y_locations = linspace(0.81, 0.99, 11), linspace(0.60, 0.80, 6)
    for index in range(5):
        x_center, y_bound = x_locations[index * 2 + 1], y_locations[index + 1]
        pyplot.fill_between([x_center - 0.01, x_center + 0.01], 0.60, y_bound, color="gray", zorder=0)
        pyplot.text(x_center, y_bound + 0.03, "1E" + str(index + 1), va="center", ha="center", fontsize=8)

    pyplot.annotate("", xy=(0.90, 0.51), xytext=(0.90, 0.45),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.text(0.90, 0.38, "runtimes of two methods\n(Fig. 4d, 4e)",
                va="center", ha="center", fontsize=8)
    pyplot.annotate("", xy=(0.90, 0.31), xytext=(0.90, 0.14),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.text(0.91, 0.23, "fit results", va="center", ha="left", fontsize=8)
    pyplot.text(0.90, 0.07, "acceleration factor in\nvarious scales (Fig. 4f)",
                va="center", ha="center", color="#C00000", fontsize=8)

    pyplot.xlabel("scalable numerical simulation", fontsize=9)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.xlim(0.02, 1.02)
    pyplot.ylim(0.00, 1.00)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["bottom"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["left"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[4:7, 2])
    results = shown_data["c"]
    distance_1, distance_2, (mesh_1_x, mesh_1_y, distribution_1), (mesh_2_x, mesh_2_y, distribution_2) = results

    levels = linspace(0.0000, 0.0012, 7)
    pyplot.contour(mesh_1_x, mesh_1_y, distribution_1, levels=levels, cmap="viridis_r",
                   linewidths=0.75, linestyles=":", zorder=1)
    levels = linspace(0.0000, 0.0012, 13)
    pyplot.contour(mesh_2_x, mesh_2_y, distribution_2, levels=levels, cmap="viridis_r",
                   linewidths=0.75, zorder=2)

    pyplot.plot([0, 60.0], [0, 0.6], color="k", lw=0.75, ls="--", zorder=0)
    pyplot.plot([0, 1050.0], [0, 1.05], color="k", lw=0.75, ls="--", zorder=0)
    pyplot.plot([0, 900.0], [0, 0.09], color="k", lw=0.75, ls="--", zorder=0)
    pyplot.text(60.0, 0.60, "1E+2", va="center", ha="center",
                bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.2"), fontsize=8)
    pyplot.text(1010, 1.05, "1E+3", va="center", ha="center",
                bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.2"), fontsize=8)
    pyplot.text(900, 0.09, "1E+4", va="center", ha="center",
                bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.2"), fontsize=8)

    pyplot.text(0, 1.50, "wasserstein", va="center", ha="left", fontsize=8)
    # noinspection SpellCheckingInspection
    pyplot.text(0, 1.35, "kullback-leibler", va="center", ha="left", fontsize=8)
    pyplot.text(515, 1.50, "= %.4f" % distance_1, va="center", ha="left", fontsize=8)
    pyplot.text(515, 1.35, "= %.4f" % distance_2, va="center", ha="left", fontsize=8)

    pyplot.text(1325, 1.50, "simulation", va="center", ha="center", fontsize=9)
    pyplot.fill_between([1300, 1350], 1.20, 1.40, ec="k", fc="none", lw=1)
    pyplot.text(1325, 1.05, "wet lab", va="center", ha="center", fontsize=9)
    pyplot.fill_between([1300, 1350], 0.75, 0.95, ec="k", fc="none", ls=":", lw=1)
    locations, colors = linspace(0.00, 0.50, 21), pyplot.get_cmap("viridis_r")(linspace(0, 1, 20))
    for former, latter, color in zip(locations[:-1], locations[1:], colors):
        pyplot.fill_between([1300, 1350], former, latter, color=color, lw=0, zorder=0)
    pyplot.fill_between([1300, 1350], 0.00, 0.50, ec="k", fc="none", lw=0.75, zorder=1)
    pyplot.hlines(linspace(0.00, 0.50, 4), 1350, 1380, lw=0.75, color="k")
    for location, label in zip(linspace(0.00, 0.50, 4), linspace(0.0000, 0.0012, 4)):
        pyplot.text(1400, location, ("%.2f" % (label * 100)) + "%", va="center", ha="left", fontsize=8)
    pyplot.text(1325, 0.60, "density", va="center", ha="center", fontsize=9)
    pyplot.xlabel("retrieval runtime of conventional pipeline", fontsize=9)
    pyplot.ylabel("retrieval runtime of SPIDER-WEB", fontsize=9)
    pyplot.xticks(arange(0, 1501, 300), arange(0, 1501, 300), fontsize=9)
    pyplot.yticks(linspace(0, 1.5, 6), ["%.2f" % v for v in linspace(0, 1.5, 6)], fontsize=9)
    pyplot.xlim(-75, 1575)
    pyplot.ylim(-0.075, 1.575)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[7:, 0])
    results = shown_data["d"]
    values_1, fitted_values_1, fit_info_our_1, fit_info_cov_1 = results

    pyplot.scatter(values_1[0], values_1[2], marker="D",
                   ec="k", fc="#9EC1D4", lw=0.75, label="conventional", zorder=2)
    pyplot.scatter(values_1[0], values_1[1], marker="D",
                   ec="k", fc="#E87651", lw=0.75, label="SPIDER-WEB", zorder=2)

    pyplot.plot(fitted_values_1[0], fitted_values_1[2], color="#9EC1D4", lw=3, alpha=0.5, zorder=0)
    pyplot.plot(fitted_values_1[0], fitted_values_1[1], color="#E87651", lw=3, alpha=0.5, zorder=0)
    pyplot.text(3.8, -3.0, "R^2 of fitted curve = %.4f" % fit_info_our_1, va="center", ha="center", fontsize=9)
    pyplot.text(2.5, 4.0, "R^2 of fitted curve = %.4f" % fit_info_cov_1, va="center", ha="center", fontsize=9)

    pyplot.legend(loc="lower right", fontsize=8, framealpha=1)
    pyplot.title("minimum error (1 out of 150)", fontsize=9)
    pyplot.xlabel("sequence diversity", fontsize=9)
    pyplot.ylabel("runtime (seconds)", fontsize=9)
    pyplot.xticks(arange(5) + 1, ["1E+1", "1E+2", "1E+3", "1E+4", "1E+5"], fontsize=9)
    labels = ["1E\N{MINUS SIGN}6", "1E\N{MINUS SIGN}4", "1E\N{MINUS SIGN}2", "1E+0", "1E+2", "1E+4", "1E+6"]
    pyplot.yticks(arange(-6, 7, 2), labels, fontsize=9)
    pyplot.xlim(0.6, 5.4)
    pyplot.ylim(-6, 6)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[7:, 1])
    results = shown_data["e"]
    values_6, fitted_values_6, fit_info_our_6, fit_info_cov_6 = results
    pyplot.scatter(values_6[0], values_6[2], marker="D",
                   ec="k", fc="#9EC1D4", lw=0.75, label="conventional", zorder=1)
    pyplot.scatter(values_6[0], values_6[1], marker="D",
                   ec="k", fc="#E87651", lw=0.75, label="SPIDER-WEB", zorder=1)
    pyplot.text(3.5, -3.0, "R^2 of fitted curve = %.4f" % fit_info_our_6, va="center", ha="center", fontsize=9)
    pyplot.text(2.5, 5.0, "R^2 of fitted curve = %.4f" % fit_info_cov_6, va="center", ha="center", fontsize=9)

    pyplot.plot(fitted_values_6[0], fitted_values_6[2], color="#9EC1D4", lw=3, alpha=0.5, zorder=0)
    pyplot.plot(fitted_values_6[0], fitted_values_6[1], color="#E87651", lw=3, alpha=0.5, zorder=0)

    pyplot.legend(loc="lower right", fontsize=8, framealpha=1)
    pyplot.title("maximum errors (6 out of 150)", fontsize=9)
    pyplot.xlabel("sequence diversity", fontsize=9)
    pyplot.ylabel("runtime (seconds)", fontsize=9)
    pyplot.xticks(arange(5) + 1, ["1E+1", "1E+2", "1E+3", "1E+4", "1E+5"], fontsize=9)
    labels = ["1E\N{MINUS SIGN}6", "1E\N{MINUS SIGN}4", "1E\N{MINUS SIGN}2", "1E+0", "1E+2", "1E+4", "1E+6"]
    pyplot.yticks(arange(-6, 7, 2), labels, fontsize=9)
    pyplot.xlim(0.6, 5.4)
    pyplot.ylim(-6, 6)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[7:, 2])
    results = shown_data["f"]

    pyplot.vlines(arange(5)[1:-1], 0, 30, lw=0.75, ls="--", color="grey", zorder=1)
    pyplot.hlines(arange(0, 31, 5)[1:-1], 0, 4, lw=0.75, ls="--", color="grey", zorder=1)

    pyplot.fill_between(results[0] - 2, results[1], results[2], fc="silver", lw=0, zorder=0)
    pyplot.plot(results[0] - 2, results[1], color="k", lw=1.5, zorder=2, label="upper bound")
    pyplot.plot(results[0] - 2, results[2], color="k", lw=1.5, ls=":", zorder=2, label="lower bound")

    pyplot.legend(loc="lower right", fontsize=8, framealpha=1)
    pyplot.xlabel("amount of data", fontsize=9)
    pyplot.ylabel("acceleration", fontsize=9)
    pyplot.xticks(arange(5), ["MB", "GB", "TB", "PB", "EB"], fontsize=9)
    pyplot.yticks(arange(0, 31, 5), ["1E+" + str(v).zfill(2) for v in arange(0, 31, 5)], fontsize=9)
    pyplot.xlim(0, 4)
    pyplot.ylim(0, 30)

    figure.align_labels()
    figure.text(0.021, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.021, 0.596, "b", va="center", ha="center", fontsize=12)
    figure.text(0.670, 0.596, "c", va="center", ha="center", fontsize=12)
    figure.text(0.021, 0.296, "d", va="center", ha="center", fontsize=12)
    figure.text(0.350, 0.296, "e", va="center", ha="center", fontsize=12)
    figure.text(0.670, 0.296, "f", va="center", ha="center", fontsize=12)

    pyplot.savefig("./fig/main-04.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main_f05():
    """
    Create Figure 5 in the main text.

    TITLE: Approximation of data retrieval efficiency.
    """
    shown_data = load_data(file_path=pkg_path + "main-05.pkl")

    figure = pyplot.figure(figsize=(10, 9), tight_layout=True)
    grid = pyplot.GridSpec(3, 2)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[0, :])

    pyplot.fill_between([0.015, 0.045], 0.920, 0.980, ec="k", fc="#D1EBF4", lw=0.75)
    pyplot.text(0.050, 0.950, "given", va="center", ha="left", fontsize=9)
    pyplot.fill_between([0.015, 0.045], 0.830, 0.890, ec="k", fc="#E1EFE2", lw=0.75)
    pyplot.text(0.050, 0.860, "collected", va="center", ha="left", fontsize=9)
    pyplot.fill_between([0.015, 0.045], 0.740, 0.800, ec="k", fc="#FAE5B7", lw=0.75)
    pyplot.text(0.050, 0.770, "fitted", va="center", ha="left", fontsize=9)
    pyplot.fill_between([0.015, 0.045], 0.650, 0.710, ec="k", fc="#F5D3D0", lw=0.75)
    pyplot.text(0.050, 0.680, "deduced", va="center", ha="left", fontsize=9)

    x_centers = linspace(0.00, 1.00, 11)
    y_centers = linspace(0.00, 1.00, 7)
    x_length, y_length = 0.085, 0.080
    locations = [[1, 3], [3, 1], [3, 5], [6, 5], [6, 1], [3, 3], [5, 3], [8, 1]]
    colors = ["#D1EBF4", "#FAE5B7", "#D1EBF4", "#E1EFE2", "#E1EFE2", "#FAE5B7", "#F5D3D0", "#D1EBF4"]
    labels = ["amount of data\n" + r"( given; denoted as $A$ )",
              "sequence diversity\n" + r"( Method; denoted as $D$ )",
              "sequence error rate\n" + "( given )",
              "correcting speed\n" + "( Fig. 5b; denoted as $S_1$)",
              "decoding speed\n" + "( Fig. 5c; denoted as $S_2$)",
              "direct retrieval rate\n" + r"( Fig. 2e )",
              "required sequence copies\n" + r"( Fig. 2f; denoted as $C$ )",
              "number of threads\n" + r"( given; denoted as $N$ )"]
    for (x_index, y_index), color, label in zip(locations, colors, labels):
        # noinspection PyTypeChecker
        pyplot.fill_between([x_centers[x_index] - x_length, x_centers[x_index] + x_length],
                            y_centers[y_index] - y_length, y_centers[y_index] + y_length,
                            ec="k", fc=color, lw=0.75, zorder=2)
        # noinspection PyTypeChecker
        pyplot.text(x_centers[x_index], y_centers[y_index], label, va="center", ha="center", fontsize=8)
    pyplot.fill_between([x_centers[7] - x_length, x_centers[7] + x_length], y_centers[3] - 0.12, y_centers[3] + 0.12,
                        ec="k", fc="#F5D3D0", lw=0.75, zorder=2)
    pyplot.text(x_centers[7], y_centers[3] + 0.04, "runtime " + r"( denoted as $R$ )" + "\n",
                va="center", ha="center", fontsize=8)
    pyplot.text(x_centers[7], y_centers[3] - 0.04,
                r"$R=\frac{D \times C}{N \times S_1}+\frac{D}{N \times S_2}$", va="center", ha="center", fontsize=12)
    pyplot.fill_between([x_centers[9] - x_length, x_centers[9] + x_length], y_centers[3] - 0.12, y_centers[3] + 0.12,
                        ec="k", fc="#F5D3D0", lw=0.75, zorder=2)
    pyplot.text(x_centers[9], y_centers[3] + 0.04, "efficiency " + r"( denoted as $E$ )" + "\n",
                va="center", ha="center", fontsize=8)
    pyplot.text(x_centers[9], y_centers[3] - 0.04,
                r"$E = \frac{A}{R}$", va="center", ha="center", fontsize=12)

    pyplot.plot([x_centers[1], x_centers[2], x_centers[2]],
                [y_centers[3], y_centers[3], y_centers[1]], lw=0.75, color="k", zorder=0)
    pyplot.annotate("", xy=(x_centers[2], y_centers[1]), xytext=(x_centers[2] + 0.015, y_centers[1]),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)

    pyplot.plot([x_centers[1], x_centers[2], x_centers[2], x_centers[9]],
                [y_centers[3], y_centers[3], y_centers[6], y_centers[6]], lw=0.75, color="k", zorder=0)
    pyplot.annotate("", xy=(x_centers[9], y_centers[6]), xytext=(x_centers[9], y_centers[3] + 0.12),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.annotate("", xy=(x_centers[4] - 0.015, y_centers[5]), xytext=(x_centers[5] + 0.015, y_centers[5]),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.annotate("", xy=(x_centers[6] - 0.015, y_centers[3]), xytext=(x_centers[6] + 0.015, y_centers[3]),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.annotate("", xy=(x_centers[3], y_centers[5] - y_length), xytext=(x_centers[3], y_centers[3] + y_length),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.annotate("", xy=(x_centers[4] - 0.015, y_centers[3]), xytext=(x_centers[4] + 0.015, y_centers[3]),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.hlines(y_centers[5], x_centers[5] + x_length, x_centers[7], lw=0.75, color="k", zorder=0)
    pyplot.annotate("", xy=(x_centers[7], y_centers[5]), xytext=(x_centers[7], y_centers[3] + 0.12),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.annotate("", xy=(x_centers[7] - 0.5 * x_length, y_centers[1] + y_length),
                    xytext=(x_centers[7] - 0.5 * x_length, y_centers[3] - 0.12),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.annotate("", xy=(x_centers[7] + 0.5 * x_length, y_centers[1] + y_length),
                    xytext=(x_centers[7] + 0.5 * x_length, y_centers[3] - 0.12),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.annotate("", xy=(x_centers[7], y_centers[0]), xytext=(x_centers[7], y_centers[3] - 0.12),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.plot([x_centers[3], x_centers[3], x_centers[7]],
                [y_centers[1] - y_length, y_centers[0], y_centers[0]], lw=0.75, color="k", zorder=0)
    pyplot.annotate("", xy=(x_centers[3], y_centers[0]), xytext=(x_centers[3], y_centers[3] - y_length),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.annotate("", xy=(x_centers[8] - 0.015, y_centers[3]), xytext=(x_centers[8] + 0.015, y_centers[3]),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)

    pyplot.xlabel("coarse-grained approximation workflow of data retrieval efficiency", fontsize=9)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.xlim(0.013, 0.987)
    pyplot.ylim(-0.01, 1.01)
    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["bottom"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["left"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[1, 0])
    results = shown_data["b"]

    labels = ["high-throughput sequencing\n(by SBS technology)",
              "single-molecule sequencing\n(by SMRT technology)",
              "single-molecule sequencing\n(by nanopore sequencing technology)"]
    for location, (label, value) in enumerate(zip(labels, results[0])):
        pyplot.vlines(log10(value), -0.5, 7.5, lw=1.5, color="#AAAAAA", zorder=0)
        # noinspection PyTypeChecker
        pyplot.text(log10(value) - 0.6, 3.5, label, bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.2"),
                    va="center", ha="center", fontsize=8, rotation=90)

    # noinspection PyTypeChecker
    pyplot.annotate("", xy=(log10(results[0][-1]), 7.0), xytext=(log10(results[1][-1]), 7.0),
                    arrowprops=dict(arrowstyle="<|-|>", color="k", shrinkA=3.5, shrinkB=1.0, lw=0.75), zorder=1)

    # noinspection PyTypeChecker
    pyplot.text((log10(results[0][-1]) + log10(results[1][-1])) / 2.0, 7.00 - 0.05,
                "{:,}X".format(int(results[1][-1] / results[0][-1])),
                bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.2"), va="center", ha="center", fontsize=9)
    colors = pyplot.get_cmap("RdYlBu_r")(linspace(0.2, 0.8, 8))
    for index, (value, color) in enumerate(zip(results[1], colors)):
        pyplot.scatter(log10(value), index, ec="k", fc=color, s=50, lw=0.75, zorder=1)
        # noinspection PyTypeChecker
        pyplot.text(log10(value) + 0.40, index - 0.05,
                    ("%.1e" % value).replace("e", "E").replace("+0", "+"),
                    va="center", ha="left", fontsize=9)
    pyplot.xlabel("correcting speed per thread (nucleotides / seconds)", fontsize=9)
    pyplot.ylabel("edit error rate", fontsize=9)
    pyplot.xticks(linspace(-5, 10, 4), ["1E\N{MINUS SIGN}5", "1E+0", "1E+5", "1E+10"], fontsize=9)
    pyplot.yticks(arange(0, 8), [("%.1f" % (v / 2.0)) + "%" for v in arange(1, 9)], fontsize=9)
    pyplot.xlim(-5.0, 10.0)
    pyplot.ylim(-0.5, 7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[1, 1])
    results = shown_data["c"]

    random.seed(2023)
    for index, speeds in enumerate(results):
        biases = random.random(size=(len(speeds),))
        biases /= min(biases)
        biases /= max(biases)
        biases = biases * 0.8 - 0.4
        pyplot.scatter(speeds / 1e7, biases + index + 1.0, s=10,
                       cmap="viridis", c=speeds / 1e7, vmin=0.4, vmax=2.5, zorder=2)
        pyplot.vlines([min(speeds / 1e7) - 0.03, max(speeds / 1e7) + 0.03], index + 0.6, index + 1.4,
                      lw=1.0, color="k", zorder=1)
        pyplot.hlines(index + 1.0, min(speeds / 1e7) - 0.03, max(speeds / 1e7) + 0.03,
                      lw=0.75, ls="--", color="k", zorder=1)
        # noinspection PyTypeChecker
        pyplot.text(min(speeds / 1e7) - 0.06, index + 0.95, "%d" % min(speeds), va="center", ha="right", fontsize=8)
        # noinspection PyTypeChecker
        pyplot.text(max(speeds / 1e7) + 0.06, index + 0.95, "%d" % max(speeds), va="center", ha="left", fontsize=8)

    pyplot.xlabel("decoding speed per thread (nucleotides / seconds)", fontsize=9)
    pyplot.ylabel("sequence number", fontsize=9)
    pyplot.xticks(linspace(0, 3, 4), ["0", "1E+7", "2E+7", "3E+7"], fontsize=9)
    pyplot.yticks(arange(1, 7), ["1E+" + str(v) for v in arange(1, 7)], fontsize=9)
    pyplot.xlim(0.0, 3.0)
    pyplot.ylim(0.4, 6.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2, 0])
    results = shown_data["d"]
    (x_values, y_values, matrix), (plotted_x_values, plotted_y_values), (range_info, points) = results

    matrix[matrix < 0] = nan
    pyplot.pcolormesh(x_values, y_values, log10(matrix).T, cmap="cool", vmin=0, vmax=4, shading="gouraud")

    x_locations = linspace(1.1, 1.7, 6)
    pyplot.text(1.4, 5.5, "thread number", ha="center", va="center", fontsize=9)
    pyplot.vlines(1.1, 4.90, 5.15, color="k", lw=0.75)
    pyplot.vlines(linspace(x_locations[1], 1.7, 5), 4.90, 5.05, color="k", lw=0.75)
    for location, label in zip(x_locations, ["N.A.", "1E0", "1E1", "1E2", "1E3", "1E4"]):
        pyplot.text(location, 4.70, label, ha="center", va="center", fontsize=8)
    pyplot.fill_between([1.1, x_locations[1], 1.7], [5.15, 5.05, 5.05], [5.15, 5.25, 5.25],
                        ec="k", fc="none", lw=0.75, zorder=2)
    pyplot.fill_between([1.1, x_locations[1]], [5.15, 5.05], [5.15, 5.25], fc="w", lw=0, zorder=1)
    colors, locations = pyplot.get_cmap("cool")(linspace(0, 1)), linspace(x_locations[1], 1.7, 51)
    for color, former, latter in zip(colors, locations[:-1], locations[1:]):
        pyplot.fill_between([former, latter], 5.05, 5.25, fc=color, lw=0, zorder=1)
    pyplot.plot(plotted_x_values, plotted_y_values, color="k", lw=0.75, ls="--", zorder=2)
    # noinspection PyTypeChecker,PyUnresolvedReferences,SpellCheckingInspection
    pyplot.text(mean(plotted_x_values[where(plotted_y_values == max(plotted_y_values))]), max(plotted_y_values) + 0.6,
                "accomplished\nusing a laptop\n(HHD=1TB, Thrs=16)", va="center", ha="center", fontsize=9)

    pyplot.annotate("", xy=points[0], xytext=(points[0][0] - 0.2, points[0][1] + 0.4),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=3, shrinkB=5, lw=0.75), zorder=1)
    pyplot.text(points[0][0] - 0.2, points[0][1] + 0.4, "%.1f KB/s" % (range_info[0] / 8.0 / 1024.0),
                va="center", ha="right", fontsize=9)
    pyplot.annotate("", xy=points[1], xytext=(points[1][0] - 0.2, points[1][1] - 0.3),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=3, shrinkB=5, lw=0.75), zorder=1)
    pyplot.text(points[1][0] - 0.2, points[1][1] - 0.3, "%.1f KB/s" % (range_info[1] / 8.0 / 1024.0),
                va="center", ha="right", fontsize=9)
    pyplot.annotate("", xy=points[2], xytext=(points[2][0] + 0.2, points[2][1] - 0.4),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=3, shrinkB=5, lw=0.75), zorder=1)
    pyplot.text(points[2][0] + 0.2, points[2][1] - 0.3, "%.1f MB/s" % (range_info[2] / 8.0 / (1024.0 ** 2)),
                va="center", ha="left", fontsize=9)
    pyplot.annotate("", xy=points[3], xytext=(points[3][0] + 0.2, points[3][1] + 0.3),
                    arrowprops=dict(arrowstyle="-|>", color="k", shrinkA=3, shrinkB=5, lw=0.75), zorder=1)
    pyplot.text(points[3][0] + 0.2, points[3][1] + 0.4, "%.1f MB/s" % (range_info[3] / 8.0 / (1024.0 ** 2)),
                va="center", ha="left", fontsize=9)
    pyplot.xlabel("data retrieval efficiency (when edit error rate = 0.5%)", fontsize=9)
    pyplot.ylabel("amount of data", fontsize=9)
    pyplot.xticks(linspace(1, 3, 5),
                  ["1 KB/s", "32 KB/s", "1 MB/s", "32 MB/s", "1 GB/s"], fontsize=9)
    pyplot.yticks(arange(1, 7), ["1 KB", "1 MB", "1 GB", "1 TB", "1 PB", "1 EB"], fontsize=9)
    pyplot.xlim(1, 3)
    pyplot.ylim(1, 6)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[2, 1])
    results = shown_data["e"]

    threshold = 150 / 1024.0
    colors = pyplot.get_cmap("RdYlBu_r")(linspace(0.2, 0.8, 8))
    lower_bounds, upper_bounds = results[:, 0] / 8.0 / (1024.0 ** 2), results[:, 1] / 8.0 / (1024.0 ** 2)
    # noinspection PyTypeChecker
    for location, lower_bound, upper_bound, color in zip(arange(8) + 0.5, lower_bounds, upper_bounds, colors):
        if lower_bound > threshold:
            pyplot.fill_between([location - 0.4, location + 0.4], lower_bound, upper_bound,
                                ec="k", fc=color, lw=0.75, zorder=0)
        else:
            pyplot.fill_between([location - 0.4, location + 0.4], lower_bound, threshold,
                                ec="k", fc="#EFEFEF", lw=0.75, zorder=0)
            pyplot.fill_between([location - 0.4, location + 0.4], threshold, upper_bound,
                                ec="k", fc=color, lw=0.75, zorder=0)
    pyplot.text(0.0, threshold + 0.1, "CD benchmark = 150 KB/s", va="bottom", ha="left", fontsize=8)
    pyplot.hlines(threshold, -0.1, 8.1, lw=0.75, ls="--", color="k", zorder=1)

    discussed_indices = array([0, 1, 3, 6, 7])
    pyplot.hlines([14.97, 13.00, 5.50], 3.5, 7.7, color="k", lw=0.75)
    pyplot.text(4.0, 14.0, "error rate", va="center", ha="center", fontsize=8)
    pyplot.text(5.5, 14.0, "lower bound", va="center", ha="center", fontsize=8)
    pyplot.text(7.0, 14.0, "upper bound", va="center", ha="center", fontsize=8)
    for location, error_info, lower_bound, upper_bound in zip([12.0, 10.5, 9.0, 7.5, 6.0],
                                                              ["0.5%", "1.0%", "2.0%", "3.5%", "4.0%"],
                                                              lower_bounds[discussed_indices],
                                                              upper_bounds[discussed_indices]):
        pyplot.text(4.0, location, error_info, va="center", ha="center", fontsize=8)
        if lower_bound > 1:
            info_1, info_2 = "%.1f MB/s" % lower_bound, "%.1f MB/s" % upper_bound
        elif lower_bound < 1 < upper_bound:
            info_1, info_2 = "%.1f KB/s" % (lower_bound * 1024.0), "%.1f MB/s" % upper_bound
        else:
            info_1, info_2 = "%.1f KB/s" % (lower_bound * 1024.0), "%.1f KB/s" % (upper_bound * 1024.0)
        if lower_bound > threshold:
            color_1, color_2 = "#329845", "#329845"
        elif lower_bound < threshold < upper_bound:
            color_1, color_2 = "#329845", "#329845"
        else:
            color_1, color_2 = "#329845", "#329845"
        pyplot.text(5.5, location, info_1, color=color_1, va="center", ha="center", fontsize=8)
        pyplot.text(7.0, location, info_2, color=color_2, va="center", ha="center", fontsize=8)

    pyplot.xlabel("edit error rate", fontsize=9)
    pyplot.ylabel("data retrieval efficiency per laptop", fontsize=9)
    pyplot.xticks(arange(8) + 0.5, [("%.1f" % (v / 2.0)) + "%" for v in arange(1, 9)], fontsize=9)
    pyplot.yticks(linspace(0, 15, 6), ["%d MB/s" % v for v in linspace(0, 15, 6)], fontsize=9)
    pyplot.xlim(-0.1, 8.1)
    pyplot.ylim(0, 15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    figure.align_labels()
    figure.text(0.021, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.021, 0.672, "b", va="center", ha="center", fontsize=12)
    figure.text(0.508, 0.672, "c", va="center", ha="center", fontsize=12)
    figure.text(0.021, 0.341, "d", va="center", ha="center", fontsize=12)
    figure.text(0.508, 0.341, "e", va="center", ha="center", fontsize=12)

    pyplot.savefig("./fig/main-05.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def main_f06():
    """
    Create Figure 6 in the main text.

    TITLE: Demonstration of real-time non-blocking data retrieval.
    """
    shown_data = load_data(file_path=pkg_path + "main-06.pkl")

    figure = pyplot.figure(figsize=(10, 9), tight_layout=True)
    grid = pyplot.GridSpec(10, 4)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[:3, :])

    frame_data = shown_data["a"]
    locations, colors = linspace(0.025, 0.550, 16), pyplot.get_cmap("binary")(linspace(0, 1, 256))[::-1]

    pyplot.text(0.012, (0.60 + 0.85) / 2.0, "30px", va="center", ha="center", fontsize=7)
    pyplot.text(0.050, 0.530, "40px", va="center", ha="center", fontsize=7)
    pyplot.plot([0.050, 0.050, 0.575, 0.575], [0.900, 0.950, 0.950, 0.900], lw=0.75, ls=":", color="k", zorder=0)
    pyplot.scatter([0.147, 0.478], [0.950, 0.950], marker=">", ec="k", fc="w", lw=0.75, zorder=1)
    pyplot.text(0.3125, 0.950, "playback direction (16 frames)", zorder=2,
                va="center", ha="center", fontsize=8, bbox=dict(fc="w", ec="w", lw=0, boxstyle="round,pad=0.4"))
    for index, location in enumerate(locations):
        pyplot.fill_between([location, location + 0.05], [0.60, 0.55], [0.85, 0.90],
                            ec="k", fc="none", lw=0.75, zorder=32 - index * 2)
        one_frame, x_locations = frame_data[index], linspace(location, location + 0.05, 41)
        for x_index, (x_former, x_latter) in enumerate(zip(x_locations[:-1], x_locations[1:])):
            y_locations_1 = linspace(0.60 - (0.05 / 40.0 * (x_index + 0)), 0.85 + (0.05 / 40.0 * (x_index + 0)), 31)
            y_locations_2 = linspace(0.60 - (0.05 / 40.0 * (x_index + 1)), 0.85 + (0.05 / 40.0 * (x_index + 1)), 31)
            for y_index, (y_former_1, y_former_2, y_latter_1, y_latter_2) in enumerate(zip(y_locations_1[:-1],
                                                                                           y_locations_1[1:],
                                                                                           y_locations_2[:-1],
                                                                                           y_locations_2[1:])):
                pyplot.fill_between([x_former, x_latter], [y_former_1, y_latter_1], [y_former_2, y_latter_2],
                                    ec="none", fc=colors[one_frame[29 - y_index, x_index]], lw=0,
                                    zorder=32 - index * 2 - 1)
    pyplot.text(0.3125, 0.470, "Muybridge race horse gallop (the first recorded motion study)", zorder=2,
                va="center", ha="center", bbox=dict(fc="w", ec="w", lw=0, boxstyle="round,pad=0.4"),
                weight="bold", fontsize=8)
    pyplot.annotate("", xy=(0.3125, 0.470), xytext=(0.3125, 0.230), zorder=1,
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(0.3205, 0.325, "encode as DNA sequences of length 2,950 nt",
                va="center", ha="left", fontsize=8)

    positions, start = linspace(0.100, 0.525, 2951), 0
    type_colors = {"r": "#CCCCCC", "s": "#F09456", "a": "#A6D08C", "p": "#8BB8E1"}
    names = {"r": "buffer", "s": "anchor", "a": "address", "p": "payload"}
    lengths = {"r": 200, "s": 10, "a": 60, "p": 110}
    for segment_type in ["r", "s", "a"] + (["s", "p"] * 20) + ["s", "a", "s", "r"]:
        if segment_type != "s":
            pyplot.fill_between([positions[start], positions[start + lengths[segment_type]]],
                                0.120, 0.180, fc=type_colors[segment_type], lw=0, zorder=0)
        else:
            pyplot.fill_between([positions[start], positions[start + lengths[segment_type]]],
                                0.120, 0.180, fc=type_colors[segment_type], lw=0, zorder=1)
        start += lengths[segment_type]

    pyplot.fill_between([0.100, 0.525], 0.120, 0.180, ec="k", fc="none", lw=0.50, zorder=2)
    pyplot.text(0.530, 0.147, "× 160", va="center", ha="left", fontsize=9)
    positions = linspace(0.000, 0.350, 9)
    for location, label, length in zip([positions[7], positions[5], positions[3], positions[1]],
                                       ["r", "s", "a", "p"], [200, 10, 60, 110]):
        pyplot.fill_between([0.002, 0.015], location - 0.02, location + 0.02, ec="k", fc=type_colors[label],
                            lw=0.5, zorder=2)
        pyplot.text(0.020, location - 0.003, names[label] + " (" + str(length) + " nt)",
                    va="center", ha="left", fontsize=7)

    pyplot.text(0.626, 0.190, "write", va="center", ha="center", fontsize=8)
    pyplot.annotate("", xy=(0.576, 0.150), xytext=(0.676, 0.150), zorder=1,
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(0.770, 0.190, "read", va="center", ha="center", fontsize=8)
    pyplot.annotate("", xy=(0.720, 0.150), xytext=(0.820, 0.150), zorder=1,
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))

    center, locations = 0.698, linspace(0.080, 0.230, 6)[1:-1]
    pyplot.fill_between([center - 0.01, center, center + 0.01], [0.080, 0.050, 0.080], locations[1] - 0.015,
                        ec="k", fc="#FFE6D9", zorder=0)
    pyplot.fill_between([center - 0.01, center, center + 0.01], [0.080, 0.050, 0.080], 0.230,
                        ec="k", fc="none", zorder=1)
    pyplot.hlines(locations[:4], center - 0.010, center - 0.005, color="k", lw=0.5, zorder=2)
    pyplot.hlines(0.215, center - 0.010, center + 0.010, color="k", lw=0.75, zorder=3)
    pyplot.fill_between([center - 0.013, center + 0.013], 0.230, 0.270,
                        ec="k", fc="w", zorder=1)
    pyplot.hlines(0.250, center - 0.013, center + 0.013, color="k", lw=0.75, zorder=3)
    pyplot.plot([center - 0.013, center - 0.017, center - 0.017, center - 0.013], [0.240, 0.240, 0.260, 0.260],
                color="k", lw=1.00, zorder=3)
    pyplot.text(center, 0.335, "library\npool", va="center", ha="center", fontsize=7)

    ax.add_patch(patches.FancyBboxPatch((0.860, 0.040), 0.020, 0.100, lw=0.75, ec="k",
                                        fc="#94E3FB", boxstyle="round,pad=0.02,rounding_size=0.01", zorder=0))
    ax.add_patch(patches.FancyBboxPatch((0.855, 0.050), 0.030, 0.200, lw=0.75, ec="k",
                                        fc="#CFCCC4", boxstyle="round,pad=0.02,rounding_size=0.01", zorder=6))
    ax.add_patch(patches.FancyBboxPatch((0.855, 0.050), 0.030, 0.200, lw=0.75, ec="k",
                                        fc="none", boxstyle="round,pad=0.02,rounding_size=0.01", zorder=8))
    pyplot.fill_between([0.835, 0.855, 0.885, 0.905], 0.263,
                        [0.263, 0.318, 0.318, 0.263], lw=0.75, ec="k", fc="#CFCCC4", zorder=2)
    pyplot.fill_between([0.835, 0.855, 0.885, 0.905], 0.263,
                        [0.263, 0.318, 0.318, 0.263], lw=0.75, ec="k", fc="none", zorder=5)
    pyplot.plot([0.855, 0.860], [0.318, 0.305], color="#898780", lw=0.30, zorder=3)
    pyplot.plot([0.885, 0.880], [0.318, 0.305], color="#898780", lw=0.30, zorder=3)
    pyplot.plot([0.835, 0.853], [0.263, 0.280], color="#898780", lw=0.30, zorder=3)
    pyplot.plot([0.905, 0.887], [0.263, 0.280], color="#898780", lw=0.30, zorder=3)
    pyplot.fill_between([0.853, 0.860, 0.880, 0.887], 0.280,
                        [0.280, 0.305, 0.305, 0.280], lw=0.30, ec="#898780", fc="k", zorder=4)

    pyplot.fill_between([0.839, 0.855, 0.885, 0.901], [0.365, 0.325, 0.325, 0.365],
                        0.365, lw=1.50, ec="#454440", fc="none", zorder=6)
    pyplot.text(0.888, 0.243, "CycloneSEQ", va="center", ha="center", fontsize=3.5, zorder=8)
    pyplot.vlines([0.870 - 0.001, 0.870 + 0.001], 0.030, 0.270, color="#898780", lw=0.30, zorder=7)
    pyplot.vlines(linspace(0.840, 0.865, 6), 0.050, 0.180, color="k", lw=0.50, ls="--", zorder=7)
    pyplot.vlines(linspace(0.875, 0.900, 6), 0.050, 0.180, color="k", lw=0.50, ls="--", zorder=7)
    pyplot.text(0.870, 0.425, "sequencing\nplatform", va="center", ha="center", fontsize=7)

    pyplot.plot([0.703, 0.703, 0.730, 0.815, 0.840], [0.705, 0.480, 0.380, 0.380, 0.280],
                color="k", lw=1.0, zorder=3)

    pyplot.text(0.690, 0.950, "computer\ntower", va="center", ha="center", fontsize=7)
    ax.add_patch(patches.FancyBboxPatch((0.685, 0.500), 0.010, 0.320, lw=0.75, ec="k",
                                        fc="#82858B", boxstyle="round,pad=0.02,rounding_size=0.02", zorder=2))
    pyplot.fill_between([0.665, 0.675, 0.705, 0.715], 0.825,
                        [0.825, 0.860, 0.860, 0.825], lw=0.75, ec="k", fc="#82858B", zorder=0)
    pyplot.scatter([0.690], [0.515], ec="k", fc="#576268", lw=0.50, s=10, zorder=4)
    pyplot.fill_between([0.673, 0.707], 0.780, 0.810, ec="k", fc="none", lw=0.50, zorder=5)
    pyplot.fill_between([0.673, 0.707], 0.730, 0.760, ec="k", fc="none", lw=0.50, zorder=5)
    pyplot.hlines([0.745, 0.795], 0.700, 0.704, color="k", lw=0.50, zorder=4)
    pyplot.fill_between([0.673, 0.681], 0.700, 0.710, ec="k", fc="none", lw=0.50, zorder=5)
    pyplot.fill_between([0.686, 0.694], 0.700, 0.710, ec="k", fc="none", lw=0.50, zorder=5)
    pyplot.fill_between([0.699, 0.707], 0.700, 0.710, ec="k", fc="k", lw=0.50, zorder=5)

    centers, labels = [0.80, 0.94], ["sequencing platform\nprocess screen", "data retrieval\ndisplay screen"]
    for center, label in zip(centers, labels):
        ax.add_patch(patches.FancyBboxPatch((center - 0.040, 0.570), 0.080, 0.300, lw=0.75, ec="k",
                                            fc="#82858B", boxstyle="round,pad=0.02,rounding_size=0.02", zorder=2))
        pyplot.fill_between([center - 0.055, center + 0.055], 0.610, 0.855, lw=0.50, ec="k", fc="w", zorder=3)
        pyplot.scatter([center], [0.583], ec="k", fc="#576268", lw=0.50, s=10, zorder=4)
        pyplot.text(center, 0.950, label, va="center", ha="center", fontsize=7)
        pyplot.fill_between([center - 0.003, center + 0.003], 0.500, 0.610,
                            lw=0.75, ec="k", fc="#82858B", zorder=1)
        pyplot.fill_between([center - 0.040, center - 0.020, center + 0.020, center + 0.040], 0.480,
                            [0.480, 0.520, 0.520, 0.480], lw=0.75, ec="k", fc="#82858B", zorder=0)

    pyplot.hlines(0.535, 0.700, 0.940, lw=0.75, color="k", zorder=0)

    random.seed(2023)
    matrix = random.random(size=(40, 8)) * 3
    # noinspection PyUnresolvedReferences
    matrix[matrix > 1] = 1
    x_locations, y_locations = linspace(0.747, 0.853, 41), linspace(0.745, 0.845, 9)
    for x_index, (x_former, x_latter) in enumerate(zip(x_locations[:-1], x_locations[1:])):
        for y_index, (y_former, y_latter) in enumerate(zip(y_locations[:-1], y_locations[1:])):
            # noinspection PyUnresolvedReferences
            pyplot.fill_between([x_former, x_latter], y_former, y_latter, alpha=0.5,
                                ec="none", fc=pyplot.get_cmap("YlGn")(matrix[x_index, y_index]), lw=0, zorder=4)
    x_locations = linspace(0.747, 0.853, 9)
    for center in [x_locations[1], x_locations[3], x_locations[5], x_locations[7]]:
        bias = x_locations[1] - x_locations[0]
        pyplot.fill_between([center - bias + 0.001, center + bias - 0.001], 0.620, 0.730,
                            lw=0.50, ec="k", fc="#EEEEEE", zorder=4)
        pyplot.hlines(linspace(0.620, 0.730, 8)[1:-1], center - bias + 0.004, center + bias - 0.004,
                      lw=0.50, color="#82858B", zorder=5)

    x_locations, y_locations = linspace(0.887, 0.947, 41), linspace(0.620, 0.845, 31)
    for x_index, (x_former, x_latter) in enumerate(zip(x_locations[:-1], x_locations[1:])):
        for y_index, (y_former, y_latter) in enumerate(zip(y_locations[:-1], y_locations[1:])):
            pyplot.fill_between([x_former, x_latter], y_former, y_latter,
                                ec="none", fc=colors[frame_data[0, 29 - y_index, x_index]], lw=0, zorder=4)
    pyplot.text(0.972, 0.822, "round 271", va="center", ha="center", fontsize=6)
    pyplot.text(0.972, 0.730, "01:40.22", va="center", ha="center", fontsize=6)
    pyplot.text(0.972, 0.645, "3200/3200", va="center", ha="center", fontsize=6)

    pyplot.xlabel("real-time non-blocking data retrieval demonstration "
                  "using single-molecule nanopore sequencing platform", fontsize=9)
    pyplot.xlim(0, 1.001)
    pyplot.ylim(0, 1)
    pyplot.xticks([])
    pyplot.yticks([])

    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["bottom"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["left"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[3:6, 0])

    x_labels, y_rates = shown_data["b"]

    for index in range(18, 35):
        if index > 18:
            pyplot.fill_between([index - 0.4, index], 0, y_rates[0, index], ec="k", fc="#FDE1DE", lw=0.5)
            pyplot.fill_between([index, index + 0.4], 0, y_rates[1, index], ec="k", fc="#D9F0FC", lw=0.5)
        else:
            pyplot.fill_between([index - 0.4, index], 0, y_rates[0, index], ec="k", fc="#FDE1DE", lw=0.5,
                                label="across sequences")
            pyplot.fill_between([index, index + 0.4], 0, y_rates[1, index], ec="k", fc="#D9F0FC", lw=0.5,
                                label="within a sequence")

    pyplot.legend(loc="upper left", fontsize=8)
    pyplot.xlabel("address tolerance (%)", fontsize=9)
    pyplot.ylabel("proportion (%)", fontsize=9)
    pyplot.xticks(arange(18, 34, 3), arange(30, 56, 5), fontsize=9)
    pyplot.yticks(linspace(0, 0.20, 5), arange(0, 21, 5), fontsize=9)
    pyplot.xlim(17.4, 33.6)
    pyplot.ylim(0, 0.20)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[3:6, 1])

    x_values, y_values, average_value = shown_data["c"]

    pyplot.fill_between(x_values, 0, y_values, ec="none", fc="#CBD2DF", lw=0.00, alpha=0.8, zorder=1)
    pyplot.plot(x_values, y_values, color="#888888", lw=1.00, zorder=2)

    pyplot.vlines(average_value, 0, 0.024, lw=0.75, ls="--", color="k", zorder=-1)
    pyplot.hlines(0.024, average_value - 0.0006, average_value + 0.0006, lw=0.75, color="k", zorder=2)
    pyplot.text(average_value, 0.025, "%.3f" % (average_value * 100) + "%", va="center", ha="center", fontsize=8)

    pyplot.xlabel("edit error rate per sequence (%)", fontsize=9)
    pyplot.ylabel("proportion (%)", fontsize=9)
    pyplot.xticks(linspace(0.002, 0.007, 6), ["%.1f" % v for v in linspace(0.2, 0.7, 6)], fontsize=9)
    pyplot.yticks(linspace(0, 0.032, 5), ["%.1f" % v for v in linspace(0, 3.2, 5)], fontsize=9)
    pyplot.xlim(0.002, 0.007)
    pyplot.ylim(0, 0.032)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[3:6, 2])

    type_data = shown_data["d"]

    labels = ["substitution", "insertion", "deletion"]

    colors = [pyplot.get_cmap("binary")(v * 0.3) for v in type_data]
    # noinspection PyTypeChecker
    ax.pie(type_data, labels=labels, colors=colors, autopct="%.3f%%",
           startangle=160, textprops={"fontsize": 9}, wedgeprops={"ec": "k", "lw": 0.75})

    ax.axis("equal")
    ax.set_xlabel("error type distribution", fontsize=9)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[3:6, 3])

    sub_data = shown_data["e"]

    pyplot.boxplot([sub_data[:, 0]], positions=[1], widths=0.6, showfliers=False, patch_artist=True,
                   medianprops=dict(color="k", lw=1.5), boxprops=dict(fc="#CBD2DF", ec="#888888", lw=0.75),
                   whiskerprops=dict(color="#888888", lw=0.75), capprops=dict(color="#888888", lw=0.75))
    pyplot.boxplot([sub_data[:, 1]], positions=[2], widths=0.6, showfliers=False, patch_artist=True,
                   medianprops=dict(color="k", lw=1.5), boxprops=dict(fc="#CBD2DF", ec="#888888", lw=0.75),
                   whiskerprops=dict(color="#888888", lw=0.75), capprops=dict(color="#888888", lw=0.75))
    pyplot.boxplot([sub_data[:, 2]], positions=[3], widths=0.6, showfliers=False, patch_artist=True,
                   medianprops=dict(color="k", lw=1.5), boxprops=dict(fc="#CBD2DF", ec="#888888", lw=0.75),
                   whiskerprops=dict(color="#888888", lw=0.75), capprops=dict(color="#888888", lw=0.75))

    pyplot.xlabel("fragment type", fontsize=9)
    pyplot.ylabel("edit error rate (%)", fontsize=9)
    pyplot.xticks([1, 2, 3], ["anchor", "address", "payload"], fontsize=9)
    pyplot.yticks(linspace(0, 0.016, 5), ["%.1f" % v for v in linspace(0, 1.6, 5)], fontsize=9)
    pyplot.xlim(0.5, 3.5)
    pyplot.ylim(0, 0.016)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[6:9, :])

    processes, bounds = shown_data["f"]

    labels, colors = ["50%", "90%", "99%", "100%"], pyplot.get_cmap("binary")(linspace(0, 1, 4))
    for index in range(5):
        former_location, former_rate = 0, 0
        for location, rate in processes[index + 1]:
            pyplot.fill_between([former_location, location], (4 - index) + 0.85, (4 - index) + 1.15,
                                ec="none", fc=pyplot.get_cmap("RdYlGn")(former_rate), lw=0, zorder=1)
            former_location, former_rate = location, rate
        for label, color in zip(labels, colors):
            pyplot.vlines(bounds[label][0][index], (4 - index) + 0.85, (4 - index) + 1.15, lw=0.75, color="k", zorder=2)
            pyplot.scatter([bounds[label][0][index]], [(4 - index) + 1.00], s=20,
                           ec="k", fc=color, lw=0.75, zorder=3)
            pyplot.text(bounds[label][0][index], (4 - index) + 1.35, "%.2f" % bounds[label][0][index],
                        va="center", ha="center", fontsize=8)

    pyplot.text(182.5, 4.20, "sequencing depth (X)", va="center", ha="center", fontsize=8)
    locations = linspace(0.80, 4.00, 7)
    pyplot.hlines([locations[0], locations[-2], locations[-1]], 165, 200, lw=0.75, color="k")
    # noinspection PyUnresolvedReferences
    text_locations = ((locations[:-1] + locations[1:]) / 2.0)[::-1]
    pyplot.text(168, text_locations[0], "trial", va="center", ha="center", fontsize=8)
    marker_locations = linspace(172, 200, 9)[1::2]
    pyplot.vlines(172, 0.80, 4.00, lw=0.75, color="k")
    for location, color in zip(marker_locations, colors):
        pyplot.scatter([location], [text_locations[0]], s=20, ec="k", fc=color, lw=0.75, zorder=3)
    for index in range(1, 6):
        pyplot.text(168, text_locations[index], str(index), va="center", ha="center", fontsize=8)
    for index_1, label in enumerate(labels):
        for index_2, depth in enumerate(bounds[label][1]):
            pyplot.text(marker_locations[index_1], text_locations[index_2 + 1], "%.1f" % depth,
                        va="center", ha="center", fontsize=8)

    pyplot.fill_between([165, 199.8], 4.60, 5.48, lw=0.75, ec="#888888", fc="none")
    pyplot.text(182.5, 5.25, "observed data retrieval rate", va="center", ha="center", fontsize=8)
    locations = linspace(165, 197, 5)
    for location, color, label in zip(locations[:-1], colors, labels):
        pyplot.scatter([location + 2.0], [4.87], s=20, ec="k", fc=color, lw=0.75)
        pyplot.text(location + 3.5, 4.84, label, va="center", ha="left", fontsize=8)

    pyplot.xlabel("retrieval runtime (second)", fontsize=9)
    pyplot.ylabel("independent replicate trial", fontsize=9)
    pyplot.xticks(arange(0, 201, 10), arange(0, 201, 10), fontsize=9)
    pyplot.yticks(arange(5) + 1, ["5", "4", "3", "2", "1"], fontsize=9)
    pyplot.xlim(0, 200)
    pyplot.ylim(0.5, 5.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[9, :])

    locations, colors = linspace(0, 1, 101), pyplot.get_cmap("RdYlGn")(linspace(0, 1, 100))
    for former, latter, color in zip(locations[:-1], locations[1:], colors):
        pyplot.fill_between([former, latter], 0, 1, ec="none", fc=color, lw=0, zorder=2)

    pyplot.title("data retrieval rate (%)", fontsize=9)
    pyplot.xticks(linspace(0, 1, 21), arange(0, 101, 5), fontsize=9)
    pyplot.yticks([])
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)

    figure.align_labels()
    figure.text(0.021, 0.990, "a", va="center", ha="center", fontsize=12)
    figure.text(0.021, 0.700, "b", va="center", ha="center", fontsize=12)
    figure.text(0.265, 0.700, "c", va="center", ha="center", fontsize=12)
    figure.text(0.506, 0.700, "d", va="center", ha="center", fontsize=12)
    figure.text(0.768, 0.700, "e", va="center", ha="center", fontsize=12)
    figure.text(0.021, 0.390, "f", va="center", ha="center", fontsize=12)

    pyplot.savefig("./fig/main-06.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


if __name__ == "__main__":
    main_f01()
    main_f02()
    main_f03()
    main_f04()
    main_f05()
    main_f06()
