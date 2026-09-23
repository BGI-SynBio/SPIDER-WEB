from logging import getLogger, CRITICAL
from matplotlib import pyplot, patheffects, rcParams
from numpy import array, linspace, arange, random, min, abs, sum, log10, max, nan, where
from scipy.stats import pearsonr, gaussian_kde
from warnings import filterwarnings

from sbw import load_data

from works import pkg_path, fig_path, tab_path

filterwarnings("ignore")

getLogger("matplotlib").setLevel(CRITICAL)

rcParams["font.family"] = "Arial"
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "Linux Libertine"
rcParams["mathtext.cal"] = "Lucida Calligraphy"
rcParams["mathtext.it"] = "Linux Libertine:italic"
rcParams["mathtext.bf"] = "Linux Libertine:bold"


def supp_f01():
    """
    Create Figure S1 in the supplementary file.

    TITLE: Complete pipeline of coding algorithm generation.
    """
    pyplot.figure(figsize=(10, 5), tight_layout=True)

    pyplot.annotate("", xy=(0.5, 3.8), xytext=(0.5, 4.2),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.fill_between([0.1, 0.9], 3.2, 3.8, fc="#DDDDDD", lw=0, zorder=1)
    pyplot.text(0.5, 3.5, "initialize vertices (k-mer)\nbased on the given constraints",
                va="center", ha="center", fontsize=9, zorder=2)
    pyplot.annotate("", xy=(0.5, 2.8), xytext=(0.5, 3.2),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.fill_between([0.1, 0.9], 2.2, 2.8, fc="#DDDDDD", lw=0, zorder=1)
    pyplot.text(0.5, 2.5, "connect remaining vertices into\na directed graph (digraph)",
                va="center", ha="center", fontsize=9, zorder=2)
    pyplot.annotate("", xy=(0.5, 1.8), xytext=(0.5, 2.2),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.fill_between([0.1, 0.5, 0.9], [1.5, 1.2, 1.5], [1.5, 1.8, 1.5], fc="#DDDDDD", lw=0, zorder=1)
    pyplot.text(0.5, 1.5, "identify vertices that do not\nmeet in/out-degree demands",
                va="center", ha="center", fontsize=9, zorder=2)
    pyplot.text(0.52, 1.05, "yes", va="center", ha="left", fontsize=9, zorder=2)
    pyplot.annotate("", xy=(0.5, 0.8), xytext=(0.5, 1.2),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.text(1.0, 2.5, "no", va="center", ha="center", fontsize=9, zorder=2,
                bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.5"))
    pyplot.fill_between([0.1, 0.9], 0.2, 0.8, fc="#DDDDDD", lw=0, zorder=1)
    pyplot.text(0.5, 0.5, "remove identified vertices",
                va="center", ha="center", fontsize=9, zorder=2)
    pyplot.plot([0.10, 0.01, 0.01], [0.50, 0.50, 2.50], lw=0.75, color="#555555", zorder=1)
    pyplot.annotate("", xy=(0.10, 2.50), xytext=(0.01, 2.50),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.hlines(1.50, 0.90, 1.00, lw=0.75, color="#555555", zorder=1)
    pyplot.vlines(1.00, 1.50, 3.50, lw=0.75, color="#555555", zorder=1)
    pyplot.annotate("", xy=(1.10, 3.50), xytext=(1.00, 3.50),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.fill_between([1.1, 1.9], 3.2, 3.8, fc="#DDDDDD", lw=0, zorder=1)
    pyplot.text(1.5, 3.5, "find the best component\nin the established digraph",
                va="center", ha="center", fontsize=9, zorder=2)
    pyplot.annotate("", xy=(2.10, 3.50), xytext=(1.90, 3.50), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color="#555555", shrinkA=0, shrinkB=0, lw=0.75, ls="--"))
    pyplot.fill_between([2.1, 2.9], 3.2, 3.8, fc="#DDDDDD", lw=0.75, zorder=1)
    pyplot.text(2.5, 3.5, "eliminate ambiguity for\nfast coding (if required)",
                va="center", ha="center", fontsize=9, zorder=2)

    pyplot.text(1.48, 3.00, "basic coding\n(Fig. S2)", va="top", ha="right", fontsize=9, zorder=2)
    pyplot.text(2.52, 3.00, "fast coding\n(Fig. 1c)", va="top", ha="left", fontsize=9, zorder=2)
    pyplot.plot([1.50, 1.50, 2.50, 2.50], [3.2, 3.0, 3.0, 3.2], lw=0.75, color="#555555", zorder=1)
    pyplot.annotate("", xy=(2.0, 2.8), xytext=(2.0, 3.0),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.fill_between([1.6, 2.4], 2.2, 2.8, fc="#DDDDDD", lw=0.75, zorder=1)
    pyplot.text(2.0, 2.5, "bind nucleotide with digits by the\nmapping table in Fig. 1c",
                va="center", ha="center", fontsize=9, zorder=2)
    pyplot.plot([2.40, 2.50, 2.50], [2.50, 2.50, 0.50], color="#555555", lw=0.75, zorder=1)
    pyplot.annotate("", xy=(2.40, 0.50), xytext=(2.50, 0.50),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.annotate("", xy=(2.0, 1.8), xytext=(2.0, 2.2), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color="#555555", shrinkA=0, shrinkB=0, lw=0.75, ls="--"))
    pyplot.fill_between([1.6, 2.4], 1.2, 1.8, fc="#DDDDDD", lw=0, zorder=1)
    pyplot.text(2.0, 1.5, "construct additional modules for\nglobal screening (if required)",
                va="center", ha="center", fontsize=9, zorder=2)
    pyplot.annotate("", xy=(2.0, 0.8), xytext=(2.0, 1.2),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)
    pyplot.fill_between([1.6, 2.4], 0.2, 0.8, fc="#DDDDDD", lw=0, zorder=1)
    pyplot.text(2.0, 0.5, "save the coding digraph and\nadditional modules (if produced)",
                va="center", ha="center", fontsize=9, zorder=2)
    pyplot.annotate("", xy=(2.0, -0.2), xytext=(2.0, 0.2),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", shrinkA=0, shrinkB=0, lw=0.75), zorder=1)

    pyplot.fill_between([2.55, 2.63], 0.45, 0.55, fc="#DDDDDD", lw=0, zorder=1)
    pyplot.annotate("", xy=(2.55, 0.2), xytext=(2.63, 0.2), zorder=1,
                    arrowprops=dict(arrowstyle="<|-", color="#555555", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.annotate("", xy=(2.55, -0.1), xytext=(2.63, -0.1), zorder=1,
                    arrowprops=dict(arrowstyle="<|-", color="#555555", shrinkA=0, shrinkB=0, lw=0.75, ls="--"))
    pyplot.text(2.66, 0.50, "step", va="center", ha="left", fontsize=9, zorder=2)
    pyplot.text(2.66, 0.20, "required", va="center", ha="left", fontsize=9, zorder=2)
    pyplot.text(2.66, -0.10, "elective", va="center", ha="left", fontsize=9, zorder=2)
    pyplot.xlim(0, 3)
    pyplot.ylim(-0.2, 4.2)
    pyplot.axis("off")

    pyplot.savefig(fig_path + "supp-01.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f02():
    """
    Create Figure S2 in the supplementary file.

    TITLE: Algorithm generation runtime under commonly used constraints.
    """
    shown_data = load_data(file_path=pkg_path + "supp-02.pkl")

    pyplot.figure(figsize=(10, 3.5), tight_layout=True)

    mesh = pyplot.pcolormesh(arange(11), arange(7), shown_data.T, cmap="spring", vmin=0, vmax=24)
    bar = pyplot.colorbar(mesh, location="right")
    bar.set_label("algorithm generation runtime (seconds)", fontsize=9)
    bar.set_ticks([0, 6, 12, 18, 24])
    bar.set_ticklabels(["0", "6", "12", "18", "24"], fontsize=9)

    pyplot.hlines(arange(5) + 1, 0, 10, colors="k", lw=0.75)
    pyplot.vlines(arange(9) + 1, 0, 6, colors="k", lw=0.75)
    for index_1 in range(10):
        for index_2 in range(6):
            pyplot.text(index_1 + 0.5, index_2 + 0.5, "%.1f" % shown_data[index_1, index_2],
                        va="center", ha="center", fontsize=9)
    pyplot.xlabel("maximum homopolymer run-length", fontsize=9)
    pyplot.ylabel("regionalized GC content (for any 10-mer)", fontsize=9)
    pyplot.xticks(arange(10) + 0.5, ["1", "2", "3", "4", "5", "6", "7", "8", "9", "N.A."], fontsize=9)
    pyplot.yticks(arange(6) + 0.5, ["50%-50%", "40%-60%", "30%-70%", "20%-80%", "10%-90%", "N.A."], fontsize=9)
    pyplot.xlim(0, 10)
    pyplot.ylim(0, 6)

    pyplot.savefig(fig_path + "supp-02.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f03():
    """
    Create Figure S3 in the supplementary file.

    TITLE: Message encoding and sequence decoding process using the basic mode.
    """
    pyplot.figure(figsize=(10, 3), tight_layout=True)

    color_groups = [["#FBE5D6", "#F4B183", "#ED7D31", "#A24A0E"], ["#E2F0D9", "#A9D18E", "#70AD47", "#4D7731"]]
    pyplot.text(0.5, 0.92, "walk of coding digraph", va="center", ha="center", fontsize=10)
    points = array([[0.5, 0.2, 0.4, 0.6, 0.8, 0.5, 0.7, 0.4, 0.6, 0.4, 0.6, 0.8],
                    [0.9, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.3, 0.3, 0.1, 0.1, 0.1]])
    points[1] -= 0.05
    pyplot.scatter(points[0], points[1], color="white", edgecolor="black", lw=0.75, s=30)
    for former, latter, info, mark, bias in [(0, 1, "{A|0}", 0, -0.07), (0, 2, "{C|1}", 0, -0.03),
                                             (0, 3, "{G|2}", 1, +0.03), (0, 4, "{T|3}", 0, +0.07),
                                             (3, 5, "{A|0}", 1, -0.03), (3, 6, "{G|1}", 0, +0.03),
                                             (5, 7, "{A|0}", 0, -0.03), (5, 8, "{C|1}", 1, +0.03),
                                             (8, 9, "{C|0}", 0, -0.05), (8, 10, "{G|1}", 0, 0),
                                             (8, 11, "{T|2}", 1, + 0.05)]:
        point_1, point_2 = (points[0][former], points[1][former]), (points[0][latter], points[1][latter])
        connection = None
        if points[0][former] < points[0][latter]:
            connection = "arc3,rad=0.2"
        elif points[0][former] > points[0][latter]:
            connection = "arc3,rad=-0.2"
        if mark:
            pyplot.annotate("", xy=point_1, xytext=point_2, zorder=1,
                            arrowprops=dict(arrowstyle="<|-", color="#000000", connectionstyle=connection,
                                            shrinkA=3, shrinkB=5, lw=0.75))
            x, y = (point_1[0] + point_2[0]) / 2 + bias, (point_1[1] + point_2[1]) / 2 + 0.01
            pyplot.plot([x - 0.05, x - 0.05, x + 0.05, x + 0.05, x - 0.05],
                        [y - 0.025, y + 0.025, y + 0.025, y - 0.025, y - 0.025], color="k", lw=0.75)
            pyplot.fill_between([x - 0.05, x], y - 0.025, y + 0.025,
                                fc=color_groups[0][{"A": 0, "C": 1, "G": 2, "T": 3}[info[1]]], lw=0, zorder=2)
            if info[1] != "T":
                pyplot.text(x - 0.025, y - 0.005, info[1], color="k", va="center", ha="center", fontsize=7.5, zorder=3)
            else:
                pyplot.text(x - 0.025, y - 0.005, info[1], color="w", va="center", ha="center", fontsize=7.5, zorder=3)
            pyplot.fill_between([x, x + 0.05], y - 0.025, y + 0.025, fc=color_groups[1][int(info[3])], lw=0, zorder=2)
            if info[3] != "3":
                pyplot.text(x + 0.025, y - 0.005, info[3], color="k", va="center", ha="center", fontsize=7.5, zorder=3)
            else:
                pyplot.text(x + 0.025, y - 0.005, info[3], color="w", va="center", ha="center", fontsize=7.5, zorder=3)
        else:
            pyplot.annotate("", xy=point_1, xytext=point_2, zorder=1,
                            arrowprops=dict(arrowstyle="<|-", color="#666666", connectionstyle=connection,
                                            shrinkA=3, shrinkB=5, lw=0.75))
    for former, latter in [(1, 2), (2, 3), (3, 4), (5, 6), (7, 8), (9, 10), (10, 11)]:
        pyplot.hlines(points[1][former], points[0][former] + 0.05, points[0][latter] - 0.05, lw=0.75, ls="--")
    pyplot.text(1.2, 0.75, "out-degree", va="center", ha="center", fontsize=10)
    pyplot.hlines(points[1][4], points[0][4] + 0.05, 1.15, lw=0.75, ls="--")
    pyplot.text(1.2, points[1][4], "4", va="center", ha="center", fontsize=9)
    pyplot.hlines(points[1][6], points[0][6] + 0.05, 1.15, lw=0.75, ls="--")
    pyplot.text(1.2, points[1][6], "2", va="center", ha="center", fontsize=9)
    pyplot.hlines(points[1][8], points[0][8] + 0.05, 1.15, lw=0.75, ls="--")
    pyplot.text(1.2, points[1][8], "2", va="center", ha="center", fontsize=9)
    pyplot.hlines(points[1][11], points[0][11] + 0.05, 1.15, lw=0.75, ls="--")
    pyplot.text(1.2, points[1][11], "3", va="center", ha="center", fontsize=9)
    pyplot.fill_between([1.45, 1.55], 0.16, 0.55, facecolor="w", edgecolor="k", linewidth=0.75)
    pyplot.fill_between([2.03, 2.17], 0.46, 0.55, facecolor="w", edgecolor="k", linewidth=0.75)
    pyplot.fill_between([2.35, 2.45], 0.16, 0.55, facecolor="w", edgecolor="k", linewidth=0.75)
    for index, info in enumerate([("$4$", r"$\times$", "$10$", r"$=$", "$42$", r"$-$", "$2$"),
                                  ("$2$", r"$\times$", "$5$", r"$=$", "$10$", r"$-$", "$0$"),
                                  ("$2$", r"$\times$", "$2$", r"$=$", "$5$", r"$-$", "$1$"),
                                  ("$3$", r"$\times$", "$0$", r"$=$", "$2$", r"$-$", "$2$")]):
        for location, string in enumerate(info):
            pyplot.text(1.5 + location * 0.15, 0.5 - index * 0.1, string, va="center", ha="center", fontsize=10)
    pyplot.annotate("", xy=(1.25, points[1][4]), xytext=(1.45, 0.5),
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.75, ls="--",
                                    connectionstyle="arc3,rad=0.1"))
    pyplot.annotate("", xy=(1.25, points[1][6]), xytext=(1.45, 0.4),
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.75, ls="--",
                                    connectionstyle="arc3,rad=0.05"))
    pyplot.annotate("", xy=(1.25, points[1][8]), xytext=(1.45, 0.3),
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.75, ls="--",
                                    connectionstyle="arc3,rad=-0.05"))
    pyplot.annotate("", xy=(1.25, points[1][11]), xytext=(1.45, 0.2),
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.75, ls="--",
                                    connectionstyle="arc3,rad=-0.1"))
    pyplot.text(3.6, 0.8, "[101010]", va="center", ha="center", fontsize=10)
    pyplot.text(3.8, 0.8, "binary message", va="center", ha="left", fontsize=10)
    pyplot.text(2.55, 0.65, r"$2^1+2^3+2^5$", va="center", ha="right", fontsize=10)
    pyplot.hlines(0.65, 2.60, 2.88, lw=0.75, ls="--")
    pyplot.text(3.55, 0.65, "decimal conversion", va="center", ha="right", color="#0070C0", fontsize=10)
    pyplot.annotate("", xy=(3.6, 0.55), xytext=(3.6, 0.75),
                    arrowprops=dict(arrowstyle="-|>", color="#0070C0", lw=2))
    pyplot.text(3.6, 0.5, "42", va="center", ha="center", fontsize=10)
    pyplot.text(3.8, 0.5, "decimal number", va="center", ha="left", fontsize=10)
    pyplot.hlines(0.35, 2.45, 2.73, lw=0.75, ls="--")
    pyplot.text(3.55, 0.35, "graph-based conversion", va="center", ha="right", color="#0070C0", fontsize=10)
    pyplot.annotate("", xy=(3.6, 0.25), xytext=(3.6, 0.45),
                    arrowprops=dict(arrowstyle="-|>", color="#0070C0", lw=2))
    pyplot.text(3.6, 0.20, "[2012]", va="center", ha="center", fontsize=10)
    pyplot.text(3.8, 0.20, "graph-based vector", va="center", ha="left", fontsize=10)
    pyplot.text(3.6, 0.12, r"$=$", va="center", ha="center", rotation="vertical", fontsize=10)
    pyplot.text(3.6, 0.04, "[GACT]", va="center", ha="center", fontsize=10)
    pyplot.text(3.8, 0.04, "DNA sequence", va="center", ha="left", fontsize=10)
    pyplot.annotate("", xy=(4.45, 0.8), xytext=(4.45, 0.04),
                    arrowprops=dict(arrowstyle="-|>", color="#0070C0", lw=2, connectionstyle="arc3,rad=0.2"))
    pyplot.text(4.85, 0.42, "decode by\nreverse steps", va="center", ha="center", color="#0070C0", fontsize=10)
    pyplot.axis("off")
    pyplot.xlim(0.1, 5.1)
    pyplot.ylim(0.0, 1.0)

    pyplot.savefig(fig_path + "supp-03.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f04():
    """
    Create Figure S4 in the supplementary file.

    TITLE: Information density comparison of DNA Fountain, SPIDER-WEB (basic mode), and SPIDER-WEB (fast mode).
    """
    shown_data = load_data(file_path=pkg_path + "supp-04.pkl")

    pyplot.figure(figsize=(10, 10.5), tight_layout=True)
    titles = ["average value of information density (DNA fountain)",
              "standard deviation of information density (DNA fountain)",
              "average value of information density (SPIDER-WEB; basic mode)",
              "standard deviation of information density (SPIDER-WEB; basic mode)",
              "average value of information density (SPIDER-WEB; fast mode)",
              "standard deviation of information density (SPIDER-WEB; fast mode)"]

    for index, ((label, matrix), title) in enumerate(zip(shown_data.items(), titles)):
        pyplot.subplot(3, 2, index + 1)
        pyplot.title(title, fontsize=9)
        pyplot.vlines(arange(10)[1:], 0, 6, color="k", lw=0.75, zorder=2)
        pyplot.hlines(arange(6)[1:], 0, 10, color="k", lw=0.75, zorder=2)

        if index % 2 == 0:
            matrix[matrix > 2] = 2.0
            matrix[matrix < 0] = nan
            mesh = pyplot.pcolormesh(arange(11), arange(7), matrix.T, cmap="cool", vmin=0, vmax=2)
            bar = pyplot.colorbar(mesh, orientation="horizontal", fraction=0.1)
            bar.set_ticks([0.00, 0.50, 1.00, 1.50, 2.00])
            bar.set_ticklabels(["0.00", "0.50", "1.00", "1.50", "2.00"], fontsize=9)

        else:
            matrix[matrix > 2] = 2.0
            matrix[matrix < 0] = nan
            mesh = pyplot.pcolormesh(arange(11), arange(7), matrix.T, cmap="autumn_r", vmin=0, vmax=0.12)
            bar = pyplot.colorbar(mesh, orientation="horizontal", fraction=0.1)
            bar.set_ticks([0.00, 0.03, 0.06, 0.09, 0.12])
            bar.set_ticklabels(["0.00", "0.03", "0.06", "0.09", "0.12"], fontsize=9)

        for index_1 in range(10):
            for index_2 in range(6):
                if matrix[index_1][index_2] >= 0:
                    pyplot.text(index_1 + 0.5, index_2 + 0.5, "%.3f" % matrix[index_1, index_2],
                                va="center", ha="center", fontsize=8)
                else:
                    pyplot.text(index_1 + 0.5, index_2 + 0.5, "N.A.",
                                va="center", ha="center", fontsize=8)

        pyplot.xlabel("maximum homopolymer run-length", fontsize=9)
        pyplot.ylabel("regionalized GC content", fontsize=9)
        pyplot.xticks(arange(10) + 0.5, ["1", "2", "3", "4", "5", "6", "7", "8", "9", "N.A."], fontsize=9)
        pyplot.yticks(arange(6) + 0.5, ["50%-50%", "40%-60%", "30%-70%", "20%-80%", "10%-90%", "N.A."], fontsize=9)
        pyplot.xlim(0, 10)
        pyplot.ylim(0, 6)

    pyplot.savefig(fig_path + "supp-04.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f05():
    """
    Create Figure S5 in the supplementary file.

    TITLE: Error-correcting performance of HEDGES.
    """
    shown_data = load_data(file_path=pkg_path + "supp-05.pkl")

    figure = pyplot.figure(figsize=(10, 4), tight_layout=True)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(1, 2, 1)
    results = shown_data["a"]

    pyplot.hlines(log10(200), 0, 9, lw=0.75, ls="--", color="silver", zorder=-1, label="reference = 200")
    for position, result in enumerate(results):
        pyplot.boxplot([log10(result)], positions=[position + 0.5], widths=0.6, showmeans=False)

    pyplot.legend(loc="upper left", fontsize=9)
    pyplot.xlabel("% of edit errors in raw sequence", fontsize=9)
    pyplot.ylabel("vertex access times using HEDGES", fontsize=9)
    pyplot.xticks(arange(0, 10) + 0.5, ["%.1f" % (v / 200.0 * 100.0) for v in arange(0, 10)], fontsize=9)
    pyplot.yticks(arange(2, 9), ["1E+" + str(v) for v in arange(2, 9)], fontsize=9)
    pyplot.xlim(-0.1, 9.1)
    pyplot.ylim(2, 8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(1, 2, 2)
    results = shown_data["b"]

    pyplot.bar(arange(9) + 0.5, results / 400.0, ec="k", fc="#F88580", lw=0.75, width=0.6)

    pyplot.legend(loc="upper right", fontsize=9)
    pyplot.xlabel("% of edit errors in raw sequence", fontsize=9)
    pyplot.ylabel("success rate using HEDGES", fontsize=9)
    pyplot.xticks(arange(0, 10) + 0.5, ["%.1f" % (v / 200.0 * 100.0) for v in arange(0, 10)], fontsize=9)
    pyplot.yticks(linspace(0, 1, 6), [str(v) + "%" for v in arange(0, 101, 20)], fontsize=9)
    pyplot.xlim(-0.1, 9.1)
    pyplot.ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    figure.align_labels()
    figure.text(0.022, 0.990, "a", va="center", ha="center", fontsize=14)
    figure.text(0.515, 0.990, "b", va="center", ha="center", fontsize=14)

    pyplot.savefig(fig_path + "supp-05.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f06():
    """
    Create Figure S5 in the supplementary file.

    TITLE: Differences in data retrieval between Composite HEDGES Nanopores and SPIDER-WEB.
    """
    shown_data = load_data(file_path=pkg_path + "supp-06.pkl")

    figure = pyplot.figure(figsize=(10, 7), tight_layout=True)
    grid = pyplot.GridSpec(3, 2)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[0, :])
    results = shown_data["a"]

    x = linspace(min(results), max(results))
    y = gaussian_kde(results)(x)
    y /= sum(y)
    y *= 100.0

    pyplot.fill_between(x, 0, y, color="silver", zorder=0)
    pyplot.plot(x, y, color="k", lw=1, zorder=1)
    pyplot.text(9, 3.85, "high-end reported value", va="center", ha="center", fontsize=9)
    pyplot.vlines(9, 0, 3.7, color="r", lw=1, ls=":", zorder=1)

    pyplot.xlabel("data retrieval efficiency of Composite HEDGES Nanopores (bit per second)", fontsize=9)
    pyplot.ylabel("proportion", fontsize=9)
    pyplot.xticks(linspace(6, 12, 13), linspace(6, 12, 13), fontsize=9)
    pyplot.yticks(arange(5), [str(v) + "%" for v in arange(5)], fontsize=9)
    pyplot.xlim(6, 12)
    pyplot.ylim(0, 4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[1:, 0])
    results = shown_data["b"]

    pyplot.title("minimum runtime (seconds) required for full recovery", fontsize=9)
    colors = pyplot.get_cmap("binary")(linspace(0, 1, 5))
    for file_size, color in zip(["1KB", "2KB", "4KB", "8KB", "16KB"], colors):
        pyplot.scatter(results[file_size][0], results[file_size][1], ec="k", fc=color, s=10, lw=0.75, zorder=3,
                       label=file_size)

    pyplot.plot([-800, 24800], [-0.8, 24.8], lw=0.75, ls="--", color="silver", zorder=1)
    pyplot.plot([-800, 24800], [-0.08, 2.48], lw=0.75, ls="--", color="silver", zorder=1)
    pyplot.plot([-800, 24800], [-0.008, 0.248], lw=0.75, ls="--", color="silver", zorder=1)
    pyplot.text(650, 0.65, "1,000-fold\nspeed-up", va="center", ha="center",
                bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.2"), fontsize=7, zorder=2)
    pyplot.text(6000, 0.60, "10,000-fold\nspeed-up", va="center", ha="center",
                bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.2"), fontsize=7, zorder=2)
    pyplot.text(21500, 0.215, "100,000-fold\nspeed-up", va="center", ha="center",
                bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.2"), fontsize=7, zorder=2)
    pyplot.legend(loc="upper right", fontsize=9, framealpha=1,
                  title="file size", title_fontsize=9)
    pyplot.xlabel("Composite HEDGES Nanopores", fontsize=9)
    pyplot.ylabel("SPIDER-WEB", fontsize=9)
    pyplot.xticks(arange(0, 24001, 4800), arange(0, 24001, 4800), fontsize=9)
    pyplot.yticks(linspace(0.2, 0.7, 6), ["%.1f" % v for v in linspace(0.2, 0.7, 6)], fontsize=9)
    pyplot.xlim(-800, 24800)
    pyplot.ylim(0.1835, 0.7175)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[1:, 1])
    results = array(shown_data["c"])

    pyplot.title("required total number of sequencing reads", fontsize=9)
    indices = where(results[0] > results[1])[0]
    pyplot.scatter(results[0][indices], results[1][indices], ec="k", fc="#71C6D4", lw=0.75, s=10, zorder=3,
                   label=str(len(indices)))
    indices = where(results[0] < results[1])[0]
    pyplot.scatter(results[0][indices], results[1][indices], ec="k", fc="#EEA1B4", lw=0.75, s=10, zorder=3,
                   label=str(len(indices)))
    pyplot.vlines(arange(0, 15001, 3000), -500, 15500, color="silver", lw=0.75, ls="--", zorder=1)
    pyplot.hlines(arange(0, 15001, 3000), -500, 15500, color="silver", lw=0.75, ls="--", zorder=1)
    pyplot.plot([-500, 15500], [-500, 15500], lw=0.75, ls="--", color="k", zorder=2)
    pyplot.legend(loc="upper left", fontsize=9, framealpha=1, ncol=2,
                  title=str(len(results[0])) + " valid cases", title_fontsize=9)
    pyplot.xlabel("Composite HEDGES Nanopores", fontsize=9)
    pyplot.ylabel("SPIDER-WEB", fontsize=9)
    pyplot.xticks(arange(0, 15001, 3000), arange(0, 15001, 3000), fontsize=9)
    pyplot.yticks(arange(0, 15001, 3000), arange(0, 15001, 3000), fontsize=9)
    pyplot.xlim(-500, 15500)
    pyplot.ylim(-500, 15500)

    figure.align_labels()
    figure.text(0.022, 0.990, "a", va="center", ha="center", fontsize=14)
    figure.text(0.022, 0.660, "b", va="center", ha="center", fontsize=14)
    figure.text(0.503, 0.660, "c", va="center", ha="center", fontsize=14)

    pyplot.savefig(fig_path + "supp-06.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f07():
    """
    Create Figure S7 in the supplementary file.

    TITLE: Effects of sequence diversity and error rate on the average unintended proportion
           across different sequence copy numbers, grouped by sequence diversity.
    """
    shown_data = load_data(file_path=pkg_path + "supp-07.pkl")

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    used_colors = pyplot.get_cmap("RdYlBu_r")(linspace(0.2, 0.8, 8))

    ax = None
    for error_index, (label, values) in enumerate(shown_data.items()):
        ax = pyplot.subplot(2, 4, error_index + 1)
        pyplot.title(("edit errors = %.1f" % ((error_index + 1) / 2)) + "%", fontsize=9)

        pyplot.vlines(arange(0, 51, 10)[1:-1], -10, 0, color="silver", lw=0.75, ls="--", zorder=0)
        pyplot.hlines(arange(-10, 1, 2)[1:-1], 0, 50, color="silver", lw=0.75, ls="--", zorder=0)

        for diversity_index, points in enumerate(values):
            pyplot.plot(arange(len(points)), points, color=used_colors[diversity_index],
                        lw=1.5, zorder=1, label="1E+" + str(diversity_index + 1))
        pyplot.xlabel("sequence copies", fontsize=9)
        pyplot.ylabel("average unintended proportion", fontsize=9)
        pyplot.xticks(arange(0, 51, 10), arange(0, 51, 10), fontsize=9)
        pyplot.yticks(arange(-10, 1, 2),
                      ["1E\N{MINUS SIGN}" + str(v).zfill(2) for v in arange(2, 11, 2)[::-1]] + ["1E+00"], fontsize=9)
        pyplot.xlim(0, 50)
        pyplot.ylim(-10, 0)

    # noinspection PyUnresolvedReferences
    handles, labels = ax.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=8, fontsize=9,
                  title="sequence diversity", title_fontsize=9, frameon=False)

    figure.align_labels()

    figure.text(0.022, 0.990, "a", va="center", ha="center", fontsize=14)
    figure.text(0.268, 0.990, "b", va="center", ha="center", fontsize=14)
    figure.text(0.514, 0.990, "c", va="center", ha="center", fontsize=14)
    figure.text(0.760, 0.990, "d", va="center", ha="center", fontsize=14)
    figure.text(0.022, 0.545, "e", va="center", ha="center", fontsize=14)
    figure.text(0.268, 0.545, "f", va="center", ha="center", fontsize=14)
    figure.text(0.514, 0.545, "g", va="center", ha="center", fontsize=14)
    figure.text(0.760, 0.545, "h", va="center", ha="center", fontsize=14)

    # noinspection PyTypeChecker
    pyplot.tight_layout(rect=[0, 0.08, 1.0, 1.0])

    pyplot.savefig(fig_path + "supp-07.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f08():
    """
    Create Figure S8 in the supplementary file.

    TITLE: Effects of sequence diversity and error rate on the average unintended proportion
           across different sequence copy numbers, grouped by error rate.
    """
    shown_data = load_data(file_path=pkg_path + "supp-08.pkl")

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    used_colors = pyplot.get_cmap("RdYlBu_r")(linspace(0.2, 0.8, 8))

    ax = None
    for diversity_index, (label, values) in enumerate(shown_data.items()):
        ax = pyplot.subplot(2, 4, diversity_index + 1)
        pyplot.title("sequence diversity = 1E+" + str(diversity_index + 1), fontsize=9)

        pyplot.vlines(arange(0, 51, 10)[1:-1], -10, 0, color="silver", lw=0.75, ls="--", zorder=0)
        pyplot.hlines(arange(-10, 1, 2)[1:-1], 0, 50, color="silver", lw=0.75, ls="--", zorder=0)

        for error_index, points in enumerate(values):
            pyplot.plot(arange(len(points)), points, color=used_colors[error_index],
                        lw=1.5, zorder=1, label="%.1f" % ((error_index + 1) / 2))

        pyplot.xlabel("sequence copies", fontsize=9)
        pyplot.ylabel("average unintended proportion", fontsize=9)
        pyplot.xticks(arange(0, 51, 10), arange(0, 51, 10), fontsize=9)
        pyplot.yticks(arange(-10, 1, 2),
                      ["1E\N{MINUS SIGN}" + str(v).zfill(2) for v in arange(2, 11, 2)[::-1]] + ["1E+00"], fontsize=9)
        pyplot.xlim(0, 50)
        pyplot.ylim(-10, 0)

    # noinspection PyUnresolvedReferences
    handles, labels = ax.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=8, fontsize=9,
                  title="% of edit errors", title_fontsize=9, frameon=False)

    figure.align_labels()
    figure.text(0.022, 0.990, "a", va="center", ha="center", fontsize=14)
    figure.text(0.268, 0.990, "b", va="center", ha="center", fontsize=14)
    figure.text(0.514, 0.990, "c", va="center", ha="center", fontsize=14)
    figure.text(0.760, 0.990, "d", va="center", ha="center", fontsize=14)
    figure.text(0.022, 0.545, "e", va="center", ha="center", fontsize=14)
    figure.text(0.268, 0.545, "f", va="center", ha="center", fontsize=14)
    figure.text(0.514, 0.545, "g", va="center", ha="center", fontsize=14)
    figure.text(0.760, 0.545, "h", va="center", ha="center", fontsize=14)

    # noinspection PyTypeChecker
    pyplot.tight_layout(rect=[0, 0.08, 1.0, 1.0])

    pyplot.savefig(fig_path + "supp-08.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f09():
    """
    Create Figure S9 in the supplementary file.

    TITLE: Inner-outer code integration and its loss tolerance.
    """
    shown_data = load_data(file_path=pkg_path + "supp-09.pkl")

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    grid = pyplot.GridSpec(5, 4)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[:2, :])

    labels_1 = ["binary messages", "binary messages", "DNA sequences",
                "DNA sequences", "binary messages", "binary messages"]
    labels_2 = ["apply outer code for payloads", "encode by SPIDER-WEB",
                "wet lab",
                "retrieve by SPIDER-WEB", "repair payloads by outer code"]

    x_locations = linspace(0.05, 0.95, 6)
    for index, label in enumerate(labels_1):
        pyplot.text(x_locations[index], 0.50, label, va="center", ha="center", fontsize=9)
    for index, (former, latter) in enumerate(zip(x_locations[:-1], x_locations[1:])):
        pyplot.annotate("", xy=(former + 0.07, 0.50), xytext=(latter - 0.07, 0.50),
                        arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=0)
        if index != 2:
            pyplot.vlines((former + latter) / 2.0, 0.30, 0.50, lw=0.75, color="k")
            pyplot.text((former + latter) / 2.0, 0.20, labels_2[index], va="center", ha="center", fontsize=9)
        else:
            pyplot.vlines((former + latter) / 2.0, 0.50, 0.70, lw=0.75, color="k")
            pyplot.text((former + latter) / 2.0, 0.80, labels_2[index], va="center", ha="center", fontsize=9)

    pyplot.axis("off")
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)

    cmap = pyplot.get_cmap("RdYlGn").copy()
    cmap.set_under("white")

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2:4, 0])

    pyplot.title("exclusive-or parity", fontsize=9)
    pyplot.pcolormesh(arange(11), arange(51), shown_data["b"][0],
                      cmap="RdYlGn", vmin=0.9, vmax=1.0, shading="gouraud")
    pyplot.xlabel("sequence loss rate", fontsize=9)
    pyplot.ylabel("outer code redundancy", fontsize=9)
    pyplot.xticks(arange(0, 11, 2), ["0%", "2%", "4%", "6%", "8%", "10%"], fontsize=9)
    pyplot.yticks(arange(0, 51, 10), ["0%", "10%", "20%", "30%", "40%", "50%"], fontsize=9)
    pyplot.xlim(0, 10)
    pyplot.ylim(0, 50)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2:4, 1])

    pyplot.title("Reed-Solomon code", fontsize=9)
    pyplot.pcolormesh(arange(11), arange(51), shown_data["b"][1],
                      cmap="RdYlGn", vmin=0.9, vmax=1.0, shading="gouraud")
    pyplot.xlabel("sequence loss rate", fontsize=9)
    pyplot.ylabel("outer code redundancy", fontsize=9)
    pyplot.xticks(arange(0, 11, 2), ["0%", "2%", "4%", "6%", "8%", "10%"], fontsize=9)
    pyplot.yticks(arange(0, 51, 10), ["0%", "10%", "20%", "30%", "40%", "50%"], fontsize=9)
    pyplot.xlim(0, 10)
    pyplot.ylim(0, 50)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2:4, 2])

    pyplot.title("fountain code", fontsize=9)
    pyplot.pcolormesh(arange(11), arange(51), shown_data["b"][2],
                      cmap="RdYlGn", vmin=0.9, vmax=1.0, shading="gouraud")

    pyplot.xlabel("sequence loss rate", fontsize=9)
    pyplot.ylabel("outer code redundancy", fontsize=9)
    pyplot.xticks(arange(0, 11, 2), ["0%", "2%", "4%", "6%", "8%", "10%"], fontsize=9)
    pyplot.yticks(arange(0, 51, 10), ["0%", "10%", "20%", "30%", "40%", "50%"], fontsize=9)
    pyplot.xlim(0, 10)
    pyplot.ylim(0, 50)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[2:4, 3])

    for index, (label, color, width) in enumerate(zip(["xor", "reed-solomon", "fountain"],
                                                      ["#E0A698", "#98B7E6", "#7CBBC6"],
                                                      [2.0, 4.0, 6.0])):
        x, y = shown_data["c"][label]
        line, = pyplot.plot(x, y, color=color, lw=width, zorder=2 - index)
        line.set_path_effects([patheffects.Stroke(linewidth=width + 0.50, foreground="k"), patheffects.Normal()])

    pyplot.plot([-0.05, +1.05], [-0.05, +1.05], lw=0.50, ls=":", color="k", zorder=3)

    pyplot.xlabel("retrieval rate with SPIDER-WEB", fontsize=9)
    pyplot.ylabel("retrieval rate\nwithout SPIDER-WEB", fontsize=9)
    pyplot.xticks(linspace(0.0, 1.0, 6), ["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=9)
    pyplot.yticks(linspace(0.0, 1.0, 6), ["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=9)
    pyplot.xlim(-0.05, +1.05)
    pyplot.ylim(-0.05, +1.05)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[4, :])

    pyplot.fill_between([0.15, 0.57], 0.40, 0.65, ec="k", fc="none", lw=0.75, zorder=1)
    locations, used_colors = linspace(0.15, 0.57, 51), pyplot.get_cmap("RdYlGn")(linspace(0, 1))
    for former, latter, color in zip(locations[:-1], locations[1:], used_colors):
        pyplot.fill_between([former, latter], 0.40, 0.65, ec="none", fc=color, zorder=0)
    locations, labels = linspace(0.15, 0.57, 11), arange(90, 101, 1)
    for index, (location, label) in enumerate(zip(locations, labels)):
        pyplot.vlines(location, 0.30, 0.40, lw=0.75, color="k")
        if index > 0:
            pyplot.text(location, 0.15, str(label) + "%", va="center", ha="center", fontsize=9)
        else:
            pyplot.text(location, 0.15, "≤ " + str(label) + "%", va="center", ha="center", fontsize=9)
    pyplot.text(0.36, 0.85, "retrieval rate", va="center", ha="center", fontsize=9)

    for location, color, label in zip([0.80, 0.50, 0.20], ["#E0A698", "#98B7E6", "#7CBBC6"],
                                      ["exclusive-or parity", "Reed-Solomon code", "fountain code"]):
        pyplot.scatter([0.85], [location], marker="s", ec="k", fc=color, lw=0.75)
        pyplot.text(0.87, location, label, va="center", ha="left", fontsize=9)

    pyplot.axis("off")
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)

    figure.align_labels()
    figure.text(0.021, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.021, 0.62, "b", va="center", ha="center", fontsize=14)
    figure.text(0.764, 0.62, "c", va="center", ha="center", fontsize=14)

    pyplot.savefig(fig_path + "supp-09.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f10():
    """
    Create Figure S10 in the supplementary file.

    TITLE: Evaluation methods and result analysis for handling runtime differences.
    """
    shown_data = load_data(file_path=pkg_path + "supp-10.pkl")

    figure = pyplot.figure(figsize=(10, 6), tight_layout=True)
    grid = pyplot.GridSpec(3, 1)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[0, 0])

    x_locations, y_locations = linspace(0.05, 0.95, 9), linspace(0, 1, 9)
    labels_1 = ["sequence correcting", "sequence ranking", "sequence decoding", ""]
    labels_2 = ["sequence correcting", "sequence ranking", "sequence decoding", "outer code correcting"]

    for index, x_location in enumerate(x_locations[1::2]):
        pyplot.text(x_location, y_locations[5], labels_1[index], va="center", ha="center", fontsize=9)
    for index, x_location in enumerate(x_locations[1::2]):
        pyplot.text(x_location, y_locations[3], labels_2[index], va="center", ha="center", fontsize=9)

    for x_location in x_locations[::2][1:-2]:
        pyplot.annotate("", xy=(x_location - 0.04, y_locations[5]), xytext=(x_location + 0.04, y_locations[5]),
                        arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=0)
    for x_location in x_locations[::2][1:-1]:
        pyplot.annotate("", xy=(x_location - 0.04, y_locations[3]), xytext=(x_location + 0.04, y_locations[3]),
                        arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=0)

    pyplot.annotate("", xy=(x_locations[0] - 0.04, y_locations[5]), xytext=(x_locations[0] + 0.04, y_locations[5]),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=0)
    pyplot.annotate("", xy=(x_locations[0] - 0.04, y_locations[3]), xytext=(x_locations[0] + 0.04, y_locations[3]),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=0)

    pyplot.annotate("", xy=(x_locations[6] - 0.04, y_locations[5]), xytext=(x_locations[8] + 0.04, y_locations[5]),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=0)
    pyplot.annotate("", xy=(x_locations[8] - 0.04, y_locations[3]), xytext=(x_locations[8] + 0.04, y_locations[3]),
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75), zorder=0)

    pyplot.plot([x_locations[0], x_locations[0], x_locations[6], x_locations[6]],
                [y_locations[5], y_locations[7], y_locations[7], y_locations[5]], lw=0.75, ls="--", color="k")
    pyplot.text((x_locations[0] + x_locations[6]) / 2.0, y_locations[7], "default runtime",
                va="center", ha="center", bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.2"), fontsize=9)

    pyplot.plot([x_locations[0], x_locations[0], x_locations[8], x_locations[8]],
                [y_locations[3], y_locations[1], y_locations[1], y_locations[3]], lw=0.75, ls="--", color="k")
    pyplot.text((x_locations[0] + x_locations[8]) / 2.0, y_locations[1], "investigated runtime",
                va="center", ha="center", bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.2"), fontsize=9)
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)
    pyplot.axis("off")

    # noinspection PyTypeChecker
    pyplot.subplot(grid[1:, 0])

    pyplot.scatter([0.0], [1.0], ec="k", fc="w", lw=0.75, zorder=2, label="default")
    pyplot.scatter(arange(1, 51), shown_data["b"][0],
                   ec="k", fc="#E0A698", lw=0.75, zorder=2, label="exclusive-or parity")
    pyplot.scatter(arange(1, 51), shown_data["b"][1],
                   ec="k", fc="#98B7E6", lw=0.75, zorder=2, label="Reed-Solomon code")
    pyplot.scatter(arange(1, 51), shown_data["b"][2],
                   ec="k", fc="#7CBBC6", lw=0.75, zorder=2, label="fountain code")

    pyplot.vlines(arange(51), 0.93, 3.07, color="k", lw=0.5, ls="--", zorder=-1)
    pyplot.hlines(linspace(1.0, 2.7, 18), -0.6, 50.6, color="k", lw=0.5, ls="--", zorder=-1)

    pyplot.legend(loc="upper left", framealpha=1, fontsize=9)
    pyplot.xlabel("outer code redundancy", fontsize=9)
    pyplot.ylabel("average (investigated runtime / default runtime)", fontsize=9)
    pyplot.xticks(arange(51), [str(v) + "%" for v in arange(51)], fontsize=9, rotation=90)
    pyplot.yticks(linspace(1.0, 2.7, 18), ["%.1f" % v for v in linspace(1.0, 2.7, 18)], fontsize=9)
    pyplot.xlim(-0.6, 50.6)
    pyplot.ylim(0.96, 2.76)

    figure.align_labels()
    figure.text(0.028, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.028, 0.70, "b", va="center", ha="center", fontsize=14)

    pyplot.savefig(fig_path + "supp-10.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f11():
    """
    Create Figure S11 in the supplementary file.

    TITLE: The performance difference between "MMSeqs2+MUSCLE5" and "MMSeqs2+USEARCH".
    """
    shown_data = load_data(file_path=pkg_path + "supp-11.pkl")

    figure = pyplot.figure(figsize=(10, 3), tight_layout=True)

    grid = pyplot.GridSpec(1, 3)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[0, 0])

    pairs = shown_data["a"]
    pyplot.plot([0.48, 1.02], [0.48, 1.02], color="k", lw=0.75, ls="--", zorder=0)

    pyplot.text(0.50, 0.90, "pearson = %.4f" % pearsonr(pairs[:, 0], pairs[:, 1])[0],
                va="center", ha="left", fontsize=8)
    pyplot.text(0.50, 0.80, ("maximum = %.1f" % (max(abs(pairs[:, 0] - pairs[:, 1])) / 10.0)) + "%",
                va="center", ha="left", fontsize=8)
    pyplot.scatter(pairs[:, 0] / 1000.0, pairs[:, 1] / 1000.0, ec="k", fc="#63A8EB", zorder=1)
    pyplot.xlabel("direct retrieval rate of MMSeqs2+MUSCLE5", fontsize=9)
    pyplot.ylabel("direct retrieval rate of MMSeqs2+USEARCH", fontsize=9)
    pyplot.xticks(linspace(0.5, 1.0, 6), [str(v) + "%" for v in arange(50, 101, 10)], fontsize=9)
    pyplot.yticks(linspace(0.5, 1.0, 6), [str(v) + "%" for v in arange(50, 101, 10)], fontsize=9)
    pyplot.xlim(0.48, 1.02)
    pyplot.ylim(0.48, 1.02)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[0, 1:])
    pairs = shown_data["b"]

    pyplot.plot([0, 50000], [0, 500], color="k", lw=0.75, ls="--", zorder=0)
    pyplot.plot([0, 5000], [0, 500], color="k", lw=0.75, ls="--", zorder=0)
    pyplot.plot([0, 500], [0, 500], color="k", lw=0.75, ls="--", zorder=0)

    pyplot.text(500, 500, "1X", va="center", ha="center", fontsize=9, zorder=2,
                bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.2"))
    pyplot.text(5000, 500, "10X", va="center", ha="center", fontsize=9, zorder=2,
                bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.2"))
    pyplot.text(49000, 500, "100X", va="center", ha="center", fontsize=9, zorder=2,
                bbox=dict(fc="w", ec="w", boxstyle="round,pad=0.2"))
    pyplot.scatter(pairs[:, 0], pairs[:, 1], ec="k", fc="#FEB697", zorder=1)
    pyplot.xlabel("runtime of MMSeqs2+MUSCLE5", fontsize=9)
    pyplot.ylabel("runtime of MMSeqs2+USEARCH", fontsize=9)
    pyplot.xticks(arange(0, 50001, 5000), arange(0, 50001, 5000), fontsize=9)
    pyplot.yticks(arange(0, 501, 100), arange(0, 501, 100), fontsize=9)
    pyplot.xlim(-1000, 51000)
    pyplot.ylim(-20, 520)

    figure.align_labels()
    figure.text(0.022, 1.000, "a", va="center", ha="center", fontsize=14)
    figure.text(0.355, 1.000, "b", va="center", ha="center", fontsize=14)

    pyplot.savefig(fig_path + "supp-11.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f12():
    """
    Create Figure S12 in the supplementary file.

    TITLE: Different data retrieval approaches in DNA-based data storage.
    """
    pyplot.figure(figsize=(10, 4), tight_layout=True)

    center_line = 0.30

    locations = array([0.10, 0.30, 0.55, 0.95])
    labels = ["preparation and preheating", "DNA sequencing", "data retrieval"]
    # noinspection PyTypeChecker
    for location, label in zip((locations[:-1] + locations[1:]) / 2.0, labels):
        pyplot.text(location, center_line, label, va="center", ha="center",
                    bbox=dict(color="#DDDDDD", boxstyle="round,pad=0.2"), fontsize=8, zorder=2)
    pyplot.fill_between([0.10, 0.95], center_line - 0.03, center_line + 0.03, color="#DDDDDD", zorder=0)
    pyplot.vlines(locations, center_line - 0.03, center_line + 0.03, color="k", lw=1.50, zorder=1)
    pyplot.hlines(center_line, 0.10, locations[1], lw=0.75, ls="--", color="k", zorder=0)
    pyplot.annotate("", xy=(locations[1], center_line), xytext=(locations[-1] + 0.02, center_line), zorder=0,
                    arrowprops=dict(arrowstyle="<|-", color="k", shrinkA=0, shrinkB=0, lw=0.75))
    pyplot.text(0.98, center_line, "time", va="center", ha="left", fontsize=10, zorder=1)

    pyplot.scatter([0.55, 0.95], [center_line + 0.04, center_line + 0.04], s=12, ec="k", fc="w", zorder=4)
    pyplot.scatter([0.32, 0.57], [center_line - 0.04, center_line - 0.04], s=12, ec="k", fc="w", zorder=4)

    pyplot.text(0.02, center_line + 0.52, "conventional retrieval approach", va="center", ha="left", fontsize=10)
    pyplot.vlines(locations[2:], center_line + 0.44, center_line + 0.60, color="k", lw=1.50, zorder=4)
    pyplot.vlines(locations[2], center_line + 0.30, center_line + 0.44, color="silver", lw=0.75, ls="--", zorder=0)
    pyplot.vlines(locations[-1], center_line + 0.04, center_line + 0.44, color="silver", lw=0.75, ls="--", zorder=0)

    ranges = linspace(0.55, 0.95, 5)
    used_centers = (ranges[1:] + ranges[:-1]) / 2.00
    labels = ["sequence clustering", "sequence alignment", "sequence decoding", "message correcting"]
    used_colors = ["#FFC9BA", "#DDA9B0", "#B38EA4", "#847692"]
    # noinspection PyTypeChecker
    for center, label in zip(used_centers, labels):
        pyplot.text(center, center_line + 0.52, label.replace(" ", "\n"), va="center", ha="center", fontsize=8, zorder=3)
    for color, former, latter in zip(used_colors, ranges[:-1], ranges[1:]):
        pyplot.fill_between([former, latter], center_line + 0.44, center_line + 0.60, color=color, lw=0, zorder=0)

    pyplot.text(0.02, center_line + 0.22, "SPIDER-WEB", va="center", ha="left", fontsize=10)
    pyplot.text(0.11, center_line + 0.19, "(regular mode)", va="center", ha="left", fontsize=8)

    pyplot.vlines([0.55], center_line + 0.04, center_line + 0.14, color="silver", lw=0.75, ls="--", zorder=0)
    pyplot.plot([0.62, 0.62, 0.85], [center_line + 0.04, center_line + 0.09, center_line + 0.14],
                color="silver", lw=0.75, ls="--", zorder=0)
    pyplot.scatter([0.62], [center_line + 0.04], s=12, ec="k", fc="w", zorder=4)
    pyplot.vlines([0.55, 0.85], center_line + 0.14, center_line + 0.30, color="k", lw=1.50, zorder=2)

    ranges = linspace(0.55, 0.85, 4)
    used_centers = (ranges[1:] + ranges[:-1]) / 2.00
    labels = ["sequence correcting", "sequence ranking", "sequence decoding"]
    used_colors = ["#D6EFDD", "#ACCCC2", "#86AAA7", "#64888D"]
    # noinspection PyTypeChecker
    for center, label in zip(used_centers, labels):
        pyplot.text(center, center_line + 0.22, label.replace(" ", "\n"), va="center", ha="center", fontsize=8, zorder=3)
    for color, former, latter in zip(used_colors, ranges[:-1], ranges[1:]):
        pyplot.fill_between([former, latter], center_line + 0.14, center_line + 0.30, color=color, lw=0, zorder=1)

    pyplot.text(0.02, center_line - 0.22, "SPIDER-WEB", va="center", ha="left", fontsize=10)
    pyplot.text(0.11, center_line - 0.25, "(non-blocking mode)", va="center", ha="left", fontsize=8)

    pyplot.vlines([0.32, 0.57], center_line - 0.14, center_line - 0.30, color="k", lw=1.50, zorder=4)
    pyplot.vlines([0.32, 0.57], center_line - 0.04, center_line - 0.14, color="silver", lw=0.75, ls="--", zorder=0)
    ranges = linspace(center_line - 0.14, center_line - 0.30, 4)
    positions = [[0.32, 0.55], [0.33, 0.56], [0.37, 0.57]]
    for index, ((former_x, latter_x), former_y, latter_y) in enumerate(zip(positions, ranges[:-1], ranges[1:])):
        pyplot.text((former_x + latter_x) / 2.0, (former_y + latter_y) / 2.0, labels[index],
                    va="center", ha="center", fontsize=8, zorder=3)
        pyplot.fill_between([former_x, latter_x], former_y, latter_y, color=used_colors[index], lw=0, zorder=1)

    pyplot.hlines(1.80, 0.0, 1.0, lw=0.75, color="k")

    pyplot.xlim(-0.01, 1.01)
    pyplot.ylim(-0.01, 0.91)
    pyplot.axis("off")

    pyplot.savefig(fig_path + "supp-12.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f13():
    """
    Create Figure S13 in the supplementary file.

    TITLE: Effects of sequence diversity and error rate on the average of maximum frequency of
           a false-positive sequence candidate in dictionary across different sequence copies,
           grouped by error rate.
    """
    shown_data = load_data(file_path=pkg_path + "supp-13.pkl")

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    used_colors = pyplot.get_cmap("RdYlBu_r")(linspace(0.2, 0.8, 8))

    ax = None
    for error_index, (label, values) in enumerate(shown_data.items()):
        ax = pyplot.subplot(2, 4, error_index + 1)
        pyplot.title(("edit errors = %.1f" % ((error_index + 1) / 2)) + "%", fontsize=9)

        pyplot.vlines(arange(0, 51, 10)[1:-1], 0, 30, color="silver", lw=0.75, ls="--", zorder=0)
        pyplot.hlines(arange(0, 31, 5)[1:-1], 0, 50, color="silver", lw=0.75, ls="--", zorder=0)

        for diversity_index in range(len(values)):
            pyplot.plot(arange(51), [0] + values[diversity_index].tolist(), color=used_colors[diversity_index],
                        lw=1.5, zorder=1, label="1E" + str(diversity_index + 1))
        pyplot.plot([0, 30], [0, 30], color="k", lw=0.75, ls="--", zorder=1)

        pyplot.xlabel("sequence copies", fontsize=9)
        pyplot.ylabel("average of maximum frequency\nof a false-positive case in dictionary", fontsize=9)
        pyplot.xticks(arange(0, 51, 10), arange(0, 51, 10), fontsize=9)
        pyplot.yticks(arange(0, 31, 5), arange(0, 31, 5), fontsize=9)
        pyplot.xlim(0, 50)
        pyplot.ylim(0, 30)

    # noinspection PyUnresolvedReferences
    handles, labels = ax.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=8, fontsize=9,
                  title="sequence diversity", title_fontsize=9, frameon=False)

    figure.align_labels()

    figure.text(0.022, 0.990, "a", va="center", ha="center", fontsize=14)
    figure.text(0.268, 0.990, "b", va="center", ha="center", fontsize=14)
    figure.text(0.514, 0.990, "c", va="center", ha="center", fontsize=14)
    figure.text(0.760, 0.990, "d", va="center", ha="center", fontsize=14)
    figure.text(0.022, 0.545, "e", va="center", ha="center", fontsize=14)
    figure.text(0.268, 0.545, "f", va="center", ha="center", fontsize=14)
    figure.text(0.514, 0.545, "g", va="center", ha="center", fontsize=14)
    figure.text(0.760, 0.545, "h", va="center", ha="center", fontsize=14)

    # noinspection PyTypeChecker
    pyplot.tight_layout(rect=[0, 0.08, 1.0, 1.0])

    pyplot.savefig(fig_path + "supp-13.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f14():
    """
    Create Figure S14 in the supplementary file.

    TITLE: Effects of sequence diversity and error rate on the average of maximum frequency of
           a false-positive sequence candidate in dictionary across different sequence copies,
           grouped by sequence diversity.
    """
    shown_data = load_data(file_path=pkg_path + "supp-14.pkl")

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    used_colors = pyplot.get_cmap("RdYlBu_r")(linspace(0.2, 0.8, 8))

    ax = None
    for diversity_index, (label, values) in enumerate(shown_data.items()):
        ax = pyplot.subplot(2, 4, diversity_index + 1)
        pyplot.title("sequence diversity = 1E+" + str(diversity_index + 1), fontsize=9)

        pyplot.vlines(arange(0, 51, 10)[1:-1], 0, 30, color="silver", lw=0.75, ls="--", zorder=0)
        pyplot.hlines(arange(0, 31, 5)[1:-1], 0, 50, color="silver", lw=0.75, ls="--", zorder=0)

        for error_index in range(len(values)):
            pyplot.plot(arange(51), [0] + values[error_index].tolist(), color=used_colors[error_index],
                        lw=1.5, zorder=1, label="%.1f" % ((error_index + 1) / 2))

        pyplot.plot([0, 30], [0, 30], color="k", lw=0.75, ls="--", zorder=1)
        pyplot.xlabel("sequence copies", fontsize=9)
        pyplot.ylabel("average of maximum frequency\nof a false-positive case in dictionary", fontsize=9)
        pyplot.xticks(arange(0, 51, 10), arange(0, 51, 10), fontsize=9)
        pyplot.yticks(arange(0, 31, 5), arange(0, 31, 5), fontsize=9)
        pyplot.xlim(0, 50)
        pyplot.ylim(0, 30)

    # noinspection PyUnresolvedReferences
    handles, labels = ax.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=8, fontsize=9,
                  title="% of edit errors", title_fontsize=9, frameon=False)

    figure.align_labels()
    figure.text(0.022, 0.990, "a", va="center", ha="center", fontsize=14)
    figure.text(0.268, 0.990, "b", va="center", ha="center", fontsize=14)
    figure.text(0.514, 0.990, "c", va="center", ha="center", fontsize=14)
    figure.text(0.760, 0.990, "d", va="center", ha="center", fontsize=14)
    figure.text(0.022, 0.545, "e", va="center", ha="center", fontsize=14)
    figure.text(0.268, 0.545, "f", va="center", ha="center", fontsize=14)
    figure.text(0.514, 0.545, "g", va="center", ha="center", fontsize=14)
    figure.text(0.760, 0.545, "h", va="center", ha="center", fontsize=14)

    # noinspection PyTypeChecker
    pyplot.tight_layout(rect=[0, 0.08, 1.0, 1.0])

    pyplot.savefig(fig_path + "supp-14.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f15():
    """
    Create Figure S15 in the supplementary file.

    TITLE: Illustration of global constraints.
    """
    figure = pyplot.figure(figsize=(10, 2), tight_layout=True)

    random.seed(2023)

    # noinspection PyTypeChecker
    segment = "".join([random.choice(["A", "C", "G", "T"]) for _ in range(10)])

    used_colors = pyplot.get_cmap("tab10")(linspace(0, 1, 10))

    ax = pyplot.subplot(1, 2, 1)

    pyplot.text(0.01, 0.80, "DNA sequence", va="center", ha="left", fontsize=9)
    pyplot.fill_between([0.20, 0.30], 0.76, 0.84, ec="k", fc="none", lw=0.75)
    pyplot.fill_between([0.90, 1.00], 0.76, 0.84, ec="k", fc="none", lw=0.75)
    pyplot.fill_between([0.30, 0.55], 0.76, 0.84, ec="k", fc="#EEEEEE", lw=0.75)
    pyplot.fill_between([0.65, 0.90], 0.76, 0.84, ec="k", fc="#EEEEEE", lw=0.75)
    locations = linspace(0.30, 0.55, 11)
    for former, latter, nucleotide in zip(locations[:-1], locations[1:], segment):
        pyplot.text((former + latter) / 2.0, 0.795, nucleotide, va="center", ha="center", fontsize=6)
    locations = linspace(0.65, 0.90, 11)
    for former, latter, nucleotide in zip(locations[:-1], locations[1:], segment):
        pyplot.text((former + latter) / 2.0, 0.795, nucleotide, va="center", ha="center", fontsize=6)
    pyplot.fill_between([0.55, 0.65], 0.76, 0.84, ec="k", fc="none", lw=0.75)
    pyplot.plot([0.55, 0.55, 0.65, 0.65], [0.86, 0.90, 0.90, 0.86], color="k", lw=0.5)
    pyplot.text(0.60, 0.95, "spacing", va="center", ha="center", fontsize=9)
    locations = linspace(0.30, 0.55, 11)
    locations_1 = (locations[:-1] + locations[1:]) / 2.0
    locations = linspace(0.65, 0.90, 11)
    locations_2 = (locations[:-1] + locations[1:]) / 2.0
    # noinspection PyTypeChecker
    for index, (former, latter) in enumerate(zip(locations_1, locations_2)):
        lower = 0.74 - (index + 1) * 0.07
        pyplot.plot([former, former, latter, latter], [0.74, lower, lower, 0.74],
                    color=used_colors[index], lw=0.5, ls="--")
        pyplot.text(0.60, lower - 0.01, str(index + 1), bbox=dict(color="w", boxstyle="round,pad=0.2"),
                    va="center", ha="center", fontsize=6)

    pyplot.xlabel("segment duplication", fontsize=9)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.xlim(0, 1.01)
    pyplot.ylim(-0.05, 1)

    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["bottom"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["left"].set_visible(False)

    ax = pyplot.subplot(1, 2, 2)

    pyplot.text(0.01, 0.80, "DNA sequence", va="center", ha="left", fontsize=9)
    pyplot.fill_between([0.20, 0.30], 0.76, 0.84, ec="k", fc="none", lw=0.75)
    pyplot.fill_between([0.90, 1.00], 0.76, 0.84, ec="k", fc="none", lw=0.75)
    pyplot.fill_between([0.30, 0.55], 0.76, 0.84, ec="k", fc="#EEEEEE", lw=0.75)
    pyplot.fill_between([0.65, 0.90], 0.76, 0.84, ec="k", fc="#EEEEEE", lw=0.75)
    locations = linspace(0.30, 0.55, 11)
    for former, latter, nucleotide in zip(locations[:-1], locations[1:], segment):
        pyplot.text((former + latter) / 2.0, 0.795, nucleotide, va="center", ha="center", fontsize=6)
    locations = linspace(0.65, 0.90, 11)
    for former, latter, nucleotide in zip(locations[:-1], locations[1:],
                                          segment.translate(str.maketrans("ACGT", "TGCA"))[::-1]):
        pyplot.text((former + latter) / 2.0, 0.795, nucleotide, va="center", ha="center", fontsize=6)
    pyplot.fill_between([0.55, 0.65], 0.76, 0.84, ec="k", fc="none", lw=0.75)
    pyplot.plot([0.55, 0.55, 0.65, 0.65], [0.86, 0.90, 0.90, 0.86], color="k", lw=0.5)
    pyplot.text(0.60, 0.95, "spacing", va="center", ha="center", fontsize=9)
    locations = linspace(0.30, 0.55, 11)
    locations_1 = (locations[:-1] + locations[1:]) / 2.0
    locations = linspace(0.65, 0.90, 11)
    # noinspection PyUnresolvedReferences
    locations_2 = ((locations[:-1] + locations[1:]) / 2.0)[::-1]
    # noinspection PyTypeChecker
    for index, (former, latter) in enumerate(zip(locations_1, locations_2)):
        lower = 0.74 - (10 - index) * 0.07
        pyplot.plot([former, former, latter, latter], [0.74, lower, lower, 0.74],
                    color=used_colors[9 - index], lw=0.5, ls="--")
        pyplot.text(0.60, lower - 0.01, str(10 - index), bbox=dict(color="w", boxstyle="round,pad=0.2"),
                    va="center", ha="center", fontsize=6)

    pyplot.xlabel("palindrome formation", fontsize=9)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.xlim(0, 1.01)
    pyplot.ylim(-0.05, 1)

    # noinspection PyUnresolvedReferences
    ax.spines["top"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["right"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["bottom"].set_visible(False)
    # noinspection PyUnresolvedReferences
    ax.spines["left"].set_visible(False)

    figure.align_labels()
    figure.text(0.021, 0.990, "a", va="center", ha="center", fontsize=14)
    figure.text(0.506, 0.990, "b", va="center", ha="center", fontsize=14)

    pyplot.savefig(fig_path + "supp-15.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f16():
    """
    Create Figure S16 in the supplementary file.

    TITLE: Valid number of encoded segments in each frame.
    """
    shown_data = load_data(file_path=pkg_path + "supp-16.pkl")

    figure = pyplot.figure(figsize=(10, 8), tight_layout=True)

    grid = pyplot.GridSpec(9, 4)

    for index_1 in range(4):
        for index_2 in range(4):
            values = shown_data[0][index_1 * 4 + index_2]

            # noinspection PyTypeChecker
            pyplot.subplot(grid[index_1 * 2: (index_1 + 1) * 2, index_2])

            pyplot.pcolormesh(arange(21), arange(11), values, cmap="RdYlGn", vmin=0, vmax=3)

            pyplot.title("frame " + str(index_1 * 4 + index_2 + 1).zfill(2), fontsize=9)
            pyplot.xlabel("column index", fontsize=9)
            pyplot.ylabel("raw index", fontsize=9)
            # noinspection PyTypeChecker
            pyplot.xticks(arange(20) + 0.5, [str(v).zfill(2) for v in arange(20) + 1], fontsize=9, rotation=90)
            # noinspection PyTypeChecker
            pyplot.yticks(arange(10) + 0.5, [str(v).zfill(2) for v in arange(10) + 1], fontsize=9)
            pyplot.xlim(0, 20)
            pyplot.ylim(0, 10)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[8, :])

    used_colors, locations = pyplot.get_cmap("RdYlGn")(linspace(0, 1, 100)), linspace(0, 3, 101)

    for former, latter, color in zip(locations[:-1], locations[1:], used_colors):
        pyplot.fill_between([former, latter], 0, 1, lw=0, fc=color)

    pyplot.vlines([log10(shown_data[1]), log10(shown_data[2])], 0, 1, lw=0.75, ls=":", color="k")
    # noinspection PyTypeChecker
    pyplot.text(log10(shown_data[1]) - 0.01, 0.46, str(shown_data[1]), va="center", ha="right", fontsize=9)
    # noinspection PyTypeChecker
    pyplot.text(log10(shown_data[2]) + 0.01, 0.46, str(shown_data[2]), va="center", ha="left", fontsize=9)
    pyplot.hlines(0.5, log10(shown_data[1]), log10(shown_data[2]), lw=0.75, ls=":", color="k")

    pyplot.xlabel("valid number of encoded segments", fontsize=9)
    pyplot.xticks(arange(4), ["1", "10", "100", "1000"], fontsize=9)
    pyplot.yticks([])
    pyplot.xlim(0, 3)
    pyplot.ylim(0, 1)

    figure.align_labels()
    figure.text(0.021, 0.990, "a", va="center", ha="center", fontsize=14)
    figure.text(0.263, 0.990, "b", va="center", ha="center", fontsize=14)
    figure.text(0.506, 0.990, "c", va="center", ha="center", fontsize=14)
    figure.text(0.750, 0.990, "d", va="center", ha="center", fontsize=14)
    figure.text(0.021, 0.768, "e", va="center", ha="center", fontsize=14)
    figure.text(0.263, 0.768, "f", va="center", ha="center", fontsize=14)
    figure.text(0.506, 0.768, "g", va="center", ha="center", fontsize=14)
    figure.text(0.750, 0.768, "h", va="center", ha="center", fontsize=14)
    figure.text(0.021, 0.550, "i", va="center", ha="center", fontsize=14)
    figure.text(0.263, 0.550, "j", va="center", ha="center", fontsize=14)
    figure.text(0.506, 0.550, "k", va="center", ha="center", fontsize=14)
    figure.text(0.750, 0.550, "l", va="center", ha="center", fontsize=14)
    figure.text(0.021, 0.330, "m", va="center", ha="center", fontsize=14)
    figure.text(0.263, 0.330, "n", va="center", ha="center", fontsize=14)
    figure.text(0.506, 0.330, "o", va="center", ha="center", fontsize=14)
    figure.text(0.750, 0.330, "p", va="center", ha="center", fontsize=14)

    pyplot.savefig(fig_path + "supp-16.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f17():
    """
    Create Figure S17 in the supplementary file.

    TITLE: The distribution of minimum spacing between duplicated 10-mers.
    """
    shown_data = load_data(file_path=pkg_path + "supp-17.pkl")

    pyplot.figure(figsize=(10, 10), tight_layout=True)
    for index_1 in range(4):
        for index_2 in range(4):
            pyplot.subplot(4, 4, index_1 * 4 + index_2 + 1)
            data = shown_data[index_1 * 4 + index_2]

            pyplot.title("frame " + str(index_1 * 4 + index_2 + 1).zfill(2), fontsize=9)

            for index in range(10):
                values = data[index]
                x = linspace(min(values), max(values), len(values))
                y = gaussian_kde(values)(x)
                y /= max(y)
                y *= 0.35
                pyplot.fill_between(log10(x), index + 0.5 - y, index + 0.5 + y, ec="k", fc="w", lw=0.75, zorder=1)
                pyplot.hlines(index + 0.5, 1, log10(x[0]), lw=0.75, ls="--", color="k", zorder=1)
                pyplot.hlines(index + 0.5, 4, log10(x[-1]), lw=0.75, ls="--", color="k", zorder=1)

            used_colors, positions = pyplot.get_cmap("RdYlGn")(linspace(0, 1, 100)), linspace(1, 3, 101)
            for former, letter, color in zip(positions[:-1], positions[1:], used_colors):
                pyplot.fill_between([former, letter], -0.1, 10.1, fc=color, lw=0, zorder=0)
            pyplot.fill_between([3, 4], -0.1, 10.1, fc=used_colors[-1], lw=0, zorder=0)

            pyplot.xlabel("minimum spacing between\nduplicated 10-mers", fontsize=9)
            pyplot.ylabel("raw index of data-coding\nDNA sequence in each frame", fontsize=9)
            pyplot.xticks(arange(1, 5), ["1E+1", "1E+2", "1E+3", "1E+4"], fontsize=9)
            # noinspection PyTypeChecker
            pyplot.yticks(arange(10) + 0.5, [str(v).zfill(2) for v in arange(10) + 1], fontsize=9)
            pyplot.xlim(1, 4)
            pyplot.ylim(-0.1, 10.1)

    pyplot.savefig(fig_path + "supp-17.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f18():
    """
    Create Figure S18 in the supplementary file.

    TITLE: The distribution of minimum spacing between a 10-mer and its reverse complement segment.
    """
    shown_data = load_data(file_path=pkg_path + "supp-18.pkl")

    pyplot.figure(figsize=(10, 10), tight_layout=True)
    for index_1 in range(4):
        for index_2 in range(4):
            pyplot.subplot(4, 4, index_1 * 4 + index_2 + 1)
            data = shown_data[index_1 * 4 + index_2]

            pyplot.title("frame " + str(index_1 * 4 + index_2 + 1).zfill(2), fontsize=9)

            for index in range(10):
                values = data[index]
                x = linspace(min(values), max(values), len(values))
                y = gaussian_kde(values)(x)
                y /= max(y)
                y *= 0.35
                pyplot.fill_between(log10(x), index + 0.5 - y, index + 0.5 + y, ec="k", fc="w", lw=0.75, zorder=1)
                pyplot.hlines(index + 0.5, 1, log10(x[0]), lw=0.75, ls="--", color="k", zorder=1)
                pyplot.hlines(index + 0.5, 4, log10(x[-1]), lw=0.75, ls="--", color="k", zorder=1)

            used_colors, positions = pyplot.get_cmap("RdYlGn")(linspace(0, 1, 100)), linspace(1, 3, 101)
            for former, letter, color in zip(positions[:-1], positions[1:], used_colors):
                pyplot.fill_between([former, letter], -0.1, 10.1, fc=color, lw=0, zorder=0)
            pyplot.fill_between([3, 4], -0.1, 10.1, fc=used_colors[-1], lw=0, zorder=0)

            pyplot.xlabel("minimum spacing between\n10-mer and its reverse complement", fontsize=9)
            pyplot.ylabel("raw index of data-coding\nDNA sequence in each frame", fontsize=9)
            pyplot.xticks(arange(1, 5), ["1E+1", "1E+2", "1E+3", "1E+4"], fontsize=9)
            # noinspection PyTypeChecker
            pyplot.yticks(arange(10) + 0.5, [str(v).zfill(2) for v in arange(10) + 1], fontsize=9)
            pyplot.xlim(1, 4)
            pyplot.ylim(-0.1, 10.1)

    pyplot.savefig(fig_path + "supp-18.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f19():
    """
    Create Figure S19 in the supplementary file.

    TITLE: Observation and modelling of instantaneous efficiency.
    """
    sub_groups, points, params = load_data(file_path=pkg_path + "supp-19.pkl")

    figure = pyplot.figure(figsize=(10, 7.5), tight_layout=True)
    grid = pyplot.GridSpec(5, 3)

    colors = ["#F59E65", "#7CC4DE", "#F06288", "#B2BAF4", "#8FF6B9"]

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[0, 0])

    index, flag = 1, True
    for group_index, sub_points in enumerate(sub_groups):
        if group_index == index - 1:
            pyplot.scatter(sub_points[:, 0], log10(sub_points[:, 1] / (8 * 1024 * 1024)),
                           lw=0.75, ec=colors[index - 1], fc="w", s=10, zorder=2, label="trial %d" % index)
        else:
            pyplot.scatter(sub_points[:, 0], log10(sub_points[:, 1] / (8 * 1024 * 1024)),
                           lw=0.75, ec="k", fc="w", s=10, alpha=0.3, zorder=1)

    pyplot.legend(loc="lower left", fontsize=9)
    pyplot.xticks(linspace(0.0, 1.0, 6), arange(0, 101, 20), fontsize=9)
    labels = ["1E\N{MINUS SIGN}7", "1E\N{MINUS SIGN}5", "1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}1", "1E+1"]
    pyplot.yticks(arange(-7, 2, 2), labels, fontsize=9)
    pyplot.xlim(0, 1.02)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[1, 0])

    index, flag = 2, True
    for group_index, sub_points in enumerate(sub_groups):
        if group_index == index - 1:
            pyplot.scatter(sub_points[:, 0], log10(sub_points[:, 1] / (8 * 1024 * 1024)),
                           lw=0.75, ec=colors[index - 1], fc="w", s=10, zorder=2, label="trial %d" % index)
        else:
            pyplot.scatter(sub_points[:, 0], log10(sub_points[:, 1] / (8 * 1024 * 1024)),
                           lw=0.75, ec="k", fc="w", s=10, alpha=0.3, zorder=1)

    pyplot.legend(loc="lower left", fontsize=9)

    pyplot.xticks(linspace(0.0, 1.0, 6), arange(0, 101, 20), fontsize=9)
    labels = ["1E\N{MINUS SIGN}7", "1E\N{MINUS SIGN}5", "1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}1", "1E+1"]
    pyplot.yticks(arange(-7, 2, 2), labels, fontsize=9)
    pyplot.xlim(0, 1.02)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[2, 0])

    index, flag = 3, True
    for group_index, sub_points in enumerate(sub_groups):
        if group_index == index - 1:
            pyplot.scatter(sub_points[:, 0], log10(sub_points[:, 1] / (8 * 1024 * 1024)),
                           lw=0.75, ec=colors[index - 1], fc="w", s=10, zorder=2, label="trial %d" % index)
        else:
            pyplot.scatter(sub_points[:, 0], log10(sub_points[:, 1] / (8 * 1024 * 1024)),
                           lw=0.75, ec="k", fc="w", s=10, alpha=0.3, zorder=1)

    pyplot.legend(loc="lower left", fontsize=9)
    pyplot.ylabel("instantaneous data retrieval efficiency (megabyte per second)", fontsize=9)
    pyplot.xticks(linspace(0.0, 1.0, 6), arange(0, 101, 20), fontsize=9)
    labels = ["1E\N{MINUS SIGN}7", "1E\N{MINUS SIGN}5", "1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}1", "1E+1"]
    pyplot.yticks(arange(-7, 2, 2), labels, fontsize=9)
    pyplot.xlim(0, 1.02)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[3, 0])

    index, flag = 4, True
    for group_index, sub_points in enumerate(sub_groups):
        if group_index == index - 1:
            pyplot.scatter(sub_points[:, 0], log10(sub_points[:, 1] / (8 * 1024 * 1024)),
                           lw=0.75, ec=colors[index - 1], fc="w", s=10, zorder=2, label="trial %d" % index)
        else:
            pyplot.scatter(sub_points[:, 0], log10(sub_points[:, 1] / (8 * 1024 * 1024)),
                           lw=0.75, ec="k", fc="w", s=10, alpha=0.3, zorder=1)

    pyplot.legend(loc="lower left", fontsize=9)
    pyplot.xticks(linspace(0.0, 1.0, 6), arange(0, 101, 20), fontsize=9)
    labels = ["1E\N{MINUS SIGN}7", "1E\N{MINUS SIGN}5", "1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}1", "1E+1"]
    pyplot.yticks(arange(-7, 2, 2), labels, fontsize=9)
    pyplot.xlim(0, 1.02)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[4, 0])

    index, flag = 5, True
    for group_index, sub_points in enumerate(sub_groups):
        if group_index == index - 1:
            pyplot.scatter(sub_points[:, 0], log10(sub_points[:, 1] / (8 * 1024 * 1024)),
                           lw=0.75, ec=colors[index - 1], fc="w", s=10, zorder=2, label="trial %d" % index)
        else:
            pyplot.scatter(sub_points[:, 0], log10(sub_points[:, 1] / (8 * 1024 * 1024)),
                           lw=0.75, ec="k", fc="w", s=10, alpha=0.3, zorder=1)

    pyplot.legend(loc="lower left", fontsize=9)
    pyplot.xlabel("data retrieval rate (%)", fontsize=9)
    pyplot.xticks(linspace(0.0, 1.0, 6), arange(0, 101, 20), fontsize=9)
    labels = ["1E\N{MINUS SIGN}7", "1E\N{MINUS SIGN}5", "1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}1", "1E+1"]
    pyplot.yticks(arange(-7, 2, 2), labels, fontsize=9)
    pyplot.xlim(0, 1.02)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # noinspection PyTypeChecker
    ax = pyplot.subplot(grid[:, 1:])

    pyplot.scatter(points[:, 0], log10(points[:, 1] / (8 * 1024 * 1024)),
                   lw=0.75, ec="k", fc="w", alpha=0.3, zorder=1, label="experimental points")

    x = linspace(min(points[:, 0]), 3199 / 3200, 100)
    y = (params[0] * (1 - x)) / (x ** params[1])
    pyplot.plot(x, log10(y / (8 * 1024 * 1024)), lw=2.0, ls="--", color="#994700", zorder=2, label="fitted curve")

    pyplot.legend(loc="lower left", fontsize=9)
    pyplot.xlabel("data retrieval rate (%)", fontsize=9)
    pyplot.xticks(linspace(0.0, 1.0, 11), arange(0, 101, 10), fontsize=9)
    labels = ["1E\N{MINUS SIGN}7", "1E\N{MINUS SIGN}6", "1E\N{MINUS SIGN}5", "1E\N{MINUS SIGN}4",
              "1E\N{MINUS SIGN}3", "1E\N{MINUS SIGN}2", "1E\N{MINUS SIGN}1", "1E+0", "1E+1"]
    pyplot.yticks(arange(-7, 2), labels, fontsize=9)
    pyplot.xlim(0, 1.01)
    pyplot.ylim(-7, 1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.345, 0.99, "b", va="center", ha="center", fontsize=14)

    pyplot.savefig(fig_path + "supp-19.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_f20():
    """
    Create Figure S20 in the supplementary file.

    TITLE: Operating factor of SPIDER-WEB in the real-world data retrieval demonstration.
    """
    shown_data = load_data(file_path=pkg_path + "supp-20.pkl")

    figure = pyplot.figure(figsize=(10, 5), tight_layout=True)
    grid = pyplot.GridSpec(6, 2)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[:5, 0])

    x_values, y_values, z_values = shown_data["a"]

    pyplot.pcolormesh(x_values, y_values, z_values.T, cmap="binary", vmin=0, vmax=0.03, shading="gouraud")

    pyplot.xlabel("retrieval stage (current runtime / full retrieval runtime)", fontsize=9)
    pyplot.ylabel("SPIDER-WEB operating factor (%)", fontsize=9)
    pyplot.xticks(linspace(0, 1, 11), ["%.1f" % v for v in linspace(0, 1, 11)], fontsize=9)
    pyplot.yticks(linspace(0, 0.1, 11), ["%d" % (v * 100) for v in linspace(0, 0.1, 11)], fontsize=9)
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 0.1)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[5, 0])

    pyplot.title("density", fontsize=9)
    locations, colors = linspace(0.00, 0.03, 101), pyplot.get_cmap("binary")(linspace(0, 1, 100))
    for former, latter, color in zip(locations[:-1], locations[1:], colors):
        pyplot.fill_between([former, latter], 0, 1, fc=color, lw=0, zorder=2)

    pyplot.xticks(linspace(0.00, 0.03, 7),
                  ["%.1f%%" % (v * 100) for v in linspace(0.0, 0.03, 7)], fontsize=9)
    pyplot.yticks([])
    pyplot.xlim(0.00, 0.03)
    pyplot.ylim(0, 1)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[:5, 1])

    matrix = shown_data["b"]

    pyplot.pcolormesh(arange(11), arange(11), matrix.T, vmin=0, vmax=0.002, cmap="summer")
    pyplot.xlabel("retrieval stage", fontsize=9)
    pyplot.ylabel("retrieval stage", fontsize=9)
    pyplot.xticks(arange(10) + 0.5,
                  ["%.1f-%.1f" % (v_1, v_2) for v_1, v_2 in zip(linspace(0.0, 0.9, 10),
                                                                linspace(0.1, 1.0, 10))], fontsize=9)
    pyplot.yticks(arange(10) + 0.5,
                  ["%.1f-%.1f" % (v_1, v_2) for v_1, v_2 in zip(linspace(0.0, 0.9, 10),
                                                                linspace(0.1, 1.0, 10))], fontsize=9)
    pyplot.xlim(0, 10)
    pyplot.ylim(0, 10)

    # noinspection PyTypeChecker
    pyplot.subplot(grid[5, 1])

    pyplot.title("wasserstein distance between two density distributions", fontsize=9)
    locations, colors = linspace(0.00, 0.002, 101), pyplot.get_cmap("summer")(linspace(0, 1, 100))
    for former, latter, color in zip(locations[:-1], locations[1:], colors):
        pyplot.fill_between([former, latter], 0, 1, fc=color, lw=0, zorder=2)

    pyplot.xticks(linspace(0.000, 0.002, 5), ["%.4f" % v for v in linspace(0.000, 0.002, 5)], fontsize=9)
    pyplot.yticks([])
    pyplot.xlim(0.000, 0.002)
    pyplot.ylim(0, 1)

    figure.align_labels()
    figure.text(0.020, 0.99, "a", va="center", ha="center", fontsize=14)
    figure.text(0.495, 0.99, "b", va="center", ha="center", fontsize=14)

    pyplot.savefig(fig_path + "supp-20.pdf", format="pdf", bbox_inches="tight", dpi=600)
    pyplot.close()


def supp_t01():
    """
    Create Table S1 in the supplementary file.

    TITLE: The count of CHN encoding failures across random trials
    """
    summarized_data = load_data(pkg_path + "supp-21.pkl")

    with open(tab_path + "supp-01.csv", "w") as file:
        file.write("file size,message number,failure number\n")
        file.write(",,trial 1,trial 2,trial 3,trial 4,trial 5\n")
        for size in ["1KB", "2KB", "4KB", "8KB", "16KB"]:
            info = summarized_data[size]
            values = [info[0], info[1][0], info[1][1], info[1][2], info[1][3], info[1][4]]
            file.write(size + "," + ",".join([str(value) for value in values]) + "\n")


def supp_t02():
    """
    Create Table S2 in the supplementary file.

    TITLE: The fitted constant theta from non-linear least-squares fitting of the empirical scaling law.
    """
    summarized_data = load_data(pkg_path + "supp-22.pkl")

    with open(tab_path + "supp-02.csv", "w") as file:
        file.write("error rate,0.5%,1.0%,1.5%,2.0%,2.5%,3.0%,3.5%,4.0%\n")
        file.write("fitted constant theta 1" + "".join([",%.4f" % v for v in summarized_data["a"]]) + "\n")
        file.write("fitted constant theta 2" + "".join([",%.4f" % v for v in summarized_data["b"]]) + "\n")
        file.write("coefficient of determination" + "".join([",%.4f" % v for v in summarized_data["c"]]) + "\n")
        file.write("root mean square error" + "".join([",%.4f" % v for v in summarized_data["d"]]) + "\n")


def supp_t03():
    """
    Create Table S3 in the supplementary file.

    TITLE: First observed instantaneous efficiency across replicate trials.
    """
    summarized_data = load_data(pkg_path + "supp-23.pkl")

    with open(tab_path + "supp-03.csv", "w") as file:
        file.write("Replicate trial,1,2,3,4,5\n")
        file.write("First instantaneous efficiency" + "".join([",%.2f" % v for v in summarized_data[0]]) + "\n")
        file.write("Observation time" + "".join([",%.4f" % v for v in summarized_data[1]]) + "\n")


def supp_t04():
    """
    Create Table S4 in the supplementary file.

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
