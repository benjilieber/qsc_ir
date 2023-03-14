import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
from matplotlib import ticker

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import seaborn as sns

sns.set(style='ticks')


def plot_results1(file_name, q_filter=None, qer_filter=None, snr_filter=None, n_filter=None, errorType="frame"):
    df = pd.read_csv(file_name)
    df["q"] = df.base
    df["qer"] = df.p_err * (-1) + 1
    df["n"] = np.ceil(np.log2(df.key_length))
    df["N"] = np.exp2(df.n)
    df["frameErrorProb"] = df.is_success * (-1) + 1

    if n_filter is not None:
        df = df[df.n.isin(n_filter)]

    grouped_by_q = df.groupby("q")
    for q in grouped_by_q.groups.keys():
        if q_filter is not None and q not in q_filter:
            continue
        cur_q_group = grouped_by_q.get_group(q)

        grouped_by_qer = cur_q_group.groupby("qer")
        for qer in grouped_by_qer.groups.keys():
            if qer_filter is not None and qer not in qer_filter:
                continue
            cur_qer_group = grouped_by_qer.get_group(qer)
            theoretic_key_rate = cur_qer_group.theoretic_key_rate.iloc[0]
            plt.axvline(x=theoretic_key_rate)

            grouped_by_N = cur_qer_group.groupby("N")
            for N in grouped_by_N.groups.keys():
                cur_group = grouped_by_N.get_group(N)
                keyRate = cur_group.key_rate
                plt.xlabel('key rate')

                if errorType == "frame":
                    errorProb = cur_group.frameErrorProb
                    plt.ylabel('frame error probability')
                elif errorType == "symbol":
                    errorProb = cur_group.symbolErrorProb
                    plt.ylabel('symbol error probability')
                elif errorType == "ml_lower_bound":
                    errorProb = cur_group.FailActualSmallerThanMin
                    plt.ylabel('ML lower bound error probability')
                else:
                    raise "Unknown error type: " + errorType

                plt.scatter(keyRate, errorProb, label=str(N), s=cur_group.block_length)

            plt.legend()
            plt.title(
                'q=' + str(q) + ', qer=' + str(qer) + ', max_block_length=' + str(cur_qer_group.block_length.max()))
            plt.show()


def plot_results2(file_name, q_filter=None, qer_filter=None, n_filter=None):
    df = pd.read_csv(file_name)

    if n_filter is not None:
        df = df[df.n.isin(n_filter)]

    df['finalRate'] = df.keyRate * (1 - df.frameErrorProb)
    grouped_by_q = df.groupby("q")
    for q in grouped_by_q.groups.keys():
        if q_filter is not None and q not in q_filter:
            continue
        cur_q_group = grouped_by_q.get_group(q)
        grouped_by_qer = cur_q_group.groupby("qer")
        for qer in grouped_by_qer.groups.keys():
            if qer_filter is not None and qer not in qer_filter:
                continue
            cur_qer_group = grouped_by_qer.get_group(qer)
            theoretic_key_rate = cur_qer_group.theoreticKeyRate.iloc[0]
            plt.axhline(y=theoretic_key_rate, linewidth=4, label="TheoreticKeyRate")
            if q == 3:
                hagai_key_rate1 = qer / 2
                plt.axhline(y=hagai_key_rate1, color='r', label="Hagai1Rate")
                hagai_key_rate2 = math.log(3, 2) * qer / 4
                plt.axhline(y=hagai_key_rate2, color='y', label="Hagai2Rate")

            grouped_by_n = cur_qer_group.groupby("n")
            for key in grouped_by_n.groups.keys():
                cur_group = grouped_by_n.get_group(key)
                plt.scatter(cur_group.maxListSize, cur_group.finalRate, label=str(key))

            plt.legend()
            plt.title('q=' + str(q) + ', qer=' + str(qer))
            plt.xlabel('maxListSize')
            plt.ylabel('final rate')
            plt.show()


def plot_results3(file_name, q_filter=None, qer_filter=None, snr_filter=None, n_filter=None, rate_filter=None,
                  max_list_size_filter=None, errorType="frame", adjustedKeyRate=False):
    df = pd.read_csv(file_name)

    if n_filter is not None:
        df = df[df.n.isin(n_filter)]
    if rate_filter is not None:
        df = df[df.rate.isin(rate_filter)]
    if max_list_size_filter is not None:
        df = df[df.maxListSize.isin(max_list_size_filter)]

    grouped_by_q = df.groupby("q")
    for q in grouped_by_q.groups.keys():
        if q_filter is not None and q not in q_filter:
            continue
        cur_q_group = grouped_by_q.get_group(q)

        grouped_by_qer = cur_q_group.groupby("qer")
        for qer in grouped_by_qer.groups.keys():
            if qer_filter is not None and qer not in qer_filter:
                continue
            cur_qer_group = grouped_by_qer.get_group(qer)

            if not adjustedKeyRate:
                cur_qer_group["origKeyRate"] = np.log2(cur_qer_group.q) * cur_qer_group.numInfoQudits / cur_qer_group.N
                minRate = np.min(cur_qer_group["origKeyRate"])
                maxRate = np.max(cur_qer_group["origKeyRate"])
            else:
                minRate = np.min(cur_qer_group["keyRate"])
                maxRate = np.max(cur_qer_group["keyRate"])

            theoretic_key_rate = cur_qer_group.theoreticKeyRate.iloc[0]
            if errorType == "frame":
                maxProb = np.max(cur_qer_group.frameErrorProb)
            elif errorType == "symbol":
                maxProb = np.max(cur_qer_group.symbolErrorProb)
            elif errorType == "ml_lower_bound":
                maxProb = np.max(cur_qer_group.FailActualSmallerThanMin)
            elif errorType == "frame_ml":
                maxProb = np.max(1.0 - cur_qer_group.SuccessActualIsMax)

            grouped_by_N = cur_qer_group.groupby("N")
            for N in grouped_by_N.groups.keys():
                plt.axvline(x=theoretic_key_rate)
                cur_N_group = grouped_by_N.get_group(N)
                grouped_by_rate = cur_N_group.groupby("rate")
                for rate in grouped_by_rate.groups.keys():
                    cur_group = grouped_by_rate.get_group(rate)

                    if not adjustedKeyRate:
                        keyRate = cur_group.origKeyRate
                        plt.xlabel('key rate')
                    else:
                        keyRate = cur_group.keyRate
                        plt.xlabel('key rate, adjusted by list size')

                    if errorType == "frame":
                        errorProb = cur_group.frameErrorProb
                        plt.ylabel('frame error probability')
                    elif errorType == "symbol":
                        errorProb = cur_group.symbolErrorProb
                        plt.ylabel('symbol error probability')
                    elif errorType == "ml_lower_bound":
                        errorProb = cur_group.FailActualSmallerThanMin
                        plt.ylabel('ML lower bound error probability')
                    elif errorType == "frame_ml":
                        errorProb = 1.0 - cur_group.SuccessActualIsMax
                        plt.ylabel('frame list-decode ML error probability')
                    else:
                        raise "Unknown error type: " + errorType

                    plt.scatter(keyRate, errorProb, label=str(
                        round(rate * math.log2(q) / theoretic_key_rate, 2)) + ", maxListSize = " + str(
                        np.max(cur_group.maxListSize)), s=10 * np.log2(cur_group.maxListSize))
                    plt.plot(keyRate, errorProb)

                plt.legend()
                plt.title('q=' + str(q) + ', qer=' + str(qer) + ', N=' + str(N))
                plt.xlim([max(0, minRate - 0.1), maxRate + 0.1])
                plt.ylim([0, maxProb])
                plt.show()


def plot_results4(file_name):
    df = pd.read_csv(file_name)
    df["q"] = df.base
    df["qer"] = df.p_err * (-1) + 1
    df["n"] = np.ceil(np.log2(df.key_length))
    df["N"] = np.exp2(df.n)
    df['key_rate_vs_theoretic_key_rate'] = df.key_rate / df.theoretic_key_rate
    n_list = sorted(df.n.unique())

    grouped_by_q = df.groupby("q")
    for q in grouped_by_q.groups.keys():
        cur_q_group = grouped_by_q.get_group(q)
        sns.relplot(data=cur_q_group, x='qer', y='key_rate_vs_theoretic_key_rate', hue='n', hue_order=n_list,
                    size='block_length')
        plt.title('q=' + str(q), y=0.965)
        plt.axhline(y=0.0)
        plt.axhline(y=1.0)
        plt.xlabel('qer')
        plt.ylabel('key rate / theoretic key rate')
        # plt.legend()
        plt.show()


def add_yield(df):
    df['yield'] = df.success_rate * df.key_rate_success_only


def add_efficiency(df):
    df.loc[df.p_err.eq(0.0), 'optimal_leak'] = np.log2(df.q - 1)
    df.loc[df.p_err.ne(0.0), 'optimal_leak'] = -df.p_err * np.log2(df.p_err) - (1 - df.p_err) * np.log2(
        (1 - df.p_err) / (df.q - 1))
    df['efficiency'] = df.encoding_size_rate * np.log2(df.q) / df.optimal_leak


def get_theoretic_key_rate_single(q, p_err):
    if p_err == 0.0:
        return math.log2(q / (q - 1))
    return p_err * np.log2(p_err) + (1 - p_err) * np.log2((1 - p_err) / (q - 1)) + math.log2(q)


def get_theoretic_key_rate(q, p_err_array):
    # p_err is a numpy array
    theoretic_key_rate = np.empty_like(p_err_array)

    theoretic_key_rate[p_err_array == 0.0] = math.log2(q / (q - 1))

    non_zero_p_err = p_err_array[p_err_array != 0.0]
    theoretic_key_rate[p_err_array != 0.0] = non_zero_p_err * np.log2(non_zero_p_err) + (1 - non_zero_p_err) * np.log2(
        (1 - non_zero_p_err) / (q - 1)) + math.log2(q)

    return theoretic_key_rate


def plot_error_exponent(df, q_filter=None, p_err_filter=None):
    df["q"] = df.q
    df["qer"] = df.p_err * (-1) + 1
    df["n"] = np.ceil(np.log2(df.N))
    df["N"] = np.exp2(df.n).astype(int)
    df["FER"] = 1 - df.success_rate

    if q_filter is not None:
        df = df[df.q.isin(q_filter)]
    if p_err_filter is not None:
        df = df[df.p_err.isin(p_err_filter)]

    color_name = "key_rate"
    color_col = "key_rate_success_only"
    use_color_log = False

    grouped_by_q = df.groupby("q")
    for q in grouped_by_q.groups.keys():
        cur_q_group = grouped_by_q.get_group(q)
        grouped_by_p_err = cur_q_group.groupby("p_err")
        for p_err in grouped_by_p_err.groups.keys():
            cur_group = grouped_by_p_err.get_group(p_err)

            fig, ax = plt.subplots()

            max_key_rate_per_xy = cur_group.groupby(["N", "FER"]).key_rate_success_only.aggregate('max').reset_index()

            scatter = ax.scatter(max_key_rate_per_xy.N, max_key_rate_per_xy.FER, s=10,
                                 c=max_key_rate_per_xy.key_rate_success_only, cmap='gist_rainbow',
                                 norm=colors.Normalize())

            plt.axhline(y=0.0, color="red")

            plt.xlabel('N')
            plt.ylabel("FER")
            legend = ax.legend(*scatter.legend_elements(), title=color_name)
            ax.add_artist(legend)

            plt.title("Error exponent - q={q},p_err={p_err}".format(q=q, p_err=p_err))
            plt.savefig("error_exponent,q={q},p_err={p_err},color={color_name}.png".format(q=q, p_err=p_err,
                                                                                           color_name=color_name))
            plt.show()


def plot_scaling_exponent(df, q_filter=None, p_err_filter=None, success_rate_filter=None):
    df["q"] = df.q
    df["qer"] = df.p_err * (-1) + 1
    df["n"] = np.ceil(np.log2(df.N))
    df["N"] = np.exp2(df.n).astype(int)
    df["gap"] = df.theoretic_key_rate - df.key_rate_success_only

    if q_filter is not None:
        df = df[df.q.isin(q_filter)]
    if p_err_filter is not None:
        df = df[df.p_err.isin(p_err_filter)]
    if success_rate_filter is not None:
        df = df[df.success_rate.isin(success_rate_filter)]

    # # color by L
    color_name = "L"
    color_col = "goal_candidates_num"
    use_color_log = True

    # # color by block length
    # color_name = "block_length"
    # color_col = "block_length"
    # use_color_log = False

    # # color by max num encoded blocks
    # color_name = "max_num_indices_to_encode"
    # color_col = "max_num_indices_to_encode"
    # use_color_log = True

    grouped_by_q = df.groupby("q")
    for q in grouped_by_q.groups.keys():
        cur_q_group = grouped_by_q.get_group(q)
        grouped_by_success_rate = cur_q_group.groupby("success_rate")
        for success_rate in grouped_by_success_rate.groups.keys():
            cur_success_rate_group = grouped_by_success_rate.get_group(success_rate)
            grouped_by_p_err = cur_success_rate_group.groupby("p_err")
            for p_err in grouped_by_p_err.groups.keys():
                cur_group = grouped_by_p_err.get_group(p_err)

                fig, ax = plt.subplots()

                min_by_color = cur_group.groupby(["N", color_col]).gap.aggregate('min').reset_index()
                if use_color_log:
                    scatter = ax.scatter(min_by_color.N, min_by_color.gap, s=10, c=min_by_color[color_col],
                                         cmap='gist_rainbow', norm=colors.LogNorm())
                else:
                    scatter = ax.scatter(min_by_color.N, min_by_color.gap, s=10,
                                         c=min_by_color[color_col], cmap='gist_rainbow', norm=colors.Normalize())

                plt.axhline(y=0.0, color="red")

                plt.xlabel('N')
                plt.ylabel("gap")
                legend = ax.legend(*scatter.legend_elements(), title=color_name)
                ax.add_artist(legend)
                # plt.title('q=' + str(q) + ', N=' + str(N))  # + ', max_block_length=' + str(grouped_by_N.block_length.max()))

                plt.title("Scaling exponent - q={q},p_err={p_err},success_rate={success_rate}".format(q=q, p_err=p_err,
                                                                                                      success_rate=success_rate))
                plt.savefig(
                    "scaling_exponent,q={q},p_err={p_err},success_rate={success_rate},color={color_name}.png".format(
                        q=q, p_err=p_err, success_rate=success_rate, color_name=color_name))
                plt.show()


def plot_vs_qser(df, y_name, q_filter=None, n_filter=None):
    df["q"] = df.q
    df["qer"] = df.p_err * (-1) + 1
    df["n"] = np.ceil(np.log2(df.N))
    df["N"] = np.exp2(df.n).astype(int)
    add_yield(df)
    add_efficiency(df)
    df['logL'] = np.log(df.goal_candidates_num)
    df['logI'] = np.log(df.max_num_indices_to_encode)
    df['Kbps'] = np.reciprocal(df.time_rate) * np.log2(df.q) / 1000
    df['total_communication_rate'] = df.total_communication_rate * np.log2(df.q)

    # # color by L
    color_name = "L"
    color_col = "goal_candidates_num"
    use_color_log = True

    # # color by block length
    # color_name = "block_length"
    # color_col = "block_length"
    # use_color_log = False

    # # color by max num encoded blocks
    # color_name = "max_num_indices_to_encode"
    # color_col = "max_num_indices_to_encode"
    # use_color_log = True

    if n_filter is not None:
        df = df[df.n.isin(n_filter)]
    if q_filter is not None:
        df = df[df.q.isin(q_filter)]

    grouped_by_q = df.groupby("q")
    for q in grouped_by_q.groups.keys():
        cur_q_group = grouped_by_q.get_group(q)
        grouped_by_N = cur_q_group.groupby("N")
        for N in grouped_by_N.groups.keys():
            cur_group = grouped_by_N.get_group(N)

            fig, ax = plt.subplots()

            Q = cur_group.p_err
            Q_range_non_zero = np.logspace(-4, -1, 100)
            Q_range = np.concatenate(([0], Q_range_non_zero))
            if y_name == "keyRate":
                y_col_name = "key_rate_success_only"

                # color_df = cur_group[color_col]
                # keyRate = cur_group[y_col_name]
                # if use_color_log:
                #   scatter = ax.scatter(Q + 0.000001, keyRate, s=1, c=color_df, cmap='gist_rainbow', norm=colors.LogNorm())
                # else:
                #   scatter = ax.scatter(Q + 0.000001, keyRate, s=1, c=color_df, cmap='gist_rainbow', norm=colors.Normalize())

                max_by_color = cur_group.groupby(["p_err", color_col])[y_col_name].aggregate('max').reset_index()
                if use_color_log:
                    scatter = ax.scatter(max_by_color.p_err + 0.000001, max_by_color[y_col_name], s=10,
                                         c=max_by_color[color_col], cmap='gist_rainbow', norm=colors.LogNorm())
                else:
                    scatter = ax.scatter(max_by_color.p_err + 0.000001, max_by_color[y_col_name], s=10,
                                         c=max_by_color[color_col], cmap='gist_rainbow', norm=colors.Normalize())

                # ax.scatter(Q + 0.000001, keyRate, s=1, c=cur_group.max_num_indices_to_encode, cmap='gist_rainbow', label=cur_group.max_num_indices_to_encode)
                theoretic_key_rate_zero = math.log2(q / (q - 1))
                theoretic_key_rate_non_zero = Q_range_non_zero * np.log2(Q_range_non_zero) + (
                        1 - Q_range_non_zero) * np.log2(
                    (1 - Q_range_non_zero) / (q - 1)) + math.log2(q)
                theoretic_key_rate = np.concatenate(([theoretic_key_rate_zero], theoretic_key_rate_non_zero))
                plt.plot(Q_range, theoretic_key_rate, 'r')
            elif y_name == "yield":
                y_col_name = "yieldRate"
                cur_group[y_col_name] = cur_group.success_rate * cur_group.key_rate_success_only

                # scatter = plt.scatter(Q, cur_group[y_col_name])

                max_by_color = cur_group.groupby(["p_err", color_col])[y_col_name].aggregate('max').reset_index()
                if use_color_log:
                    scatter = ax.scatter(max_by_color.p_err + 0.000001, max_by_color[y_col_name], s=10,
                                         c=max_by_color[color_col], cmap='gist_rainbow', norm=colors.LogNorm())
                else:
                    scatter = ax.scatter(max_by_color.p_err + 0.000001, max_by_color[y_col_name], s=10,
                                         c=max_by_color[color_col], cmap='gist_rainbow', norm=colors.Normalize())

                theoretic_key_rate_zero = math.log2(q / (q - 1))
                theoretic_key_rate_non_zero = Q_range_non_zero * np.log2(Q_range_non_zero) + (
                        1 - Q_range_non_zero) * np.log2(
                    (1 - Q_range_non_zero) / (q - 1)) + math.log2(q)
                theoretic_key_rate = np.concatenate(([theoretic_key_rate_zero], theoretic_key_rate_non_zero))
                plt.plot(Q_range, theoretic_key_rate, 'r')
            elif y_name == "efficiency":
                y_col_name = "efficiency"

                def calc_optimal_leak_rate(QSER):
                    if QSER == 0:
                        return np.log2(q - 1)
                    else:
                        return (-(QSER + 0.000001) * np.log2(QSER + 0.000001) - (
                                1 - (QSER + 0.000001)) * np.log2((1 - QSER) / (q - 1)))

                cur_group["optimal_leak_rate"] = Q.apply(lambda QSER: calc_optimal_leak_rate(QSER))

                cur_group[y_col_name] = cur_group.encoding_size_rate * np.log2(q) / cur_group.optimal_leak_rate

                # scatter = plt.scatter(Q, cur_group[y_col_name])
                min_by_color = cur_group.groupby(["p_err", color_col])[y_col_name].aggregate('min').reset_index()
                if use_color_log:
                    scatter = ax.scatter(min_by_color.p_err + 0.000001, min_by_color[y_col_name], s=10,
                                         c=min_by_color[color_col], cmap='gist_rainbow', norm=colors.LogNorm())
                else:
                    scatter = ax.scatter(min_by_color.p_err + 0.000001, min_by_color[y_col_name], s=10,
                                         c=min_by_color[color_col], cmap='gist_rainbow', norm=colors.Normalize())
                plt.axhline(y=1.0, color="r")
            else:
                raise "bad y_name!"

            ax.set_xscale("log")
            # plt.xticks(Q.unique()+0.00001)
            # ax.set_xscale("log")
            # ax.set_xlim(Q.min()+1, Q.max()+1)
            if y_name in ["keyRate", "yield"]:
                ax.set_ylim(0.0, 0.61)
            elif y_name == "efficiency":
                ax.set_ylim(0.99, 1.3)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x - 0.000001)))
            ax.xaxis.set_major_locator(ticker.FixedLocator(Q + 0.000001))
            plt.xlabel('QSER')
            plt.ylabel(y_name)
            legend = ax.legend(*scatter.legend_elements(), title=color_name)
            ax.add_artist(legend)
            # plt.title('q=' + str(q) + ', N=' + str(N))  # + ', max_block_length=' + str(grouped_by_N.block_length.max()))

            plt.title("N = " + str(N))
            plt.savefig("{y_name}-vs-QSER,q={q},N={N},color={color_name}.png".format(y_name=y_name, q=q, N=N,
                                                                                     color_name=color_name))
            plt.show()


def plot_key_rate_vs_qser(file_name):
    return plot_vs_qser(file_name, "keyRate")


def plot_yield_vs_qser(file_name):
    return plot_vs_qser(file_name, "yield")


def plot_efficiency_vs_qser(file_name):
    return plot_vs_qser(file_name, "efficiency")


# def plot_Kbps_vs_yield(df, q_filter=None, n_filter=None, color_col=None):
#     return plot_y_vs_x(df, "yield", "Kbps", q_filter, n_filter, color_col=color_col)
# def plot_communication_vs_yield(df, q_filter=None, n_filter=None, color_col=None):
#     return plot_y_vs_x(df, "yield", "total_communication_rate", q_filter, n_filter, color_col=color_col)

def plot_time_vs_yield(df, q_filter=None, n_filter=None, p_err_filter=None):
    df["q"] = df.q
    df["qer"] = df.p_err * (-1) + 1
    df["n"] = np.ceil(np.log2(df.N))
    df["N"] = np.exp2(df.n).astype(int)

    if q_filter is not None:
        df = df[df.q.isin(q_filter)]
    if n_filter is not None:
        df = df[df.n.isin(n_filter)]
    if p_err_filter is not None:
        df = df[df.p_err.isin(p_err_filter)]

    grouped_by_q = df.groupby("q")
    for q in grouped_by_q.groups.keys():
        cur_q_group = grouped_by_q.get_group(q)
        grouped_by_p_err = cur_q_group.groupby("p_err")
        for p_err in grouped_by_p_err.groups.keys():
            cur_group = grouped_by_p_err.get_group(p_err)

            fig, ax = plt.subplots()

            cur_group["yield_rate"] = cur_group.success_rate * cur_group.key_rate_success_only

            min_by_color = cur_group.groupby(["N", "time_rate"]).yield_rate.aggregate('min').reset_index()
            scatter = ax.scatter(min_by_color.yield_rate, min_by_color.time_rate, s=1, c=min_by_color.N,
                                 cmap='gist_rainbow', norm=colors.LogNorm())

            if p_err == 0:
                theoretic_key_rate = math.log2(q) - math.log2((q - 1))
            else:
                theoretic_key_rate = p_err * np.log2(p_err) + (1 - p_err) * np.log2((1 - p_err) / (q - 1)) + math.log2(
                    q)
            plt.axvline(x=theoretic_key_rate, color="r")

            plt.xlabel('yield_rate')
            ax.set_yscale("log")
            plt.ylabel("time_rate (log(s/N))")
            legend = ax.legend(*scatter.legend_elements(), title="N")
            ax.add_artist(legend)

            plt.title("Yield rate vs time rate - q={q},p_err={p_err}".format(q=q, p_err=p_err))
            plt.savefig("yield_rate-vs-time_rate,q={q},p_err={p_err},color=N.png".format(q=q, p_err=p_err))
            plt.show()


def plot_time_vs_N(df, q_filter=None, n_filter=None, p_err_filter=None):
    df["q"] = df.q
    df["qer"] = df.p_err * (-1) + 1
    df["n"] = np.ceil(np.log2(df.N))
    df["N"] = np.exp2(df.n).astype(int)

    if q_filter is not None:
        df = df[df.q.isin(q_filter)]
    if n_filter is not None:
        df = df[df.n.isin(n_filter)]
    if p_err_filter is not None:
        df = df[df.p_err.isin(p_err_filter)]

    grouped_by_q = df.groupby("q")
    for q in grouped_by_q.groups.keys():
        cur_q_group = grouped_by_q.get_group(q)
        grouped_by_p_err = cur_q_group.groupby("p_err")
        for p_err in grouped_by_p_err.groups.keys():
            cur_group = grouped_by_p_err.get_group(p_err)

            fig, ax = plt.subplots()

            cur_group["yield_rate"] = cur_group.success_rate * cur_group.key_rate_success_only

            max_by_color = cur_group.groupby(["N", "time_rate"]).yield_rate.aggregate('max').reset_index()
            scatter = ax.scatter(max_by_color.N, max_by_color.time_rate, s=1, c=max_by_color.yield_rate,
                                 cmap='gist_rainbow', norm=colors.LogNorm())

            plt.xlabel('N, log scale')
            ax.set_xscale("log")
            plt.ylabel("time_rate (s/N), log scale")
            ax.set_yscale("log")
            legend = ax.legend(*scatter.legend_elements(), title="yield_rate")
            ax.add_artist(legend)

            plt.title("N vs time rate - q={q},p_err={p_err}".format(q=q, p_err=p_err))
            plt.savefig("N-vs-time_rate,q={q},p_err={p_err},color=yield_rate.png".format(q=q, p_err=p_err))
            plt.show()


def plot_time_vs_block_length(df, q_filter=None, n_filter=None, p_err_filter=None):
    df["q"] = df.q
    df["qer"] = df.p_err * (-1) + 1
    df["n"] = np.ceil(np.log2(df.N))
    df["N"] = np.exp2(df.n).astype(int)

    if q_filter is not None:
        df = df[df.q.isin(q_filter)]
    if n_filter is not None:
        df = df[df.n.isin(n_filter)]
    if p_err_filter is not None:
        df = df[df.p_err.isin(p_err_filter)]

    grouped_by_q = df.groupby("q")
    for q in grouped_by_q.groups.keys():
        cur_q_group = grouped_by_q.get_group(q)
        grouped_by_p_err = cur_q_group.groupby("p_err")
        for p_err in grouped_by_p_err.groups.keys():
            cur_p_err_group = grouped_by_p_err.get_group(p_err)
            grouped_by_N = cur_p_err_group.groupby("N")
            for N in grouped_by_N.groups.keys():
                cur_group = grouped_by_N.get_group(N)

                fig, ax = plt.subplots()

                cur_group["yield_rate"] = cur_group.success_rate * cur_group.key_rate_success_only

                scatter = ax.scatter(cur_group.block_length, cur_group.time_rate, s=1, c=cur_group.yield_rate,
                                     cmap='gist_rainbow', norm=colors.LogNorm())

                plt.xlabel('block length')
                plt.ylabel("time_rate (s/N), log scale")
                plt.ylim([0.001, 1])
                ax.set_yscale("log")
                legend = ax.legend(*scatter.legend_elements(), title="yield_rate")
                ax.add_artist(legend)

                plt.title("Block length vs time rate - q={q},p_err={p_err},N={N}".format(q=q, p_err=p_err, N=N))
                plt.savefig(
                    "block_length-vs-time_rate,q={q},p_err={p_err},N={N},color=yield_rate.png".format(q=q, p_err=p_err,
                                                                                                      N=N))
                plt.show()


def sanity_checks(df):
    grouped_by_success_rate = df.groupby("success_rate")
    for success_rate in grouped_by_success_rate.groups.keys():
        cur_success_rate_group = grouped_by_success_rate.get_group(success_rate)
        print("sample size: " + str(len(cur_success_rate_group)))
        print("desired success rate: " + str(success_rate))
        print("actual success rate: " + str(cur_success_rate_group["is_success"].mean()))
        print("----------------------------------")


q_filter = [3]
file_name = "results/history_agg.csv"
df = pd.read_csv(file_name)
df = df[df.key_length < 8000]

# sanity_checks("fake_results.csv")

# plot_key_rate_vs_qser(df)
# plot_yield_vs_qser(df)
# plot_efficiency_vs_qser(df)
# plot_error_exponent(df)
# plot_scaling_exponent(df)
# plot_time_vs_yield(df)
# plot_time_vs_N(df)
plot_time_vs_block_length(df)
