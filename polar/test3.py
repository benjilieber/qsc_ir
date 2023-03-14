import sys
import os
sys.path.append(os.getcwd())

import math
from timeit import default_timer as timer

from scalar_distributions import QaryMemorylessDistribution


def make_xVectorDistribution_fromQaryMemorylessDistribution(q, xyDistribution, length, use_log=False):
    def make_xVectorDistribution():
        xDistribution = QaryMemorylessDistribution.QaryMemorylessDistribution(q)
        xDistribution.probs = [xyDistribution.calcXMarginals()]
        xVectorDistribution = xDistribution.makeQaryMemorylessVectorDistribution(length, None, use_log=use_log)
        return xVectorDistribution

    return make_xVectorDistribution

def make_xyVectorDistribution_fromQaryMemorylessDistribution(xyDistribution, use_log=False):
    def make_xyVectorDistribution(receivedWord):
        length = len(receivedWord)
        useTrellis = False

        if useTrellis:
            xyVectorDistribution = xyDistribution.makeQaryTrellisDistribution(length, receivedWord)
        else:
            xyVectorDistribution = xyDistribution.makeQaryMemorylessVectorDistribution(length, receivedWord, use_log)

        return xyVectorDistribution

    return make_xyVectorDistribution

def test(q, listDecode=False, maxListSize=None, checkSize=None, numInfoIndices=None, verbosity=False):
    print("q = " + str(q))

    p = 0.99
    L = 100
    n = 8
    N = 2 ** n

    upperBoundOnErrorProbability = 0.1

    xDistribution = None
    xyDistribution = QaryMemorylessDistribution.makeQSC(q, p)

    frozenSet = getFrozenSet(q, N, n, L, "QSC", xDistribution, xyDistribution, p, upperBoundOnErrorProbability, numInfoIndices, verbosity=verbosity)

    # print("Rate = ", N - len(frozenSet), "/", N, " = ", (N - len(frozenSet)) / N)

    numberOfTrials = 200

    make_xVectorDistribution = make_xVectorDistribution_fromQaryMemorylessDistribution(q, xyDistribution, N)
    simulateChannel = simulateChannel_fromQaryMemorylessDistribution(xyDistribution)
    make_xyVectorDistribution = make_xyVectorDistribution_fromQaryMemorylessDistribution(xyDistribution)

    if not listDecode:
        QaryPolarEncoderDecoder.encodeDecodeSimulation(q, N, make_xVectorDistribution, simulateChannel,
                                                       make_xyVectorDistribution, numberOfTrials, frozenSet,
                                                       verbosity=verbosity)
    else:
        QaryPolarEncoderDecoder.encodeListDecodeSimulation(q, N, make_xVectorDistribution,
                                                           simulateChannel,
                                                           make_xyVectorDistribution, numberOfTrials, frozenSet,
                                                           maxListSize, checkSize, verbosity=verbosity)

    # # trustXYProbs = False
    # trustXYProbs = True
    # PolarEncoderDecoder.genieEncodeDecodeSimulation(N, make_xVectorDistribuiton, make_codeword, simulateChannel, make_xyVectorDistribution, numberOfTrials, upperBoundOnErrorProbability, trustXYProbs)


def test_ir(q, channel_type="QSC", maxListSize=None, checkSize=0, numberOfTrials=200, ir_version=1, numInfoIndices=None, use_log=False, verbosity=False):
    p = 0.98
    L = 100
    n = 6
    N = 2 ** n
    upperBoundOnErrorProbability = 0.1

    xDistribution = None
    xyDistribution = QaryMemorylessDistribution.makeQSC(q, p)

    if channel_type != "QSC":
        raise "TODO: Implement this for AWGN channel"
    frozenSet = getFrozenSet(q, N, n, L, channel_type, xDistribution, xyDistribution, p, upperBoundOnErrorProbability, numInfoIndices, verbosity=verbosity)

    simulateChannel = simulateChannel_fromQaryMemorylessDistribution(xyDistribution)
    make_xyVectorDistribution = make_xyVectorDistribution_fromQaryMemorylessDistribution(xyDistribution, use_log)

    if maxListSize is None:
        mixingFactor = max(frozenSet)+1 - len(frozenSet)
        maxListSize = mixingFactor ** q
        print(maxListSize)
    QaryPolarEncoderDecoder.irSimulation(q, N, simulateChannel,
                                          make_xyVectorDistribution, numberOfTrials, frozenSet, maxListSize, checkSize, use_log=use_log,
                                          verbosity=verbosity, ir_version=ir_version)

def test_ir_per_config(q, L, n, maxListSize, numTrials, channel_type="QSC", qer=None, snr=None, rate=None, numInfoIndices=None, frozenSet=None, use_log=False, verbosity=False, file_name=None):
    N = 2 ** n
    assert(channel_type in ["QSC", "AWGN"])

    if numInfoIndices is None:
        assert(rate is not None)
        numInfoIndices = math.floor(rate * N)

    xDistribution = None
    if channel_type == "QSC":
        xyDistribution = QaryMemorylessDistribution.makeQSC(q, qer)
    elif channel_type == "AWGN":
        xyDistribution = QaryMemorylessDistribution.makeAWGN(q, snr, rate)

    if frozenSet is None:
        frozenSet = getFrozenSet(q, N, n, L, channel_type, xDistribution, xyDistribution, qer=qer, snr=snr, rate=rate, numInfoIndices=numInfoIndices, verbosity=verbosity)

    if maxListSize is None:
        mixingFactor = max(frozenSet) + 1 - len(frozenSet)
        maxListSize = mixingFactor ** q
        print(maxListSize)

    if verbosity:
        print("q=" + str(q)
              + ", channelType=" + str(channel_type)
              + ", qer=" + str(qer)
              + ", snr=" + str(snr)
              + ", rate=" + str(rate)
              + ", n=" + str(n)
              + ", L=" + str(L)
              + ", numInfoQudits=" + str(numInfoIndices)
              + ", maxListSize=" + str(maxListSize)
              + ", numTrials=" + str(numTrials))

    if channel_type == "AWGN":
        raise "TODO: Implement this for AWGN channel"

    make_xVectorDistribution = make_xVectorDistribution_fromQaryMemorylessDistribution(q, xyDistribution, N, use_log)
    simulateChannel = simulateChannel_fromQaryMemorylessDistribution(xyDistribution)
    make_xyVectorDistribution = make_xyVectorDistribution_fromQaryMemorylessDistribution(xyDistribution, use_log)

    start = timer()
    frame_error_prob, symbol_error_prob, key_rate, prob_result_list = QaryPolarEncoderDecoder.irSimulation(q, N, simulateChannel,
                                         make_xyVectorDistribution, numTrials, frozenSet, maxListSize, use_log=use_log, verbosity=verbosity)
    end = timer()
    time_rate = (end - start) / (numTrials * N)
    if file_name is not None:
        write_header(file_name)
        write_result(file_name, q, qer, snr, calc_theoretic_key_rate(q, channel_type, qer=qer, snr=snr, rate=rate), n, N, L, "degradingUpgrading",
                     numInfoIndices, rate, maxListSize, frame_error_prob, symbol_error_prob, key_rate, time_rate,
                     numTrials, prob_result_list,
                     verbosity=verbosity)
    return frame_error_prob, symbol_error_prob, key_rate, time_rate, maxListSize, prob_result_list

def test_ir_and_record_range(file_name, q_range, L, n_range, numTrials, channel_type="QSC", qer_range=None, snr_range=None, maxListRange=None, rate_range=None, numInfoQuditsDelta=0, numInfoQuditsRadius=10, maxListSizeCompleteRange=False, explicitNumInfoQuditsRange=None, largeMaxListSize=False, use_log=False, verbosity=False):
    write_header(file_name)
    assert(channel_type in ["QSC", "AWGN"])
    if channel_type != "QSC":
        raise "TODO: Implement this for AWGN channel"
    for q in q_range:
        if channel_type == "QSC":
            snr_range = [None] * len(qer_range)
        elif channel_type == "AWGN":
            qer_range = [None] * len(snr_range)
        for qer, snr in zip(qer_range, snr_range):
            for n in n_range:
                N = 2**n
                if rate_range is not None:
                    numInfoQuditsRange = [math.floor(rate * N) for rate in rate_range]
                else:
                    assert (explicitNumInfoQuditsRange is not None)
                    numInfoQuditsRange = explicitNumInfoQuditsRange
                    rate_range = [None] * len(numInfoQuditsRange)
                for numInfoQudits, rate in zip(numInfoQuditsRange, rate_range):
                    theoretic_key_rate = calc_theoretic_key_rate(q, channel_type, qer=qer, snr=snr, rate=rate)
                    # theoretic_num_info_qudits = math.ceil(N*theoretic_key_rate*math.log(2, q))
                    if largeMaxListSize:
                        frame_error_prob, symbol_error_prob, key_rate, time_rate, maxListSize, prob_result_list = test_ir_per_config(q=q, channel_type=channel_type, qer=qer, snr=snr, rate=rate, L=L, n=n, maxListSize=None, numInfoIndices=numInfoQudits,
                                                                             numTrials=numTrials, use_log=use_log, verbosity=verbosity)
                        write_result(file_name, q, qer, snr, theoretic_key_rate, n, N, L, "degradingUpgrading",
                                     numInfoQudits, rate, maxListSize, frame_error_prob, symbol_error_prob, key_rate, time_rate, numTrials, prob_result_list,
                                     verbosity=verbosity)
                        continue

                    if maxListRange is None:
                        maxListRange = list(range(1, 21))
                    cur_frame_error_prob = None
                    prev_frame_error_prob = None
                    for maxListSize in maxListRange:
                        if not maxListSizeCompleteRange and prev_frame_error_prob is not None and (cur_frame_error_prob >= prev_frame_error_prob + 0.1):
                            break
                        if math.log2(q)*numInfoQudits < math.log2(maxListSize):
                            break
                        frame_error_prob, symbol_error_prob, key_rate, time_rate, maxListSize, prob_result_list = test_ir_per_config(q=q, channel_type=channel_type, qer=qer, snr=snr, rate=rate, L=L, n=n, maxListSize=maxListSize, numInfoIndices=numInfoQudits, numTrials=numTrials, use_log=use_log, verbosity=verbosity)
                        write_result(file_name, q, qer, snr, theoretic_key_rate, n, N, L, "degradingUpgrading", numInfoQudits, rate, maxListSize, frame_error_prob, symbol_error_prob, key_rate, time_rate, numTrials, prob_result_list, verbosity=verbosity)
                        prev_frame_error_prob = cur_frame_error_prob
                        cur_frame_error_prob = frame_error_prob

# test(2)
# test(3)
# test(3, listDecode=True, maxListSize=1, checkSize=1)
# test(3, listDecode=True, maxListSize=3, checkSize=4)
# test_ir(2)
# test_ir(2, 100, 20)
# test_ir(3, 10, 1, numberOfTrials=1, numInfoIndices=5, verbosity=True)
# test_ir(3)
# test_ir(3, ir_version=2)

maxListRange = [1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049]
n_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
use_log = False

test_ir_and_record_range(file_name="../results/old/results_nospc.csv", q_range=[3], qer_range=[0.01], L=100, n_range=n_range, numTrials=100, rate_range=[0.94, 0.95], maxListRange=maxListRange, maxListSizeCompleteRange=True, use_log=use_log, verbosity=True)
# test_ir_and_record_range(file_name="results_nospc.csv", q_range=[3], qer_range=[0.02], L=100, n_range=n_range, numTrials=100, rate_range=[0.9, 0.92], maxListRange=maxListRange, maxListSizeCompleteRange=True, use_log=use_log, verbosity=True)
# test_ir_and_record_range(file_name="results_nospc.csv", q_range=[3], qer_range=[0.05], L=100, n_range=n_range, numTrials=100, rate_range=[0.79, 0.81], maxListRange=maxListRange, maxListSizeCompleteRange=True, use_log=use_log, verbosity=True)

# test_ir_and_record_range(file_name="results_fastsscl.csv", q_range=[3], qer_range=[0.95], L=100, n_range=[11, 12, 13, 14], numTrials=100, rate_range=[0.24, 0.26], maxListRange=maxListRange, maxListSizeCompleteRange=True, use_log=use_log, verbosity=True)
# test_ir_and_record_range(file_name="results_fastsscl.csv", q_range=[3], qer_range=[0.98], L=100, n_range=[11, 12, 13, 14], numTrials=100, rate_range=[0.27, 0.28], maxListRange=maxListRange, maxListSizeCompleteRange=True, use_log=use_log, verbosity=True)
# test_ir_and_record_range(file_name="results_fastsscl.csv", q_range=[3], qer_range=[0.99], L=100, n_range=[11, 12, 13, 14], numTrials=100, rate_range=[0.31, 0.32], maxListRange=maxListRange, maxListSizeCompleteRange=True, use_log=use_log, verbosity=True)
# test_ir_and_record_range(file_name="results_fastsscl.csv", q_range=[3], qer_range=[1.0], L=100, n_range=[11, 12, 13, 14], numTrials=100, rate_range=[0.36, 0.37], maxListRange=maxListRange, maxListSizeCompleteRange=True, use_log=use_log, verbosity=True)

# calcFrozenSetsOnly(q_range=[3], L=100, n_range=[12], qer_range=[1.0], verbosity=True)

# import plot
# q_list = [3]
# qer_list = [1.0]
# rate_list = None  # [0.94, 0.95]
# max_list_size_list = None # [2, 4, 8, 16, 32, 64, 128]  # [729, 2187]
# n_list = [6, 7, 8, 9, 10, 11, 12, 13, 14]
# plot.plot_results3("results.csv", q_filter=q_list, qer_filter=qer_list, n_filter=n_list, rate_filter=rate_list, max_list_size_filter=max_list_size_list, errorType="frame", adjustedKeyRate=True)


# plot.plot_results3("results.csv", q_filter=q_list, qer_filter=qer_list, n_filter=n_list, rate_filter=rate_list, max_list_size_filter=max_list_size_list, errorType="frame", adjustedKeyRate=True)
# plot.plot_results1("results.csv", q_filter=q_list, qer_filter=qer_list, n_filter=n_list, errorType="ml_lower_bound", adjustedKeyRate=False) # log probs
# plot.plot_results2("results9.csv", q_filter=q_list, qer_filter=qer_list, n_filter=n_list)
# diff_frozen_range([2, 3], [4, 5, 6, 7, 8, 9, 10], 100, 150, qer_range=[0.98, 0.99, 1.0])
# diff_frozen_range([3], [6], 100, 200, qer_range=[0.98])
# test_ir_per_config(q=3, qer=0.01, L=100, n=8, maxListSize=10, numInfoIndices=230, numTrials=100, use_log=True, verbosity=True)
#
# test_ir_per_config(q=2, snr=1.0, L=100, n=11, maxListSize=1, rate=0.5, numTrials=100, use_log=True, verbosity=True)
# calcFrozenSetsOnly(q_range=[3], L=100, n_range=[13, 14], qer_range=[1.0], verbosity=True)

# snr_list = list(np.arange(1.0, 2.25, 0.25))
# snr_list = list(np.arange(2.25, 3.25, 0.25))
# for snr in snr_list:
#     # print("snr=" + str(snr))
#     # qer = snr_to_qer(2, snr, 0.5)
#     # print("qer=" + str(qer))
#     # calcFrozenSetsOnly(q_range=[2], L=100, n_range=[11], snr_range=[snr], rate_range=[0.5], verbosity=True)

# for maxListSize in [1, 2, 4, 8, 16, 32, 64]:
#     for numInfoIndices in [90_000, 90_100, 90_200, 90_300]:
#         test_ir_per_config(q=2, qer=0.02, L=20, n=20, maxListSize=maxListSize, numInfoIndices=numInfoIndices, numTrials=100, use_log=True, verbosity=True, file_name="results11.csv")
# check_SPC_property(q=3, L=100, n=8, channel_type="QSC", qer=1.0, rate=0.36, verbosity=True)
# test_ir_per_config(q=3, qer=1.0, L=100, n=10, maxListSize=30, numInfoIndices=600, numTrials=100, use_log=False, verbosity=True)