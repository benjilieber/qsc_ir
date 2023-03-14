# q, qer, snr, calc_theoretic_key_rate(q, channel_type, qer=qer, snr=snr, rate=rate),
#                          n, N, L, "degradingUpgrading",
#                          numInfoIndices, rate, maxListSize, frame_error_prob, symbol_error_prob, key_rate, time_rate,
#                          numTrials, prob_result_list

def write_header(file_name):
    prob_result_names = [probResult.name for probResult in QaryPolarEncoderDecoder.ProbResult]
    header = ["q", "qer", "snr", "theoreticKeyRate", "n", "N", "L", "frozenBitsAlgorithm", "numInfoQudits", "rate",
              "maxListSize", "frameErrorProb", "symbolErrorProb", "keyRate", "yield", "efficiency", "timeRate",
              "d"] + prob_result_names
    try:
        with open(file_name, 'r') as f:
            for row in f:
                assert (row.rstrip('\n').split(",") == header)
                return
    except FileNotFoundError:
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    except AssertionError:
        raise AssertionError(f"Header of {file_name} is bad.")


def write_result(file_name, q, qer, snr, theoretic_key_rate, n, N, L, frozenBitsAlgorithm, numInfoQudits, rate,
                 maxListSize, frame_error_prob, symbol_error_prob, key_rate, time_rate, numTrials, prob_result_list,
                 verbosity=False):
    if verbosity:
        print("writing results")
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        if q == 2:
            yld = (1 - frame_error_prob) * numInfoQudits * math.log(q, 2)
            efficiency = numInfoQudits * math.log(q, 2) / (-qer * math.log(qer, 2) - (1 - qer) * math.log(1 - qer, 2))
        else:
            yld = None
            efficiency = None
        probResultCounter = Counter(prob_result_list)
        if verbosity:
            print(probResultCounter)
        prob_result_stats = [probResultCounter[probResult] / numTrials for probResult in
                             QaryPolarEncoderDecoder.ProbResult]
        writer.writerow(
            [q, qer, snr, theoretic_key_rate, n, N, L, frozenBitsAlgorithm, numInfoQudits, rate, maxListSize,
             frame_error_prob, symbol_error_prob, key_rate, yld, efficiency, time_rate, numTrials] + prob_result_stats)
