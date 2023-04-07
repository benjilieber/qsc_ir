import math
import os
import platform

import numpy as np

from cfg import CodeStrategy, Cfg
from polar.scalar_distributions import qary_memoryless_distribution


class IndexType:
    frozen = False
    info = True


class PolarCfg(Cfg):

    def __init__(self,
                 orig_cfg=None,
                 q=None,
                 p_err=None,
                 N=None,
                 key_rate=None,
                 num_info_indices=None,
                 success_rate=None,
                 relative_gap_rate=None,
                 constr_l=None,
                 scl_l=None,
                 check_length=0,
                 use_log=None,
                 raw_results_file_path=None,
                 agg_results_file_path=None,
                 verbosity=False):
        super().__init__(orig_cfg=orig_cfg,
                         q=q,
                         N=N,
                         p_err=p_err,
                         use_log=use_log,
                         check_length=check_length,
                         code_strategy=CodeStrategy.polar,
                         raw_results_file_path=raw_results_file_path,
                         agg_results_file_path=agg_results_file_path,
                         verbosity=verbosity)

        self.n = int(math.log2(self.N))

        assert ((key_rate is not None) + (relative_gap_rate is not None) + (success_rate is not None) + (
                    num_info_indices is not None) == 1)
        self.desired_key_rate = key_rate
        self.desired_relative_gap_rate = relative_gap_rate
        self.desired_success_rate = success_rate
        self.desired_num_info_indices = num_info_indices
        if num_info_indices is not None:
            self.num_info_indices = num_info_indices
        elif key_rate is not None:
            self.num_info_indices = int(math.ceil(key_rate * math.log(2, self.q) * self.N)) + self.check_length
        elif relative_gap_rate is not None:
            self.num_info_indices = int(
                math.ceil(self._theoretic_key_q_rate() * relative_gap_rate * self.N)) + self.check_length
        else:
            self.num_info_indices = None  # Will be set during the polar code construction below
        self.constr_l = constr_l
        self.x_dist = None
        self.xy_dist = qary_memoryless_distribution.makeQSC(self.q, self.qer)
        self.index_types, self.frozen_set, self.info_set = self._calc_index_types()
        self.num_frozen_indices = self.N - self.num_info_indices

        if scl_l is not None:
            self.list_size = scl_l
        else:
            mixing_factor = max(self.frozen_set) + 1 - len(self.frozen_set)
            self.list_size = mixing_factor ** self.q

    def log_dict(self):
        super_dict = super().log_dict()
        specific_dict = {"polar_n": self.n,
                         "polar_desired_key_rate": self.desired_key_rate,
                         "polar_desired_num_info_indices": self.desired_num_info_indices,
                         "polar_desired_relative_gap_rate": self.desired_relative_gap_rate,
                         "polar_desired_success_rate": self.desired_success_rate,
                         "polar_num_info_indices": self.num_info_indices,
                         "polar_num_frozen_indices": self.num_frozen_indices,
                         "polar_constr_l": self.constr_l}
        assert (set(specific_dict.keys()) == set(specific_log_header()))
        return {**super_dict, **specific_dict}

    def _calc_index_types(self):
        """
        Constructs the code, i.e. defines which bits are informational and which are frozen.
        Two behaviours are possible:
        1) If there is previously saved data with the sorted indices of this config's channel
        and construction method, it loads this data and uses it to define sets of informational and frozen bits.
        2) Otherwise, it calls the preferred method from the dict of methods to construct the code. Then, it
        saves sorted indices of channels and finally defines sets of informational and frozen bits.
        """
        assert (self.n >= 0)
        assert (self.constr_l > 0)
        assert (self.xy_dist is not None)
        tv, pe = self._calc_tv_pe_degrading_upgrading()
        return self._calc_index_types_from_tv_pe(tv, pe)

    def _calc_tv_pe_degrading_upgrading(self):
        directory_name = self.get_construction_path()
        tv_construction_name = directory_name + '{}.npy'.format("DegradingUpgrading_L=" + str(self.constr_l) + "_tv")
        pe_construction_name = directory_name + '{}.npy'.format("DegradingUpgrading_L=" + str(self.constr_l) + "_pe")
        if self.verbosity:
            print(tv_construction_name)
            print(pe_construction_name)

        # If the files with data exist, load them
        if os.path.isfile(tv_construction_name) and os.path.isfile(pe_construction_name):
            tv = np.load(tv_construction_name, allow_pickle=True)
            pe = np.load(pe_construction_name, allow_pickle=True)
            return tv, pe

        # Otherwise, obtain construction and save them to the files
        else:
            if self.verbosity:
                print("Calculating TV and Pe vectors...")

            # Upgrade
            if self.x_dist is not None:
                x_dists = [[self.x_dist]]
                for m in range(1, self.n + 1):
                    x_dists.append([])
                    for dist in x_dists[m - 1]:
                        x_dists[m].append(dist.minus_transform().upgrade(self.constr_l))
                        x_dists[m].append(dist.plus_transform().upgrade(self.constr_l))

            # Degrade
            xy_dists = [[self.xy_dist]]
            for m in range(1, self.n + 1):
                xy_dists.append([])
                for dist in xy_dists[m - 1]:
                    xy_dists[m].append(dist.minus_transform().degrade(self.constr_l))
                    xy_dists[m].append(dist.plus_transform().degrade(self.constr_l))

            tv = []
            pe = []
            for i in range(self.N):
                if self.x_dist is not None:
                    tv.append(x_dists[self.n][i].totalVariation())
                else:
                    tv.append(0.0)
                pe.append(xy_dists[self.n][i].errorProb())

            if self.verbosity:
                print("Done calculating!")
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)
            np.save(tv_construction_name, tv)
            np.save(pe_construction_name, pe)

            return tv, pe

    def get_construction_path(self):
        construction_path = os.path.dirname(os.path.abspath(__file__))

        if platform.system() == 'Windows':
            sep = '\\'
        else:
            sep = '/'
        construction_path += sep + 'polar_codes_constructions' + sep
        construction_path += 'q={}'.format(self.q) + sep
        construction_path += 'N={}'.format(self.N) + sep
        construction_path += 'QER={}'.format(self.qer) + sep
        return construction_path

    def _calc_index_types_from_tv_pe(self, tv, pe):
        if self.verbosity:
            print("num_info_indices = ", self.num_info_indices)
        tv_plus_pe = np.add(tv, pe)
        sorted_indices = sorted(range(self.N), key=lambda k: tv_plus_pe[k])

        if self.num_info_indices is None:
            error_sum = 0.0
            self.num_info_indices = 0
            while error_sum < 1 - self.desired_success_rate and self.num_info_indices < self.N:
                i = sorted_indices[self.num_info_indices]
                if tv_plus_pe[i] + error_sum <= 1 - self.desired_success_rate:
                    error_sum += tv_plus_pe[i]
                    self.num_info_indices += 1
                else:
                    break

        self.success_rate = 1 - sum(tv_plus_pe[sorted_indices[:self.num_info_indices]])

        index_types = np.zeros(self.N, dtype=bool)
        index_types[sorted_indices[self.num_info_indices:]] = IndexType.frozen
        index_types[sorted_indices[:self.num_info_indices]] = IndexType.info
        segment_size = self.N
        while segment_size > 1:
            for i in range(self.N // segment_size):
                if sum(index_types[i * segment_size: (i + 1) * segment_size] == IndexType.frozen) == 1 and index_types[
                    i * segment_size] == IndexType.info:
                    index_types[i * segment_size] = IndexType.frozen
                    index_types[i * segment_size + 1: (i + 1) * segment_size] = IndexType.info
            segment_size //= 2
        frozen_set = set([i for i, index_type in enumerate(index_types) if index_type == IndexType.frozen])
        info_set = set([i for i, index_type in enumerate(index_types) if index_type == IndexType.info])

        if self.verbosity:
            print("frozen set = ", frozen_set)
            if self.num_info_indices is None:
                print("fraction of info indices = ", 1.0 - len(frozen_set) / self.N)

        return index_types, frozen_set, info_set

    def get_num_info_indices_to_success_prob_map(self):
        tv, pe = self._calc_tv_pe_degrading_upgrading()
        tv_plus_pe = np.add(tv, pe)
        sorted_indices = sorted(range(self.N), key=lambda k: tv_plus_pe[k])

        error_sum = 0.0
        i_ord = 1
        success_rate_list = [0.0 for _ in range(self.N+1)]
        while i_ord < self.N:
            i = sorted_indices[i_ord-1]
            error_sum += tv_plus_pe[i]
            i_ord += 1
            success_rate_list[i_ord] = 1.0-error_sum
        return list(range(self.N+1)), success_rate_list


def specific_log_header():
    return ["polar_n",
            "polar_desired_key_rate",
            "polar_desired_num_info_indices",
            "polar_desired_relative_gap_rate",
            "polar_desired_success_rate",
            "polar_num_info_indices",
            "polar_num_frozen_indices",
            "polar_constr_l"]


def specific_log_header_params():
    return ["polar_num_info_indices",
            "polar_constr_l"]
