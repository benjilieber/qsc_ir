import math
import os
import time
from timeit import default_timer as timer

import numpy as np

from polar.polar_encoder_decoder import PolarEncoderDecoder
from polar.scalar_distributions.qary_memoryless_distribution import QaryMemorylessDistribution
from protocol import Protocol


class PolarProtocol(Protocol):
    def __init__(self, cfg, a, b):
        super().__init__(cfg=cfg, a=a, b=b)

        np.random.seed([os.getpid(), int(str(time.time() % 1)[2:10])])

        self.total_leak = 0
        self.total_communication_size = 0

        self.cur_candidates_num = 1

    def run(self):
        start = timer()
        cpu_start = time.process_time()

        encoder_decoder = PolarEncoderDecoder(self.cfg)
        w, u = encoder_decoder.calculate_syndrome_and_complement(self.a)
        a_key = encoder_decoder.get_message_info_bits(u)
        x_b = self.b
        # x_b = np.mod(np.add(b, polarTransformOfQudits(self.cfg.q, w)), self.cfg.q)
        xy_vec_dist = self.xy_vec_dist_given_input_vec(x_b)
        frozen_vec = (encoder_decoder.get_message_frozen_bits(w) * (self.cfg.q - 1)) % self.cfg.q

        b_key_list, a_guess_list = encoder_decoder.list_decode(xy_vec_dist=xy_vec_dist,
                                                               frozen_values=frozen_vec)

        end = timer()
        cpu_end = time.process_time()

        return self.get_results(key_length=self.cfg.num_info_indices,
                                a_guess_list=a_guess_list,
                                a_key=a_key,
                                b_key_list=b_key_list,
                                leak_size=self.cfg.num_frozen_indices * math.log2(self.cfg.q),
                                matrix_size=0.0,
                                bob_communication_size=1.0,
                                time=end - start,
                                cpu_time=cpu_end - cpu_start)

    def xy_vec_dist_given_input_vec(self, received_vec):
        return self.cfg.xy_dist.makeQaryMemorylessVectorDistribution(self.cfg.N, received_vec, self.cfg.use_log)

    def x_vec_dist(self):
        x_dist = QaryMemorylessDistribution(self.cfg.q)
        x_dist.probs = [self.cfg.xy_dist.calcXMarginals()]
        x_vec_dist = x_dist.makeQaryMemorylessVectorDistribution(self.cfg.N, None, use_log=self.cfg.use_log)
        return x_vec_dist
