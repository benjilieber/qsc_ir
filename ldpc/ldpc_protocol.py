import math
import os
import time

import numpy as np

from ldpc.ldpc_cfg import Decoder
from ldpc.ldpc_decoder import LdpcDecoder
from ldpc.ldpc_generator import LdpcGenerator
from protocol import Protocol


class LdpcProtocol(Protocol):
    def __init__(self, cfg, a, b):
        super().__init__(cfg=cfg, a=a, b=b)

        np.random.seed([os.getpid(), int(str(time.time() % 1)[2:10])])

        self.total_leak = 0
        self.total_communication_size = 0

        self.cur_candidates_num = 1

    def run(self):
        assert (self.cfg.list_size == 1)

        start = time.time()
        cpu_start = time.process_time()
        code_gen = LdpcGenerator(self.cfg)
        encoding_matrix = code_gen.generate_gallagher_matrix(self.cfg.syndrome_length)
        encoded_a = encoding_matrix * self.a
        decoder = LdpcDecoder(self.cfg, encoding_matrix, encoded_a)
        if self.cfg.decoder == Decoder.bp:
            candidates_left, stats = decoder.decode_belief_propagation(self.b, 20)
        else:
            raise "No support for iterative ldpc decoder"
        end = time.time()
        cpu_end = time.process_time()
        leak_size = self.cfg.syndrome_length * math.log2(self.cfg.q)
        matrix_size = self.cfg.syndrome_length * self.cfg.sparsity * math.log2(self.cfg.N) * math.log2(self.cfg.q - 1)
        bob_communication_size = 1.0
        a_guess_list = stats.a_guess_list if (len(candidates_left) > 0) else []
        return self.get_results(key_length=self.cfg.N,
                                a_guess_list=a_guess_list,
                                a_key=self.a,
                                b_key_list=a_guess_list,
                                leak_size=leak_size,
                                matrix_size=matrix_size,
                                bob_communication_size=bob_communication_size,
                                time=end - start,
                                cpu_time=cpu_end - cpu_start)
