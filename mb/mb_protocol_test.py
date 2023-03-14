import unittest

from key_generator import KeyGenerator
from mb_cfg import MbCfg, IndicesToEncodeStrategy
from mb_protocol import MbProtocol


class MbProtocolTest(unittest.TestCase):
    def test_run(self):
        protocol = MbProtocol(MbCfg(q=3, block_length=3, num_blocks=4, max_candidates_num=2),
                              [0, 1, 2, 0, 0, 2, 2, 0, 1, 1, 1, 1],
                              [2, 0, 1, 1, 1, 0, 1, 2, 2, 2, 0, 2])
        self.assertEqual(protocol.run()[0].is_success, True)
        protocol = MbProtocol(MbCfg(q=3, block_length=3, num_blocks=4), [0, 1, 2, 0, 0, 2, 2, 0, 1, 1, 1, 1],
                              [2, 0, 1, 1, 1, 0, 1, 2, 2, 2, 0, 2])
        self.assertEqual(protocol.run()[0].is_success, True)
        protocol = MbProtocol(MbCfg(q=3, block_length=4, num_blocks=3), [0, 1, 2, 0, 0, 2, 2, 0, 1, 1, 1, 1],
                              [2, 0, 1, 1, 1, 0, 1, 2, 2, 2, 0, 2])
        self.assertEqual(protocol.run()[0].is_success, True)
        protocol = MbProtocol(MbCfg(q=3, block_length=5, num_blocks=3), [0, 1, 2, 0, 0, 2, 2, 0, 1, 1, 1, 1, 0, 1, 2],
                              [2, 0, 1, 1, 1, 0, 1, 2, 2, 2, 0, 2, 2, 0, 1])
        self.assertEqual(protocol.run()[0].is_success, True)

    def test_run_large_no_error(self):
        p_err = 0.0
        cfg = MbCfg(q=3, block_length=6, num_blocks=100, p_err=p_err,
                    success_rate=1.0, max_candidates_num=10,
                    indices_to_encode_strategy=IndicesToEncodeStrategy.most_candidate_blocks,
                    max_num_indices_to_encode=3)
        key_generator = KeyGenerator(p_err=p_err, key_length=600)
        a, b = key_generator.generate_keys()
        protocol = MbProtocol(cfg, a, b)
        self.assertTrue(protocol.run()[0].is_success)

    def test_run_large_error(self):
        p_err = 0.01
        cfg = MbCfg(q=3, block_length=6, num_blocks=100, p_err=p_err,
                    success_rate=0.99, max_candidates_num=10,
                    indices_to_encode_strategy=IndicesToEncodeStrategy.most_candidate_blocks,
                    max_num_indices_to_encode=10)
        key_generator = KeyGenerator(p_err=p_err, key_length=600)
        a, b = key_generator.generate_keys()
        protocol = MbProtocol(cfg, a, b)
        self.assertTrue(protocol.run()[0].is_success)


if __name__ == '__main__':
    unittest.main()
