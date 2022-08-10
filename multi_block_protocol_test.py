import unittest

from key_generator import KeyGenerator
from multi_block_protocol import MultiBlockProtocol
from protocol_configs import ProtocolConfigs, IndicesToEncodeStrategy, CodeGenerationStrategy


class MultiBlockProtocolTest(unittest.TestCase):
    def test_run(self):
        protocol = MultiBlockProtocol(ProtocolConfigs(base=3, block_length=3, num_blocks=4, max_candidates_num=2),
                                  [0, 1, 2, 0, 0, 2, 2, 0, 1, 1, 1, 1],
                                  [2, 0, 1, 1, 1, 0, 1, 2, 2, 2, 0, 2])
        self.assertEqual(protocol.run().is_success, True)
        protocol = MultiBlockProtocol(ProtocolConfigs(base=3, block_length=3, num_blocks=4), [0, 1, 2, 0, 0, 2, 2, 0, 1, 1, 1, 1],
                                  [2, 0, 1, 1, 1, 0, 1, 2, 2, 2, 0, 2])
        self.assertEqual(protocol.run().is_success, True)
        protocol = MultiBlockProtocol(ProtocolConfigs(base=3, block_length=4, num_blocks=3), [0, 1, 2, 0, 0, 2, 2, 0, 1, 1, 1, 1],
                                  [2, 0, 1, 1, 1, 0, 1, 2, 2, 2, 0, 2])
        self.assertEqual(protocol.run().is_success, True)
        protocol = MultiBlockProtocol(ProtocolConfigs(base=3, block_length=5, num_blocks=3), [0, 1, 2, 0, 0, 2, 2, 0, 1, 1, 1, 1, 0, 1, 2],
                                  [2, 0, 1, 1, 1, 0, 1, 2, 2, 2, 0, 2, 2, 0, 1])
        self.assertEqual(protocol.run().is_success, True)

    def test_run_large_no_error(self):
        p_err = 0.0
        cfg = ProtocolConfigs(base=3, block_length=6, num_blocks=100, p_err=p_err,
                              success_rate=1.0, max_candidates_num=10,
                              indices_to_encode_strategy=IndicesToEncodeStrategy.MOST_CANDIDATE_BLOCKS,
                              code_generation_strategy=CodeGenerationStrategy.LINEAR_CODE,
                              max_num_indices_to_encode=3)
        key_generator = KeyGenerator(p_err=p_err, key_length=600)
        a, b = key_generator.generate_keys()
        protocol = MultiBlockProtocol(cfg, a, b)
        self.assertTrue(protocol.run().is_success)

    def test_run_large_error(self):
        p_err = 0.01
        cfg = ProtocolConfigs(base=3, block_length=6, num_blocks=100, p_err=p_err,
                              success_rate=0.99, max_candidates_num=10,
                              indices_to_encode_strategy=IndicesToEncodeStrategy.MOST_CANDIDATE_BLOCKS,
                              code_generation_strategy=CodeGenerationStrategy.LINEAR_CODE,
                              max_num_indices_to_encode=10)
        key_generator = KeyGenerator(p_err=p_err, key_length=600)
        a, b = key_generator.generate_keys()
        protocol = MultiBlockProtocol(cfg, a, b)
        self.assertTrue(protocol.run().is_success)

if __name__ == '__main__':
    unittest.main()
