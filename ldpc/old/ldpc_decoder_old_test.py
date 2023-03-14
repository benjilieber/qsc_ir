import math
import os
import sys

import util

sys.path.append(os.getcwd())

import unittest
from ldpc_decoder import LdpcDecoder
from ldpc_generator import LdpcGenerator
from protocol_configs import CodeGenerationStrategy
from key_generator import KeyGenerator


class LdpcDecoderTest(unittest.TestCase):
    # def test_decode_single_block1(self): # successful
    #     encoding_matrix = [[0, 1, 0, 1, 1, 0, 0, 2],
    #                        [2, 1, 1, 0, 0, 1, 0, 0],
    #                        [0, 0, 1, 0, 0, 2, 1, 1],
    #                        [1, 0, 0, 2, 1, 0, 1, 0]]
    #     encoded_a = [2, 1, 1, 2, 0, 2, 0, 0]
    #     b = [0, 2, 2, 0]
    #     decoder = LdpcDecoder(3, 0.01, encoding_matrix, encoded_a)
    #     a_guess = decoder.decode_belief_propagation(b, 20, 0)
    #     print(a_guess)
    #     print(encoded_a)
    #     print(np.matmul(a_guess, encoding_matrix) % 3)

    # def test_decode_single_block2(self):
    #     block_length = 8
    #     encoding_matrix = [[1, 2, 1, 0],
    #                        [1, 0, 1, 2],
    #                        [0, 2, 2, 1],
    #                        [1, 1, 0, 1],
    #                        [2, 0, 1, 2],
    #                        [1, 1, 2, 0],
    #                        [0, 2, 1, 1],
    #                        [1, 2, 0, 1]]
    #     a = [1, 0, 1, 1, 0, 2, 2, 0]
    #     encoded_a = [1, 2, 0, 1]
    #     b = [0, 2, 2, 0, 1, 1, 0, 1]
    #     decoder = LdpcDecoder(3, 0.01, encoding_matrix, encoded_a, use_log=False)
    #     a_guess = decoder.decode_belief_propagation(b, 500, num_forked_indices=2)
    #
    #     deltas = product([1, 2], repeat=block_length)
    #     for cur_delta in deltas:
    #         cur_candidate = np.mod(np.add(b, cur_delta), [3] * block_length).astype(int)
    #         if np.array_equal(np.matmul(cur_candidate, encoding_matrix) % 3, encoded_a):
    #             print("valid candidate:")
    #             print(cur_candidate)
    #
    #     print(a_guess)
    #     print(np.matmul(a, encoding_matrix) % 3)
    #     print(np.matmul(a_guess, encoding_matrix) % 3)

    # def test_decode_single_block(self):
    #     block_length = 8
    #     encoding_matrix = [[0, 1, 0, 1],
    #                        [1, 1, 0, 0],
    #                        [0, 1, 1, 0],
    #                        [1, 0, 0, 1],
    #                        [1, 0, 0, 1],
    #                        [0, 1, 1, 0],
    #                        [0, 0, 1, 1],
    #                        [1, 0, 1, 0]]
    #     encoded_a = [1, 1, 1, 1]
    #     b = [0, 1, 1, 0, 1, 0, 1, 0]
    #     decoder = LdpcDecoder(3, 0.01, encoding_matrix, encoded_a)
    #     print(decoder.decode_belief_propagation(b, 100))
    #
    #     deltas = product([1, 2], repeat=block_length)
    #     for cur_delta in deltas:
    #         cur_candidate = np.mod(np.add(b, cur_delta), [3] * block_length).astype(int)
    #         if np.array_equal(np.matmul(cur_candidate, encoding_matrix) % 3, encoded_a):
    #             print("valid candidate:")
    #             print(cur_candidate)

    # def test_decode_single_block(self): # 5 forked indices works
    #     block_length = 12
    #     encoding_matrix = [[1, 0, 0, 1, 0, 0],
    #                        [0, 1, 0, 0, 1, 0],
    #                        [0, 0, 1, 0, 0, 1],
    #                        [0, 0, 1, 1, 0, 0],
    #                        [1, 0, 0, 0, 1, 0],
    #                        [0, 1, 0, 0, 0, 1],
    #                        [0, 1, 0, 1, 0, 0],
    #                        [0, 0, 1, 0, 1, 0],
    #                        [1, 0, 0, 0, 0, 1],
    #                        [1, 1, 0, 0, 0, 0],
    #                        [0, 0, 1, 1, 0, 0],
    #                        [0, 0, 0, 0, 1, 1]]
    #     encoded_a = [1, 1, 1, 1, 0, 2]
    #     b = [0, 1, 1, 0, 1, 0, 1, 0, 2, 2, 0, 1]
    #     decoder = LdpcDecoder(3, 0.01, encoding_matrix, encoded_a)
    #     print(decoder.decode_belief_propagation(b, 100, num_forked_indices=5))
    #
    #     deltas = product([1, 2], repeat=block_length)
    #     for cur_delta in deltas:
    #         cur_candidate = np.mod(np.add(b, cur_delta), [3] * block_length).astype(int)
    #         if np.array_equal(np.matmul(cur_candidate, encoding_matrix) % 3, encoded_a):
    #             print("valid candidate:")
    #             print(cur_candidate)

    # def test_decode_single_block(self): # (20, 15) -> 2 forked indices works (only 1 valid candidate)
    #     block_length = 20
    #     encoding_matrix = [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    #                        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    #                        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    #                        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    #                        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    #                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    #                        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    #                        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    #                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    #                        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #                        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    #                        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    #                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    #                        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    #                        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    #                        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    #                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
    #     encoded_a = [2, 2, 2, 2, 0, 2, 1, 1, 1, 0, 0, 2, 1, 0, 2]
    #     b = [2, 2, 0, 1, 0, 2, 0, 1, 0, 1, 2, 0, 1, 2, 1, 1, 0, 1, 2, 1]
    #     decoder = LdpcDecoder(3, 0.01, encoding_matrix, encoded_a)
    #     print(decoder.decode_belief_propagation(b, 15, num_forked_indices=2))
    #
    #     # deltas = product([1, 2], repeat=block_length)
    #     # for cur_delta in deltas:
    #     #     cur_candidate = np.mod(np.add(b, cur_delta), [3] * block_length).astype(int)
    #     #     if np.array_equal(np.matmul(cur_candidate, encoding_matrix) % 3, encoded_a):
    #     #         print("valid candidate:")
    #     #         print(cur_candidate)

    # def test_decode_single_block(self):  # random (20, 15) -> 2 forked indices works
    #     block_length = 20
    #     num_encoding_columns = 15
    #     sparsity = 4
    #     cfg = ProtocolConfigs(m=3, block_length=block_length, num_blocks=1,
    #                           code_generation_strategy=CodeGenerationStrategy.LDPC_CODE, sparsity=sparsity)
    #     code_gen = CodeGenerator(cfg)
    #     encoding_matrix = code_gen.generate_encoding_matrix(num_encoding_columns)
    #     key_generator = KeyGenerator(p_err=0, key_length=block_length)
    #     a, b = key_generator.generate_keys()
    #     encoder = Encoder(3, block_length)
    #     encoded_a = encoder.encode_single_block(encoding_matrix, a)
    #     decoder = LdpcDecoder(3, 0.01, encoding_matrix, encoded_a)
    #     print(decoder.decode_belief_propagation(b, 15, num_forked_indices=2))
    #
    #     deltas = product([1, 2], repeat=block_length)
    #     for cur_delta in deltas:
    #         cur_candidate = np.mod(np.add(b, cur_delta), [3] * block_length).astype(int)
    #         if np.array_equal(np.matmul(cur_candidate, encoding_matrix) % 3, encoded_a):
    #             print("valid candidate:")
    #             print(cur_candidate)

    # def test_decode_single_block(self):  # random (30, 20) -> 10 forked indices works
    #     block_length = 30
    #     num_encoding_columns = 20
    #     sparsity = 6
    #     cfg = ProtocolConfigs(m=3, block_length=block_length, num_blocks=1,
    #                           code_generation_strategy=CodeGenerationStrategy.LDPC_CODE, sparsity=sparsity)
    #     code_gen = CodeGenerator(cfg)
    #     encoding_matrix = code_gen.generate_encoding_matrix(num_encoding_columns)
    #     key_generator = KeyGenerator(p_err=0, key_length=block_length)
    #     a, b = key_generator.generate_keys()
    #     encoder = Encoder(3, block_length)
    #     encoded_a = encoder.encode_single_block(encoding_matrix, a)
    #     decoder = LdpcDecoder(3, 0.01, encoding_matrix, encoded_a)
    #     print(decoder.decode_belief_propagation(b, 15, num_forked_indices=10))
    #     print("a:")
    #     print(a)

    # deltas = product([1, 2], repeat=block_length)
    # for cur_delta in deltas:
    #     cur_candidate = np.mod(np.add(b, cur_delta), [3] * block_length).astype(int)
    #     if np.array_equal(np.matmul(cur_candidate, encoding_matrix) % 3, encoded_a):
    #         print("valid candidate:")
    #         print(cur_candidate)
    # def test_decode_single_block(self):  # random (1000, 750, 4) -> ? forked indices works
    #     block_length = 9000
    #     num_encoding_columns = 6000
    #     sparsity = 3
    #     cfg = ProtocolConfigs(m=3, block_length=block_length, num_blocks=1,
    #                           code_generation_strategy=CodeGenerationStrategy.LDPC_CODE, sparsity=sparsity)
    #     code_gen = CodeGenerator(cfg)
    #     encoding_matrix = code_gen.generate_encoding_matrix(num_encoding_columns)
    #     key_generator = KeyGenerator(p_err=0, key_length=block_length)
    #     a, b = key_generator.generate_keys()
    #     encoder = Encoder(3, block_length)
    #     encoded_a = encoder.encode_single_block(encoding_matrix, a)
    #     decoder = LdpcDecoder(3, 0.0, encoding_matrix, encoded_a)
    #     a_guess = decoder.decode_belief_propagation(b, 200, a=a)
    #     print("a:")
    #     print(a)
    #     print("a_guess:")
    #     print(a_guess)
    #     print("error:")
    #     print(util.hamming_single_block(a, a_guess))

    # def test_sparse(self):
    #     block_length = 20
    #     encoding_matrix = [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    #                        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    #                        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    #                        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    #                        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    #                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    #                        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    #                        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    #                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    #                        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #                        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    #                        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    #                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    #                        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    #                        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    #                        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    #                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
    #     encoded_a = [2, 2, 2, 2, 0, 2, 1, 1, 1, 0, 0, 2, 1, 0, 2]
    #     b = [2, 2, 0, 1, 0, 2, 0, 1, 0, 1, 2, 0, 1, 2, 1, 1, 0, 1, 2, 1]
    #     normal_decoder = LdpcDecoder(3, 0.01, encoding_matrix, encoded_a)
    #     sparse_decoder = LdpcDecoder(3, 0.01, encoding_matrix, encoded_a, use_sparse=True)
    #     a_normal_guess = normal_decoder.decode_belief_propagation(b, 15, num_forked_indices=2)
    #     a_sparse_guess = sparse_decoder.decode_belief_propagation(b, 15, num_forked_indices=2)
    #     print("encoded_a:")
    #     print(encoded_a)
    #     print("normal results:")
    #     print(a_normal_guess)
    #     print(np.matmul(a_normal_guess, encoding_matrix) % 3)
    #     print("sparse results:")
    #     print(a_sparse_guess)
    #     print(np.matmul(a_sparse_guess, encoding_matrix) % 3)

    # def test_decode_random_ldpc_matrix(self):
    #     n = 50
    #     num_encoding_columns = 50
    #     sparsity = 5
    #     cfg = ProtocolConfigs(m=3, block_length=n, num_blocks=1,
    #                           code_generation_strategy=CodeGenerationStrategy.LDPC_CODE, sparsity=sparsity)
    #     code_gen = CodeGenerator(cfg)
    #     encoding_matrix = code_gen.generate_encoding_matrix(num_encoding_columns)
    #     np.random.seed([os.getppid(), int(str(time.time() % 1)[2:10])])
    #     key_generator = KeyGenerator(p_err=0, key_length=n)
    #     a, b = key_generator.generate_keys()
    #     encoder = Encoder(3, n)
    #     encoded_a = encoder.encode_single_block(encoding_matrix, a)
    #
    #     decoder = LdpcDecoder(3, 0.01, encoding_matrix, encoded_a, use_log=True)
    #     a_guess, best_a_sample = decoder.decode_belief_propagation(b, 100, 0, 100)
    #     print("a:")
    #     print(a)
    #     print("a_guess error:")
    #     print(util.hamming_single_block(a, a_guess))
    #     print("best_a_sample error:")
    #     print(util.hamming_single_block(a, best_a_sample))
    #     print("b:")
    #     print(b)
    #     print("encoding_matrix:")
    #     print(encoding_matrix)
    #     print("encoded_a:")
    #     print(encoded_a)
    #     print(np.matmul(a, encoding_matrix) % 3)
    #     print(np.matmul(a_guess, encoding_matrix) % 3)

    # def test_sparse_matrix_invariants(self):
    #     block_length = 900
    #     num_encoding_columns = 600
    #     sparsity = 3
    #     cfg = ProtocolConfigs(m=3, block_length=block_length, num_blocks=1,
    #                           code_generation_strategy=CodeGenerationStrategy.LDPC_CODE, sparsity=sparsity)
    #     code_gen = CodeGenerator(cfg, use_sparse=True, use_extended=False)
    #     encoding_matrix = code_gen.generate_encoding_matrix(num_encoding_columns)
    #     key_generator = KeyGenerator(p_err=0, key_length=block_length)
    #     a, b = key_generator.generate_keys()
    #     for _ in range(50):
    #         a = key_generator.generate_complement_key(b)
    #         encoded_a = encoding_matrix.encode(a)
    #         decoder = LdpcDecoder(3, 0.0, encoding_matrix, encoded_a, use_sparse_matrix=True)
    #         a_guess, trajectory = decoder.decode_belief_propagation(b, 100, a=a)
    #         # print("a:")
    #         # print(a)
    #         # print("a_guess:")
    #         # print(a_guess)
    #         # print("error:")
    #         print(util.hamming_single_block(a, a_guess))
    #         print(trajectory)

    # def test_sparse_matrix(self):
    #     block_length = 9000
    #     num_encoding_columns = 6000
    #     sparsity = 3
    #     p_err = 0.01
    #     cfg = ProtocolConfigs(m=3, block_length=block_length, num_blocks=1,
    #                           code_generation_strategy=CodeGenerationStrategy.LDPC_CODE, sparsity=sparsity)
    #     code_gen = CodeGenerator(cfg, use_sparse=True, use_extended=False)
    #     for i in range(10):
    #         encoding_matrix = code_gen.generate_encoding_matrix(num_encoding_columns)
    #         key_generator = KeyGenerator(p_err=p_err, key_length=block_length)
    #         a, b = key_generator.generate_keys()
    #         encoded_a = encoding_matrix.encode(a)
    #
    #         print("regular run")
    #         decoder_no_forking = LdpcDecoder(3, p_err, encoding_matrix, encoded_a, use_sparse_matrix=True, use_forking=False)
    #         candidates_left, a_guess_list, encoding_error_trajectory_list, error_trajectory_list, entropy_sum_trajectory_list, final_entropies_list, a_guess_entropy_sum_list = decoder_no_forking.decode_belief_propagation(b, 200, a=a)
    #         if len(candidates_left):
    #             print(util.hamming_single_block(a, a_guess_list[0]))
    #             print(error_trajectory_list[0])
    #             print(encoding_error_trajectory_list[0])
    #             print(
    #                 [int(entropy_sum) if not (math.isnan(entropy_sum) or isinstance(entropy_sum, str)) else entropy_sum
    #                  for entropy_sum in entropy_sum_trajectory_list[0]])
    #
    #         print("with hints")
    #         decoder_with_hints = LdpcDecoder(3, p_err, encoding_matrix, encoded_a, use_sparse_matrix=True,
    #                                          use_hints=True)
    #         candidates_left, a_guess_list, encoding_error_trajectory_list, error_trajectory_list, entropy_sum_trajectory_list, final_entropies_list, a_guess_entropy_sum_list = decoder_with_hints.decode_belief_propagation(
    #             b, 200, a=a)
    #         if len(candidates_left):
    #             print("number of hints: " + str(sum([isinstance(error, str) for error in error_trajectory_list[0]])))
    #             print(util.hamming_single_block(a, a_guess_list[0]))
    #             print(error_trajectory_list[0])
    #             print(encoding_error_trajectory_list[0])
    #             print(
    #                 [int(entropy_sum) if not (isinstance(entropy_sum, str) or math.isnan(entropy_sum)) else entropy_sum
    #                  for entropy_sum in entropy_sum_trajectory_list[0]])
    #
    #         print("with forking")
    #         decoder_with_forking = LdpcDecoder(3, p_err, encoding_matrix, encoded_a, use_sparse_matrix=True, use_forking=True, max_candidates_num=4)
    #         candidates_left, a_guess_list, encoding_error_trajectory_list, error_trajectory_list, entropy_sum_trajectory_list, final_entropies_list, a_guess_entropy_sum_list = decoder_with_forking.decode_belief_propagation(b, 200, a=a)
    #         for i in range(len(a_guess_list)):
    #             if i in candidates_left:
    #                 print("candidate " + str(i) + ":")
    #                 print(util.hamming_single_block(a, a_guess_list[i]))
    #                 if util.closeness_single_block(b, a_guess_list[i]) > 0:
    #                     print("invalid candidate!!!")
    #             else:
    #                 continue
    #                 # print("stopped trajectory " + str(i) + ":")
    #             print(error_trajectory_list[i])
    #             print(encoding_error_trajectory_list[i])
    #             print([int(entropy_sum) if (not isinstance(entropy_sum, str) and not math.isnan(entropy_sum)) else entropy_sum for entropy_sum in entropy_sum_trajectory_list[i]])

    # def test_hints(self):
    #     block_length = 9000
    #     num_encoding_columns = 6000
    #     sparsity = 6
    #     p_err = 0.0
    #     cfg = ProtocolConfigs(m=3, block_length=block_length, num_blocks=1,
    #                           code_generation_strategy=CodeGenerationStrategy.LDPC_CODE, sparsity=sparsity)
    #     code_gen = CodeGenerator(cfg, use_sparse=True, use_extended=False)
    #     for i in range(10):
    #         encoding_matrix = code_gen.generate_encoding_matrix(num_encoding_columns)
    #         key_generator = KeyGenerator(p_err=p_err, key_length=block_length)
    #         a, b = key_generator.generate_keys()
    #         encoded_a = encoding_matrix.encode(a)
    #
    #         decoder_with_hints = LdpcDecoder(3, p_err, encoding_matrix, encoded_a, use_sparse_matrix=True,
    #                                          use_hints=True)
    #         candidates_left, a_guess_list, encoding_error_trajectory_list, error_trajectory_list, entropy_sum_trajectory_list, final_entropies_list, a_guess_entropy_sum_list = decoder_with_hints.decode_belief_propagation(
    #             b, 100, a=a)
    #         if len(candidates_left):
    #             print("number of hints: " + str(sum([isinstance(error, str) for error in error_trajectory_list[0]])))
    #             print(util.hamming_single_block(a, a_guess_list[0]))
    #             print(error_trajectory_list[0])
    #             print(encoding_error_trajectory_list[0])
    #             print(
    #                 [int(entropy_sum) if not (isinstance(entropy_sum, str) or math.isnan(entropy_sum)) else entropy_sum
    #                  for entropy_sum in entropy_sum_trajectory_list[0]])
    #         else:
    #             print("no candidates left")

    # def test_binary(self):
    #     m = 2
    #     block_length = 10000
    #     num_encoding_columns = 1000
    #     sparsity = 500
    #     p_err = 0.02
    #     num_rounds = 100
    #     cfg = ProtocolConfigs(m=m, block_length=block_length, num_blocks=1,
    #                           code_generation_strategy=CodeGenerationStrategy.LDPC_CODE, sparsity=sparsity)
    #     code_gen = CodeGenerator(cfg, use_sparse=True, use_extended=False)
    #     for i in range(10):
    #         encoding_matrix = code_gen.generate_encoding_matrix(num_encoding_columns)
    #         key_generator = KeyGenerator(p_err=p_err, key_length=block_length, m=m)
    #         a, b = key_generator.generate_keys()
    #         encoded_a = encoding_matrix.encode(a)
    #
    #         # print("regular run")
    #         # decoder_no_forking = LdpcDecoder(m, p_err, encoding_matrix, encoded_a, use_sparse_matrix=True, use_forking=False)
    #         # candidates_left, a_guess_list, encoding_error_trajectory_list, error_trajectory_list, entropy_sum_trajectory_list, final_entropies_list, a_guess_entropy_sum_list = decoder_no_forking.decode_belief_propagation(b, num_rounds, a=a)
    #         # if len(candidates_left):
    #         #     print(util.hamming_single_block(a, a_guess_list[0]))
    #         #     print(error_trajectory_list[0])
    #         #     print(encoding_error_trajectory_list[0])
    #         #     print(
    #         #         [int(entropy_sum) if not (math.isnan(entropy_sum) or isinstance(entropy_sum, str)) else entropy_sum
    #         #          for entropy_sum in entropy_sum_trajectory_list[0]])
    #
    #         # print("with hints")
    #         # decoder_with_hints = LdpcDecoder(m, p_err, encoding_matrix, encoded_a, use_sparse_matrix=True,
    #         #                                  use_hints=True)
    #         # candidates_left, a_guess_list, encoding_error_trajectory_list, error_trajectory_list, entropy_sum_trajectory_list, final_entropies_list, a_guess_entropy_sum_list = decoder_with_hints.decode_belief_propagation(
    #         #     b, num_rounds, a=a)
    #         # if len(candidates_left):
    #         #     print("number of hints: " + str(sum([isinstance(error, str) for error in error_trajectory_list[0]])))
    #         #     print(util.hamming_single_block(a, a_guess_list[0]))
    #         #     print(error_trajectory_list[0])
    #         #     print(encoding_error_trajectory_list[0])
    #         #     print(
    #         #         [int(entropy_sum) if not (isinstance(entropy_sum, str) or math.isnan(entropy_sum)) else entropy_sum
    #         #          for entropy_sum in entropy_sum_trajectory_list[0]])
    #
    #         print("with forking")
    #         decoder_with_forking = LdpcDecoder(m, p_err, encoding_matrix, encoded_a, use_sparse_matrix=True, use_forking=True, max_candidates_num=4)
    #         candidates_left, a_guess_list, encoding_error_trajectory_list, error_trajectory_list, entropy_sum_trajectory_list, final_entropies_list, a_guess_entropy_sum_list = decoder_with_forking.decode_belief_propagation(b, num_rounds, a=a)
    #         for i in range(len(a_guess_list)):
    #             if i in candidates_left:
    #                 print("candidate " + str(i) + ":")
    #                 print(util.hamming_single_block(a, a_guess_list[i]))
    #                 if util.closeness_single_block(b, a_guess_list[i]) > 0:
    #                     print("invalid candidate!!!")
    #             else:
    #                 continue
    #                 # print("stopped trajectory " + str(i) + ":")
    #             print(error_trajectory_list[i])
    #             print(encoding_error_trajectory_list[i])
    #             print([int(entropy_sum) if (not isinstance(entropy_sum, str) and not math.isnan(entropy_sum)) else entropy_sum for entropy_sum in entropy_sum_trajectory_list[i]])

    # def test_unique_solution(self):
    #     block_length = 21
    #     num_encoding_columns = 14
    #     sparsity = 3
    #     p_err = 0.0
    #     cfg = ProtocolConfigs(m=3, block_length=block_length, num_blocks=1,
    #                           code_generation_strategy=CodeGenerationStrategy.LDPC_CODE, sparsity=sparsity)
    #     code_gen = CodeGenerator(cfg, use_sparse=True, use_extended=False)
    #     for i in range(10):
    #         encoding_matrix = code_gen.generate_encoding_matrix(num_encoding_columns)
    #         key_generator = KeyGenerator(p_err=p_err, key_length=block_length)
    #         a, b = key_generator.generate_keys()
    #         encoded_a = encoding_matrix.encode(a)
    #
    #         deltas = product([1, 2], repeat=block_length)
    #         valid_candidates = []
    #         for cur_delta in deltas:
    #             cur_candidate = np.mod(np.add(b, cur_delta), [3] * block_length).astype(int)
    #             if np.array_equal(encoding_matrix.encode(cur_candidate), encoded_a):
    #                 valid_candidates.append(cur_candidate)
    #         num_diff_indices = np.sum(np.sum(np.diff(valid_candidates, axis=0), axis=0) > 0)
    #         print("setup " + str(i) + " number valid candidates: " + str(len(valid_candidates)) + "\tnumber diff indices " + str(num_diff_indices))

    def test_nonregular(self):
        block_length = 210
        num_encoding_columns = 200
        sparsity = 3
        p_err = 0.0
        num_rounds = 100
        cfg = cfg(base=3, block_length=block_length, num_blocks=1,
                  code_generation_strategy=CodeGenerationStrategy.ldpc, sparsity=sparsity)
        code_gen = LdpcGenerator(cfg, use_sparse=True, use_extended=True)
        for i in range(10):
            encoding_matrix = code_gen.generate_encoding_matrix(num_encoding_columns)
            key_generator = KeyGenerator(p_err=p_err, key_length=block_length)
            a, b = key_generator.generate_keys()
            encoded_a = encoding_matrix.encode(a)
            decoder_no_forking = LdpcDecoder(3, p_err, encoding_matrix, encoded_a, use_sparse_matrix=True,
                                             use_forking=False)
            candidates_left, a_guess_list, encoding_error_trajectory_list, error_trajectory_list, entropy_sum_trajectory_list, final_entropies_list, a_guess_entropy_sum_list = decoder_no_forking.decode_belief_propagation(
                b, num_rounds, a=a)
            if len(candidates_left):
                print(util.hamming_single_block(a, a_guess_list[0]))
                print(error_trajectory_list[0])
                print(encoding_error_trajectory_list[0])
                print(
                    [int(entropy_sum) if not (math.isnan(entropy_sum) or isinstance(entropy_sum, str)) else entropy_sum
                     for entropy_sum in entropy_sum_trajectory_list[0]])


if __name__ == '__main__':
    unittest.main()
