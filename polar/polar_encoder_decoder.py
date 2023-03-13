import itertools
import random
from timeit import default_timer as timer
from enum import Enum
import numpy as np

from polar.polar_cfg import IndexType


class ProbResult(Enum):
    SuccessActualIsMax = 0
    SuccessActualSmallerThanMax = 1
    FailActualLargerThanMax = 2
    FailActualIsMax = 3
    FailActualWithinRange = 4
    FailActualSmallerThanMin = 5

class PolarEncoderDecoder:
    def __init__(self, cfg, use_log=False):
        self.cfg = cfg

        self.prob_list = None
        self.actual_prob = None
        self.cfg.use_log = use_log

    def encode(self, x_vec_dist, information_vec):
        u_index = 0
        information_vec_index = 0
        assert (len(x_vec_dist) == self.cfg.N)
        assert (len(information_vec) == self.cfg.num_info_indices)

        (encoded_vector, next_u_index, next_information_vec_index) = self.recursive_encode_decode(information_vec, u_index,
                                                                                                  information_vec_index,
                                                                                                  x_vec_dist)

        assert (next_u_index == len(encoded_vector) == len(x_vec_dist) == self.cfg.N)
        assert (next_information_vec_index == len(information_vec) == self.cfg.num_info_indices)
        return encoded_vector

    def decode(self, x_vec_dist, xy_vec_dist):
        u_index = 0
        information_vec_index = 0

        information_vec = np.full(self.cfg.num_info_indices, -1, dtype=np.int64)

        assert (len(x_vec_dist) == len(xy_vec_dist) == self.cfg.N)

        (encoded_vector, next_u_index, next_information_vec_index) = self.recursive_encode_decode(information_vec, u_index,
                                                                                                  information_vec_index,
                                                                                                  x_vec_dist,
                                                                                                  xy_vec_dist)

        assert (next_u_index == len(encoded_vector) == self.cfg.N)
        assert (next_information_vec_index == len(information_vec) == self.cfg.num_info_indices)

        return information_vec

    def list_decode(self, xy_vec_dist, frozen_values, check_matrix, check_value, actual_information=None):
        u_index = 0
        information_vec_index = 0

        information_list = np.full((self.cfg.scl_l * self.cfg.q, self.cfg.num_info_indices), -1, dtype=np.int64)
        frozen_values_iterator = None
        if len(frozen_values):
            frozen_values_iterator = np.nditer(frozen_values, flags=['f_index'])

        assert (len(xy_vec_dist) == self.cfg.N)

        self.actual_information = actual_information
        if self.cfg.use_log:
            self.actual_prob = 0.0
            self.prob_list = np.array([0.0])
        else:
            self.actual_prob = 1.0
            self.prob_list = np.array([1.0])

        start = timer()
        (information_list, encoded_vector_list, next_u_index, next_information_vector_index, final_list_size,
         original_indices_map, actual_encoding) = self.recursive_list_decode(information_list, u_index, information_vec_index,
                                                                             [xy_vec_dist], frozen_values_iterator, in_list_size=1, actual_xy_vec_dist=xy_vec_dist)
        end = timer()

        assert (1 <= final_list_size <= self.cfg.scl_l)
        assert (len(encoded_vector_list) == final_list_size)
        assert (next_u_index == len(encoded_vector_list[0]) == self.cfg.N)
        assert (next_information_vector_index == self.cfg.num_info_indices)
        assert (len(original_indices_map) == final_list_size)
        assert (np.count_nonzero(original_indices_map) == 0)

        if actual_information is not None:
            explicit_probs, normalization = normalize([self.calc_explicit_prob(information, frozen_values, xy_vec_dist) for information
                                                       in information_list[:self.cfg.scl_l]], use_log=self.cfg.use_log)
            actual_explicit_prob = self.calc_explicit_prob(actual_information, frozen_values, xy_vec_dist)
            if self.cfg.use_log:
                actual_explicit_prob = actual_explicit_prob - normalization
            else:
                actual_explicit_prob = actual_explicit_prob / normalization

            for i, information in enumerate(information_list[:self.cfg.scl_l]):
                if np.array_equal(information, actual_information):
                    max_prob = max(self.prob_list)
                    if self.prob_list[i] == max_prob:
                        prob_result = ProbResult.SuccessActualIsMax
                    else:
                        prob_result = ProbResult.SuccessActualSmallerThanMax
                    return information, prob_result

            max_prob = max(self.prob_list)
            if self.actual_prob > max_prob:
                prob_result = ProbResult.FailActualLargerThanMax
            elif self.actual_prob == max_prob:
                prob_result = ProbResult.FailActualIsMax
            elif self.actual_prob >= min(self.prob_list):
                prob_result = ProbResult.FailActualWithinRange
            else:
                prob_result = ProbResult.FailActualSmallerThanMin
            return information_list[0], prob_result

        candidate_list = np.array([np.array_equal(np.matmul(information, check_matrix) % self.cfg.q, check_value) for information in information_list[:self.cfg.scl_l]])
        if True in candidate_list:
            for i, val in enumerate(candidate_list):
                if val:
                    return information_list[i], None

        return information_list[0], None

    def calc_explicit_prob(self, information, frozenValues, xyVectorDistribution):
        guess = polarTransformOfQudits(self.cfg.q, self.mergeInfoAndFrozen(information, frozenValues))
        probs_list = [xyProb[guess[i]] for i, xyProb in enumerate(xyVectorDistribution.probs)]
        if self.cfg.use_log:
            guess_prob = sum(probs_list)
        else:
            guess_prob = np.prod(probs_list)
        return guess_prob

    def mergeInfoAndFrozen(self, actualInformation, frozenValues):
        mergedVector = np.empty(self.cfg.N, dtype=np.int)
        mergedVector[list(self.cfg.info_set)] = actualInformation
        mergedVector[list(self.cfg.frozen_set)] = frozenValues
        return mergedVector

    def recursive_encode_decode(self, information, uIndex, informationVectorIndex, xVectorDistribution,
                                xyVectorDistribution=None, marginalizedUProbs=None):
        # By default, we assume encoding, and add small corrections for decoding.
        encodedVector = np.full(len(xVectorDistribution), -1, dtype=np.int64)
        decoding = xyVectorDistribution is not None

        if len(xVectorDistribution) == 1:
            if self.cfg.index_types[uIndex] == IndexType.info:
                if decoding:
                    marginalizedVector = xyVectorDistribution.calcMarginalizedProbabilities()
                    information[informationVectorIndex] = np.argmax(marginalizedVector)
                encodedVector[0] = information[informationVectorIndex]
                next_uIndex = uIndex + 1
                next_informationVectorIndex = informationVectorIndex + 1
            else:
                # marginalizedVector = xVectorDistribution.calcMarginalizedProbabilities()
                # encodedVector[0] = min(
                #     np.searchsorted(np.cumsum(marginalizedVector), self.randomlyGeneratedNumbers[uIndex]),
                #     self.cfg.q - 1)
                encodedVector[0] = 0
                next_uIndex = uIndex + 1
                next_informationVectorIndex = informationVectorIndex

            # should we return the marginalized probabilities?
            if marginalizedUProbs is not None:
                vectorDistribution = xyVectorDistribution or xVectorDistribution
                marginalizedVector = vectorDistribution.calcMarginalizedProbabilities()
                marginalizedUProbs.append(marginalizedVector)

            return (encodedVector, next_uIndex, next_informationVectorIndex)
        else:
            xMinusVectorDistribution = xVectorDistribution.minusTransform()
            xMinusVectorDistribution.normalize()
            if decoding:
                xyMinusVectorDistribution = xyVectorDistribution.minusTransform()
                xyMinusVectorDistribution.normalize()
            else:
                xyMinusVectorDistribution = None

            (minusEncodedVector, next_uIndex, next_informationVectorIndex) = self.recursive_encode_decode(information,
                                                                                                          uIndex,
                                                                                                          informationVectorIndex,
                                                                                                          xMinusVectorDistribution,
                                                                                                          xyMinusVectorDistribution,
                                                                                                          marginalizedUProbs)

            xPlusVectorDistribution = xVectorDistribution.plusTransform(minusEncodedVector)
            xPlusVectorDistribution.normalize()
            if decoding:
                xyPlusVectorDistribution = xyVectorDistribution.plusTransform(minusEncodedVector)
                xyPlusVectorDistribution.normalize()
            else:
                xyPlusVectorDistribution = None

            uIndex = next_uIndex
            informationVectorIndex = next_informationVectorIndex
            (plusEncodedVector, next_uIndex, next_informationVectorIndex) = self.recursive_encode_decode(information,
                                                                                                         uIndex,
                                                                                                         informationVectorIndex,
                                                                                                         xPlusVectorDistribution,
                                                                                                         xyPlusVectorDistribution,
                                                                                                         marginalizedUProbs)

            halfLength = len(xVectorDistribution) // 2

            for halfi in range(halfLength):
                encodedVector[2 * halfi] = (minusEncodedVector[halfi] + plusEncodedVector[halfi]) % self.cfg.q
                encodedVector[2 * halfi + 1] = (-plusEncodedVector[halfi] + self.cfg.q) % self.cfg.q

            return (encodedVector, next_uIndex, next_informationVectorIndex)

    def recursive_list_decode(self, informationList, uIndex, informationVectorIndex,
                              xyVectorDistributionList, frozenValuesIterator=None, marginalizedUProbs=None, in_list_size=1, maxListSize=1,
                              actual_xy_vec_dist=None):
        assert (in_list_size <= maxListSize)
        assert (in_list_size == len(self.prob_list))
        segmentSize = len(xyVectorDistributionList[0])

        if segmentSize == 1:
            if self.cfg.index_types[uIndex] == IndexType.info:
                encodedVectorList = np.full((in_list_size * self.cfg.q, segmentSize), -1, dtype=np.int64)
                newProbList = np.empty(in_list_size * self.cfg.q, dtype=np.float)
                for i in range(in_list_size):
                    information = informationList[i]
                    marginalizedVector = xyVectorDistributionList[i].calcMarginalizedProbabilities()
                    start = timer()
                    for s in range(self.cfg.q):
                        if s > 0:
                            informationList[s * in_list_size + i] = information  # branch the paths q times
                        informationList[s * in_list_size + i][informationVectorIndex] = s
                        encodedVectorList[s * in_list_size + i][0] = s
                        if self.cfg.use_log:
                            newProbList[s * in_list_size + i] = self.prob_list[i] + marginalizedVector[s]
                        else:
                            newProbList[s * in_list_size + i] = self.prob_list[i] * marginalizedVector[s]
                    end = timer()
                    self.info_time += end - start
                newListSize = in_list_size * self.cfg.q
                next_uIndex = uIndex + 1
                next_informationVectorIndex = informationVectorIndex + 1

                if newListSize > maxListSize:
                    if self.cfg.use_log:
                        newListSize = min(newListSize - np.isneginf(newProbList).sum(), maxListSize)
                    else:
                        newListSize = min(np.count_nonzero(newProbList), maxListSize)
                    indices_to_keep = np.argpartition(newProbList, -newListSize)[-newListSize:]
                    original_indices_map = indices_to_keep % in_list_size
                    start = timer()
                    informationList[0:newListSize] = informationList[indices_to_keep]
                    informationList[newListSize:, :] = -1
                    end = timer()
                    self.info_time += end - start
                    encodedVectorList = encodedVectorList[indices_to_keep]
                    newProbList = newProbList[indices_to_keep]
                else:
                    original_indices_map = np.tile(np.arange(in_list_size), self.cfg.q)

                self.prob_list, normalizationWeight = normalize(newProbList, use_log=self.cfg.use_log)

                if actual_xy_vec_dist is not None:
                    actualEncodedVector = [self.actual_information[informationVectorIndex]]
                    if self.cfg.use_log:
                        self.actual_prob += actual_xy_vec_dist.calcMarginalizedProbabilities()[actualEncodedVector[0]] - normalizationWeight
                    else:
                        self.actual_prob *= actual_xy_vec_dist.calcMarginalizedProbabilities()[actualEncodedVector[0]] / normalizationWeight
            else:
                newListSize = in_list_size
                frozenValue = frozenValuesIterator[0]
                encodedVectorList = np.full((in_list_size, segmentSize), frozenValue, dtype=np.int64)
                if actual_xy_vec_dist is not None:
                    actualEncodedVector = [frozenValue]
                frozenValuesIterator.iternext()
                # encodedVectorList = np.full((inListSize, segmentSize), 0, dtype=np.int64)
                next_uIndex = uIndex + 1
                next_informationVectorIndex = informationVectorIndex
                original_indices_map = np.arange(in_list_size)
                if self.cfg.use_log:
                    self.prob_list, normalizationWeight = normalize([prob + xyVectorDistributionList[i].calcMarginalizedProbabilities()[frozenValue] for i, prob in enumerate(self.prob_list)], use_log=self.cfg.use_log)
                    self.actual_prob += actual_xy_vec_dist.calcMarginalizedProbabilities()[frozenValue] - normalizationWeight
                else:
                    self.prob_list, normalizationWeight = normalize([prob * xyVectorDistributionList[i].calcMarginalizedProbabilities()[frozenValue] for i, prob in enumerate(self.prob_list)], use_log=self.cfg.use_log)
                    self.actual_prob *= actual_xy_vec_dist.calcMarginalizedProbabilities()[frozenValue] / normalizationWeight

            return (informationList, encodedVectorList, next_uIndex, next_informationVectorIndex, newListSize,
                    original_indices_map, actualEncodedVector)
        else:
            numInfoBits = np.sum((self.cfg.index_types[uIndex:uIndex + segmentSize] == IndexType.info))

            # Rate-0 node
            if numInfoBits == 0:
                frozenVector = np.empty(segmentSize, dtype=np.int64)
                for i in range(segmentSize):
                    frozenVector[i] = frozenValuesIterator[0]
                    frozenValuesIterator.iternext()
                encodedVector = polarTransformOfQudits(self.cfg.q, frozenVector)
                encodedVectorList = [encodedVector] * in_list_size
                if self.cfg.use_log:
                    newProbList = [prob + np.sum([xyVectorDistribution.probs[i, encodedVector[i]] for i in range(segmentSize)]) for xyVectorDistribution, prob in zip(xyVectorDistributionList, self.prob_list)]
                    self.prob_list, normalizationWeight = normalize(newProbList, use_log=self.cfg.use_log)
                    self.actual_prob += np.sum([actual_xy_vec_dist.probs[i, encodedVector[i]] for i in range(segmentSize)]) - normalizationWeight
                else:
                    newProbList = [prob * np.product([xyVectorDistribution.probs[i, encodedVector[i]] for i in range(segmentSize)]) for xyVectorDistribution, prob in zip(xyVectorDistributionList, self.prob_list)]
                    self.prob_list, normalizationWeight = normalize(newProbList, use_log=self.cfg.use_log)
                    self.actual_prob *= np.product([actual_xy_vec_dist.probs[i, encodedVector[i]] for i in
                                                    range(segmentSize)]) / normalizationWeight

                next_uIndex = uIndex + segmentSize
                next_informationVectorIndex = informationVectorIndex
                newListSize = in_list_size
                originalIndicesMap = np.arange(in_list_size)
                actualEncodedVector = encodedVector
                return (informationList, encodedVectorList, next_uIndex, next_informationVectorIndex, newListSize,
                 originalIndicesMap, actualEncodedVector)

            # Rep node
            if numInfoBits == 1:
                k = np.where(self.cfg.index_types[uIndex:uIndex + segmentSize] == IndexType.info)[0][0]
                inputVectorSplits = np.empty((self.cfg.q, segmentSize), dtype=np.int64)
                for i in range(segmentSize):
                    if i != k:
                        inputVectorSplits[:, i] = frozenValuesIterator[0]
                        frozenValuesIterator.iternext()
                    else:
                        inputVectorSplits[:, i] = np.arange(self.cfg.q)
                encodedVectorSplits = np.array([polarTransformOfQudits(self.cfg.q, v) for v in inputVectorSplits])

                newProbList = np.empty(in_list_size * self.cfg.q, dtype=np.float)
                for i in range(in_list_size):
                    information = informationList[i]
                    start = timer()
                    for s in range(self.cfg.q):
                        if s > 0:
                            informationList[s * in_list_size + i] = information  # branch the paths q times
                        informationList[s * in_list_size + i][informationVectorIndex] = s
                        if self.cfg.use_log:
                            newProbList[s * in_list_size + i] = self.prob_list[i] + np.sum([xyVectorDistributionList[i].probs[j, encodedVectorSplits[s, j]] for j in range(segmentSize)])
                        else:
                            newProbList[s * in_list_size + i] = self.prob_list[i] * np.product([xyVectorDistributionList[i].probs[j, encodedVectorSplits[s, j]] for j in range(segmentSize)])
                    end = timer()
                    self.info_time += end - start
                newListSize = in_list_size * self.cfg.q

                if newListSize > maxListSize:
                    if self.cfg.use_log:
                        newListSize = min(newListSize - np.isneginf(newProbList).sum(), maxListSize)
                    else:
                        newListSize = min(np.count_nonzero(newProbList), maxListSize)
                    indices_to_keep = np.argpartition(newProbList, -newListSize)[-newListSize:]
                    original_indices_map = indices_to_keep % in_list_size
                    start = timer()
                    informationList[0:newListSize] = informationList[indices_to_keep]
                    informationList[newListSize:, :] = -1
                    end = timer()
                    self.info_time += end - start
                    encodedVectorList = encodedVectorSplits[indices_to_keep // in_list_size]
                    newProbList = newProbList[indices_to_keep]
                else:
                    encodedVectorList = np.repeat(encodedVectorSplits, in_list_size, axis=0)
                    original_indices_map = np.tile(np.arange(in_list_size), self.cfg.q)

                self.prob_list, normalizationWeight = normalize(newProbList, use_log=self.cfg.use_log)

                if actual_xy_vec_dist is not None:
                    actualEncodedVector = encodedVectorSplits[self.actual_information[informationVectorIndex]]
                    if self.cfg.use_log:
                        self.actual_prob += np.sum([actual_xy_vec_dist.probs[i, actualEncodedVector[i]] for i in range(segmentSize)]) - normalizationWeight
                    else:
                        self.actual_prob *= np.product([actual_xy_vec_dist.probs[i, actualEncodedVector[i]] for i in range(segmentSize)]) / normalizationWeight

                next_uIndex = uIndex + segmentSize
                next_informationVectorIndex = informationVectorIndex + 1
                return (informationList, encodedVectorList, next_uIndex, next_informationVectorIndex, newListSize,
                        original_indices_map, actualEncodedVector)

            # Rate-1 node
            if numInfoBits == segmentSize:
                fork_size = self.cfg.q ** 2
                encodedVectorList = np.empty((in_list_size * fork_size, segmentSize), dtype=np.int64)
                newProbList = np.empty(in_list_size * fork_size, dtype=np.float)
                for i in range(in_list_size):
                    # Pick the 2 least reliable indices
                    [j1, j2] = self.pickLeastReliableIndices(xyVectorDistributionList[i].probs, 2)
                    # Fork there
                    encodedVectorList[fork_size*i : fork_size*(i + 1)], newProbList[fork_size*i : fork_size*(i + 1)] = self.forkIndices(self.prob_list[i], xyVectorDistributionList[i].probs, segmentSize, [j1, j2])
                newListSize = in_list_size * fork_size

                # Prune
                if newListSize > maxListSize:
                    if self.cfg.use_log:
                        newListSize = min(newListSize - np.isneginf(newProbList).sum(), maxListSize)
                    else:
                        newListSize = min(np.count_nonzero(newProbList), maxListSize)
                    indices_to_keep = np.argpartition(newProbList, -newListSize)[-newListSize:]
                    encodedVectorList = encodedVectorList[indices_to_keep]
                    newProbList = newProbList[indices_to_keep]
                    original_indices_map = indices_to_keep // fork_size
                else:
                    original_indices_map = np.repeat(np.arange(in_list_size), fork_size)

                # Normalize
                self.prob_list, normalizationWeight = normalize(newProbList, use_log=self.cfg.use_log)
                if actual_xy_vec_dist is not None:
                    actualEncodedVector = polarTransformOfQudits(self.cfg.q, self.actual_information[informationVectorIndex:informationVectorIndex + segmentSize])
                    if self.cfg.use_log:
                        self.actual_prob += np.sum([actual_xy_vec_dist.probs[i, actualEncodedVector[i]] for i in
                                                    range(segmentSize)]) - normalizationWeight
                    else:
                        self.actual_prob *= np.product([actual_xy_vec_dist.probs[i, actualEncodedVector[i]] for i in
                                                        range(segmentSize)]) / normalizationWeight

                # Update informationList
                start = timer()
                informationList[0:newListSize] = informationList[original_indices_map]
                informationList[0:newListSize, informationVectorIndex:informationVectorIndex+segmentSize] = [polarTransformOfQudits(self.cfg.q, encodedVector) for encodedVector in encodedVectorList]
                informationList[newListSize:] = -1
                end = timer()
                self.info_time += end - start

                next_uIndex = uIndex + segmentSize
                next_informationVectorIndex = informationVectorIndex + segmentSize

                return (informationList, encodedVectorList, next_uIndex, next_informationVectorIndex, newListSize,
                        original_indices_map, actualEncodedVector)

            # SPC node
            if numInfoBits == segmentSize - 1:
                num_forked_indices = 3

                fork_size = self.cfg.q ** num_forked_indices
                encodedVectorList = np.empty((in_list_size * fork_size, segmentSize), dtype=np.int64)
                newProbList = np.empty(in_list_size * fork_size, dtype=np.float)
                frozenValue = frozenValuesIterator[0]
                for i in range(in_list_size):
                    # Pick the least reliable indices
                    leastReliableIndices = self.pickLeastReliableIndices(xyVectorDistributionList[i].probs, num_forked_indices + 1)
                    # Fork there
                    encodedVectorList[fork_size*i: fork_size*(i + 1)], newProbList[fork_size*i: fork_size*(i + 1)] = self.forkIndicesSpc(self.prob_list[i], xyVectorDistributionList[i].probs, segmentSize, leastReliableIndices, frozenValue)
                frozenValuesIterator.iternext()
                newListSize = in_list_size * fork_size

                # Prune
                if newListSize > maxListSize:
                    if self.cfg.use_log:
                        newListSize = min(newListSize - np.isneginf(newProbList).sum(), maxListSize)
                    else:
                        newListSize = min(np.count_nonzero(newProbList), maxListSize)
                    indices_to_keep = np.argpartition(newProbList, -newListSize)[-newListSize:]
                    encodedVectorList = encodedVectorList[indices_to_keep]
                    newProbList = newProbList[indices_to_keep]
                    original_indices_map = indices_to_keep // fork_size
                else:
                    original_indices_map = np.repeat(np.arange(in_list_size), fork_size)

                # Normalize
                self.prob_list, normalizationWeight = normalize(newProbList, use_log=self.cfg.use_log)
                if actual_xy_vec_dist is not None:
                    actualEncodedVector = polarTransformOfQudits(self.cfg.q, np.concatenate(([frozenValue], self.actual_information[informationVectorIndex:informationVectorIndex + segmentSize - 1]), axis=None))
                    if self.cfg.use_log:
                        self.actual_prob += np.sum([actual_xy_vec_dist.probs[i, actualEncodedVector[i]] for i in
                                                    range(segmentSize)]) - normalizationWeight
                    else:
                        self.actual_prob *= np.product([actual_xy_vec_dist.probs[i, actualEncodedVector[i]] for i in
                                                        range(segmentSize)]) / normalizationWeight

                # Update informationList
                start = timer()
                informationList[0:newListSize] = informationList[original_indices_map]
                informationList[0:newListSize, informationVectorIndex:informationVectorIndex+segmentSize-1] = [polarTransformOfQudits(self.cfg.q, encodedVector)[1:] for encodedVector in encodedVectorList]
                informationList[newListSize:] = -1
                end = timer()
                self.info_time += end - start

                next_uIndex = uIndex + segmentSize
                next_informationVectorIndex = informationVectorIndex + segmentSize - 1

                return (informationList, encodedVectorList, next_uIndex, next_informationVectorIndex, newListSize,
                        original_indices_map, actualEncodedVector)

            start = timer()
            xyMinusVectorDistributionList = []
            for i in range(in_list_size):
                xyMinusVectorDistribution = xyVectorDistributionList[i].minusTransform()
                xyMinusVectorDistribution.normalize()
                xyMinusVectorDistributionList.append(xyMinusVectorDistribution)
            # xyMinusVectorDistributionList, xyMinusNormalizationVector = normalizeDistList(xyMinusVectorDistributionList)
            end = timer()
            self.transform_time += end-start
            if actual_xy_vec_dist is not None:
                actualXyMinusVectorDistribution = actual_xy_vec_dist.minusTransform()
                actualXyMinusVectorDistribution.normalize()
                # actualXyMinusVectorDistribution.normalize(xyMinusNormalizationVector)

            (minusInformationList, minusEncodedVectorList, next_uIndex, next_informationVectorIndex, minusListSize,
             minusOriginalIndicesMap, minusActualEncodedVector) = self.recursive_list_decode(informationList, uIndex, informationVectorIndex,
                                                                                             xyMinusVectorDistributionList, frozenValuesIterator, marginalizedUProbs,
                                                                                             in_list_size, maxListSize, actual_xy_vec_dist=actualXyMinusVectorDistribution)

            start = timer()
            xyPlusVectorDistributionList = []
            for i in range(minusListSize):
                origI = minusOriginalIndicesMap[i]

                xyPlusVectorDistribution = xyVectorDistributionList[origI].plusTransform(minusEncodedVectorList[i])
                xyPlusVectorDistribution.normalize()
                xyPlusVectorDistributionList.append(xyPlusVectorDistribution)
            # xyPlusVectorDistributionList, xyPlusNormalizationVector = normalizeDistList(xyPlusVectorDistributionList)
            end = timer()
            self.transform_time += end-start
            if actual_xy_vec_dist is not None:
                actualXyPlusVectorDistribution = actual_xy_vec_dist.plusTransform(minusActualEncodedVector)
                actualXyPlusVectorDistribution.normalize()
                # actualXyPlusVectorDistribution.normalize(xyPlusNormalizationVector)

            uIndex = next_uIndex
            informationVectorIndex = next_informationVectorIndex
            (plusInformationList, plusEncodedVectorList, next_uIndex, next_informationVectorIndex, plusListSize,
             plusOriginalIndicesMap, plusActualEncodedVector) = self.recursive_list_decode(minusInformationList, uIndex, informationVectorIndex,
                                                                                           xyPlusVectorDistributionList, frozenValuesIterator, marginalizedUProbs,
                                                                                           minusListSize, maxListSize, actual_xy_vec_dist=actualXyPlusVectorDistribution)

            newListSize = plusListSize

            encodedVectorList = np.full((newListSize, segmentSize), -1, dtype=np.int64)
            # halfLength = segmentSize // 2

            start = timer()
            for i in range(newListSize):
                minusI = plusOriginalIndicesMap[i]
                encodedVectorList[i][::2] = minusEncodedVectorList[minusI] + plusEncodedVectorList[i]
                encodedVectorList[i][1::2] = -plusEncodedVectorList[i]
                encodedVectorList[i] %= self.cfg.q
                # for halfi in range(halfLength):
                #     encodedVectorList[i][2 * halfi] = (minusEncodedVectorList[minusI][halfi] + plusEncodedVectorList[i][
                #         halfi]) % self.cfg.q
                #     encodedVectorList[i][2 * halfi + 1] = (-plusEncodedVectorList[i][halfi] + self.cfg.q) % self.cfg.q
            end = timer()
            self.encoding_time += end-start

            if actual_xy_vec_dist is not None:
                actualEncodedVector = np.full(segmentSize, -1, dtype=np.int64)
                actualEncodedVector[::2] = np.add(minusActualEncodedVector, plusActualEncodedVector)
                actualEncodedVector[1::2] = np.array(plusActualEncodedVector) * (-1)
                actualEncodedVector %= self.cfg.q
                # for halfi in range(halfLength):
                #     actualEncodedVector[2 * halfi] = (minusActualEncodedVector[halfi] + plusActualEncodedVector[
                #         halfi]) % self.cfg.q
                #     actualEncodedVector[2 * halfi + 1] = (-plusActualEncodedVector[halfi] + self.cfg.q) % self.cfg.q

            originalIndicesMap = minusOriginalIndicesMap[plusOriginalIndicesMap]

            return (plusInformationList, encodedVectorList, next_uIndex, next_informationVectorIndex, newListSize,
                    originalIndicesMap, actualEncodedVector)

    def pickLeastReliableIndices(self, xyDist, numUnreliableIndices):
        scores = [self.reliability(cur_probs) for cur_probs in xyDist]
        return np.argpartition(scores, range(-numUnreliableIndices, 1))[-numUnreliableIndices:]

    def reliability(self, probs):
        max_probs = np.partition(probs, -2)[-2:]
        if self.cfg.use_log:
            return max_probs[0] - max_probs[1]
        else:
            return max_probs[0] / max_probs[1]

    def forkIndices(self, cur_prob, xyDist, segmentSize, indicesToFork):
        numIndicesToFork = len(indicesToFork)
        if numIndicesToFork != 2:
            raise "No support for number of indices to fork: " + str(numIndicesToFork)

        num_forks = self.cfg.q ** numIndicesToFork
        forks = np.empty((num_forks, segmentSize), dtype=np.int64)
        constant_indices_mask = np.ones(segmentSize, dtype=bool)
        constant_indices_mask[indicesToFork] = False

        forks[:, constant_indices_mask] = np.argmax(xyDist[constant_indices_mask], axis=1)
        forks[:, indicesToFork] = list(itertools.product(np.arange(self.cfg.q), repeat=numIndicesToFork))

        fork_prob_combinations = list(itertools.product(xyDist[indicesToFork[0]], xyDist[indicesToFork[1]]))
        if self.cfg.use_log:
            base_prob = cur_prob + sum(np.max(xyDist[constant_indices_mask], axis=1))
            prob_list = np.sum(fork_prob_combinations, axis=1) + base_prob
        else:
            base_prob = cur_prob * np.product(np.max(xyDist[constant_indices_mask], axis=1))
            prob_list = np.product(fork_prob_combinations, axis=1) * base_prob

        return forks, prob_list

    def forkIndicesSpc(self, cur_prob, xyDist, segmentSize, leastReliableIndices, frozenValue):
        numIndicesToFork = len(leastReliableIndices) - 1
        dependentIndex = leastReliableIndices[-1]
        indicesToFork = leastReliableIndices[:-1]
        # dependentIndex = leastReliableIndices[0]
        # indicesToFork = leastReliableIndices[1:]

        num_forks = self.cfg.q ** numIndicesToFork
        forks = np.empty((num_forks, segmentSize), dtype=np.int64)
        constant_indices_mask = np.ones(segmentSize, dtype=bool)
        constant_indices_mask[leastReliableIndices] = False

        forked_values = np.argmax(xyDist[constant_indices_mask], axis=1)
        forks[:, constant_indices_mask] = forked_values
        base_frozen_delta = (frozenValue - sum(forked_values)) % self.cfg.q
        forks[:, indicesToFork] = list(itertools.product(np.arange(self.cfg.q), repeat=numIndicesToFork))
        forks[:, dependentIndex] = np.mod(
            (base_frozen_delta - np.sum(forks[:, indicesToFork], axis=1)), self.cfg.q)

        fork_prob_combinations = np.array([[xyDist[i, fork[i]] for i in leastReliableIndices] for fork in forks])
        if self.cfg.use_log:
            base_prob = cur_prob + sum(np.max(xyDist[constant_indices_mask], axis=1))
            prob_list = np.sum(fork_prob_combinations, axis=1) + base_prob
        else:
            base_prob = cur_prob * np.product(np.max(xyDist[constant_indices_mask], axis=1))
            prob_list = np.product(fork_prob_combinations, axis=1) * base_prob

        return forks, prob_list

    def calculate_syndrome_and_complement(self, u_message):
        y = polarTransformOfQudits(self.cfg.q, u_message)

        w = np.copy(y)
        w[list(self.cfg.info_set)] = 0
        w[list(self.cfg.frozen_set)] *= self.cfg.q - 1
        w[list(self.cfg.frozen_set)] %= self.cfg.q

        u = y
        u[list(self.cfg.frozen_set)] = 0

        return w, u

    def get_message_info_bits(self, u_message):
        return u_message[list(self.cfg.info_set)]

    def get_message_frozen_bits(self, u_message):
        return u_message[list(self.cfg.frozen_set)]

def normalize(prob_list, use_log=False):
    maxProb = np.max(prob_list)
    if use_log:
        return prob_list - maxProb, maxProb
    else:
        return prob_list / maxProb, maxProb

def calcNormalizationVector(dist_list):
    segment_size = len(dist_list[0].probs)
    normalization = np.zeros(segment_size)
    for i in range(segment_size):
        normalization[i] = max([dist.probs[i].max(axis=0) for dist in dist_list])
    return normalization

def normalizeDistList(dist_list):
    normalization_vector = calcNormalizationVector(dist_list)
    for dist in dist_list:
        dist.normalize(normalization_vector)
    return dist_list, normalization_vector

def polarTransformOfQudits(q, xvec):
    # print("xvec =", xvec)
    if len(xvec) == 1:
        return xvec
    else:
        if len(xvec) % 2 != 0:
            print(xvec)
        assert (len(xvec) % 2 == 0)

        vfirst = []
        vsecond = []
        for i in range((len(xvec) // 2)):
            vfirst.append((xvec[2 * i] + xvec[2 * i + 1]) % q)
            vsecond.append((q - xvec[2 * i + 1]) % q)

        ufirst = polarTransformOfQudits(q, vfirst)
        usecond = polarTransformOfQudits(q, vsecond)

        return np.concatenate((ufirst, usecond))