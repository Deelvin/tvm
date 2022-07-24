# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=consider-using-enumerate, invalid-name, invalid-sequence-index
"""
Cost model optimizer based on simulated annealing
"""

import heapq
import logging
import time

import numpy as np

from ..utils import sample_ints
from .model_based_tuner import ModelOptimizer, knob2point, point2knob

logger = logging.getLogger("autotvm")


class SimulatedAnnealingOptimizer(ModelOptimizer):
    """parallel simulated annealing optimization algorithm

    Parameters
    ----------
    task: Task
        The tuning task
    n_iter: int
        The number of iterations of simulated annealing
    temp: float or Array of float
        If is a single float, then use a constant temperature.
        If is an Array, then perform linear cooling from temp[0] to temp[1]
    early_stop: int, optional
        Stop iteration if the optimal set do not change in `early_stop` rounds
    log_interval: int, optional
        Print log every `log_interval` iterations
    """

    def __init__(
        self,
        task,
        n_iter=500,
        temp=(1, 0),
        persistent=True,
        parallel_size=128,
        early_stop=50,
        log_interval=50,
    ):
        super(SimulatedAnnealingOptimizer, self).__init__()

        self.task = task
        self.dims = [len(x) for x in self.task.config_space.space_map.values()] # ICE 

        self.n_iter = n_iter
        self.temp = temp
        self.persistent = persistent
        self.parallel_size = min(parallel_size, len(self.task.config_space))
        self.early_stop = early_stop or 1e9
        self.log_interval = log_interval
        self.points = None

    def find_maximums(self, model, num, exclusive):
        tic = time.time()
        temp, n_iter, early_stop, log_interval = (
            self.temp,
            self.n_iter,
            self.early_stop,
            self.log_interval,
        )

        if self.persistent and self.points is not None:
            points = self.points
        else:
            points = np.array(sample_ints(0, len(self.task.config_space), self.parallel_size, self.task.config_space.check_index))

        scores = model.predict(points) # ICE ERROR ANDEY predict invalid points?

        # build heap and insert initial points
        heap_items = [(float("-inf"), -1 - i) for i in range(num)] # num???
        heapq.heapify(heap_items)
        in_heap = set(exclusive) # exclusive ??? filter?
        # print("ICE find_maximums exclusive in_heap", repr(in_heap), flush=True)
        in_heap.update([x[1] for x in heap_items])
        # print("ICE find_maximums exclusive in_heap updated", repr(in_heap), flush=True)
        for s, p in zip(scores, points):
            if s > heap_items[0][0] and p not in in_heap:
                pop = heapq.heapreplace(heap_items, (s, p))
                in_heap.remove(pop[1])
                in_heap.add(p)

        k = 0
        k_last_modify = 0

        if isinstance(temp, (tuple, list, np.ndarray)):
            t = temp[0]
            cool = 1.0 * (temp[0] - temp[1]) / (n_iter + 1)
        else:
            t = temp
            cool = 0
        # print("ICE random_walk fi:",self.task.config_space.filtered_indexes, "a2f:", self.task.config_space.a2f_indexes, flush=True)
        while k < n_iter and k < k_last_modify + early_stop:
            new_points = np.empty_like(points)
            for i, p in enumerate(points):
                # if (self.task.config_space.a2f_indexes) and (p not in self.task.config_space.a2f_indexes):
                #     print("ICE Found index which should be in ice_unblocked but it is not, ", p)
                new_points[i] = random_walk(p, self.dims, self.task.config_space.check_index, len(self.task.config_space))
                # if self.check_index(p):
                #     new_points[i] = self.task.config_space.random_walk(p)
                
                # new_point = None
                # if (self.task.config_space.filtered_indexes) and p in self.task.config_space.filtered_indexes:
                #     p = self.task.config_space.filtered_indexes[p]
                #     # p = p + self.ice_offset[i] - self.ice_offset[i]
                # elif (self.task.config_space.filtered_indexes) and (p not in self.task.config_space.filtered_indexes):
                #     print("Found index which should be in filtered_indexes but it is not, ", p)
                # while new_point == None:
                #     new_point = random_walk(p, self.dims)
                #     if (self.task.config_space.a2f_indexes) and (new_point not in self.task.config_space.a2f_indexes):
                #         new_point = None
                #     elif self.task.config_space.a2f_indexes:
                #         new_point = self.task.config_space.a2f_indexes[new_point]
                # new_points[i] = new_point


            new_scores = model.predict(new_points)

            ac_prob = np.exp(np.minimum((new_scores - scores) / (t + 1e-5), 1))
            ac_index = np.random.random(len(ac_prob)) < ac_prob

            points[ac_index] = new_points[ac_index]
            scores[ac_index] = new_scores[ac_index]

            for s, p in zip(new_scores, new_points):
                if s > heap_items[0][0] and p not in in_heap:
                    pop = heapq.heapreplace(heap_items, (s, p))
                    in_heap.remove(pop[1])
                    in_heap.add(p)
                    k_last_modify = k

            k += 1
            t -= cool

            if log_interval and k % log_interval == 0:
                t_str = "%.2f" % t
                logger.debug(
                    "SA iter: %d\tlast_update: %d\tmax-0: %.2f\tmax-1: %.2f\ttemp: %s\t"
                    "elapsed: %.2f",
                    k,
                    k_last_modify,
                    heap_items[0][0],
                    np.max([v for v, _ in heap_items]),
                    t_str,
                    time.time() - tic,
                )

        heap_items.sort(key=lambda item: -item[0])
        heap_items = [x for x in heap_items if x[0] >= 0]
        logger.debug(
            "SA iter: %d\tlast_update: %d\telapsed: %.2f", k, k_last_modify, time.time() - tic
        )
        logger.debug("SA Maximums: %s", heap_items)

        if self.persistent:
            self.points = points

        return [x[1] for x in heap_items]


    def random_walk(point, dims, check_index, size):
        from tvm.autotvm.tuner.model_based_tuner import knob2point, point2knob
        """random walk as local transition

        Parameters
        ----------
        p: int
            index of the ConfigEntity
        dims: Array of int
            sizes of each dimension

        Returns
        -------
        new_p: int
            new neighborhood index
        """
        # transform to knob form
        knob = point2knob(point, dims)
        new_knob = knob.copy()
        unsuitable = set([point])
        new_point = None
        # mutate
        while new_point in unsuitable and len(unsuitable) < size:
            from_i = np.random.randint(len(knob))
            to_v = np.random.randint(dims[from_i])
            new_knob[from_i] = to_v

        # transform to index form
            new_point = knob2point(new_knob, dims)
            if not check_index(new_point):
                unsuitable.add(new_point)
            if not len(unsuitable) < size:
                logger.debug("random_walk did not find a new suitable point. The original will be returned")
                return point

        return new_point