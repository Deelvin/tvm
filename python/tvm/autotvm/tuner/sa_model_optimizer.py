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
        print("ICE SimulatedAnnealingOptimizer", task, flush=True)
        self.task = task
        self.dims = [len(x) for x in self.task.config_space.space_map.values()] # ICE TODO

        self.n_iter = n_iter
        self.temp = temp
        self.persistent = persistent
        self.parallel_size = min(parallel_size, self.task.config_space.checked_length)
        # self.parallel_size = min(parallel_size, len(self.task.config_space))
        self.early_stop = early_stop or 1e9
        self.log_interval = log_interval
        self.points = None

    def find_maximums(self, model, num, exclusive):
        # print("ICE find_maximums", num, "exclusive", exclusive, flush=True)
        tic = time.time()
        temp, n_iter, early_stop, log_interval = (
            self.temp,
            self.n_iter,
            self.early_stop,
            self.log_interval,
        )

        if self.persistent and self.points is not None:
            # print("ICE find_maximums self.points", flush=True)
            points = self.points
        else:
            # print("ICE find_maximums sample_ints", flush=True)
            points = np.array(sample_ints(0, len(self.task.config_space), self.parallel_size, self.task.config_space))

        # print("ICE find_maximums predict", flush=True)
        scores = model.predict(points) # ICE TODO

        # build heap and insert initial points
        heap_items = [(float("-inf"), -1 - i) for i in range(num)] # ICE TODO
        heapq.heapify(heap_items)
        in_heap = set(exclusive) # ICE TODO

        in_heap.update([x[1] for x in heap_items])

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
        while k < n_iter and k < k_last_modify + early_stop:
            # print("ICE find_maximums while", k, n_iter, k_last_modify + early_stop, flush=True)
            new_points = np.empty_like(points)
            for i, p in enumerate(points):
                # print(i)
                # print(p)
                # print(self.dims)
                # print(self.task.config_space.check_index)
                # print(self.task.config_space)
                # print(len(self.task.config_space))
                new_points[i] = SimulatedAnnealingOptimizer.random_walk(point=p, dims=self.dims, 
                    check_index=self.task.config_space.check_index, size=len(self.task.config_space))

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
        # print("ICE find_maximums end", flush=True)
        return [x[1] for x in heap_items]


    def random_walk(point, dims, check_index, size):
        # print("random_walk.point",point)
        # print("random_walk.dims",dims)
        # print("random_walk.check_index",check_index)
        # print("random_walk.size",size)
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
        new_point = point
        # mutate
        while new_point in unsuitable:
            # print("random_walk.new_point",new_point)
            # print("random_walk.unsuitable",unsuitable)
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
        # print("random_walk.new_point end",new_point)
        return new_point