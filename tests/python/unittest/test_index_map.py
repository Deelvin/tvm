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

import pytest

import tvm
from tvm.tir import IndexMap
from tvm.ir import assert_structural_equal


def assert_equal_index_map(map1: IndexMap, map2: IndexMap) -> None:

    iters_1 = map1.map_indices(map2.initial_indices)
    iters_2 = map2.final_indices
    assert len(iters_1) == len(iters_2)

    analyzer = tvm.arith.Analyzer()
    for iter1, iter2 in zip(iters_1, iters_2):
        assert analyzer.can_prove_equal(iter1, iter2)


def test_index_mapping():
    index_map = IndexMap.from_func(lambda i: [i // 4, i % 4])

    assert_structural_equal(index_map.map_indices([0]), [0, 0])
    assert_structural_equal(index_map.map_indices([3]), [0, 3])
    assert_structural_equal(index_map.map_indices([4]), [1, 0])
    assert_structural_equal(index_map.map_indices([42]), [10, 2])


def test_shape_mapping():
    index_map = IndexMap.from_func(lambda i: [i // 4, i % 4])

    assert_structural_equal(index_map.map_shape([4]), [1, 4])
    assert_structural_equal(index_map.map_shape([16]), [4, 4])

    assert_structural_equal(index_map.map_shape([14]), [4, 4])


def test_inverse():
    index_map = IndexMap.from_func(lambda i: [i // 4, i % 4])
    expected_inverse = IndexMap.from_func(lambda i, j: [4 * i + j])

    assert index_map.inverse([16]).is_equivalent_to(expected_inverse)


def test_nonbijective_inverse_gives_error():
    index_map = IndexMap.from_func(lambda i: [i // 4, i % 4])

    with pytest.raises(tvm.TVMError):
        index_map.inverse([14])


dynamic_N = tvm.tir.Var("N", "int32")
padding_test_case = tvm.testing.parameter(
    by_dict={
        "no_padding": dict(
            forward=lambda i: [i // 4, i % 4],
            inverse=lambda i, j: [4 * i + j],
            pre_shape=[16],
            post_shape=[4, 4],
            padding=lambda i, j: tvm.runtime.convert(False),
        ),
        "right_padding": dict(
            forward=lambda i: [i // 4, i % 4],
            inverse=lambda i, j: [4 * i + j],
            pre_shape=[15],
            post_shape=[4, 4],
            padding=lambda i, j: tvm.tir.And(i == 3, j >= 3),
        ),
        "left_padding": dict(
            forward=lambda i: [(i + 1) // 4, (i + 1) % 4],
            inverse=lambda i, j: [4 * i + j - 1],
            pre_shape=[15],
            post_shape=[4, 4],
            padding=lambda i, j: tvm.tir.And(i == 0, j < 1),
        ),
        "left_and_right_padding": dict(
            forward=lambda i: [(i + 1) // 4, (i + 1) % 4],
            inverse=lambda i, j: [4 * i + j - 1],
            pre_shape=[14],
            post_shape=[4, 4],
            padding=lambda i, j: tvm.tir.Or(
                tvm.tir.And(i == 0, j < 1),
                tvm.tir.And(i == 3, j >= 3),
            ),
        ),
        "dynamic_size": dict(
            forward=lambda i: [i // 4, i % 4],
            inverse=lambda i, j: [4 * i + j],
            pre_shape=[dynamic_N],
            post_shape=[(dynamic_N - 1) // 4 + 1, 4],
            padding=lambda i, j: tvm.tir.And(
                dynamic_N % (-4) != 0,
                tvm.tir.And(i == dynamic_N // 4, j >= dynamic_N % 4),
            ),
        ),
        "2d_padding": dict(
            forward=lambda i, j: [(i + 1) // 4, (j + 5) // 8, (i + 1) % 4, (j + 5) % 8],
            inverse=lambda i_outer, j_outer, i_inner, j_inner: [
                4 * i_outer + i_inner - 1,
                8 * j_outer + j_inner - 5,
            ],
            pre_shape=[14, 31],
            post_shape=[
                4,  # ceildiv(left_pad + i.extent, 4) = ceildiv(1 + 14, 4) = 4
                5,  # ceildiv(left_pad + j.extent, 8) = ceildiv(5 + 31, 8) = 5
                4,  # Range of iter%4
                8,  # Range of iter%8
            ],
            padding=lambda i_outer, j_outer, i_inner, j_inner: tvm.tir.Or(
                tvm.tir.Or(
                    tvm.tir.And(i_outer == 0, i_inner < 1),
                    tvm.tir.And(i_outer == 3, i_inner >= 3),
                ),
                tvm.tir.Or(
                    tvm.tir.And(j_outer == 0, j_inner < 5),
                    tvm.tir.And(j_outer == 4, j_inner >= 4),
                ),
            ),
        ),
        "multiple_right_padding": dict(
            forward=lambda i: [i // 32, (i // 4) % 8, i % 4],
            inverse=lambda i, j, k: [32 * i + 4 * j + k],
            pre_shape=[116],
            post_shape=[4, 8, 4],
            padding=lambda i, j, k: tvm.tir.And(i == 3, 4 * j + k >= 20),
        ),
        "multiple_right_padding_transpose": dict(
            forward=lambda i: [(i // 4) % 8, i // 32, i % 4],
            inverse=lambda j, i, k: [32 * i + 4 * j + k],
            pre_shape=[116],
            post_shape=[8, 4, 4],
            padding=lambda j, i, k: tvm.tir.And(i == 3, 4 * j + k >= 20),
        ),
        "multiple_left_padding": dict(
            forward=lambda i: [(i + 5) // 32, ((i + 5) // 4) % 8, (i + 5) % 4],
            inverse=lambda i, j, k: [32 * i + 4 * j + k - 5],
            pre_shape=[123],
            post_shape=[4, 8, 4],
            padding=lambda i, j, k: tvm.tir.And(i == 0, j * 4 + k < 5),
        ),
        "multiple_left_padding_with_transpose": dict(
            forward=lambda i: [((i + 5) // 4) % 8, (i + 5) // 32, (i + 5) % 4],
            inverse=lambda j, i, k: [32 * i + 4 * j + k - 5],
            pre_shape=[123],
            post_shape=[8, 4, 4],
            padding=lambda j, i, k: tvm.tir.And(i == 0, j * 4 + k < 5),
        ),
    }
)


def test_nonsurjective_inverse(padding_test_case):
    index_map = IndexMap.from_func(padding_test_case["forward"])

    inverse, padding_predicate = index_map.non_surjective_inverse(padding_test_case["pre_shape"])
    expected_inverse = IndexMap.from_func(padding_test_case["inverse"])
    assert inverse.is_equivalent_to(expected_inverse)

    post_shape = index_map.map_shape(padding_test_case["pre_shape"])
    tvm.ir.assert_structural_equal(post_shape, padding_test_case["post_shape"])

    expected_predicate = padding_test_case["padding"](*inverse.initial_indices)

    # Can't use analyzer.can_prove_equal, because it can't simplify
    # expressions like `(4*i+j >= 14) - (4*i+j >= 14)`.
    analyzer = tvm.arith.Analyzer()
    expected_predicate = analyzer.simplify(expected_predicate)
    padding_predicate = analyzer.simplify(padding_predicate)
    tvm.ir.assert_structural_equal(padding_predicate, expected_predicate)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
