/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file graph_executor_debug.cc
 */
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>
#include <tvm/ir/expr.h>

#include <numeric>

namespace tvm {
namespace runtime {

/*!
 * \brief Graph executor with dyn batch support.
 *
 *  This is the extension of GraphExecutor class to handle
 *  input tensors with arbitrary batch size
 */
class GraphExecutorDynBatch : public ModuleNode {
 public:
  const char* type_key() const final override { return "GraphExecutorDynBatch"; }

  /*!
   * \brief Struct to represent batch position in in/out tensors.
   * Literally that is a array<tuple<idx_of_tensor(int), batch_axis(int), is_output(bool)>>
   */
  using DynBatchConfig = tvm::runtime::Array<tvm::runtime::Array<ObjectRef>>;

  void Init(Module origin, DynBatchConfig config) {
    ICHECK(origin.defined());
    ICHECK_EQ(origin->type_key(), std::string("GraphExecutor"));

    TypedPackedFunc<NDArray(int)> get_input = origin.GetFunction("get_input");
    TypedPackedFunc<NDArray(int)> get_output = origin.GetFunction("get_output");
    TypedPackedFunc<int()> get_num_outputs = origin.GetFunction("get_num_outputs");
    TypedPackedFunc<int()> get_num_inputs = origin.GetFunction("get_num_inputs");

    auto num_inputs = get_num_inputs();
    auto num_outputs = get_num_outputs();

    // Convert
    std::vector<std::tuple<int, int, bool>> conf;
    for (const auto &l : config) {
      auto input_idx = l[0].as<IntImmNode>()->value; // int
      auto batch_dim = l[1].as<IntImmNode>()->value; // int
      auto is_output = l[2].as<IntImmNode>()->value; // bool
      conf.push_back({input_idx, batch_dim, is_output});
    }

    ShapeTuple::index_type origin_batch_size = -1;
    std::vector<int32_t> input_batch_axis(num_inputs, -1);
    std::vector<int32_t> output_batch_axis(num_outputs, -1);

    // Validate config
    for (const auto &p : conf) {
      int idx = std::get<0>(p);
      int axis = std::get<1>(p);
      bool is_output = std::get<2>(p);
      auto batch_size = is_output ? get_output(idx).Shape()[axis] : get_input(idx).Shape()[axis];

      if (origin_batch_size == -1) {
        origin_batch_size = batch_size;
      } else {
        ICHECK_EQ(batch_size, origin_batch_size);
      }
      if (is_output) {
        ICHECK_EQ(output_batch_axis[idx], -1);
        output_batch_axis[idx] = axis;
      } else {
        ICHECK_EQ(input_batch_axis[idx], -1);
        input_batch_axis[idx] = axis;
      }
    }

    origin_batch_size_ = origin_batch_size;
    cur_batch_size_ = origin_batch_size;

    origin_ = origin;
    set_input_zero_copy_ = origin.GetFunction("set_input_zero_copy");
    set_output_zero_copy_ = origin.GetFunction("set_output_zero_copy");
    get_input_index_ = origin.GetFunction("get_input_index");
    run_ = origin.GetFunction("run");

    num_inputs_ = num_inputs;
    num_outputs_ = num_outputs;

    input_batch_axis_ = input_batch_axis;
    output_batch_axis_ = output_batch_axis;

    // Initial values for cur tensors
    orig_inputs_.resize(num_inputs);
    for (int i = 0; i < num_inputs; i++)
      orig_inputs_[i] = get_input(i);

    orig_outputs_.resize(num_outputs);
    for (int i = 0; i < num_outputs; i++)
      orig_outputs_[i] = get_output(i);

    // By default it uses original tensors
    cur_inputs_ = orig_inputs_;
    cur_outputs_ = orig_outputs_;

    // Default state is batch coordinated
    is_dyn_batch_coordinated_ = true;
  }

  /*!
   * \brief GetFunction Get the function based on input.
   * \param name The function which needs to be invoked.
   * \param sptr_to_self Packed function pointer.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) override;

  /*!
   * \brief set index-th input to the graph.
   * \param index The input index.
   * \param data_in The input data.
   */
  void SetInput(int index, DLTensor* data_in) {
    ShapeTuple shape {data_in->shape, data_in->shape + data_in->ndim};
    if (shape == orig_inputs_[index].Shape()) {
      // store in origin input tensor plus
      orig_inputs_[index].CopyFrom(data_in);
      cur_inputs_[index] = orig_inputs_[index];
    } else {
      // Recreate a new one input tensor
      auto new_data_in = NDArray::Empty(shape, data_in->dtype, data_in->device);
      new_data_in.CopyFrom(data_in);
      cur_inputs_[index] = new_data_in;
    }
    RecalculateDynBatchState();
  }

  /*!
   * \brief set index-th input to the graph without copying the data
   * \param index The input index.
   * \param data_ref The input data that is referred.
   */
  void SetInputZeroCopy(int index, DLTensor* data_ref) {
    auto dl_data = DLManagedTensor{*data_ref};  // fake DLManagedTensor, no deleter
    auto data = NDArray::FromDLPack(&dl_data);
    cur_inputs_[index] = data;
    RecalculateDynBatchState();
  }

  /*!
   * \brief set index-th output to the graph without copying the data.
   * \param index The output index.
   * \param data_ref The output data that is referred.
   */
  void SetOutputZeroCopy(int index, DLTensor* data_ref) {
    // fake DLManagedTensor(no deleter). It doesn't hold original tensor
    auto dl_data = DLManagedTensor{*data_ref};
    auto data = NDArray::FromDLPack(&dl_data);
    cur_outputs_[index] = data;
    RecalculateDynBatchState();
  }

  /*!
   * \brief Return NDArray for given input index.
   * \param index The input index.
   *
   * \return NDArray corresponding to given input node index.
   */
  NDArray GetInput(int index) const { return cur_inputs_[index]; }

  /*!
   * \brief Return NDArray for given output index.
   * \param index The output index.
   *
   * \return NDArray corresponding to given output node index.
   */
  NDArray GetOutput(int index) const { return cur_outputs_[index]; }

  /*!
   * Process data by chunks
   */
  void Run() {
    if (!is_dyn_batch_coordinated_)
      LOG(FATAL) << "Input tensors shapes are not coordinated by batch dimension";

    if (!use_external_output_)
      ReallocateOutputs();

    std::vector<NDArray> inputs(num_inputs_);
    std::vector<NDArray> outputs(num_outputs_);

    ICHECK_EQ(cur_batch_size_ % origin_batch_size_, 0) << "Currently support only ";

    int num_of_slices = cur_batch_size_ / origin_batch_size_;
    for (int s = 0; s < num_of_slices; s++) {
      // Specify inputs
      for (size_t i = 0; i < cur_inputs_.size(); i++) {
        auto batch_axis = input_batch_axis_[i];

        if (batch_axis == -1) {
          inputs[i] = cur_inputs_[i];
        }
        if (batch_axis == 0) {
          inputs[i] =
              GetSliceView(cur_inputs_[i], batch_axis, origin_batch_size_ * s, origin_batch_size_);
        }
        if (batch_axis > 0 || !inputs[i].defined()) {
          CopySlice<true>(cur_inputs_[i], orig_inputs_[i], batch_axis, origin_batch_size_ * s,
                          origin_batch_size_);
          inputs[i] = orig_inputs_[i];
        }
      }
      // Specify outputs
      for (size_t i = 0; i < cur_outputs_.size(); i++) {
        auto batch_axis = output_batch_axis_[i];

        outputs[i] = cur_outputs_[i];
        if (batch_axis == 0) {
          outputs[i] =
              GetSliceView(cur_outputs_[i], batch_axis, origin_batch_size_ * s, origin_batch_size_);
        }
        if (batch_axis > 0 || !outputs[i].defined()) {
          outputs[i] = orig_outputs_[i];
        }
      }

      for (int i = 0; i < num_inputs_; i++) set_input_zero_copy_(i, inputs[i]);
      for (int i = 0; i < num_outputs_; i++) set_output_zero_copy_(i, outputs[i]);
      run_();

      // Copy output if it was not a view
      for (size_t i = 0; i < cur_outputs_.size(); i++) {
        if (outputs[i] == orig_outputs_[i] && outputs[i] != cur_outputs_[i])
          CopySlice<false>(orig_outputs_[i], cur_outputs_[i], output_batch_axis_[i],
                           origin_batch_size_ * s, origin_batch_size_);
      }
    }
  }

 private:
  static NDArray GetSliceView(NDArray arr, int axis, int offset, int size) {
    ICHECK(arr.IsContiguous());
    auto shape = arr.Shape();
    ICHECK_GT(shape.size(), axis);
    if (shape[axis] == size) return arr;

    ICHECK_EQ(axis, 0);
    std::vector<ShapeTuple::index_type> view_shape(shape.begin(), shape.end());
    view_shape[0] = size;
    auto batch_stride =
        std::accumulate(view_shape.begin() + 1, view_shape.end(), 1, std::multiplies<>());
    auto offset_byte = offset * batch_stride * arr.DataType().bytes();

    // There is a requirement on data alignment for all tensors.
    if (offset_byte % kAllocAlignment != 0) return {};

    auto slice_view = arr.CreateView(view_shape, arr->dtype).ToDLPack();
    // TODO: Looks like kernels doesn't support offsets.. have to move data ptr
//    slice_view->dl_tensor.byte_offset = offset * arr.DataType().bytes();
    slice_view->dl_tensor.data = static_cast<uint8_t*>(slice_view->dl_tensor.data) + offset_byte;

    return NDArray::FromDLPack(slice_view);
  }

  /*!
   * Copy of none contiguous slice representation.
   *
   * Naive implementation:
   * Reshape to [outer_dim, batch, inner_dim]
   * for (od : range(inner_dim))
   *   for (b : range(offset, size))  // |
   *     for (id : range(inner_dim))  // | Copy of contiguous memory
   *        *dst++ = src[od, b, id]   // |
   */
  template <bool SLICE_SRC>
  static void CopySlice(NDArray &src, NDArray &dst, int axis, int offset, int size) {
    auto shape = SLICE_SRC ? src.Shape() : dst.Shape();
    ICHECK_GT(shape.size(), axis);
    ICHECK(src.IsContiguous());

    auto outer_dim = std::accumulate(shape.begin(), shape.begin() + axis, 1, std::multiplies<>());
    auto slice_dim = shape[axis];
    auto inner_dim = std::accumulate(shape.begin() + axis + 1, shape.end(), 1,
                                     std::multiplies<>());

    auto chunk_src_arr = src.CreateView({size, inner_dim}, src->dtype);
    auto chunk_dst_arr = dst.CreateView({size, inner_dim}, dst->dtype);

    auto chunk_src_dl = *chunk_src_arr.operator->();
    auto chunk_dst_dl = *chunk_dst_arr.operator->();

    uint64_t chunk_src_stride_byte = SLICE_SRC ? inner_dim * slice_dim : inner_dim * size;
    uint64_t chunk_src_offset_byte = SLICE_SRC ? inner_dim * offset : 0;
    uint64_t chunk_dst_stride_byte = !SLICE_SRC ? inner_dim * slice_dim : inner_dim * size;
    uint64_t chunk_dst_offset_byte = !SLICE_SRC ? inner_dim * offset : 0;

    chunk_src_dl.byte_offset += chunk_src_offset_byte * src.DataType().bytes();
    chunk_dst_dl.byte_offset += chunk_dst_offset_byte * dst.DataType().bytes();

    for (int o = 0; o < outer_dim; o++) {
      NDArray::CopyFromTo(&chunk_src_dl, &chunk_dst_dl);
      chunk_src_dl.byte_offset += chunk_src_stride_byte * src.DataType().bytes();
      chunk_dst_dl.byte_offset += chunk_dst_stride_byte * dst.DataType().bytes();
    }
  }

  void RecalculateDynBatchState() {
    is_dyn_batch_coordinated_ = true;
    cur_batch_size_ = -1;
    for (int i = 0; i < num_inputs_; i++) {
      auto batch_dims = input_batch_axis_[i];
      if (batch_dims == -1) continue;  // no batch dim in that input

      auto batch_size = cur_inputs_[i].Shape()[batch_dims];
      if (cur_batch_size_ == -1) {
        cur_batch_size_ = batch_size;
      } else {
        is_dyn_batch_coordinated_ &= cur_batch_size_ == batch_size;
      }
    }
  }

  void ReallocateOutputs() {
    ICHECK(is_dyn_batch_coordinated_);
    ICHECK(!use_external_output_);
    for (int i = 0; i < num_outputs_; i++) {
      auto batch_axis = output_batch_axis_[i];
      if (batch_axis == -1) continue;

      auto cur_shape = cur_outputs_[i].Shape();
      if (cur_shape[batch_axis] != cur_batch_size_) {
        auto orig_data = orig_outputs_[i];

        // reallocate with cur batch size
        std::vector<ShapeTuple::index_type> new_cur_shape {cur_shape.begin(), cur_shape.end()};
        new_cur_shape[batch_axis] = cur_batch_size_;

        cur_outputs_[i] = NDArray::Empty(ShapeTuple(new_cur_shape), orig_data->dtype,
                                         orig_data->device);
      }
    }
  }

  Module origin_;  // just a holder
  TypedPackedFunc<void(int, NDArray)> set_input_zero_copy_;
  TypedPackedFunc<void(int, NDArray)> set_output_zero_copy_;
  TypedPackedFunc<int(std::string)> get_input_index_;
  PackedFunc run_;

  std::vector<int32_t> input_batch_axis_;
  std::vector<int32_t> output_batch_axis_;

  bool is_dyn_batch_coordinated_;
  bool use_external_output_;
  ShapeTuple::index_type origin_batch_size_;
  ShapeTuple::index_type cur_batch_size_;

  int num_inputs_;
  int num_outputs_;

  std::vector<NDArray> orig_inputs_;
  std::vector<NDArray> orig_outputs_;
  std::vector<NDArray> cur_inputs_;
  std::vector<NDArray> cur_outputs_;
};

/*!
 * \brief GetFunction Get the function based on input.
 * \param name The function which needs to be invoked.
 * \param sptr_to_self Packed function pointer.
 */
PackedFunc GraphExecutorDynBatch::GetFunction(const std::string& name,
                                           const ObjectPtr<Object>& sptr_to_self) {
  // return member functions during query.
  if (name == "get_output") {
    // TODO: it should support 2 args mode
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 1);
      if (String::CanConvertFrom(args[0])) {
        *rv = this->GetOutput(get_input_index_(args[0]));
      } else {
        *rv = this->GetOutput(args[0]);
      }
    });
  } else if (name == "get_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 1);
      if (String::CanConvertFrom(args[0])) {
        std::string name = args[0].operator String();
        *rv = this->GetInput(get_input_index_(name));
      } else {
        *rv = this->GetInput(args[0]);
      }
    });
  } else if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 2);
      if (String::CanConvertFrom(args[0])) {
        auto idx = get_input_index_(args[0]);
        this->SetInput(idx, args[1]);
      } else {
        this->SetInput(args[0], args[1]);
      }
    });
  } else if (name == "set_input_zero_copy") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 2);
      if (String::CanConvertFrom(args[0])) {
        this->SetInputZeroCopy(get_input_index_(args[0]), args[1]);
      } else {
        this->SetInputZeroCopy(args[0], args[1]);
      }
    });
  } else if (name == "set_output_zero_copy") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0])) {
        this->SetOutputZeroCopy(get_input_index_(args[0]), args[1]);
      } else {
        this->SetOutputZeroCopy(args[0], args[1]);
      }
    });
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 0);
      this->Run();
    });
  } else {
    return origin_->GetFunction(name, sptr_to_self);
  }
}

/*!
 * \brief GraphExecutorDebugCreate Get the function based on input.
 * \param sym_json The graph symbol in json format.
 * \param m Compiled module which will be loaded.
 * \param devs All devices.
 */
Module GraphExecutorDynBatchCreate(const tvm::runtime::Module& m,
                                   GraphExecutorDynBatch::DynBatchConfig config) {
  auto exec = make_object<GraphExecutorDynBatch>();
  exec->Init(m, config);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.graph_executor_dyn_batch.create")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
  ICHECK_GE(args.num_args, 2) << "The expected number of arguments for "
                                 "graph_executor_dyn_batch.create is 2, but it has "
                                  << args.num_args;

  *rv = GraphExecutorDynBatchCreate(args[0], args[1]);
});
}  // namespace runtime
}  // namespace tvm
