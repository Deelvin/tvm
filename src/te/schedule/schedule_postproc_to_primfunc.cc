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
 * \file schedule_postproc_to_primfunc.cc
 *
 * \brief Translate the function body generated by ScheduleOps
 *  with te related dialects that incorporates Tensor
 *  into the Stmts to a PrimFunc.
 *
 *  Perform this translation before running any TIR optimizations.
 *
 *  Rationale: The body generated by ScheduleOps is not
 *  a formal PrimFunc and cannot be used for further optimization.
 *  This function canonicalize that body and creates a formal PrimFunc.
 *
 *  List of actions taken by the function:
 *  - Remove occurences of te::Tensor, te::Operation in the IR
 *    and replace them by corresponding IR nodes via tir::Buffer.
 *  - Add annotation of extern buffers using the buffer_map field
 *    in the PrimFunc type.
 */
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include <functional>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace te {

// create a buffer for tensor.
Buffer CreateBufferFor(const Tensor& tensor, String storage_scope = "") {
  std::string name = tensor->op->name;
  if (tensor->op->num_outputs() != 1) {
    name += ".v" + std::to_string(tensor->value_index);
  }
  Buffer buffer = decl_buffer(tensor->shape, tensor->dtype, name, storage_scope);

  return buffer;
}

// A remapper that maps tensor to buffer
class TensorToBufferMapper : public StmtExprMutator {
 public:
  explicit TensorToBufferMapper(std::unordered_map<Tensor, Buffer> buffer_map)
      : buffer_map_(buffer_map) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    auto ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();
    if (op->attr_key == tir::attr::double_buffer_scope ||
        op->attr_key == tir::attr::rolling_buffer_scope) {
      Stmt body = op->body;
      Operation operation = Downcast<Operation>(op->node);
      for (int i = operation->num_outputs(); i != 0; --i) {
        Buffer buffer = GetOrAllocBuffer(operation.output(i - 1));
        body = AttrStmt(buffer, op->attr_key, op->value, body);
      }
      return body;
    } else if (op->attr_key == tir::attr::buffer_bind_scope) {
      Array<ObjectRef> tuple = Downcast<Array<ObjectRef>>(op->node);
      Tensor tensor = Downcast<Tensor>(tuple[1]);
      return AttrStmt(Array<ObjectRef>{tuple[0], GetOrAllocBuffer(tensor)}, op->attr_key, op->value,
                      op->body);
    } else if (op->attr_key == tir::attr::buffer_dim_align ||
               op->attr_key == tir::attr::prefetch_scope) {
      Tensor tensor = Downcast<Tensor>(op->node);
      Buffer buffer = GetOrAllocBuffer(tensor);
      return AttrStmt(buffer, op->attr_key, op->value, op->body);
    } else if (op->attr_key == tir::attr::layout_transforms ||
               op->attr_key == tir::attr::axis_separators) {
      auto arr = Downcast<Array<ObjectRef>>(op->node);
      ICHECK_EQ(arr.size(), 2);

      Stmt body = op->body;

      Tensor tensor = Downcast<Tensor>(arr[0]);
      Buffer buffer = GetBuffer(tensor);

      return AttrStmt(Array<ObjectRef>{buffer, arr[1]}, op->attr_key, 1, body);
    } else {
      return ret;
    }
  }

  Stmt VisitStmt_(const ProducerRealizeNode* op) final {
    Tensor tensor = Downcast<Tensor>(op->producer);
    Buffer buffer = GetOrAllocBuffer(tensor, op->storage_scope);

    auto ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<ProducerRealizeNode>();

    return BufferRealize(buffer, op->bounds, op->condition, op->body);
  }

  Stmt VisitStmt_(const ProducerStoreNode* op) final {
    Tensor tensor = Downcast<Tensor>(op->producer);
    Buffer buffer = GetBuffer(tensor);

    auto ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<ProducerStoreNode>();

    return BufferStore(buffer, op->value, GetIndices(op->indices, buffer->shape));
  }

  PrimExpr VisitExpr_(const ProducerLoadNode* op) final {
    auto ret = StmtExprMutator::VisitExpr_(op);
    op = ret.as<ProducerLoadNode>();
    Tensor tensor = Downcast<Tensor>(op->producer);
    Buffer buffer = GetBuffer(tensor);
    return tir::BufferLoad(buffer, GetIndices(op->indices, buffer->shape));
  }

 private:
  Buffer GetOrAllocBuffer(const Tensor& tensor, String storage_scope = "") {
    return GetBuffer(tensor, storage_scope, true);
  }

  Buffer GetBuffer(const Tensor& tensor, String storage_scope = "", bool allow_alloc = false) {
    auto it = buffer_map_.find(tensor);
    if (it != buffer_map_.end()) return it->second;
    ICHECK(allow_alloc) << "Cannot find the Realization point of tensor " << tensor;

    auto buffer = CreateBufferFor(tensor, storage_scope);
    buffer_map_[tensor] = buffer;
    return buffer;
  }

  Array<PrimExpr> GetIndices(const Array<PrimExpr>& tensor_indices,
                             const Array<PrimExpr>& buffer_shape) {
    if (tensor_indices.size() == buffer_shape.size()) {
      return tensor_indices;
    } else if (tensor_indices.size() == 1) {
      // Workaround to support previous behavior of tensor indexing by
      // a single index, treating the tensor as if were already
      // flattened by a row-major traversal.
      PrimExpr unravel = tensor_indices[0];
      Array<PrimExpr> rev_indices;
      for (size_t i = buffer_shape.size(); i > 0; i--) {
        PrimExpr dim = buffer_shape[i - 1];
        rev_indices.push_back(indexmod(unravel, dim));
        unravel = indexdiv(unravel, dim);
      }
      return Array<PrimExpr>(rev_indices.rbegin(), rev_indices.rend());
    } else {
      LOG(FATAL) << "Cannot produce indices for " << buffer_shape.size()
                 << "-dimensional TIR buffer using " << tensor_indices.size()
                 << "-dimensional tensor indices.";
      return {};
    }
  }

  // Maps tensor to buffer.
  std::unordered_map<Tensor, Buffer> buffer_map_;
};

/*! Collect the physical layout map of all tensors in the statement. */
class LayoutTransformAttrUnwrapper : StmtExprMutator {
 public:
  static tir::PrimFunc Apply(tir::PrimFunc func) {
    // Collect the physical layout annotations in the body, which may
    // refer to input arguments.
    auto layout_map = Collector::Collect(func->body);

    if (layout_map.size()) {
      func = WithAttr(std::move(func), "layout_transform_map", layout_map);

      auto write_ptr = func.CopyOnWrite();
      write_ptr->body = LayoutTransformAttrUnwrapper()(func->body);
    }

    return func;
  }

  LayoutTransformAttrUnwrapper() {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    auto ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();

    if (op->attr_key == tir::attr::layout_transforms) {
      return op->body;
    } else {
      return ret;
    }
  }

 private:
  /*! Collect the physical layout information of all tensors in the statement.
   *
   * Must be done before constructing the buffers, since the
   * attributes could either apply to the external buffers or to
   * internal allocations.
   */
  class Collector : StmtExprVisitor {
   public:
    static Map<Buffer, Array<IndexMap>> Collect(Stmt stmt) {
      Collector collector;
      collector(std::move(stmt));
      return std::move(collector.layout_map_);
    }

    Collector() {}

    void VisitStmt_(const AttrStmtNode* op) final {
      if (op->attr_key == tir::attr::layout_transforms) {
        auto arr = Downcast<Array<ObjectRef>>(op->node);
        ICHECK_EQ(arr.size(), 2);

        auto buffer = Downcast<Buffer>(arr[0]);
        auto layout_transforms = Downcast<Array<IndexMap>>(arr[1]);
        layout_map_.Set(buffer, layout_transforms);
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    Map<Buffer, Array<IndexMap>> layout_map_;
  };

  std::unordered_map<const BufferNode*, Buffer> buffer_remap_;

  Map<Buffer, Array<IndexMap>> layout_map_;
};

/*! Move axis_separators from an attribute to a buffer property. */
class AxisSeparatorsAttrUnwrapper : StmtExprMutator {
 public:
  static tir::PrimFunc Apply(tir::PrimFunc func) {
    // Collect the physical layout annotations in the body, which may
    // refer to input arguments.
    auto axis_separators_map = Collector::Collect(func->body);

    if (axis_separators_map.size()) {
      auto write_ptr = func.CopyOnWrite();
      auto pass = AxisSeparatorsAttrUnwrapper(axis_separators_map);
      write_ptr->buffer_map = pass.UpdateExternBufferMap(func->buffer_map);
      write_ptr->body = pass(func->body);
      if (auto map = func->attrs.GetAttr<Map<Buffer, Array<IndexMap>>>("layout_transform_map")) {
        func = WithAttr(std::move(func), "layout_transform_map", pass.UpdateIndexMap(map.value()));
      }
    }

    return func;
  }

  explicit AxisSeparatorsAttrUnwrapper(Map<Buffer, Array<IntImm>> axis_separators_map)
      : axis_separators_map_(axis_separators_map) {}

  Map<Var, Buffer> UpdateExternBufferMap(const Map<Var, Buffer>& orig) {
    Map<Var, Buffer> output;
    for (const auto& kv : orig) {
      output.Set(kv.first, GetRemappedBuffer(kv.second));
    }
    return output;
  }

  Map<Buffer, Array<IndexMap>> UpdateIndexMap(const Map<Buffer, Array<IndexMap>>& orig) {
    Map<Buffer, Array<IndexMap>> output;
    for (const auto& kv : orig) {
      output.Set(GetRemappedBuffer(kv.first), kv.second);
    }
    return output;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    auto ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();

    if (op->attr_key == tir::attr::axis_separators) {
      return op->body;
    } else if (op->attr_key == tir::attr::buffer_bind_scope) {
      Array<ObjectRef> tuple = Downcast<Array<ObjectRef>>(op->node);
      Buffer view_buffer = Downcast<Buffer>(tuple[0]);
      Buffer source_buffer = Downcast<Buffer>(tuple[1]);
      return AttrStmt(
          Array<ObjectRef>{GetRemappedBuffer(view_buffer), GetRemappedBuffer(source_buffer)},
          op->attr_key, op->value, op->body);
    } else {
      return ret;
    }
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    auto node = Downcast<BufferRealize>(StmtExprMutator::VisitStmt_(op));
    return VisitBufferAccess(std::move(node));
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    return VisitBufferAccess(std::move(node));
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return VisitBufferAccess(std::move(node));
  }

 private:
  template <typename Node>
  Node VisitBufferAccess(Node node) {
    Buffer new_buf = GetRemappedBuffer(node->buffer);
    if (!node->buffer.same_as(new_buf)) {
      auto writer = node.CopyOnWrite();
      writer->buffer = new_buf;
    }
    return node;
  }

  Buffer GetRemappedBuffer(Buffer buf) {
    // If this buffer has already been remapped, then return the
    // previous value.
    auto key = buf.get();
    {
      auto it = buffer_remap_.find(key);
      if (it != buffer_remap_.end()) {
        return it->second;
      }
    }

    // Otherwise, check if we need to add axis_separators to this
    // buffer.
    auto lookup = axis_separators_map_.Get(buf);
    if (lookup) {
      Array<IntImm> axis_separators = lookup.value();
      if (axis_separators.size()) {
        auto write_ptr = buf.CopyOnWrite();
        write_ptr->axis_separators = axis_separators;
      }
    }

    // And cache the result for next time.
    buffer_remap_[key] = buf;

    return buf;
  }

  /*! Collect the axis separator information of all tensors in the statement.
   *
   * Must be done before constructing the buffers, since the
   * attributes could either apply to the external buffers or to
   * internal allocations.
   */
  class Collector : StmtExprVisitor {
   public:
    static Map<Buffer, Array<IntImm>> Collect(Stmt stmt) {
      Collector collector;
      collector(std::move(stmt));
      return std::move(collector.axis_separators_map_);
    }

    Collector() {}

    void VisitStmt_(const AttrStmtNode* op) final {
      if (op->attr_key == tir::attr::axis_separators) {
        auto arr = Downcast<Array<ObjectRef>>(op->node);
        ICHECK_EQ(arr.size(), 2);

        auto buffer = Downcast<Buffer>(arr[0]);
        auto axis_separators = Downcast<Array<IntImm>>(arr[1]);
        axis_separators_map_.Set(buffer, axis_separators);
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    Map<Buffer, Array<IntImm>> axis_separators_map_;
  };

  std::unordered_map<const BufferNode*, Buffer> buffer_remap_;

  Map<Buffer, Array<IntImm>> axis_separators_map_;
};

// Helper to assign extenal const buffers
class AStmtNodeVisitor final : public StmtExprVisitor {
  public:
    AStmtNodeVisitor(Map<tir::Var, tir::Buffer>& buffer_map,
                     Array<tir::Var>& params)
    : buffer_map_(&buffer_map)
    , params_(&params)
    {}

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->node->IsInstance<ArrayNode>() &&
        op->attr_key == tir::attr::buffer_bind_scope &&
        buffer_map_ != nullptr &&
        params_ != nullptr) {
      Array<ObjectRef> arr = Downcast<Array<ObjectRef>>(op->node);
      ICHECK_EQ(arr.size(), 2U);
      tir::Buffer buffer = Downcast<Buffer>(arr[1]);
      bool found = false;
      for (auto i : (*buffer_map_)) {
        if (i.second == buffer) {
          found = true;
          break;
        }
      }
      if (!found) {
        tir::Var bptr(buffer->name, PrimType(DataType::Handle()));
        params_->push_back(bptr);
        buffer_map_->Set(bptr, buffer);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
private:
  Map<tir::Var, tir::Buffer>* buffer_map_ = nullptr;
  Array<tir::Var>* params_ = nullptr;
};

PrimFunc SchedulePostProcToPrimFunc(Array<ObjectRef> arg_list, Stmt body,
                                    Optional<Map<Tensor, Buffer>> extern_buffer_opt) {
  std::unordered_map<Tensor, Buffer> extern_tensor_map;

  if (extern_buffer_opt.defined()) {
    auto v = extern_buffer_opt.value();
    extern_tensor_map = std::unordered_map<Tensor, Buffer>(v.begin(), v.end());
  }

  Array<tir::Var> params;
  Map<tir::Var, tir::Buffer> buffer_map;

  for (auto arg : arg_list) {
    if (auto* n = arg.as<tir::VarNode>()) {
      tir::Var var = GetRef<tir::Var>(n);
      params.push_back(GetRef<tir::Var>(n));
    } else if (auto* n = arg.as<te::TensorNode>()) {
      te::Tensor tensor = GetRef<te::Tensor>(n);
      ICHECK(!extern_tensor_map.count(tensor));

      tir::Buffer buffer = CreateBufferFor(tensor);
      tir::Var bptr(buffer->name, PrimType(DataType::Handle()));
      params.push_back(bptr);
      buffer_map.Set(bptr, buffer);
      extern_tensor_map[tensor] = buffer;
    } else if (auto* n = arg.as<tir::BufferNode>()) {
      tir::Buffer buffer = GetRef<tir::Buffer>(n);
      tir::Var bptr(buffer->name, PrimType(DataType::Handle()));
      params.push_back(bptr);
      buffer_map.Set(bptr, buffer);
    } else {
      LOG(FATAL) << "Expected argument to be Var, Tensor, or Buffer, but received "
                 << arg->GetTypeKey();
    }
  }

  body = TensorToBufferMapper(std::move(extern_tensor_map))(std::move(body));

  // workaround which allows to assign intermediate buffers as inputs
  AStmtNodeVisitor visitor(buffer_map, params);
  visitor(body);
  PrimFunc func = tir::PrimFunc(params, body, VoidType(), buffer_map);

  func = LayoutTransformAttrUnwrapper::Apply(std::move(func));
  func = AxisSeparatorsAttrUnwrapper::Apply(std::move(func));

  // We mark this PrimFunc as coming from a TE schedule
  func = WithAttr(func, "from_legacy_te_schedule", Bool(true));

  return func;
}

TVM_REGISTER_GLOBAL("schedule.SchedulePostProcToPrimFunc")
    .set_body_typed(SchedulePostProcToPrimFunc);

}  // namespace te
}  // namespace tvm
