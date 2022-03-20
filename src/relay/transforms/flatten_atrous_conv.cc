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
 * \file src/relay/transforms/quantize_fake_quantization.cc
 * \brief A pass for taking fake quantized graphs and converting them
 * to actual integer operations.
 */

// #include "fake_quantization_to_integer.h"

#include <tvm/ir/affine_type.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/transform.h>

#include <tvm/topi/broadcast.h>

#include <unordered_map>

#include "../qnn/utils.h"

namespace tvm {
namespace relay {

/* Description of FakeQuantizationToInteger
 *
 * The purpose of this pass is to find regions of the graph that follow
 * the general pattern:
 *
 *   x    w
 *   |    |
 *   dq   dq
 *    \   /
 *     op1
 *      |
 *     op2
 *      |
 *      q
 *
 * and convert them into subgraphs with actual integer operations on x and w
 *
 * The pass does this via a multi-pass approach:
 *
 * The main pass is a MixedModeMutator that traverses the full graph searching for
 * quantize operations
 *
 * The second pass is an ExprVisitor that recursively searches for subgraphs leading to the
 * quantize for subtraphs bounded by dequantize operations. This pass extracts the affine
 * types of the inputs for later processing, where affine denotes the transformation
 * x_real = (x_affine - zero_point) * scale
 *
 * The third pass is an ExprMutator that recursively rewrites the subgraphs using packed funcs
 * registered with the FTVMFakeQuantizationToInteger attribute. These packed funcs rewrite
 * the ops based on the affine types of their inputs and then return the affine types of the
 * new rewriten ops to pass that information down the stack during rewrite.
 *
 * After the second and third passes run, the first pass replaces the quantize with the
 * rewritten subgraph and the processing continues
 *
 *
 * After that an additional QAT pass can be enabled by use_qat flag. The goal of the pass is to find
 * operations in those regions(which were not successfully converted by the main pass) that can
 * still be converted into quantized form. The idea is to find and transform operations with
 * dequantized inputs one by one individually. Only operations for which all parameters can be
 * explicitly calculated are allowed. For example, if on the above general  pattern op2 is not
 * registered with the FTVMFakeQuantizationToInteger attribute, op1 operation can still be
 * converted. Converted pattern below:
 *
 *   x    w
 *   |    |
 *    \   /
 *     op1
 *      |
 *     dq
 *      |
 *     op2
 *      |
 *      q
 *
 * This pass works in the same multi-pass approach.
 */

using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;
using ExprMap = std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual>;
using AffineTypeMap = Map<Expr, AffineType>;

using FTVMFakeQuantizationToInteger =
    runtime::TypedPackedFunc<Array<ObjectRef>(const Expr& expr, const AffineTypeMap& map)>;


class FlattenAtrousConvSubgraphExtractor : public ExprVisitor {
 public:

bool has_conv2d = false;
bool has_space_to_batch_nd = false;
bool has_batch_to_space_nd = false;

  Op batch_to_space_nd = Op::Get("nn.batch_to_space_nd");
  Op conv2d = Op::Get("nn.conv2d");
  Op space_to_batch_nd = Op::Get("nn.space_to_batch_nd");

  const ExprSet GetSubgraph(const Expr& expr) {
    expr_call_node_ = expr.as<CallNode>();
    ICHECK(expr_call_node_ != nullptr);
    //ICHECK(is_op_enabled_for_optional_fq2i(expr_call_node_));
    ICHECK(expr_call_node_->op == batch_to_space_nd);
    


    VisitExpr(expr);

    ExprSet subgraph;
    //if (is_fake_quantized_) {
    // std::cout << "subgraph: " << std::endl;
    if (has_conv2d && has_space_to_batch_nd && has_batch_to_space_nd) {
      for (auto kv : this->visit_counter_) {
        if (auto call_node = GetRef<ObjectRef>(kv.first).as<CallNode>()) {
          //// std::cout << "call_node: " << call_node->op << " " << kv.second << std::endl;
          //if (call_node != expr_call_node_) {
            subgraph.insert(Downcast<Expr>(GetRef<ObjectRef>(kv.first)));
          //}
        }
      }
    }
    // std::cout << "//subgraph" << std::endl;
    return subgraph;
  }
  const AffineTypeMap GetAffineTypes() { return affine_types_; }
  void VisitExpr(const Expr& expr) override {
    // When looking for fake quantized subgraphs, we only support data-flow regions of the graph,
    // i.e. call nodes/tuples/constants/etc. If we see anything else (like control flow) we
    // abort the rewrite.
    if (expr.as<CallNode>() == nullptr && expr.as<OpNode>() == nullptr &&
        expr.as<TupleNode>() == nullptr && expr.as<TupleGetItemNode>() == nullptr &&
        expr.as<ConstantNode>() == nullptr) {
      DLOG(INFO) << "FakeQuantizationToInteger found a non - dataflow op inside a fake quantize "
                    "region, aborting this rewrite";
      is_fake_quantized_ = false;
    } else {
      ExprVisitor::VisitExpr(expr);
    }
  }

 protected:
  void VisitExpr_(const CallNode* call_node) override {
    //if (call_node->op == dequantize_op_) {
    //  const auto* attrs = call_node->attrs.as<qnn::DequantizeAttrs>();
    //  ICHECK(attrs != nullptr);

    //  affine_types_.Set(
    //      GetRef<Expr>(call_node),
    //      TensorAffineType(
    //          call_node->args[1], call_node->args[2],
    //          tvm::relay::transform::InferTypeLocal(call_node->args[0]).as<TensorTypeNode>()->dtype,
    //          attrs->axis));
    //} else 

      if (has_batch_to_space_nd == false) {
        ICHECK(has_conv2d == false);
        ICHECK(has_space_to_batch_nd == false);
        ICHECK(call_node->op == batch_to_space_nd);
        has_batch_to_space_nd = true;
         for (auto arg : call_node->args) {
          VisitExpr(arg);
        }
      } else if (has_conv2d == false) {
        ICHECK(has_batch_to_space_nd == true);
        ICHECK(has_space_to_batch_nd == false);
        ICHECK(call_node->op == conv2d);
        has_conv2d = true;
         for (auto arg : call_node->args) {
          VisitExpr(arg);
        }
      } else if (has_space_to_batch_nd == false) {
        ICHECK(has_batch_to_space_nd == true);
        ICHECK(has_conv2d == true);
        ICHECK(call_node->op == space_to_batch_nd);
        has_space_to_batch_nd = true;
      } 
      // error

      //if (call_node == expr_call_node_) {
      //for (auto arg : call_node->args) {
      //  VisitExpr(arg);
      //}
    //} else {
      // run normally on everything else.
      //ExprVisitor::VisitExpr_(call_node);
    //}
  }

  const Op dequantize_op_ = Op::Get("qnn.dequantize");
  bool is_fake_quantized_ = true;
  AffineTypeMap affine_types_;
  const CallNode* expr_call_node_ = nullptr;
};

class FlattenAtrousConvSubgraphMutator : public ExprMutator {
 public:
  FlattenAtrousConvSubgraphMutator(ExprSet subgraph, bool tf)
      : subgraph_(subgraph), _tf(tf){}

  Expr MutateSubgraph(const Expr& expr) {
    if (subgraph_.size() == 0) {
      return expr;
    }

    quantize_node_ = expr.as<CallNode>();
    ICHECK(quantize_node_);
    //ICHECK(is_op_enabled_for_optional_fq2i(quantize_node_));

    for (auto node : subgraph_) {
      const Op op = Downcast<Op>(node.as<CallNode>()->op);

      //if (node.as<CallNode>()->op != dequantize_op_) {
      //    DLOG(INFO) << "Not dequantization was found in the input arguments for "
      //               << AsText(op, false) << std::endl;
      //    return expr;
      //}
    }
    try {
      return Mutate(expr);
    } catch (std::exception& e) {
        DLOG(INFO) << "Ran into an error rewriting a subgraph, skipping" << expr << std::endl;
      
        // std::cout << "Ran into an error(" << e.what()  << ") rewriting a subgraph, skipping" << expr << std::endl;
        return expr;
    }
  }
  const CallNode* conv2d_node = nullptr;
  const CallNode* b2s_node = nullptr;
  const CallNode* s2b_node = nullptr;
  bool _tf = true;
 protected:
  //%79 = nn.space_to_batch_nd(%78,
  //%80 = nn.conv2d(%79, meta[relay.Constant][51] 
  //%81 = nn.batch_to_space_nd(%80,
  //=>>>
  //nn.conv2d(%78, meta[relay.Constant][51] 
  Expr VisitExpr_(const CallNode* call_node) {
    if (_tf) 
      return VisitExpr_tf(call_node);
    else 
      return VisitExpr_mace(call_node);
  }

  Expr VisitExpr_tf(const CallNode* call_node) {
    // std::cout << "VisitExpr_: " << call_node->op << std::endl;
    // Expr out;

    Op op = Downcast<Op>(call_node->op);

    Expr expr;
      if (op == batch_to_space_nd) {
        b2s_node = call_node;
        expr = ExprMutator::VisitExpr_(b2s_node); // 1
        expr = expr.as<CallNode>()->args[0];
        // // std::cout << "after batch_to_space_nd expr# " << expr << std::endl;
        // // std::cout << "VisitExpr_: " << call_node->op << "\n => " << expr << std::endl;
        return expr;
      } else if (op == conv2d) {
        conv2d_node = call_node;
        expr =  ExprMutator::VisitExpr_(conv2d_node); // 2
        expr = expr.as<CallNode>()->args[0];
        // // std::cout << "after conv2d expr# " << expr << std::endl;
        // // std::cout << "VisitExpr_: " << call_node->op << "\n => " << expr << std::endl;
        return expr;
      } else if (op == space_to_batch_nd) {  // 3
        s2b_node = call_node;

        ICHECK(b2s_node != nullptr);
        const auto* b2s_attrs = b2s_node->attrs.as<BatchToSpaceNDAttrs>();
        ICHECK(b2s_attrs != nullptr);
        b2s_attrs->block_shape;
        b2s_attrs->crops;

        ICHECK(conv2d_node != nullptr);
        const auto* conv2d_attrs = conv2d_node->attrs.as<Conv2DAttrs>();
        ICHECK(conv2d_attrs != nullptr);
        conv2d_attrs->strides;
        conv2d_attrs->padding;
        conv2d_attrs->dilation;
        conv2d_attrs->groups;
        conv2d_attrs->channels;
        conv2d_attrs->kernel_size;
        conv2d_attrs->data_layout;
        conv2d_attrs->kernel_layout;
        conv2d_attrs->out_layout;
        conv2d_attrs->out_dtype;

        ICHECK(s2b_node != nullptr);
        const auto* s2b_attrs = s2b_node->attrs.as<SpaceToBatchNDAttrs>();
        ICHECK(s2b_attrs != nullptr);
        s2b_attrs->block_shape;
        s2b_attrs->paddings;
        s2b_attrs->pad_value;

        /*
        {"BatchToSpaceND", b2s_node
            {
                {"Conv2D|DepthwiseConv2dNative", conv2d_node
                    {
                        {"SpaceToBatchND", s2b_node
                            {
                                {"*"},          // Input to the flattened op. s2b_node->args[0]
                                {"*"},          // block_shape s2b_attrs->block_shape
                                {"*"}           // paddings s2b_attrs->paddings
                            }
                        },
                        {"*"}                   // filter conv2d_node->args[1]
                    }
                },
                {"*"},                          // block_shape b2s_attrs->block_shape
                {"*"}                           // crops b2s_attrs->crops
            }
        }
        */

        // Find all the nodes we expect in the subgraph.
        const auto& batch_to_space_node = b2s_node;      // match.node;
        const auto& conv_node = conv2d_node;             // // match.inputs[0].node;
        const auto& filter_node = conv2d_node->args[1];  // match.inputs[0].inputs[1].node;
        const auto& input_node = s2b_node->args[0];  // match.inputs[0].inputs[0].inputs[0].node;
        const auto& space_to_batch_block_shape_node = s2b_attrs->block_shape;  // match.inputs[0].inputs[0].inputs[1].node;

        // The atrous rate value is inferred from the block shape.
        // Tensor block_shape = GetNodeTensorAttr(space_to_batch_block_shape_node, "value");
        const int32_t block_height = space_to_batch_block_shape_node[0];  // block_shape.flat<int32>()(0);
                                                 // space_to_batch_block_shape_node[0]
        const int32_t block_width = space_to_batch_block_shape_node[1];  // block_shape.flat<int32>()(1);

        // Compute the upsampled filter.
        // const auto& filter = GetNodeTensorAttr(filter_node, "value");
        Array<PrimExpr> filter_shape = filter_node->type_as<TensorTypeNode>()->shape;
        const int32_t filter_height = qnn::get_const_int(filter_shape[0]);  // filter.dim_size(0);
        const int32_t filter_width = qnn::get_const_int(filter_shape[1]);   // filter.dim_size(1);
        const int32_t in_channels = qnn::get_const_int(filter_shape[2]);    // filter.dim_size(2);
        const int32_t out_channels = qnn::get_const_int(filter_shape[3]);   // filter.dim_size(3);
        // std::cout << "filter_height " << filter_height << std::endl;
        // std::cout << "filter_width " << filter_width << std::endl;
        // std::cout << "in_channels " << in_channels << std::endl;
        // std::cout << "out_channels " << out_channels << std::endl;

        // ???
        const int32_t upsampled_filter_height = (filter_height - 1) * block_height + 1;
        const int32_t upsampled_filter_width = (filter_width - 1) * block_width + 1;
        // std::cout << "upsampled_filter_height " << upsampled_filter_height << std::endl;
        // std::cout << "upsampled_filter_width " << upsampled_filter_width << std::endl;
        // std::cout << "upsampled_filter " << "(" << upsampled_filter_height << "," << upsampled_filter_width << "," << in_channels << "," << out_channels << ")" << std::endl;
        // Tensor upsampled_filter(DT_FLOAT, TensorShape({upsampled_filter_height, upsampled_filter_width, in_channels, out_channels}));
        
        const auto pre_weight_node = filter_node.as<ConstantNode>();
        if (!pre_weight_node) 
          std::cout << "weight is not ConstantNode" << std::endl;
          //return post;


        // check weight dtype & shape
        auto&& pre_weight = pre_weight_node->data;
        
        auto dtype = pre_weight.DataType();
        DLDevice dev_cpu0_{DLDeviceType::kDLCPU, 0};
        auto weight_data = runtime::NDArray::Empty({upsampled_filter_height, upsampled_filter_width, in_channels, out_channels}, dtype, dev_cpu0_);          
        // auto const_weight_data = Constant(weight_data);
        
        TensorType tensor_type = GetRef<TensorType>(filter_node->type_as<TensorTypeNode>());
        float scalar_value = 1;
        int tensor_num_elements = qnn::get_const_int(tensor_type->Size());
        std::vector<float> tensor_values(tensor_num_elements, scalar_value);
        std::vector<int64_t> tensor_shape = {upsampled_filter_height, upsampled_filter_width, in_channels, out_channels};
        Constant const_weight_data = MakeConstantTensor<float>(DataType::Float(32), tensor_shape, tensor_values);
        // auto const_weight_data_expr = GetRef<Expr>(tvm::relay::transform::InferTypeLocal(const_weight_data));
        // Constant c();
        // Var
        // filter_node->type_as<TensorTypeNode>();
        // MakeConstantTensor(DataType::Float(32), )
        
      /*
        auto filter_eigen = filter.tensor<float, 4>();
        auto upsampled_filter_eigen = upsampled_filter.tensor<float, 4>();

        upsampled_filter_eigen.setZero();
          */
        for (int h = 0; h < filter_height; ++h) {
         for (int w = 0; w < filter_width; ++w) {
           for (int c_in = 0; c_in < in_channels; ++c_in) {
             for (int c_out = 0; c_out < out_channels; ++c_out) {
              //  upsampled_filter_eigen(block_height * h, block_width * w, c_in, c_out) = filter_eigen(h, w, c_in, c_out);
              //  // std::cout << ""
             }
           }
         }
        }
      

        // NodeDef upsampled_filter_node;
        // upsampled_filter_node.set_op("Const");
        // upsampled_filter_node.set_name(filter_node.name());
        // SetNodeAttr("dtype", DT_FLOAT, &upsampled_filter_node);
        // SetNodeTensorAttr<float>("value", upsampled_filter, &upsampled_filter_node);

        // // Set up the new flattened version of the convolution op.
        // NodeDef flattened_conv_node;

        // flattened_conv_node.set_name(batch_to_space_node.name());
        // flattened_conv_node.set_op(conv_node.op());
        // flattened_conv_node.set_device(conv_node.device());

        // AddNodeInput(input_node.name(), &flattened_conv_node);
        // AddNodeInput(upsampled_filter_node.name(), &flattened_conv_node);

        // CopyNodeAttr(conv_node, "T", "T", &flattened_conv_node);
        // CopyNodeAttr(conv_node, "strides", "strides", &flattened_conv_node);
        // SetNodeAttr("padding", "SAME", &flattened_conv_node);
        // CopyNodeAttr(conv_node, "data_format", "data_format", &flattened_conv_node);

        // if (conv_node.op() == "Conv2D") {
        //   CopyNodeAttr(conv_node, "use_cudnn_on_gpu", "use_cudnn_on_gpu", &flattened_conv_node);
        // }

        // new_nodes->push_back(input_node);
        // new_nodes->push_back(upsampled_filter_node);
        // new_nodes->push_back(flattened_conv_node);

        // static inline Expr Conv2D(Expr data, Expr weight, Array<IndexExpr> strides,
        //                          Array<IndexExpr> padding, Array<IndexExpr> dilation, int groups,
        //                          IndexExpr channels, Array<IndexExpr> kernel_size,
        //                          std::string data_layout, std::string kernel_layout,
        //                          std::string out_layout, DataType out_dtype)
        // std::cout << "Conv2D################################################ " << std::endl;
        // std::cout << "data: " << s2b_node->args[0]->checked_type() << std::endl;
        // std::cout << "weight: " << conv2d_node->args[1] << std::endl;  // filter
        // std::cout << "strides: " << conv2d_attrs->strides << std::endl;
        // std::cout << "padding: " << conv2d_attrs->padding << std::endl;
        // std::cout << "dilation: " << conv2d_attrs->dilation << std::endl;
        // std::cout << "groups: " << conv2d_attrs->groups << std::endl;
        // std::cout << "channels: " << conv2d_attrs->channels << std::endl;
        // std::cout << "kernel_size: " << conv2d_attrs->kernel_size << std::endl;
        // std::cout << "data_layout: " << conv2d_attrs->data_layout << std::endl;
        // std::cout << "kernel_layout: " << conv2d_attrs->kernel_layout << std::endl;
        // std::cout << "out_layout: " << conv2d_attrs->out_layout << std::endl;
        // std::cout << "out_dtype: " << conv2d_attrs->out_dtype << std::endl;
        // std::cout << "/Conv2D################################################ " << std::endl;
        // self.replace_quantize_info(b2s_op, conv_op) # op, replace_op // ICE!!!
        //         "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
        // "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        // "dimensions respectively. Default to be same as input layout.");

        // relay::TensorType
        // Expr MakeQnnConv2D(Expr data, Expr weight, Expr input_zero_point, Expr kernel_zero_point,
        //          Expr input_scale, Expr kernel_scale, Array<IndexExpr> strides,
        //          Array<IndexExpr> padding, Array<IndexExpr> dilation, int groups,
        //          IndexExpr channels, Array<IndexExpr> kernel_size, String data_layout,
        //          String kernel_layout, String out_layout, DataType out_dtype) {
       /* Array<IndexExpr> padding = {0, 2, 2, 0};*/
        Array<IndexExpr> dilation = { b2s_attrs->block_shape[0], b2s_attrs->block_shape[1] };
        //Array<IndexExpr> kernel_size = {5, 5};
        //conv2d_attrs->padding.size();
        int padding_size = conv2d_attrs->padding.size();
        const int64_t* a = tir::as_const_int(conv2d_attrs->padding[0]);
        // std::cout << "padding_size s " << padding_size << std::endl;
        // std::cout << "a " << (*a) << std::endl;

        if (padding_size > 0 && (*tir::as_const_int(conv2d_attrs->padding[0])) > 0) {
          // std::cout << "SAME" << std::endl;
        }
        //if (padding_size > 0 && (conv2d_attrs->padding[0] > 0) {
        //  // std::cout << "padding_arg.i = PaddingMode.SAME.value" << std::endl;
        //}

        /*relay::Expr*/ filter_node;

        // attr["padding"] == "SAME" : 
        // stride_h, stride_w = attr["strides"]
        tvm::PrimExpr stride_h = conv2d_attrs->strides[0]; // ICE TODO [1,1]
        tvm::PrimExpr stride_w = conv2d_attrs->strides[1];  

        // kernel_h, kernel_w = attr["kernel_shape"]

        tvm::PrimExpr kernel_h = upsampled_filter_height;
        //tvm::PrimExpr kernel_h = conv2d_attrs->kernel_size[0];
        tvm::PrimExpr kernel_w = upsampled_filter_width;  
        //tvm::PrimExpr kernel_w = conv2d_attrs->kernel_size[1];  
        

        auto& pdata_shape = s2b_node->args[0];
//#Check whether output shapes attribute is set and not None
//        if (opname == "conv_transpose" and len(attr["_output_shapes"]) > 0 and
//            attr["_output_shapes"][0]):
//                pdata_shape = attr["_output_shapes"][0]

        //if attr["data_format"] == "NHWC":
        tvm::PrimExpr in_h;
        tvm::PrimExpr in_w;
        if (conv2d_attrs->data_layout == "NHWC") {
          //        in_h = pdata_shape[1]
          //        in_w = pdata_shape[2]
          in_h = pdata_shape->type_as<TensorTypeNode>()->shape[1];
          in_w = pdata_shape->type_as<TensorTypeNode>()->shape[2];
        } else {
          //        in_h = pdata_shape[2]
          //        in_w = pdata_shape[3]
          in_h = pdata_shape->type_as<TensorTypeNode>()->shape[2];
          in_w = pdata_shape->type_as<TensorTypeNode>()->shape[3];
        }

        //dilation_h = attr["dilations"][0]
        //dilation_w = attr["dilations"][1]
        tvm::PrimExpr dilation_h = conv2d_attrs->dilation[0];
        tvm::PrimExpr dilation_w = conv2d_attrs->dilation[1];
        //dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        //dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        tvm::PrimExpr dilated_kernel_h = (kernel_h - 1) * dilation_h + 1;
        tvm::PrimExpr dilated_kernel_w = (kernel_w - 1) * dilation_w + 1;


        //def _get_pad_pair(input1d, kernel1d, stride1d):
        //    out1d = (input1d + stride1d - 1) // stride1d
        //    pad = np.maximum((out1d - 1) * stride1d + kernel1d - input1d, 0)
        //    pad_before = pad // 2
        //    pad_after = pad - pad_before
        //    return [pad_before, pad_after]

        auto _get_pad_pair = [](tvm::PrimExpr input1d, tvm::PrimExpr kernel1d, tvm::PrimExpr stride1d) {
          tvm::PrimExpr out1d = (input1d + stride1d - 1);  // stride1d
          PrimExpr l = (out1d - 1) * stride1d + kernel1d - input1d;

          PrimExpr pad = topi::maximum(l, 0);
          PrimExpr pad_before = topi::divide(pad, 2);
            
          PrimExpr pad_after = pad - pad_before;

           return Array<tvm::PrimExpr>{pad_before, pad_after};
        };

        //pad_v = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
        //pad_h = _get_pad_pair(in_w, dilated_kernel_w, stride_w)
        Array<tvm::PrimExpr> pad_v = _get_pad_pair(in_h, dilated_kernel_h, stride_h);
        Array<tvm::PrimExpr> pad_h = _get_pad_pair(in_w, dilated_kernel_w, stride_w);

        //Array<IndexExpr> padding = {};
        Array<IndexExpr> padding = {pad_v[0], pad_h[0], pad_v[1], pad_h[1]};
        Array<IndexExpr> kernel_size = {upsampled_filter_height, upsampled_filter_width};
        // std::cout << "new padding: " << padding << std::endl;

        expr = Conv2D( // qnn.conv2d
            /* data */ s2b_node->args[0], 
            // /* weight */ const_weight_data_expr, 
            /* weight */ const_weight_data, 
            // /* weight */conv2d_node->args[1], 
            conv2d_attrs->strides, 
            padding,
            //conv2d_attrs->padding, 
            //s2b_attrs->paddings[0],
            //dilation,
             conv2d_attrs->dilation, // mb default
            conv2d_attrs->groups, 
            conv2d_attrs->channels, 
            kernel_size,
            //conv2d_attrs->kernel_size,
            conv2d_attrs->data_layout,
            conv2d_attrs->kernel_layout,
            conv2d_attrs->out_layout,
            conv2d_attrs->out_dtype
        );

        // std::cout << "after space_to_batch_nd expr# " << expr << std::endl;
        // std::cout << "VisitExpr_: " << call_node->op << "\n => " << expr << std::endl;
        return expr;
        // expr = GetRef<Expr>(s2b_node);

        // std::cout << "expr dtype: " << tvm::relay::transform::InferTypeLocal(expr) << std::endl;
        //// std::cout << "conv2d_attrs->out_dtype: " << conv2d_attrs->out_dtype << std::endl;
        //// std::cout << "expr: " << expr.as<CallNode>()->checked_type() << std::endl;

      }
      //if (op == batch_to_space_nd) {
      //  expr = call_node->args[0];
      //}
      //if (op == space_to_batch_nd) {
      //  expr = expr.as<CallNode>()->args[0];
      //}
      //expr = GetRef<Expr>(s2b_node);
      // std::cout << "VisitExpr_: " << call_node->op  << "\n => " << expr << std::endl;
      return expr;
      //  expr = GetRef<Expr>(call_node);
      //} else {
      //  expr = ExprMutator::VisitExpr_(call_node);
      //}
      // Call the rewrite
      //Array<ObjectRef> vals = fqfq[op](expr, affine_types_);
      // Save the outputs of the rewrite
      //ICHECK(vals.size() == 2)
      //    << "got the wrong number of returned arguments from FTVMFakeQuantizationToInteger for "
      //    << AsText(op, false);
      //out = Downcast<Expr>(vals[0]);

      //affine_types_.Set(out, Downcast<AffineType>(vals[1]));

      //if (call_node == quantize_node_) {
        //out = qnn::MakeDequantize(out, vals[1].as<TensorAffineTypeNode>()->scale,
        //                          vals[1].as<TensorAffineTypeNode>()->zero_point,
        //                          vals[1].as<TensorAffineTypeNode>()->axis);
      //}
    // } else {
    //   ICHECK(false) << "When rewriting a fake quantized graph, found an invalid node "
    //                 << AsText(GetRef<Expr>(call_node), false);
    // }
    // return  ;
  }

  Expr VisitExpr_mace(const CallNode* call_node) {
    // std::cout << "VisitExpr_: " << call_node->op << std::endl;
    Expr out;
    // static auto fqfq = Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMFakeQuantizationToInteger");

    // space_to_batch_nd
    // conv2d
    // batch_to_space_nd


//%79 = nn.space_to_batch_nd(%78,
//%80 = nn.conv2d(%79, meta[relay.Constant][51] 
//%81 = nn.batch_to_space_nd(%80,
//=>>>
//nn.conv2d(%78, meta[relay.Constant][51] 

    Op op = Downcast<Op>(call_node->op);
    // if (fqfq.count(op)) {
    Expr expr;
      if (op == batch_to_space_nd) {
        b2s_node = call_node;
        expr = ExprMutator::VisitExpr_(b2s_node); // 1
        expr = expr.as<CallNode>()->args[0];
        // std::cout << "after batch_to_space_nd expr# " << expr << std::endl;
        // std::cout << "VisitExpr_: " << call_node->op << "\n => " << expr << std::endl;
        return expr;
      } else if (op == conv2d) {
        conv2d_node = call_node;
        expr =  ExprMutator::VisitExpr_(conv2d_node); // 2
        expr = expr.as<CallNode>()->args[0];
        // std::cout << "after conv2d expr# " << expr << std::endl;
        // std::cout << "VisitExpr_: " << call_node->op << "\n => " << expr << std::endl;
        return expr;
      } else if (op == space_to_batch_nd) {  // 3
        s2b_node = call_node;

        ICHECK(b2s_node != nullptr);
        const auto* b2s_attrs = b2s_node->attrs.as<BatchToSpaceNDAttrs>();
        ICHECK(b2s_attrs != nullptr);
        b2s_attrs->block_shape;
        b2s_attrs->crops;

        ICHECK(conv2d_node != nullptr);
        const auto* conv2d_attrs = conv2d_node->attrs.as<Conv2DAttrs>();
        ICHECK(conv2d_attrs != nullptr);
        conv2d_attrs->strides;
        conv2d_attrs->padding;
        conv2d_attrs->dilation;
        conv2d_attrs->groups;
        conv2d_attrs->channels;
        conv2d_attrs->kernel_size;
        conv2d_attrs->data_layout;
        conv2d_attrs->kernel_layout;
        conv2d_attrs->out_layout;
        conv2d_attrs->out_dtype;

        ICHECK(s2b_node != nullptr);
        const auto* s2b_attrs = s2b_node->attrs.as<SpaceToBatchNDAttrs>();
        ICHECK(s2b_attrs != nullptr);
        s2b_attrs->block_shape;
        s2b_attrs->paddings;
        s2b_attrs->pad_value;

        // static inline Expr Conv2D(Expr data, Expr weight, Array<IndexExpr> strides,
        //                          Array<IndexExpr> padding, Array<IndexExpr> dilation, int groups,
        //                          IndexExpr channels, Array<IndexExpr> kernel_size,
        //                          std::string data_layout, std::string kernel_layout,
        //                          std::string out_layout, DataType out_dtype)
        // std::cout << "Conv2D################################################ " << std::endl;
        // std::cout << "data: " << s2b_node->args[0]->checked_type() << std::endl;
        // std::cout << "weight: " << conv2d_node->args[1] << std::endl;  // filter
        // std::cout << "strides: " << conv2d_attrs->strides << std::endl;
        // std::cout << "padding: " << conv2d_attrs->padding << std::endl;
        // std::cout << "dilation: " << conv2d_attrs->dilation << std::endl;
        // std::cout << "groups: " << conv2d_attrs->groups << std::endl;
        // std::cout << "channels: " << conv2d_attrs->channels << std::endl;
        // std::cout << "kernel_size: " << conv2d_attrs->kernel_size << std::endl;
        // std::cout << "data_layout: " << conv2d_attrs->data_layout << std::endl;
        // std::cout << "kernel_layout: " << conv2d_attrs->kernel_layout << std::endl;
        // std::cout << "out_layout: " << conv2d_attrs->out_layout << std::endl;
        // std::cout << "out_dtype: " << conv2d_attrs->out_dtype << std::endl;
        // std::cout << "/Conv2D################################################ " << std::endl;
        // self.replace_quantize_info(b2s_op, conv_op) # op, replace_op // ICE!!!
        //         "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
        // "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        // "dimensions respectively. Default to be same as input layout.");

        // relay::TensorType
        // Expr MakeQnnConv2D(Expr data, Expr weight, Expr input_zero_point, Expr kernel_zero_point,
        //          Expr input_scale, Expr kernel_scale, Array<IndexExpr> strides,
        //          Array<IndexExpr> padding, Array<IndexExpr> dilation, int groups,
        //          IndexExpr channels, Array<IndexExpr> kernel_size, String data_layout,
        //          String kernel_layout, String out_layout, DataType out_dtype) {
        //Array<IndexExpr> padding = {0, 2, 2, 0};
        Array<IndexExpr> dilation = { b2s_attrs->block_shape[0], b2s_attrs->block_shape[1] };
        Array<IndexExpr> strides = { 1, 1 };
        //Array<IndexExpr> kernel_size = {5, 5};
        //conv2d_attrs->padding.size();
        int padding_size = conv2d_attrs->padding.size();
        const int64_t* a = tir::as_const_int(conv2d_attrs->padding[0]);
        // std::cout << "padding_size " << padding_size << std::endl;
        // std::cout << "a " << (*a) << std::endl;

        if (padding_size > 0 && (*tir::as_const_int(conv2d_attrs->padding[0])) > 0) {
          // std::cout << "SAME" << std::endl;
        }
        //if (padding_size > 0 && (conv2d_attrs->padding[0] > 0) {
        //  // std::cout << "padding_arg.i = PaddingMode.SAME.value" << std::endl;
        //}

        
        // attr["padding"] == "SAME" :
        // stride_h, stride_w = attr["strides"]
        tvm::PrimExpr stride_h = conv2d_attrs->strides[0]; // ICE TODO [1,1]
        tvm::PrimExpr stride_w = conv2d_attrs->strides[1];

        // kernel_h, kernel_w = attr["kernel_shape"]
        tvm::PrimExpr kernel_h = conv2d_attrs->kernel_size[0];
        tvm::PrimExpr kernel_w = conv2d_attrs->kernel_size[1];

        auto& pdata_shape = s2b_node->args[0];
        //#Check whether output shapes attribute is set and not None
        //        if (opname == "conv_transpose" and len(attr["_output_shapes"]) > 0 and
        //            attr["_output_shapes"][0]):
        //                pdata_shape = attr["_output_shapes"][0]

        // if attr["data_format"] == "NHWC":
        tvm::PrimExpr in_h;
        tvm::PrimExpr in_w;
        if (conv2d_attrs->data_layout == "NHWC") {
          //        in_h = pdata_shape[1]
          //        in_w = pdata_shape[2]
          in_h = pdata_shape->type_as<TensorTypeNode>()->shape[1];
          in_w = pdata_shape->type_as<TensorTypeNode>()->shape[2];
        } else {
          //        in_h = pdata_shape[2]
          //        in_w = pdata_shape[3]
          in_h = pdata_shape->type_as<TensorTypeNode>()->shape[2];
          in_w = pdata_shape->type_as<TensorTypeNode>()->shape[3];
        }

        // dilation_h = attr["dilations"][0]
        // dilation_w = attr["dilations"][1]
        //tvm::PrimExpr dilation_h = conv2d_attrs->dilation[0];
        tvm::PrimExpr dilation_h = dilation[0];
        //tvm::PrimExpr dilation_w = conv2d_attrs->dilation[1];
        tvm::PrimExpr dilation_w = dilation[1];
        // dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        // dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        tvm::PrimExpr dilated_kernel_h = (kernel_h - 1) * dilation_h + 1;
        tvm::PrimExpr dilated_kernel_w = (kernel_w - 1) * dilation_w + 1;

        // def _get_pad_pair(input1d, kernel1d, stride1d):
        //    out1d = (input1d + stride1d - 1) // stride1d
        //    pad = np.maximum((out1d - 1) * stride1d + kernel1d - input1d, 0)
        //    pad_before = pad // 2
        //    pad_after = pad - pad_before
        //    return [pad_before, pad_after]

        auto _get_pad_pair = [](tvm::PrimExpr input1d, tvm::PrimExpr kernel1d,
                                tvm::PrimExpr stride1d) {
          tvm::PrimExpr out1d = (input1d + stride1d - 1);  // stride1d
          PrimExpr l = (out1d - 1) * stride1d + kernel1d - input1d;

          PrimExpr pad = topi::maximum(l, 0);
          PrimExpr pad_before = topi::divide(pad, 2);  //
          PrimExpr pad_after = pad - pad_before;
          
          //cast(DataType::Int(pad_before->dtype->bits), pad_before)

          return Array<tvm::PrimExpr>{pad_before, pad_after};
        };

        // pad_v = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
        // pad_h = _get_pad_pair(in_w, dilated_kernel_w, stride_w)
        Array<tvm::PrimExpr> pad_v = _get_pad_pair(in_h, dilated_kernel_h, stride_h);
        Array<tvm::PrimExpr> pad_h = _get_pad_pair(in_w, dilated_kernel_w, stride_w);

        //Array<IndexExpr> padding = {1,1,1,1};
        Array<IndexExpr> padding = {pad_v[0], pad_h[0], pad_v[1], pad_h[1]};

        //Array<IndexExpr> padding = {pad_v[0], pad_h[0] + 1, pad_v[1], 0};
        //Array<IndexExpr> padding = {pad_v[0], 0, pad_v[1], pad_h[1]+1};
        //Array<IndexExpr> padding = {0, 0, pad_v[1]+1, pad_h[1]+1};
        //Array<IndexExpr> padding = {1, 1, pad_v[1]+1, pad_h[1]+1};

        //Array<IndexExpr> padding = {0, 0, 2, 2}; +
        //Array<IndexExpr> padding = {0, 1, 1, 2};
        //Array<IndexExpr> padding = {0, 1, 2, 1};+
        //Array<IndexExpr> padding = {0, 2, 0, 2};
        //Array<IndexExpr> padding = {0, 2, 1, 1};
        //Array<IndexExpr> padding = {0, 2, 2, 0};+
        //Array<IndexExpr> padding = {1, 0, 1, 2};
        //Array<IndexExpr> padding = {1, 0, 2, 1};
        //Array<IndexExpr> padding = {1, 1, 0, 2};
        //Array<IndexExpr> padding = {1, 1, 1, 1};+
        //Array<IndexExpr> padding = {1, 1, 2, 0};
        //Array<IndexExpr> padding = {1, 2, 0, 1};
        //Array<IndexExpr> padding = {1, 2, 1, 0};+
        //Array<IndexExpr> padding = {2, 0, 0, 2};+
        //Array<IndexExpr> padding = {2, 0, 1, 1};
        //Array<IndexExpr> padding = {2, 0, 2, 0};
        //Array<IndexExpr> padding = {2, 1, 0, 1};+
        //Array<IndexExpr> padding = {2, 1, 1, 0};
        //Array<IndexExpr> padding = {2, 2, 0, 0};+

        // std::cout << "new padding: " << padding << std::endl;


        // /*relay::Expr*/ filter_node;
        expr = Conv2D( // qnn.conv2d
            /* data */ s2b_node->args[0], 
            /* weight */conv2d_node->args[1], 
            // conv2d_attrs->strides, 
            strides,
            padding,
            //conv2d_attrs->padding, 
            //s2b_attrs->paddings[0],
             dilation,
            //conv2d_attrs->dilation, 
            conv2d_attrs->groups, 
            conv2d_attrs->channels, 
            //kernel_size,
             conv2d_attrs->kernel_size,
            conv2d_attrs->data_layout,
            conv2d_attrs->kernel_layout,
            conv2d_attrs->out_layout,
            conv2d_attrs->out_dtype
        );

        // std::cout << "after space_to_batch_nd expr# " << expr << std::endl;
        // std::cout << "VisitExpr_: " << call_node->op << "\n => " << expr << std::endl;
        return expr;
        // expr = GetRef<Expr>(s2b_node);

        // std::cout << "expr dtype: " << tvm::relay::transform::InferTypeLocal(expr) << std::endl;
        //// std::cout << "conv2d_attrs->out_dtype: " << conv2d_attrs->out_dtype << std::endl;
        //// std::cout << "expr: " << expr.as<CallNode>()->checked_type() << std::endl;

      }
      //if (op == batch_to_space_nd) {
      //  expr = call_node->args[0];
      //}
      //if (op == space_to_batch_nd) {
      //  expr = expr.as<CallNode>()->args[0];
      //}
      //expr = GetRef<Expr>(s2b_node);
      // std::cout << "VisitExpr_: " << call_node->op  << "\n => " << expr << std::endl;
      return expr;
      //  expr = GetRef<Expr>(call_node);
      //} else {
      //  expr = ExprMutator::VisitExpr_(call_node);
      //}
      // Call the rewrite
      //Array<ObjectRef> vals = fqfq[op](expr, affine_types_);
      // Save the outputs of the rewrite
      //ICHECK(vals.size() == 2)
      //    << "got the wrong number of returned arguments from FTVMFakeQuantizationToInteger for "
      //    << AsText(op, false);
      //out = Downcast<Expr>(vals[0]);

      //affine_types_.Set(out, Downcast<AffineType>(vals[1]));

      //if (call_node == quantize_node_) {
        //out = qnn::MakeDequantize(out, vals[1].as<TensorAffineTypeNode>()->scale,
        //                          vals[1].as<TensorAffineTypeNode>()->zero_point,
        //                          vals[1].as<TensorAffineTypeNode>()->axis);
      //}
    // } else {
    //   ICHECK(false) << "When rewriting a fake quantized graph, found an invalid node "
    //                 << AsText(GetRef<Expr>(call_node), false);
    // }
    // return  ;
  }

  // Expr Rewrite_(const CallNode* pre, const Expr& post) override {

  //   // %27 = qnn.dequantize(%24, 0.299955f /* ty=float32 */, 0 /* ty=int32 */, axis=1) /* ty=Tensor[(1, 48, 1, 1), float32] */;
  //   // %28 = cast(%26, dtype="float32") /* ty=Tensor[(12, 48, 1, 1), float32] */;

  //   if (pre->op == cast_op_) {
  //     // // std::cout << "pre->args[0]->checked_type() " <<  pre->args[0]->checked_type() << std::endl;
  //     // // std::cout << "pre->checked_type() " << pre->checked_type() << std::endl;
  //     // // std::cout << "pre->args[0]->checked_type().as<TensorTypeNode>()->dtype " <<  pre->args[0]->checked_type().as<TensorTypeNode>()->dtype << std::endl;
  //     // // std::cout << "pre->attrs.as<CastAttrs>()->dtype " << pre->attrs.as<CastAttrs>()->dtype << std::endl;

  //     if(pre->args[0]->checked_type().as<TensorTypeNode>()->dtype == pre->attrs.as<CastAttrs>()->dtype) {
  //       // std::cout << "pre->op " << pre->args[0].as<CallNode>()->op << std::endl;
  //       // std::cout << "post->op " << post.as<CallNode>()->args[0].as<CallNode>()->op << std::endl;
  //       // std::cout << "shape.size() " << pre->args[0]->checked_type().as<TensorTypeNode>()->shape.size() << std::endl;
  //       // std::cout << "is_scalar " << pre->attrs.as<CastAttrs>()->dtype.is_scalar() <<  std::endl;
  //       // // std::cout << "FastCastMutator post " << post.as<CallNode>()->args[0] << std::endl;
  //       // // std::cout << "FastCastMutator pre  " << pre->args[0] << std::endl;
  //       return post.as<CallNode>()->args[0];
  //     }
  //   }
  //   return post;
  // }

  Expr VisitExpr_(const TupleNode* node) {
    Expr expr = ExprMutator::VisitExpr_(node);
    auto new_node = expr.as<TupleNode>();
    Array<TensorAffineType> types;
    for (Expr field : new_node->fields) {
      // ICHECK(affine_types_[field].as<TensorAffineTypeNode>());
      // types.push_back(Downcast<TensorAffineType>(affine_types_[field]));
    }
    //affine_types_.Set(expr, TupleAffineType(types));
    return expr;
  }

  Expr VisitExpr_(const TupleGetItemNode* node) {
    Expr expr = ExprMutator::VisitExpr_(node);
    // auto tuple_type = affine_types_[expr.as<TupleGetItemNode>()->tuple].as<TupleAffineTypeNode>();
    //affine_types_.Set(expr, tuple_type->types[node->index]);
    return expr;
  }

  Op batch_to_space_nd = Op::Get("nn.batch_to_space_nd");
  Op conv2d = Op::Get("nn.conv2d");
  Op space_to_batch_nd = Op::Get("nn.space_to_batch_nd");
  ExprSet subgraph_;
  //AffineTypeMap affine_types_;
  //const Op dequantize_op_ = Op::Get("qnn.dequantize");
  const CallNode* quantize_node_ = nullptr;
};

class FlattenAtrousConvRewriter : public MixedModeMutator {
 public:
  FlattenAtrousConvRewriter(bool tf) : _tf(tf) {}

 protected:
  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (const CallNode* call_node = post.as<CallNode>()) {
      if (call_node->op == batch_to_space_nd) {
        // std::cout << call_node->op << std::endl; 
        FlattenAtrousConvSubgraphExtractor extractor;
        ExprSet subgraph = extractor.GetSubgraph(post);
        //AffineTypeMap affine_types = extractor.GetAffineTypes();
        Expr out = FlattenAtrousConvSubgraphMutator(subgraph, _tf).MutateSubgraph(post);
        // std::cout << "out\n" << out << std::endl;
        //Expr out = post;
        return out;
      }
    }
    return post;
  }
  bool _tf = true;
  Op batch_to_space_nd = Op::Get("nn.batch_to_space_nd");
  Op conv2d = Op::Get("nn.conv2d");
  Op space_to_batch_nd = Op::Get("nn.space_to_batch_nd");
};

Expr FlattenAtrousConv(const Expr& expr, const IRModule& mod, bool tf) {
   //std::cout << "FlattenAtrousConv before     \n" << expr << std::endl;
  auto fac_expr = FlattenAtrousConvRewriter(tf).Mutate(expr);
   //std::cout << "FlattenAtrousConv after pass \n" << fac_expr << std::endl;
  //auto faci = tvm::relay::InferType(fac_expr);
  //// std::cout << "FlattenAtrousConv after infer\n" << faci << std::endl;
  return fac_expr;
  //return faci;
  // return FlattenAtrousConvRewriter().Mutate(expr);
}


namespace transform {

Pass FlattenAtrousConv(bool tf) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(FlattenAtrousConv(f, m, tf));
      };
  return CreateFunctionPass(pass_func, 0, "FlattenAtrousConv", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FlattenAtrousConv")
    .set_body_typed(FlattenAtrousConv);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
