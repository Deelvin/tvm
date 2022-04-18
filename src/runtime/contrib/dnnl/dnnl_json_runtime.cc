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
 * \file src/runtime/contrib/dnnl/dnnl_json_runtime.cc
 * \brief A simple JSON runtime for DNNL.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <regex>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "dnnl.hpp"
#include "dnnl_node_helper.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class DNNLJSONRuntime : public JSONRuntimeBase {
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;

 public:
  DNNLJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String>& const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names),
        g_explorer_(nodes_, data_entry_, node_row_ptr_, engine_) {}

  const char* type_key() const override { return "dnnl_json"; }

  static std::string get_version() {
    auto v = dnnl_version();
    std::stringstream ver_strm;
    ver_strm << v->major << '.' << v->minor << '.' << v->patch;
    return ver_strm.str();
  }

  void Init(const Array<NDArray>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    // Setup constants entries for weights.
    SetupConstants(consts);
    // Init internal DNNL specific objects
    BuildEngine();
  }

  /**
   * Override of GetFunction methods to replace main symbol_name_ implementation with
   * thread safe one.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) override {
    if (this->symbol_name_ == name) {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK(this->initialized_) << "The module has not been initialized";

        ICHECK_EQ(args.size(), input_var_eid_.size() + outputs_.size())
            << "Found mismatch in the number of provided data entries and required.";

        Run(args);
      });
    } else {
      return JSONRuntimeBase::GetFunction(name, sptr_to_self);
    }
  }

  /**
   * @brief Thread safe version of base method Run.
   *
   * The main purpose of this overwrite is to make symbol_name_ function thread safe.
   * The base implementation of that method is using SetInputOutputBuffers() which
   * is not thread safe and lead to changes in DNNLJSONRuntime itself.
   *
   * @param args kernel arguments
   */
  void Run(const TVMArgs& args) const {
    auto io_data_provider = makeIoDataProvider(args);
    // Execute primitives one by one
    for (const auto& act : net_) {
      auto req_args = std::get<TensorRegistry::ArgReqSet>(act);
      auto prim = std::get<dnnl::primitive>(act);

      // Find proper dnnl::memory buffer based on provided ArgRequisite
      auto mem_args = tensor_registry_.solve(req_args, io_data_provider);
      prim.execute(stream_, mem_args);
    }
  }

  /** @brief Stub implementation */
  void Run() override { LOG(ERROR) << "Unimplemented. Should never be called."; }

 private:

   /** Receive tensor memory buffer handler based from provided arg */
  static void* extractDataHandle(const TVMArgValue& val) {
    ICHECK(val.type_code() == kTVMNDArrayHandle || val.type_code() == kTVMDLTensorHandle)
        << "Expect NDArray or DLTensor";
    void* hdl = nullptr;
    if (val.IsObjectRef<NDArray>()) {
      NDArray arr = val;
      hdl = arr.operator->()->data;
    } else {
      hdl = val.operator DLTensor*()->data;
    }
    return hdl;
  }

  TensorRegistry::ExtDataProvider makeIoDataProvider(const TVMArgs& args) const {
    std::map<uint32_t, void*> io_map;  // eid to data handler

    int i = 0;
    for (auto e : input_var_eid_) io_map[e] = extractDataHandle(args[i++]);
    for (auto e : outputs_) io_map[EntryID(e)] = extractDataHandle(args[i++]);

    // lambda with captured IO data handlers
    return [io_map](uint32_t eid) -> void* { return io_map.at(eid); };
  }

  std::set<uint32_t> makeIoEids() const {
    std::set<uint32_t> io_set;  // eid of inputs and outputs
    for (auto e : input_var_eid_) io_set.insert(e);
    for (auto e : outputs_) io_set.insert(EntryID(e));
    return io_set;
  }

  struct SubmitAttr {
    enum AttrType { None, ZeroCopyRequest };

    SubmitAttr() {}
    SubmitAttr(AttrType type, const TensorRequisite& tr, int flag)
        : type_(type), tr_(tr), flag_(flag) {}

    AttrType type_ = AttrType::None;
    const TensorRequisite tr_ = {};
    int flag_ = 0;
  };

  // Helper function to register primitive into execution queue
  void submit(const dnnl::primitive& prim, const std::unordered_map<int, TensorRequisite>& tr_args,
              const SubmitAttr attr = {}) {
    // collection of post action. Dst primitive processing will be stored here
    TensorRegistry::ActionQue post_actions;

    // Helper func to register TensorRequisite and store corresponding Actions in proper place
    auto register_tr = [this, &post_actions](const TensorRequisite& tr) {
      TensorRegistry::ArgReq arg_req;
      TensorRegistry::ActionQue actions;
      std::tie(arg_req, actions) = tensor_registry_.registerTR(tr);

      auto& action_queue = tr.isReversed() ? post_actions : net_;
      action_queue.insert(action_queue.end(), actions.begin(), actions.end());
      return arg_req;
    };

    // Register all provided TR arguments
    std::unordered_map<int, TensorRegistry::ArgReq> arg_reqs;
    for (const auto& kvp : tr_args) {
      const auto& tr = kvp.second;
      const auto& key = kvp.first;

      if (!tr.defined()) continue;  // empty arg is admitted. Just skip it
      arg_reqs[key] = register_tr(tr);
    }

    // ZeroCopyRequest or Inplace memory
    if (attr.type_ == SubmitAttr::ZeroCopyRequest) {
      auto zero_copy_src_tr = attr.tr_;
      auto zero_copy_dst_tr = tr_args.at(attr.flag_);
      auto zero_copy_src_ar = register_tr(zero_copy_src_tr);
      auto zero_copy_dst_ar = arg_reqs.at(attr.flag_);

      // Register copy action direct before main primitive
      dnnl::reorder::primitive_desc io_copy_pd(engine_, zero_copy_src_tr.desc(), engine_,
                                               zero_copy_dst_tr.desc());
      net_.push_back({dnnl::reorder(io_copy_pd),
                      {{DNNL_ARG_SRC, zero_copy_src_ar}, {DNNL_ARG_DST, zero_copy_dst_ar}}});
    }

    // Register main primitive
    net_.push_back({prim, arg_reqs});

    // Register post actions
    net_.insert(net_.end(), post_actions.begin(), post_actions.end());
  }

  // Build up the engine based on the input graph.

  std::map<std::string, dnnl::algorithm> elt_name2algo{
      {"abs", dnnl::algorithm::eltwise_abs},
      {"exp", dnnl::algorithm::eltwise_exp},
      {"log", dnnl::algorithm::eltwise_log},
      {"sqrt", dnnl::algorithm::eltwise_sqrt},
      {"round", dnnl::algorithm::eltwise_round},
      {"logsumexp", dnnl::algorithm::eltwise_logsigmoid},
      {"nn.relu", dnnl::algorithm::eltwise_relu},
      {"nn.leaky_relu", dnnl::algorithm::eltwise_relu},
      {"tanh", dnnl::algorithm::eltwise_tanh},
      {"sigmoid", dnnl::algorithm::eltwise_logistic},
      {"clip", dnnl::algorithm::eltwise_clip},
  };

  std::map<std::string, tag> layout_dict{
      {"", tag::any},
      {"NCW", tag::ncw},
      {"NWC", tag::nwc},
      {"OIW", tag::oiw},
      {"GOIW", tag::goiw},
      {"NCHW", tag::nchw},
      {"NHWC", tag::nhwc},
      {"OIHW", tag::oihw},
      {"GOIHW", tag::goihw},
      {"NCDHW", tag::ncdhw},
      {"NDHWC", tag::ndhwc},
      {"OIDHW", tag::oidhw},
      {"GOIDHW", tag::goidhw},
      {"IOHW", tag::iohw},
      {"GIOHW", tag::giohw},
      {"IODHW", tag::iodhw},
      {"GIODHW", tag::giodhw},

      // Blocking layout.
      {"NCW8c", tag::nCw8c},
      {"NCW16c", tag::nCw16c},
      {"OIW16i16o", tag::OIw8i8o},
      {"OIW16i16o", tag::OIw16i16o},
      {"OWI8o", tag::Owi8o},
      {"OWI16o", tag::Owi16o},
      {"NCHW4c", tag::nChw4c},
      {"NCHW8c", tag::nChw8c},
      {"NCHW16c", tag::nChw16c},
      {"OIHW8i8o", tag::OIhw8i8o},
      {"IOHW8i8o", tag::any},
      {"OIHW16i16o", tag::OIhw16i16o},
      {"IOHW16i16o", tag::IOhw16i16o},
      {"GOIHW4i4o", tag::gOIhw4i4o},
      {"GOIHW8i8o", tag::gOIhw8i8o},
      {"GOIHW16i16o", tag::gOIhw16i16o},
      {"OHWI8o", tag::Ohwi8o},
      {"OHWI16o", tag::Ohwi16o},
      {"OHWI32o", tag::Ohwi32o},
      {"OHWI48o", tag::Ohwi48o},
      {"OHWI64o", tag::Ohwi64o},
      {"GOIHW8g", tag::Goihw8g},
      {"GOIHW16g", tag::Goihw16g},
      {"NCDHW8c", tag::nCdhw8c},
      {"NCDHW16c", tag::nCdhw16c},
      {"OIDHW16i16o", tag::OIdhw16i16o},
      {"IODHW16i16o", tag::IOdhw16i16o},
      {"OIDHW8i8o", tag::OIdhw8i8o},
      {"IODHW8i8o", tag::any},
      {"ODHWI8o", tag::Odhwi8o},
      {"ODHWI16o", tag::Odhwi16o},
      {"ODHWI32o", tag::Odhwi32o},
      {"ODHWI48o", tag::Odhwi48o},
      {"ODHWI64o", tag::Odhwi64o},
  };

  bool ParsingOpName(const std::string op_name, dnnl::primitive_attr& attr) {
    // Define RegExp.
    std::regex bias_add_pat(".*_bias.*");
    std::regex relu_pat(".*_relu.*");
    std::regex tanh_pat(".*_tanh.*");
    std::regex sigmoid_pat(".*_sigmoid.*");

    // Parsing post-ops.
    dnnl::post_ops ops;
    if (std::regex_match(op_name, relu_pat)) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 0.f);
    }
    if (std::regex_match(op_name, tanh_pat)) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_tanh, 0.f, 0.f);
    }
    if (std::regex_match(op_name, sigmoid_pat)) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_logistic, 0.f, 0.f);
    }
    attr.set_post_ops(ops);

    // Parsing bias_add.
    return std::regex_match(op_name, bias_add_pat) ? true : false;
  }

  dnnl::memory::dims Transform2Dims(const std::vector<int>& vals,
                                    bool dilates = false) {
    dnnl::memory::dims dims(vals.begin(), vals.end());
    if (dilates) {
      std::transform(dims.begin(), dims.end(), dims.begin(),
                     [](int v) { return v - 1; });
    }
    return dims;
  }

  void BuildEngine() {
    engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
    stream_ = dnnl::stream(engine_);
    tensor_registry_ = TensorRegistry(engine_, makeIoEids());

    std::regex conv_pat(".*conv[1-3]d.*");
    std::regex deconv_pat(".*deconv[1-3]d.*");
    std::regex conv_transpose_pat(".*conv[1-3]d_transpose.*");
    std::regex dense_pat(".*dense.*");
    std::regex max_pool_pat(".*max_pool[1-3]d");
    std::regex avg_pool_pat(".*avg_pool[1-3]d");

    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();

        if (std::regex_match(op_name, deconv_pat) ||
            std::regex_match(op_name, conv_transpose_pat)) {
          Deconvolution(nid);
        } else if (std::regex_match(op_name, conv_pat)) {
          Convolution(nid);
        } else if("dnnl.qnn.dense_dequantize" == op_name) {
          DenseDequantize(nid);
        } else if ("dnnl.qnn.dense_add_req" == op_name) {
          DenseAddRequantize(nid);
        } else if (std::regex_match(op_name, dense_pat)) {
          Dense(nid);
        } else if ("nn.batch_norm" == op_name) {
          BatchNorm(nid);
        } else if (std::regex_match(op_name, max_pool_pat)) {
          Pooling(nid, dnnl::algorithm::pooling_max);
        } else if (std::regex_match(op_name, avg_pool_pat)) {
          Pooling(nid, dnnl::algorithm::pooling_avg);
        } else if (elt_name2algo.count(op_name)) {
          Eltwise(nid);
        } else if ("nn.softmax" == op_name) {
          Softmax(nid);
        } else if ("add" == op_name) {
          Binary(nid, dnnl::algorithm::binary_add);
        } else if ("multiply" == op_name) {
          Binary(nid, dnnl::algorithm::binary_mul);
        } else if ("dnnl.qnn.matmul_dequantize" == op_name ||
                   "dnnl.qnn.matmul_dequantize_div" == op_name ||
                   "dnnl.qnn.matmul_req" == op_name) {
          QnnMatmul(nid);
        } else if ("dnnl.layer.normalize" == op_name) {
          LayerNorm(nid);
        } else if ("dnnl.softmax_qnn.quantize" == op_name) {
          QnnSoftmax(nid);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
    tensor_registry_.finalize();
  }

  void my_print(const std::vector<int>& dims) {
    std::cout << "size=" << dims.size() << " [ ";
    for (auto d : dims)
      std::cout << d << " ";
    std::cout << "]" << std::endl;
  }

  void my_dims_print(const dnnl::memory::dims& dims) {
    std::cout << "size=" << dims.size() << " [ ";
    for (auto d : dims)
      std::cout << d << " ";
    std::cout << "]" << std::endl;
  }

  void Convolution(const size_t& nid) {
    auto node = NodeHelper{nid, g_explorer_};

    // Fix position inputs
    auto data_tr = node.getInput(0);
    auto kernel_tr = node.getInput(1);
    auto output_tr = node.getOutput(0);

    // Parse general conv attributes
    auto strides = node.getAttr<std::vector<int>>("strides");
    auto padding = node.getAttr<std::vector<int>>("padding");
    auto dilation = node.getAttr<std::vector<int>>("dilation");
    //auto groups = node.getAttr<dnnl::memory::dim>("groups");

    decltype(padding) padding_l(padding.begin(), padding.begin() + padding.size() / 2);
    decltype(padding) padding_r(padding.end() - padding.size() / 2, padding.end());

    auto data_layout = node.getAttr<std::string>("data_layout");
    auto kernel_layout = node.getAttr<std::string>("kernel_layout");

    auto activation = node.getAttr<std::vector<std::string>>("activation", {"none"});
    auto bias_idx = node.getAttr<int>("bias_idx", {"-1"});
    auto sum_idx = node.getAttr<int>("sum_idx", {"-1"});
    auto sum_scl_idx = node.getAttr<int>("sum_scl_idx", {"-1"});
    auto o_scl_idx = node.getAttr<int>("o_scl_idx", {"-1"});
    auto dst_zp_idx = node.getAttr<int>("dst_zp_idx", {"-1"});

    // may be empty in case if '-1'
    auto bias_tr = node.getInput(bias_idx);
    auto sum_tr = node.getInput(sum_idx);
    auto sum_scl_tr = node.getInput(sum_scl_idx);
    auto o_scl_tr = node.getInput(o_scl_idx);
    auto dst_zp_tr = node.getInput(dst_zp_idx);

        // Check layout.
    if (layout_dict.find(data_layout) == layout_dict.end()) {
      LOG(FATAL) << "Unsupported data layout for conv: " << data_layout;
    }

    if (layout_dict.find(kernel_layout) == layout_dict.end()) {
      layout_dict.insert({kernel_layout, tag::any});
      LOG(WARNING) << "Unregistered kernel layout for conv: " << kernel_layout
                   << ", transfer to tag::any";
    }

    // permute corresponding with provided layouts
    auto data_permutation = utils::permutation(data_layout);
    auto kernel_permutation = utils::permutation(kernel_layout);
    auto data_reshape = utils::reshape(data_tr.dims(), data_layout);
    auto kernel_reshape = utils::reshape(kernel_tr.dims(), kernel_layout);

    data_tr = data_tr.permute(data_permutation).reshape(data_reshape);
    kernel_tr = kernel_tr.permute(kernel_permutation).reshape(kernel_reshape);
    sum_tr = sum_tr.permute(data_permutation).reshape(data_reshape);
    output_tr = output_tr.permute(data_permutation);

    // TODO(@apeskov): temp WA. while codegen is not able to guarantee 1D format of bias data
    bias_tr = bias_tr.squeeze();

    // Group weight format
    /*if (groups > 1) {
      LOG(FATAL) << "Not checked 1";
      auto k_dims = kernel_tr.dims();  // OIHW -> GOIHW
      k_dims[0] /= groups;
      k_dims.insert(k_dims.begin(), groups);
      kernel_tr = kernel_tr.reshape(k_dims);
    }*/

    // Attributes setting
    dnnl::primitive_attr attr;
    ParsingOpName(node.GetOpName(), attr);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    if (dst_zp_tr) {
      auto zp = dst_zp_tr.getConstScalarData<int32_t>();
      // Per channel zp is not supported. It was merged into BIAS
      attr.set_zero_points(DNNL_ARG_DST, 0, {zp});
    }

    if (o_scl_tr) {
      ICHECK(o_scl_tr.isConstant());
      auto data = o_scl_tr.getConstDataLikeVec<float>();
      attr.set_output_scales(data.size() == 1 ? 0 : (1 << 1), data);
    }

    /*if (activation[0] != "none") {
      LOG(FATAL) << "Not checked 2";
      auto a_type = utils::convert2dnnl_activation(activation[0]);
      auto a_scale = node.getInput(std::stoi(activation[1])).getConstScalarData<float>();
      auto a_alfa = node.getInput(std::stoi(activation[2])).getConstScalarData<float>();
      auto a_beta = node.getInput(std::stoi(activation[3])).getConstScalarData<float>();

      auto ops = attr.get_post_ops();
      ops.append_eltwise(a_scale, a_type, a_alfa, a_beta);
      attr.set_post_ops(ops);
    }*/

    if (sum_scl_tr) {
      auto scl = sum_scl_tr.getConstScalarData<float>();
      auto ops = attr.get_post_ops();
      ops.append_sum(scl);
      attr.set_post_ops(ops);
    }

    dnnl::memory::dims strides_dims = Transform2Dims(strides);
    dnnl::memory::dims dilates_dims = Transform2Dims(dilation, true);
    dnnl::memory::dims padding_l_dims = Transform2Dims(padding_l);
    dnnl::memory::dims padding_r_dims = Transform2Dims(padding_r);

    // Conv description
    auto conv_d = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
        data_tr.layout(layout_dict[data_layout]).desc(),
        kernel_tr.layout(layout_dict[kernel_layout]).desc(),
        bias_tr.layoutAny().desc(), output_tr.layoutAny().desc(),
        strides_dims, dilates_dims, padding_l_dims, padding_r_dims);
    auto conv_pd = dnnl::convolution_forward::primitive_desc(conv_d, attr, engine_);
    auto conv = dnnl::convolution_forward(conv_pd);

    // Specify proper layouts
    data_tr = data_tr.requestLayout(conv_pd.src_desc());
    kernel_tr = kernel_tr.requestLayout(conv_pd.weights_desc());
    output_tr = output_tr.requestLayout(conv_pd.dst_desc());
    bias_tr = bias_tr.requestLayout(conv_pd.bias_desc());

    auto scratchpad_tr = node.makeScratchpad(conv_pd.scratchpad_desc());

    // Inplace request for conv+sum pattern. Match input with dst tensor
    auto submit_attr =
        sum_tr ? SubmitAttr{SubmitAttr::ZeroCopyRequest, sum_tr, DNNL_ARG_DST} : SubmitAttr{};

    // Register prim to execute
    submit(conv,
           {{DNNL_ARG_SRC, data_tr},
            {DNNL_ARG_WEIGHTS, kernel_tr},
            {DNNL_ARG_BIAS, bias_tr},
            {DNNL_ARG_SCRATCHPAD, scratchpad_tr},
            {DNNL_ARG_DST, output_tr}},
           submit_attr);
  }

  void Deconvolution(const size_t& nid) {
    auto node = NodeHelper{nid, g_explorer_};

    // Fix position inputs
    auto data_tr = node.getInput(0);
    auto kernel_tr = node.getInput(1);
    auto output_tr = node.getOutput(0);

    // Parse general conv attributes
    auto strides = node.getAttr<std::vector<int>>("strides");
    auto padding = node.getAttr<std::vector<int>>("padding");
    auto dilation = node.getAttr<std::vector<int>>("dilation");
    auto groups = node.getAttr<dnnl::memory::dim>("groups");

    decltype(padding) padding_l(padding.begin(), padding.begin() + padding.size() / 2);
    decltype(padding) padding_r(padding.end() - padding.size() / 2, padding.end());

    auto data_layout = node.getAttr<std::string>("data_layout");
    auto kernel_layout = node.getAttr<std::string>("kernel_layout");

    auto activation = node.getAttr<std::vector<std::string>>("activation", {"none"});
    auto bias_idx = node.getAttr<int>("bias_idx", {"-1"});
    auto sum_idx = node.getAttr<int>("sum_idx", {"-1"});

    // may be empty in case if '-1'
    auto bias_tr = node.getInput(bias_idx);
    auto sum_tr = node.getInput(sum_idx);

    // Check layout.
    if (layout_dict.find(data_layout) == layout_dict.end()) {
      LOG(FATAL) << "Unsupported data layout for deconv: " << data_layout;
    }

    if (layout_dict.find(kernel_layout) == layout_dict.end()) {
      layout_dict.insert({kernel_layout, tag::any});
      LOG(WARNING) << "Unregistered kernel layout for deconv: " << kernel_layout
                   << ", transfer to tag::any";
    }

    // permute corresponding with provided layouts
    auto data_permutation = utils::permutation(data_layout);
    auto kernel_permutation = utils::permutation(kernel_layout);
    auto data_reshape = utils::reshape(data_tr.dims(), data_layout);
    auto kernel_reshape = utils::reshape(kernel_tr.dims(), kernel_layout);

    data_tr = data_tr.permute(data_permutation).reshape(data_reshape);
    kernel_tr = kernel_tr.permute(kernel_permutation).reshape(kernel_reshape);
    sum_tr = sum_tr.permute(data_permutation).reshape(data_reshape);
    output_tr = output_tr.permute(data_permutation);

    // TODO(@apeskov): temp WA. while codegen is not able to guarantee 1D format of bias data
    bias_tr = bias_tr.squeeze();

    // Group weight format
    if (groups > 1) {
      LOG(FATAL) << "Not checked 1";
      auto k_dims = kernel_tr.dims();  // OIHW -> GOIHW
      k_dims[0] /= groups;
      k_dims.insert(k_dims.begin(), groups);
      kernel_tr = kernel_tr.reshape(k_dims);
    }

    // Attributes setting
    dnnl::primitive_attr attr;
    ParsingOpName(node.GetOpName(), attr);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    /*if (activation[0] != "none") {
      auto a_type = utils::convert2dnnl_activation(activation[0]);
      auto a_scale = node.getInput(std::stoi(activation[1])).getConstScalarData<float>();
      auto a_alfa = node.getInput(std::stoi(activation[2])).getConstScalarData<float>();
      auto a_beta = node.getInput(std::stoi(activation[3])).getConstScalarData<float>();

      auto ops = attr.get_post_ops();
      ops.append_eltwise(a_scale, a_type, a_alfa, a_beta);
      attr.set_post_ops(ops);
    }*/

    dnnl::memory::dims strides_dims = Transform2Dims(strides);
    dnnl::memory::dims dilates_dims = Transform2Dims(dilation, true);
    dnnl::memory::dims padding_l_dims = Transform2Dims(padding_l);
    dnnl::memory::dims padding_r_dims = Transform2Dims(padding_r);

    // Conv description
    auto deconv_d = dnnl::deconvolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct,
        data_tr.layout(layout_dict[data_layout]).desc(),
        kernel_tr.layout(layout_dict[kernel_layout]).desc(),
        bias_tr.layoutAny().desc(), output_tr.layoutAny().desc(),
        strides_dims, dilates_dims, padding_l_dims, padding_r_dims);
    auto deconv_pd = dnnl::deconvolution_forward::primitive_desc(deconv_d, attr, engine_);
    auto deconv = dnnl::deconvolution_forward(deconv_pd);

    // Specify proper layouts
    data_tr = data_tr.requestLayout(deconv_pd.src_desc());
    kernel_tr = kernel_tr.requestLayout(deconv_pd.weights_desc());
    output_tr = output_tr.requestLayout(deconv_pd.dst_desc());
    bias_tr = bias_tr.requestLayout(deconv_pd.bias_desc());

    auto scratchpad_tr = node.makeScratchpad(deconv_pd.scratchpad_desc());

    // Inplace request for deconv+sum pattern. Match input with dst tensor
    auto submit_attr =
        sum_tr ? SubmitAttr{SubmitAttr::ZeroCopyRequest, sum_tr, DNNL_ARG_DST} : SubmitAttr{};

    // Register prim to execute
    submit(deconv,
           {{DNNL_ARG_SRC, data_tr},
            {DNNL_ARG_WEIGHTS, kernel_tr},
            {DNNL_ARG_BIAS, bias_tr},
            {DNNL_ARG_SCRATCHPAD, scratchpad_tr},
            {DNNL_ARG_DST, output_tr}},
           submit_attr);
  }

  void Dense(const size_t& nid) {
    auto node = NodeHelper{nid, g_explorer_};
    auto op_name = node.GetOpName();

    auto src_tr = node.getInput(0);
    auto wgh_tr = node.getInput(1);
    auto dst_tr = node.getOutput(0);

    auto activation = node.getAttr<std::vector<std::string>>("activation", {"none"});
    auto bias_idx = node.getAttr<int>("bias_idx", {"-1"});
    auto sum_idx = node.getAttr<int>("sum_idx", {"-1"});
    auto sum_scl_idx = node.getAttr<int>("sum_scl_idx", {"-1"});
    auto o_scl_idx = node.getAttr<int>("o_scl_idx", {"-1"});
    auto dst_zp_idx = node.getAttr<int>("dst_zp_idx", {"-1"});

    // may be empty in case if '-1'
    auto bias_tr = node.getInput(bias_idx);
    auto sum_tr = node.getInput(sum_idx);
    auto sum_scl_tr = node.getInput(sum_scl_idx);
    auto o_scl_tr = node.getInput(o_scl_idx);
    auto dst_zp_tr = node.getInput(dst_zp_idx);

    // TODO(@apeskov): temp WA. while codegen is not able to guarantee 1D format of bias data
    bias_tr = bias_tr.squeeze();

    // Attributes setting
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    ParsingOpName(op_name, attr);

    ICHECK(!dst_zp_tr) << "DNNL doesn't support input zero point for optimized primitives."
                          "Should be merged into bias";

    if (o_scl_tr) {
      LOG(FATAL) << "Unsupported op: o_scl_tr";
      ICHECK(o_scl_tr.isConstant());
      auto data = o_scl_tr.getConstDataLikeVec<float>();
      attr.set_output_scales(data.size() == 1 ? 0 : (1 << 1), data);
    }

    if (activation[0] != "none") {
      LOG(FATAL) << "Unsupported op: activation";
      auto a_type = utils::convert2dnnl_activation(activation[0]);
      auto a_scale = node.getInput(std::stoi(activation[1])).getConstScalarData<float>();
      auto a_alfa = node.getInput(std::stoi(activation[2])).getConstScalarData<float>();
      auto a_beta = node.getInput(std::stoi(activation[3])).getConstScalarData<float>();

      auto ops = attr.get_post_ops();
      ops.append_eltwise(a_scale, a_type, a_alfa, a_beta);
      attr.set_post_ops(ops);
    }

    if (sum_scl_tr) {
      LOG(FATAL) << "Unsupported op: sum_scl_tr";
      auto scl = sum_scl_tr.getConstScalarData<float>();
      auto ops = attr.get_post_ops();
      ops.append_sum(scl);
      attr.set_post_ops(ops);
    }

    // Dense description.
    auto dense_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, src_tr.layoutAny().desc(), wgh_tr.layoutAny().desc(),
        bias_tr.layoutAny().desc(), dst_tr.layoutAny().desc());
    auto dense_pd = dnnl::inner_product_forward::primitive_desc(dense_d, attr, engine_);
    auto dense = dnnl::inner_product_forward(dense_pd);

    // Select proper layout
    src_tr = src_tr.requestLayout(dense_pd.src_desc());
    wgh_tr = wgh_tr.requestLayout(dense_pd.weights_desc());
    dst_tr = dst_tr.requestLayout(dense_pd.dst_desc());

    auto scratch_pad_d = node.makeScratchpad(dense_pd.scratchpad_desc());

    // Inplace request for conv+sum pattern. Match input with dst tensor
    auto submit_attr = sum_tr 
                     ? SubmitAttr{SubmitAttr::ZeroCopyRequest, sum_tr, DNNL_ARG_DST}
                     : SubmitAttr{};

    submit(dense, {{DNNL_ARG_SRC, src_tr},
            {DNNL_ARG_WEIGHTS, wgh_tr},
            {DNNL_ARG_BIAS, bias_tr},
            {DNNL_ARG_SCRATCHPAD, scratch_pad_d},
            {DNNL_ARG_DST, dst_tr}},
           submit_attr);
  }

  void BatchNorm(const size_t& nid) {
    auto node = NodeHelper{nid, g_explorer_};

    auto src_tr = node.getInput(0);
    auto gamma_tr = node.getInput(1);
    auto beta_tr = node.getInput(2);
    auto mean_tr = node.getInput(3);
    auto variance_tr = node.getInput(4);
    auto dst_tr = node.getOutput(0);

    auto axis = node.getAttr<int>("axis");
    auto epsilon = node.getAttr<float>("epsilon");
    auto center = node.getAttr<bool>("center");
    auto scale = node.getAttr<bool>("scale");

    // TODO(@apeskov): Add support of all type of axis, center and scale args
    ICHECK(axis == 1);
    ICHECK(center);
    ICHECK(scale);

    // TODO(@apeskov): Should it use "any" layout to select proper one?
    auto bn_d = dnnl::batch_normalization_forward::desc(
        dnnl::prop_kind::forward_inference, dst_tr.desc(), epsilon,
        dnnl::normalization_flags::use_global_stats | dnnl::normalization_flags::use_scale_shift);
    auto bn_pd = dnnl::batch_normalization_forward::primitive_desc(bn_d, engine_);
    auto bn = dnnl::batch_normalization_forward(bn_pd);

    src_tr = src_tr.requestLayout(bn_pd.src_desc());
    dst_tr = dst_tr.requestLayout(bn_pd.dst_desc());
    mean_tr = mean_tr.requestLayout(bn_pd.mean_desc());
    variance_tr = variance_tr.requestLayout(bn_pd.variance_desc());

    // TODO(@apeskov): DNNL v2.5 and late has API for separate scale and shift
    //                 it will eliminate requirements of data copy.
    // Prepare concatenated Scale and Shift tensor
    auto scale_shift_tr = node.makeTemp(bn_pd.weights_desc(), g_explorer_.generateUniqueEID());
    auto sc_sh_dims = scale_shift_tr.dims();
    ICHECK(sc_sh_dims.size() == 2);
    ICHECK(sc_sh_dims[0] == 2);
    sc_sh_dims[0] /= 2;
    auto scale_tr = scale_shift_tr.crop(sc_sh_dims, {0, 0}).squeeze();
    auto shift_tr = scale_shift_tr.crop(sc_sh_dims, {1, 0}).squeeze();

    auto register_copy = [this](const TensorRequisite& src, const TensorRequisite& dst) {
      dnnl::reorder::primitive_desc copy_pd(engine_, src.desc(), engine_, dst.desc());
      submit(dnnl::reorder(copy_pd), {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
    };

    register_copy(gamma_tr, scale_tr);
    register_copy(beta_tr, shift_tr);

    submit(bn, {{DNNL_ARG_SRC, src_tr},
                {DNNL_ARG_DST, dst_tr},
                {DNNL_ARG_SCALE_SHIFT, scale_shift_tr},
                {DNNL_ARG_MEAN, mean_tr},
                {DNNL_ARG_VARIANCE, variance_tr}});
  }

  void Pooling(const size_t& nid, dnnl::algorithm algo) {
    auto node = NodeHelper{nid, g_explorer_};

    // Fix position inputs
    auto data_tr = node.getInput(0);
    auto output_tr = node.getOutput(0);

    // Parse general pool attributes
    auto strides = node.getAttr<std::vector<int>>("strides");
    auto padding = node.getAttr<std::vector<int>>("padding");
    auto dilation = node.getAttr<std::vector<int>>("dilation");
    auto kernel = node.getAttr<std::vector<int>>("pool_size");

    decltype(padding) padding_l(padding.begin(), padding.begin() + padding.size() / 2);
    decltype(padding) padding_r(padding.end() - padding.size() / 2, padding.end());

    auto layout = node.getAttr<std::string>("layout");

    // Check layout.
    if (layout_dict.find(layout) == layout_dict.end()) {
      LOG(FATAL) << "Unsupported layout for pooling: " << layout;
    }

    // Attributes related to AvgPool
    if (algo == dnnl::algorithm::pooling_avg) {
      int int_countpad = node.getAttr<int>("count_include_pad");
      bool count_include_pad = int_countpad != 0 ? true : false;
      algo = count_include_pad ? dnnl::algorithm::pooling_avg_include_padding
                               : dnnl::algorithm::pooling_avg_exclude_padding;
    }

    dnnl::memory::dims kernel_dims = Transform2Dims(kernel);
    dnnl::memory::dims strides_dims = Transform2Dims(strides);
    dnnl::memory::dims dilates_dims = Transform2Dims(dilation, true);
    dnnl::memory::dims padding_l_dims = Transform2Dims(padding_l);
    dnnl::memory::dims padding_r_dims = Transform2Dims(padding_r);

    // Pooling description.
    auto pool_desc = dnnl::pooling_forward::desc(dnnl::prop_kind::forward_inference, algo,
                                                 data_tr.layout(layout_dict[layout]).desc(),
                                                 output_tr.layoutAny().desc(), strides_dims,
                                                 kernel_dims, padding_l_dims, padding_r_dims);

    auto pool_pd = dnnl::pooling_forward::primitive_desc(pool_desc, engine_, true);
    auto pool = dnnl::pooling_forward(pool_pd);
    
    // Specify proper layouts
    data_tr = data_tr.requestLayout(pool_pd.src_desc());
    output_tr = output_tr.requestLayout(pool_pd.dst_desc());

    submit(pool, {{DNNL_ARG_SRC, data_tr}, {DNNL_ARG_DST, output_tr}});
  }

  void Eltwise(const size_t& nid) {
    auto node = NodeHelper{nid, g_explorer_};
    auto op_name = node.GetOpName();
    auto algo = elt_name2algo[op_name];

    auto src_tr = node.getInput(0);
    auto dst_tr = node.getOutput(0);
    ICHECK(src_tr.dims() == dst_tr.dims());
    // Eltwise op required same layout for src/dst
    src_tr = src_tr.requestLayout(dst_tr.desc());

    float alpha = 0., beta = 0.;
    if (op_name == "clip") {
      alpha = node.getAttr<float>("a_min");
      beta = node.getAttr<float>("a_max");
    } else if (op_name == "nn.leaky_relu") {
      alpha = node.getAttr<float>("alpha");
    }

    auto eltwise_d = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
                                                 algo, dst_tr.desc(), alpha, beta);
    auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(eltwise_d, engine_);
    auto eltwise = dnnl::eltwise_forward(eltwise_pd);

    submit(eltwise, {{DNNL_ARG_SRC, src_tr}, {DNNL_ARG_DST, dst_tr}});
  }

  void Softmax(const size_t& nid) {
    auto node = NodeHelper{nid, g_explorer_};

    auto src_tr = node.getInput(0);
    auto dst_tr = node.getOutput(0);

    auto axis = node.getAttr<int>("axis");
    if (axis < 0) {
      axis = src_tr.dims().size() + axis;
    }

    // Support of softmax_v2 appears since version 2.6 in oneDNN
#if ((DNNL_VERSION_MAJOR == 2) && (DNNL_VERSION_MINOR >= 6)) || (DNNL_VERSION_MAJOR > 2)
    dnnl::primitive_attr attr;
    auto q_scale      = node.getInputByAttrName("q_scale_idx");
    auto q_zero_point = node.getInputByAttrName("q_zp_idx");
    if (q_scale && q_zero_point) {
      auto q_scale_const = q_scale.getConstScalarData<float>();
      auto q_zp_const = q_zero_point.getConstScalarData<int32_t>();
      // zp != 0 is unsupported case
      ICHECK_EQ(q_zp_const, 0);
      float scale = 1.0f / q_scale_const;
      attr.set_output_scales(0, {scale});
    }
    auto softmax_d = dnnl::softmax_v2_forward::desc(dnnl::prop_kind::forward_inference,
                                                    dnnl::algorithm::softmax_accurate,
                                                    src_tr.desc(), dst_tr.desc(), axis);
    auto softmax_pd = dnnl::softmax_v2_forward::primitive_desc(softmax_d, attr, engine_);
    auto softmax = dnnl::softmax_v2_forward(softmax_pd);
#else
    // Softmax description.
    auto softmax_d = dnnl::softmax_forward::desc(dnnl::prop_kind::forward_inference,
                                                 src_tr.desc(), axis);
    auto softmax_pd = dnnl::softmax_forward::primitive_desc(softmax_d, engine_);
    auto softmax = dnnl::softmax_forward(softmax_pd);
#endif

    src_tr = src_tr.requestLayout(softmax_pd.src_desc());
    dst_tr = dst_tr.requestLayout(softmax_pd.dst_desc());

    // TODO: support in-place calculation.
    submit(softmax, {{DNNL_ARG_SRC, src_tr}, {DNNL_ARG_DST, dst_tr}});
  }

  void Binary(const size_t& nid, dnnl::algorithm algo) {
    auto node = NodeHelper{nid, g_explorer_};

    auto lhs_tr = node.getInput(0);
    auto rhs_tr = node.getInput(1);
    auto out_tr = node.getOutput(0);

    lhs_tr = lhs_tr.broadcast(out_tr.dims());
    rhs_tr = rhs_tr.broadcast(out_tr.dims());

    // Any layouts cannot be used for binary prim
    auto binary_d = dnnl::binary::desc(algo, lhs_tr.desc(), rhs_tr.desc(),
                                       out_tr.desc());
    auto binary_pd = dnnl::binary::primitive_desc(binary_d, engine_);
    auto binary = dnnl::binary(binary_pd);

    // Request proper layouts
    lhs_tr = lhs_tr.requestLayout(binary_pd.src0_desc());
    rhs_tr = rhs_tr.requestLayout(binary_pd.src1_desc());
    out_tr = out_tr.requestLayout(binary_pd.dst_desc());

    submit(binary, {{DNNL_ARG_SRC_0, lhs_tr}, {DNNL_ARG_SRC_1, rhs_tr},
                    {DNNL_ARG_DST, out_tr}});
  }

  void DenseDequantize(const size_t& nid) {
    auto node = NodeHelper{nid, g_explorer_};

    auto src_tr = node.getInput(0);
    auto wgh_tr = node.getInput(1);

    auto dst_tr = node.getOutput(0);
    // Convert 3D -> 2D tensor
    ICHECK(dst_tr.dims()[0] == 1);
    auto dst_tr_2d = dst_tr.squeeze({0});

    auto activation = node.getAttr<std::vector<std::string>>("activation", {"none"});
    auto src_zero_point = node.getInputByAttrName("src_zp_idx");
    auto dst_zero_point = node.getInputByAttrName("dst_zp_idx");
    auto deq_scale      = node.getInputByAttrName("deq_scale_idx");
    auto bias_tr        = node.getInputByAttrName("bias_idx");
    auto q_scale        = node.getInputByAttrName("q_scale_idx");
    auto q_zero_point   = node.getInputByAttrName("q_zp_idx");

    auto src_zp_const = src_zero_point.getConstScalarData<int32_t>();
    auto dst_zp_const = dst_zero_point.getConstScalarData<int32_t>();
    auto deq_scl_const = deq_scale.getConstScalarData<float>();

    dnnl::primitive_attr attr;
    attr.set_output_scales(0, { deq_scl_const});
    attr.set_zero_points(DNNL_ARG_SRC, 0, {src_zp_const});
    attr.set_zero_points(DNNL_ARG_DST, 0, {dst_zp_const});

    if (activation[0] != "none") {
      auto a_type = utils::convert2dnnl_activation(activation[0]);
      auto a_scale = node.getInput(std::stoi(activation[1])).getConstScalarData<float>();
      auto a_alfa = node.getInput(std::stoi(activation[2])).getConstScalarData<float>();
      auto a_beta = node.getInput(std::stoi(activation[3])).getConstScalarData<float>();

      auto ops = attr.get_post_ops();
      ops.append_eltwise(a_scale, a_type, a_alfa, a_beta);
      attr.set_post_ops(ops);
    }

    if (q_scale && q_zero_point) {
      auto q_scale_const = q_scale.getConstScalarData<float>();
      auto q_zp_const = q_zero_point.getConstScalarData<int32_t>();
      auto ops = attr.get_post_ops();
      float alpha = 1.0f / q_scale_const;
      ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, alpha, (float)q_zp_const);
      attr.set_post_ops(ops);
    }

    // Dense description.
    auto dense_d = (bias_tr) ? dnnl::inner_product_forward::desc(
                                    dnnl::prop_kind::forward_inference, src_tr.layoutAny().desc(),
                                    wgh_tr.layoutAny().desc(), bias_tr.layoutAny().desc(),
                                    dst_tr_2d.layoutAny().desc())
                             : dnnl::inner_product_forward::desc(
                                    dnnl::prop_kind::forward_inference, src_tr.layoutAny().desc(),
                                    wgh_tr.layoutAny().desc(), dst_tr_2d.layoutAny().desc());
    auto dense_pd = dnnl::inner_product_forward::primitive_desc(dense_d, attr, engine_);
    auto dense = dnnl::inner_product_forward(dense_pd);

    // Select proper layout
    src_tr = src_tr.requestLayout(dense_pd.src_desc());
    wgh_tr = wgh_tr.requestLayout(dense_pd.weights_desc());
    dst_tr = dst_tr_2d.requestLayout(dense_pd.dst_desc());

    std::unordered_map<int, TensorRequisite> tr_args =
        {{DNNL_ARG_SRC, src_tr}, {DNNL_ARG_WEIGHTS, wgh_tr}, {DNNL_ARG_DST, dst_tr}};
    if (bias_tr)
      tr_args[DNNL_ARG_BIAS] = bias_tr.requestLayout(dense_pd.bias_desc());

    submit(dense, tr_args);
  }

  void DenseAddRequantize(const size_t& nid) {
    auto node = NodeHelper{nid, g_explorer_};

    auto src_tr = node.getInput(0);
    auto wgh_tr = node.getInput(1);
    auto bias_tr = node.getInput(2);

    ICHECK_EQ(src_tr.dims().size(), 2) << "input shape should have 2 dimentions.";
    ICHECK_EQ(wgh_tr.dims().size(), 2) << "weights shape should have 2 dimentions.";

    auto dst_tr = node.getOutput(0);

    // Convert -> 2D tensor
    dnnl::memory::dims new_out_dims = {src_tr.dims()[0], wgh_tr.dims()[0]};
    auto dst_tr_2d = dst_tr.reshape(new_out_dims);

    ICHECK_EQ(dst_tr_2d.dims().size(), 2) << "output shape should have 2 dimentions.";

    auto o_scl = node.getInputByAttrName("o_scl_idx");
    ICHECK(o_scl.isConstant());
    auto data = o_scl.getConstDataLikeVec<float>();

    dnnl::primitive_attr attr;
    attr.set_output_scales(data.size() == 1 ? 0 : (1 << 1), data);

    // Dense description.
    auto dense_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference,
        src_tr.layoutAny().desc(), wgh_tr.layoutAny().desc(),
        bias_tr.layoutAny().desc(), dst_tr_2d.layoutAny().desc());
    auto dense_pd = dnnl::inner_product_forward::primitive_desc(dense_d, attr, engine_);
    auto dense = dnnl::inner_product_forward(dense_pd);

    // Select proper layout
    src_tr = src_tr.requestLayout(dense_pd.src_desc());
    wgh_tr = wgh_tr.requestLayout(dense_pd.weights_desc());
    bias_tr = bias_tr.requestLayout(dense_pd.bias_desc());
    dst_tr = dst_tr_2d.requestLayout(dense_pd.dst_desc());

    submit(dense, {{DNNL_ARG_SRC, src_tr},
                   {DNNL_ARG_WEIGHTS, wgh_tr},
                   {DNNL_ARG_BIAS, bias_tr},
                   {DNNL_ARG_DST, dst_tr}});
  }

  void QnnMatmul(const size_t& nid) {
    auto node = NodeHelper{nid, g_explorer_};

    auto src_tr = node.getInput(0);
    auto wgh_tr = node.getInput(1);

    ICHECK_EQ(src_tr.dims().size(), 3) << "input shape should have 3 dimentions.";
    ICHECK_EQ(wgh_tr.dims().size(), 3) << "weights shape should have 3 dimentions.";

    auto dst_tr = node.getOutput(0);
    // Convert 4D -> 3D tensor
    if (dst_tr.dims().size() == 4) {
      ICHECK(dst_tr.dims()[0] == 1);
      dst_tr = dst_tr.squeeze({0});
    }
    ICHECK_EQ(dst_tr.dims().size(), 3) << "output shape should have 3 dimentions.";

    auto o_scale = node.getInputByAttrName("o_scale_idx");
    auto o_scl_const = o_scale.getConstScalarData<float>();

    dnnl::primitive_attr attr;
    attr.set_output_scales(0, { o_scl_const});

    // Matmul description.
    auto matmul_d = dnnl::matmul::desc(src_tr.layoutAny().desc(),
        wgh_tr.layoutAny().desc(), dst_tr.layoutAny().desc());
    auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, engine_);
    auto matmul = dnnl::matmul(matmul_pd);

    // Select proper layout
    src_tr = src_tr.requestLayout(matmul_pd.src_desc());
    wgh_tr = wgh_tr.requestLayout(matmul_pd.weights_desc());
    dst_tr = dst_tr.requestLayout(matmul_pd.dst_desc());

    submit(matmul, {{DNNL_ARG_SRC, src_tr},
                    {DNNL_ARG_WEIGHTS, wgh_tr},
                    {DNNL_ARG_DST, dst_tr}});
  }

  void LayerNorm(const size_t& nid) {
    auto node = NodeHelper{nid, g_explorer_};

    auto data_tr = node.getInput(0);
    auto dst_tr = node.getOutput(0);

    auto e_tr = node.getInputByAttrName("epsilon_idx");
    auto epsilon = e_tr.getConstScalarData<float>();

    auto scale_tr = node.getInputByAttrName("scale_idx");
    auto shift_tr = node.getInputByAttrName("shift_idx");

    // Just for check
    ICHECK_EQ(node.getAttr<int>("axis"), -1);

    auto layer_norm_desc = dnnl::layer_normalization_forward::desc(
            dnnl::prop_kind::forward_inference, data_tr.desc(), epsilon,
            dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift);
    auto l_norm_pd = dnnl::layer_normalization_forward::primitive_desc(layer_norm_desc, engine_);
    auto l_norm = dnnl::layer_normalization_forward(l_norm_pd);

    data_tr = data_tr.requestLayout(l_norm_pd.src_desc());
    dst_tr = dst_tr.requestLayout(l_norm_pd.dst_desc());

    auto mean_tr = node.makeTemp(l_norm_pd.mean_desc(),
                                 g_explorer_.generateUniqueEID());
    auto variance_tr = node.makeTemp(l_norm_pd.variance_desc(),
                                     g_explorer_.generateUniqueEID());

    submit(l_norm, {{DNNL_ARG_SRC, data_tr},
                    {DNNL_ARG_DST, dst_tr},
                    {DNNL_ARG_SCALE, scale_tr},
                    {DNNL_ARG_SHIFT, shift_tr},
                    {DNNL_ARG_MEAN, mean_tr},
                    {DNNL_ARG_VARIANCE, variance_tr}});
  }

  void QnnSoftmax(const size_t& nid) {
    auto node = NodeHelper{nid, g_explorer_};

    auto src_tr = node.getInput(0);
    auto dst_tr = node.getOutput(0);

    auto axis = node.getAttr<int>("axis");

    // Support of softmax_v2 appears since version 2.6 in oneDNN
#if ((DNNL_VERSION_MAJOR == 2) && (DNNL_VERSION_MINOR >= 6)) || (DNNL_VERSION_MAJOR > 2)
    dnnl::primitive_attr attr;
    auto q_scale      = node.getInputByAttrName("q_scale_idx");
    auto q_zero_point = node.getInputByAttrName("q_zp_idx");
    if (q_scale && q_zero_point) {
      auto q_scale_const = q_scale.getConstScalarData<float>();
      auto q_zp_const = q_zero_point.getConstScalarData<int32_t>();
      // zp != 0 is unsupported case
      ICHECK_EQ(q_zp_const, 0);
      float scale = 1.0f / q_scale_const;
      attr.set_output_scales(0, {scale});
    }
    auto softmax_d = dnnl::softmax_v2_forward::desc(dnnl::prop_kind::forward_inference,
                                                    dnnl::algorithm::softmax_accurate,
                                                    src_tr.desc(), dst_tr.desc(), axis);
    auto softmax_pd = dnnl::softmax_v2_forward::primitive_desc(softmax_d, attr, engine_);
    auto softmax = dnnl::softmax_v2_forward(softmax_pd);
#else
    // Softmax description.
    auto softmax_d = dnnl::softmax_forward::desc(dnnl::prop_kind::forward_inference,
                                                 src_tr.desc(), axis);
    auto softmax_pd = dnnl::softmax_forward::primitive_desc(softmax_d, engine_);
    auto softmax = dnnl::softmax_forward(softmax_pd);
#endif

    src_tr = src_tr.requestLayout(softmax_pd.src_desc());
    dst_tr = dst_tr.requestLayout(softmax_pd.dst_desc());

    // TODO: support in-place calculation.
    submit(softmax, {{DNNL_ARG_SRC, src_tr}, {DNNL_ARG_DST, dst_tr}});
  }

  // Generate DNNL memory description and infer the data layout by the given shape.
  inline dnnl::memory::desc GenDNNLMemDescByShape(const dnnl::memory::dims& shape, dt dtype) {
    dnnl::memory::desc data_md;
    switch (shape.size()) {
      case 2:
        data_md = dnnl::memory::desc({shape, dtype, tag::ab});
        break;
      case 3:
        data_md = dnnl::memory::desc({shape, dtype, tag::abc});
        break;
      case 4:
        data_md = dnnl::memory::desc({shape, dtype, tag::abcd});
        break;
      case 5:
        data_md = dnnl::memory::desc({shape, dtype, tag::abcde});
        break;
      default:
        LOG(FATAL) << "Unsupported data shape dimension: " << shape.size();
        break;
    }
    return data_md;
  }

  /** The dnnl engine. */
  dnnl::engine engine_;
  /** The dnnl stream. */
  dnnl::stream stream_;
  /** Tensor registry which manages all real dnnl memory objects */
  TensorRegistry tensor_registry_;
  /** The network layers that are represented as dnnl primitives plus there args. */
  TensorRegistry::ActionQue net_;
  /** Utility object */
  GraphExplorer g_explorer_;
};

runtime::Module DNNLJSONRuntimeCreate(const String& symbol_name, const String& graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<DNNLJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.DNNLJSONRuntimeCreate").set_body_typed(DNNLJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_dnnl_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<DNNLJSONRuntime>);

TVM_REGISTER_GLOBAL("runtime.module.dnnl_version").set_body_typed(DNNLJSONRuntime::get_version);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
