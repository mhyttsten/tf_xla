/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_KERNELS_MKL_MKL_MATMUL_OPS_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_MKL_MKL_MATMUL_OPS_COMMON_H_
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh {
public:
   std::string _s;
   int _indent = 0;
   std::string _functionName;
   bool _isFile = false;
   std::string _fileName;
   std::string _envMHIndent;
   int _lineNumber;
   bool _filtered = false;
   bool _otherThread = false;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
      _functionName = functionName;
      _lineNumber = lineNumber;

      // Check if tracing is enabled
      const char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }
      // Should we trace of filter?
      const char* env_filter = std::getenv("MHTRACER_FILTER");
      if (env_filter != nullptr) {
         std::string sfilter = std::string(env_filter);
         std::string sLineNumber = std::to_string(lineNumber);
         while (true) {
            std::size_t ioE = sfilter.find(";");
            if (sfilter.size() == 0) {
               break;
            }
            std::string cfs = sfilter.substr(0, ioE);
            std::size_t ioFileName = cfs.find("|");
            std::string fFileName  = cfs.substr(0, ioFileName);
            std::size_t ioFunctionName = cfs.find("|", ioFileName+1);
            std::string fFunctionName  = cfs.substr(ioFileName+1, ioFunctionName-ioFileName-1);
            std::string fLineNumber    = cfs.substr(ioFunctionName+1, cfs.size()-ioFunctionName-1);

            if (  (fFileName == "*" || fFileName == fileName)
               && (fFunctionName == "*" || fFunctionName == functionName)
               && (fLineNumber == "*" || fLineNumber == sLineNumber)) {
              _filtered = true;
               return;
            }

            if (ioE == std::string::npos) {
               sfilter = "";
            } else {
               sfilter = sfilter.substr(ioE+1, sfilter.size()-ioE-1);
            }
         }
      }

      // Create log string
      std::string ostr;

      // Assign indent spaces (tied to PID and TID)
      pid_t pid = getpid();
      std::thread::id tid = std::this_thread::get_id();
      std::stringstream pid_dash_tid_ss;
      pid_dash_tid_ss << pid << "-" << tid;
      std::string pid_dash_tid_str = pid_dash_tid_ss.str();
      _envMHIndent = "MHTRACER_INDENT_";
      char* env_indent = std::getenv(_envMHIndent.c_str());
      if (env_indent != nullptr) {
         _indent = std::stoi(std::string(env_indent));
      }
      _s.assign(_indent, ' ');

      // Check that reporting matches pid/tid
      const char* env_pid_dash_tid = std::getenv("MHTRACER_PID_DASH_TID");
      if (env_pid_dash_tid != nullptr) {
         std::string env_pid_dash_tid_str(env_pid_dash_tid);
         if (env_pid_dash_tid_str != pid_dash_tid_str) {
            _otherThread = true;
         }
      }
      else {  // PID-THREAD not set, set it for the first time (starter thread)
         setenv("MHTRACER_PID_DASH_TID", pid_dash_tid_str.c_str(), 1);
      }

      std::string paramStr;
      for (int i=0; i < params.size(); i++) {
         auto e = params[i];
         while (e.find("\n") != std::string::npos) {
            size_t pos = e.find("\n");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<NL>");
         }
         while (e.find("[") != std::string::npos) {
            size_t pos = e.find("[");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<LB>");
         }
         while (e.find("]") != std::string::npos) {
            size_t pos = e.find("]");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<RB>");
         }
         paramStr += e;
         if ((i+1) < params.size()) {
            paramStr += ", ";
         }
      }

      const char* env_dont_print_pid_dash_tid = std::getenv("MHTRACER_DONT_PRINT_PID_DASH_TID");
      if (env_dont_print_pid_dash_tid != nullptr) {
         pid_dash_tid_str = "";
      }
      if (_otherThread) {
         functionName = "MHOT_" + functionName;
      }
      ostr += _s + functionName + 
         + " [1]"
         + " [" + prefix + "]"
         + " [" + paramStr + "]"
         + " [" + pid_dash_tid_str + " "
         +    std::to_string(lineNumber)
         +    " @ " + fileName + "]\n";

      // Log to file
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_USEFILE") != std::string::npos) {
         _isFile = true;
         _fileName = "/tmp/mhtracer_" + pid_dash_tid_str + ".log";
         std::ofstream os;
         os.open(_fileName, std::ofstream::out | std::ofstream::app);
         os << ostr << "";
         os.close();
      }
      // Log to stdout
      else {
         std::cout << ostr << "";
      }

      // Increment indent spaces
      if (_otherThread) {
         return;
      }
      _indent += 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh() {
      // Check if tracing is enabled
      char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }

      // Don't update indent if tracing was filtered or from another thread
      if (_filtered || _otherThread) {
         return;
      }

      _indent -= 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
};


#ifdef INTEL_MKL
#include <memory>
#include <string>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "dnnl.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/onednn_env_vars.h"
#ifdef DNNL_AARCH64_USE_ACL
#include "tensorflow/core/platform/mutex.h"
#endif

using dnnl::inner_product_forward;
using dnnl::primitive_attr;
using dnnl::prop_kind;
using dnnl::stream;

namespace tensorflow {
static Eigen::internal::CacheSizes cache_sizes = Eigen::internal::CacheSizes();

typedef Eigen::ThreadPoolDevice CPUDevice;
inline bool ExecuteSingleThreadedGemm(int m, int n, int k, int bytes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "ExecuteSingleThreadedGemm");

  // Ideally we would like to determine blocking and then come up with
  // a heuristic but what we are targeting are very small models whose
  // total size is < x*L2. So we will do this simple calculation
  // to determine if the matrix multiplication should be run on a single thread.
  // TODO(Intel-tf): this needs to be vastly improved, perhaps at a lower level
  // than the integration.
  ptrdiff_t l2_size = cache_sizes.m_l2;
  constexpr float kHeuristicMultiplier = 1.01;
  const float mul_size = bytes * (m * n + k * (m + n));
  const float l2_heur = l2_size * kHeuristicMultiplier;
  return mul_size < l2_heur;
}

// This structure aggregates multiple inputs to MklDnnMatMul* methods.
struct MklDnnMatMulFwdParams {
  memory::dims src_dims;
  memory::dims weight_dims;
  memory::dims bias_dims;
  memory::dims dst_dims;
  memory::format_tag src_format;
  memory::format_tag weight_format;
  memory::format_tag dst_format;
  string dtypes = string("");
  bool const_weight;
#ifdef DNNL_AARCH64_USE_ACL
  void* weight_address = nullptr;
#endif
  struct PostOpParam {
    string name;
    std::vector<float> param;
  };
  std::vector<PostOpParam> post_op_params;

  MklDnnMatMulFwdParams(
      memory::dims src_dims, memory::dims weight_dims, memory::dims bias_dims,
      memory::dims dst_dims,
      memory::format_tag src_format = memory::format_tag::any,
      memory::format_tag weight_format = memory::format_tag::any,
      memory::format_tag dst_format = memory::format_tag::any,
      bool const_weight = false)
      : src_dims(src_dims),
        weight_dims(weight_dims),
        bias_dims(bias_dims),
        dst_dims(dst_dims),
        src_format(src_format),
        weight_format(weight_format),
        dst_format(dst_format),
        const_weight(const_weight) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_1(mht_1_v, 263, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "MklDnnMatMulFwdParams");
}
};

// With quantization, input, weight, bias, and output can have different types.
// So we use different template parameters for each type.
// TODO(intel-tf): The template type "T" is currently used to match the
// templatized class MklPrimitiveFactory (tensorflow/core/util/mkl_util.h).
// In the future, with the removal of "T" from MklPrimitiveFactory, this class
// needs to drop "T".
template <typename T, typename Tinput, typename Tweight, typename Tbias,
          typename Toutput>
class MklDnnMatMulFwdPrimitive : public MklPrimitive {
 public:
  explicit MklDnnMatMulFwdPrimitive(
      const MklDnnMatMulFwdParams& matmulFwdParams)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_2(mht_2_v, 281, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "MklDnnMatMulFwdPrimitive");

    // Create matmul primitive
    if (context_.matmul_fwd == nullptr) {
      Setup(matmulFwdParams);
    }
  }

  ~MklDnnMatMulFwdPrimitive() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_3(mht_3_v, 291, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "~MklDnnMatMulFwdPrimitive");
}

  dnnl::memory::desc GetScratchPadDesc() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_4(mht_4_v, 296, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "GetScratchPadDesc");

    return context_.fwd_pd->scratchpad_desc();
  }

  // Inner-product forward execute with bias:
  //  - src_data: input data buffer of src
  //  - weight_data: input data buffer of weight
  //  - bias_data: input data buffer of bias
  //  - dst_data: output data buffer of dst
  //  - sp_data: scratchpad data
  void Execute(const Tinput* src_data, const Tweight* weight_data,
               const Tbias* bias_data, Toutput* dst_data, void* sp_data,
               std::shared_ptr<stream> fwd_stream) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_5(mht_5_v, 311, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "Execute");

#ifdef DNNL_AARCH64_USE_ACL
    mutex_lock lock(primitive_execution_mu_);
#endif
#ifndef ENABLE_ONEDNN_OPENMP
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<Tinput*>(src_data)), *fwd_stream);
    context_.weight_mem->set_data_handle(
        static_cast<void*>(const_cast<Tweight*>(weight_data)), *fwd_stream);
    context_.bias_mem->set_data_handle(
        static_cast<void*>(const_cast<Tbias*>(bias_data)));
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data),
                                      *fwd_stream);
    context_.sp_mem->set_data_handle(sp_data, *fwd_stream);
#else
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<Tinput*>(src_data)));
    context_.weight_mem->set_data_handle(
        static_cast<void*>(const_cast<Tweight*>(weight_data)));
    context_.bias_mem->set_data_handle(
        static_cast<void*>(const_cast<Tbias*>(bias_data)));
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data));
    context_.sp_mem->set_data_handle(sp_data);
#endif  // !ENABLE_ONEDNN_OPENMP

    execute_primitives(context_.fwd_primitives, fwd_stream, context_.net_args);

    // After execution, set data handle back
    context_.src_mem->set_data_handle(DummyData);
    context_.weight_mem->set_data_handle(DummyData);
    context_.bias_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
  }

  std::shared_ptr<dnnl::inner_product_forward::primitive_desc>
  GetPrimitiveDesc() const {
    return context_.fwd_pd;
  }

 private:
  // Primitive reuse context for inner-product Fwd op
  struct MklDnnMatMulFwdContext {
    // oneDNN memory.
    std::shared_ptr<dnnl::memory> src_mem;
    std::shared_ptr<dnnl::memory> weight_mem;
    std::shared_ptr<dnnl::memory> bias_mem;
    std::shared_ptr<dnnl::memory> dst_mem;
    std::shared_ptr<dnnl::memory> sp_mem;

    // Descriptor and primitive-descriptor for forward inner-product.
    std::shared_ptr<dnnl::inner_product_forward::desc> fwd_desc;
    std::shared_ptr<dnnl::inner_product_forward::primitive_desc> fwd_pd;

    // Memory descriptors.
    std::shared_ptr<dnnl::memory::desc> src_md;
    std::shared_ptr<dnnl::memory::desc> weight_md;
    std::shared_ptr<dnnl::memory::desc> bias_md;
    std::shared_ptr<dnnl::memory::desc> dst_md;

    // Inner-product primitive.
    std::shared_ptr<dnnl::primitive> matmul_fwd;
    std::vector<dnnl::primitive> fwd_primitives;

    std::vector<std::unordered_map<int, memory>> net_args;

    MklDnnMatMulFwdContext()
        : src_mem(nullptr),
          weight_mem(nullptr),
          bias_mem(nullptr),
          dst_mem(nullptr),
          sp_mem(nullptr),
          fwd_desc(nullptr),
          fwd_pd(nullptr),
          src_md(nullptr),
          weight_md(nullptr),
          bias_md(nullptr),
          dst_md(nullptr),
          matmul_fwd(nullptr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_6(mht_6_v, 391, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "MklDnnMatMulFwdContext");
}
  };

  void Setup(const MklDnnMatMulFwdParams& matmul_fwd_params) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_7(mht_7_v, 397, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "Setup");

    // Create memory descriptors for inner-product data without specified
    // format.
    context_.src_md.reset(new memory::desc({matmul_fwd_params.src_dims},
                                           MklDnnType<Tinput>(),
                                           matmul_fwd_params.src_format));

    context_.weight_md.reset(new memory::desc({matmul_fwd_params.weight_dims},
                                              MklDnnType<Tweight>(),
                                              matmul_fwd_params.weight_format));

    context_.dst_md.reset(new memory::desc({matmul_fwd_params.dst_dims},
                                           MklDnnType<Toutput>(),
                                           matmul_fwd_params.dst_format));

    context_.bias_md.reset(new memory::desc({matmul_fwd_params.bias_dims},
                                            MklDnnType<Tbias>(),
                                            memory::format_tag::any));
    // Create an inner-product.
    context_.fwd_desc.reset(new inner_product_forward::desc(
        matmul_fwd_params.const_weight ? prop_kind::forward_inference
                                       : prop_kind::forward_training,
        *context_.src_md, *context_.weight_md, *context_.bias_md,
        *context_.dst_md));
    context_.fwd_pd.reset(new inner_product_forward::primitive_desc(
        *context_.fwd_desc, cpu_engine_));

    // Check if there is any fusion as post-ops
    auto const& post_op_params = matmul_fwd_params.post_op_params;
    dnnl::primitive_attr post_ops_attr;
    post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    dnnl::post_ops post_ops;
    if (!post_op_params.empty()) {
      for (auto const& post_op_param : post_op_params) {
        if (post_op_param.name == "relu" || post_op_param.name == "leakyrelu") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, dnnl::algorithm::eltwise_relu,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "relu6") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale,
                                  dnnl::algorithm::eltwise_bounded_relu,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "elu") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, dnnl::algorithm::eltwise_elu,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "gelu_approximate") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, dnnl::algorithm::eltwise_gelu_tanh,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "gelu_exact") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, dnnl::algorithm::eltwise_gelu_erf,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "tanh") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, dnnl::algorithm::eltwise_tanh,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "logistic") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, dnnl::algorithm::eltwise_logistic,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "output_scale") {
          DCHECK_EQ(post_op_param.param.size(), 1);
          std::vector<float> scales;
          scales.push_back(post_op_param.param[0]);
          post_ops_attr.set_output_scales(0, scales);
        } else if (post_op_param.name == "sum") {
          DCHECK_EQ(post_op_param.param.size(), 1);
          float op_scale = post_op_param.param[0];
          post_ops.append_sum(op_scale);

        } else {
          DCHECK((post_op_param.name == "relu") ||
                 (post_op_param.name == "relu6") ||
                 (post_op_param.name == "elu") ||
                 (post_op_param.name == "tanh") ||
                 (post_op_param.name == "logistic") ||
                 (post_op_param.name == "sum") ||
                 (post_op_param.name == "leakyrelu") ||
                 (post_op_param.name == "output_scale"));
        }
      }
      post_ops_attr.set_post_ops(post_ops);
      context_.fwd_pd.reset(new inner_product_forward::primitive_desc(
          *context_.fwd_desc, post_ops_attr, cpu_engine_));
    } else {
      context_.fwd_pd.reset(new inner_product_forward::primitive_desc(
          *context_.fwd_desc, post_ops_attr, cpu_engine_));
    }

    // Create memory primitive based on dummy data
    context_.src_mem.reset(
        new memory(context_.fwd_pd.get()->src_desc(), cpu_engine_, DummyData));
    context_.weight_mem.reset(new memory(context_.fwd_pd.get()->weights_desc(),
                                         cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new memory(context_.fwd_pd.get()->dst_desc(), cpu_engine_, DummyData));
    context_.bias_mem.reset(new memory({{matmul_fwd_params.bias_dims},
                                        MklDnnType<Tbias>(),
                                        memory::format_tag::x},
                                       cpu_engine_, DummyData));
    auto scratchpad_md = context_.fwd_pd->scratchpad_desc();
    context_.sp_mem.reset(
        new dnnl::memory(scratchpad_md, cpu_engine_, DummyData));

    // Create inner-product primitive.
    context_.matmul_fwd.reset(new inner_product_forward(*context_.fwd_pd));
    context_.net_args.push_back({{DNNL_ARG_SRC, *context_.src_mem},
                                 {DNNL_ARG_WEIGHTS, *context_.weight_mem},
                                 {DNNL_ARG_BIAS, *context_.bias_mem},
                                 {DNNL_ARG_SCRATCHPAD, *context_.sp_mem},
                                 {DNNL_ARG_DST, *context_.dst_mem}});

    context_.fwd_primitives.push_back(*context_.matmul_fwd);
    return;
  }

  struct MklDnnMatMulFwdContext context_;

#ifdef DNNL_AARCH64_USE_ACL
  // Guards Execution()
  mutex primitive_execution_mu_;
#endif
};

template <typename T, typename Tinput, typename Tweight, typename Tbias,
          typename Toutput>
class MklDnnMatMulFwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklDnnMatMulFwdPrimitive<T, Tinput, Tweight, Tbias, Toutput>* Get(
      const MklDnnMatMulFwdParams& mkldnn_matmul_fwd_dims, bool do_not_cache) {
    MklDnnMatMulFwdPrimitive<T, Tinput, Tweight, Tbias, Toutput>* matmul_fwd =
        nullptr;

    if (do_not_cache) {
      // Always create new primitive
      matmul_fwd =
          new MklDnnMatMulFwdPrimitive<T, Tinput, Tweight, Tbias, Toutput>(
              mkldnn_matmul_fwd_dims);
    } else {
      // Try to find a suitable one in pool
      matmul_fwd = dynamic_cast<
          MklDnnMatMulFwdPrimitive<T, Tinput, Tweight, Tbias, Toutput>*>(
          MklDnnMatMulFwdPrimitiveFactory<T, Tinput, Tweight, Tbias,
                                          Toutput>::GetInstance()
              .GetMklDnnMatMulFwd(mkldnn_matmul_fwd_dims));
      if (matmul_fwd == nullptr) {
        matmul_fwd =
            new MklDnnMatMulFwdPrimitive<T, Tinput, Tweight, Tbias, Toutput>(
                mkldnn_matmul_fwd_dims);
        MklDnnMatMulFwdPrimitiveFactory<T, Tinput, Tweight, Tbias,
                                        Toutput>::GetInstance()
            .SetMklDnnMatMulFwd(mkldnn_matmul_fwd_dims, matmul_fwd);
      }
    }
    return matmul_fwd;
  }

 private:
  MklDnnMatMulFwdPrimitiveFactory() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_8(mht_8_v, 582, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "MklDnnMatMulFwdPrimitiveFactory");
}
  ~MklDnnMatMulFwdPrimitiveFactory() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_9(mht_9_v, 586, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "~MklDnnMatMulFwdPrimitiveFactory");
}

  static MklDnnMatMulFwdPrimitiveFactory& GetInstance() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_10(mht_10_v, 591, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "GetInstance");

    static MklDnnMatMulFwdPrimitiveFactory instance_;
    return instance_;
  }

  static string CreateKey(const MklDnnMatMulFwdParams& mkldnn_matmul_fwd_dims) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_11(mht_11_v, 599, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "CreateKey");

    string prefix = "matmul_fwd_";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(mkldnn_matmul_fwd_dims.src_dims);
    key_creator.AddAsKey(mkldnn_matmul_fwd_dims.weight_dims);
    key_creator.AddAsKey(mkldnn_matmul_fwd_dims.bias_dims);
    key_creator.AddAsKey(mkldnn_matmul_fwd_dims.dst_dims);
    key_creator.AddAsKey(mkldnn_matmul_fwd_dims.dtypes);
    key_creator.AddAsKey(mkldnn_matmul_fwd_dims.weight_format);
#ifdef DNNL_AARCH64_USE_ACL
    key_creator.AddAsKey(mkldnn_matmul_fwd_dims.weight_address);
#endif

    // Generate keys for post-ops
    for (auto const& post_op_param : mkldnn_matmul_fwd_dims.post_op_params) {
      if (post_op_param.name == "relu" || post_op_param.name == "relu6" ||
          post_op_param.name == "elu" || post_op_param.name == "tanh" ||
          post_op_param.name == "logistic" ||
          post_op_param.name == "leakyrelu" ||
          post_op_param.name == "gelu_approximate" ||
          post_op_param.name == "gelu_exact") {
        DCHECK_EQ(post_op_param.param.size(), 3);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
        key_creator.AddAsKey(post_op_param.param[1]);
        key_creator.AddAsKey(post_op_param.param[2]);
      } else if (post_op_param.name == "sum") {
        DCHECK_EQ(post_op_param.param.size(), 1);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
      } else if (post_op_param.name == "output_scale") {
        DCHECK_EQ(post_op_param.param.size(), 1);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
      } else {
        return string("not_a_key");
      }
    }
    return key_creator.GetKey();
  }

  MklPrimitive* GetMklDnnMatMulFwd(
      const MklDnnMatMulFwdParams& mkldnn_matmul_fwd_dims) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_12(mht_12_v, 645, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "GetMklDnnMatMulFwd");

    string key = CreateKey(mkldnn_matmul_fwd_dims);
    return this->GetOp(key);
  }

  void SetMklDnnMatMulFwd(const MklDnnMatMulFwdParams& mkldnn_matmul_fwd_dims,
                          MklPrimitive* op) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_13(mht_13_v, 654, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "SetMklDnnMatMulFwd");

    string key = CreateKey(mkldnn_matmul_fwd_dims);
    this->SetOp(key, op);
  }
};

template <class Tweight, class Toutput>
class MklDnnMatMulOpBase : public OpKernel {
 public:
  explicit MklDnnMatMulOpBase(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_14(mht_14_v, 667, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "MklDnnMatMulOpBase");
}
  void Compute(OpKernelContext* context) override = 0;

  // Allocate output tensor.
  virtual void AllocateOutputTensor(
      OpKernelContext* context,
      const inner_product_forward::primitive_desc& mkldnn_matmul_prim_desc,
      const memory::dims& output_dims_mkl_order,
      MklTensorFormat output_tf_format, Tensor** output_tensor,
      bool native_format = false) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_15(mht_15_v, 679, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "AllocateOutputTensor");

    DCHECK(output_tensor);
    auto dst_pd = mkldnn_matmul_prim_desc.dst_desc();

    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(true);
    output_mkl_shape.SetMklLayout(&dst_pd);
    output_mkl_shape.SetElemType(MklDnnType<Toutput>());
    output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                 output_dims_mkl_order, output_tf_format);

    TensorShape output_tf_shape;
    output_tf_shape.AddDim((dst_pd.get_size() / sizeof(Toutput)));

    if (native_format) {
      output_tf_shape = output_mkl_shape.GetTfShape();
    }
    // Allocate Output Tensor
    AllocateOutputSetMklShape(context, kOutputIndexDst, output_tensor,
                              output_tf_shape, output_mkl_shape, native_format);
  }

  // TF_LOCKS_EXCLUDED annotation ensures that the lock (mu_) cannot
  // be acquired before entering the function, since it is acquired
  // inside the function.
  inline bool IsWeightCacheEmpty(OpKernelContext* context)
      TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock lock(mu_);
    return (weight_oi_.NumElements() == 0);
  }

  // Cache the converted weight in a tensor.
  // Only one thread can execute this method at any given time.
  void CacheWeight(
      OpKernelContext* context,
      const std::shared_ptr<dnnl::inner_product_forward::primitive_desc>&
          matmul_fwd_pd,
      Tweight* weight_data, const Tensor& weight_tensor,
      MklDnnData<Tweight>& weight, const memory::desc& weight_md)
      TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    const Tensor& weight_t = weight_oi_;

    // If the weights are already cached, there's nothing to do
    if (weight_t.NumElements() > 0) {
      return;
    }

    // reorder and cache the weight
    weight.SetUsrMem(weight_md, &weight_tensor);
    weight.CheckReorderToOpMem(matmul_fwd_pd.get()->weights_desc(), cpu_engine_,
                               context);
    weight_data = static_cast<Tweight*>(weight.GetOpMem().get_data_handle());

    size_t weight_size = matmul_fwd_pd.get()->weights_desc().get_size();
    TensorShape weight_tf_shape;
    weight_tf_shape.AddDim(weight_size / sizeof(Tweight));

    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Tweight>::value,
                                          weight_tf_shape, &weight_oi_));

    void* weight_oi_t_data = weight.GetTensorBuffer(&weight_oi_);
    memcpy(weight_oi_t_data, weight_data, weight_size);

    // cache the memory descriptor
    auto expected_md = matmul_fwd_pd->weights_desc();
    TensorShape weight_mkl_format;
    weight_mkl_format.AddDim(sizeof(expected_md) / sizeof(Tweight));

    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Tweight>::value,
                                          weight_mkl_format, &weight_oi_md_));
    *reinterpret_cast<memory::desc*>(weight_oi_md_.flat<Tweight>().data()) =
        expected_md;
  }

  Tweight* GetCachedWeight(OpKernelContext* context,
                           const memory::desc& expected_md)
      TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock lock(mu_);
    const Tensor& weight_t = weight_oi_;
    const Tensor& weight_md_t = weight_oi_md_;

    // Check if the memory descriptor of the cached weight is same as
    // expected_md. if so use the cached memory, else return NULL
    if (weight_md_t.flat<Tweight>().size()) {
      const memory::desc& stored_md =
          *(static_cast<memory::desc*>(weight_md_t.data()));
      if (stored_md == expected_md) {
        return static_cast<Tweight*>(
            const_cast<Tweight*>(weight_t.flat<Tweight>().data()));
      }
    }
    return nullptr;
  }

  engine cpu_engine_ = engine(engine::kind::cpu, 0);

 protected:
  // Tensor to save reordered weight
  mutex mu_;
  Tensor weight_oi_ TF_GUARDED_BY(mu_);
  Tensor weight_oi_md_ TF_GUARDED_BY(mu_);

  bool is_weight_const_;

  const int kInputIndexSrc = 0;
  const int kInputIndexWeight = 1;
  const int kInputIndexBias = 2;
  const int kOutputIndexDst = 0;
};

using dnnl::matmul;

namespace {

struct MklMatMulParams {
  memory::dims a_dims;
  memory::dims b_dims;
  memory::dims c_dims;
  memory::dims a_strides;
  memory::dims b_strides;
  memory::dims c_strides;
#ifdef DNNL_AARCH64_USE_ACL
  int aarch64_counter;
#endif
  struct PostOpParam {
    string name;
    std::vector<float> param;
    memory::dims dims;
    memory::data_type data_type;
    memory::format_tag format_tag;
  };
  std::vector<PostOpParam> post_op_params;

  MklMatMulParams(memory::dims a_dims, memory::dims b_dims, memory::dims c_dims,
                  memory::dims a_strides, memory::dims b_strides,
                  memory::dims c_strides)
      : a_dims(a_dims),
        b_dims(b_dims),
        c_dims(c_dims),
        a_strides(a_strides),
        b_strides(b_strides),
        c_strides(c_strides) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_16(mht_16_v, 826, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "MklMatMulParams");
}
};

template <typename Tlhs, typename Trhs, typename Toutput>
class MklMatMulPrimitive : public MklPrimitive {
 public:
  explicit MklMatMulPrimitive(const MklMatMulParams& params)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_17(mht_17_v, 836, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "MklMatMulPrimitive");

    // Create matmul primitive
    Setup(params);
  }

  ~MklMatMulPrimitive() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_18(mht_18_v, 844, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "~MklMatMulPrimitive");
}

  dnnl::memory::desc GetScratchPadDesc() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_19(mht_19_v, 849, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "GetScratchPadDesc");

    return context_.prim_desc->scratchpad_desc();
  }
  void Execute(const std::shared_ptr<stream>& stream, const Tlhs* a_data,
               const Trhs* b_data, const Toutput* c_data, void* sp_data,
               void* mul_data = nullptr, void* add_data = nullptr) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_20(mht_20_v, 857, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "Execute");

#ifdef DNNL_AARCH64_USE_ACL
    mutex_lock lock(primitive_execution_mu_);
#endif
#ifndef ENABLE_ONEDNN_OPENMP
    context_.a_mem->set_data_handle(
        static_cast<void*>(const_cast<Tlhs*>(a_data)), *stream);
    context_.b_mem->set_data_handle(
        static_cast<void*>(const_cast<Trhs*>(b_data)), *stream);
    context_.c_mem->set_data_handle(
        static_cast<void*>(const_cast<Toutput*>(c_data)), *stream);
    context_.sp_mem->set_data_handle(sp_data, *stream);

    if (mul_data != nullptr)
      context_.mul_mem->set_data_handle(mul_data, *stream);
    if (add_data != nullptr)
      context_.add_mem->set_data_handle(add_data, *stream);
#else
    context_.a_mem->set_data_handle(
        static_cast<void*>(const_cast<Tlhs*>(a_data)));
    context_.b_mem->set_data_handle(
        static_cast<void*>(const_cast<Trhs*>(b_data)));
    context_.c_mem->set_data_handle(
        static_cast<void*>(const_cast<Toutput*>(c_data)));
    context_.sp_mem->set_data_handle(sp_data);
    if (mul_data != nullptr) context_.mul_mem->set_data_handle(mul_data);
    if (add_data != nullptr) context_.add_mem->set_data_handle(add_data);
#endif  // !ENABLE_ONEDNN_OPENMP
    execute_primitives(context_.matmul_primitives, stream, context_.net_args);

    // After execution, set data handle back
    context_.a_mem->set_data_handle(DummyData);
    context_.b_mem->set_data_handle(DummyData);
    context_.c_mem->set_data_handle(DummyData);
    context_.sp_mem->set_data_handle(DummyData);
    if (mul_data != nullptr) context_.mul_mem->set_data_handle(DummyData);
    if (add_data != nullptr) context_.add_mem->set_data_handle(DummyData);
  }

 private:
  // Primitive reuse context for MatMul op
  struct MklMatMulContext {
    // oneDNN memory.
    std::shared_ptr<dnnl::memory> a_mem;
    std::shared_ptr<dnnl::memory> b_mem;
    std::shared_ptr<dnnl::memory> c_mem;
    std::shared_ptr<dnnl::memory> mul_mem;
    std::shared_ptr<dnnl::memory> add_mem;
    std::shared_ptr<dnnl::memory> sp_mem;

    // Descriptor and primitive-descriptor for MatMul.
    std::shared_ptr<matmul::desc> desc;
    std::shared_ptr<matmul::primitive_desc> prim_desc;

    // Memory descriptors.
    std::shared_ptr<dnnl::memory::desc> a_md;
    std::shared_ptr<dnnl::memory::desc> b_md;
    std::shared_ptr<dnnl::memory::desc> c_md;
    std::shared_ptr<dnnl::memory::desc> mul_md;
    std::shared_ptr<dnnl::memory::desc> add_md;

    // MatMul primitive.
    std::vector<dnnl::primitive> matmul_primitives;
    std::vector<std::unordered_map<int, memory>> net_args;

    MklMatMulContext()
        : a_mem(nullptr),
          b_mem(nullptr),
          c_mem(nullptr),
          mul_mem(nullptr),
          add_mem(nullptr),
          sp_mem(nullptr),
          desc(nullptr),
          prim_desc(nullptr),
          a_md(nullptr),
          b_md(nullptr),
          c_md(nullptr),
          mul_md(nullptr),
          add_md(nullptr) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_21(mht_21_v, 938, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "MklMatMulContext");
}
  };

  void Setup(const MklMatMulParams& params) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_22(mht_22_v, 944, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "Setup");

    std::shared_ptr<dnnl::primitive> matmul_primitive = nullptr;

    // Create MatMul descriptor and primitive descriptor.
    context_.a_md.reset(new memory::desc({params.a_dims}, MklDnnType<Tlhs>(),
                                         params.a_strides));

    context_.b_md.reset(new memory::desc({params.b_dims}, MklDnnType<Trhs>(),
                                         params.b_strides));

    context_.c_md.reset(new memory::desc({params.c_dims}, MklDnnType<Toutput>(),
                                         params.c_strides));

    // Create matmul.
    context_.desc.reset(
        new matmul::desc(*context_.a_md, *context_.b_md, *context_.c_md));

    // Check if there is any fusion as post-ops
    auto const& post_op_params = params.post_op_params;
    dnnl::primitive_attr post_ops_attr;
    dnnl::post_ops post_ops;
    if (!post_op_params.empty()) {
      for (auto const& post_op_param : post_op_params) {
        if (post_op_param.name == "output_scale") {
          DCHECK_EQ(post_op_param.param.size(), 1);
          std::vector<float> scales;
          scales.push_back(post_op_param.param[0]);
          post_ops_attr.set_output_scales(0, scales);
        } else if (post_op_param.name == "mul") {
          context_.mul_md.reset(new memory::desc({post_op_param.dims},
                                                 post_op_param.data_type,
                                                 post_op_param.format_tag));
          post_ops.append_binary(dnnl::algorithm::binary_mul, *context_.mul_md);
        } else if (post_op_param.name == "add") {
          context_.add_md.reset(new memory::desc({post_op_param.dims},
                                                 post_op_param.data_type,
                                                 post_op_param.format_tag));
          post_ops.append_binary(dnnl::algorithm::binary_add, *context_.add_md);
        } else {
          DCHECK((post_op_param.name == "output_scale"));
        }
      }
      post_ops_attr.set_post_ops(post_ops);
    }
    post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    context_.prim_desc.reset(
        new matmul::primitive_desc(*context_.desc, post_ops_attr, cpu_engine_));

    // Create memory primitive based on dummy data.
    context_.a_mem.reset(
        new dnnl::memory(*context_.a_md, cpu_engine_, DummyData));
    context_.b_mem.reset(
        new dnnl::memory(*context_.b_md, cpu_engine_, DummyData));
    context_.c_mem.reset(
        new dnnl::memory(*context_.b_md, cpu_engine_, DummyData));
    auto scratchpad_md = context_.prim_desc->scratchpad_desc();
    context_.sp_mem.reset(
        new dnnl::memory(scratchpad_md, cpu_engine_, DummyData));

    // Create matmul primitive.
    matmul_primitive.reset(new dnnl::matmul(*context_.prim_desc));
    context_.net_args.push_back({{DNNL_ARG_SRC, *context_.a_mem},
                                 {DNNL_ARG_WEIGHTS, *context_.b_mem},
                                 {DNNL_ARG_SCRATCHPAD, *context_.sp_mem},
                                 {DNNL_ARG_DST, *context_.c_mem}});
    if (!post_op_params.empty()) {
      int count = 0;
      for (auto const& post_op_param : post_op_params) {
        if (post_op_param.name == "mul") {
          context_.mul_mem.reset(
              new dnnl::memory(*context_.mul_md, cpu_engine_, DummyData));
          context_.net_args[0].insert(
              {DNNL_ARG_ATTR_MULTIPLE_POST_OP(count) | DNNL_ARG_SRC_1,
               *context_.mul_mem});
          count++;
        } else if (post_op_param.name == "add") {
          context_.add_mem.reset(
              new dnnl::memory(*context_.add_md, cpu_engine_, DummyData));
          context_.net_args[0].insert(
              {DNNL_ARG_ATTR_MULTIPLE_POST_OP(count) | DNNL_ARG_SRC_1,
               *context_.add_mem});
          count++;
        }
      }
    }

    context_.matmul_primitives.push_back(*matmul_primitive);
    return;
  }

  struct MklMatMulContext context_;
#ifdef DNNL_AARCH64_USE_ACL
  mutex primitive_execution_mu_;
#endif
};

template <typename T, typename Tlhs, typename Trhs, typename Toutput>
class MklMatMulPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklMatMulPrimitive<Tlhs, Trhs, Toutput>* Get(
      const MklMatMulParams& params, bool do_not_cache) {
    MklMatMulPrimitive<Tlhs, Trhs, Toutput>* matmul_prim = nullptr;

    if (do_not_cache) {
      // Always create new primitive
      matmul_prim = new MklMatMulPrimitive<Tlhs, Trhs, Toutput>(params);
    } else {
      // Try to find a suitable one in pool
      matmul_prim = dynamic_cast<MklMatMulPrimitive<Tlhs, Trhs, Toutput>*>(
          MklMatMulPrimitiveFactory<T, Tlhs, Trhs, Toutput>::GetInstance()
              .GetMklMatMul(params));
      if (matmul_prim == nullptr) {
        matmul_prim = new MklMatMulPrimitive<Tlhs, Trhs, Toutput>(params);
        MklMatMulPrimitiveFactory<T, Tlhs, Trhs, Toutput>::GetInstance()
            .SetMklMatMul(params, matmul_prim);
      }
    }

    return matmul_prim;
  }

 private:
  MklMatMulPrimitiveFactory() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_23(mht_23_v, 1069, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "MklMatMulPrimitiveFactory");
}
  ~MklMatMulPrimitiveFactory() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_24(mht_24_v, 1073, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "~MklMatMulPrimitiveFactory");
}

  static MklMatMulPrimitiveFactory& GetInstance() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_25(mht_25_v, 1078, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "GetInstance");

    static MklMatMulPrimitiveFactory instance_;
    return instance_;
  }

  static string CreateKey(const MklMatMulParams& params) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_26(mht_26_v, 1086, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "CreateKey");

    string prefix = "matmul_";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(params.a_dims);
    key_creator.AddAsKey(params.b_dims);
    key_creator.AddAsKey(params.c_dims);
    key_creator.AddAsKey(params.a_strides);
    key_creator.AddAsKey(params.b_strides);
    key_creator.AddAsKey(params.c_strides);
    key_creator.AddAsKey(typeid(T).name());
#ifdef DNNL_AARCH64_USE_ACL
    key_creator.AddAsKey(params.aarch64_counter);
#endif
    key_creator.AddAsKey(typeid(Tlhs).name());
    key_creator.AddAsKey(typeid(Trhs).name());
    key_creator.AddAsKey(typeid(Toutput).name());

    // Generate keys for post-ops
    for (auto const& post_op_param : params.post_op_params) {
      if (post_op_param.name == "output_scale") {
        DCHECK_EQ(post_op_param.param.size(), 1);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
      } else if (post_op_param.name == "mul" || post_op_param.name == "add") {
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.dims);
      } else {
        return string("not_a_key");
      }
    }
    return key_creator.GetKey();
  }

  MklPrimitive* GetMklMatMul(const MklMatMulParams& params) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_27(mht_27_v, 1123, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "GetMklMatMul");

    string key = CreateKey(params);
    return this->GetOp(key);
  }

  void SetMklMatMul(const MklMatMulParams& params, MklPrimitive* op) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_28(mht_28_v, 1131, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "SetMklMatMul");

    string key = CreateKey(params);
    this->SetOp(key, op);
  }
};

template <typename T>
void dnnl_gemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
               float alpha, const T* a, int64_t lda, const T* b, int64_t ldb,
               float beta, T* c, int64_t ldc, OpKernelContext* ctx = nullptr) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("transa: '" + std::string(1, transa) + "'");
   mht_29_v.push_back("transb: '" + std::string(1, transb) + "'");
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_ops_commonDTh mht_29(mht_29_v, 1145, "", "./tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h", "dnnl_gemm");

  using dims = dnnl::memory::dims;

  // Prepare strides based on the transa and transb flags: transposed
  // matrices have strides swapped
  dims a_dims = dims{m, k};
  dims b_dims = dims{k, n};
  dims c_dims = dims{m, n};
  dims a_strides = tolower(transa) == 'n' ? dims{lda, 1} : dims{1, lda};
  dims b_strides = tolower(transb) == 'n' ? dims{ldb, 1} : dims{1, ldb};
  dims c_strides = dims{ldc, 1};

  // MklMatMul uses const alpha and beta, make guarantee here to ensure
  // they are never changed.
  DCHECK_EQ(alpha, 1.0f);
  DCHECK_EQ(beta, 0.f);

  MklMatMulParams params(a_dims, b_dims, c_dims, a_strides, b_strides,
                         c_strides);
  MklMatMulPrimitive<T, T, T>* matmul_prim =
      MklMatMulPrimitiveFactory<T, T, T, T>::Get(params, 0);

  UserScratchPad<unsigned char> scratch_pad;
  scratch_pad.AllocateSPTensor(matmul_prim, ctx);
  // Execute matmul primitive.
  auto st = ExecuteSingleThreadedGemm(m, n, k, sizeof(T));
  std::shared_ptr<stream> cpu_stream;
  MklDnnThreadPool eigen_tp(ctx, st ? 1 : -1);
  cpu_stream.reset(CreateStream(&eigen_tp, matmul_prim->GetEngine()));
  matmul_prim->Execute(cpu_stream, a, b, c, scratch_pad.Get());
}

}  // anonymous namespace

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_MATMUL_OPS_COMMON_H_
