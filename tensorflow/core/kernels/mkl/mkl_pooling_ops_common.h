/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_MKL_POOLING_OPS_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_MKL_MKL_POOLING_OPS_COMMON_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh() {
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

#include "dnnl.hpp"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/padding.h"
#ifdef DNNL_AARCH64_USE_ACL
#include "tensorflow/core/platform/mutex.h"
#endif

namespace tensorflow {

using dnnl::pooling_backward;
using dnnl::pooling_forward;
using dnnl::prop_kind;
using dnnl::stream;

using PoolingFwdPd = dnnl::pooling_forward::primitive_desc;
using PoolingBwdPd = dnnl::pooling_backward::primitive_desc;

struct MklPoolingParams {
  memory::dims src_dims;
  memory::dims dst_dims;
  memory::dims filter_dims;
  memory::dims strides;
  memory::dims padding_left;
  memory::dims padding_right;
  dnnl::algorithm alg_kind;
  dnnl::prop_kind prop_kind;
  memory::format_tag src_format;
  memory::desc src_md;
  bool native_format;

  MklPoolingParams(memory::dims src_dims, memory::dims dst_dims,
                   memory::dims filter_dims, memory::dims strides,
                   memory::dims padding_left, memory::dims padding_right,
                   dnnl::algorithm alg_kind, dnnl::prop_kind prop_kind,
                   memory::format_tag src_format, memory::desc src_md,
                   bool native_format)
      : src_dims(src_dims),
        dst_dims(dst_dims),
        filter_dims(filter_dims),
        strides(strides),
        padding_left(padding_left),
        padding_right(padding_right),
        alg_kind(alg_kind),
        prop_kind(prop_kind),
        src_format(src_format),
        src_md(src_md),
        native_format(native_format) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_0(mht_0_v, 240, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "MklPoolingParams");
}
};

template <typename T>
class MklPoolingFwdPrimitive : public MklPrimitive {
 public:
  explicit MklPoolingFwdPrimitive(const MklPoolingParams& fwdParams)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_1(mht_1_v, 250, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "MklPoolingFwdPrimitive");

    if (context_.fwd == nullptr) Setup(fwdParams);
  }

  ~MklPoolingFwdPrimitive() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_2(mht_2_v, 257, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "~MklPoolingFwdPrimitive");
}

  // Pooling forward execute
  //   src_data:  input data buffer of src
  //   ws_data:   output data buffer of workspace
  //   dst_data:  output data buffer of dst
  void Execute(const T* src_data, T* dst_data, void* ws_data,
               std::shared_ptr<stream> fwd_stream);

  std::shared_ptr<PoolingFwdPd> GetPoolingFwdPd() const {
    return context_.fwd_pd;
  }

  memory::format_tag GetSrcMemoryFormat() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_3(mht_3_v, 273, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "GetSrcMemoryFormat");
 return context_.src_fmt; }
  memory::format_tag GetDstMemoryFormat() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_4(mht_4_v, 277, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "GetDstMemoryFormat");
 return context_.dst_fmt; }

 private:
  void Setup(const MklPoolingParams& fwdParams);

  struct PoolingFwdContext {
    // Algorithm.
    dnnl::algorithm alg_kind;

    // Kind of propagation, forward or backward.
    dnnl::prop_kind prop_kind;

    // Expected memory format.
    memory::format_tag src_fmt;
    memory::format_tag dst_fmt;
    memory::format_tag ws_fmt;

    // Workspace shape.
    memory::dims ws_dims;
    memory::data_type ws_dt;
    size_t ws_size;

    // oneDNN memory, just dummy data.
    std::shared_ptr<dnnl::memory> ws_mem;
    std::shared_ptr<dnnl::memory> src_mem;
    std::shared_ptr<dnnl::memory> dst_mem;

    // Pooling forward descriptor and primitive descriptor.
    std::shared_ptr<dnnl::pooling_forward::desc> fwd_desc;
    std::shared_ptr<PoolingFwdPd> fwd_pd;

    // Memory descriptor.
    std::shared_ptr<dnnl::memory::desc> src_md;
    std::shared_ptr<dnnl::memory::desc> dst_md;

    // Pooling primitive
    std::shared_ptr<dnnl::pooling_forward> fwd;
    std::shared_ptr<dnnl::stream> fwd_stream;
    std::vector<dnnl::primitive> fwd_primitives;

    std::vector<std::unordered_map<int, memory>> net_args;

    PoolingFwdContext()
        : src_fmt(memory::format_tag::any),
          dst_fmt(memory::format_tag::any),
          ws_fmt(memory::format_tag::any),
          ws_mem(nullptr),
          src_mem(nullptr),
          dst_mem(nullptr),
          fwd_desc(nullptr),
          fwd_pd(nullptr),
          src_md(nullptr),
          dst_md(nullptr),
          fwd(nullptr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_5(mht_5_v, 333, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "PoolingFwdContext");
}
  };

  struct PoolingFwdContext context_;

#ifdef DNNL_AARCH64_USE_ACL
  mutex primitive_execution_mu_;
#endif
};

template <typename T>
class MklPoolingFwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklPoolingFwdPrimitive<T>* Get(const MklPoolingParams& fwdParams) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_6(mht_6_v, 349, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "Get");

    MklPoolingFwdPrimitive<T>* pooling_forward = nullptr;

    // Get pooling primitive from the pool
    pooling_forward = static_cast<MklPoolingFwdPrimitive<T>*>(
        MklPoolingFwdPrimitiveFactory<T>::GetInstance().GetPoolingFwd(
            fwdParams));

    if (pooling_forward == nullptr) {
      pooling_forward = new MklPoolingFwdPrimitive<T>(fwdParams);
      MklPoolingFwdPrimitiveFactory<T>::GetInstance().SetPoolingFwd(
          fwdParams, pooling_forward);
    }
    return pooling_forward;
  }

  static MklPoolingFwdPrimitiveFactory& GetInstance() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_7(mht_7_v, 368, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "GetInstance");

    static MklPoolingFwdPrimitiveFactory instance_;
    return instance_;
  }

 private:
  MklPoolingFwdPrimitiveFactory() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_8(mht_8_v, 377, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "MklPoolingFwdPrimitiveFactory");
}
  ~MklPoolingFwdPrimitiveFactory() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_9(mht_9_v, 381, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "~MklPoolingFwdPrimitiveFactory");
}

  // The key to be created will be used to get/set pooling
  // primitive op from reuse perspective.
  // A pooling key is a string which concates key parameters
  // as well as algorithm kind (max versus avg).
  static string CreateKey(const MklPoolingParams& fwdParams) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_10(mht_10_v, 390, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "CreateKey");

    string prefix = "pooling_fwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(fwdParams.src_dims);
    key_creator.AddAsKey(fwdParams.dst_dims);
    key_creator.AddAsKey(fwdParams.filter_dims);
    key_creator.AddAsKey(fwdParams.strides);
    key_creator.AddAsKey(fwdParams.padding_left);
    key_creator.AddAsKey(fwdParams.padding_right);
    key_creator.AddAsKey<int>(static_cast<int>(fwdParams.alg_kind));
    key_creator.AddAsKey<int>(static_cast<int>(fwdParams.prop_kind));
    return key_creator.GetKey();
  }

  MklPrimitive* GetPoolingFwd(const MklPoolingParams& fwdParams) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_11(mht_11_v, 408, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "GetPoolingFwd");

    string key = CreateKey(fwdParams);
    return this->GetOp(key);
  }

  void SetPoolingFwd(const MklPoolingParams& fwdParams, MklPrimitive* op) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_12(mht_12_v, 416, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "SetPoolingFwd");

    string key = CreateKey(fwdParams);
    this->SetOp(key, op);
  }
};

template <typename T>
class MklPoolingBwdPrimitive : public MklPrimitive {
 public:
  explicit MklPoolingBwdPrimitive(const MklPoolingParams& bwdParams)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_13(mht_13_v, 429, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "MklPoolingBwdPrimitive");

    if (context_.bwd == nullptr) Setup(bwdParams);
  }

  ~MklPoolingBwdPrimitive() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_14(mht_14_v, 436, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "~MklPoolingBwdPrimitive");
}

  // Pooling backward execute
  //   diff_dst_data:  input data buffer of diff_dst
  //   diff_src_data:  output data buffer of diff_src
  //   ws_data:        input data buffer of workspace
  void Execute(const T* diff_dst_data, T* diff_src_data, const void* ws_data,
               std::shared_ptr<stream> bwd_stream);

 public:
  std::shared_ptr<PoolingFwdPd> GetPoolingFwdPd() const {
    return context_.fwd_pd;
  }
  std::shared_ptr<PoolingBwdPd> GetPoolingBwdPd() const {
    return context_.bwd_pd;
  }

  dnnl::memory::data_type GetWorkspaceDataType() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_15(mht_15_v, 456, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "GetWorkspaceDataType");

    return context_.ws_dt;
  }

 private:
  void Setup(const MklPoolingParams& bwdParams);

  // Primitive reuse context for pooling bwd ops
  struct PoolingBwdContext {
    // Algorithm.
    dnnl::algorithm alg_kind;

    // Expected memory format.
    memory::format_tag diff_src_fmt;
    memory::format_tag diff_dst_fmt;
    memory::format_tag ws_fmt;

    // Workspace attribute.
    dnnl::memory::dims ws_dims;
    dnnl::memory::data_type ws_dt;

    // oneDNN memory.
    std::shared_ptr<dnnl::memory> ws_mem;
    std::shared_ptr<dnnl::memory> diff_src_mem;
    std::shared_ptr<dnnl::memory> diff_dst_mem;

    // Memory descriptors.
    std::shared_ptr<dnnl::memory::desc> src_md;
    std::shared_ptr<dnnl::memory::desc> dst_md;

    // Forward and backward pooling descriptors and primitive descriptors.
    std::shared_ptr<dnnl::pooling_forward::desc> fwd_desc;
    std::shared_ptr<dnnl::pooling_backward::desc> bwd_desc;
    std::shared_ptr<PoolingFwdPd> fwd_pd;
    std::shared_ptr<PoolingBwdPd> bwd_pd;

    // Backward pooling primitive.
    std::shared_ptr<dnnl::pooling_backward> bwd;
    std::shared_ptr<dnnl::stream> bwd_stream;

    std::vector<dnnl::primitive> bwd_primitives;
    std::vector<std::unordered_map<int, memory>> net_args;

    PoolingBwdContext()
        : diff_src_fmt(memory::format_tag::any),
          diff_dst_fmt(memory::format_tag::any),
          ws_fmt(memory::format_tag::any),
          ws_mem(nullptr),
          diff_src_mem(nullptr),
          diff_dst_mem(nullptr),
          src_md(nullptr),
          dst_md(nullptr),
          fwd_desc(nullptr),
          bwd_desc(nullptr),
          fwd_pd(nullptr),
          bwd_pd(nullptr),
          bwd(nullptr) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_16(mht_16_v, 515, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "PoolingBwdContext");
}
  };

  struct PoolingBwdContext context_;
#ifdef DNNL_AARCH64_USE_ACL
  mutex primitive_execution_mu_;
#endif
};

template <typename T>
class MklPoolingBwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklPoolingBwdPrimitive<T>* Get(const MklPoolingParams& bwdParams) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_17(mht_17_v, 530, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "Get");

    MklPoolingBwdPrimitive<T>* pooling_backward = nullptr;

    // Find a pooling backward primitive from the pool.
    // If it does not exist, create a new one.
    pooling_backward = static_cast<MklPoolingBwdPrimitive<T>*>(
        MklPoolingBwdPrimitiveFactory<T>::GetInstance().GetPoolingBwd(
            bwdParams));
    if (pooling_backward == nullptr) {
      pooling_backward = new MklPoolingBwdPrimitive<T>(bwdParams);
      MklPoolingBwdPrimitiveFactory<T>::GetInstance().SetPoolingBwd(
          bwdParams, pooling_backward);
    }
    return pooling_backward;
  }

  static MklPoolingBwdPrimitiveFactory& GetInstance() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_18(mht_18_v, 549, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "GetInstance");

    static MklPoolingBwdPrimitiveFactory instance_;
    return instance_;
  }

 private:
  MklPoolingBwdPrimitiveFactory() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_19(mht_19_v, 558, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "MklPoolingBwdPrimitiveFactory");
}
  ~MklPoolingBwdPrimitiveFactory() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_20(mht_20_v, 562, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "~MklPoolingBwdPrimitiveFactory");
}

  // The key to be created will be used to get/set pooling
  // primitive op from reuse perspective.
  // A pooling key is a string which concates key parameters
  // as well as algorithm kind (max versus avg).
  static string CreateKey(const MklPoolingParams& bwdParams) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_21(mht_21_v, 571, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "CreateKey");

    string prefix = "pooling_bwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(bwdParams.src_dims);
    key_creator.AddAsKey(bwdParams.dst_dims);
    key_creator.AddAsKey(bwdParams.filter_dims);
    key_creator.AddAsKey(bwdParams.strides);
    key_creator.AddAsKey(bwdParams.padding_left);
    key_creator.AddAsKey(bwdParams.padding_right);
    key_creator.AddAsKey<int>(static_cast<int>(bwdParams.alg_kind));
    return key_creator.GetKey();
  }

  MklPrimitive* GetPoolingBwd(const MklPoolingParams& bwdParams) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_22(mht_22_v, 588, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "GetPoolingBwd");

    string key = CreateKey(bwdParams);
    return this->GetOp(key);
  }

  void SetPoolingBwd(const MklPoolingParams& bwdParams, MklPrimitive* op) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_23(mht_23_v, 596, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "SetPoolingBwd");

    string key = CreateKey(bwdParams);
    this->SetOp(key, op);
  }
};

typedef Eigen::ThreadPoolDevice CPUDevice;

struct MklPoolParameters {
  int depth;

  int tensor_in_planes;  // Pool3D
  int tensor_in_cols;
  int tensor_in_rows;
  int tensor_in_batch;

  int window_planes;  // Pool3D
  int window_rows;
  int window_cols;
  int depth_window;

  int planes_stride;  // Pool3D
  int row_stride;
  int col_stride;
  int depth_stride;

  int64 out_planes;  // Pool3D
  int64 out_height;
  int64 out_width;
  int out_depth;

  int64 pad_P1;  // Pool3D
  int64 pad_P2;  // Pool3D
  int64 pad_left;
  int64 pad_right;
  int64 pad_top;
  int64 pad_bottom;
  int pad_depth;

  TensorFormat data_format;
  MklPoolParameters()
      : depth(0),
        tensor_in_planes(0),
        tensor_in_cols(0),
        tensor_in_rows(0),
        tensor_in_batch(0),
        window_planes(0),
        window_rows(0),
        window_cols(0),
        depth_window(0),
        planes_stride(0),
        row_stride(0),
        col_stride(0),
        depth_stride(0),
        out_planes(0),
        out_height(0),
        out_width(0),
        out_depth(0),
        pad_P1(0),
        pad_P2(0),
        pad_left(0),
        pad_right(0),
        pad_top(0),
        pad_bottom(0),
        pad_depth(0),
        data_format(TensorFormat::FORMAT_NCHW) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_24(mht_24_v, 664, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "MklPoolParameters");
}

  // Updates context->status if there is an invalid input.
  void Init(OpKernelContext* context, const std::vector<int32>& ksize,
            const std::vector<int32>& stride, Padding padding,
            TensorFormat data_format, const TensorShape& tensor_in_shape);
  void Init(OpKernelContext* context, const std::vector<int32>& ksize,
            const std::vector<int32>& stride, Padding padding,
            TensorFormat data_format, const MklDnnShape* mkl_in_shape);

 private:
  // Common initialization for TensorFlow and MKL formats
  void Init(OpKernelContext* context, const std::vector<int32>& ksize,
            const std::vector<int32>& stride, Padding padding,
            TensorFormat data_format);
};

template <class T>
class MklPoolingOpBase : public OpKernel {
 public:
  explicit MklPoolingOpBase(OpKernelConstruction* context)
      : OpKernel(context), workspace_enabled_(false) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_25(mht_25_v, 688, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "MklPoolingOpBase");

    string data_format;
    if (std::is_same<T, qint8>::value || std::is_same<T, quint8>::value) {
      // Current quantized convolution doesn't have data_format attribute.
      data_format = "NHWC";
    } else {
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    }
    OP_REQUIRES(context, FormatFromString(data_format, &this->data_format_tf_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &this->ksize_));
    OP_REQUIRES(context, this->ksize_.size() == 4 || this->ksize_.size() == 5,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 or 5 dimensions"));
    for (int i = 0; i < this->ksize_.size(); ++i) {
      OP_REQUIRES(context, this->ksize_[i] > 0,
                  errors::InvalidArgument("Sliding window ksize for dimension ",
                                          i, " was zero."));
    }

    OP_REQUIRES_OK(context, context->GetAttr("strides", &this->stride_));
    OP_REQUIRES(context, this->stride_.size() == 4 || this->stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 or 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &this->padding_));
    OP_REQUIRES(context, this->ksize_[0] == 1 && this->stride_[0] == 1,
                errors::Unimplemented("Pooling is not yet supported on the "
                                      "batch dimension."));
    bool is_pool2d = (this->ksize_.size() == 4);
    this->tensor_format_mkldnn_ =
        is_pool2d ? TFDataFormatToMklDnnDataFormat(this->data_format_tf_)
                  : TFDataFormatToMklDnn3DDataFormat(this->data_format_tf_);

    this->data_format_mkldnn_ =
        MklTensorFormatToMklDnnDataFormat(this->tensor_format_mkldnn_);

    // We may not get this attribute for this node if it does not go through
    // graph rewrite pass. So we do not check for error while retrieving this
    // attribute value.
    auto status =
        context->GetAttr("workspace_enabled", &this->workspace_enabled_);
    (void)status;
  }
  void Compute(OpKernelContext* context) override = 0;

 protected:
  // Calculate output shape of pooling op in oneDNN and TensorFlow order.
  // oneDNN uses NCHW(Pool2D) or NCDHW(Pool3D) for output order.
  // But TensorFlow output will be in NHWC/NCHW(Pool2D) or
  // NDHWC/NCDHW(Pool3D) format depending on data format. Function expects
  // output height and width to have already been int32 bounds-checked.
  void GetOutputDims(const MklPoolParameters& mkl_pool_params,
                     memory::dims* output_dims_mkl_order) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_26(mht_26_v, 743, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "GetOutputDims");

    if (this->ksize_.size() == 4) {
      // Pooling2D: oneDNN always needs output in NCHW format.
      *output_dims_mkl_order = {mkl_pool_params.tensor_in_batch,
                                mkl_pool_params.out_depth,
                                static_cast<int>(mkl_pool_params.out_height),
                                static_cast<int>(mkl_pool_params.out_width)};
    } else {
      // Pooling3D: oneDNN always needs output in NCDHW format.
      *output_dims_mkl_order = {mkl_pool_params.tensor_in_batch,
                                mkl_pool_params.out_depth,
                                static_cast<int>(mkl_pool_params.out_planes),
                                static_cast<int>(mkl_pool_params.out_height),
                                static_cast<int>(mkl_pool_params.out_width)};
    }
  }

  void InitMklPoolParameters(OpKernelContext* context,
                             MklPoolParameters* pool_params,
                             const MklDnnShape& original_input_mkl_shape,
                             const TensorShape& input_tensor_shape) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_27(mht_27_v, 766, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "InitMklPoolParameters");

    if (!original_input_mkl_shape.IsMklTensor()) {
      pool_params->Init(context, this->ksize_, this->stride_, this->padding_,
                        this->data_format_tf_, input_tensor_shape);
    } else {
      pool_params->Init(context, this->ksize_, this->stride_, this->padding_,
                        this->data_format_tf_, &original_input_mkl_shape);
    }
  }

  void PoolParamsToDims(const MklPoolParameters* pool_params,
                        memory::dims* filter_dims, memory::dims* strides,
                        memory::dims* padding_left, memory::dims* padding_right,
                        bool is_pool2d) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_28(mht_28_v, 782, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "PoolParamsToDims");

    if (is_pool2d) {
      // Pool2D
      *filter_dims =
          memory::dims({pool_params->window_rows, pool_params->window_cols});
      *strides =
          memory::dims({pool_params->row_stride, pool_params->col_stride});
      *padding_left = memory::dims({static_cast<int>(pool_params->pad_top),
                                    static_cast<int>(pool_params->pad_left)});
      *padding_right = memory::dims({static_cast<int>(pool_params->pad_bottom),
                                     static_cast<int>(pool_params->pad_right)});
    } else {
      // Pool3D
      *filter_dims =
          memory::dims({pool_params->window_planes, pool_params->window_rows,
                        pool_params->window_cols});
      *strides =
          memory::dims({pool_params->planes_stride, pool_params->row_stride,
                        pool_params->col_stride});

      *padding_left = memory::dims({static_cast<int>(pool_params->pad_P1),
                                    static_cast<int>(pool_params->pad_top),
                                    static_cast<int>(pool_params->pad_left)});
      *padding_right = memory::dims({static_cast<int>(pool_params->pad_P2),
                                     static_cast<int>(pool_params->pad_bottom),
                                     static_cast<int>(pool_params->pad_right)});
    }
  }

  void AllocateEmptyOutputTensor(OpKernelContext* context,
                                 const int kOutputIndex,
                                 MklPoolParameters* pool_params,
                                 const memory::dims output_dims_mkl_order,
                                 Tensor** output_tensor) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_29(mht_29_v, 818, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "AllocateEmptyOutputTensor");

    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(false);
    TensorShape output_tf_shape;
    if (pool_params->data_format == TensorFormat::FORMAT_NCHW) {
      output_tf_shape = MklDnnDimsToTFShape(output_dims_mkl_order);
    } else {
      memory::dims output_dims_order;
      // determine Pooling2D (NHWC) or Pooling3D (NDHWC)
      if (this->ksize_.size() == 4) {
        output_dims_order = {pool_params->tensor_in_batch,
                             static_cast<int>(pool_params->out_height),
                             static_cast<int>(pool_params->out_width),
                             pool_params->out_depth};
      } else {
        output_dims_order = {pool_params->tensor_in_batch,
                             static_cast<int>(pool_params->out_planes),
                             static_cast<int>(pool_params->out_height),
                             static_cast<int>(pool_params->out_width),
                             pool_params->out_depth};
      }
      output_tf_shape = MklDnnDimsToTFShape(output_dims_order);
    }
    AllocateOutputSetMklShape(context, kOutputIndex, output_tensor,
                              output_tf_shape, output_mkl_shape,
                              native_format_);
    DCHECK(output_tensor);
  }

  // Checks to make sure that the memory we need to allocate
  // is a multiple of sizeof(T)
  // returns the number of elements
  size_t GetNumTElements(const memory::desc& pd) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_30(mht_30_v, 853, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "GetNumTElements");

    size_t num_bytes = pd.get_size();
    size_t ret_val = num_bytes / sizeof(T);
    if (num_bytes % sizeof(T) != 0) {
      ret_val++;
    }
    return ret_val;
  }

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_tf_;
  MklTensorFormat tensor_format_mkldnn_;
  memory::format_tag data_format_mkldnn_;
  bool workspace_enabled_;
  bool native_format_ = false;
};

template <class T>
class MklPoolingForwardOpBase : public MklPoolingOpBase<T> {
 public:
  explicit MklPoolingForwardOpBase<T>(OpKernelConstruction* context)
      : MklPoolingOpBase<T>(context) {}
  void Compute(OpKernelContext* context) override = 0;

 protected:
  void ConfigureInput(OpKernelContext* context,
                      const MklDnnShape& input_mkl_shape,
                      const Tensor& input_tensor,
                      MklPoolParameters* pool_params,
                      MklDnnData<T>* dnn_data_input) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_31(mht_31_v, 887, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "ConfigureInput");

    DCHECK(pool_params);
    DCHECK(dnn_data_input);
    TensorShape input_tensor_shape = input_tensor.shape();
    if (input_tensor.NumElements() != 0) {
      memory::desc input_md =
          input_mkl_shape.IsMklTensor()
              ? input_mkl_shape.GetMklLayout()
              : memory::desc(
                    (this->ksize_.size() == 4)
                        ? TFShapeToMklDnnDimsInNCHW(input_tensor_shape,
                                                    this->data_format_tf_)
                        : TFShapeToMklDnnDimsInNCDHW(input_tensor_shape,
                                                     this->data_format_tf_),
                    MklDnnType<T>(), this->data_format_mkldnn_);
      dnn_data_input->SetUsrMem(input_md, &input_tensor);

      if (this->ksize_.size() == 5) {
        // Pool3D
        std::vector<dnnl::memory::dim> input_sizes(5, -1);
        input_sizes[MklDnnDims3D::Dim3d_N] = input_md.data.dims[0];
        input_sizes[MklDnnDims3D::Dim3d_C] = input_md.data.dims[1];
        input_sizes[MklDnnDims3D::Dim3d_D] = input_md.data.dims[2];
        input_sizes[MklDnnDims3D::Dim3d_H] = input_md.data.dims[3];
        input_sizes[MklDnnDims3D::Dim3d_W] = input_md.data.dims[4];
        dnn_data_input->SetOpMemDesc(input_sizes, this->data_format_mkldnn_);
      }
    }
    this->InitMklPoolParameters(context, pool_params, input_mkl_shape,
                                input_tensor_shape);
  }

  void AllocateOutputTensor(OpKernelContext* context,
                            const PoolingFwdPd& pool_fwd_prim_desc,
                            const memory::dims output_dims_mkl_order,
                            const MklTensorFormat& output_tf_format,
                            Tensor** output_tensor) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_32(mht_32_v, 926, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "AllocateOutputTensor");

    TensorShape output_tf_shape;
    DCHECK(output_tensor);
    memory::desc dst_pd = pool_fwd_prim_desc.dst_desc();

    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(true);
    output_mkl_shape.SetMklLayout(&dst_pd);
    output_mkl_shape.SetElemType(MklDnnType<T>());
    output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                 output_dims_mkl_order, output_tf_format);
    // Only allocate enough space for the elements we need.
    output_tf_shape.AddDim(this->GetNumTElements(dst_pd));

    if (this->native_format_) {
      output_tf_shape = output_mkl_shape.GetTfShape();
    }
    AllocateOutputSetMklShape(context, kOutputTensorIndexOutput, output_tensor,
                              output_tf_shape, output_mkl_shape,
                              this->native_format_);
    DCHECK(*output_tensor);
  }

  void SanityCheckInput(OpKernelContext* context, const Tensor& input_tensor,
                        const MklDnnShape& input_mkl_shape) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_33(mht_33_v, 953, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "SanityCheckInput");

    if (!input_mkl_shape.IsMklTensor()) {
      OP_REQUIRES(context, input_tensor.dims() == 4 || input_tensor.dims() == 5,
                  errors::InvalidArgument("Input must be 4 or 5-dimensional"));
    } else {
      OP_REQUIRES(
          context,
          input_mkl_shape.GetDimension() == 4 ||
              input_mkl_shape.GetDimension() == 5,
          errors::InvalidArgument("Input shape must be 4 or 5-dimensional"));
    }
  }
  const int kInputTensorIndexInput = 0;
  const int kOutputTensorIndexOutput = 0;
};  // MklPoolingForwardBaseOp

template <class T>
class MklPoolingBackwardOpBase : public MklPoolingOpBase<T> {
 public:
  explicit MklPoolingBackwardOpBase<T>(OpKernelConstruction* context)
      : MklPoolingOpBase<T>(context) {}
  void Compute(OpKernelContext* context) override = 0;

 protected:
  const int kOutputTensorIndexOutput = 0;

  void AllocateOutputTensor(OpKernelContext* context,
                            const PoolingBwdPd& pool_bkwd_prim_desc,
                            const memory::dims output_dims_mkl_order,
                            const MklTensorFormat& output_tf_format,
                            Tensor** output_tensor) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_pooling_ops_commonDTh mht_34(mht_34_v, 986, "", "./tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h", "AllocateOutputTensor");

    DCHECK(output_tensor);
    memory::desc dst_pd = pool_bkwd_prim_desc.diff_src_desc();
    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(true);
    output_mkl_shape.SetMklLayout(&dst_pd);
    output_mkl_shape.SetElemType(MklDnnType<T>());
    output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                 output_dims_mkl_order, output_tf_format);

    TensorShape output_tf_shape;
    output_tf_shape.AddDim(this->GetNumTElements(dst_pd));
    if (this->native_format_) {
      output_tf_shape = output_mkl_shape.GetTfShape();
    }
    AllocateOutputSetMklShape(context, kOutputTensorIndexOutput, output_tensor,
                              output_tf_shape, output_mkl_shape,
                              this->native_format_);
    DCHECK(*output_tensor);
  }
};

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_POOLING_OPS_COMMON_H_
