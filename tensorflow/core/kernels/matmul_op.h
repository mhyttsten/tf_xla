/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_MATMUL_OP_H_
#define TENSORFLOW_CORE_KERNELS_MATMUL_OP_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_opDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_opDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_opDTh() {
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


#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/hash/hash.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

namespace tensorflow {
namespace functor {

// Helpers to define tensor<T> needed by MatMul op.
template <typename T>
struct MatMulTypes {
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned>
      out_type;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                           Eigen::Aligned>
      in_type;
};

template <typename Device, typename In0, typename In1, typename Out,
          typename DimPair>
void MatMul(const Device& d, Out out, In0 in0, In1 in1,
            const DimPair& dim_pair) {
  out.device(d) = in0.contract(in1, dim_pair);
}

template <typename Device, typename T>
struct MatMulFunctor {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(
      const Device& d, typename MatMulTypes<T>::out_type out,
      typename MatMulTypes<T>::in_type in0,
      typename MatMulTypes<T>::in_type in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair);
};

}  // end namespace functor

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Encapsulate all the shape information that is used in matmul operations.
class MatmulParameters {
 public:
  MatmulParameters(bool transa, bool transb, uint64 m, uint64 n, uint64 k,
                   DataType dtype, int device_id)
      : transa_(transa),
        transb_(transb),
        m_(m),
        n_(n),
        k_(k),
        dtype_(dtype),
        device_id_(device_id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_opDTh mht_0(mht_0_v, 242, "", "./tensorflow/core/kernels/matmul_op.h", "MatmulParameters");

    hash_code_ = transa;
    hash_code_ = Hash64Combine(hash_code_, transb);
    hash_code_ = Hash64Combine(hash_code_, m);
    hash_code_ = Hash64Combine(hash_code_, n);
    hash_code_ = Hash64Combine(hash_code_, k);
    hash_code_ = Hash64Combine(hash_code_, dtype);
    hash_code_ = Hash64Combine(hash_code_, device_id);
  }
  bool operator==(const MatmulParameters& other) const {
    return this->get_data_as_tuple() == other.get_data_as_tuple();
  }

  bool operator!=(const MatmulParameters& other) const {
    return !(*this == other);
  }
  uint64 hash() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_opDTh mht_1(mht_1_v, 261, "", "./tensorflow/core/kernels/matmul_op.h", "hash");
 return hash_code_; }

  string ToString() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_opDTh mht_2(mht_2_v, 266, "", "./tensorflow/core/kernels/matmul_op.h", "ToString");

    // clang-format off
    return strings::StrCat(
        transa_, ", ", transb_, ", ",
        m_, ", ", n_, ", ", k_,
        dtype_, ", ", device_id_);
    // clang-format on
  }

 private:
  typedef std::tuple<bool, bool, int64, int64, int64, DataType, int>
      ParameterDataType;

  ParameterDataType get_data_as_tuple() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_opDTh mht_3(mht_3_v, 282, "", "./tensorflow/core/kernels/matmul_op.h", "get_data_as_tuple");

    return std::make_tuple(transa_, transb_, m_, n_, k_, dtype_, device_id_);
  }

  bool transa_;
  bool transb_;
  uint64 m_;
  uint64 n_;
  uint64 k_;
  DataType dtype_;
  int device_id_;
  uint64 hash_code_;
};

typedef Eigen::GpuDevice GPUDevice;

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MATMUL_OP_H_
