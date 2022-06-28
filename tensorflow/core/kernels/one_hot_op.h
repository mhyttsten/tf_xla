/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc

#ifndef TENSORFLOW_CORE_KERNELS_ONE_HOT_OP_H_
#define TENSORFLOW_CORE_KERNELS_ONE_HOT_OP_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSone_hot_opDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSone_hot_opDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSone_hot_opDTh() {
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

// Generator definition for OneHotOp, must be compilable by nvcc.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace generator {

template <typename T, typename TI>
class OneGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  OneGenerator(const typename TTypes<TI>::ConstMatrix& indices,
               const typename TTypes<T>::ConstScalar& on_value,
               const typename TTypes<T>::ConstScalar& off_value)
      : indices_(indices), on_value_(on_value), off_value_(off_value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSone_hot_opDTh mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/one_hot_op.h", "OneGenerator");
}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<Eigen::DenseIndex, 3>& pre_depth_suff) const {
    return (indices_(pre_depth_suff[0], pre_depth_suff[2]) == pre_depth_suff[1])
               ? on_value_()
               : off_value_();
  }

 private:
  const typename TTypes<TI>::ConstMatrix indices_;
  const typename TTypes<T>::ConstScalar on_value_;
  const typename TTypes<T>::ConstScalar off_value_;
};

}  // namespace generator

namespace functor {

template <typename Device, typename T, typename TI>
struct OneHot {
  EIGEN_ALWAYS_INLINE static void Compute(
      const Device& d, const typename TTypes<TI>::ConstMatrix& indices,
      const typename TTypes<T>::ConstScalar& on_value,
      const typename TTypes<T>::ConstScalar& off_value,
      typename TTypes<T, 3>::Tensor* output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSone_hot_opDTh mht_1(mht_1_v, 240, "", "./tensorflow/core/kernels/one_hot_op.h", "Compute");

    generator::OneGenerator<T, TI> generator(indices, on_value, off_value);
    output->device(d) = output->generate(generator);
  }
};

template <typename T, typename TI>
struct OneHot<CPUDevice, T, TI> {
  EIGEN_ALWAYS_INLINE static void Compute(
      const CPUDevice& d, const typename TTypes<TI>::ConstMatrix& indices,
      const typename TTypes<T>::ConstScalar& on_value,
      const typename TTypes<T>::ConstScalar& off_value,
      typename TTypes<T, 3>::Tensor* output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSone_hot_opDTh mht_2(mht_2_v, 255, "", "./tensorflow/core/kernels/one_hot_op.h", "Compute");

    // Pre-fill output with `off_value`.
    output->device(d) = output->constant(off_value());

    // Iterate through indices and update on_value elements in the output.
    Eigen::Index prefix_size = output->dimensions()[0];
    Eigen::Index depth_size = output->dimensions()[1];
    Eigen::Index suffix_size = output->dimensions()[2];

    // Cost of setting one `on_value` coefficient.
    double bytes_loaded = sizeof(T);
    double bytes_stored = sizeof(T);
    double cycles = 0.0;
    const Eigen::TensorOpCost cost(bytes_loaded, bytes_stored, cycles);

    if (suffix_size == 1) {
      const auto func = [&](Eigen::Index start, Eigen::Index end) -> void {
        for (Eigen::Index i = start; i < end; ++i) {
          const TI depth = internal::SubtleMustCopy(indices(i, 0));
          if (FastBoundsCheck(depth, depth_size)) {
            (*output)(i, depth, 0) = on_value();
          }
        }
      };
      d.parallelFor(prefix_size, cost, func);
    } else {
      const auto func = [&](Eigen::Index start, Eigen::Index end) -> void {
        for (Eigen::Index i = start; i < end; ++i) {
          const Eigen::Index d0 = i / suffix_size;
          const Eigen::Index d1 = i - (d0 * suffix_size);
          const TI depth = internal::SubtleMustCopy(indices(d0, d1));
          if (FastBoundsCheck(depth, depth_size)) {
            (*output)(d0, depth, d1) = on_value();
          }
        }
      };
      d.parallelFor(prefix_size * suffix_size, cost * suffix_size, func);
    }
  }
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_ONE_HOT_OP_H_
