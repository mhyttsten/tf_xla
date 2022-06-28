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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_MKL_QUANTIZED_CONV_OPS_H_
#define TENSORFLOW_CORE_KERNELS_MKL_MKL_QUANTIZED_CONV_OPS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_quantized_conv_opsDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_quantized_conv_opsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_quantized_conv_opsDTh() {
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

#ifdef INTEL_MKL

namespace tensorflow {
template <class T>
float MklFloatForOneQuantizedLevel(float range_min, float range_max) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_quantized_conv_opsDTh mht_0(mht_0_v, 195, "", "./tensorflow/core/kernels/mkl/mkl_quantized_conv_ops.h", "MklFloatForOneQuantizedLevel");

  int64 highest = static_cast<int64_t>(Eigen::NumTraits<T>::highest());
  int64 lowest = static_cast<int64_t>(Eigen::NumTraits<T>::lowest());

  // Adjusting for having a symmetric range.
  // for example: for 8-bit [-127, 127] as opposed to [-128, 127].
  if (lowest < -highest) ++lowest;

  const float float_for_one_quantized_level =
      (range_max - range_min) / (highest - lowest);
  return float_for_one_quantized_level;
}

template <class T1, class T2, class T3>
void MklQuantizationRangeForMultiplication(float min_a, float max_a,
                                           float min_b, float max_b,
                                           float* min_c, float* max_c) {
  const float a_float_for_one_quant_level =
      MklFloatForOneQuantizedLevel<T1>(min_a, max_a);
  const float b_float_for_one_quant_level =
      MklFloatForOneQuantizedLevel<T2>(min_b, max_b);

  const int64 c_highest = static_cast<int64_t>(Eigen::NumTraits<T3>::highest());
  const int64 c_lowest = static_cast<int64_t>(Eigen::NumTraits<T3>::lowest());
  const float c_float_for_one_quant_level =
      a_float_for_one_quant_level * b_float_for_one_quant_level;

  *min_c = c_float_for_one_quant_level * c_lowest;
  *max_c = c_float_for_one_quant_level * c_highest;
}

template <class T1, class T2, class T3>
void MklQuantizationRangeForMultiplication(float min_a, float max_a,
                                           const Tensor& min_b_vector,
                                           const Tensor& max_b_vector,
                                           Tensor** min_c_vector,
                                           Tensor** max_c_vector) {
  DCHECK(min_b_vector.NumElements() == (*min_c_vector)->NumElements());
  DCHECK(max_b_vector.NumElements() == (*max_c_vector)->NumElements());
  size_t n_channel = min_b_vector.NumElements();
  const int64 c_highest = static_cast<int64_t>(Eigen::NumTraits<T3>::highest());
  const int64 c_lowest = static_cast<int64_t>(Eigen::NumTraits<T3>::lowest());
  const float* min_b = min_b_vector.flat<float>().data();
  const float* max_b = max_b_vector.flat<float>().data();
  float* min_c = (*min_c_vector)->flat<float>().data();
  float* max_c = (*max_c_vector)->flat<float>().data();

#ifdef ENABLE_ONEDNN_OPENMP
#pragma omp parallel for
#endif  // ENABLE_ONEDNN_OPENMP
  // TODO(intel-tf): Add eigen parallel_for
  for (int64_t n = 0; n < n_channel; ++n) {
    float a_float_for_one_quant_level =
        MklFloatForOneQuantizedLevel<T1>(min_a, max_a);
    float b_float_for_one_quant_level =
        MklFloatForOneQuantizedLevel<T2>(min_b[n], max_b[n]);
    float c_float_for_one_quant_level =
        a_float_for_one_quant_level * b_float_for_one_quant_level;
    min_c[n] = c_float_for_one_quant_level * c_lowest;
    max_c[n] = c_float_for_one_quant_level * c_highest;
  }
}

}  // namespace tensorflow

#endif  // INTEL_MKL

#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_QUANTIZED_CONV_OPS_H_
