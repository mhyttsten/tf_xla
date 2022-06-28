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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_MULTITHREADED_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_MULTITHREADED_CONV_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSmultithreaded_convDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSmultithreaded_convDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSmultithreaded_convDTh() {
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


#include <assert.h>
#include <stdint.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/optimized/eigen_spatial_convolutions.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace multithreaded_ops {

// Shorthands for the types we need when interfacing with the EigenTensor
// library.
typedef Eigen::TensorMap<
    Eigen::Tensor<float, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    EigenMatrix;
typedef Eigen::TensorMap<
    Eigen::Tensor<const float, 2, Eigen::RowMajor, Eigen::DenseIndex>,
    Eigen::Aligned>
    ConstEigenMatrix;

typedef Eigen::TensorMap<
    Eigen::Tensor<float, 4, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    EigenTensor;
typedef Eigen::TensorMap<
    Eigen::Tensor<const float, 4, Eigen::RowMajor, Eigen::DenseIndex>,
    Eigen::Aligned>
    ConstEigenTensor;

// Utility functions we need for the EigenTensor API.
template <typename Device, typename T>
struct MatMulConvFunctor {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(
      const Device& d, EigenMatrix out, ConstEigenMatrix in0,
      ConstEigenMatrix in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair) {
    out.device(d) = in0.contract(in1, dim_pair);
  }
};

template <class T>
class EigenTensorConvFunctor {
 private:
  Eigen::PaddingType RuntimePadding2EigenPadding(PaddingType padding) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSmultithreaded_convDTh mht_0(mht_0_v, 241, "", "./tensorflow/lite/kernels/internal/optimized/multithreaded_conv.h", "RuntimePadding2EigenPadding");

    switch (padding) {
      case PaddingType::kValid:
        return Eigen::PADDING_VALID;
      case PaddingType::kSame:
        return Eigen::PADDING_SAME;
      case PaddingType::kNone:
        assert(false);  // should never get here.
        return Eigen::PADDING_VALID;
    }
    return Eigen::PADDING_SAME;  // Prevent compiler warning about missing
                                 // return
  }

 public:
  void operator()(const Eigen::ThreadPoolDevice& device, const T* input_data,
                  int input_batches, int input_height, int input_width,
                  int input_depth, const T* filter_data, int filter_height,
                  int filter_width, int filter_count, int stride_rows,
                  int stride_cols, int pad_width, int pad_height,
                  PaddingType padding, T* output_data, int output_height,
                  int output_width) {
    const bool is_1x1_kernel = (filter_height == 1 && filter_width == 1 &&
                                stride_rows == 1 && stride_cols == 1);
    if (is_1x1_kernel) {
      // For 1x1 kernel, the 2D convolution is reduced to matrix
      // multiplication.
      const int conv_width = output_height * output_width;
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      EigenMatrix output(output_data, input_batches * conv_width, filter_count);
      ConstEigenMatrix input(input_data, input_batches * conv_width,
                             input_depth);
      ConstEigenMatrix filter(filter_data, input_depth, filter_count);
      MatMulConvFunctor<Eigen::ThreadPoolDevice, T>()(device, output, input,
                                                      filter, dim_pair);
    } else if (filter_height == input_height && filter_width == input_width &&
               pad_width == 0 && pad_height == 0) {
      // If the input data and filter have the same height/width,
      // the 2D convolution is reduced to matrix multiplication.
      const int k =  // Length of reduction dimension.
          filter_width * filter_height * input_depth;
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      EigenMatrix output(output_data, input_batches, filter_count);
      ConstEigenMatrix input(input_data, input_batches, k);
      ConstEigenMatrix filter(filter_data, k, filter_count);
      MatMulConvFunctor<Eigen::ThreadPoolDevice, T>()(device, output, input,
                                                      filter, dim_pair);
    } else {
      EigenTensor output(output_data, input_batches, output_height,
                         output_width, filter_count);
      ConstEigenTensor input(input_data, input_batches, input_height,
                             input_width, input_depth);
      ConstEigenTensor filter(filter_data, filter_height, filter_width,
                              input_depth, filter_count);
      output.device(device) =
          Eigen::SpatialConvolution(input, filter, stride_cols, stride_rows,
                                    RuntimePadding2EigenPadding(padding));
    }
  }
};

inline void Conv(const Eigen::ThreadPoolDevice& device,
                 const ConvParams& params, const RuntimeShape& input_shape,
                 const float* input_data, const RuntimeShape& filter_shape,
                 const float* filter_data, const RuntimeShape& bias_shape,
                 const float* bias_data, const RuntimeShape& output_shape,
                 float* output_data, const RuntimeShape& im2col_shape,
                 float* im2col_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSmultithreaded_convDTh mht_1(mht_1_v, 313, "", "./tensorflow/lite/kernels/internal/optimized/multithreaded_conv.h", "Conv");

  // Nest profiling under "Conv", to aggregate with other kernels.
  ruy::profiler::ScopeLabel label("Conv");
  ruy::profiler::ScopeLabel inner_label("Multithreaded EigenTensor");

  // im2col data should not be generated for the multi-thread supporting case.
  TFLITE_DCHECK(!im2col_data);
  (void)im2col_shape;
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const PaddingType padding = params.padding_type;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  EigenTensorConvFunctor<float> conv_functor;
  conv_functor(device, input_data, batches, input_height, input_width,
               input_depth, filter_data, filter_height, filter_width,
               output_depth, stride_height, stride_width, pad_height, pad_width,
               padding, output_data, output_height, output_width);

  optimized_ops::AddBiasAndEvalActivationFunction(
      output_activation_min, output_activation_max, bias_shape, bias_data,
      output_shape, output_data);
}

}  // namespace multithreaded_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_MULTITHREADED_CONV_H_
