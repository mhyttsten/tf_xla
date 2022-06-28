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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/operations.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {

Padding2D& Padding2D::operator=(const Padding2D& value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "=");

  prepended = value.prepended;
  appended = value.appended;
  return *this;
}

bool Padding2D::operator==(const Padding2D& value) {
  return this->prepended == value.prepended && this->appended == value.appended;
}

bool Padding2D::operator!=(const Padding2D& value) { return !(*this == value); }

Padding2D& Padding2D::operator-(const Padding2D& value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_1(mht_1_v, 217, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "-");

  prepended.h -= value.prepended.h;
  prepended.w -= value.prepended.w;
  appended.h -= value.appended.h;
  appended.w -= value.appended.w;
  return *this;
}

Padding3D& Padding3D::operator=(const Padding3D& value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_2(mht_2_v, 228, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "=");

  prepended = value.prepended;
  appended = value.appended;
  return *this;
}

bool Padding3D::operator==(const Padding3D& value) {
  return this->prepended == value.prepended && this->appended == value.appended;
}

bool Padding3D::operator!=(const Padding3D& value) { return !(*this == value); }

Padding3D& Padding3D::operator-(const Padding3D& value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_3(mht_3_v, 243, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "-");

  prepended.h -= value.prepended.h;
  prepended.w -= value.prepended.w;
  prepended.d -= value.prepended.d;
  appended.h -= value.appended.h;
  appended.w -= value.appended.w;
  appended.d -= value.appended.d;
  return *this;
}

std::string ToString(enum OperationType op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_4(mht_4_v, 256, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "ToString");

  switch (op) {
    case OperationType::ABS:
      return "abs";
    case OperationType::ADD:
      return "add";
    case OperationType::BATCH_NORMALIZATION:
      return "batch_normalization";
    case OperationType::BATCH_TO_SPACE:
      return "batch_to_space";
    case OperationType::BATCHED_MATMUL:
      return "batched_matmul";
    case OperationType::CONCAT:
      return "concat";
    case OperationType::CONSTANT:
      return "const";
    case OperationType::CONVOLUTION_2D:
      return "convolution_2d";
    case OperationType::CONVOLUTION_TRANSPOSED:
      return "convolution_transposed";
    case OperationType::COPY:
      return "copy";
    case OperationType::COS:
      return "cos";
    case OperationType::DENSIFY:
      return "densify";
    case OperationType::DEPTHWISE_CONVOLUTION:
      return "depthwise_convolution";
    case OperationType::DEPTH_TO_SPACE:
      return "depth_to_space";
    case OperationType::DIV:
      return "div";
    case OperationType::ELU:
      return "elu";
    case OperationType::EQUAL:
      return "equal";
    case OperationType::EXP:
      return "exp";
    case OperationType::FLOOR:
      return "floor";
    case OperationType::FLOOR_DIV:
      return "floor_div";
    case OperationType::FLOOR_MOD:
      return "floor_mod";
    case OperationType::FULLY_CONNECTED:
      return "fully_connected";
    case OperationType::FULLY_CONNECTED_INT8:
      return "fully_connected_int8";
    case OperationType::GATHER:
      return "gather";
    case OperationType::GREATER:
      return "greater";
    case OperationType::GREATER_EQUAL:
      return "greater_equal";
    case OperationType::HARD_SWISH:
      return "hard_swish";
    case OperationType::LESS:
      return "less";
    case OperationType::LESS_EQUAL:
      return "less_equal";
    case OperationType::LOG:
      return "log";
    case OperationType::LSTM:
      return "lstm";
    case OperationType::MAXIMUM:
      return "maximum";
    case OperationType::MAX_UNPOOLING_2D:
      return "max_unpooling";
    case OperationType::MEAN:
      return "mean";
    case OperationType::MEAN_STDDEV_NORMALIZATION:
      return "mean_stddev_normalization";
    case OperationType::MINIMUM:
      return "minimum";
    case OperationType::MUL:
      return "mul";
    case OperationType::NEG:
      return "neg";
    case OperationType::NOT_EQUAL:
      return "not_equal";
    case OperationType::PAD:
      return "pad";
    case OperationType::POOLING_2D:
      return "pooling_2d";
    case OperationType::POW:
      return "pow";
    case OperationType::PRELU:
      return "prelu";
    case OperationType::QUANTIZE_AND_DEQUANTIZE:
      return "quantize_and_dequantize";
    case OperationType::REDUCE_MAXIMUM:
      return "reduce_maximum";
    case OperationType::REDUCE_MINIMUM:
      return "reduce_minimum";
    case OperationType::REDUCE_PRODUCT:
      return "reduce_product";
    case OperationType::REDUCE_SUM:
      return "reduce_sum";
    case OperationType::RELU:
      return "relu";
    case OperationType::RESAMPLER:
      return "resampler";
    case OperationType::RESHAPE:
      return "reshape";
    case OperationType::RESIZE:
      return "resize";
    case OperationType::RSQRT:
      return "rsqrt";
    case OperationType::SIGMOID:
      return "sigmoid";
    case OperationType::SIN:
      return "sin";
    case OperationType::SLICE:
      return "slice";
    case OperationType::SOFTMAX:
      return "softmax";
    case OperationType::SPACE_TO_BATCH:
      return "space_to_batch";
    case OperationType::SPACE_TO_DEPTH:
      return "space_to_depth";
    case OperationType::SPLIT:
      return "split";
    case OperationType::SQRT:
      return "sqrt";
    case OperationType::SQUARE:
      return "square";
    case OperationType::SQUARED_DIFF:
      return "squared_diff";
    case OperationType::SUB:
      return "subtract";
    case OperationType::TANH:
      return "tanh";
    case OperationType::TILE:
      return "tile";
    case OperationType::TRANSPOSE:
      return "transpose";
    case OperationType::UNKNOWN:
      return "unknown_operation";
  }
}

OperationType OperationTypeFromString(const std::string& name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_5(mht_5_v, 401, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "OperationTypeFromString");

  static const auto* operations =
      new std::unordered_map<std::string, OperationType>({
          {"abs", OperationType::ABS},
          {"add", OperationType::ADD},
          {"batch_normalization", OperationType::BATCH_NORMALIZATION},
          {"batched_matmul", OperationType::BATCHED_MATMUL},
          {"concat", OperationType::CONCAT},
          {"const", OperationType::CONSTANT},
          {"convolution_2d", OperationType::CONVOLUTION_2D},
          {"convolution_transposed", OperationType::CONVOLUTION_TRANSPOSED},
          {"copy", OperationType::COPY},
          {"cos", OperationType::COS},
          {"densify", OperationType::DENSIFY},
          {"depthwise_convolution", OperationType::DEPTHWISE_CONVOLUTION},
          {"depth_to_space", OperationType::DEPTH_TO_SPACE},
          {"div", OperationType::DIV},
          {"elu", OperationType::ELU},
          {"equal", OperationType::EQUAL},
          {"exp", OperationType::EXP},
          {"floor", OperationType::FLOOR},
          {"floor_div", OperationType::FLOOR_DIV},
          {"floor_mod", OperationType::FLOOR_MOD},
          {"fully_connected", OperationType::FULLY_CONNECTED},
          {"fully_connected_int8", OperationType::FULLY_CONNECTED_INT8},
          {"gather", OperationType::GATHER},
          {"greater", OperationType::GREATER},
          {"greater_equal", OperationType::GREATER_EQUAL},
          {"hard_swish", OperationType::HARD_SWISH},
          {"less", OperationType::LESS},
          {"less_equal", OperationType::LESS_EQUAL},
          {"log", OperationType::LOG},
          {"lstm", OperationType::LSTM},
          {"maximum", OperationType::MAXIMUM},
          {"max_unpooling", OperationType::MAX_UNPOOLING_2D},
          {"mean", OperationType::MEAN},
          {"mean_stddev_normalization",
           OperationType::MEAN_STDDEV_NORMALIZATION},
          {"minimum", OperationType::MINIMUM},
          {"mul", OperationType::MUL},
          {"neg", OperationType::NEG},
          {"not_equal", OperationType::NOT_EQUAL},
          {"pad", OperationType::PAD},
          {"pooling_2d", OperationType::POOLING_2D},
          {"pow", OperationType::POW},
          {"prelu", OperationType::PRELU},
          {"quantize_and_dequantize", OperationType::QUANTIZE_AND_DEQUANTIZE},
          {"reduce_maximum", OperationType::REDUCE_MAXIMUM},
          {"reduce_minimum", OperationType::REDUCE_MINIMUM},
          {"reduce_product", OperationType::REDUCE_PRODUCT},
          {"reduce_sum", OperationType::REDUCE_SUM},
          {"relu", OperationType::RELU},
          {"resampler", OperationType::RESAMPLER},
          {"resize", OperationType::RESIZE},
          {"reshape", OperationType::RESHAPE},
          {"rsqrt", OperationType::RSQRT},
          {"sigmoid", OperationType::SIGMOID},
          {"sin", OperationType::SIN},
          {"slice", OperationType::SLICE},
          {"softmax", OperationType::SOFTMAX},
          {"space_to_depth", OperationType::SPACE_TO_DEPTH},
          {"split", OperationType::SPLIT},
          {"sqrt", OperationType::SQRT},
          {"square", OperationType::SQUARE},
          {"squared_diff", OperationType::SQUARED_DIFF},
          {"subtract", OperationType::SUB},
          {"tanh", OperationType::TANH},
          {"tile", OperationType::TILE},
          {"transpose", OperationType::TRANSPOSE},
      });
  auto op = operations->find(name);
  return op == operations->end() ? OperationType::UNKNOWN : op->second;
}

namespace {

template <typename T>
T DivideRoundUp(T n, T divisor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_6(mht_6_v, 481, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "DivideRoundUp");

  return (n - 1) / divisor + 1;
}

int32_t CalculateOutputSizeBeforeStrides(int32_t input, int32_t kernel,
                                         int32_t padding, int32_t dilation) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_7(mht_7_v, 489, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputSizeBeforeStrides");

  const int32_t dilated_kernel = (kernel - 1) * dilation + 1;
  return input + padding - dilated_kernel + 1;
}

template <Axis T>
int32_t CalculateOutputWithoutStrides(const BHWC& input,
                                      const Convolution2DAttributes& attr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_8(mht_8_v, 499, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputWithoutStrides");

  return CalculateOutputSizeBeforeStrides(
      input.get<T>(), attr.weights.shape.get<T>(),
      attr.padding.prepended.get<T>() + attr.padding.appended.get<T>(),
      attr.dilations.get<T>());
}

template <Axis T>
int32_t CalculateOutputWithoutStrides(const BHWDC& input,
                                      const Convolution3DAttributes& attr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_9(mht_9_v, 511, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputWithoutStrides");

  return CalculateOutputSizeBeforeStrides(
      input.get<T>(), attr.weights.shape.get<T>(),
      attr.padding.prepended.get<T>() + attr.padding.appended.get<T>(),
      attr.dilations.get<T>());
}

template <Axis T>
int32_t CalculateOutputWithoutStrides(const BHWC& input,
                                      const Pooling2DAttributes& attr) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_10(mht_10_v, 523, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputWithoutStrides");

  return CalculateOutputSizeBeforeStrides(
      input.get<T>(), attr.kernel.get<T>(),
      attr.padding.prepended.get<T>() + attr.padding.appended.get<T>(),
      /*dilation=*/1);
}

template <Axis T>
int32_t CalculateOutputWithoutStrides(const BHWDC& input,
                                      const Pooling3DAttributes& attr) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_11(mht_11_v, 535, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputWithoutStrides");

  return CalculateOutputSizeBeforeStrides(
      input.get<T>(), attr.kernel.get<T>(),
      attr.padding.prepended.get<T>() + attr.padding.appended.get<T>(),
      /*dilation=*/1);
}

template <Axis T>
int32_t CalculateOutput(const BHWC& input,
                        const ConvolutionTransposedAttributes& attr) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_12(mht_12_v, 547, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutput");

  return (input.get<T>() - 1) * attr.stride.get<T>() -
         (attr.padding.prepended.get<T>() + attr.padding.appended.get<T>()) +
         attr.weights.shape.get<T>() + attr.adjacent.get<T>();
}

template <Axis T>
int32_t CalculateOutput(const BHWDC& input,
                        const ConvolutionTransposed3DAttributes& attr) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_13(mht_13_v, 558, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutput");

  return (input.get<T>() - 1) * attr.stride.get<T>() -
         (attr.padding.prepended.get<T>() + attr.padding.appended.get<T>()) +
         attr.weights.shape.get<T>();
}

inline int32_t StridedSize(int32_t size, int32_t stride) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_14(mht_14_v, 567, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "StridedSize");

  return stride == 0 ? -1 : DivideRoundUp(size, stride);
}

template <Axis AxisT, typename AttrT>
int32_t CalculateOutput(const BHWC& input, const AttrT& attr) {
  return StridedSize(CalculateOutputWithoutStrides<AxisT>(input, attr),
                     attr.strides.template get<AxisT>());
}

template <Axis AxisT, typename AttrT>
int32_t CalculateOutput(const BHWDC& input, const AttrT& attr) {
  return StridedSize(CalculateOutputWithoutStrides<AxisT>(input, attr),
                     attr.strides.template get<AxisT>());
}

int32_t CalculateSamePadding(int32_t input, int32_t kernel, int32_t dilation,
                             int32_t stride) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_15(mht_15_v, 587, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  const int32_t dilated_kernel = (kernel - 1) * dilation + 1;
  return std::max(0, dilated_kernel - (input - 1) % stride - 1);
}

// Returns a padding that should be present to make sure image size stays
// the same.
template <Axis AxisT>
int32_t CalculateSamePadding(const BHWC& input,
                             const Convolution2DAttributes& attr) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_16(mht_16_v, 599, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return CalculateSamePadding(
      input.get<AxisT>(), attr.weights.shape.get<AxisT>(),
      attr.dilations.get<AxisT>(), attr.strides.get<AxisT>());
}

// Returns a padding that should be present to make sure image size stays
// the same.
template <Axis AxisT>
int32_t CalculateSamePadding(const BHWDC& input,
                             const Convolution3DAttributes& attr) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_17(mht_17_v, 612, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return CalculateSamePadding(
      input.get<AxisT>(), attr.weights.shape.get<AxisT>(),
      attr.dilations.get<AxisT>(), attr.strides.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWC& input,
                             const ConvolutionTransposedAttributes& attr) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_18(mht_18_v, 623, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return CalculateSamePadding(input.get<AxisT>(),
                              attr.weights.shape.get<AxisT>(),
                              /*dilation=*/1, attr.stride.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWDC& input,
                             const ConvolutionTransposed3DAttributes& attr) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_19(mht_19_v, 634, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return CalculateSamePadding(input.get<AxisT>(),
                              attr.weights.shape.get<AxisT>(),
                              /*dilation=*/1, attr.stride.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWC& input,
                             const Pooling2DAttributes& attr) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_20(mht_20_v, 645, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return CalculateSamePadding(input.get<AxisT>(), attr.kernel.get<AxisT>(),
                              /*dilation=*/1, attr.strides.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWDC& input,
                             const Pooling3DAttributes& attr) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_21(mht_21_v, 655, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return CalculateSamePadding(input.get<AxisT>(), attr.kernel.get<AxisT>(),
                              /*dilation=*/1, attr.strides.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWC& input,
                             const MaxUnpooling2DAttributes& attr) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_22(mht_22_v, 665, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return CalculateSamePadding(input.get<AxisT>(), attr.kernel.get<AxisT>(),
                              /*dilation=*/1, attr.strides.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWDC& input,
                             const MaxUnpooling3DAttributes& attr) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_23(mht_23_v, 675, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return CalculateSamePadding(input.get<AxisT>(), attr.kernel.get<AxisT>(),
                              /*dilation=*/1, attr.strides.get<AxisT>());
}

Padding2D MakeSamePadding(const BHWC& input,
                          const ConvolutionTransposedAttributes& attr) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_24(mht_24_v, 684, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "MakeSamePadding");

  int32_t padding_height = CalculateSamePadding<Axis::HEIGHT>(input, attr);
  int32_t padding_width = CalculateSamePadding<Axis::WIDTH>(input, attr);
  Padding2D padding;
  padding.prepended = HW(padding_height / 2, padding_width / 2);
  padding.appended = HW(padding_height - padding_height / 2,
                        padding_width - padding_width / 2);
  return padding;
}

Padding3D MakeSamePadding(const BHWDC& input,
                          const ConvolutionTransposed3DAttributes& attr) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_25(mht_25_v, 698, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "MakeSamePadding");

  int32_t padding_height = CalculateSamePadding<Axis::HEIGHT>(input, attr);
  int32_t padding_width = CalculateSamePadding<Axis::WIDTH>(input, attr);
  int32_t padding_depth = CalculateSamePadding<Axis::DEPTH>(input, attr);
  Padding3D padding;
  padding.prepended =
      HWD(padding_height / 2, padding_width / 2, padding_depth / 2);
  padding.appended =
      HWD(padding_height - padding_height / 2,
          padding_width - padding_width / 2, padding_depth - padding_depth / 2);
  return padding;
}

// If padding depends on input, convert it into fixed padding.
template <class AttrT>
Padding2D MakeSamePadding(const BHWC& input, const AttrT& attr) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_26(mht_26_v, 716, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "MakeSamePadding");

  int32_t padding_height = CalculateSamePadding<Axis::HEIGHT>(input, attr);
  int32_t padding_width = CalculateSamePadding<Axis::WIDTH>(input, attr);
  Padding2D padding;
  padding.prepended = HW(padding_height / 2, padding_width / 2);
  padding.appended = HW(padding_height - padding_height / 2,
                        padding_width - padding_width / 2);
  return padding;
}

// If padding depends on input, convert it into fixed padding.
template <class AttrT>
Padding3D MakeSamePadding(const BHWDC& input, const AttrT& attr) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_27(mht_27_v, 731, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "MakeSamePadding");

  int32_t padding_height = CalculateSamePadding<Axis::HEIGHT>(input, attr);
  int32_t padding_width = CalculateSamePadding<Axis::WIDTH>(input, attr);
  int32_t padding_depth = CalculateSamePadding<Axis::DEPTH>(input, attr);
  Padding3D padding;
  padding.prepended =
      HWD(padding_height / 2, padding_width / 2, padding_depth / 2);
  padding.appended =
      HWD(padding_height - padding_height / 2,
          padding_width - padding_width / 2, padding_depth - padding_depth / 2);
  return padding;
}

}  // namespace

BHWC CalculateOutputShape(const BHWC& input,
                          const MaxUnpooling2DAttributes& attr) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_28(mht_28_v, 750, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWC(input.b,
              input.h * attr.strides.h - attr.padding.prepended.h -
                  attr.padding.appended.h,
              input.w * attr.strides.w - attr.padding.prepended.w -
                  attr.padding.appended.w,
              input.c);
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const MaxUnpooling3DAttributes& attr) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_29(mht_29_v, 763, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWDC(input.b,
               input.h * attr.strides.h - attr.padding.prepended.h -
                   attr.padding.appended.h,
               input.w * attr.strides.w - attr.padding.prepended.w -
                   attr.padding.appended.w,
               input.d * attr.strides.d - attr.padding.prepended.d -
                   attr.padding.appended.d,
               input.c);
}

BHWC CalculateOutputShape(const BHWC& input, const Pooling2DAttributes& attr) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_30(mht_30_v, 777, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
              CalculateOutput<Axis::WIDTH>(input, attr), input.c);
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const Pooling3DAttributes& attr) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_31(mht_31_v, 786, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWDC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
               CalculateOutput<Axis::WIDTH>(input, attr),
               CalculateOutput<Axis::DEPTH>(input, attr), input.c);
}

BHWC CalculateOutputShape(const BHWC& input,
                          const Convolution2DAttributes& attr) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_32(mht_32_v, 796, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
              CalculateOutput<Axis::WIDTH>(input, attr),
              attr.weights.shape.get<Axis::OUTPUT_CHANNELS>());
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const Convolution3DAttributes& attr) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_33(mht_33_v, 806, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWDC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
               CalculateOutput<Axis::WIDTH>(input, attr),
               CalculateOutput<Axis::DEPTH>(input, attr),
               attr.weights.shape.get<Axis::OUTPUT_CHANNELS>());
}

BHWC CalculateOutputShape(const BHWC& input,
                          const ConvolutionTransposedAttributes& attr) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_34(mht_34_v, 817, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
              CalculateOutput<Axis::WIDTH>(input, attr),
              attr.weights.shape.get<Axis::OUTPUT_CHANNELS>());
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const ConvolutionTransposed3DAttributes& attr) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_35(mht_35_v, 827, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWDC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
               CalculateOutput<Axis::WIDTH>(input, attr),
               CalculateOutput<Axis::DEPTH>(input, attr),
               attr.weights.shape.get<Axis::OUTPUT_CHANNELS>());
}

BHWC CalculateOutputShape(const BHWC& input,
                          const DepthwiseConvolution2DAttributes& attr) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_36(mht_36_v, 838, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
              CalculateOutput<Axis::WIDTH>(input, attr),
              attr.weights.shape.get<Axis::OUTPUT_CHANNELS>() *
                  attr.weights.shape.get<Axis::INPUT_CHANNELS>());
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const DepthwiseConvolution3DAttributes& attr) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_37(mht_37_v, 849, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWDC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
               CalculateOutput<Axis::WIDTH>(input, attr),
               CalculateOutput<Axis::DEPTH>(input, attr),
               attr.weights.shape.get<Axis::OUTPUT_CHANNELS>() *
                   attr.weights.shape.get<Axis::INPUT_CHANNELS>());
}

BHWC CalculateOutputShape(const BHWC& input, const SliceAttributes& attr) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_38(mht_38_v, 860, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWC(StridedSize(attr.ends.b - attr.starts.b, attr.strides.b),
              StridedSize(attr.ends.h - attr.starts.h, attr.strides.h),
              StridedSize(attr.ends.w - attr.starts.w, attr.strides.w),
              StridedSize(attr.ends.c - attr.starts.c, attr.strides.c));
}

BHWDC CalculateOutputShape(const BHWDC& input, const Slice3DAttributes& attr) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_39(mht_39_v, 870, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWDC(StridedSize(attr.ends.b - attr.starts.b, attr.strides.b),
               StridedSize(attr.ends.h - attr.starts.h, attr.strides.h),
               StridedSize(attr.ends.w - attr.starts.w, attr.strides.w),
               StridedSize(attr.ends.d - attr.starts.d, attr.strides.d),
               StridedSize(attr.ends.c - attr.starts.c, attr.strides.c));
}

BHWC CalculateOutputShape(const BHWC& input, const PadAttributes& attr) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_40(mht_40_v, 881, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWC(attr.appended.b + attr.prepended.b + input.b,
              attr.appended.h + attr.prepended.h + input.h,
              attr.appended.w + attr.prepended.w + input.w,
              attr.appended.c + attr.prepended.c + input.c);
}

BHWDC CalculateOutputShape(const BHWDC& input, const Pad3DAttributes& attr) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_41(mht_41_v, 891, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWDC(attr.appended.b + attr.prepended.b + input.b,
               attr.appended.h + attr.prepended.h + input.h,
               attr.appended.w + attr.prepended.w + input.w,
               attr.appended.d + attr.prepended.d + input.d,
               attr.appended.c + attr.prepended.c + input.c);
}

BHWC CalculateOutputShape(const BHWC& input,
                          const FullyConnectedAttributes& attr) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_42(mht_42_v, 903, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWC(input.b, 1, 1, attr.weights.shape.o);
}

BHWC CalculateOutputShape(const BHWC& input, const MeanAttributes& attr) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_43(mht_43_v, 910, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  const int b = attr.dims.find(Axis::BATCH) == attr.dims.end() ? input.b : 1;
  const int h = attr.dims.find(Axis::HEIGHT) == attr.dims.end() ? input.h : 1;
  const int w = attr.dims.find(Axis::WIDTH) == attr.dims.end() ? input.w : 1;
  const int c = attr.dims.find(Axis::CHANNELS) == attr.dims.end() ? input.c : 1;
  return BHWC(b, h, w, c);
}

BHWDC CalculateOutputShape(const BHWDC& input, const MeanAttributes& attr) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_44(mht_44_v, 921, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  const int b = attr.dims.find(Axis::BATCH) == attr.dims.end() ? input.b : 1;
  const int h = attr.dims.find(Axis::HEIGHT) == attr.dims.end() ? input.h : 1;
  const int w = attr.dims.find(Axis::WIDTH) == attr.dims.end() ? input.w : 1;
  const int d = attr.dims.find(Axis::DEPTH) == attr.dims.end() ? input.d : 1;
  const int c = attr.dims.find(Axis::CHANNELS) == attr.dims.end() ? input.c : 1;
  return BHWDC(b, h, w, d, c);
}

absl::Status CalculateOutputShape(const std::vector<BHWC>& input,
                                  const ConcatAttributes& attr,
                                  BHWC* output_shape) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_45(mht_45_v, 935, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  BHWC new_shape = input[0];
  switch (attr.axis) {
    case Axis::CHANNELS:
      for (int i = 1; i < input.size(); i++) {
        if (input[i].h != new_shape.h || input[i].w != new_shape.w ||
            input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Height, Width and Batch must be the same when concatenating "
              "by channels axis");
        }
        new_shape.c += input[i].c;
      }
      break;
    case Axis::HEIGHT:
      for (int i = 1; i < input.size(); i++) {
        if (input[i].w != new_shape.w || input[i].c != new_shape.c ||
            input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Channels, Width and Batch must be the same when concatenating "
              "by height axis");
        }
        new_shape.h += input[i].h;
      }
      break;
    case Axis::WIDTH:
      for (int i = 1; i < input.size(); i++) {
        if (input[i].h != new_shape.h || input[i].c != new_shape.c ||
            input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Height, Channels and Batch must be the same when concatenating "
              "by width axis");
        }
        new_shape.w += input[i].w;
      }
      break;
    case Axis::BATCH:
      for (int i = 1; i < input.size(); i++) {
        if (input[i].h != new_shape.h || input[i].c != new_shape.c ||
            input[i].w != new_shape.w) {
          return absl::InvalidArgumentError(
              "Width, Height and Channels must be the same when concatenating "
              "by batch axis");
        }
        new_shape.b += input[i].b;
      }
      break;
    default:
      return absl::InvalidArgumentError("Invalid axis");
      break;
  }
  *output_shape = new_shape;
  return absl::OkStatus();
}

absl::Status CalculateOutputShape(const std::vector<BHWDC>& input,
                                  const ConcatAttributes& attr,
                                  BHWDC* output_shape) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_46(mht_46_v, 995, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  BHWDC new_shape = input[0];
  switch (attr.axis) {
    case Axis::CHANNELS:
      for (int i = 1; i < input.size(); ++i) {
        if (input[i].h != new_shape.h || input[i].w != new_shape.w ||
            input[i].d != new_shape.d || input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Height, Width, Batch and Depth must be the same when "
              "concatenating "
              "by channels axis");
        }
        new_shape.c += input[i].c;
      }
      break;
    case Axis::HEIGHT:
      for (int i = 1; i < input.size(); ++i) {
        if (input[i].w != new_shape.w || input[i].c != new_shape.c ||
            input[i].d != new_shape.d || input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Width, Depth, Batch and Channels must be the same when "
              "concatenating "
              "by height axis");
        }
        new_shape.h += input[i].h;
      }
      break;
    case Axis::WIDTH:
      for (int i = 1; i < input.size(); ++i) {
        if (input[i].h != new_shape.h || input[i].c != new_shape.c ||
            input[i].d != new_shape.d || input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Height, Depth, Batch and Channels must be the same when "
              "concatenating "
              "by width axis");
        }
        new_shape.w += input[i].w;
      }
      break;
    case Axis::DEPTH:
      for (int i = 1; i < input.size(); ++i) {
        if (input[i].w != new_shape.w || input[i].h != new_shape.h ||
            input[i].c != new_shape.c || input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Width, Height, Batch and Channels must be the same when "
              "concatenating "
              "by depth axis");
        }
        new_shape.d += input[i].d;
      }
      break;
    case Axis::BATCH:
      for (int i = 1; i < input.size(); ++i) {
        if (input[i].w != new_shape.w || input[i].h != new_shape.h ||
            input[i].c != new_shape.c || input[i].d != new_shape.d) {
          return absl::InvalidArgumentError(
              "Width, Height, Depth and Channels must be the same when "
              "concatenating "
              "by batch axis");
        }
        new_shape.b += input[i].b;
      }
      break;
    default:
      return absl::InvalidArgumentError("Invalid axis");
  }
  *output_shape = new_shape;
  return absl::OkStatus();
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const Convolution2DAttributes& attr) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_47(mht_47_v, 1069, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return MakeSamePadding(input, attr);
}

Padding3D CalculateSamePadding(const BHWDC& input,
                               const Convolution3DAttributes& attr) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_48(mht_48_v, 1077, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return MakeSamePadding(input, attr);
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const ConvolutionTransposedAttributes& attr) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_49(mht_49_v, 1085, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return MakeSamePadding(input, attr);
}

Padding3D CalculateSamePadding(const BHWDC& input,
                               const ConvolutionTransposed3DAttributes& attr) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_50(mht_50_v, 1093, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return MakeSamePadding(input, attr);
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const DepthwiseConvolution2DAttributes& attr) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_51(mht_51_v, 1101, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return MakeSamePadding(input, attr);
}

Padding3D CalculateSamePadding(const BHWDC& input,
                               const DepthwiseConvolution3DAttributes& attr) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_52(mht_52_v, 1109, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return MakeSamePadding(input, attr);
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const Pooling2DAttributes& attr) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_53(mht_53_v, 1117, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return MakeSamePadding(input, attr);
}

Padding3D CalculateSamePadding(const BHWDC& input,
                               const Pooling3DAttributes& attr) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_54(mht_54_v, 1125, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return MakeSamePadding(input, attr);
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const MaxUnpooling2DAttributes& attr) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_55(mht_55_v, 1133, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return MakeSamePadding(input, attr);
}

Padding3D CalculateSamePadding(const BHWDC& input,
                               const MaxUnpooling3DAttributes& attr) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_56(mht_56_v, 1141, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateSamePadding");

  return MakeSamePadding(input, attr);
}

float CalculateResizeScale(int32_t input_size, int32_t output_size,
                           const Resize2DAttributes& attr) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_57(mht_57_v, 1149, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateResizeScale");

  return attr.align_corners && input_size > 1 && output_size > 1
             ? static_cast<float>(input_size - 1) / (output_size - 1)
             : static_cast<float>(input_size) / output_size;
}

float CalculateResizeScale(int32_t input_size, int32_t output_size,
                           const Resize3DAttributes& attr) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_58(mht_58_v, 1159, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateResizeScale");

  return attr.align_corners && input_size > 1 && output_size > 1
             ? static_cast<float>(input_size - 1) / (output_size - 1)
             : static_cast<float>(input_size) / output_size;
}

BHWC CalculateOutputShape(const BHWC& input, const Resize2DAttributes& attr) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_59(mht_59_v, 1168, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWC(input.b, attr.new_shape.h, attr.new_shape.w, input.c);
}

BHWDC CalculateOutputShape(const BHWDC& input, const Resize3DAttributes& attr) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_60(mht_60_v, 1175, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWDC(input.b, attr.new_shape.h, attr.new_shape.w, attr.new_shape.d,
               input.c);
}

BHWC CalculateOutputShape(const BHWC& input, const TransposeAttributes& attr) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_61(mht_61_v, 1183, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWC(input.get(attr.perm.b), input.get(attr.perm.h),
              input.get(attr.perm.w), input.get(attr.perm.c));
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const Transpose3DAttributes& attr) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_62(mht_62_v, 1192, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "CalculateOutputShape");

  return BHWDC(input.get(attr.perm.b), input.get(attr.perm.h),
               input.get(attr.perm.w), input.get(attr.perm.d),
               input.get(attr.perm.c));
}

FullyConnectedAttributes DequatizeFullyConnectedAttr(
    const FullyConnectedInt8Attributes& attr) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSoperationsDTcc mht_63(mht_63_v, 1202, "", "./tensorflow/lite/delegates/gpu/common/operations.cc", "DequatizeFullyConnectedAttr");

  FullyConnectedAttributes dequant_attr;
  dequant_attr.weights.id = attr.weights.id;
  dequant_attr.weights.shape = attr.weights.shape;
  dequant_attr.weights.data.resize(
      dequant_attr.weights.shape.DimensionsProduct());
  dequant_attr.bias = attr.bias;

  // weights dequantization to float32
  for (int i = 0; i < attr.weights.data.size(); i++) {
    const int32_t val = attr.weights.data[i];
    dequant_attr.weights.data[i] = attr.scale * (val - attr.zero_point);
  }
  return dequant_attr;
}

}  // namespace gpu
}  // namespace tflite
