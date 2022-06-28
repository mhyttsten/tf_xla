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
#ifndef TENSORFLOW_LITE_KERNELS_KERNEL_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_KERNEL_UTIL_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_utilDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_utilDTh() {
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


#include <stdint.h>

#include <limits>
#ifndef TF_LITE_STATIC_MEMORY
#include <string>
#endif  // TF_LITE_STATIC_MEMORY

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {

// A fair number of functions in this header have historically been inline.
// It is ok to change functions to not be inline if the latency with
// benchmark_model for MobileNet + MobileBERT is unaffected. If such a change is
// made, move the newly non-inlined function declarations to the top of this
// header file.

// Note: You must check if result is not null:
//
//   TfLiteTensor* my_tensor = GetInput(context, node, kMyTensorIdx);
//   TF_LITE_ENSURE(context, my_tensor != nullptr);
//
// This is because the index might point to the optional tensor constant
// (kTfLiteOptionalTensor) in which case there is no tensor to return.
const TfLiteTensor* GetInput(const TfLiteContext* context,
                             const TfLiteNode* node, int index);

// Same as `GetInput` but returns boolean and uses output argument for tensor.
//
//   TfLiteTensor* my_tensor;
//   TF_LITE_ENSURE_OK(context,
//                     GetInputSafe(context, node, kMyTensorIdx, &my_tensor));
//   // can use my_tensor directly from here onwards, it is not nullptr
//
// Should be used in cases where the binary size is too large.
TfLiteStatus GetInputSafe(const TfLiteContext* context, const TfLiteNode* node,
                          int index, const TfLiteTensor** tensor);

// Note: You must check if result is not null:
//
//   TfLiteTensor* my_tensor = GetVariableInput(context, node, kMyTensorIdx);
//   TF_LITE_ENSURE(context, my_tensor != nullptr);
//
// This is because the index might point to the optional tensor constant
// (kTfLiteOptionalTensor) in which case there is no tensor to return.
TfLiteTensor* GetVariableInput(TfLiteContext* context, const TfLiteNode* node,
                               int index);

// Note: You must check if result is not null:
//
//   TfLiteTensor* my_tensor = GetOutput(context, node, kMyTensorIdx);
//   TF_LITE_ENSURE(context, my_tensor != nullptr);
//
// This is because the index might point to the optional tensor constant
// (kTfLiteOptionalTensor) in which case there is no tensor to return.
TfLiteTensor* GetOutput(TfLiteContext* context, const TfLiteNode* node,
                        int index);

// Same as `GetOutput` but returns boolean and uses output argument for tensor.
//
//   TfLiteTensor* my_tensor;
//   TF_LITE_ENSURE_OK(context,
//                     GetOutputSafe(context, node, kMyTensorIdx, &my_tensor));
//   // can use my_tensor directly from here onwards, it is not nullptr
//
// Should be used in cases where the binary size is too large.
TfLiteStatus GetOutputSafe(const TfLiteContext* context, const TfLiteNode* node,
                           int index, TfLiteTensor** tensor);

// Note: You must check if result is not null:
//
//   TfLiteTensor* my_tensor = GetOptionalInputTensor(context, node, kIdx);
//   TF_LITE_ENSURE(context, my_tensor != nullptr);
//
// This is because the index might point to the optional tensor constant
// (kTfLiteOptionalTensor) in which case there is no tensor to return.
//
// Deprecated. GetInput has the same functionality.
const TfLiteTensor* GetOptionalInputTensor(const TfLiteContext* context,
                                           const TfLiteNode* node, int index);

#ifndef TF_LITE_STATIC_MEMORY
// Note: You must check if result is not null:
//
//   TfLiteTensor* my_tensor = GetTemporary(context, node, kMyTensorIdx);
//   TF_LITE_ENSURE(context, my_tensor != nullptr);
//
// This is because the index might point to the optional tensor constant
// (kTfLiteOptionalTensor) in which case there is no tensor to return.
TfLiteTensor* GetTemporary(TfLiteContext* context, const TfLiteNode* node,
                           int index);

// Same as `GetTemporary` but returns boolean and uses output argument for
// tensor.
//
//   TfLiteTensor* my_tensor;
//   TF_LITE_ENSURE_OK(context,
//                     GetTemporarySafe(context, node, kMyTensorIdx,
//                     &my_tensor));
//   // can use my_tensor directly from here onwards, it is not nullptr
//
// Should be used in cases where the binary size is too large.
TfLiteStatus GetTemporarySafe(const TfLiteContext* context,
                              const TfLiteNode* node, int index,
                              TfLiteTensor** tensor);

// Note: You must check if result is not null:
//
//   TfLiteTensor* my_tensor = GetIntermediates(context, node, kMyTensorIdx);
//   TF_LITE_ENSURE(context, my_tensor != nullptr);
//
// This is because the index might point to the optional tensor constant
// (kTfLiteOptionalTensor) in which case there is no tensor to return.
const TfLiteTensor* GetIntermediates(TfLiteContext* context,
                                     const TfLiteNode* node, int index);

// Same as `GetIntermediates` but returns boolean and uses output argument for
// tensor.
//
//   TfLiteTensor* my_tensor;
//   TF_LITE_ENSURE_OK(context,
//                     GetIntermediatesSafe(context, node, kMyTensorIdx,
//                     &my_tensor));
//   // can use my_tensor directly from here onwards, it is not nullptr
//
// Should be used in cases where the binary size is too large.
TfLiteStatus GetIntermediatesSafe(const TfLiteContext* context,
                                  const TfLiteNode* node, int index,
                                  TfLiteTensor** tensor);
#endif  // TF_LITE_STATIC_MEMORY

inline int NumDimensions(const TfLiteTensor* t) { return t->dims->size; }
inline int SizeOfDimension(const TfLiteTensor* t, int dim) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_utilDTh mht_0(mht_0_v, 320, "", "./tensorflow/lite/kernels/kernel_util.h", "SizeOfDimension");

  return t->dims->data[dim];
}

inline int NumInputs(const TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_utilDTh mht_1(mht_1_v, 327, "", "./tensorflow/lite/kernels/kernel_util.h", "NumInputs");

  return node->inputs == nullptr ? 0 : node->inputs->size;
}
inline int NumOutputs(const TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_utilDTh mht_2(mht_2_v, 333, "", "./tensorflow/lite/kernels/kernel_util.h", "NumOutputs");

  return node->outputs == nullptr ? 0 : node->outputs->size;
}

#ifndef TF_LITE_STATIC_MEMORY
inline int NumIntermediates(const TfLiteNode* node) {
  return node->intermediates->size;
}
#endif  // TF_LITE_STATIC_MEMORY

inline int64_t NumElements(const TfLiteIntArray* dims) {
  int64_t count = 1;
  for (int i = 0; i < dims->size; ++i) {
    count *= dims->data[i];
  }
  return count;
}

inline int64_t NumElements(const TfLiteTensor* t) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_utilDTh mht_3(mht_3_v, 354, "", "./tensorflow/lite/kernels/kernel_util.h", "NumElements");

  return NumElements(t->dims);
}

// Determines whether tensor is constant.
// TODO(b/138199592): Introduce new query which checks for constant OR
// persistent-read-only, which would be useful for most tensor kernels that
// are potentially dynamic based on the input tensor value availability at the
// time of prepare.
inline bool IsConstantTensor(const TfLiteTensor* tensor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_utilDTh mht_4(mht_4_v, 366, "", "./tensorflow/lite/kernels/kernel_util.h", "IsConstantTensor");

  return tensor->allocation_type == kTfLiteMmapRo;
}

inline bool IsConstantOrPersistentTensor(const TfLiteTensor* tensor) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_utilDTh mht_5(mht_5_v, 373, "", "./tensorflow/lite/kernels/kernel_util.h", "IsConstantOrPersistentTensor");

  return IsConstantTensor(tensor) ||
         (tensor->allocation_type == kTfLitePersistentRo);
}

// Determines whether tensor is dynamic. Note that a tensor can be non-const and
// not dynamic. This function specifically checks for a dynamic tensor.
inline bool IsDynamicTensor(const TfLiteTensor* tensor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_utilDTh mht_6(mht_6_v, 383, "", "./tensorflow/lite/kernels/kernel_util.h", "IsDynamicTensor");

  return tensor->allocation_type == kTfLiteDynamic;
}

// Sets tensor to dynamic.
inline void SetTensorToDynamic(TfLiteTensor* tensor) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_utilDTh mht_7(mht_7_v, 391, "", "./tensorflow/lite/kernels/kernel_util.h", "SetTensorToDynamic");

  if (tensor->allocation_type != kTfLiteDynamic) {
    tensor->allocation_type = kTfLiteDynamic;
    tensor->data.raw = nullptr;
  }
}

// Sets tensor to persistent and read-only.
inline void SetTensorToPersistentRo(TfLiteTensor* tensor) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_utilDTh mht_8(mht_8_v, 402, "", "./tensorflow/lite/kernels/kernel_util.h", "SetTensorToPersistentRo");

  if (tensor->allocation_type != kTfLitePersistentRo) {
    tensor->allocation_type = kTfLitePersistentRo;
    tensor->data.raw = nullptr;
  }
}

// Determines whether it is a hybrid op - one that has float inputs and
// quantized weights.
inline bool IsHybridOp(const TfLiteTensor* input, const TfLiteTensor* weight) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_utilDTh mht_9(mht_9_v, 414, "", "./tensorflow/lite/kernels/kernel_util.h", "IsHybridOp");

  return ((weight->type == kTfLiteUInt8 || weight->type == kTfLiteInt8) &&
          input->type == kTfLiteFloat32);
}

// Check dimensionality match and populate OpData for Conv and DepthwiseConv.
TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int32_t* per_channel_shift);

TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int32_t* per_channel_shift,
    int num_channels);

// Calculates the multiplication factor for a quantized convolution (or
// quantized depthwise convolution) involving the given tensors. Returns an
// error if the scales of the tensors are not compatible.
TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              const TfLiteTensor* bias,
                                              TfLiteTensor* output,
                                              double* multiplier);

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              TfLiteTensor* output,
                                              double* multiplier);

// Calculates the useful quantized range of an activation layer given its
// activation tensor.
TfLiteStatus CalculateActivationRangeQuantized(TfLiteContext* context,
                                               TfLiteFusedActivation activation,
                                               TfLiteTensor* output,
                                               int32_t* act_min,
                                               int32_t* act_max);

// Calculates the useful range of an activation layer given its activation
// tensor.a
template <typename T>
void CalculateActivationRange(TfLiteFusedActivation activation,
                              T* activation_min, T* activation_max) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_utilDTh mht_10(mht_10_v, 466, "", "./tensorflow/lite/kernels/kernel_util.h", "CalculateActivationRange");

  if (activation == kTfLiteActRelu) {
    *activation_min = 0;
    *activation_max = std::numeric_limits<T>::max();
  } else if (activation == kTfLiteActRelu6) {
    *activation_min = 0;
    *activation_max = 6;
  } else if (activation == kTfLiteActReluN1To1) {
    *activation_min = -1;
    *activation_max = 1;
  } else {
    *activation_min = std::numeric_limits<T>::lowest();
    *activation_max = std::numeric_limits<T>::max();
  }
}

// Return true if the given tensors have the same shape.
bool HaveSameShapes(const TfLiteTensor* input1, const TfLiteTensor* input2);

#if !defined(TF_LITE_STATIC_MEMORY)
// Gets the output shape from the input tensor.
TfLiteStatus GetOutputShapeFromInput(TfLiteContext* context,
                                     const TfLiteTensor* input,
                                     TfLiteIntArray** output_shape);

const std::string GetShapeDebugString(const TfLiteIntArray* shape);

#endif  // !defined(TF_LITE_STATIC_MEMORY)

// Calculates the output_shape that is necessary for element-wise operations
// with broadcasting involving the two input tensors.
TfLiteStatus CalculateShapeForBroadcast(TfLiteContext* context,
                                        const TfLiteTensor* input1,
                                        const TfLiteTensor* input2,
                                        TfLiteIntArray** output_shape);

// Calculates the output_shape that is necessary for element-wise operations
// with broadcasting involving the three input tensors.
TfLiteStatus CalculateShapeForBroadcast(TfLiteContext* context,
                                        const TfLiteTensor* input1,
                                        const TfLiteTensor* input2,
                                        const TfLiteTensor* input3,
                                        TfLiteIntArray** output_shape);

// Return the size of given type in bytes. Return 0 in in case of string.
int TfLiteTypeGetSize(TfLiteType type);

// Whether the current platform is mobile (Android or iOS).
bool IsMobilePlatform();

// Returns whether there is unspecified dimension in the tensor's dim signature.
bool HasUnspecifiedDimension(const TfLiteTensor* tensor);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_KERNEL_UTIL_H_
