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
#ifndef TENSORFLOW_LITE_TOCO_TOOLING_UTIL_H_
#define TENSORFLOW_LITE_TOCO_TOOLING_UTIL_H_
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
class MHTracer_DTPStensorflowPSlitePStocoPStooling_utilDTh {
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
   MHTracer_DTPStensorflowPSlitePStocoPStooling_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStooling_utilDTh() {
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


#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"
#include "tensorflow/lite/toco/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

// TODO(aselle): Replace with using a container specific hash override instead.
namespace std {
template <>
struct hash<toco::OperatorType> {
  size_t operator()(const toco::OperatorType& op) const {
    return std::hash<size_t>()(static_cast<size_t>(op));
  }
};
}  // namespace std

namespace toco {

constexpr int kLogLevelModelChanged = 1;
constexpr int kLogLevelModelUnchanged = 2;

absl::string_view FindLongestCommonPrefix(absl::string_view a,
                                          absl::string_view b);
std::string LogName(const Operator& op);

std::string ArrayDataTypeName(ArrayDataType data_type);

// Returns true if the given array is specified as a model input array.
bool IsInputArray(const Model& model, const std::string& array_name);
// Returns true if the given array is specified as a model output array.
bool IsOutputArray(const Model& model, const std::string& array_name);

bool IsArrayConsumed(const Model& model, const std::string& name);
int CountTrueOutputs(const Model& model, const Operator& op);

int CountOpsWithInput(const Model& model, const std::string& array_name);
bool DeleteArrayIfUnused(const std::string& array_name, Model* model);

// Deletes the op and any of its input and output arrays if they are unused
// after the op has been deleted.
void DeleteOpAndArrays(Model* model, const Operator* op);

std::vector<std::unique_ptr<Operator>>::const_iterator FindOpWithOutput(
    const Model& model, const std::string& array_name);
Operator* GetOpWithOutput(const Model& model, const std::string& array_name);

std::vector<std::unique_ptr<Operator>>::iterator FindOpWithOutput(
    Model& model, const std::string& array_name);

std::vector<std::unique_ptr<Operator>>::const_iterator FindOpWithInput(
    const Model& model, const std::string& array_name);

std::vector<std::unique_ptr<Operator>>::iterator FindOpWithInput(
    Model& model, const std::string& array_name);

Operator* GetOpWithInput(const Model& model, const std::string& array_name);
Operator* GetFirstOpWithInput(const Model& model,
                              const std::string& array_name);

// Replaces all uses of the |old_array_name| with the |new_array_name|.
void ReplaceArrayUsage(Model* model, const std::string& old_array_name,
                       const std::string& new_array_name);

std::vector<std::unique_ptr<Operator>>::const_iterator FindOp(
    const Model& model, const Operator* op);
std::vector<std::unique_ptr<Operator>>::iterator FindOp(Model& model,
                                                        const Operator* op);

const char* OperatorTypeName(OperatorType type);
std::string HelpfulOperatorTypeName(const Operator& op);

// Whether the operator can be fused with an activation function. Note that this
// will return false by default for new operators; fusing support is opt-in.
bool OperatorSupportsFusedActivation(OperatorType type);

void DumpGraphvizVideoFrame(const Model& model);
void LogDump(int log_level, const std::string& message, const Model& model);
void LogSummary(int log_level, const std::string& message, const Model& model);

// TODO(b/36075966): Clean up when dims superseded by array shape.
void ExtendShape(Shape* shape, int new_shape_size);

// TODO(b/36075966): Clean up when dims superseded by array shape.
void UnextendShape(Shape* shape, int new_shape_size);

// Checks that all dimensions of 'shape' are at least 1. Note that scalars,
// lacking dimensions, satisfy this condition and are considered non-empty.
bool IsNonEmpty(const Shape& shape);

// Given two shapes with potentially different dimensionality and dimension
// arrays d0 and d1. Without loss of generality, assume that shape0 may have
// higher dimensionality (length(d0) >= length(d1)). Then shape0 and shape1
// "agree up to broadcasting" if:
// - When walking the d0 and d1 from back to front with indices i0, i1,
//   d0[i0] == d1[i1] or d0[i0] == 1 or d1[i1] == 1, for each dimension until
//   i1 == 0 (inclusive).
bool ShapesAgreeUpToBroadcasting(const Shape& shape0, const Shape& shape1);

// A stricter constraint than ShapesAgreeUpToBroadcasting().
//
// Given two shapes with potentially different dimensionality and dimension
// arrays d0 and d1. Without loss of generality, assume that shape0 may have
// higher dimensionality (length(d0) >= length(d1)). Then shape0 and shape1
// "agree up to extending" if:
// - When walking the d0 and d1 from back to front with indices i0, i1,
//   d0[i0] == d1[i1] for each dimension until i1 == 0 (inclusive).
// - For the remaining indices [0..i0), d0[i0] == 1.
bool ShapesAgreeUpToExtending(const Shape& shape0, const Shape& shape1);

inline ::tflite::RuntimeShape ToRuntimeShape(const Shape& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPStooling_utilDTh mht_0(mht_0_v, 309, "", "./tensorflow/lite/toco/tooling_util.h", "ToRuntimeShape");

  return ::tflite::RuntimeShape(shape.dimensions_count(), shape.dims().data());
}

bool IsArrayFullyConnectedWeights(const Model& model, const std::string& name);

// If there is a wildcard dimension (-1), this may return a negative value.
int RequiredBufferSizeForShape(const Shape& shape);

bool IsConstantParameterArray(const Model& model, const std::string& name);

// Compares two constant parameter arrays for exact equality.
bool CompareConstantArrays(const Array& lhs_array, const Array& rhs_array);

void CheckNoMissingArray(const Model& model);
void CheckInvariants(const Model& model);

void CheckModelCounts(const Model& model);

void FixOperatorOrdering(Model* model);
void FixNoMissingArray(Model* model);
void FixNoOrphanedArray(Model* model);

// Fixes input/output arrays that may have issues during export or inference.
void FixEdgeArrays(Model* model);

// Finds and deduplicates large constant arrays in the model.
// After constant propagation runs it's possible to end up with several of the
// same large array (whether they be zeros or otherwise).
//
// |min_size| is used to adjust the minimum size in bytes of an array before
// it's considered for deduping. As deduping can make the graphs more difficult
// to read this helps prevent small arrays from spidering out.
void DedupeConstantArrays(Model* model, size_t min_size);

// Copies the contents of an array into another.
// Expects that the shape and data type match.
template <ArrayDataType A>
void CopyArrayBuffer(const Array& source_array, Array* target_array) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPStooling_utilDTh mht_1(mht_1_v, 350, "", "./tensorflow/lite/toco/tooling_util.h", "CopyArrayBuffer");

  int source_buffer_size = RequiredBufferSizeForShape(source_array.shape());
  int target_buffer_size = RequiredBufferSizeForShape(target_array->shape());
  CHECK_EQ(source_buffer_size, target_buffer_size)
      << "Buffer sizes must match in element count";
  CHECK(source_array.data_type == target_array->data_type)
      << "Data types must match";
  if (source_array.buffer) {
    const auto& source_buffer = source_array.GetBuffer<A>();
    auto& target_buffer = target_array->GetMutableBuffer<A>();
    target_buffer.data = source_buffer.data;
  }
}

// Inserts a no-op reshape operator between the source array and the target
// array. This effectively just copies the data.
void InsertCopyOperator(Model* model, const std::string& source_array_name,
                        const std::string& target_array_name);

// Clones an array with all data and parameters.
void CloneArray(Model* model, const std::string& source_array_name,
                const std::string& target_array_name);

void ResolveModelFlags(const ModelFlags& model_flags, Model* model);

template <typename T>
T ConvertOperator(Operator* o, OperatorType type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPStooling_utilDTh mht_2(mht_2_v, 379, "", "./tensorflow/lite/toco/tooling_util.h", "ConvertOperator");

  if (o != nullptr && o->type == type) {
    return static_cast<T>(o);
  }

  return nullptr;
}

void CheckIsReadyForQuantization(const Model& model);

bool ReshapeIsEquivalentToTranspose(const Model& model,
                                    const TensorFlowReshapeOperator* op,
                                    bool allow_extra_unary_dims);

inline int Offset(const Shape& shape, const std::vector<int>& indices) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPStooling_utilDTh mht_3(mht_3_v, 396, "", "./tensorflow/lite/toco/tooling_util.h", "Offset");

  DCHECK_EQ(shape.dimensions_count(), indices.size());
  const int dims_count = shape.dimensions_count();
  int offset = 0;
  for (int i = 0; i < dims_count; i++) {
    const int index = indices[i];
    DCHECK(index >= 0 && index < shape.dims(i));
    offset *= shape.dims(i);
    offset += index;
  }
  return offset;
}

inline std::vector<int> ReverseOffset(const Shape& shape, int index) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, RequiredBufferSizeForShape(shape));
  const int dims_count = shape.dimensions_count();
  std::vector<int> indices(dims_count);
  int residual = index;
  for (int i = dims_count - 1; i >= 0; i--) {
    indices[i] = residual % shape.dims(i);
    residual /= shape.dims(i);
  }
  return indices;
}

int ElementSize(ArrayDataType data_type);

void DropMinMax(Model* model, const std::string& array_name);

bool IsAllocatableTransientArray(const Model& model,
                                 const std::string& array_name);

void CreateOrCheckRnnStateArray(const std::string& name, int size,
                                int state_num_dims, Model* model);

std::string AvailableArrayName(const Model& model, const std::string& name);

// Formats a shape as a string: [ dims(0), dims(1), ..., dims(num_dims-1) ].
std::string ShapeToString(const Shape& shape);

void PrintArrayShape(Model* model, const std::string& name);

void MakeArrayDims(int num_dims, int batch, int height, int width, int depth,
                   std::vector<int>* out_dims);

// Defines a constant int32 array with the provided values formatted for use
// as op parameters.
std::string CreateInt32Array(Model* model, const std::string& param_name,
                             const std::vector<int>& value);

bool EstimateArithmeticOpsCount(const Model& model, const Operator& op,
                                int64_t* result);
bool EstimateArithmeticOpsCount(const Model& model, int64_t* result);
std::string FormattedNumber(int64_t x);

int AxesCount(AxesOrder axes_order);

// Returns the permutation of the dimensions based on the input axes order and
// output axes order.
void GetShuffleShape(AxesOrder input_axes_order, AxesOrder output_axes_order,
                     std::vector<int>* shuffle);

// Extend shuffle is designed to match ExtendShape, which pads the shape with
// unit dimensions at the beginning.
void ExtendShuffle(const std::vector<int>& input_shuffle, int newdim,
                   std::vector<int>* extended_shuffle);

void ShuffleDims(const Shape& input_shape, AxesOrder input_axes_order,
                 AxesOrder output_axes_order, Shape* output_shape);
void ShuffleArray(const Shape& input_shape, AxesOrder input_axes_order,
                  AxesOrder output_axes_order, const Shape& output_shape,
                  const float* input_data, float* output_data);
void ShuffleArray(const Shape& input_shape, AxesOrder input_axes_order,
                  AxesOrder output_axes_order, const Shape& output_shape,
                  const uint8* input_data, uint8* output_data);

// Returns true if it may be OK for any graph transformation to ever discard
// that array. The idea is that we can't ever discard arrays that are either
// an input or an output of the whole graph, or that appear in RNN back-edges,
// as that would undercut explicit flags that the user might pass.
bool IsDiscardableArray(const Model& model, const std::string& array_name);

void CheckFinalDataTypesSatisfied(const Model& model);

ArrayDataType ConvertIODataTypeToArrayDataType(IODataType type);

// The process of building models varies according to the import format.
//
// (a) In some cases, such as model-proto format, the model should be fully
// specified. In these cases, no extra action should be taken by this function.
// (b) In other cases, such as TF graphdef format, the desired types of RNN
// arrays are not specified directly in the model, neither can they be inferred.
// However, we can set the types of RNN destination arrays to float. This breaks
// any cycles such as when resolution of the type of an RNN source array depends
// on the type of its destination array.
//
// This function is applied after the main import, after resolution of flags and
// after application of ArraysExtraInfo. It only defaults destination RNN arrays
// to float. If the model is subsequently quantized, it is assumed that the
// model contains sufficient information for that to be completed. If it is
// already quantized, then case (a) should hold.
void FinishBuildingRNNStates(Model* model);

void UseArraysExtraInfo(Model* model, bool quantize_output);

// Calculates the number of elements in tensor given a shape. Shape elements
// are assumed to be of type T, while the result total is of type U. If U
// doesn't have enough range to represent the sum of elements, an error is
// returned.
template <typename T, typename U>
tensorflow::Status NumElements(const std::vector<T>& shape, U* num_elements) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPStooling_utilDTh mht_4(mht_4_v, 510, "", "./tensorflow/lite/toco/tooling_util.h", "NumElements");

  static_assert(
      std::numeric_limits<T>::max() <= std::numeric_limits<uint64_t>::max(),
      "vector type exceed capabilities of NumElements");

  *num_elements = 1;
  for (const T& dim : shape) {
    if (dim < 0) {
      // TensorFlow's shapes sometimes include -1 to represent an "unknown"
      // size but TOCO isn't able to create arrays of unknown sizes and will
      // crash in RequiredBufferSizeForShape().
      return tensorflow::errors::InvalidArgument(
          "Tensor shape should not include negative values");
    }
    if (*num_elements != 0 &&
        static_cast<uint64_t>(dim) >
            std::numeric_limits<U>::max() / *num_elements) {
      *num_elements = 0;
      return tensorflow::errors::InvalidArgument("Tensor shape is too large");
    }
    *num_elements *= dim;
  }
  return tensorflow::Status::OK();
}

// A model file may have shuffled FC weights.
// When that happens, we want to de-shuffle them immediately on import,
// so that the rest of toco doesn't need to know about shuffled weights.
void UndoWeightsShuffling(Model* model);

// Copies minmax, quantization_params, and narrow_range.
void CopyMinMaxAndQuantizationRelatedFields(const Array& src, Array* dst);

// Delete Array if it's discardable and not referenced as input or output array
// by any other op than the specified op.
bool DeleteArrayIfUnusedOutsideOfOp(const std::string& array_name,
                                    const Operator* op, Model* model);

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TOOLING_UTIL_H_
