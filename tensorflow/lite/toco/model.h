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
#ifndef TENSORFLOW_LITE_TOCO_MODEL_H_
#define TENSORFLOW_LITE_TOCO_MODEL_H_
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
class MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh {
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
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh() {
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


#include <complex>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/toco_port.h"
#include "tensorflow/lite/toco/toco_types.h"

namespace toco {

using tflite::QuantizationParams;

enum class OperatorType : uint8 {
  kNone,
  // General-purpose neural network operators.
  kAdd,
  kAddN,
  kAveragePool,
  kBatchMatMul,
  kBatchNormalization,
  kCeil,
  kConv,
  kConcatenation,
  kCos,
  kDepthwiseConv,
  kDepthToSpace,
  kSpaceToDepth,
  kDequantize,
  kDiv,
  kExp,
  kExpandDims,
  kFill,
  kFloorDiv,
  kFloorMod,
  kFullyConnected,
  kL2Normalization,
  kL2Pool,
  kLstmCell,
  kUnidirectionalSequenceLstm,
  kLocalResponseNormalization,
  kLog,
  kLogistic,
  kMaxPool,
  kFakeQuant,
  kMul,
  kOneHot,
  kRandomUniform,
  kRange,
  kRank,
  kRelu,
  kRelu1,
  kRelu6,
  kPRelu,
  kHardSwish,
  kSoftmax,
  kLogSoftmax,
  kSub,
  kTanh,
  kTransposeConv,
  kCast,
  kFloor,
  kRound,
  kGather,
  kResizeBilinear,
  kSin,
  kSpaceToBatchND,
  kPack,
  kBatchToSpaceND,
  kPad,
  kPadV2,
  kReduceProd,  // Reduction product
  kStridedSlice,
  kSlice,
  kSqueeze,
  kMean,
  kArgMax,
  // The SVDF Op is a decomposition of a densely connected Op into
  // low rank filters. For details:
  // https://research.google.com/pubs/pub43813.html
  kSvdf,
  // Special operators used for importing TensorFlow nodes.
  // The general intent is to have some graph transformation either
  // drop them or rewrite them as general-purpose operators.
  kAll,
  kAssert,
  kConcat,
  kConcatV2,
  kGreater,
  kGreaterEqual,
  kIdentity,
  kLess,
  kLessEqual,
  kReduceMax,  //  Reduction Max
  kMaximum,    //  Element-wise Maximum
  kReduceMin,  //  Reduction Min
  kMinimum,    //  Element-wise Minimum
  kMatMul,
  kMerge,
  kNeg,
  kReshape,
  kRsqrt,
  kShape,
  kSplit,
  kSplitV,
  kSqrt,
  kSquare,
  kSquaredDifference,
  kSum,
  kSwitch,
  kTile,
  kTranspose,
  kTopK_V2,
  kDynamicPartition,
  kDynamicStitch,
  // An unsupported TF operation. It's only needed to be able to represent TF
  // graph internally and is expected to be dropped by graph transformations.
  kUnsupported,
  // Finally, TensorFlow uses different conventions for axes ordering,
  // see AxesOrder, and this cannot always be resolved at the time of importing
  // nodes, as TensorFlow parameters may be constant-expression subgraphs
  // instead of being given as plain constant arrays. So we need to insert
  // special nodes in the graph to shuffle axes.
  kReorderAxes,
  kSegmentSum,
  kSelect,
  kSelectV2,
  kSparseToDense,
  kEqual,
  kNotEqual,
  kPow,
  kArgMin,
  kAny,
  kLogicalAnd,
  kLogicalNot,
  kLogicalOr,
  kCTCBeamSearchDecoder,
  kUnpack,
  kZerosLike,
  kResizeNearestNeighbor,
  kLeakyRelu,
  kAbs,
  kMirrorPad,
  kUnique,
  kUnidirectionalSequenceRnn,
  kBidirectionalSequenceLstm,
  kReverseV2,
  kBidirectionalSequenceRnn,
  kGatherNd,
  kWhere,
  kElu,
  kReverseSequence,
  kMatrixDiag,
  kMatrixSetDiag,
  kMatrixDiagV2,
  kMatrixSetDiagV2,
  kMatrixDiagV3,
  kMatrixSetDiagV3,
  kScatterNd,
  // Debugging operators.
  kNumericVerify
};

// Helper to deal with TensorFlow arrays using a different ordering of
// dimensions
// ("axes") than our own.
// TODO(benoitjacob): Ultimately, we shouldn't have any "ordering" of axes,
// we should have associative arrays mapping symbolic axes identifiers (like
// "output_depth") to dimensions. We would then not need this anymore.
enum class AxesOrder {
  kOneAxis,  // one-dimensional array, one unique axis.
  kCR,       // column-major matrix storage order. Our standard.
  kRC,       // row-major matrix storage order. TensorFlow default.
  kOHWI,     // Our standard for conv weights
  kHWIO,     // TensorFlow conv weights
  k1HWO,     // Our standard for DepthwiseConv weights
  kHWIM,     // TensorFlow DepthwiseConv weights
  kNHWC,     // TensorFlow activations
  kHWOI,     // TensorFlow back-prop conv weights
};

// The type of the scalars in an array.
// Note that the type does not by itself tell whether the values in the array
// are non-quantized (can be accessed directly) or quantized (must be
// interpreted in conjunction with QuantizationParams).
//
// In practice though:
//   float values are never quantized
//   uint8 values are always quantized
//   int32 values are sometimes quantized (depending on whether
//   QuantizationParams are present).
//   complex values are never quantized
//   other types are never quantized at the moment.
//
// kNone means that we don't know the data type yet, or that we don't care
// because we'll be dropping the array anyway (e.g. some exotic array types
// may be involved only in debug-only subgraphs that we may not be interested
// in actually supporting).
enum class ArrayDataType : uint8 {
  kNone,  // 0
  kBool,
  kFloat,
  kInt8,
  kUint8,
  kInt16,  // 5
  kUint16,
  kInt32,
  kUint32,
  kInt64,
  kUint64,  // 10
  kString,
  kComplex64,
  kFloat16,
  kFloat64,
  kComplex128,
};

// Compile-time logic to map ArrayDataType to the corresponding C++ scalar type
template <ArrayDataType A>
struct DataTypeImpl {};
template <>
struct DataTypeImpl<ArrayDataType::kNone> {
  typedef int Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kBool> {
  typedef bool Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kFloat> {
  typedef float Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kInt8> {
  typedef int8 Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kUint8> {
  typedef uint8 Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kInt16> {
  typedef int16 Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kUint16> {
  typedef uint16 Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kInt32> {
  typedef int32 Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kUint32> {
  typedef uint32 Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kInt64> {
  typedef int64_t Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kUint64> {
  typedef uint64 Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kString> {
  typedef std::string Type;
};
template <>
struct DataTypeImpl<ArrayDataType::kComplex64> {
  typedef std::complex<float> Type;
};

template <ArrayDataType A>
using DataType = typename DataTypeImpl<A>::Type;

// Base class for type-specific buffer types.
struct GenericBuffer {
  // Non-default-constructible: only ArrayDataType-specific subclass
  // objects may be constructed.
  GenericBuffer() = delete;
  // Non-copyable-or-movable: we should only store pointers-to-Buffer
  // in containers, not Operators themselves, so there should be no
  // copy or move.
  GenericBuffer(const GenericBuffer&) = delete;
  GenericBuffer(const GenericBuffer&&) = delete;

  // We need a virtual destructor so we can store pointers-to-Buffer
  // in containers and have the containers call the right subclass destructor.
  virtual ~GenericBuffer() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_0(mht_0_v, 483, "", "./tensorflow/lite/toco/model.h", "~GenericBuffer");
}

  virtual int Length() const = 0;

  const ArrayDataType type;

 protected:
  // Constructor used by subclasses for specific ArrayDataType's.
  explicit GenericBuffer(ArrayDataType t) : type(t) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_1(mht_1_v, 494, "", "./tensorflow/lite/toco/model.h", "GenericBuffer");
}
};

// Type-specific buffer, containing type-specific storage.
template <ArrayDataType A>
struct Buffer : GenericBuffer {
  Buffer() : GenericBuffer(A) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_2(mht_2_v, 503, "", "./tensorflow/lite/toco/model.h", "Buffer");
}

  int Length() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_3(mht_3_v, 508, "", "./tensorflow/lite/toco/model.h", "Length");
 return data.size(); }

  std::vector<DataType<A>> data;
};

class Shape {
 public:
  // For Shape, we stick to half-way encapsulation for now:
  // we hide the raw dims_ member, but expose it raw by accessors
  // because from some brainstorming, it's not at all easy to
  // anticipate which flavor of more hermetic encapsulation would
  // actually buy us future-proof-ness without being needlessly
  // cumbersome.
  Shape() {}
  Shape(std::initializer_list<int> dim_list) : dims_(dim_list) {}

  void ReplaceDims(std::initializer_list<int> dim_list) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_4(mht_4_v, 527, "", "./tensorflow/lite/toco/model.h", "ReplaceDims");

    dims_ = std::vector<int>(dim_list);
  }

  const std::vector<int>& dims() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_5(mht_5_v, 534, "", "./tensorflow/lite/toco/model.h", "dims");
 return dims_; }
  std::vector<int>* mutable_dims() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_6(mht_6_v, 538, "", "./tensorflow/lite/toco/model.h", "mutable_dims");
 return &dims_; }
  const int dimensions_count() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_7(mht_7_v, 542, "", "./tensorflow/lite/toco/model.h", "dimensions_count");
 return dims_.size(); }

  // We still have that one convenience accessor to avoid
  // the awkward double bracket issue:  shape.dims()[i].
  int dims(int i) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_8(mht_8_v, 549, "", "./tensorflow/lite/toco/model.h", "dims");

    // Always check for out-of-bounds accesses, even in optimized builds where
    // standard assertions are disabled. Out-of-bounds access here is a common
    // occurrence.
    CHECK_GE(i, 0);
    CHECK_GT(dims_.size(), i);
    return dims_[i];
  }

  bool operator==(const Shape& comp) const {
    return (this->dims_ == comp.dims());
  }

  bool operator!=(const Shape& comp) const { return !((*this) == comp); }

 private:
  std::vector<int> dims_;
};

// Base class for all operator classes.
struct Operator {
  // Non-default-constructible: only OperatorType-specific subclass
  // objects may be constructed.
  Operator() = delete;
  // Non-copyable-or-movable: we should only store pointers-to-Operator
  // in containers, not Operators themselves, so there should be no
  // copy or move.
  Operator(const Operator&) = delete;
  Operator(const Operator&&) = delete;

  // We need a virtual destructor so we can store pointers-to-Operator
  // in containers and have the containers call the right subclass destructor.
  virtual ~Operator() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_9(mht_9_v, 584, "", "./tensorflow/lite/toco/model.h", "~Operator");
}

  // The specific type of operator. Corresponds 1:1 to subclasses.
  const OperatorType type;

  // The activation function that may be fused into this operator,
  // or None if no activation function is fused.
  FusedActivationFunctionType fused_activation_function;

  // Input arrays: either activation arrays or constant array parameters.
  // We refer to them by their name, not by their address; the mapping of
  // names to addresses is given by the Model, which owns both Operator's and
  // Array's. Thus, an Operator on its own doesn't contain much information,
  // it is meant to be used in conjunction with the Model that owns it.
  std::vector<std::string> inputs;

  // Output activation arrays. Same comments as for inputs apply here too.
  std::vector<std::string> outputs;

  // If true, the operator has more outputs than are listed in the 'outputs'
  // member. These need to be resolved by some graph transformation.
  // This flag is only here to indicate that an operator should not be
  // discarded as unused, even if from its 'outputs' member alone it
  // looks unused.
  bool unresolved_outputs = false;

  // A serialized tensorflow::NodeDef string.
  // The field is filled only when importing from TensorFlow.
  // It's guaranteed to be filled for `TensorFlowUnsupportedOperator`.
  // It's not guaranteed to be filled for other ops. Ops created by graph
  // transformations won't have TensorFlow NodeDef.
  std::string tensorflow_node_def;

 protected:
  // Constructor used by subclasses for specific OperatorType's.
  explicit Operator(OperatorType t)
      : type(t),
        fused_activation_function(FusedActivationFunctionType::kNone) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_10(mht_10_v, 624, "", "./tensorflow/lite/toco/model.h", "Operator");
}
};

// Padding types for Conv-like operators. This is how padding is typically
// specified in model files. But for inference, we will need to resolve this
// to a FixedPadding, see below.
enum class PaddingType { kNone, kSame, kValid };

// Padding as resolved for a specific layer shape, as needed for inference.
// For a given layer shape, a given padding type will resolve to a choice of
// a number of padding rows and columns, which we call the padding height and
// width respectively.
struct FixedPadding {
  int width = 0;
  int height = 0;
};

// "Universal" padding struct containing both a generic PaddingType (as
// represented in a model file), and a FixedPadding (as needed for inference).
// The latter is resolved during the PropagateFixedSizes pass.
struct Padding {
  FixedPadding& GetOrCreateFixedPadding() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_11(mht_11_v, 648, "", "./tensorflow/lite/toco/model.h", "GetOrCreateFixedPadding");

    if (!fixed) {
      FixedPadding* ptr = new FixedPadding;
      fixed = std::unique_ptr<FixedPadding>(ptr);
    }
    return *fixed;
  }

  Padding() : type(PaddingType::kNone) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_12(mht_12_v, 659, "", "./tensorflow/lite/toco/model.h", "Padding");
}
  PaddingType type;
  std::unique_ptr<FixedPadding> fixed;
};

// "Convolutional" layer, as represented in model files.
//
// Inputs:
//   inputs[0]: required: the input activations array
//   inputs[1]: required: the Conv weights
//   inputs[2]: optional: the bias vector, specifying the biases for each output
//   channel.
//
// Outputs:
//   outputs[0]: required: the output activations array
//   outputs[1]: optional: the intermediate array of im2col-replicated input
//                         activations. Present when targeting implementations
//                         of Conv layers as Im2col+GEMM.
//
// TensorFlow equivalent: Conv2D
struct ConvOperator : Operator {
  ConvOperator() : Operator(OperatorType::kConv) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_13(mht_13_v, 683, "", "./tensorflow/lite/toco/model.h", "ConvOperator");
}
  Padding padding;
  int stride_width = 0;
  int stride_height = 0;
  // A dilation_rate of 0 is invalid and this field is an optional attribute.
  // Thus initializing it to 1 to allow default conv behavior when the
  // attribute is not present.
  int dilation_width_factor = 1;
  int dilation_height_factor = 1;
};

// CTCBeamSearchDecoder operator:
//
// Inputs:
//   inputs[0]: required: the logits.
//   inputs[1]: required: sequence length.
//   inputs[2]: optional: beam width.
//   inputs[3]: optional: top paths.
//   inputs[4]: optional: merge repeated.
//
//  Outputs:
//    outputs[0]: decoded.
//    outputs[1]: log probability.
//
// TensorFlow equivalent: CTCBeamSearchDecoder
struct CTCBeamSearchDecoderOperator : Operator {
  CTCBeamSearchDecoderOperator()
      : Operator(OperatorType::kCTCBeamSearchDecoder) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_14(mht_14_v, 713, "", "./tensorflow/lite/toco/model.h", "CTCBeamSearchDecoderOperator");
}
  int beam_width;
  int top_paths;
  bool merge_repeated = true;
};

// Depthwise-separable convolution operator.
//
// Inputs:
//   inputs[0]: required: the input activations array
//   inputs[1]: required: the DepthwiseConv weights
//   inputs[2]: optional: the bias vector, specifying the biases for each output
//   channel.
//
// TensorFlow equivalent: DepthwiseConv2dNative
struct DepthwiseConvOperator : Operator {
  DepthwiseConvOperator() : Operator(OperatorType::kDepthwiseConv) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_15(mht_15_v, 732, "", "./tensorflow/lite/toco/model.h", "DepthwiseConvOperator");
}
  Padding padding;
  int stride_height = 0;
  int stride_width = 0;
  int depth_multiplier = 0;
  // A dilation_rate of 0 is invalid and this field is an optional attribute.
  // Thus initializing it to 1 to allow default conv behavior when the
  // attribute is not present.
  int dilation_width_factor = 1;
  int dilation_height_factor = 1;
};

// Depth-to-space transform operator.
//
// Inputs:
//   inputs[0]: required: the input activations array
//
// TensorFlow equivalent: DepthToSpace
struct DepthToSpaceOperator : Operator {
  DepthToSpaceOperator() : Operator(OperatorType::kDepthToSpace) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_16(mht_16_v, 754, "", "./tensorflow/lite/toco/model.h", "DepthToSpaceOperator");
}
  int block_size = 0;
};

// Space-to-depth transform operator.
//
// Inputs:
//   inputs[0]: required: the input activations array
//
// TensorFlow equivalent: SpaceToDepth
struct SpaceToDepthOperator : Operator {
  SpaceToDepthOperator() : Operator(OperatorType::kSpaceToDepth) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_17(mht_17_v, 768, "", "./tensorflow/lite/toco/model.h", "SpaceToDepthOperator");
}
  int block_size = 0;
};

// Fully-connected operator.
//
// Inputs:
//   inputs[0]: required: the input activations array
//   inputs[1]: required: the FullyConnected weights
//   inputs[2]: optional: the bias vector, specifying the biases for each output
//   channel.
//
// TensorFlow equivalent: a pair consisting of a Reshape node reshaping the
// input activations as a matrix, followed by a MatMul node.
struct FullyConnectedOperator : Operator {
  FullyConnectedOperator() : Operator(OperatorType::kFullyConnected) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_18(mht_18_v, 786, "", "./tensorflow/lite/toco/model.h", "FullyConnectedOperator");
}
  FullyConnectedWeightsFormat weights_format =
      FullyConnectedWeightsFormat::kDefault;

  // `keep_num_dims` is supported in the FullyConnected kernel version 5, but
  // it's never supported by Toco.
  bool keep_num_dims = false;
};

// Dequantization operator, converting a quantized array of integers with
// quantization parameters specifying how these integers correspond to real
// numbers
// (see QuantizationParams) to an output activations array of floating-point
// values.
//
// In floating-point image models, there is typically a Dequantization operator
// at the very beginning, converting the input image RGB data, consisting of
// uint8 integer values, to floating-point input activations. That is where
// image model parameters such as "mean_value" and "std_value" are typically
// handled.
//
// This is the only operator type that converts from quantized to
// floating-point,
// and there is at the moment no operator type at all to convert from
// floating-point
// to quantized. Every other operator does either float->float or
// quantized->quantized.
//
// Inputs:
//   inputs[0]: required: the input quantized activations array
//
// TensorFlow equivalent: Dequantize
struct DequantizeOperator : Operator {
  DequantizeOperator() : Operator(OperatorType::kDequantize) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_19(mht_19_v, 822, "", "./tensorflow/lite/toco/model.h", "DequantizeOperator");
}
};

// Numeric verification operator, converting a quantized array of integers with
// quantization parameters specifying how these integers correspond to real
// numbers
// (see QuantizationParams) and verify them with an array of floating-point
// values.

// Inputs:
//   inputs[0]: required: the input quantized activations array
//   inputs[1]: required: the input reference activations array
//
// TensorFlow equivalent: Dequantize
struct NumericVerifyOperator : Operator {
  NumericVerifyOperator() : Operator(OperatorType::kNumericVerify) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_20(mht_20_v, 840, "", "./tensorflow/lite/toco/model.h", "NumericVerifyOperator");
}
};

// Batch-normalization operator.
//
// We only support batch-normalization using pre-learned moments, so this is
// just
// computing (input - mean) * multiplier + offset. As such, this can be
// expressed as a combination of Add and Mul nodes, and indeed this is how
// we break it down during tooling for the purpose of fusing it into
// other operators.
//
// Inputs:
//   inputs[0]: required: the input activations array
//   inputs[1]: required: the learned mean array
//   inputs[2]: required: the learned multiplier array
//   inputs[3]: required: the learned offset array
//
// TensorFlow equivalent: a combination of Add and Mul nodes
struct BatchNormalizationOperator : Operator {
  BatchNormalizationOperator()
      : Operator(OperatorType::kBatchNormalization),
        global_normalization(false) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_21(mht_21_v, 865, "", "./tensorflow/lite/toco/model.h", "BatchNormalizationOperator");
}
  bool global_normalization;
};

// L2-normalization operator.
//
// Inputs:
//   inputs[0]: required: the input activations array
//
// TensorFlow equivalent: none. In TensorFlow, L2 normalization is implemented
// by a sub-graph of operators implementing L2-normalization
// from lower-level arithmetic nodes; during tooling, we identify such
// sub-graphs
// and replace them by L2NormalizationOperator's. See IdentifyL2Normalization.
struct L2NormalizationOperator : Operator {
  L2NormalizationOperator() : Operator(OperatorType::kL2Normalization) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_22(mht_22_v, 883, "", "./tensorflow/lite/toco/model.h", "L2NormalizationOperator");
}
};

// LSTM Cell operator.
//
// Inputs:
//   inputs[0]: required: the input data array
//   inputs[1]: required: the previous output activations array
//   inputs[2]: required: the learned weights array
//   inputs[3]: required: the learned biases array
//   inputs[4]: required: the previous output state
//   outputs[0]: required: the output activations array
//   outputs[1]: required: the new state array
//
// TensorFlow equivalent: none. In TensorFlow, an LSTM is implemented
// with a sub-graph of lower-level arithmetic nodes; during tooling, we identify
// such sub-graphs and replace them with LstmCells. See IdentifyLstmCell().
struct LstmCellOperator : Operator {
  enum Inputs {
    DATA_INPUT = 0,
    PREV_ACTIV_INPUT = 1,
    WEIGHTS_INPUT = 2,
    BIASES_INPUT = 3,
    PREV_STATE_INPUT = 4,
    NUM_INPUTS = 5
  };
  enum Outputs {
    ACTIV_OUTPUT = 0,
    STATE_OUTPUT = 1,
    CONCAT_TEMP = 2,
    ACTIV_TEMP = 3,
    NUM_OUTPUTS = 4
  };
  enum KernelType {
    KERNEL_BASIC = 0,
    KERNEL_FULL = 1,
  };

  LstmCellOperator()
      : Operator(OperatorType::kLstmCell), kernel_type(KERNEL_BASIC) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_23(mht_23_v, 925, "", "./tensorflow/lite/toco/model.h", "LstmCellOperator");
}

  KernelType kernel_type;
};

struct UnidirectionalSequenceLstmOperator : Operator {
  UnidirectionalSequenceLstmOperator()
      : Operator(OperatorType::kUnidirectionalSequenceLstm) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_24(mht_24_v, 935, "", "./tensorflow/lite/toco/model.h", "UnidirectionalSequenceLstmOperator");
}
};

struct BidirectionalSequenceLstmOperator : Operator {
  BidirectionalSequenceLstmOperator()
      : Operator(OperatorType::kBidirectionalSequenceLstm) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_25(mht_25_v, 943, "", "./tensorflow/lite/toco/model.h", "BidirectionalSequenceLstmOperator");
}
  bool merge_outputs;
};

struct BidirectionalSequenceRnnOperator : Operator {
  BidirectionalSequenceRnnOperator()
      : Operator(OperatorType::kBidirectionalSequenceRnn) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_26(mht_26_v, 952, "", "./tensorflow/lite/toco/model.h", "BidirectionalSequenceRnnOperator");
}
  bool merge_outputs;
};

// Element-wise multiplication operator.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: Mul
struct MulOperator : Operator {
  MulOperator() : Operator(OperatorType::kMul) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_27(mht_27_v, 967, "", "./tensorflow/lite/toco/model.h", "MulOperator");
}
};

// Element-wise Abs operator:
//   x -> abs(x)
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: abs
struct AbsOperator : Operator {
  AbsOperator() : Operator(OperatorType::kAbs) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_28(mht_28_v, 981, "", "./tensorflow/lite/toco/model.h", "AbsOperator");
}
};

// Element-wise HardSwish operator:
//   x -> x * relu6(x+3)/6
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: hard_swish
struct HardSwishOperator : Operator {
  HardSwishOperator() : Operator(OperatorType::kHardSwish) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_29(mht_29_v, 995, "", "./tensorflow/lite/toco/model.h", "HardSwishOperator");
}
};

// Elu
//   f(x) -> exp(x) - 1 for x < 0, x for x >= 0.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Elu
struct EluOperator : Operator {
  EluOperator() : Operator(OperatorType::kElu) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_30(mht_30_v, 1009, "", "./tensorflow/lite/toco/model.h", "EluOperator");
}
};

// Element-wise Relu operator:
//   x -> max(0, x)
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Relu
struct ReluOperator : Operator {
  ReluOperator() : Operator(OperatorType::kRelu) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_31(mht_31_v, 1023, "", "./tensorflow/lite/toco/model.h", "ReluOperator");
}
};

// Element-wise Relu1 operator:
//   x -> min(max(x, -1), 1)
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: none. We can construct the operator with Minimum
// and Maximum operations
struct Relu1Operator : Operator {
  Relu1Operator() : Operator(OperatorType::kRelu1) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_32(mht_32_v, 1038, "", "./tensorflow/lite/toco/model.h", "Relu1Operator");
}
};

// Element-wise Relu6 operator:
//   x -> max(0, min(6, x))
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Relu6
struct Relu6Operator : Operator {
  Relu6Operator() : Operator(OperatorType::kRelu6) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_33(mht_33_v, 1052, "", "./tensorflow/lite/toco/model.h", "Relu6Operator");
}
};

// PRelu
//   f(x) = alpha * x for x < 0, f(x) = x for x >= 0.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the alpha array
//
// Equivalent to keras.layers.PReLU.
struct PReluOperator : Operator {
  PReluOperator() : Operator(OperatorType::kPRelu) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_34(mht_34_v, 1067, "", "./tensorflow/lite/toco/model.h", "PReluOperator");
}
};

// LeakyRelu
//   x -> max(x, alpha * x)
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: LeakyRelu
struct LeakyReluOperator : Operator {
  LeakyReluOperator() : Operator(OperatorType::kLeakyRelu) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_35(mht_35_v, 1081, "", "./tensorflow/lite/toco/model.h", "LeakyReluOperator");
}

  float alpha = 0.2f;  // 0.2 matches the default value for the TF op attribute.
};

// Element-wise Logistic operator:
//   x -> Logistic(x) = 1 / (1 + exp(-x))
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Sigmoid
struct LogisticOperator : Operator {
  LogisticOperator() : Operator(OperatorType::kLogistic) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_36(mht_36_v, 1097, "", "./tensorflow/lite/toco/model.h", "LogisticOperator");
}
};

// Element-wise natural log operator:
//   x -> ln(x)
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Log
struct LogOperator : Operator {
  LogOperator() : Operator(OperatorType::kLog) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_37(mht_37_v, 1111, "", "./tensorflow/lite/toco/model.h", "LogOperator");
}
};

// Element-wise Tanh operator:
//   x -> Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Tanh
struct TanhOperator : Operator {
  TanhOperator() : Operator(OperatorType::kTanh) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_38(mht_38_v, 1125, "", "./tensorflow/lite/toco/model.h", "TanhOperator");
}
};

// Element-wise Sin operator:
//   x -> Sin(x) = sin(x)
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Sin
struct SinOperator : Operator {
  SinOperator() : Operator(OperatorType::kSin) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_39(mht_39_v, 1139, "", "./tensorflow/lite/toco/model.h", "SinOperator");
}
};

// Element-wise addition operator.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: Add
struct AddOperator : Operator {
  AddOperator() : Operator(OperatorType::kAdd) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_40(mht_40_v, 1153, "", "./tensorflow/lite/toco/model.h", "AddOperator");
}
};

// Element-wise addition operator for N inputs.
//
// Inputs:
//   inputs[i]: The i-th array to add together to form the output.
//
// TensorFlow equivalent: AddN
struct AddNOperator : Operator {
  AddNOperator() : Operator(OperatorType::kAddN) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_41(mht_41_v, 1166, "", "./tensorflow/lite/toco/model.h", "AddNOperator");
}
};

// Concatenation operator: concatenates its inputs
// along the axis.
//
// Inputs: this operator accepts any number >= 1 of inputs.
//   inputs[i]: the i-th array to concatenate.
//
// TensorFlow equivalent: Concat.
struct ConcatenationOperator : Operator {
  ConcatenationOperator() : Operator(OperatorType::kConcatenation) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_42(mht_42_v, 1180, "", "./tensorflow/lite/toco/model.h", "ConcatenationOperator");
}
  int axis = 0;
};

// Reordering dimensions. Used only during tooling to transform graphs from
// the TensorFlow format.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: none. This is only useful to convert between formats.
struct ReorderAxesOperator : Operator {
  ReorderAxesOperator() : Operator(OperatorType::kReorderAxes) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_43(mht_43_v, 1195, "", "./tensorflow/lite/toco/model.h", "ReorderAxesOperator");
}
  AxesOrder input_axes_order;
  AxesOrder output_axes_order;
};

// Average-pooling operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: AveragePool
struct AveragePoolOperator : Operator {
  AveragePoolOperator() : Operator(OperatorType::kAveragePool) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_44(mht_44_v, 1210, "", "./tensorflow/lite/toco/model.h", "AveragePoolOperator");
}
  Padding padding;
  int stride_height = 0;
  int stride_width = 0;
  int kheight = 0;
  int kwidth = 0;
};

// Local response normalization operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: LRN
struct LocalResponseNormalizationOperator : Operator {
  LocalResponseNormalizationOperator()
      : Operator(OperatorType::kLocalResponseNormalization) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_45(mht_45_v, 1229, "", "./tensorflow/lite/toco/model.h", "LocalResponseNormalizationOperator");
}

  int range = 0;
  float bias = 0.f;
  float alpha = 0.f;
  float beta = 0.f;
};

// Max-pooling operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: MaxPool
struct MaxPoolOperator : Operator {
  MaxPoolOperator() : Operator(OperatorType::kMaxPool) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_46(mht_46_v, 1247, "", "./tensorflow/lite/toco/model.h", "MaxPoolOperator");
}
  Padding padding;
  int stride_height = 0;
  int stride_width = 0;
  int kheight = 0;
  int kwidth = 0;
};

// L2-pooling operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: none. Can be shimmed by squaring+avgpool+sqrt.
struct L2PoolOperator : Operator {
  L2PoolOperator() : Operator(OperatorType::kL2Pool) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_47(mht_47_v, 1265, "", "./tensorflow/lite/toco/model.h", "L2PoolOperator");
}
  Padding padding;
  int stride_height = 0;
  int stride_width = 0;
  int kheight = 0;
  int kwidth = 0;
};

// The expected [min, max] range of values in a given array.
// Used for quantization only.
// This information typically comes from special nodes found in quantized
// models, see FakeQuantOperator, and is used during quantization to resolve
// actual quantization parameters (see QuantizationParams).
struct MinMax {
  double min = 0.;
  double max = 0.;
};

inline bool operator==(const MinMax& m1, const MinMax& m2) {
  return m1.min == m2.min && m1.max == m2.max;
}

inline bool operator!=(const MinMax& m1, const MinMax& m2) {
  return m1.min != m2.min || m1.max != m2.max;
}

// Fake-quantization operator. This does two things:
//   - Annotate its input and output arrays with MinMax information,
//   - Arithmetic-wise, this operator rounds incoming activation values
//     to the nearest representable value on the scale of 256
//     values from the min to the max value dictated by its MinMax info.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: optional: the 'min' value, if it has not yet been resolved
//              to a constant.
//   inputs[2]: optional: the 'max' value, if it has not yet been resolved
//              to a constant.
//
// TensorFlow equivalent: FakeQuantWithMinMaxVars, FakeQuantWithMinMaxArgs.
struct FakeQuantOperator : Operator {
  FakeQuantOperator() : Operator(OperatorType::kFakeQuant) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_48(mht_48_v, 1309, "", "./tensorflow/lite/toco/model.h", "FakeQuantOperator");
}
  std::unique_ptr<MinMax> minmax;
  int num_bits = 8;
  bool narrow_range = false;
};

// Element-wise division operator.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: Div
struct DivOperator : Operator {
  DivOperator() : Operator(OperatorType::kDiv) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_49(mht_49_v, 1326, "", "./tensorflow/lite/toco/model.h", "DivOperator");
}
};

// Element-wise identity (x->x) operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Identity
struct TensorFlowIdentityOperator : Operator {
  TensorFlowIdentityOperator() : Operator(OperatorType::kIdentity) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_50(mht_50_v, 1339, "", "./tensorflow/lite/toco/model.h", "TensorFlowIdentityOperator");
}
};

// Batch matrix multiplication operator. This comes from a tf.matmul where one
// of the operands has rank 3 or more.
//
// Inputs:
//   inputs[0]: required: the left-hand side matrix
//   inputs[1]: required: the right-hand side matrix
//
// TensorFlow equivalent: MatMul
struct BatchMatMulOperator : Operator {
  BatchMatMulOperator() : Operator(OperatorType::kBatchMatMul) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_51(mht_51_v, 1354, "", "./tensorflow/lite/toco/model.h", "BatchMatMulOperator");
}
  bool adj_x = false;
  bool adj_y = false;
};

// General matrix multiplication operator. We don't want to support general
// matrix multiplication at inference time, so we resolve it during tooling
// to more specific operator types, namely, FullyConnected.
//
// Inputs:
//   inputs[0]: required: the left-hand side matrix
//   inputs[1]: required: the right-hand side matrix
//
// TensorFlow equivalent: MatMul
struct TensorFlowMatMulOperator : Operator {
  TensorFlowMatMulOperator() : Operator(OperatorType::kMatMul) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_52(mht_52_v, 1372, "", "./tensorflow/lite/toco/model.h", "TensorFlowMatMulOperator");
}
  bool transpose_a = false;
  bool transpose_b = false;
};

// Padding operator. Pads a tensor with zeros.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the padding array
//
// This operation pads a `input` with zeros according to the `paddings` you
// specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
// rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
// how many zeros to add before the contents of `input` in that dimension, and
// `paddings[D, 1]` indicates how many zeros to add after the contents of
// `input` in that dimension.
//
// TensorFlow equivalent: Pad
struct PadOperator : Operator {
  PadOperator() : Operator(OperatorType::kPad) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_53(mht_53_v, 1395, "", "./tensorflow/lite/toco/model.h", "PadOperator");
}

  std::vector<int> left_padding;
  std::vector<int> right_padding;
};

// PaddingV2 operator. Pads a tensor with the given constant value.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the padding array
//   inputs[2]: required: the scalar constant_values
//
// This operation pads input according to the paddings and constant_values you
// specify. paddings is an integer tensor with shape [Dn, 2], where n is the
// rank of input. For each dimension D of input, paddings[D, 0] indicates how
// many padding values to add before the contents of input in that dimension,
// and paddings[D, 1] indicates how many padding values to add after the
// contents of input in that dimension. constant_values is a scalar tensor of
// the same type as input that indicates the value to use for padding input.
//
// TensorFlow equivalent: PadV2
struct PadV2Operator : Operator {
  PadV2Operator() : Operator(OperatorType::kPadV2) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_54(mht_54_v, 1421, "", "./tensorflow/lite/toco/model.h", "PadV2Operator");
}

  std::vector<int> left_padding;
  std::vector<int> right_padding;
};

// Strided slice operator.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the begin array
//   inputs[2]: required: the end array
//   inputs[3]: optional: the strides array
//
// TensorFlow equivalent: StridedSlice
struct StridedSliceOperator : Operator {
  StridedSliceOperator() : Operator(OperatorType::kStridedSlice) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_55(mht_55_v, 1440, "", "./tensorflow/lite/toco/model.h", "StridedSliceOperator");
}

  std::vector<int> start_indices;
  std::vector<int> stop_indices;
  std::vector<int> strides;

  int begin_mask;
  int ellipsis_mask;
  int end_mask;
  int new_axis_mask;
  int shrink_axis_mask;

  StridedSliceOperator(const StridedSliceOperator& other)
      : Operator(OperatorType::kStridedSlice) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_56(mht_56_v, 1456, "", "./tensorflow/lite/toco/model.h", "StridedSliceOperator");

    inputs = other.inputs;
    outputs = other.outputs;

    start_indices = other.start_indices;
    stop_indices = other.stop_indices;
    strides = other.strides;

    begin_mask = other.begin_mask;
    ellipsis_mask = other.ellipsis_mask;
    end_mask = other.end_mask;
    new_axis_mask = other.new_axis_mask;
    shrink_axis_mask = other.shrink_axis_mask;
  }

  void PadIndices(int dim_count) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_57(mht_57_v, 1474, "", "./tensorflow/lite/toco/model.h", "PadIndices");

    // Add indices and mask bits to fully include extra dimensions
    CHECK_GE(dim_count, start_indices.size());
    CHECK_EQ(start_indices.size(), stop_indices.size());
    CHECK_EQ(stop_indices.size(), strides.size());

    for (int i = start_indices.size(); i < dim_count; i++) {
      start_indices.push_back(0);
      stop_indices.push_back(0);
      strides.push_back(1);
      begin_mask |= 1 << i;
      end_mask |= 1 << i;
    }
  }

  void ReverseIndices() {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_58(mht_58_v, 1492, "", "./tensorflow/lite/toco/model.h", "ReverseIndices");

    CHECK_EQ(start_indices.size(), stop_indices.size());
    CHECK_EQ(stop_indices.size(), strides.size());

    std::reverse(start_indices.begin(), start_indices.end());
    std::reverse(stop_indices.begin(), stop_indices.end());
    std::reverse(strides.begin(), strides.end());

    begin_mask = toco::port::ReverseBits32(static_cast<uint32>(begin_mask)) >>
                 (32 - start_indices.size());
    ellipsis_mask =
        toco::port::ReverseBits32(static_cast<uint32>(ellipsis_mask)) >>
        (32 - start_indices.size());
    end_mask = toco::port::ReverseBits32(static_cast<uint32>(end_mask)) >>
               (32 - start_indices.size());
    new_axis_mask =
        toco::port::ReverseBits32(static_cast<uint32>(new_axis_mask)) >>
        (32 - start_indices.size());
    shrink_axis_mask =
        toco::port::ReverseBits32(static_cast<uint32>(shrink_axis_mask)) >>
        (32 - start_indices.size());
  }
};

// Reshaping operator, reshaping its input array to a two-dimensional shape
// (a "matrix"). This is used in the TensorFlow format, in conjunction with
// MatMul nodes, to implement fully-connected layers.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: optional: the output tensor shape
//
// TensorFlow equivalent: Reshape --- except that we only support a special case
// here, where the output shape is a matrix (2D) shape.
struct TensorFlowReshapeOperator : Operator {
  TensorFlowReshapeOperator() : Operator(OperatorType::kReshape) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_59(mht_59_v, 1530, "", "./tensorflow/lite/toco/model.h", "TensorFlowReshapeOperator");
}
  std::vector<int> shape;
};

// Removes dimensions of size 1 from the shape of a tensor.
// https://www.tensorflow.org/api_docs/python/tf/squeeze
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Squeeze
struct SqueezeOperator : Operator {
  SqueezeOperator() : Operator(OperatorType::kSqueeze) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_60(mht_60_v, 1545, "", "./tensorflow/lite/toco/model.h", "SqueezeOperator");
}

  std::vector<int> squeeze_dims;
};

// Inputs:
//   inputs[0]: required: the output shape
//   inputs[1]: required: the weights
//   inputs[2]: required: the input activations array
//   inputs[3]: optional: the bias vector, specifying the biases for each output
//                        channel.
//   NOTE: The input activations is NOT the first input.
//
//
// Outputs:
//   outputs[0]: required: the output activations array
//
// TensorFlow equivalent: Conv2DBackpropInput
struct TransposeConvOperator : Operator {
  enum Inputs {
    OUTPUT_SHAPE = 0,
    WEIGHTS = 1,
    DATA_INPUT = 2,
    BIAS = 3,
  };

  TransposeConvOperator() : Operator(OperatorType::kTransposeConv) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_61(mht_61_v, 1574, "", "./tensorflow/lite/toco/model.h", "TransposeConvOperator");
}
  Padding padding;
  int stride_width = 0;
  int stride_height = 0;
  // Dilation is possible with transpose convolution, but Tensorflow does not
  // currently support it, so we omit it.
};

// Given a tensor input, this operation calculates element-wise exponential
// (y = e^x).
//
// Inputs:
//   inputs[0]: required: input tensor
//
// TensorFlow equivalent: Exp
struct ExpOperator : Operator {
  ExpOperator() : Operator(OperatorType::kExp) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_62(mht_62_v, 1593, "", "./tensorflow/lite/toco/model.h", "ExpOperator");
}
};

// Given a tensor input, this operation calculates element-wise exponential
// (y = cos(x)).
//
// Inputs:
//   inputs[0]: required: input tensor
//
// TensorFlow equivalent: Cos
struct CosOperator : Operator {
  CosOperator() : Operator(OperatorType::kCos) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_63(mht_63_v, 1607, "", "./tensorflow/lite/toco/model.h", "CosOperator");
}
};

// Given a tensor input, this operation inserts a dimension of 1 at the
// dimension index axis of input's shape. The dimension index axis starts at
// zero; if you specify a negative number for axis it is counted backward from
// the end.
//
// Inputs:
//   inputs[0]: required: input tensor
//   inputs[1]: required: 0-D (scalar). Specifies the dimension index at which
//   to expand the shape of input
//
// TensorFlow equivalent: ExpandDims
struct ExpandDimsOperator : Operator {
  ExpandDimsOperator() : Operator(OperatorType::kExpandDims) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_64(mht_64_v, 1625, "", "./tensorflow/lite/toco/model.h", "ExpandDimsOperator");
}
};

// Creates a tensor of shape dims and fills it with the given scalar value.
// Output type will be the same as the given scalar value.
//
// Inputs:
//   inputs[0]: required: 1-D (int32) - the shape of the output tensor
//   inputs[1]: required: 0-D (scalar) - value to fill the tensor with
//
// TensorFlow equivalent: Fill
struct FillOperator : Operator {
  FillOperator() : Operator(OperatorType::kFill) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_65(mht_65_v, 1640, "", "./tensorflow/lite/toco/model.h", "FillOperator");
}
};

// Element-wise floor division operator.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: FloorDiv
struct FloorDivOperator : Operator {
  FloorDivOperator() : Operator(OperatorType::kFloorDiv) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_66(mht_66_v, 1654, "", "./tensorflow/lite/toco/model.h", "FloorDivOperator");
}
};

// Element-wise floor mod operator.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: FloorMod
struct FloorModOperator : Operator {
  FloorModOperator() : Operator(OperatorType::kFloorMod) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_67(mht_67_v, 1668, "", "./tensorflow/lite/toco/model.h", "FloorModOperator");
}
};

struct RandomUniformOperator : Operator {
  RandomUniformOperator() : Operator(OperatorType::kRandomUniform) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_68(mht_68_v, 1675, "", "./tensorflow/lite/toco/model.h", "RandomUniformOperator");
}
  ArrayDataType dtype = ArrayDataType::kNone;
  int64_t seed;
  int64_t seed2;
};

// Creates a sequence of numbers that begins at start and extends by increments
// of delta up to but not including limit.
//
// The dtype of the resulting tensor is inferred from the inputs unless it is
// provided explicitly.
//
// Inputs:
//   inputs[0]: required: the start
//   inputs[1]: required: the limit
//   inputs[2]: required: the delta
//
// TensorFlow equivalent: Range
struct RangeOperator : Operator {
  RangeOperator() : Operator(OperatorType::kRange) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_69(mht_69_v, 1697, "", "./tensorflow/lite/toco/model.h", "RangeOperator");
}
  ArrayDataType dtype = ArrayDataType::kNone;
};

// Rank operator. Extracts the rank of the tensor.
//
// Inputs:
//   inputs[0]: required: the input array
//
// This operation outputs a 0-D int32 Tensor representing the rank of input.
//
// TensorFlow equivalent: Rank.
struct TensorFlowRankOperator : Operator {
  TensorFlowRankOperator() : Operator(OperatorType::kRank) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_70(mht_70_v, 1713, "", "./tensorflow/lite/toco/model.h", "TensorFlowRankOperator");
}
  ArrayDataType output_data_type = ArrayDataType::kInt32;
};

// Element-wise negation (-x) operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Neg
struct NegOperator : Operator {
  NegOperator() : Operator(OperatorType::kNeg) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_71(mht_71_v, 1727, "", "./tensorflow/lite/toco/model.h", "NegOperator");
}
};

// Element-wise select operator choosing elements from inputs[1] or input[2]
//
// Inputs:
//  inputs[0]: required: boolean mask per index
//  inputs[1]: required: tensor of values if true
//  inputs[2]: required: tensor of values if false
//
//  TensorFlow equivalent: Select
struct SelectOperator : Operator {
  SelectOperator() : Operator(OperatorType::kSelect) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_72(mht_72_v, 1742, "", "./tensorflow/lite/toco/model.h", "SelectOperator");
}
};

// Element-wise reciprocal-square-root (x^-0.5) operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Rsqrt
struct TensorFlowRsqrtOperator : Operator {
  TensorFlowRsqrtOperator() : Operator(OperatorType::kRsqrt) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_73(mht_73_v, 1755, "", "./tensorflow/lite/toco/model.h", "TensorFlowRsqrtOperator");
}
};

// Stacks a list of rank-R tensors into one rank-(R+1) tensor.
//
// Packs the list of tensors in values into a tensor with rank one higher than
// each tensor in values, by packing them along the axis dimension. Given a list
// of length N of tensors of shape (A, B, C);.
//
// Inputs: this operator accepts any number >= 1 of inputs.
//   inputs[i]: the i-th array to merge.
//
// TensorFlow equivalent: Pack
struct PackOperator : Operator {
  PackOperator() : Operator(OperatorType::kPack) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_74(mht_74_v, 1772, "", "./tensorflow/lite/toco/model.h", "PackOperator");
}
  int values_count;
  int axis = 0;
  ArrayDataType dtype = ArrayDataType::kNone;
};

// Shape operator. Extracts the shape of the tensor.
//
// Inputs:
//   inputs[0]: required: the input array
//
// This operation outputs a 1-D integer tensor representing the shape of
// the input.
//
// TensorFlow equivalent: Shape.
struct TensorFlowShapeOperator : Operator {
  TensorFlowShapeOperator() : Operator(OperatorType::kShape) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_75(mht_75_v, 1791, "", "./tensorflow/lite/toco/model.h", "TensorFlowShapeOperator");
}
  ArrayDataType output_data_type = ArrayDataType::kInt32;
};

// Element-wise square-root (x^0.5) operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Sqrt
struct TensorFlowSqrtOperator : Operator {
  TensorFlowSqrtOperator() : Operator(OperatorType::kSqrt) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_76(mht_76_v, 1805, "", "./tensorflow/lite/toco/model.h", "TensorFlowSqrtOperator");
}
};

// Element-wise square (x*x) operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Square
struct TensorFlowSquareOperator : Operator {
  TensorFlowSquareOperator() : Operator(OperatorType::kSquare) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_77(mht_77_v, 1818, "", "./tensorflow/lite/toco/model.h", "TensorFlowSquareOperator");
}
};

// Element-wise squared difference ((x-y)*(x-y)) operator.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: SquaredDifference
struct SquaredDifferenceOperator : Operator {
  SquaredDifferenceOperator() : Operator(OperatorType::kSquaredDifference) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_78(mht_78_v, 1832, "", "./tensorflow/lite/toco/model.h", "SquaredDifferenceOperator");
}
};

// Transposes a tensor.
//
// By default, this operation performs a regular matrix transpose on 2-D input
// tensors.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Transpose
struct TransposeOperator : Operator {
  TransposeOperator() : Operator(OperatorType::kTranspose) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_79(mht_79_v, 1848, "", "./tensorflow/lite/toco/model.h", "TransposeOperator");
}
  std::vector<int> perm;
};

// Element-wise subtraction operator.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: Sub
struct SubOperator : Operator {
  SubOperator() : Operator(OperatorType::kSub) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_80(mht_80_v, 1863, "", "./tensorflow/lite/toco/model.h", "SubOperator");
}
};

// Sum reduction: computes the sum of all of entries across the axes.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Sum
struct TensorFlowSumOperator : Operator {
  TensorFlowSumOperator() : Operator(OperatorType::kSum) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_81(mht_81_v, 1876, "", "./tensorflow/lite/toco/model.h", "TensorFlowSumOperator");
}
  std::vector<int> axis;
  bool keep_dims = false;
};

// Prod reduction: computes the product of all of entries across the axes.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Prod
struct TensorFlowProdOperator : Operator {
  TensorFlowProdOperator() : Operator(OperatorType::kReduceProd) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_82(mht_82_v, 1891, "", "./tensorflow/lite/toco/model.h", "TensorFlowProdOperator");
}
  std::vector<int> axis;
  bool keep_dims = false;
};

// TensorFlow Tile equivalent. Refer to TensorFlow documentation for details.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: int array with length of rank(input[0])
struct TensorFlowTileOperator : Operator {
  TensorFlowTileOperator() : Operator(OperatorType::kTile) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_83(mht_83_v, 1905, "", "./tensorflow/lite/toco/model.h", "TensorFlowTileOperator");
}
};

// TensorFlow Slice equivalent. Refer to TensorFlow documentation for details.
struct SliceOperator : Operator {
  SliceOperator() : Operator(OperatorType::kSlice) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_84(mht_84_v, 1913, "", "./tensorflow/lite/toco/model.h", "SliceOperator");
}

  std::vector<int> begin;
  std::vector<int> size;
};

// TensorFlow Split equivalent. Refer to TensorFlow documentation for details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
struct TensorFlowSplitOperator : Operator {
  TensorFlowSplitOperator() : Operator(OperatorType::kSplit) {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_85(mht_85_v, 1926, "", "./tensorflow/lite/toco/model.h", "TensorFlowSplitOperator");
}
  int num_split = 0;
};

// TensorFlow SplitV equivalent. Refer to TensorFlow documentation for details.
struct TensorFlowSplitVOperator : Operator {
  TensorFlowSplitVOperator() : Operator(OperatorType::kSplitV) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_86(mht_86_v, 1935, "", "./tensorflow/lite/toco/model.h", "TensorFlowSplitVOperator");
}
  int num_split = 0;
};

// TensorFlow Concat equivalent. Refer to TensorFlow documentation for details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Concretely, once the concat dim becomes known, if it is the depth
// dimension then we can change this op into a DepthConcatenation op.
// Otherwise, we hope for some other graph transformation to drop this node.
struct TensorFlowConcatOperator : Operator {
  TensorFlowConcatOperator() : Operator(OperatorType::kConcat) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_87(mht_87_v, 1949, "", "./tensorflow/lite/toco/model.h", "TensorFlowConcatOperator");
}
};

// TensorFlow ConcatV2 equivalent. Refer to TensorFlow documentation for
// details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Concretely, once the concat dim becomes known, if it is the depth
// dimension then we can change this op into a DepthConcatenation op.
// Otherwise, we hope for some other graph transformation to drop this node.
struct TensorFlowConcatV2Operator : Operator {
  TensorFlowConcatV2Operator() : Operator(OperatorType::kConcatV2) {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_88(mht_88_v, 1963, "", "./tensorflow/lite/toco/model.h", "TensorFlowConcatV2Operator");
}
};

// TensorFlow Merge equivalent. Refer to TensorFlow documentation for details.
//
// Inputs: this operator accepts any number >= 1 of inputs.
//   inputs[i]: the i-th array to merge.
//
// It is expected that graph transformations will drop all but exactly one
// of the inputs, at which point the Merge node will be equivalent to an
// Identity node forwarding the remaining input.
//
// Note: We do not currently support runtime control flow: we only support
// control flow that can be resolved at tooling time (independently of input
// activations).
struct TensorFlowMergeOperator : Operator {
  TensorFlowMergeOperator() : Operator(OperatorType::kMerge) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_89(mht_89_v, 1982, "", "./tensorflow/lite/toco/model.h", "TensorFlowMergeOperator");
}
};

// TensorFlow Switch equivalent. Refer to TensorFlow documentation for details.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the boolean predicate, given as an array of size 1
//     and of type kBool, will determine which output gets selected.
//
// Outputs: a TensorFlow Switch node always has exactly two outputs. Depending
// on the boolean value that the input predicate resolves to (see note below),
// one or the other of the outputs will be 'selected': the input array will be
// forwarded to the 'selected output' as if by a Identity node, while the other
// output will be discarded, and any graph edge connecting that discarded output
// will be dropped. The rule for selecting outputs is as follows:
//   outputs[0] will be selected if the input predicate resolves to 'true'.
//   outputs[1] will be selected if the input predicate resolves to 'false'.
//
// Note: We do not currently support runtime control flow: we only support
// control flow that can be resolved at tooling time (independently of input
// activations).
struct TensorFlowSwitchOperator : Operator {
  TensorFlowSwitchOperator() : Operator(OperatorType::kSwitch) {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_90(mht_90_v, 2008, "", "./tensorflow/lite/toco/model.h", "TensorFlowSwitchOperator");
}
};

// TensorFlow All equivalent. Refer to TensorFlow documentation for details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Typically, this is only used as an input to an Assert node, so can be
// removed as an unused node as we drop Assert nodes.
struct TensorFlowAllOperator : Operator {
  TensorFlowAllOperator() : Operator(OperatorType::kAll) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_91(mht_91_v, 2020, "", "./tensorflow/lite/toco/model.h", "TensorFlowAllOperator");
}
};

// TensorFlow Assert equivalent. Refer to TensorFlow documentation for details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Typically, we just drop Assert nodes.
struct TensorFlowAssertOperator : Operator {
  TensorFlowAssertOperator() : Operator(OperatorType::kAssert) {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_92(mht_92_v, 2031, "", "./tensorflow/lite/toco/model.h", "TensorFlowAssertOperator");
}
};

// TensorFlow Less equivalent. Refer to TensorFlow documentation for details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Typically, this is only used as an input to an Assert node, so can be
// removed as an unused node as we drop Assert nodes.
struct TensorFlowLessOperator : Operator {
  TensorFlowLessOperator() : Operator(OperatorType::kLess) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_93(mht_93_v, 2043, "", "./tensorflow/lite/toco/model.h", "TensorFlowLessOperator");
}
};

// TensorFlow LessEqual equivalent. Refer to TensorFlow documentation for
// details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Typically, this is only used as an input to an Assert node, so can be
// removed as an unused node as we drop Assert nodes.
struct TensorFlowLessEqualOperator : Operator {
  TensorFlowLessEqualOperator() : Operator(OperatorType::kLessEqual) {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_94(mht_94_v, 2056, "", "./tensorflow/lite/toco/model.h", "TensorFlowLessEqualOperator");
}
};

// TensorFlow Less equivalent. Refer to TensorFlow documentation for details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Typically, this is only used as an input to an Assert node, so can be
// removed as an unused node as we drop Assert nodes.
struct TensorFlowGreaterOperator : Operator {
  TensorFlowGreaterOperator() : Operator(OperatorType::kGreater) {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_95(mht_95_v, 2068, "", "./tensorflow/lite/toco/model.h", "TensorFlowGreaterOperator");
}
};

// TensorFlow GreaterEqual equivalent. Refer to TensorFlow documentation for
// details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Typically, this is only used as an input to an Assert node, so can be
// removed as an unused node as we drop Assert nodes.
struct TensorFlowGreaterEqualOperator : Operator {
  TensorFlowGreaterEqualOperator() : Operator(OperatorType::kGreaterEqual) {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_96(mht_96_v, 2081, "", "./tensorflow/lite/toco/model.h", "TensorFlowGreaterEqualOperator");
}
};

// TensorFlow Equal equivalent. Refer to TensorFlow documentation for
// details.
// Not fully supported, just a placeholder to handle TensorFlow graphs and
// support graph transformations to other operator types by matching sub-graphs.
// Typically, this is only used as an input to an Assert node, so can be
// removed as an unused node as we drop Assert nodes.
struct TensorFlowEqualOperator : Operator {
  TensorFlowEqualOperator() : Operator(OperatorType::kEqual) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_97(mht_97_v, 2094, "", "./tensorflow/lite/toco/model.h", "TensorFlowEqualOperator");
}
};

// TensorFlow Not Equal equivalent. Refer to TensorFlow documentation for
// details.
struct TensorFlowNotEqualOperator : Operator {
  TensorFlowNotEqualOperator() : Operator(OperatorType::kNotEqual) {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_98(mht_98_v, 2103, "", "./tensorflow/lite/toco/model.h", "TensorFlowNotEqualOperator");
}
};

// Max reduction: computes the max of all of entries across the axes.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Max
struct TensorFlowMaxOperator : Operator {
  TensorFlowMaxOperator() : Operator(OperatorType::kReduceMax) {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_99(mht_99_v, 2116, "", "./tensorflow/lite/toco/model.h", "TensorFlowMaxOperator");
}
  std::vector<int> axis;
  bool keep_dims = false;
};

// Min reduction: computes the min of all of entries across the axes.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Min
struct TensorFlowMinOperator : Operator {
  TensorFlowMinOperator() : Operator(OperatorType::kReduceMin) {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_100(mht_100_v, 2131, "", "./tensorflow/lite/toco/model.h", "TensorFlowMinOperator");
}
  std::vector<int> axis;
  bool keep_dims = false;
};

// Element-wise maximum operator. Currently it only supports scalar as
// the second operand.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: Maximum
struct TensorFlowMaximumOperator : Operator {
  TensorFlowMaximumOperator() : Operator(OperatorType::kMaximum) {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_101(mht_101_v, 2148, "", "./tensorflow/lite/toco/model.h", "TensorFlowMaximumOperator");
}
};

// Element-wise minimum operator. Currently it only supports scalar as
// the second operand.
//
// Inputs:
//   inputs[0]: required: the left-hand side array
//   inputs[1]: required: the right-hand side array
//
// TensorFlow equivalent: Minimum
struct TensorFlowMinimumOperator : Operator {
  TensorFlowMinimumOperator() : Operator(OperatorType::kMinimum) {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_102(mht_102_v, 2163, "", "./tensorflow/lite/toco/model.h", "TensorFlowMinimumOperator");
}
};

// General TF operation, unsupported by tf.mini. Expected to be dropped by
// graph transformations.
struct TensorFlowUnsupportedOperator : Operator {
  TensorFlowUnsupportedOperator() : Operator(OperatorType::kUnsupported) {
   std::vector<std::string> mht_103_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_103(mht_103_v, 2172, "", "./tensorflow/lite/toco/model.h", "TensorFlowUnsupportedOperator");
}

  // The original TF operation type. Used for diagnostic purposes.
  std::string tensorflow_op;
  // A boolean indicating if the unsupported op should be treated as quantized.
  bool quantized = false;
  // A boolean indicating if the unsupported op output should allow float values
  // in quantized mode.
  bool support_output_type_float_in_quantized_op = false;
  // Output data types
  std::vector<ArrayDataType> output_data_types;
  // Output shapes.
  std::vector<Shape> output_shapes;
};

// Softmax activation function.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Softmax
struct SoftmaxOperator : Operator {
  SoftmaxOperator() : Operator(OperatorType::kSoftmax) {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_104(mht_104_v, 2197, "", "./tensorflow/lite/toco/model.h", "SoftmaxOperator");
}
  float beta = 0.f;
};

// LogSoftmax activation function.
//
// Inputs:
//   inputs[0]: required: the logits input array
//
// TensorFlow equivalent: LogSoftmax
struct LogSoftmaxOperator : Operator {
  LogSoftmaxOperator() : Operator(OperatorType::kLogSoftmax) {
   std::vector<std::string> mht_105_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_105(mht_105_v, 2211, "", "./tensorflow/lite/toco/model.h", "LogSoftmaxOperator");
}

  // LogSoftmax can in principal have very large negative output, depending on
  // the input size.  However, input x_i that is less than x_max-10 is
  // accumulated as exp(x_i-x_max), which is truncated to zero.
  //
  // Since we effectively disregard smallish inputs in the normalizing factor,
  // we also drop them in the output (set to minimum output), and in doing so
  // make better use of the quantization range / resolution.
  static constexpr float kOutputRangeMin = -16.0;
};

// Cast operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Cast
struct CastOperator : Operator {
  CastOperator() : Operator(OperatorType::kCast) {
   std::vector<std::string> mht_106_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_106(mht_106_v, 2233, "", "./tensorflow/lite/toco/model.h", "CastOperator");
}
  ArrayDataType src_data_type = ArrayDataType::kNone;
  ArrayDataType dst_data_type = ArrayDataType::kNone;
};

// Floor operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Floor
struct FloorOperator : Operator {
  FloorOperator() : Operator(OperatorType::kFloor) {
   std::vector<std::string> mht_107_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_107(mht_107_v, 2248, "", "./tensorflow/lite/toco/model.h", "FloorOperator");
}
};

// Ceil operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Ceil
struct CeilOperator : Operator {
  CeilOperator() : Operator(OperatorType::kCeil) {
   std::vector<std::string> mht_108_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_108(mht_108_v, 2261, "", "./tensorflow/lite/toco/model.h", "CeilOperator");
}
};

// Round operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Round
struct RoundOperator : Operator {
  RoundOperator() : Operator(OperatorType::kRound) {
   std::vector<std::string> mht_109_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_109(mht_109_v, 2274, "", "./tensorflow/lite/toco/model.h", "RoundOperator");
}
};

// Gather operator. It gathers slices from params according to indices.
// Only 1-D indices are supported at the moment.
//
// Inputs:
//   inputs[0]: required: the params array
//   inputs[1]: required: the indices to gather
//   inputs[2]: optional: axis
//
// TensorFlow equivalent: Gather
struct GatherOperator : Operator {
  GatherOperator() : Operator(OperatorType::kGather) {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_110(mht_110_v, 2290, "", "./tensorflow/lite/toco/model.h", "GatherOperator");
}
  // Axis is populated explicitly or implicitly from the axis input by
  // ResolveGatherAttributes. An empty axis indicates that the axis has not yet
  // be resolved.
  absl::optional<int> axis;

  // This field is not used by the standard TF Lite export but it is still need
  // for legacy Gather implementations.
  int input_rank = 0;
};

// GatherNd operator. It gathers slices from params according to indices.
//
// Inputs:
//   inputs[0]: required: the params array
//   inputs[1]: required: the indices to gather
//
// TensorFlow equivalent: GatherNd
struct GatherNdOperator : Operator {
  GatherNdOperator() : Operator(OperatorType::kGatherNd) {
   std::vector<std::string> mht_111_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_111(mht_111_v, 2312, "", "./tensorflow/lite/toco/model.h", "GatherNdOperator");
}
};

// ArgMax operator. It returns the index of the maximum value along axis.
//
// Inputs:
//   inputs[0]: required: the input tensor
//   inputs[1]: optional: 0-D (scalar) axis
//
// TensorFlow equivalent: ArgMax
struct ArgMaxOperator : Operator {
  ArgMaxOperator() : Operator(OperatorType::kArgMax) {
   std::vector<std::string> mht_112_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_112(mht_112_v, 2326, "", "./tensorflow/lite/toco/model.h", "ArgMaxOperator");
}
  ArrayDataType output_data_type = ArrayDataType::kInt64;
};

// ArgMin operator. It returns the index of the minimum value along axis.
//
// Inputs:
//   inputs[0]: required: the input tensor
//   inputs[1]: optional: 0-D (scalar) axis
//
// TensorFlow equivalent: ArgMin
struct ArgMinOperator : Operator {
  ArgMinOperator() : Operator(OperatorType::kArgMin) {
   std::vector<std::string> mht_113_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_113(mht_113_v, 2341, "", "./tensorflow/lite/toco/model.h", "ArgMinOperator");
}
  ArrayDataType output_data_type = ArrayDataType::kInt64;
};

// ResizeBilinear operator. It resizes input images with bilinear interpolation.
// It does not support align_corners at the moment.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the new image size
//
// TensorFlow equivalent: ResizeBilinear
struct ResizeBilinearOperator : Operator {
  ResizeBilinearOperator() : Operator(OperatorType::kResizeBilinear) {
   std::vector<std::string> mht_114_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_114(mht_114_v, 2357, "", "./tensorflow/lite/toco/model.h", "ResizeBilinearOperator");
}

  bool align_corners = false;
  bool half_pixel_centers = false;
};

// ResizeNearestNeighborOperator operator. It resizes input images with nearest
// neighbor interpolation. It does not support align_corners at the moment.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the new image size
//
// TensorFlow equivalent: ResizeNearestNeighbor
struct ResizeNearestNeighborOperator : Operator {
  ResizeNearestNeighborOperator()
      : Operator(OperatorType::kResizeNearestNeighbor) {
   std::vector<std::string> mht_115_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_115(mht_115_v, 2376, "", "./tensorflow/lite/toco/model.h", "ResizeNearestNeighborOperator");
}

  bool align_corners = false;
  bool half_pixel_centers = false;
};

// SpaceToBatchND operator. It divides spatial dimensions into a grid of
// blocks and interleaves these blocks with the batch dimension. Currently,
// only 2-d blocks are supported.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the block shape
//   inputs[2]: required: the paddings
//
// TensorFlow equivalent: SpaceToBatchND
struct SpaceToBatchNDOperator : Operator {
  SpaceToBatchNDOperator() : Operator(OperatorType::kSpaceToBatchND) {
   std::vector<std::string> mht_116_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_116(mht_116_v, 2396, "", "./tensorflow/lite/toco/model.h", "SpaceToBatchNDOperator");
}

  std::vector<int> block_shape;
  std::vector<int> before_paddings;
  std::vector<int> after_paddings;
};

// BatchToSpaceND operator. Rearranges data from batch into blocks of
// spatial data. Currently, only 2-d blocks are supported.
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: the block shape
//   inputs[2]: required: the crops
//
// TensorFlow equivalent: BatchToSpaceND
struct BatchToSpaceNDOperator : Operator {
  BatchToSpaceNDOperator() : Operator(OperatorType::kBatchToSpaceND) {
   std::vector<std::string> mht_117_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_117(mht_117_v, 2416, "", "./tensorflow/lite/toco/model.h", "BatchToSpaceNDOperator");
}

  std::vector<int> block_shape;
  std::vector<int> before_crops;
  std::vector<int> after_crops;
};

// Mean operator.
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Mean
struct MeanOperator : Operator {
  MeanOperator() : Operator(OperatorType::kMean) {
   std::vector<std::string> mht_118_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_118(mht_118_v, 2433, "", "./tensorflow/lite/toco/model.h", "MeanOperator");
}

  std::vector<int> axis;
  bool keep_dims = false;
};

// Svdf operator:
//
// Inputs:
//   inputs[0]: required: the input array
//   inputs[1]: required: weights_feature
//   inputs[2]: required: weights_time
//   inputs[3]: optional: bias
struct SvdfOperator : Operator {
  SvdfOperator() : Operator(OperatorType::kSvdf) {
   std::vector<std::string> mht_119_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_119(mht_119_v, 2450, "", "./tensorflow/lite/toco/model.h", "SvdfOperator");
}
  int rank;
};

// TopKV2 operator.
//
// Inputs:
//    input tensor and top_k scalar.
struct TopKV2Operator : Operator {
  TopKV2Operator() : Operator(OperatorType::kTopK_V2) {
   std::vector<std::string> mht_120_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_120(mht_120_v, 2462, "", "./tensorflow/lite/toco/model.h", "TopKV2Operator");
}
};

// DynamicPartition operator:
//
// Inputs:
//  inputs[0]: required: data.
//  inputs[1]: required: partitions.
//
// TensorFlow equivalent: DynamicPartition
struct DynamicPartitionOperator : Operator {
  DynamicPartitionOperator() : Operator(OperatorType::kDynamicPartition) {
   std::vector<std::string> mht_121_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_121(mht_121_v, 2476, "", "./tensorflow/lite/toco/model.h", "DynamicPartitionOperator");
}
  int num_partitions;
};

// DynamicStitch operator:
//
// Inputs:
//  inputs[0,N): required: indices.
//  inputs[N,2N): required: data.
//
// TensorFlow equivalent: DynamicStitch/ParallelDynamicStitch
struct DynamicStitchOperator : Operator {
  DynamicStitchOperator() : Operator(OperatorType::kDynamicStitch) {
   std::vector<std::string> mht_122_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_122(mht_122_v, 2491, "", "./tensorflow/lite/toco/model.h", "DynamicStitchOperator");
}
  int num_partitions;
};

// SparseToDense operator:
//
// Inputs:
// Inputs[0]: required: sparse_indices.
// Inputs[1]: required: output_shape.
// Inputs[2]: required: sparse_values.
//
// TensorFlow equivalent: SparseToDense.
struct SparseToDenseOperator : Operator {
  SparseToDenseOperator() : Operator(OperatorType::kSparseToDense) {
   std::vector<std::string> mht_123_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_123(mht_123_v, 2507, "", "./tensorflow/lite/toco/model.h", "SparseToDenseOperator");
}
  bool validate_indices;
};

// Pow operator:
//
// Inputs:
// Inputs[0]: required: A tensor.
// Inputs[1]: required: A tensor.
//
// TensorFlow equivalent: Pow.
struct PowOperator : Operator {
  PowOperator() : Operator(OperatorType::kPow) {
   std::vector<std::string> mht_124_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_124(mht_124_v, 2522, "", "./tensorflow/lite/toco/model.h", "PowOperator");
}
};

// Any operator:
//
// Inputs:
// Inputs[0]: required: A boolean input tensor.
// Inputs[1]: required: reduction_indices.
//
// TensorFlow equivalent: tf.reduce_any.
struct TensorFlowAnyOperator : Operator {
  TensorFlowAnyOperator() : Operator(OperatorType::kAny) {
   std::vector<std::string> mht_125_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_125(mht_125_v, 2536, "", "./tensorflow/lite/toco/model.h", "TensorFlowAnyOperator");
}
  std::vector<int> axis;
  bool keep_dims = false;
};

// LogicalAnd operator:
//
// Inputs:
// Inputs[0]: required: A boolean tensor.
// Inputs[1]: required: A boolean tensor.
//
// TensorFlow equivalent: tf.logical_and.
struct LogicalAndOperator : Operator {
  LogicalAndOperator() : Operator(OperatorType::kLogicalAnd) {
   std::vector<std::string> mht_126_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_126(mht_126_v, 2552, "", "./tensorflow/lite/toco/model.h", "LogicalAndOperator");
}
};

// LogicalNot operator:
//
// Inputs:
// Inputs[0]: required: A boolean tensor.
//
// TensorFlow equivalent: tf.logical_not.
struct LogicalNotOperator : Operator {
  LogicalNotOperator() : Operator(OperatorType::kLogicalNot) {
   std::vector<std::string> mht_127_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_127(mht_127_v, 2565, "", "./tensorflow/lite/toco/model.h", "LogicalNotOperator");
}
};

// OneHot operator:
//
// Inputs:
// Inputs[0]: required: indices.
// Inputs[1]: required: depth.
// Inputs[2]: required: on_value.
// Inputs[3]: required: off_value.
//
// TensorFlow equivalent: OneHot.
struct OneHotOperator : Operator {
  enum Inputs {
    INDICES_INPUT = 0,
    DEPTH_INPUT = 1,
    ON_VALUE_INPUT = 2,
    OFF_VALUE_INPUT = 3,
  };

  OneHotOperator() : Operator(OperatorType::kOneHot) {
   std::vector<std::string> mht_128_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_128(mht_128_v, 2588, "", "./tensorflow/lite/toco/model.h", "OneHotOperator");
}
  int axis = -1;
};

// LogicalOr operator:
//
// Inputs:
// Inputs[0]: required: A Bool tensor.
// Inputs[1]: required: A Bool tensor.
//
// TensorFlow equivalent: LogicalOr.
struct LogicalOrOperator : Operator {
  LogicalOrOperator() : Operator(OperatorType::kLogicalOr) {
   std::vector<std::string> mht_129_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_129(mht_129_v, 2603, "", "./tensorflow/lite/toco/model.h", "LogicalOrOperator");
}
};

// Unpack operator:
//
// Inputs:
// Inputs[0]: required: A boolean input tensor.
// Inputs[1]: required: reduction_indices.
//
// TensorFlow equivalent: tf.unstack.
struct UnpackOperator : Operator {
  UnpackOperator() : Operator(OperatorType::kUnpack) {
   std::vector<std::string> mht_130_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_130(mht_130_v, 2617, "", "./tensorflow/lite/toco/model.h", "UnpackOperator");
}
  int num;
  int axis;
  ArrayDataType dtype = ArrayDataType::kNone;
};

// ZerosLike operator:
//
// Inputs:
// inputs[0]: required: the input array
//
// TensorFlow equivalent: tf.zeros_like
struct TensorFlowZerosLikeOperator : Operator {
  TensorFlowZerosLikeOperator() : Operator(OperatorType::kZerosLike) {
   std::vector<std::string> mht_131_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_131(mht_131_v, 2633, "", "./tensorflow/lite/toco/model.h", "TensorFlowZerosLikeOperator");
}
};

// ReverseV2 operator:
//
// Inputs:
// Inputs[0]: required: the input array.
//
// TensorFlow equivalent: ReverseV2.
struct ReverseV2Operator : Operator {
  ReverseV2Operator() : Operator(OperatorType::kReverseV2) {
   std::vector<std::string> mht_132_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_132(mht_132_v, 2646, "", "./tensorflow/lite/toco/model.h", "ReverseV2Operator");
}
};

enum class MirrorPadMode { kNone, kSymmetric, kReflect };

// MirrorPad Operator:
//
// Inputs:
// Inputs[0]: required: input tensor to be padded.
// Inputs[1]: required: 2 Column matrix specifying padding sizes. The number of
// rows must be the same as the rank of the input.
// Inputs[2]: required: REFLECT or SYMMETRIC.
//
// TensorFlow equivalent: MirrorPad.
struct MirrorPadOperator : Operator {
  MirrorPadOperator() : Operator(OperatorType::kMirrorPad) {
   std::vector<std::string> mht_133_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_133(mht_133_v, 2664, "", "./tensorflow/lite/toco/model.h", "MirrorPadOperator");
}
  // mode is either SYMMETRIC or REFLECT.
  MirrorPadMode mode;
};

// ReverseSequence operator:
//
// Inputs:
// Inputs[0]: required: the input array.
// Inputs[1]: required: the lengths of the elements to be reversed.
//
// TensorFlow equivalent: tf.reverse_sequence.
struct ReverseSequenceOperator : Operator {
  ReverseSequenceOperator() : Operator(OperatorType::kReverseSequence) {
   std::vector<std::string> mht_134_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_134(mht_134_v, 2680, "", "./tensorflow/lite/toco/model.h", "ReverseSequenceOperator");
}
  int seq_dim;
  int batch_dim = 0;
};

// Unique Operator:
//
// Inputs:
//   inputs[0]: required: the input array
//
// TensorFlow equivalent: Unique
struct UniqueOperator : Operator {
  UniqueOperator() : Operator(OperatorType::kUnique) {
   std::vector<std::string> mht_135_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_135(mht_135_v, 2695, "", "./tensorflow/lite/toco/model.h", "UniqueOperator");
}
  ArrayDataType idx_out_type = ArrayDataType::kInt32;
};

struct UnidirectionalSequenceRnnOperator : Operator {
  UnidirectionalSequenceRnnOperator()
      : Operator(OperatorType::kUnidirectionalSequenceRnn) {
   std::vector<std::string> mht_136_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_136(mht_136_v, 2704, "", "./tensorflow/lite/toco/model.h", "UnidirectionalSequenceRnnOperator");
}
  bool time_major;
  FusedActivationFunctionType fused_activation_function;
};

// Where Operator:
// Return the coordinates of the true values in condition tensor in row-major
// order.
//
// Inputs:
//  inputs[0]: required: boolean condition tensor
//
//  TensorFlow equivalent: Where
struct WhereOperator : Operator {
  WhereOperator() : Operator(OperatorType::kWhere) {
   std::vector<std::string> mht_137_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_137(mht_137_v, 2721, "", "./tensorflow/lite/toco/model.h", "WhereOperator");
}
};

// Matrix Diag Operator:
// Construct a batched diagonal tensor with given batched diagonal values.
// Inputs: A tensor of values that will be on the diagonal of the returned
//         tensor.
struct MatrixDiagOperator : Operator {
  MatrixDiagOperator() : Operator(OperatorType::kMatrixDiag) {
   std::vector<std::string> mht_138_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_138(mht_138_v, 2732, "", "./tensorflow/lite/toco/model.h", "MatrixDiagOperator");
}
};

// Matrix Diag Operator V2:
// Construct a batched diagonal tensor with given batched diagonal values.
// Not fully supported, contains 4 extra inputs compared to MatrixDiag. Behave
// like MatrixDiag when default parameters are used.
struct MatrixDiagV2Operator : Operator {
  MatrixDiagV2Operator() : Operator(OperatorType::kMatrixDiagV2) {
   std::vector<std::string> mht_139_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_139(mht_139_v, 2743, "", "./tensorflow/lite/toco/model.h", "MatrixDiagV2Operator");
}
};

// Matrix Diag Operator V3:
// Construct a batched diagonal tensor with given batched diagonal values.
// Not fully supported, contains 5 extra inputs compared to MatrixDiag. Behave
// like MatrixDiag when default parameters are used.
// V3 is only different from V2 because it has an extra attribute (align) which
// controls the alignment of diagonals in the band matrix (compact) format.
// The alignment in V2 contradicts with the default alignment in V3 so V2 is
// skipped. (It has never been, and should never be, exposed in the public API.)
struct MatrixDiagV3Operator : Operator {
  MatrixDiagV3Operator() : Operator(OperatorType::kMatrixDiagV3) {
   std::vector<std::string> mht_140_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_140(mht_140_v, 2758, "", "./tensorflow/lite/toco/model.h", "MatrixDiagV3Operator");
}
};

// Matrix Set Diag Operator:
// Construct a batched diagonal tensor with given input and diagonal values.
// Input is a rank (k+1) tensor of values.
// diagonal is a rank (k) tensor of values that will be on the diagonal
// of the returned output. Output is rank k+1.
//         tensor.
struct MatrixSetDiagOperator : Operator {
  MatrixSetDiagOperator() : Operator(OperatorType::kMatrixSetDiag) {
   std::vector<std::string> mht_141_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_141(mht_141_v, 2771, "", "./tensorflow/lite/toco/model.h", "MatrixSetDiagOperator");
}
};

// Matrix Set Diag Operator V2:
// Construct a batched diagonal tensor with given input and diagonal values.
// Not fully supported, contains 1 extra inputs compared to MatrixSetDiag.
// Behave like MatrixSetDiag when default parameters are used.
struct MatrixSetDiagV2Operator : Operator {
  MatrixSetDiagV2Operator() : Operator(OperatorType::kMatrixSetDiagV2) {
   std::vector<std::string> mht_142_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_142(mht_142_v, 2782, "", "./tensorflow/lite/toco/model.h", "MatrixSetDiagV2Operator");
}
};

// Matrix Set Diag Operator V3:
// Construct a batched diagonal tensor with given input and diagonal values.
// Not fully supported, contains 2 extra inputs compared to MatrixSetDiag.
// Behave like MatrixSetDiag when default parameters are used.
// V3 is only different from V2 because it has an extra attribute (align) which
// controls the alignment of diagonals in the band matrix (compact) format.
// The alignment in V2 contradicts with the default alignment in V3 so V2 is
// skipped. (It has never been, and should never be, exposed in the public API.)
struct MatrixSetDiagV3Operator : Operator {
  MatrixSetDiagV3Operator() : Operator(OperatorType::kMatrixSetDiagV3) {
   std::vector<std::string> mht_143_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_143(mht_143_v, 2797, "", "./tensorflow/lite/toco/model.h", "MatrixSetDiagV3Operator");
}
};

struct ScatterNdOperator : Operator {
  ScatterNdOperator() : Operator(OperatorType::kScatterNd) {
   std::vector<std::string> mht_144_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_144(mht_144_v, 2804, "", "./tensorflow/lite/toco/model.h", "ScatterNdOperator");
}
};

struct SegmentSumOperator : Operator {
  SegmentSumOperator() : Operator(OperatorType::kSegmentSum) {
   std::vector<std::string> mht_145_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_145(mht_145_v, 2811, "", "./tensorflow/lite/toco/model.h", "SegmentSumOperator");
}
};

// Alloc's are used for transient arrays only. An Alloc specifies which interval
// of the "transient_data" workspace buffer passed to inference functions, is to
// be used for the transient array at hand. The 'start' and 'end' values are
// offsets from the start of the workspace buffer, expressed in bytes.
struct Alloc {
  int64_t start = 0;
  int64_t end = 0;
};

inline bool operator<(const Alloc& a, const Alloc& b) {
   std::vector<std::string> mht_146_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_146(mht_146_v, 2826, "", "./tensorflow/lite/toco/model.h", "operator<");

  return a.start < b.start;
}

// Array represents an array (either a constant parameter array or an
// activations array) in a Model.
struct Array {
  template <ArrayDataType A>
  const Buffer<A>& GetBuffer() const {
   std::vector<std::string> mht_147_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_147(mht_147_v, 2837, "", "./tensorflow/lite/toco/model.h", "GetBuffer");

    DCHECK(buffer);
    DCHECK(buffer->type == A);
    return *static_cast<const Buffer<A>*>(buffer.get());
  }
  template <ArrayDataType A>
  Buffer<A>& GetMutableBuffer() {
   std::vector<std::string> mht_148_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_148(mht_148_v, 2846, "", "./tensorflow/lite/toco/model.h", "GetMutableBuffer");

    if (!buffer) {
      Buffer<A>* ptr = new Buffer<A>;
      buffer = std::unique_ptr<GenericBuffer>(ptr);
    }
    DCHECK(buffer);
    DCHECK(buffer->type == A);
    return *static_cast<Buffer<A>*>(buffer.get());
  }
  Alloc& GetOrCreateAlloc() {
   std::vector<std::string> mht_149_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_149(mht_149_v, 2858, "", "./tensorflow/lite/toco/model.h", "GetOrCreateAlloc");

    if (!alloc) {
      alloc = std::unique_ptr<Alloc>(new Alloc);
    }
    return *alloc;
  }
  MinMax& GetOrCreateMinMax() {
   std::vector<std::string> mht_150_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_150(mht_150_v, 2867, "", "./tensorflow/lite/toco/model.h", "GetOrCreateMinMax");

    if (!minmax) {
      minmax = std::unique_ptr<MinMax>(new MinMax);
    }
    return *minmax;
  }
  MinMax& GetMinMax() const {
   std::vector<std::string> mht_151_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_151(mht_151_v, 2876, "", "./tensorflow/lite/toco/model.h", "GetMinMax");

    DCHECK(minmax);
    return *minmax;
  }
  QuantizationParams& GetOrCreateQuantizationParams() {
   std::vector<std::string> mht_152_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_152(mht_152_v, 2883, "", "./tensorflow/lite/toco/model.h", "GetOrCreateQuantizationParams");

    if (!quantization_params) {
      quantization_params =
          std::unique_ptr<QuantizationParams>(new QuantizationParams);
    }
    return *quantization_params;
  }
  QuantizationParams& GetQuantizationParams() const {
   std::vector<std::string> mht_153_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_153(mht_153_v, 2893, "", "./tensorflow/lite/toco/model.h", "GetQuantizationParams");

    DCHECK(quantization_params);
    return *quantization_params;
  }

  // The data type of the actual elements of this array, that is:
  //  - If there is a buffer (see 'buffer' member), it must be of the same
  //    type.
  //  - If there is no buffer, meaning that this is a runtime (i.e. activations)
  //    array, then this specifies the type of elements that there will be
  //    at runtime.
  //
  // Note that this only specifies the storage type of elements; this does
  // not specify whether these are to be treated as 'real' or 'quantized'
  // values.
  // That is decided by whether the 'quantization_params' member is null.
  ArrayDataType data_type = ArrayDataType::kNone;
  // The final value that data_type should have at the end of graph
  // transformations
  ArrayDataType final_data_type = ArrayDataType::kNone;
  // The dimensions of this array --- this specifies both sizes and strides
  // (the storage layout).
  //
  // Issues with shape handling that remain include:
  //   - No way to distinguish between 0-dimensional dims and missing dims.
  //   - No way to describe dims that may be runtime-variable.
  //   - Addressing of dims by integer index differs in different graph formats
  //     (TensorFlow vs. other frameworks vs. what we have informally grown
  //     within toco).
  //     This is currently quite messy; see ReorderAxesOperator which is how we
  //     bridge some of these discrepancies at the moment. This is overdue for
  //     a redesign; I'm thinking that it would be nice to have more flexible
  //     dims that allow mapping 1:1, cleanly, dims as they are in various
  //     formats,
  //     then explicitly convert between different conventions.

  // Proto-style accessors
  bool has_shape() const {
   std::vector<std::string> mht_154_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_154(mht_154_v, 2933, "", "./tensorflow/lite/toco/model.h", "has_shape");
 return array_shape != nullptr; }
  const Shape& shape() const {
   std::vector<std::string> mht_155_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_155(mht_155_v, 2937, "", "./tensorflow/lite/toco/model.h", "shape");

    CHECK(has_shape());
    return *array_shape;
  }
  Shape* mutable_shape() {
   std::vector<std::string> mht_156_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_156(mht_156_v, 2944, "", "./tensorflow/lite/toco/model.h", "mutable_shape");

    if (!array_shape) {
      array_shape.reset(new Shape);
    }
    return array_shape.get();
  }
  void copy_shape(const Shape& src_shape) {
   std::vector<std::string> mht_157_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_157(mht_157_v, 2953, "", "./tensorflow/lite/toco/model.h", "copy_shape");
 *mutable_shape() = src_shape; }
  void clear_shape() {
   std::vector<std::string> mht_158_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_158(mht_158_v, 2957, "", "./tensorflow/lite/toco/model.h", "clear_shape");
 array_shape = nullptr; }

  // The constant buffer backing this array. This is non-null if and only if
  // this is a constant parameter array. Conversely, this is null for
  // activations arrays.
  //
  // Note that this buffer is pure storage. In the case of quantized values,
  // it only stores the quantized values, it does not know by itself about the
  // quantization parameters necessary to interprete these values, that is
  // in the separate 'quantization_params' field. In fact, this 'buffer' field
  // does no even know whether values are quantized. It only has a data_type,
  // which must equal the 'data_type' member here, and which only describes
  // the storage type of element, does not tell whether they are quantized i.e.
  // whether they are to be interpreted with quantization_params.
  std::unique_ptr<GenericBuffer> buffer;
  // Only for activation arrays (i.e. when 'buffer' is null).
  // Only for code generation.
  //
  // Describes the allocation of this array within the workspace buffer
  // allocated
  // for all transient arrays.
  std::unique_ptr<Alloc> alloc;
  // Describes the [min, max] range of values
  // to be assumed when determining quantization_params.
  //
  // Only used for quantization. In fact, only used for determining
  // quantization_params.
  //
  // Used for both constant arrays (those having a 'buffer') and non-constant
  // arrays (activations). Indeed, it is important to use the same min-max range
  // as was used during training, even if that min-max range is slightly wrong
  // w.r.t. actual buffer elements. Doing otherwise would defeat the point of
  // re-training for quantization.
  std::unique_ptr<MinMax> minmax;
  // Quantization parameters. The non-null-ness of this pointer is what
  // defines whether this array is quantized or not.
  //
  // If this is non-null, then these quantization parameters are to be used
  // to assign a meaning as real numbers to the elements of this array.
  std::unique_ptr<QuantizationParams> quantization_params;
  // narrow_range is a detail of how toco handles FakeQuant operators with
  // narrow_range, see
  // https://www.tensorflow.org/api_docs/python/tf/fake_quant_with_min_max_vars
  //
  // For more context about what that is useful for, see the big comment in
  // graph_transformations/ensure_uint8_weights_safe_for_fast_int8_kernels.cc
  //
  // The narrow_range flag applies only to quantized arrays, and changes
  // their quantization in the following way when it is set to 'true':
  // 1. The computation of {zero_point, scale} from {min, max} needs to be
  //    amended so that the real min value will get quantized to
  //    (min_quantized_value + 1) instead of just (min_quantized_value).
  //    E.g. for uint8 quantization, the real min value should get quantized to
  //    the uint8 value 1, not 0.
  // 2. Quantized values should get clamped to the interval
  //    [min_quantized_value + 1, max_value]. Equivalently, the
  //    min_quantized_value should get nudged to (min_quantized_value + 1).
  // The reason why 1. does not imply 2. is that real values may not belong to
  // the stated [min, max] interval. Concretely, weights recorded at the last
  // learning step may not fall in the [min, max] interval recorded over
  // previous learning steps, as the values evolve across learning steps.
  //
  // Rationale why this is directly a field on Array:
  // - This can't be just a field on FakeQuantOperator, because
  //   FakeQuantOperators are gone (DropFakeQuant) before we get to using that
  //   information (Quantize). We need a place to store that bit in the interim.
  // - This can't be in QuantizationParams because we need to record this
  //   ahead of quantization, and QuantizationParams are only created during
  //   quantization.
  // - This could be in MinMax, but that would be an abuse of what MinMax is
  //   about, and would break existing code that assumes that a MinMax is just
  //   a min and a max. Unlike MinMax which is agnostic as to the quantized
  //   data type, narrow_range refers to values in the quantized data type.
  bool narrow_range = false;

 private:
  std::unique_ptr<Shape> array_shape;
};

// Our Model struct, represents an entire model (our "top-level" struct).
// Owns everything.
class Model {
 public:
  using ArrayMap = std::unordered_map<std::string, std::unique_ptr<Array>>;

  bool HasArray(const std::string& name) const {
   std::vector<std::string> mht_159_v;
   mht_159_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_159(mht_159_v, 3046, "", "./tensorflow/lite/toco/model.h", "HasArray");

    return arrays.count(name) > 0;
  }
  Array& GetArray(const std::string& name) const {
   std::vector<std::string> mht_160_v;
   mht_160_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_160(mht_160_v, 3053, "", "./tensorflow/lite/toco/model.h", "GetArray");

    DCHECK(HasArray(name)) << "Array not found: " << name;
    return *arrays.at(name);
  }
  Array& GetOrCreateArray(const std::string& name) {
   std::vector<std::string> mht_161_v;
   mht_161_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_161(mht_161_v, 3061, "", "./tensorflow/lite/toco/model.h", "GetOrCreateArray");

    // Make sure name is not used by an optional array
    DCHECK(!optional_arrays.count(name));
    if (!HasArray(name)) {
      Array* ptr = new Array;
      arrays[name] = std::unique_ptr<Array>(ptr);
    }
    Array& result = GetArray(name);
    return result;
  }
  void CreateOptionalArray(const std::string& name) {
   std::vector<std::string> mht_162_v;
   mht_162_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_162(mht_162_v, 3075, "", "./tensorflow/lite/toco/model.h", "CreateOptionalArray");

    DCHECK(!arrays.count(name) && !optional_arrays.count(name));
    optional_arrays.insert(name);
  }
  bool IsOptionalArray(const std::string& name) const {
   std::vector<std::string> mht_163_v;
   mht_163_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_163(mht_163_v, 3083, "", "./tensorflow/lite/toco/model.h", "IsOptionalArray");

    return optional_arrays.count(name);
  }

  // Note that this invalidates all array iterators.
  void EraseArray(const std::string& name) {
   std::vector<std::string> mht_164_v;
   mht_164_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_164(mht_164_v, 3092, "", "./tensorflow/lite/toco/model.h", "EraseArray");
 arrays.erase(name); }
  void EraseArrays(std::function<bool(const std::string&)> discardable) {
   std::vector<std::string> mht_165_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_165(mht_165_v, 3096, "", "./tensorflow/lite/toco/model.h", "EraseArrays");

    for (auto it = arrays.begin(); it != arrays.end();) {
      if (discardable(it->first)) {
        it = arrays.erase(it);
      } else {
        ++it;
      }
    }
  }
  const ArrayMap& GetArrayMap() const {
   std::vector<std::string> mht_166_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_166(mht_166_v, 3108, "", "./tensorflow/lite/toco/model.h", "GetArrayMap");
 return arrays; }
  ArrayMap& GetMutableArrayMap() {
   std::vector<std::string> mht_167_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_167(mht_167_v, 3112, "", "./tensorflow/lite/toco/model.h", "GetMutableArrayMap");
 return arrays; }

  int64_t ArithmeticOpsCount() const {
   std::vector<std::string> mht_168_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_168(mht_168_v, 3117, "", "./tensorflow/lite/toco/model.h", "ArithmeticOpsCount");
 return ops_count; }

  void AddInvalidInputArray(std::string invalid_input_array) {
   std::vector<std::string> mht_169_v;
   mht_169_v.push_back("invalid_input_array: \"" + invalid_input_array + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_169(mht_169_v, 3123, "", "./tensorflow/lite/toco/model.h", "AddInvalidInputArray");

    invalid_input_arrays_.insert(invalid_input_array);
  }

  const std::unordered_set<std::string>& GetInvalidInputArrays() const {
   std::vector<std::string> mht_170_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodelDTh mht_170(mht_170_v, 3130, "", "./tensorflow/lite/toco/model.h", "GetInvalidInputArrays");

    return invalid_input_arrays_;
  }

  // Optional arrays are used for optional tensors,
  // these tensors do not have data, but with reserved names as op inputs.
  std::set<std::string> optional_arrays;

  // The list of operators. Notice how it's a list of unique_ptr's, implying
  // that the Model is what owns Operator's and keeps them alive.
  std::vector<std::unique_ptr<Operator>> operators;

  // Generic flags, a place where we combine information passed to us via
  // command-line parameters (e.g. --input_width=N) with information that
  // we may or may not find in the input model file.
  ModelFlags flags;
  // For code-generation only: required size of the transient_data buffer
  std::size_t transient_data_size = 0;
  // For code-generation only: required alignment of the transient_data buffer
  std::size_t transient_data_alignment = 0;
  // Arithmetic operations performed in the model.
  int64_t ops_count = 0;

 private:
  // The associative array mapping names to Array's.
  // Notice how it's a container of unique_ptr's, implying
  // that the Model is what owns Array's and keeps them alive.
  // The Operator's refer to these Array's by their name strings, not by their
  // addresses. See Operator::inputs, Operator::outputs.
  std::unordered_map<std::string, std::unique_ptr<Array>> arrays;

  // Invalid input arrays.
  std::unordered_set<std::string> invalid_input_arrays_;
};

// OperatorSignature contains the information required to making versioning
// decisions.
struct OperatorSignature {
  // The operator.
  const Operator* op;

  // The model in which the operator resides.
  const Model* model;
};
}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_MODEL_H_
