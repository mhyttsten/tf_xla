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
class MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc() {
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

#include "tensorflow/stream_executor/dnn.h"

#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"

namespace stream_executor {
namespace dnn {

namespace {

bool ProtoMapIsSubset(const google::protobuf::Map<int64_t, int64_t>& x,
                      const google::protobuf::Map<int64_t, int64_t>& y) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_0(mht_0_v, 199, "", "./tensorflow/stream_executor/dnn.cc", "ProtoMapIsSubset");

  for (const auto& ypair : y) {
    const auto it = x.find(ypair.first);
    if (it == x.end() || it->second != ypair.second) return false;
  }
  return true;
}

bool ProtoMapsEqual(const google::protobuf::Map<int64_t, int64_t>& x,
                    const google::protobuf::Map<int64_t, int64_t>& y) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_1(mht_1_v, 211, "", "./tensorflow/stream_executor/dnn.cc", "ProtoMapsEqual");

  return ProtoMapIsSubset(x, y) && ProtoMapIsSubset(y, x);
}

}  // namespace

constexpr DataType ToDataType<float>::value;
constexpr DataType ToDataType<double>::value;
constexpr DataType ToDataType<Eigen::half>::value;
constexpr DataType ToDataType<Eigen::bfloat16>::value;
constexpr DataType ToDataType<int8>::value;
constexpr DataType ToDataType<int32>::value;
constexpr DataType ToDataType<std::complex<float>>::value;
constexpr DataType ToDataType<std::complex<double>>::value;

AlgorithmDesc::AlgorithmDesc(
    int64_t engine_id,
    const std::vector<std::pair<int64_t, int64_t>>& tuning_knobs,
    absl::optional<uint64_t> workspace_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_2(mht_2_v, 232, "", "./tensorflow/stream_executor/dnn.cc", "AlgorithmDesc::AlgorithmDesc");

  proto_.set_is_cudnn_frontend(true);
  proto_.set_algo_id(engine_id);
  if (workspace_size) {
    proto_.mutable_workspace_size()->set_value(*workspace_size);
  }
  for (const auto& pair : tuning_knobs) {
    (*proto_.mutable_tuning_knobs())[pair.first] = pair.second;
  }
}

uint64_t AlgorithmDesc::hash() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_3(mht_3_v, 246, "", "./tensorflow/stream_executor/dnn.cc", "AlgorithmDesc::hash");

  return tensorflow::DeterministicProtoHash64(proto_);
}

bool AlgorithmDesc::operator==(const AlgorithmDesc& other) const {
  if (is_cudnn_frontend()) {
    return other.is_cudnn_frontend() && algo_id() == other.algo_id() &&
           ProtoMapsEqual(proto_.tuning_knobs(), other.proto_.tuning_knobs());
  }
  return !other.is_cudnn_frontend() && algo_id() == other.algo_id() &&
         tensor_ops_enabled() == other.tensor_ops_enabled();
}

std::string AlgorithmDesc::ToString() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_4(mht_4_v, 262, "", "./tensorflow/stream_executor/dnn.cc", "AlgorithmDesc::ToString");

  if (is_cudnn_frontend()) {
    // Format similarly to cudnn_frontend::ExecutionPlan::getTag(), e.g.
    // "eng2{k1=2,k3=4}".
    return absl::StrFormat(
        "eng%d{%s}", proto_.algo_id(),
        absl::StrJoin(
            proto_.tuning_knobs(), ",",
            [](std::string* out,
               const google::protobuf::Map<int64_t, int64_t>::value_type& pair) {
              absl::StrAppendFormat(out, "k%d=%d", pair.first, pair.second);
            }));
  }
  if (tensor_ops_enabled()) {
    return absl::StrCat(algo_id(), "#TC");
  } else {
    return absl::StrCat(algo_id());
  }
}

std::vector<std::pair<int64_t, int64_t>> AlgorithmDesc::TuningKnobs() const {
  std::vector<std::pair<int64_t, int64_t>> result;
  result.reserve(proto_.tuning_knobs().size());
  for (const auto& pair : proto_.tuning_knobs()) {
    result.emplace_back(pair.first, pair.second);
  }
  return result;
}

bool DnnSupport::GetConvolveAlgorithms(
    CudaComputeCapability cuda_compute_capability,
    std::vector<AlgorithmDesc>* out_algorithms) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_5(mht_5_v, 296, "", "./tensorflow/stream_executor/dnn.cc", "DnnSupport::GetConvolveAlgorithms");

  return false;
}

port::Status DnnSupport::GetConvolveRunners(
    bool /* use_cudnn_frontend */, dnn::ConvolutionKind /*kind*/,
    dnn::DataType /*input_type*/, dnn::DataType /*output_type*/,
    Stream* /*stream*/, const dnn::BatchDescriptor& /*input_descriptor*/,
    DeviceMemoryBase /*input_data*/,
    const dnn::FilterDescriptor& /*filter_descriptor*/,
    DeviceMemoryBase /*filter_data*/,
    const dnn::BatchDescriptor& /*output_descriptor*/,
    DeviceMemoryBase /*output_data*/,
    const dnn::ConvolutionDescriptor& /*convolution_descriptor*/,
    bool /*use_fallback*/, ScratchAllocator* /*scratch_allocator*/,
    std::vector<std::unique_ptr<const dnn::ConvRunner>>* /*exec_plans*/) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_6(mht_6_v, 314, "", "./tensorflow/stream_executor/dnn.cc", "DnnSupport::GetConvolveRunners");

  return port::UnimplementedError("GetConvolveRunners not implemented.");
}

port::StatusOr<std::unique_ptr<const dnn::ConvRunner>>
DnnSupport::ConvolveRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::ConvolutionKind kind, dnn::DataType element_type,
    dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_7(mht_7_v, 328, "", "./tensorflow/stream_executor/dnn.cc", "DnnSupport::ConvolveRunnerFromDesc");

  return port::UnimplementedError("ConvolveRunnerFromDesc not implemented.");
}

port::Status DnnSupport::GetFusedConvolveRunners(
    bool use_cudnn_frontend, dnn::ConvolutionKind kind,
    dnn::DataType element_type, dnn::DataType bias_type,
    dnn::DataType output_type, double conv_input_scale, double side_input_scale,
    Stream* stream, const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& bias_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor, bool use_fallback,
    dnn::ActivationMode activation_mode,
    std::vector<std::unique_ptr<const dnn::FusedConvRunner>>* out_exec_plans) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_8(mht_8_v, 345, "", "./tensorflow/stream_executor/dnn.cc", "DnnSupport::GetFusedConvolveRunners");

  return port::UnimplementedError("GetFusedConvolveRunners not implemented.");
}

port::StatusOr<std::unique_ptr<const dnn::FusedConvRunner>>
DnnSupport::FusedConvolveRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::ConvolutionKind kind, dnn::DataType element_type,
    dnn::DataType bias_type, dnn::DataType output_type, double conv_scale,
    double side_input_scale, const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& bias_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    dnn::ActivationMode activation_mode) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_9(mht_9_v, 362, "", "./tensorflow/stream_executor/dnn.cc", "DnnSupport::FusedConvolveRunnerFromDesc");

  return port::UnimplementedError(
      "FusedConvolveRunnerFromDesc not implemented.");
}

bool DnnSupport::GetMIOpenConvolveAlgorithms(
    dnn::ConvolutionKind /*kind*/, dnn::DataType /*element_type*/,
    Stream* /*stream*/, const dnn::BatchDescriptor& /*input_descriptor*/,
    DeviceMemoryBase input_data,
    const dnn::FilterDescriptor& /*filter_descriptor*/,
    DeviceMemoryBase filter_data,
    const dnn::BatchDescriptor& /*output_descriptor*/,
    DeviceMemoryBase output_data,
    const dnn::ConvolutionDescriptor& /*convolution_descriptor*/,
    ScratchAllocator* scratch_allocator,
    std::vector<ProfileResult>* /*out_algorithms*/) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_10(mht_10_v, 380, "", "./tensorflow/stream_executor/dnn.cc", "DnnSupport::GetMIOpenConvolveAlgorithms");

  return false;
}

bool DnnSupport::GetRnnAlgorithms(std::vector<AlgorithmDesc>* out_algorithms) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_11(mht_11_v, 387, "", "./tensorflow/stream_executor/dnn.cc", "DnnSupport::GetRnnAlgorithms");

  return false;
}

bool DnnSupport::GetConvolveBackwardDataAlgorithms(
    CudaComputeCapability cuda_compute_capability,
    std::vector<AlgorithmDesc>* out_algorithms) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_12(mht_12_v, 396, "", "./tensorflow/stream_executor/dnn.cc", "DnnSupport::GetConvolveBackwardDataAlgorithms");

  return false;
}

bool DnnSupport::GetConvolveBackwardFilterAlgorithms(
    CudaComputeCapability cuda_compute_capability,
    std::vector<AlgorithmDesc>* out_algorithms) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_13(mht_13_v, 405, "", "./tensorflow/stream_executor/dnn.cc", "DnnSupport::GetConvolveBackwardFilterAlgorithms");

  return false;
}

std::string QuantizedActivationModeString(QuantizedActivationMode mode) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_14(mht_14_v, 412, "", "./tensorflow/stream_executor/dnn.cc", "QuantizedActivationModeString");

  switch (mode) {
    case dnn::QuantizedActivationMode::k8Bit:
      return "uint8";
    case dnn::QuantizedActivationMode::k16Bit:
      return "uint16";
    case dnn::QuantizedActivationMode::k32Bit:
      return "int32";
    default:
      return absl::StrCat("unknown: ", static_cast<int32_t>(mode));
  }
}

std::string ActivationModeString(ActivationMode mode) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_15(mht_15_v, 428, "", "./tensorflow/stream_executor/dnn.cc", "ActivationModeString");

  switch (mode) {
    case ActivationMode::kNone:
      return "none";
    case ActivationMode::kSigmoid:
      return "sigmoid";
    case ActivationMode::kRelu:
      return "relu";
    case ActivationMode::kRelu6:
      return "relu6";
    case ActivationMode::kReluX:
      return "reluX";
    case ActivationMode::kTanh:
      return "tanh";
    case ActivationMode::kBandPass:
      return "bandpass";
    default:
      return absl::StrCat("unknown: ", static_cast<int32_t>(mode));
  }
}

std::string ElementwiseOperationString(ElementwiseOperation op) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_16(mht_16_v, 452, "", "./tensorflow/stream_executor/dnn.cc", "ElementwiseOperationString");

  switch (op) {
    case ElementwiseOperation::kAdd:
      return "add";
    case ElementwiseOperation::kMultiply:
      return "multiply";
    default:
      return absl::StrCat("unknown: ", static_cast<int32_t>(op));
  }
}

std::string DataLayoutString(DataLayout layout) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_17(mht_17_v, 466, "", "./tensorflow/stream_executor/dnn.cc", "DataLayoutString");

  switch (layout) {
    case DataLayout::kYXDepthBatch:
      return "YXDepthBatch";
    case DataLayout::kYXBatchDepth:
      return "YXBatchDepth";
    case DataLayout::kBatchYXDepth:
      return "BatchYXDepth";
    case DataLayout::kBatchDepthYX:
      return "BatchDepthYX";
    case DataLayout::kBatchDepthYX4:
      return "BatchDepthYX4";
    case DataLayout::kBatchDepthYX32:
      return "BatchDepthYX32";
    default:
      return absl::StrCat("unknown: ", static_cast<int32_t>(layout));
  }
}

std::string FilterLayoutString(FilterLayout layout) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_18(mht_18_v, 488, "", "./tensorflow/stream_executor/dnn.cc", "FilterLayoutString");

  switch (layout) {
    case FilterLayout::kOutputInputYX:
      return "OutputInputYX";
    case FilterLayout::kOutputYXInput:
      return "OutputYXInput";
    case FilterLayout::kOutputInputYX4:
      return "OutputInputYX4";
    case FilterLayout::kOutputInputYX32:
      return "OutputInputYX32";
    case FilterLayout::kInputYXOutput:
      return "InputYXOutput";
    case FilterLayout::kYXInputOutput:
      return "YXInputOutput";
    default:
      return absl::StrCat("unknown: ", static_cast<int32_t>(layout));
  }
}

std::string PadAlignmentString(PadAlignment alignment) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_19(mht_19_v, 510, "", "./tensorflow/stream_executor/dnn.cc", "PadAlignmentString");

  switch (alignment) {
    case PadAlignment::kDefault:
      return "default";
    case PadAlignment::kCudnnPadding:
      return "cuDNN padding";
    case PadAlignment::kTensorFlowPadding:
      return "TensorFlow padding";
    default:
      return absl::StrCat("unknown: ", static_cast<int32_t>(alignment));
  }
}

std::ostream& operator<<(std::ostream& str, dnn::PadAlignment alignment) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_20(mht_20_v, 526, "", "./tensorflow/stream_executor/dnn.cc", "operator<<");

  return str << PadAlignmentString(alignment);
}

std::string ShortPoolingModeString(PoolingMode mode) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_21(mht_21_v, 533, "", "./tensorflow/stream_executor/dnn.cc", "ShortPoolingModeString");

  switch (mode) {
    case PoolingMode::kMaximum:
      return "Max";
    case PoolingMode::kAverage:
      return "Avg";
    default:
      return absl::StrCat("unknown: ", static_cast<int32_t>(mode));
  }
}

struct ConvDimIndices {
  union {
    struct {
      int depth_idx;
      int batch_idx;
      int spatial_idx;
    } data;
    struct {
      int output_idx;
      int input_idx;
      int spatial_idx;
    } filter;
  };
};

ConvDimIndices GetDimIndices(const DataLayout& layout, const int data_dims) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_22(mht_22_v, 562, "", "./tensorflow/stream_executor/dnn.cc", "GetDimIndices");

  ConvDimIndices dim_indices;
  switch (layout) {
    case DataLayout::kYXBatchDepth:
      dim_indices.data.depth_idx = data_dims - 1;
      dim_indices.data.batch_idx = data_dims - 2;
      dim_indices.data.spatial_idx = 0;
      break;

    case DataLayout::kYXDepthBatch:
      dim_indices.data.depth_idx = data_dims - 2;
      dim_indices.data.batch_idx = data_dims - 1;
      dim_indices.data.spatial_idx = 0;
      break;

    case DataLayout::kBatchYXDepth:
      dim_indices.data.depth_idx = data_dims - 1;
      dim_indices.data.batch_idx = 0;
      dim_indices.data.spatial_idx = 1;
      break;

    case DataLayout::kBatchDepthYX:
    case DataLayout::kBatchDepthYX4:
    case DataLayout::kBatchDepthYX32:
      dim_indices.data.depth_idx = 1;
      dim_indices.data.batch_idx = 0;
      dim_indices.data.spatial_idx = 2;
      break;

    default:
      LOG(FATAL) << "Unknown layout " << layout;
  }

  return dim_indices;
}

ConvDimIndices GetDimIndices(const FilterLayout& layout, const int data_dims) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_23(mht_23_v, 601, "", "./tensorflow/stream_executor/dnn.cc", "GetDimIndices");

  ConvDimIndices dim_indices;
  switch (layout) {
    case FilterLayout::kOutputInputYX:
    case FilterLayout::kOutputInputYX4:
    case FilterLayout::kOutputInputYX32:
      dim_indices.filter.input_idx = 1;
      dim_indices.filter.output_idx = 0;
      dim_indices.filter.spatial_idx = 2;
      break;

    case FilterLayout::kOutputYXInput:
      dim_indices.filter.input_idx = data_dims - 1;
      dim_indices.filter.output_idx = 0;
      dim_indices.filter.spatial_idx = 1;
      break;

    case FilterLayout::kInputYXOutput:
      dim_indices.filter.input_idx = 0;
      dim_indices.filter.output_idx = data_dims - 1;
      dim_indices.filter.spatial_idx = 1;
      break;

    case FilterLayout::kYXInputOutput:
      dim_indices.filter.input_idx = data_dims - 2;
      dim_indices.filter.output_idx = data_dims - 1;
      dim_indices.filter.spatial_idx = 0;
      break;

    default:
      LOG(FATAL) << "Unknown layout " << layout;
  }

  return dim_indices;
}

std::vector<int64_t> ReorderDims(const std::vector<int64_t>& input,
                                 const DataLayout& from, const DataLayout& to) {
  if (from == to) return input;

  ConvDimIndices from_indices = GetDimIndices(from, input.size());
  ConvDimIndices to_indices = GetDimIndices(to, input.size());

  std::vector<int64_t> reordered(input.size());
  reordered[to_indices.data.batch_idx] = input[from_indices.data.batch_idx];
  reordered[to_indices.data.depth_idx] = input[from_indices.data.depth_idx];

  int spatial_idx_from = from_indices.data.spatial_idx;
  int spatial_idx_to = to_indices.data.spatial_idx;
  for (size_t i = 0; i < input.size() - 2;
       i++, spatial_idx_from++, spatial_idx_to++) {
    reordered[spatial_idx_to] = input[spatial_idx_from];
  }

  return reordered;
}

std::vector<int64_t> ReorderDims(const std::vector<int64_t>& input,
                                 const FilterLayout& from,
                                 const FilterLayout& to) {
  if (from == to) return input;

  ConvDimIndices from_indices = GetDimIndices(from, input.size());
  ConvDimIndices to_indices = GetDimIndices(to, input.size());

  std::vector<int64_t> reordered(input.size());
  reordered[to_indices.filter.output_idx] =
      input[from_indices.filter.output_idx];
  reordered[to_indices.filter.input_idx] = input[from_indices.filter.input_idx];

  int spatial_idx_from = from_indices.filter.spatial_idx;
  int spatial_idx_to = to_indices.filter.spatial_idx;
  for (size_t i = 0; i < input.size() - 2;
       i++, spatial_idx_from++, spatial_idx_to++) {
    reordered[spatial_idx_to] = input[spatial_idx_from];
  }

  return reordered;
}

// -- AlgorithmConfig

std::string AlgorithmConfig::ToString() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_24(mht_24_v, 686, "", "./tensorflow/stream_executor/dnn.cc", "AlgorithmConfig::ToString");

  std::string algo = "none";
  if (algorithm().has_value()) {
    algo = algorithm()->ToString();
  }
  std::string algo_no_scratch = "none";
  if (algorithm_no_scratch().has_value()) {
    algo_no_scratch = algorithm_no_scratch()->ToString();
  }
  return absl::StrCat(algo, ", ", algo_no_scratch);
}

// -- BatchDescriptor

BatchDescriptor::BatchDescriptor(int ndims)
    : value_max_(0.0),
      value_min_(0.0),
      quantized_activation_mode_(QuantizedActivationMode::k8Bit) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_25(mht_25_v, 706, "", "./tensorflow/stream_executor/dnn.cc", "BatchDescriptor::BatchDescriptor");

  tensor_.mutable_dimensions()->Resize(ndims + 2, 0);
  set_layout(DataLayout::kYXDepthBatch);
}

BatchDescriptor::BatchDescriptor() : BatchDescriptor(/*ndims=*/2) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_26(mht_26_v, 714, "", "./tensorflow/stream_executor/dnn.cc", "BatchDescriptor::BatchDescriptor");
}

std::vector<int64_t> BatchDescriptor::full_dims(
    const DataLayout& layout) const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_27(mht_27_v, 720, "", "./tensorflow/stream_executor/dnn.cc", "BatchDescriptor::full_dims");

  std::vector<int64_t> bdyx_dims(ndims() + 2);
  bdyx_dims[0] = count();
  bdyx_dims[1] = feature_map_count();
  std::copy(spatial_size().begin(), spatial_size().end(),
            bdyx_dims.begin() + 2);
  return ReorderDims(bdyx_dims, DataLayout::kBatchDepthYX, layout);
}

std::vector<int64_t> BatchDescriptor::full_strides(
    const DataLayout& layout) const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_28(mht_28_v, 733, "", "./tensorflow/stream_executor/dnn.cc", "BatchDescriptor::full_strides");

  std::vector<int64_t> phys_dims = full_dims(this->layout());
  std::vector<int64_t> phys_strides(phys_dims.size());
  phys_strides[ndims() + 1] = 1;
  for (int i = ndims(); i >= 0; i--) {
    phys_strides[i] = phys_strides[i + 1] * phys_dims[i + 1];
  }
  return ReorderDims(phys_strides, this->layout(), layout);
}

std::vector<int64_t> BatchDescriptor::vectorized_dims(const DataLayout& layout,
                                                      int vector_size,
                                                      int vector_dim) const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_29(mht_29_v, 748, "", "./tensorflow/stream_executor/dnn.cc", "BatchDescriptor::vectorized_dims");

  std::vector<int64_t> bdyx_dims = full_dims(dnn::DataLayout::kBatchDepthYX);
  if (vector_dim != -1) {
    bdyx_dims[vector_dim] /= vector_size;
  }
  return dnn::ReorderDims(bdyx_dims, dnn::DataLayout::kBatchDepthYX, layout);
}

std::vector<int64_t> BatchDescriptor::vectorized_strides(
    const DataLayout& layout, int vector_size, int vector_dim) const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_30(mht_30_v, 760, "", "./tensorflow/stream_executor/dnn.cc", "BatchDescriptor::vectorized_strides");

  std::vector<int64_t> phys_dims =
      vectorized_dims(this->layout(), vector_size, vector_dim);
  std::vector<int64_t> phys_strides(phys_dims.size());
  phys_strides[phys_dims.size() - 1] = 1;
  for (int i = phys_dims.size() - 2; i >= 0; i--) {
    phys_strides[i] = phys_strides[i + 1] * phys_dims[i + 1];
  }
  return ReorderDims(phys_strides, this->layout(), layout);
}

void BatchDescriptor::CloneFrom(const BatchDescriptor& other) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_31(mht_31_v, 774, "", "./tensorflow/stream_executor/dnn.cc", "BatchDescriptor::CloneFrom");

  tensor_ = other.tensor_;
  value_max_ = other.value_max_;
  value_min_ = other.value_min_;
  quantized_activation_mode_ = other.quantized_activation_mode_;
}

std::string BatchDescriptor::ToString() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_32(mht_32_v, 784, "", "./tensorflow/stream_executor/dnn.cc", "BatchDescriptor::ToString");

  std::string spatial;
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&spatial, "%d ", spatial_size()[i]);
  }
  return absl::StrFormat(
      "{count: %d feature_map_count: %d spatial: %s "
      "value_min: %f value_max: %f layout: %s}",
      count(), feature_map_count(), spatial, value_min_, value_max_,
      DataLayoutString(layout()));
}

std::string BatchDescriptor::ToShortString() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_33(mht_33_v, 799, "", "./tensorflow/stream_executor/dnn.cc", "BatchDescriptor::ToShortString");

  // All the constituent strings are less than 15 characters, so the
  // small string optimization ensures that there will be at most one
  // heap memory allocation.
  std::string depth = absl::StrCat("d", feature_map_count());
  std::string batch = absl::StrCat("b", count());

  std::string spatial = "s";
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&spatial, "%d ", spatial_size()[i]);
  }

  std::string suffix;
  if (value_min() != value_max()) {
    absl::StrAppend(&suffix, "[", value_min(), ";", value_max(), "]");
  }
  if (quantized_activation_mode() == QuantizedActivationMode::k16Bit) {
    suffix += "_16bit";
  }

  switch (layout()) {
    case DataLayout::kYXDepthBatch:
      return absl::StrCat(spatial, depth, batch, suffix);
    case DataLayout::kYXBatchDepth:
      return absl::StrCat(spatial, batch, depth, suffix);
    case DataLayout::kBatchYXDepth:
      return absl::StrCat(batch, spatial, depth, suffix);
    case DataLayout::kBatchDepthYX:
      return absl::StrCat(batch, depth, spatial, suffix);
    case DataLayout::kBatchDepthYX4:
    case DataLayout::kBatchDepthYX32:
      return absl::StrCat(batch, depth, spatial, suffix, "(VECT_C)");
    default:
      LOG(FATAL) << "Unknown layout " << static_cast<int32>(layout());
      return "";  // Avoid return warning (unreachable)
  }
}

int64_t BatchDescriptor::NodesPerFeatureMap() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_34(mht_34_v, 840, "", "./tensorflow/stream_executor/dnn.cc", "BatchDescriptor::NodesPerFeatureMap");

  int64_t ret = 1;
  for (int i = 0; i < ndims(); i++) {
    ret *= spatial_size()[i];
  }
  return ret;
}

int64_t BatchDescriptor::NodesAcrossFeatureMaps() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_35(mht_35_v, 851, "", "./tensorflow/stream_executor/dnn.cc", "BatchDescriptor::NodesAcrossFeatureMaps");

  return NodesPerFeatureMap() * feature_map_count();
}

int64_t BatchDescriptor::ElementCount() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_36(mht_36_v, 858, "", "./tensorflow/stream_executor/dnn.cc", "BatchDescriptor::ElementCount");

  return count() * feature_map_count() * NodesPerFeatureMap();
}

int64_t BatchDescriptor::FullyConnectedWeightCount(
    const BatchDescriptor& input, const BatchDescriptor& output) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_37(mht_37_v, 866, "", "./tensorflow/stream_executor/dnn.cc", "BatchDescriptor::FullyConnectedWeightCount");

  return input.NodesAcrossFeatureMaps() * output.NodesAcrossFeatureMaps();
}

int64_t BatchDescriptor::FullyConnectedBiasCount(
    const BatchDescriptor& output) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_38(mht_38_v, 874, "", "./tensorflow/stream_executor/dnn.cc", "BatchDescriptor::FullyConnectedBiasCount");

  return output.NodesAcrossFeatureMaps();
}

BatchDescriptor BatchDescriptor::DepthConcatenateOutputDescriptor(
    port::ArraySlice<dnn::BatchDescriptor> inputs) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_39(mht_39_v, 882, "", "./tensorflow/stream_executor/dnn.cc", "BatchDescriptor::DepthConcatenateOutputDescriptor");

  if (inputs.empty()) {
    return BatchDescriptor();
  }
  int feature_map_count = 0;
  for (const auto& dimensions : inputs) {
    feature_map_count += dimensions.feature_map_count();
  }
  BatchDescriptor output = inputs[0];
  output.set_feature_map_count(feature_map_count);
  return output;
}

TensorDescriptorProto BatchDescriptor::ToProto(DataType data_type) const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_40(mht_40_v, 898, "", "./tensorflow/stream_executor/dnn.cc", "BatchDescriptor::ToProto");

  CHECK_EQ(0.0, value_max_);
  CHECK_EQ(0.0, value_min_);
  CHECK(quantized_activation_mode_ == QuantizedActivationMode::k8Bit);

  TensorDescriptorProto ret = tensor_;
  ret.set_data_type(data_type);
  return ret;
}

// -- FilterDescriptor

FilterDescriptor::FilterDescriptor(int ndims) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_41(mht_41_v, 913, "", "./tensorflow/stream_executor/dnn.cc", "FilterDescriptor::FilterDescriptor");

  tensor_.mutable_dimensions()->Resize(ndims + 2, 0);
  set_layout(FilterLayout::kOutputInputYX);
}

FilterDescriptor::FilterDescriptor() : FilterDescriptor(/*ndims=*/2) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_42(mht_42_v, 921, "", "./tensorflow/stream_executor/dnn.cc", "FilterDescriptor::FilterDescriptor");
}

FilterDescriptor::~FilterDescriptor() {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_43(mht_43_v, 926, "", "./tensorflow/stream_executor/dnn.cc", "FilterDescriptor::~FilterDescriptor");
}

void FilterDescriptor::CloneFrom(const FilterDescriptor& other) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_44(mht_44_v, 931, "", "./tensorflow/stream_executor/dnn.cc", "FilterDescriptor::CloneFrom");

  tensor_ = other.tensor_;
}

std::string FilterDescriptor::ToString() const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_45(mht_45_v, 938, "", "./tensorflow/stream_executor/dnn.cc", "FilterDescriptor::ToString");

  std::string desc = absl::StrFormat(
      "{output_feature_map_count: %d input_feature_map_count: %d "
      "layout: %s shape: ",
      output_feature_map_count(), input_feature_map_count(),
      FilterLayoutString(layout()));
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&desc, "%d ", input_filter_dims()[i]);
  }
  absl::StrAppend(&desc, "}");

  return desc;
}

std::string FilterDescriptor::ToShortString() const {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_46(mht_46_v, 955, "", "./tensorflow/stream_executor/dnn.cc", "FilterDescriptor::ToShortString");

  // All the constituent strings are less than 15 characters, so the
  // small string optimization ensures that there will be at most one
  // heap memory allocation.
  std::string od = absl::StrCat("od", output_feature_map_count());
  std::string id = absl::StrCat("id", input_feature_map_count());

  std::string spatial = "s";
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&spatial, "%d ", input_filter_dims()[i]);
  }

  switch (layout()) {
    case FilterLayout::kOutputInputYX:
      return absl::StrCat(od, id, spatial);
    case FilterLayout::kOutputYXInput:
      return absl::StrCat(od, spatial, id);
    case FilterLayout::kOutputInputYX4:
    case FilterLayout::kOutputInputYX32:
      return absl::StrCat(od, id, spatial, "(VECT_C)");
    case FilterLayout::kInputYXOutput:
      return absl::StrCat(id, spatial, od);
    case FilterLayout::kYXInputOutput:
      return absl::StrCat(spatial, id, od);
    default:
      LOG(FATAL) << "Unknown layout " << static_cast<int32>(layout());
      return "";  // Avoid return warning (unreachable)
  }
}

int64_t FilterDescriptor::ComputeWeightCount() const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_47(mht_47_v, 988, "", "./tensorflow/stream_executor/dnn.cc", "FilterDescriptor::ComputeWeightCount");

  int64_t ret = output_feature_map_count() * input_feature_map_count();
  for (int i = 0; i < ndims(); i++) {
    ret *= input_filter_dims()[i];
  }
  return ret;
}

std::vector<int64_t> FilterDescriptor::full_dims(
    const FilterLayout& layout) const {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_48(mht_48_v, 1000, "", "./tensorflow/stream_executor/dnn.cc", "FilterDescriptor::full_dims");

  std::vector<int64_t> oiyx_dims(ndims() + 2);
  oiyx_dims[0] = output_feature_map_count();
  oiyx_dims[1] = input_feature_map_count();
  std::copy(input_filter_dims().begin(), input_filter_dims().end(),
            oiyx_dims.begin() + 2);
  return ReorderDims(oiyx_dims, FilterLayout::kOutputInputYX, layout);
}

std::vector<int64_t> FilterDescriptor::full_strides(
    const FilterLayout& layout) const {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_49(mht_49_v, 1013, "", "./tensorflow/stream_executor/dnn.cc", "FilterDescriptor::full_strides");

  std::vector<int64_t> phys_dims = full_dims(this->layout());
  std::vector<int64_t> phys_strides(phys_dims.size());
  phys_strides[ndims() + 1] = 1;
  for (int i = ndims(); i >= 0; i--) {
    phys_strides[i] = phys_strides[i + 1] * phys_dims[i + 1];
  }
  return ReorderDims(phys_strides, this->layout(), layout);
}

std::vector<int64_t> FilterDescriptor::vectorized_dims(
    const FilterLayout& layout, int vector_size, int vector_dim) const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_50(mht_50_v, 1027, "", "./tensorflow/stream_executor/dnn.cc", "FilterDescriptor::vectorized_dims");

  std::vector<int64_t> oiyx_dims = full_dims(dnn::FilterLayout::kOutputInputYX);
  if (vector_dim != -1) {
    oiyx_dims[vector_dim] /= vector_size;
  }
  return ReorderDims(oiyx_dims, FilterLayout::kOutputInputYX, layout);
}

std::vector<int64_t> FilterDescriptor::vectorized_strides(
    const FilterLayout& layout, int vector_size, int vector_dim) const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_51(mht_51_v, 1039, "", "./tensorflow/stream_executor/dnn.cc", "FilterDescriptor::vectorized_strides");

  std::vector<int64_t> phys_dims =
      vectorized_dims(this->layout(), vector_size, vector_dim);
  std::vector<int64_t> phys_strides(phys_dims.size());
  phys_strides[phys_dims.size() - 1] = 1;
  for (int i = phys_dims.size() - 2; i >= 0; i--) {
    phys_strides[i] = phys_strides[i + 1] * phys_dims[i + 1];
  }
  return ReorderDims(phys_strides, this->layout(), layout);
}

TensorDescriptorProto FilterDescriptor::ToProto(DataType data_type) const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_52(mht_52_v, 1053, "", "./tensorflow/stream_executor/dnn.cc", "FilterDescriptor::ToProto");

  TensorDescriptorProto ret = tensor_;
  ret.set_data_type(data_type);
  return ret;
}

// -- ConvolutionDescriptor

ConvolutionDescriptor::ConvolutionDescriptor(int ndims) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_53(mht_53_v, 1064, "", "./tensorflow/stream_executor/dnn.cc", "ConvolutionDescriptor::ConvolutionDescriptor");

  proto_.mutable_paddings()->Resize(ndims, 0);
  proto_.mutable_strides()->Resize(ndims, 1);
  proto_.mutable_dilations()->Resize(ndims, 1);
  proto_.set_group_count(1);
  proto_.set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);
}

ConvolutionDescriptor::ConvolutionDescriptor()
    : ConvolutionDescriptor(/*ndims=*/2) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_54(mht_54_v, 1076, "", "./tensorflow/stream_executor/dnn.cc", "ConvolutionDescriptor::ConvolutionDescriptor");
}

ConvolutionDescriptor::~ConvolutionDescriptor() {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_55(mht_55_v, 1081, "", "./tensorflow/stream_executor/dnn.cc", "ConvolutionDescriptor::~ConvolutionDescriptor");
}

std::string ConvolutionDescriptor::ToString() const {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_56(mht_56_v, 1086, "", "./tensorflow/stream_executor/dnn.cc", "ConvolutionDescriptor::ToString");

  std::string padding;
  std::string strides;
  std::string dilations;
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&padding, "%d ", this->padding()[i]);
    absl::StrAppendFormat(&strides, "%d ", this->strides()[i]);
    absl::StrAppendFormat(&dilations, "%d ", this->dilations()[i]);
  }

  return absl::StrFormat(
      "{zero_padding: %s pad_alignment: %s filter_strides: %s dilation_rates: "
      "%s}",
      padding, PadAlignmentString(pad_alignment()), strides, dilations);
}

std::string ConvolutionDescriptor::ToShortString() const {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_57(mht_57_v, 1105, "", "./tensorflow/stream_executor/dnn.cc", "ConvolutionDescriptor::ToShortString");

  std::string desc;
  for (int i = 0; i < ndims(); i++) {
    if (i > 0) absl::StrAppend(&desc, "_");
    absl::StrAppendFormat(&desc, "p%d:%d", i, padding()[i]);
  }
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&desc, "_s%d:%d", i, strides()[i]);
  }
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&desc, "_d%d:%d", i, dilations()[i]);
  }
  return desc;
}

// -- PoolingDescriptor

PoolingDescriptor::PoolingDescriptor(int ndims)
    : mode_(dnn::PoolingMode::kMaximum),
      ndims_(ndims),
      propagate_nans_(false),
      window_(ndims, 0),
      padding_(ndims, 0),
      strides_(ndims, 1) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_58(mht_58_v, 1131, "", "./tensorflow/stream_executor/dnn.cc", "PoolingDescriptor::PoolingDescriptor");
}

PoolingDescriptor::PoolingDescriptor() : PoolingDescriptor(/*ndims=*/2) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_59(mht_59_v, 1136, "", "./tensorflow/stream_executor/dnn.cc", "PoolingDescriptor::PoolingDescriptor");
}

void PoolingDescriptor::CloneFrom(const PoolingDescriptor& other) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_60(mht_60_v, 1141, "", "./tensorflow/stream_executor/dnn.cc", "PoolingDescriptor::CloneFrom");

  mode_ = other.mode_;
  ndims_ = other.ndims_;
  window_ = other.window_;
  padding_ = other.padding_;
  strides_ = other.strides_;
  propagate_nans_ = other.propagate_nans_;
}

std::string PoolingDescriptor::ToString() const {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_61(mht_61_v, 1153, "", "./tensorflow/stream_executor/dnn.cc", "PoolingDescriptor::ToString");

  const char* mode_string =
      mode_ == dnn::PoolingMode::kMaximum ? "kMaximum" : "kAverage";

  std::string window, strides, padding;
  for (int i = 0; i < ndims_; i++) {
    absl::StrAppendFormat(&window, "%d ", window_[i]);
    absl::StrAppendFormat(&strides, "%d ", strides_[i]);
    absl::StrAppendFormat(&padding, "%d", padding_[i]);
  }

  const char* propagate_string = propagate_nans_ ? "Yes" : "No";

  return absl::StrFormat(
      "{mode: %s window: %s strides: %s padding: %s propagate NaNs: %s}",
      mode_string, window, strides, padding, propagate_string);
}

std::string PoolingDescriptor::ToShortString() const {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_62(mht_62_v, 1174, "", "./tensorflow/stream_executor/dnn.cc", "PoolingDescriptor::ToShortString");

  std::string window, strides, padding;
  for (int i = 0; i < ndims_; i++) {
    absl::StrAppendFormat(&window, "_w%d:%d", i, window_[i]);
    absl::StrAppendFormat(&strides, "_s%d:%d", i, strides_[i]);
    absl::StrAppendFormat(&padding, "_p%d:%d", i, padding_[i]);
  }
  return absl::StrCat(mode_ == dnn::PoolingMode::kMaximum ? "max" : "avg",
                      window, strides, padding,
                      propagate_nans_ ? "propagate_nans" : "ignore_nans");
}

// -- NormalizeDescriptor

NormalizeDescriptor::NormalizeDescriptor()
    : bias_(0.0),
      range_(0),
      alpha_(0.0),
      beta_(0.0),
      wrap_around_(false),
      segment_size_(0) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_63(mht_63_v, 1197, "", "./tensorflow/stream_executor/dnn.cc", "NormalizeDescriptor::NormalizeDescriptor");
}

void NormalizeDescriptor::CloneFrom(const NormalizeDescriptor& other) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_64(mht_64_v, 1202, "", "./tensorflow/stream_executor/dnn.cc", "NormalizeDescriptor::CloneFrom");

  bias_ = other.bias_;
  range_ = other.range_;
  alpha_ = other.alpha_;
  beta_ = other.beta_;
  wrap_around_ = other.wrap_around_;
  segment_size_ = other.segment_size_;
}

std::string NormalizeDescriptor::ToString() const {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_65(mht_65_v, 1214, "", "./tensorflow/stream_executor/dnn.cc", "NormalizeDescriptor::ToString");

  return absl::StrFormat(
      "{bias: %f range: %d alpha: %f beta: %f wrap_around: %d "
      "segment_size: %d}",
      bias_, range_, alpha_, beta_, wrap_around_, segment_size_);
}

std::string NormalizeDescriptor::ToShortString() const {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_66(mht_66_v, 1224, "", "./tensorflow/stream_executor/dnn.cc", "NormalizeDescriptor::ToShortString");

  return absl::StrCat("bias:", bias_, "_range:", range_, "_alpha:", alpha_,
                      "_beta:", beta_, "_wrap:", wrap_around_,
                      "_size:", segment_size_);
}

bool DnnSupport::IsStatusOk(const port::Status& status, bool report_error) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_67(mht_67_v, 1233, "", "./tensorflow/stream_executor/dnn.cc", "DnnSupport::IsStatusOk");

  if (status.ok()) {
    return true;
  }
  if (report_error) {
    LOG(ERROR) << status.error_message();
  }
  return false;
}

port::Status DnnSupport::DoCtcLoss(
    Stream* stream, dnn::DataType element_type,
    const RnnStateTensorDescriptor& probs_desc,
    const DeviceMemoryBase probs_data, absl::Span<const int> labels_data,
    absl::Span<const int> labels_lengths_data,
    absl::Span<const int> input_lengths_data, DeviceMemoryBase costs_data,
    const RnnStateTensorDescriptor& grads_desc, DeviceMemoryBase grads_data,
    DeviceMemory<uint8> scratch_memory, int ctc_loss_algo_id) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPSstream_executorPSdnnDTcc mht_68(mht_68_v, 1253, "", "./tensorflow/stream_executor/dnn.cc", "DnnSupport::DoCtcLoss");

  return port::UnimplementedError("CtcLoss not implemented");
}

}  // namespace dnn
}  // namespace stream_executor
