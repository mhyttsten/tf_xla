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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.h"

#include "absl/strings/str_format.h"
#include "tensorflow/cc/ops//array_ops.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/weights.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

bool IsQuantizeAndDequantizeOp(const Node* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.cc", "IsQuantizeAndDequantizeOp");

  return absl::c_find(kQuantizationOpNames, node->def().op()) !=
         kQuantizationOpNames.end();
}

namespace {

// Provides quantizing and dequantizing tensor scales for a given dynamic range.
// Borrowed from TF quantization kernel logic.
template <typename T>
QuantizationScales<T, 1> ComputeQuantizationRange(bool signed_input,
                                                  int num_bits,
                                                  bool narrow_range,
                                                  T* min_range, T* max_range) {
  // Calculate the range for the simulated integer quantization:
  // e.g. [-127,127] for signed = true, narrow_range = true, num_bits = 8,
  // or [-128,127] for signed = true, narrow_range = false, num_bits = 8,
  // or [0, 255] for signed = false, num_bits = 8.
  const int64_t min_quantized =
      signed_input ? narrow_range ? -(1ULL << (num_bits - 1)) + 1
                                  : -(1ULL << (num_bits - 1))
                   : 0;
  const int64_t max_quantized =
      signed_input ? (1ULL << (num_bits - 1)) - 1 : (1ULL << num_bits) - 1;
  // Determine the maximum scaling factor that would scale
  // [min_range, max_range] to not exceed [min_quantized, max_quantized],
  // while keeping 0 unchanged.
  const T scale_from_min_side = (min_quantized * *min_range > 0)
                                    ? min_quantized / *min_range
                                    : std::numeric_limits<T>::max();
  const T scale_from_max_side = (max_quantized * *max_range > 0)
                                    ? max_quantized / *max_range
                                    : std::numeric_limits<T>::max();

  QuantizationScales<T, 1> scales;
  // Note: Avoids changing the side of the range that determines scale.
  if (scale_from_min_side < scale_from_max_side) {
    scales.quantize_scale[0] = scale_from_min_side;
    scales.dequantize_scale[0] = *min_range / min_quantized;
    *max_range = max_quantized * scales.dequantize_scale[0];
  } else {
    scales.quantize_scale[0] = scale_from_max_side;
    scales.dequantize_scale[0] = *max_range / max_quantized;
    *min_range = min_quantized * scales.dequantize_scale[0];
  }
  return scales;
}

// Prepares the input for a QDQ node in explicit precision mode, returning a
// ITensor pointer. If the input is weights, we convert it to a ITensor by
// adding a constant layer.
StatusOr<nvinfer1::ITensor*> ExlicitQDQInputToTensor(
    TRTNetworkBuilder* builder, OpConverterParams* params,
    const TRT_TensorOrWeights& input) {
  if (input.is_tensor()) {
    return input.tensor()->trt_tensor();
  }
  if (!IS_TRT_VERSION_GE(8, 0, 0, 0) && input.weights().count() > 1) {
    LOG(WARNING) << absl::StrCat(
        "QDQ per-channel for weights not "
        "implemented, assuming uniform scaling");
  }
  TRT_ShapedWeights trt_weights = input.weights();
  StatusOr<nvinfer1::IConstantLayer*> weights_const =
      builder->WeightsToConstant(trt_weights.GetTrtWeights(),
                                 trt_weights.Shape());
  TRT_ENSURE_PTR_OK(weights_const);
  params->converter->SetLayerName(*weights_const, params->node_def, "const");
  nvinfer1::ITensor* qdq_input = (*weights_const)->getOutput(0);
  std::string name = absl::StrCat((*weights_const)->getName(), "_output");
  qdq_input->setName(name.c_str());
  return qdq_input;
}

}  // namespace

// Carries traits for each specific quantization op type for conversion.
// Specialization for template parameter T should be given for each TF C++
// quantization op.
template <typename T>
struct QDQOpSpec {};

template <>
struct QDQOpSpec<ops::QuantizeAndDequantizeV2> {
  static constexpr std::array<InputArgSpec, 3> InputSpec() {
    return {
        InputArgSpec::Create("input", TrtInputArg::kBoth),
        InputArgSpec::Create("input_min", TrtInputArg::kWeight),
        InputArgSpec::Create("input_max", TrtInputArg::kWeight),
    };
  }

  struct Attrs {
    float min_range;
    float max_range;
    bool narrow_range;
    std::string round_mode;
    UniformQuantizationScales scales;
  };

  static Status ValidateQDQForExplicitPrecision(
      const std::vector<TRT_TensorOrWeights>& inputs, const NodeDef& node_def,
      Attrs* args) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc mht_1(mht_1_v, 307, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.cc", "ValidateQDQForExplicitPrecision");

    AttrSlice attrs(node_def);
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "round_mode", &args->round_mode));
    if (args->round_mode != "HALF_TO_EVEN") {
      LOG(WARNING) << node_def.op() << ": " << node_def.name()
                   << " has round_mode=" << args->round_mode
                   << ", but for TensorRT conversion, "
                      "round_mode=HALF_TO_EVEN is recommended.";
    }
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "narrow_range", &args->narrow_range));
    if (args->narrow_range) {
      LOG(WARNING) << node_def.op() << ": " << node_def.name()
                   << " has narrow_range=true, but for TensorRT conversion, "
                      "narrow_range=false is recommended.";
    }
    args->min_range = inputs.at(1).weights().template GetPointer<float>()[0];
    args->max_range = inputs.at(2).weights().template GetPointer<float>()[0];
    const int num_bits = 8;
    args->scales = ComputeQuantizationRange<float>(
        /*signed_input=*/true, num_bits, args->narrow_range, &args->min_range,
        &args->max_range);
    TRT_ENSURE(args->scales.dequantize_scale[0] != 0);
    TRT_ENSURE(args->scales.quantize_scale[0] != 0);
    return Status::OK();
  }

  // Converts in explicit precision mode. In this mode, QDQ operations are
  // directly converted into TensorRT quantizing and dequantizing scale
  // operations.
  static Status ConvertExplicit(OpConverterParams* params, const Attrs& args) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc mht_2(mht_2_v, 339, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.cc", "ConvertExplicit");

    const auto& node_def = params->node_def;

    StatusOr<TRTNetworkBuilder> builder = TRTNetworkBuilder::Create(
        params->converter->network(), params->weight_store);

    StatusOr<nvinfer1::ITensor*> qdq_input =
        ExlicitQDQInputToTensor(&*builder, params, params->inputs.at(0));
    TRT_ENSURE_PTR_OK(qdq_input);

    // TODO(cbate): check this condition exists for TRT8? Outline this block to
    // a "reshape policy".
    const int required_dims = params->use_implicit_batch ? 3 : 4;
    const nvinfer1::Dims idims = (*qdq_input)->getDimensions();
    nvinfer1::Dims intermediate_dims = idims;
    TRT_ENSURE(idims.nbDims > 0);
    if (idims.nbDims < required_dims) {
      const int nb_extra_dims = required_dims - idims.nbDims;
      intermediate_dims.nbDims = required_dims;
      std::vector<int> ones(nb_extra_dims, 1);
      TRT_ENSURE(ones.size() == nb_extra_dims && nb_extra_dims > 0);

      if (!params->use_implicit_batch) {
        intermediate_dims.d[0] = idims.d[0];
        std::copy(ones.begin(), ones.end(), intermediate_dims.d + 1);
        std::copy_n(idims.d + 1, idims.nbDims - 1,
                    intermediate_dims.d + ones.size() + 1);
      } else {
        std::copy(ones.begin(), ones.end(), intermediate_dims.d);
        std::copy_n(idims.d, idims.nbDims, intermediate_dims.d + ones.size());
      }

      LOG(WARNING) << absl::StrCat(
          node_def.name(), ":", node_def.op(), ": tensor ",
          (*qdq_input)->getName(), " has shape ", DebugString(idims),
          " but TRT scale layer requires at least 3 dims excluding batch dim, "
          "trying to recover by inserting 1's to create shape ",
          DebugString(intermediate_dims));
      StatusOr<nvinfer1::IShuffleLayer*> reshape =
          builder->Reshape(*qdq_input, intermediate_dims);
      TRT_ENSURE_PTR_OK(reshape);
      *qdq_input = (*reshape)->getOutput(0);
    }

    VLOG(1) << "[ExplicitPrecision]" << node_def.op() << ": " << node_def.name()
            << " computed scales: " << args.scales << " from min/max ranges "
            << args.min_range << "/" << args.max_range;

    StatusOr<nvinfer1::ILayer*> qdq =
        builder->UniformQuantizeDequantizeExplicit(
            *qdq_input, args.scales.quantize_scale[0],
            args.scales.dequantize_scale[0], node_def.name());
    TRT_ENSURE_PTR_OK(qdq);
    ITensorProxyPtr final_output = (*qdq)->getOutput(0);
    if (idims.nbDims != intermediate_dims.nbDims) {
      StatusOr<nvinfer1::IShuffleLayer*> undo_reshape =
          builder->Reshape(*qdq_input, idims);
      TRT_ENSURE_PTR_OK(undo_reshape);
      final_output = (*undo_reshape)->getOutput(0);
    }
    params->outputs->push_back(final_output);
    return Status::OK();
  }
};

template <>

struct QDQOpSpec<ops::QuantizeAndDequantizeV3> {
  static constexpr std::array<InputArgSpec, 4> InputSpec() {
    return {
        InputArgSpec::Create("input", TrtInputArg::kBoth),
        InputArgSpec::Create("min", TrtInputArg::kWeight),
        InputArgSpec::Create("max", TrtInputArg::kWeight),
        InputArgSpec::Create("num_bits", TrtInputArg::kWeight),
    };
  }
  // Use same attributes and conversion functions as QDQV2.
  using Attrs = QDQOpSpec<ops::QuantizeAndDequantizeV2>::Attrs;

  static Status ValidateQDQForExplicitPrecision(
      const std::vector<TRT_TensorOrWeights>& inputs, const NodeDef& node_def,
      Attrs* args) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc mht_3(mht_3_v, 423, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.cc", "ValidateQDQForExplicitPrecision");

    return QDQOpSpec<
        ops::QuantizeAndDequantizeV2>::ValidateQDQForExplicitPrecision(inputs,
                                                                       node_def,
                                                                       args);
  }

  static Status ConvertExplicit(OpConverterParams* params, const Attrs& args) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc mht_4(mht_4_v, 433, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.cc", "ConvertExplicit");

    return QDQOpSpec<ops::QuantizeAndDequantizeV2>::ConvertExplicit(params,
                                                                    args);
  }
};

template <>

struct QDQOpSpec<ops::FakeQuantWithMinMaxVars> {
  static constexpr std::array<InputArgSpec, 3> InputSpec() {
    return {
        InputArgSpec::Create("input", TrtInputArg::kBoth),
        InputArgSpec::Create("min", TrtInputArg::kWeight),
        InputArgSpec::Create("max", TrtInputArg::kWeight),
    };
  }
  struct Attrs {
    int num_bits;
    bool narrow_range;
  };

  static Status ValidateQDQForExplicitPrecision(
      const std::vector<TRT_TensorOrWeights>& inputs, const NodeDef& node_def,
      Attrs* args) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc mht_5(mht_5_v, 459, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.cc", "ValidateQDQForExplicitPrecision");

    return errors::Unimplemented("");
  }

  static Status ConvertExplicit(OpConverterParams* params, const Attrs& args) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc mht_6(mht_6_v, 466, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.cc", "ConvertExplicit");

    return errors::Unimplemented("");
  }
};

template <>

struct QDQOpSpec<ops::FakeQuantWithMinMaxArgs> {
  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return {
        InputArgSpec::Create("input", TrtInputArg::kBoth),
    };
  }

  struct Attrs {
    float min;
    float max;
    int num_bits;
    bool narrow_range;
  };

  static Status ValidateQDQForExplicitPrecision(
      const std::vector<TRT_TensorOrWeights>& inputs, const NodeDef& node_def,
      Attrs* args) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc mht_7(mht_7_v, 492, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.cc", "ValidateQDQForExplicitPrecision");

    return errors::Unimplemented("");
  }

  static Status ConvertExplicit(OpConverterParams* params, const Attrs& args) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc mht_8(mht_8_v, 499, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.cc", "ConvertExplicit");

    return errors::Unimplemented("");
  }
};

// Converts QDQ operations in non-explicit precision mode. This is the original
// "ConvertQuantize" function. In this mode, Q/DQ operations are no-ops and are
// instead used to set the dynamic range of the input tensor.
Status ConvertDynamicRangeMode(OpConverterParams* params) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc mht_9(mht_9_v, 510, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.cc", "ConvertDynamicRangeMode");

  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;

  float min_range = 0.0f;
  float max_range = 0.0f;
  AttrSlice attrs(params->node_def);

  if (node_def.op() == "FakeQuantWithMinMaxArgs") {
    // Get ranges via node attributes.
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "min", &min_range));
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "max", &max_range));
  } else if (node_def.op() == "FakeQuantWithMinMaxVars" ||
             node_def.op() == "QuantizeAndDequantizeV2" ||
             node_def.op() == "QuantizeAndDequantizeV3") {
    // Get ranges via inputs.
    auto get_weights_value = [&inputs](int index) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc mht_10(mht_10_v, 529, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.cc", "lambda");

      const auto* raw_weights = inputs.at(index).weights().GetPointer<float>();
      return raw_weights[0];
    };
    min_range = get_weights_value(1);
    max_range = get_weights_value(2);
  } else {
    return errors::InvalidArgument("Unknown quantization op ", node_def.op(),
                                   ", at ", node_def.name());
  }
  if (params->validation_only) {
    return Status::OK();
  }

  // Store ranges for tensor
  ITensorProxyPtr input0 = inputs.at(0).tensor();
  params->converter->ProvideQuantizationRange(&input0, min_range, max_range);
  // Sometimes, TRT may not quantize a tensor, either because it chooses to
  // execute a higher precision kernel or because of op fusion. In these
  // cases, accuracy will suffer if the model was trained to expect
  // quantization at that tensor. We should consider adding a clip(tensor,
  // min_range, max_range) operation here to ensure that any arbitrarily
  // placed quantize node will execute as expected. However, this will
  // negatively affect performance. If users train their models in a way which
  // models inference as close as possible (i.e. not quantizing in place where
  // fusion will occur), then there is no problem with the current
  // implementation.
  params->outputs->push_back(inputs.at(0));
  return Status::OK();
}

template <typename TFOpType>
class ConvertQDQ : public OpConverterBase<ConvertQDQ<TFOpType>> {
 public:
  explicit ConvertQDQ(OpConverterParams* params)
      : OpConverterBase<ConvertQDQ<TFOpType>>(params) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc mht_11(mht_11_v, 567, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.cc", "ConvertQDQ");
}

  static constexpr auto InputSpec() { return QDQOpSpec<TFOpType>::InputSpec(); }

  // Disable the non-applicable data type check by providing empty string.
  static constexpr const char* NodeDefDataTypeAttributeName() { return ""; }

  Status ValidateDynamicRangeINT8Mode() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc mht_12(mht_12_v, 577, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.cc", "ValidateDynamicRangeINT8Mode");

    // The condition ensures we only call the conversion once. We should break
    // this function up into validation and conversion.
    if (this->params_->validation_only) {
      return ConvertDynamicRangeMode(this->params_);
    }
    return Status::OK();
  }

  Status Validate() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc mht_13(mht_13_v, 589, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.cc", "Validate");

    if (!this->params_->use_explicit_precision) {
      return ValidateDynamicRangeINT8Mode();
    }
    return OpSpec::ValidateQDQForExplicitPrecision(
        this->params_->inputs, this->params_->node_def, &attrs_);
  }

  Status Convert() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSquantization_opsDTcc mht_14(mht_14_v, 600, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.cc", "Convert");

    if (!this->params_->use_explicit_precision) {
      return ConvertDynamicRangeMode(this->params_);
    }
    return OpSpec::ConvertExplicit(this->params_, attrs_);
  }

  using OpSpec = QDQOpSpec<TFOpType>;
  using OpSpecAttrs = typename QDQOpSpec<TFOpType>::Attrs;
  OpSpecAttrs attrs_;
};

REGISTER_DEFAULT_TRT_OP_CONVERTER(
    MakeConverterFunction<ConvertQDQ<ops::QuantizeAndDequantizeV2>>(),
    "QuantizeAndDequantizeV2");
REGISTER_DEFAULT_TRT_OP_CONVERTER(
    MakeConverterFunction<ConvertQDQ<ops::QuantizeAndDequantizeV3>>(),
    "QuantizeAndDequantizeV3");
REGISTER_DEFAULT_TRT_OP_CONVERTER(
    MakeConverterFunction<ConvertQDQ<ops::FakeQuantWithMinMaxVars>>(),
    "FakeQuantWithMinMaxVars");
REGISTER_DEFAULT_TRT_OP_CONVERTER(
    MakeConverterFunction<ConvertQDQ<ops::FakeQuantWithMinMaxArgs>>(),
    "FakeQuantWithMinMaxArgs");

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
