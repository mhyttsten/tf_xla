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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSfill_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSfill_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSfill_opsDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

#if IS_TRT_VERSION_GE(8, 2, 0, 0)

class ConvertFill : public OpConverterBase<ConvertFill> {
 public:
  explicit ConvertFill(OpConverterParams* params)
      : OpConverterBase<ConvertFill>(params) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSfill_opsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/fill_ops.cc", "ConvertFill");
}

  static constexpr std::array<DataType, 3> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32};
  }

  static constexpr std::array<InputArgSpec, 2> InputSpec() {
    return std::array<InputArgSpec, 2>{
        InputArgSpec::Create("dims", TrtInputArg::kBoth),
        InputArgSpec::Create("value", TrtInputArg::kBoth)};
  }

  Status Validate() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSfill_opsDTcc mht_1(mht_1_v, 215, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/fill_ops.cc", "Validate");

    const auto& params = *this->params_;

    if (params.use_implicit_batch) {
      return errors::Unimplemented(
          "Conversion for Fill is not implemented in"
          "implicit batch mode");
    }

    const auto& inputs = params.inputs;
    const auto& node_def = params.node_def;
    const TRT_TensorOrWeights& dims_input = inputs.at(0);

    nvinfer1::DataType dims_type = dims_input.TrtDType();
    if (dims_type != nvinfer1::DataType::kINT32) {
      return errors::InvalidArgument("The dims parameter of ", node_def.op(),
                                     " operation in ", node_def.name(),
                                     " is expected to be of type ",
                                     DebugString(nvinfer1::DataType::kINT32),
                                     " type, got ", DebugString(dims_type));
    }

    int nbDims = dims_input.GetTrtDims().nbDims;
    if (nbDims < 0) {
      return errors::InvalidArgument("The shape of parameter ", node_def.op(),
                                     " operation in ", node_def.name(),
                                     " cannot be partial.");
    }
    return Status::OK();
  }

  Status Convert() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSfill_opsDTcc mht_2(mht_2_v, 249, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/fill_ops.cc", "Convert");

    const auto& params = *this->params_;
    const auto& inputs = params.inputs;
    auto* converter = params.converter;
    auto* network = converter->network();
    const auto& node_def = params.node_def;

    const bool is_dims_static = inputs[0].is_weights();
    const bool is_value_static = inputs[1].is_weights();

    const TRT_TensorOrWeights& dims_input = inputs.at(0);
    const TRT_TensorOrWeights& value_input = inputs.at(1);

    int nbDims = dims_input.GetTrtDims().d[0];

    nvinfer1::Dims trt_dims{0};
    if (is_dims_static) {
      const auto dims_weights = dims_input.weights();
      DimsAdapter dims_adapter(dims_weights.GetSpan<int32>());
      dims_adapter.TrtDims(&trt_dims);
    }

    // TensorRT IFillLayer requires a rank 0 scalar.
    ITensorProxyPtr scalar_tensor;
    nvinfer1::Dims scalar_dims;
    scalar_dims.nbDims = 0;
    nvinfer1::DataType value_type = value_input.TrtDType();
    if (is_value_static) {
      scalar_tensor =
          converter->CreateConstantLayer(value_input.weights(), scalar_dims);
    } else {
      TF_RETURN_IF_ERROR(PrepareTensorForShape(
          converter, value_input, scalar_dims, params.validation_only,
          &scalar_tensor, node_def));
    }

    auto builder = TRTNetworkBuilder::Create(network, params.weight_store);
    nvinfer1::Dims beta_shape{1, {nbDims}};
    StatusOr<nvinfer1::IConstantLayer*> const_layer =
        builder->Constant(0, beta_shape, value_type);
    TF_RETURN_IF_ERROR(const_layer.status());
    ITensorProxyPtr empty_beta_tensor = (*const_layer)->getOutput(0);

    nvinfer1::IFillLayer* layer =
        network->addFill(trt_dims, nvinfer1::FillOperation::kLINSPACE);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    if (!is_dims_static) {
      layer->setInput(0, *dims_input.tensor()->trt_tensor());
    }
    layer->setInput(1, *scalar_tensor->trt_tensor());
    layer->setInput(2, *empty_beta_tensor->trt_tensor());
    converter->SetLayerName(layer, node_def, "fill");
    ITensorProxyPtr output_tensor = layer->getOutput(0);
    AddOutput(TRT_TensorOrWeights(output_tensor));
    return Status::OK();
  }
};

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertFill>(), "Fill");

#endif  // IS_TRT_VERSION_GE(8, 2, 0, 0)

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
