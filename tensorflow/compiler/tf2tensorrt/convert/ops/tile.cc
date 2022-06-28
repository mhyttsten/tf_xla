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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPStileDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPStileDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPStileDTcc() {
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

class ConvertTile : public OpConverterBase<ConvertTile> {
 public:
  explicit ConvertTile(OpConverterParams *params)
      : OpConverterBase<ConvertTile>(params) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPStileDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/tile.cc", "ConvertTile");
}

  static constexpr std::array<DataType, 3> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32};
  }

  static constexpr std::array<InputArgSpec, 2> InputSpec() {
    return std::array<InputArgSpec, 2>{
        InputArgSpec::Create("input_tensor", TrtInputArg::kBoth),
        InputArgSpec::Create("weight", TrtInputArg::kBoth)};
  }

  Status Validate() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPStileDTcc mht_1(mht_1_v, 213, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/tile.cc", "Validate");

    const auto &params = *this->params_;
    const auto &inputs = params.inputs;

    const auto &repl = inputs.at(1);
    if (params.use_implicit_batch && repl.is_tensor()) {
      return errors::InvalidArgument(
          "Conversion for Tile is not implemented for multipliers "
          "passed as a tensor in implicit batch mode.");
    }

    nvinfer1::DataType dtype;
    const int *multiplies;
    if (repl.is_weights()) {
      TFTRT_CHECK_SHAPE_TENSOR(repl.weights().GetTensor());
      dtype = repl.weights().TrtDType();
      multiplies = repl.weights().GetPointer<int>();
    } else {
      dtype = repl.tensor()->getType();
      multiplies = nullptr;
    }

    if (dtype != nvinfer1::DataType::kINT32) {
      return errors::InvalidArgument(
          "The replication parameter of the ", params.node_def.op(),
          " operation in ", params.node_def.name(), " is expected to be of ",
          DebugString(nvinfer1::DataType::kINT32), " type, got ",
          DebugString(dtype), ".");
    }

    const auto dims = inputs.at(0).GetTrtDims();
    const auto nb_dims =
        dims.nbDims +
        (params.use_implicit_batch && inputs.at(0).is_tensor() ? 1 : 0);
    if (multiplies) {
      const int mult_numb = repl.weights().count();
      if (mult_numb != nb_dims) {
        return errors::InvalidArgument(
            "The length of the replication vector (", mult_numb,
            ") of the Tile operation in '", params.node_def.name(),
            "' is expected to be equal to the rank of the input vector (",
            nb_dims, ").");
      }

      if (std::any_of(multiplies, multiplies + nb_dims,
                      [](int i) { return i <= 0; })) {
        const auto &mul = absl::StrJoin(multiplies, multiplies + nb_dims, ", ");
        return errors::InvalidArgument(
            "All replications of the Tile operation in ",
            params.node_def.name(), " should be positive, got (", mul, ").");
      }
    } else {
      const auto &repl_dims = repl.GetTrtDims();
      if (repl_dims.nbDims != 1) {
        return errors::InvalidArgument(
            "When replications are defined as a tensor, that tensor must be "
            "1-dimensional. Got ",
            repl_dims.nbDims, "-dimensional tensor.");
      }

      // Check the number of elements in multiplyer for tensors with non-dynamic
      // shape
      if (repl_dims.d[0] >= 0 && repl_dims.d[0] != nb_dims) {
        return errors::InvalidArgument(
            "When replications are defined as a tensor, "
            "the number of its elements (",
            repl_dims.d[0], ") must be equal to the rank of the input tensor (",
            nb_dims, ").");
      }
    }

    return Status::OK();
  }

  Status Convert() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPStileDTcc mht_2(mht_2_v, 290, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/tile.cc", "Convert");

    const auto &params = *this->params_;
    const auto &inputs = params.inputs;
    auto *converter = params.converter;
    auto *network = converter->network();
    const auto &tensor = inputs.at(0);
    const auto &replics = inputs.at(1);
    const auto dims = tensor.GetTrtDims();
    const auto nb_dims = dims.nbDims;

    nvinfer1::Dims size{nb_dims, {1}};
    bool dynamic_flag = replics.is_tensor();
    if (!dynamic_flag) {
      const auto dim_adj =
          params.use_implicit_batch && tensor.is_tensor() ? 1 : 0;
      const auto *pSize = dims.d;
      dynamic_flag = std::any_of(pSize + 1 - dim_adj, pSize + nb_dims,
                                 [](int i) { return i < 0; });
      const int *pMultiplies = replics.weights().GetPointer<int>() + dim_adj;
      for (int i = 1 - dim_adj; i < nb_dims; i++)
        size.d[i] = pMultiplies[i] * pSize[i];
    }

    StatusOr<TRTNetworkBuilder> builder;
    if (tensor.is_weights() || (dynamic_flag && replics.is_weights())) {
      builder =
          TRTNetworkBuilder::Create(converter->network(), params.weight_store);
      TRT_ENSURE_OK(builder);
    }

    ITensorProxyPtr input_tensor;
    if (tensor.is_weights()) {
      StatusOr<nvinfer1::IConstantLayer *> weights_const =
          builder->WeightsToConstant(tensor.weights().GetTrtWeights(), dims);
      TRT_ENSURE_PTR_OK(weights_const);
      input_tensor = (*weights_const)->getOutput(0);
    } else {
      input_tensor = tensor.tensor();
    }

    auto &input_trt_tensor = *input_tensor->trt_tensor();
    nvinfer1::ITensor *target_shape = nullptr;
    if (dynamic_flag) {
      nvinfer1::ITensor *mult;
      if (replics.is_weights()) {
        StatusOr<nvinfer1::IConstantLayer *> weights_const =
            builder->WeightsToConstant(replics.weights().GetTrtWeights(),
                                       replics.GetTrtDims());
        TRT_ENSURE_PTR_OK(weights_const);
        mult = (*weights_const)->getOutput(0);
      } else {
        const ITensorProxyPtr multiplies = replics.tensor()->trt_tensor();
        mult = multiplies->trt_tensor();
      }

      nvinfer1::ITensor *shape =
          network->addShape(input_trt_tensor)->getOutput(0);
      target_shape = network
                         ->addElementWise(*shape, *mult,
                                          nvinfer1::ElementWiseOperation::kPROD)
                         ->getOutput(0);
    }

    nvinfer1::Dims start{nb_dims, {}};
    DimsAdapter stride(std::vector<int>(nb_dims, 1));
    auto layer =
        network->addSlice(input_trt_tensor, start, size, stride.AsTrtDims());
    layer->setMode(nvinfer1::SliceMode::kWRAP);
    if (target_shape) layer->setInput(2, *target_shape);

    converter->SetLayerName(layer, params.node_def.name(), "to_tile");
    ITensorProxyPtr output_tensor = layer->getOutput(0);
    if (tensor.is_weights() && params.use_implicit_batch) {
      // Reshape output tensor by removing first dimension.
      DimsAdapter adap(output_tensor->getDimensions());
      TF_RETURN_IF_ERROR(adap.RemoveBatchDimension());

      TF_RETURN_IF_ERROR(PrepareTensorForShape(
          params.converter, TRT_TensorOrWeights(output_tensor),
          adap.AsTrtDims(), false, &output_tensor, params.node_def));
    }

    AddOutput(TRT_TensorOrWeights(output_tensor));
    return Status::OK();
  }
};

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertTile>(), "Tile");

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
