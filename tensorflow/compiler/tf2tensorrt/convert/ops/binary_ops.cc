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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSbinary_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSbinary_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSbinary_opsDTcc() {
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

#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {
const binaryOperationMap *BinaryOperationMap() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSbinary_opsDTcc mht_0(mht_0_v, 193, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/binary_ops.cc", "BinaryOperationMap");

  static auto *const m = new binaryOperationMap({
      {"Add", nvinfer1::ElementWiseOperation::kSUM},
      {"AddV2", nvinfer1::ElementWiseOperation::kSUM},
      {"Mul", nvinfer1::ElementWiseOperation::kPROD},
      {"Sub", nvinfer1::ElementWiseOperation::kSUB},
      {"Div", nvinfer1::ElementWiseOperation::kDIV},
      {"FloorDiv", nvinfer1::ElementWiseOperation::kFLOOR_DIV},
      {"RealDiv", nvinfer1::ElementWiseOperation::kDIV},
      {"Minimum", nvinfer1::ElementWiseOperation::kMIN},
      {"Maximum", nvinfer1::ElementWiseOperation::kMAX},
      {"Pow", nvinfer1::ElementWiseOperation::kPOW},
  });
  return m;
}

class ConvertBinaryImpl {
 protected:
  ConvertBinaryImpl(const binaryOperationMap *pOperMap) : pOperMap_(pOperMap) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSbinary_opsDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/binary_ops.cc", "ConvertBinaryImpl");
}

  Status ImplValidate(const OpConverterParams &params) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSbinary_opsDTcc mht_2(mht_2_v, 219, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/binary_ops.cc", "ImplValidate");

    const auto &node_def = params.node_def;
    const auto op = node_def.op();
    const auto op_pair = pOperMap_->find(op);
    if (op_pair == pOperMap_->end()) {
      return errors::Unimplemented("Binary op: ", op, " not supported");
    }

    // Constant folding should have been done by TensorFlow.
    const auto &inputs = params.inputs;
    if (inputs.at(0).is_weights() && inputs.at(1).is_weights()) {
      return errors::Unimplemented(
          "Constant folding is falled back to TensorFlow, binary op '", op,
          "' received both input as constant");
    }

    nvinfer1::Dims broadcasted_dims[2];
    TF_RETURN_IF_ERROR(GetTrtBroadcastShape(
        inputs.at(0), inputs.at(1), true, params.use_implicit_batch,
        broadcasted_dims, broadcasted_dims + 1));

    for (int i = 0; i < 2; i++) {
      tensor_[i] = nullptr;
      // This will also convert constants to tensors.
      TF_RETURN_IF_ERROR(PrepareTensorForShape(
          params.converter, inputs.at(i), broadcasted_dims[i],
          params.validation_only, tensor_ + i, node_def, i));
    }
    operation_ = op_pair->second;
    return Status::OK();
  }

  Status ImplConvert(const OpConverterParams &params) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSbinary_opsDTcc mht_3(mht_3_v, 254, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/binary_ops.cc", "ImplConvert");

    const auto &node_def = params.node_def;
    // Add ElementWise layer.
    nvinfer1::ILayer *layer = params.converter->network()->addElementWise(
        *tensor_[0]->trt_tensor(), *tensor_[1]->trt_tensor(), operation_);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());

    if (params.use_explicit_precision) {
      layer->setPrecision(nvinfer1::DataType::kFLOAT);
    }

    params.converter->SetLayerName(layer, node_def);
    params.outputs->push_back(TRT_TensorOrWeights(layer->getOutput(0)));
    return Status::OK();
  }
  static constexpr std::array<InputArgSpec, 2> InputSpec() {
    return std::array<InputArgSpec, 2>{
        InputArgSpec::Create("x", TrtInputArg::kBoth),
        InputArgSpec::Create("y", TrtInputArg::kBoth)};
  }

 private:
  const binaryOperationMap *pOperMap_;
  ITensorProxyPtr tensor_[2];
  nvinfer1::ElementWiseOperation operation_;
};

class ConvertBinary : public OpConverterBase<ConvertBinary>,
                      protected ConvertBinaryImpl {
 public:
  explicit ConvertBinary(OpConverterParams *params)
      : OpConverterBase<ConvertBinary>(params),
        ConvertBinaryImpl(BinaryOperationMap()) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSbinary_opsDTcc mht_4(mht_4_v, 289, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/binary_ops.cc", "ConvertBinary");
}

  static constexpr std::array<DataType, 3> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32};
  }

  static constexpr std::array<InputArgSpec, 2> InputSpec() {
    return ConvertBinaryImpl::InputSpec();
  }

  Status Validate() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSbinary_opsDTcc mht_5(mht_5_v, 302, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/binary_ops.cc", "Validate");
 return ImplValidate(*params_); }
  Status Convert() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSbinary_opsDTcc mht_6(mht_6_v, 306, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/binary_ops.cc", "Convert");
 return ImplConvert(*params_); }
};

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertBinary>(),
                                  GetOperationNames(*BinaryOperationMap()));

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
