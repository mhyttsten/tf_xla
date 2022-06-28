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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSunary_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSunary_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSunary_opsDTcc() {
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

const std::unordered_map<string, nvinfer1::UnaryOperation>*
UnaryOperationMap() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSunary_opsDTcc mht_0(mht_0_v, 195, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/unary_ops.cc", "UnaryOperationMap");

  static auto* const m =
      new std::unordered_map<string, nvinfer1::UnaryOperation>({
        {"Neg", nvinfer1::UnaryOperation::kNEG},
            {"Exp", nvinfer1::UnaryOperation::kEXP},
            {"Log", nvinfer1::UnaryOperation::kLOG},
            {"Sqrt", nvinfer1::UnaryOperation::kSQRT},
            {"Abs", nvinfer1::UnaryOperation::kABS},
            {"Reciprocal", nvinfer1::UnaryOperation::kRECIP},
            {"Sin", nvinfer1::UnaryOperation::kSIN},
            {"Cos", nvinfer1::UnaryOperation::kCOS},
            {"Tan", nvinfer1::UnaryOperation::kTAN},
            {"Sinh", nvinfer1::UnaryOperation::kSINH},
            {"Cosh", nvinfer1::UnaryOperation::kCOSH},
            {"Asin", nvinfer1::UnaryOperation::kASIN},
            {"Acos", nvinfer1::UnaryOperation::kACOS},
            {"Atan", nvinfer1::UnaryOperation::kATAN},
            {"Asinh", nvinfer1::UnaryOperation::kASINH},
            {"Acosh", nvinfer1::UnaryOperation::kACOSH},
            {"Atanh", nvinfer1::UnaryOperation::kATANH},
            {"Ceil", nvinfer1::UnaryOperation::kCEIL},
            {"Floor", nvinfer1::UnaryOperation::kFLOOR},
            {"Erf", nvinfer1::UnaryOperation::kERF},
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
            {"Round", nvinfer1::UnaryOperation::kROUND},
            {"Sign", nvinfer1::UnaryOperation::kSIGN},
#endif
      });
  return m;
}

class ConvertUnary : public OpConverterBase<ConvertUnary> {
 public:
  explicit ConvertUnary(OpConverterParams* params)
      : OpConverterBase<ConvertUnary>(params) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSunary_opsDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/unary_ops.cc", "ConvertUnary");
}

  static constexpr std::array<DataType, 3> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF};
  }

  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return std::array<InputArgSpec, 1>{
        InputArgSpec::Create("x", TrtInputArg::kTensor)};
  }

  Status Validate() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSunary_opsDTcc mht_2(mht_2_v, 246, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/unary_ops.cc", "Validate");

    const auto& op = params_->node_def.op();
    if (UnaryOperationMap()->find(op) == UnaryOperationMap()->end()) {
      return errors::Unimplemented("Unary op: ", op, " not supported");
    }

    if (params_->use_implicit_batch && (op == "Sign" || op == "Round")) {
      return errors::Unimplemented("Unary op: '", params_->node_def.op(),
                                   "' is not supported in implicit batch mode");
    }

    return Status::OK();
  }

  Status Convert() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSunary_opsDTcc mht_3(mht_3_v, 263, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/unary_ops.cc", "Convert");

    const auto& params = *this->params_;
    const auto& node_def = params.node_def;
    auto* converter = params.converter;

    const auto op_pair = UnaryOperationMap()->find(node_def.op());
    ITensorProxyPtr tensor = params.inputs.at(0).tensor();
    nvinfer1::IUnaryLayer* layer =
        converter->network()->addUnary(*tensor->trt_tensor(), op_pair->second);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    converter->SetLayerName(layer, node_def);

    params.outputs->push_back(TRT_TensorOrWeights(layer->getOutput(0)));
    return Status::OK();
  }
};

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertUnary>(),
                                  GetOperationNames(*UnaryOperationMap()));

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
