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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_testDTcc() {
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
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter.h"

#include <gtest/gtest.h>
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

using ::tensorflow::testing::IsOk;
using ::tensorflow::testing::StatusIs;
using ::testing::HasSubstr;

class ExampleOpConverter : public OpConverterBase<ExampleOpConverter> {
 public:
  explicit ExampleOpConverter(OpConverterParams* params)
      : OpConverterBase<ExampleOpConverter>(params) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter_test.cc", "ExampleOpConverter");
}

  static constexpr const char* NodeDefDataTypeAttributeName() {
    return "data_type";
  }

  static constexpr std::array<DataType, 2> AllowedDataTypes() {
    return {DataType::DT_FLOAT};
  }

  static constexpr std::array<InputArgSpec, 2> InputSpec() {
    return std::array<InputArgSpec, 2>{
        InputArgSpec::Create("input_tensor", TrtInputArg::kTensor),
        InputArgSpec::Create("weight", TrtInputArg::kWeight)};
  }

  Status Validate() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_testDTcc mht_1(mht_1_v, 225, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter_test.cc", "Validate");
 return Status::OK(); }

  Status Convert() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_testDTcc mht_2(mht_2_v, 230, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter_test.cc", "Convert");

    AddOutput(TRT_TensorOrWeights(nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims{1, {1, 1, 1}}, 1));
    return Status::OK();
  }
};

TEST(TestOpConverterBase, TestOpConverterBase) {
  // Register a converter which uses the base converter class.
  GetOpConverterRegistry()->Register(
      "FakeFunc", 1, MakeConverterFunction<ExampleOpConverter>());

  NodeDef def;
  def.set_op("FakeFunc");
  auto converter = Converter::Create(TrtPrecisionMode::FP32, false,
                                     Logger::GetLogger(), false, "test_engine");
  EXPECT_THAT(converter, IsOk());

  // Base class should check attribute with key given by
  // Impl::NodeDefDataTypeAttributeName().
  Status conversion_status = (*converter)->ConvertNode(def);
  EXPECT_THAT(conversion_status,
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Attribute with name data_type not found")));

  // Add partial inputs to the node and make the converter aware.
  def.mutable_input()->Add("input1");
  conversion_status = (*converter)
                          ->AddInputTensor("input1", nvinfer1::DataType::kFLOAT,
                                           nvinfer1::Dims{4, {1, 1, 1, 1}}, 1);
  EXPECT_THAT(conversion_status, IsOk());

  // Base class method should check number of inputs.
  AddNodeAttr("data_type", DT_FLOAT, &def);
  conversion_status = (*converter)->ConvertNode(def);
  EXPECT_THAT(conversion_status, StatusIs(error::INTERNAL));

  // Add second input to the node and make the converter aware.
  def.mutable_input()->Add("input2");
  conversion_status = (*converter)
                          ->AddInputTensor("input2", nvinfer1::DataType::kFLOAT,
                                           nvinfer1::Dims{4, {1, 1, 1, 1}}, 1);
  EXPECT_THAT(conversion_status, IsOk());

  // Base class validation should check the type (Constant or Tensor) of the
  // inputs.
  conversion_status = (*converter)->ConvertNode(def);
  EXPECT_THAT(
      conversion_status,
      StatusIs(error::UNIMPLEMENTED,
               HasSubstr("input \"weight\" for FakeFunc must be a constant")));

  // Correct input2 so that it is a weight.
  (*converter)->TensorsMap().erase("input2");
  (*converter)
      ->TensorsMap()
      .insert(std::make_pair("input2", TRT_TensorOrWeights(TRT_ShapedWeights(
                                           nvinfer1::DataType::kFLOAT))));

  // With the correct input types, check that the converter is called and sets
  // one output tensor.
  conversion_status = (*converter)->ConvertNode(def);
  EXPECT_THAT(conversion_status, IsOk());
  EXPECT_EQ((*converter)->TensorsMap().size(), 3U);

  GetOpConverterRegistry()->Clear("FakeFunc");
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif
