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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSunpack_quantize_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSunpack_quantize_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSunpack_quantize_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

// The following tests make sure the quantize operation for unpack has the
// correct quantization parameters
namespace {

class UnpackQuantizeTest : public ::testing::Test {
 protected:
  UnpackQuantizeTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSunpack_quantize_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/toco/graph_transformations/tests/unpack_quantize_test.cc", "UnpackQuantizeTest");
}

  // Prepare a hypothetical TOCO model with one unpack operator in it
  // together with 2 arrays as its outputs.

  // Since we are testing quantization in action, we are going to have all
  // inputs as kFloat. Outputs are also kFloat. This will force the
  // transformation operation to
  // 1. calculate min and max of the input.
  // 2. insert dequantization nodes after quantized outputs of Unpack operation.
  void PrepareModel(Model* model, int axis) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSunpack_quantize_testDTcc mht_1(mht_1_v, 214, "", "./tensorflow/lite/toco/graph_transformations/tests/unpack_quantize_test.cc", "PrepareModel");

    std::vector<std::string> unpack_output_names = {"unpack_out0",
                                                    "unpack_out1"};
    model->flags.add_output_arrays(unpack_output_names[0]);
    model->flags.add_output_arrays(unpack_output_names[1]);
    const std::string unpack_input_name("unpack_op_input");

    const int kDim = 2;
    const int kElementPerDim = 2;
    const int kBufSize = 4;
    static float in_buf[kBufSize] = {0.0, 1.0, 2.0, 3.0};

    // Input arrays is going to be in kFloat since in this case quantization
    // transformation will be forced to calculate min and max of the input.
    Array& in_array = model->GetOrCreateArray(unpack_input_name);
    in_array.data_type = ArrayDataType::kFloat;

    // Initialize shape for the input array.
    Shape* in_array_shape = in_array.mutable_shape();
    std::vector<int>* in_array_shape_dim = in_array_shape->mutable_dims();
    for (int i = 0; i < kDim; i++) {
      in_array_shape_dim->push_back(kElementPerDim);
    }
    auto& in_array_buffer =
        in_array.GetMutableBuffer<toco::ArrayDataType::kFloat>();
    in_array_buffer.data.resize(kBufSize);
    auto* buf_ptr = in_array_buffer.data.data();
    std::copy(in_buf, in_buf + kBufSize, buf_ptr);

    auto* unpack_op = new UnpackOperator;
    unpack_op->axis = axis;
    unpack_op->inputs = {unpack_input_name};
    unpack_op->outputs = unpack_output_names;

    // Configuring the necessary outputs. The outputs also happen to be in
    // kFloat. This is because during quantization transformation data types for
    // these arrays are going to be forced to be kUint8.
    for (const std::string& unpack_output_name : unpack_output_names) {
      Array& out_array = model->GetOrCreateArray(unpack_output_name);
      out_array.GetOrCreateMinMax();
      out_array.data_type = ArrayDataType::kFloat;
      out_array.GetMutableBuffer<ArrayDataType::kFloat>().data.resize(
          kElementPerDim);

      Shape* out_array_shape = out_array.mutable_shape();
      std::vector<int>* out_array_shape_dim = out_array_shape->mutable_dims();
      out_array_shape_dim->resize(kDim - 1);
      for (int i = 0; i < kDim - 1; i++) {
        (*out_array_shape_dim)[i] = kElementPerDim;
      }
    }

    model->operators.push_back(std::unique_ptr<Operator>(unpack_op));
  }
};
}  // namespace

TEST_F(UnpackQuantizeTest, CheckUnpackPreservesQuantizationParameters) {
  using testing::ElementsAre;
  using testing::ElementsAreArray;
  Model model;
  const int axis = 0;
  PrepareModel(&model, axis);

  GraphTransformationsSet graph_transformation_set;
  graph_transformation_set.Add(new toco::Quantize);
  bool modified;
  ASSERT_TRUE((*graph_transformation_set.begin())
                  ->Run(&model, /*op_index=*/0, &modified)
                  .ok());

  const std::string output_name = model.flags.output_arrays(0);

  // Quantization transformation inserts NODE_NAME_DEQUANTIZE operations,
  // effectively making them the new outputs of the array. Old outputs of the
  // array are being fed into dequantization nodes. Furthermore, dequantize
  // nodes are being set as model outputs in model flags.  Therefore, we get the
  // following configuration OriginalInput->Unpack->OriginalOutputQuantized->
  // ->Dequantize. In fact we are interested in quantization parameters of
  // OriginalOutputQuantized array, hence using the original string constants
  // from the test fixture preparation code.
  const auto& unpack_input_array = model.GetArray("unpack_op_input");
  const auto& unpack_array0 = model.GetArray("unpack_out0");
  const auto& unpack_array1 = model.GetArray("unpack_out1");
  // Checking quantization params match, minmax match for array1
  EXPECT_THAT(unpack_input_array.quantization_params->zero_point,
              unpack_array0.quantization_params->zero_point);
  EXPECT_THAT(unpack_input_array.quantization_params->scale,
              unpack_array0.quantization_params->scale);
  EXPECT_THAT(unpack_input_array.minmax->min, unpack_array0.minmax->min);
  EXPECT_THAT(unpack_input_array.minmax->max, unpack_array0.minmax->max);

  // ...and for array2
  EXPECT_THAT(unpack_input_array.quantization_params->zero_point,
              unpack_array1.quantization_params->zero_point);
  EXPECT_THAT(unpack_input_array.quantization_params->scale,
              unpack_array1.quantization_params->scale);
  EXPECT_THAT(unpack_input_array.minmax->min, unpack_array1.minmax->min);
  EXPECT_THAT(unpack_input_array.minmax->max, unpack_array1.minmax->max);
}
}  // namespace toco
