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
class MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_opPSsimple_tflite_op_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_opPSsimple_tflite_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_opPSsimple_tflite_op_testDTcc() {
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
#include "tensorflow/lite/kernels/shim/test_op/simple_tflite_op.h"

#include <cstring>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace {

class SimpleOpModel : public SingleOpModel {
 public:
  // Builds the op model and feeds in inputs, ready to invoke.
  SimpleOpModel(const std::vector<uint8_t>& op_options,
                const std::vector<tflite::TensorType>& input_types,
                const std::vector<std::vector<int>>& input_shapes,
                const std::string& input0,
                const std::vector<std::vector<int64_t>>& input1,
                const std::vector<tflite::TensorType>& output_types) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("input0: \"" + input0 + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_opPSsimple_tflite_op_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/lite/kernels/shim/test_op/simple_tflite_op_test.cc", "SimpleOpModel");

    // Define inputs.
    std::vector<int> input_idx;
    for (const auto input_type : input_types) {
      input_idx.push_back(AddInput(input_type));
    }
    // Define outputs.
    for (const auto output_type : output_types) {
      output_idx_.push_back(AddOutput(output_type));
    }
    // Build the interpreter.
    SetCustomOp(OpName_SIMPLE_OP(), op_options, Register_SIMPLE_OP);
    BuildInterpreter(input_shapes);
    // Populate inputs.
    PopulateStringTensor(input_idx[0], {input0});
    for (int i = 0; i < input1.size(); ++i) {
      PopulateTensor(input_idx[1 + i], input1[i]);
    }
  }

  template <typename T>
  std::vector<T> GetOutput(const int i) {
    return ExtractVector<T>(output_idx_[i]);
  }

  std::vector<int> GetOutputShape(const int i) {
    return GetTensorShape(output_idx_[i]);
  }

 protected:
  // Tensor indices
  std::vector<int> output_idx_;
};

TEST(SimpleOpModel, OutputSize_5_N_2) {
  // Test input
  flexbuffers::Builder builder;
  builder.Map([&]() {
    builder.Int("output1_size", 5);
    builder.String("output2_suffix", "foo");
    builder.Int("N", 2);
  });
  builder.Finish();
  std::vector<std::vector<int>> input_shapes = {{}, {}, {2}};
  std::vector<tflite::TensorType> input_types = {tflite::TensorType_STRING,
                                                 tflite::TensorType_INT64,
                                                 tflite::TensorType_INT64};
  std::vector<tflite::TensorType> output_types = {
      tflite::TensorType_INT32, tflite::TensorType_FLOAT32,
      tflite::TensorType_STRING, tflite::TensorType_INT64,
      tflite::TensorType_INT64};
  const std::string input0 = "abc";
  const std::vector<std::vector<int64_t>> input1 = {{123}, {456, 789}};
  // Run the op
  SimpleOpModel m(/*op_options=*/builder.GetBuffer(), input_types, input_shapes,
                  input0, input1, output_types);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  // Assertions
  EXPECT_THAT(m.GetOutput<int>(0), testing::ElementsAre(0, 1, 2, 3, 4));
  EXPECT_THAT(m.GetOutput<float>(1),
              testing::ElementsAre(0, 0.5, 1.0, 1.5, 2.0));
  EXPECT_THAT(m.GetOutput<std::string>(2),
              testing::ElementsAre("0", "1", "2", "foo"));
  EXPECT_THAT(m.GetOutput<int64_t>(3), testing::ElementsAre(124));
  EXPECT_THAT(m.GetOutputShape(3), testing::ElementsAre());
  EXPECT_THAT(m.GetOutput<int64_t>(4), testing::ElementsAre(457, 790));
  EXPECT_THAT(m.GetOutputShape(4), testing::ElementsAre(2));
}

TEST(SimpleOpModel, OutputSize_3_N_0) {
  // Test input
  flexbuffers::Builder builder;
  builder.Map([&]() {
    builder.Int("output1_size", 3);
    builder.String("output2_suffix", "foo");
    builder.Int("N", 0);
  });
  builder.Finish();
  std::vector<std::vector<int>> input_shapes = {{}};
  std::vector<tflite::TensorType> input_types = {tflite::TensorType_STRING};
  std::vector<tflite::TensorType> output_types = {tflite::TensorType_INT32,
                                                  tflite::TensorType_FLOAT32,
                                                  tflite::TensorType_STRING};
  const std::string input0 = "abcde";
  const std::vector<std::vector<int64_t>> input1;
  // Run the op
  SimpleOpModel m(/*op_options=*/builder.GetBuffer(), input_types, input_shapes,
                  input0, input1, output_types);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  // Assertions
  EXPECT_THAT(m.GetOutput<int>(0), testing::ElementsAre(0, 1, 2, 3, 4));
  EXPECT_THAT(m.GetOutput<float>(1), testing::ElementsAre(0, 0.5, 1.0));
  EXPECT_THAT(m.GetOutput<std::string>(2),
              testing::ElementsAre("0", "1", "2", "3", "4", "foo"));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
