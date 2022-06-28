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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSresolve_constant_unary_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSresolve_constant_unary_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSresolve_constant_unary_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

void RunResolveSum(const std::vector<float>& input,
                   const std::vector<int>& input_shape,
                   const std::vector<int>& axis,
                   const std::vector<int>& output_shape,
                   const std::vector<float>& expected_output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSresolve_constant_unary_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/toco/graph_transformations/tests/resolve_constant_unary_test.cc", "RunResolveSum");

  Model model;
  const std::string output_name("output");
  model.flags.add_output_arrays(output_name);
  Array& input0 = model.GetOrCreateArray("input0");
  Array& input1 = model.GetOrCreateArray("input1");
  Array& output = model.GetOrCreateArray(output_name);

  *input0.mutable_shape()->mutable_dims() = input_shape;
  input0.data_type = ArrayDataType::kFloat;
  input0.GetMutableBuffer<ArrayDataType::kFloat>().data = input;

  *input1.mutable_shape()->mutable_dims() = {static_cast<int>(axis.size())};
  input1.GetMutableBuffer<ArrayDataType::kInt32>().data = axis;
  input1.data_type = ArrayDataType::kInt32;

  *output.mutable_shape()->mutable_dims() = output_shape;

  auto sum_op = absl::make_unique<TensorFlowSumOperator>();
  sum_op->keep_dims = true;
  sum_op->inputs = {"input0", "input1"};
  sum_op->outputs = {output_name};
  model.operators.push_back(std::move(sum_op));
  bool modified;
  ASSERT_TRUE(ResolveConstantUnaryOperator().Run(&model, 0, &modified).ok());
  EXPECT_EQ(model.GetArray("output").GetBuffer<ArrayDataType::kFloat>().data,
            expected_output);
  EXPECT_EQ(model.GetArray("output").shape().dims(), output_shape);
}

// Reduce a 2d array across axis 0
TEST(ResolveConstantUnary, ResolveSumAxis0_2D) {
  // clang-format off
  RunResolveSum(
      // Input data
      {3, 1, 4, 1,
       5, 9, 2, 6,
       5, 3, 5, 8},

      // Input shape
      {3, 4},

      // Axes
      {0},

      // Expected output shape,
      {1, 4},

      // Expected output
      {13, 13, 11, 15});
  // clang-format on
}

// Reduce a 2d array across axis 1
TEST(ResolveConstantUnary, ResolveSumAxis1_2D) {
  // clang-format off
  RunResolveSum(
      // Input data
      {3, 1, 4, 1,
       5, 9, 2, 6,
       5, 3, 5, 8},

      // Input shape
      {3, 4},

      // Axes
      {1},

      // Expected output shape,
      {3, 1},

      // Expected output
      {9, 22, 21});
  // clang-format on
}

// Reduce a 3d tensor across axes 0 and 2.
TEST(ResolveConstantUnary, ResolveSumAxis0_2_3D) {
  // clang-format off
  RunResolveSum(
      // Input data
      {  0,   1,   2,
         3,  10,  11,
        12,  13,  20,
        21,  22,  23,

       100, 101, 102,
       103, 110, 111,
       112, 113, 120,
       121, 122, 123,

       200, 201, 202,
       203, 210, 211,
       212, 213, 220,
       221, 222, 223 },

      // Input shape
      {3, 4, 3},

      // Axes
      {0, 2},

      // Expected output shape,
      {1, 4, 1},

      // Expected output, generated using octave.
      { 909, 972, 1035, 1098});
  // clang-format on
}

}  // namespace
}  // namespace toco
