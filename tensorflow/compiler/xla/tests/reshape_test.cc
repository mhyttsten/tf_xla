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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSreshape_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSreshape_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSreshape_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

// Use a bool parameter to indicate whether to use bfloat16.
class ReshapeTest : public ::testing::WithParamInterface<bool>,
                    public ClientLibraryTestBase {
 public:
  ReshapeTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSreshape_testDTcc mht_0(mht_0_v, 216, "", "./tensorflow/compiler/xla/tests/reshape_test.cc", "ReshapeTest");
 set_use_bfloat16(GetParam()); }

  ErrorSpec zero_error_spec_{0.0};
};

// Collapses 2-dimensional pseudo-scalar (single-element array) to 1 dimension.
XLA_TEST_P(ReshapeTest, CollapseTrivial1x1) {
  XlaBuilder builder(TestName());
  Array2D<float> input_array(1, 1);
  input_array.Fill(1.0f);
  auto input_literal = LiteralUtil::CreateR2FromArray2D(input_array);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(
                      0, input_literal, "parameter", &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{0, 1});

  auto expected_literal = LiteralUtil::CreateR1<float>({1.0f});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, CollapseTrivialR1EmptyDims) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateR1<float>({1.0f});
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(
                      0, input_literal, "parameter", &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{});

  auto expected_literal = LiteralUtil::CreateR1<float>({1.0f});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, CollapseTrivialR1OnlyDim) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateR1<float>({1.0f});
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(
                      0, input_literal, "parameter", &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{0});

  auto expected_literal = LiteralUtil::CreateR1<float>({1.0f});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Collapses 2-dimensional pseudo-scalar (single-element array) to scalar.
XLA_TEST_P(ReshapeTest, SingleElementArrayToScalar) {
  XlaBuilder builder(TestName());
  Array2D<float> input_array(1, 1);
  input_array.Fill(1.0f);
  auto input_literal = LiteralUtil::CreateR2FromArray2D(input_array);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(
                      0, input_literal, "parameter", &builder, &parameter));
  auto reshape = Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1},
                         /*new_sizes=*/{});
  auto new_shape = builder.GetShape(reshape).ConsumeValueOrDie();

  auto expected_literal = LiteralUtil::CreateR0<float>(1.0f);
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, ScalarToSingleElementArray) {
  XlaBuilder builder(TestName());

  Literal param0_literal = LiteralUtil::CreateR0<float>(1.0f);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, param0_literal, "param0",
                                                    &builder, &parameter));
  auto a = Neg(parameter);
  Reshape(/*operand=*/a, /*dimensions=*/{}, /*new_sizes=*/{1});

  auto expected_literal = LiteralUtil::CreateR1<float>({-1.0f});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, Trivial0x3) {
  XlaBuilder builder(TestName());
  Array2D<float> input_array(0, 3);
  auto input_literal = LiteralUtil::CreateR2FromArray2D(input_array);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{0, 1});
  auto expected_literal = LiteralUtil::CreateR1<float>({});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, Trivial0x3WithParameter) {
  XlaBuilder builder(TestName());

  Literal param0_literal =
      LiteralUtil::CreateR2FromArray2D<float>(Array2D<float>(0, 3));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, param0_literal, "param0",
                                                    &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{0, 1});
  auto expected_literal = LiteralUtil::CreateR1<float>({});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, Trivial3x0) {
  XlaBuilder builder(TestName());
  Array2D<float> input_array(3, 0);
  auto input_literal = LiteralUtil::CreateR2FromArray2D(input_array);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{0, 1});
  auto expected_literal = LiteralUtil::CreateR1<float>({});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Collapses a 2-dimensional row vector to 1 dimension.
XLA_TEST_P(ReshapeTest, Trivial1x3) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateR2<float>({{1.0f, 2.0f, 3.0f}});
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{0, 1});
  auto expected_literal = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Collapses a 2-dimensional column vector to 1 dimension.
XLA_TEST_P(ReshapeTest, Trivial3x1) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateR2<float>({{1.0f}, {2.0f}, {3.0f}});
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{0, 1});
  auto expected_literal = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Splits an empty vector into an empty matrix.
XLA_TEST_P(ReshapeTest, R1ToR2_0_To_2x0) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateR1<float>({});
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0},
          /*new_sizes=*/{2, 0});
  auto expected_literal = LiteralUtil::CreateR2<float>({{}, {}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Splits a vector into a matrix.
XLA_TEST_P(ReshapeTest, R1ToR2_6_To_2x3) {
  XlaBuilder builder(TestName());
  auto input_literal =
      LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0},
          /*new_sizes=*/{2, 3});
  auto expected_literal =
      LiteralUtil::CreateR2<float>({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Transposes a 2x0 array to a 0x2 array.
XLA_TEST_P(ReshapeTest, Reshape0x2To2x0) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(Array2D<float>(0, 2));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1},
          /*new_sizes=*/{2, 0});
  auto expected_literal = LiteralUtil::CreateR2<float>({{}, {}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Transposes a 2-dimensional row vector to a column vector.
XLA_TEST_P(ReshapeTest, ReshapeRowToCol) {
  XlaBuilder builder(TestName());
  auto simple = MakeLinspaceArray2D(1.0f, 3.0f, 1, 3);
  auto input_literal = LiteralUtil::CreateFromArray(*simple);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1},
          /*new_sizes=*/{3, 1});

  auto expected = ReferenceUtil::TransposeArray2D(*simple);
  auto expected_literal = LiteralUtil::CreateFromArray(*expected);
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Transposes a 2-dimensional array.
XLA_TEST_P(ReshapeTest, TransposeAsReshape) {
  XlaBuilder builder(TestName());
  auto a4x3 = MakeLinspaceArray2D(1.0f, 12.0f, 4, 3);
  auto input_literal = LiteralUtil::CreateFromArray(*a4x3);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{1, 0},
          /*new_sizes=*/{3, 4});

  auto expected = ReferenceUtil::TransposeArray2D(*a4x3);
  auto expected_literal = LiteralUtil::CreateFromArray(*expected);
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Transposes a 0x4 array with XlaBuilder::Transpose.
XLA_TEST_P(ReshapeTest, Transpose0x4) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(Array2D<float>(0, 4));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Transpose(parameter, {1, 0});
  auto expected_literal = LiteralUtil::CreateR2<float>({{}, {}, {}, {}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Transposes a 2-dimensional array with ComputationBuilder::Trans.
XLA_TEST_P(ReshapeTest, Transpose4x3) {
  XlaBuilder builder(TestName());
  auto a4x3 = MakeLinspaceArray2D(1.0f, 12.0f, 4, 3);
  auto input_literal = LiteralUtil::CreateFromArray(*a4x3);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Transpose(parameter, {1, 0});

  auto expected = ReferenceUtil::TransposeArray2D(*a4x3);
  auto expected_literal = LiteralUtil::CreateFromArray(*expected);
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Reshapes an empty 2-dimensional array with dimensions that are not just a
// rearrangement of the originals (split), but no reordering (no shuffle).
XLA_TEST_P(ReshapeTest, ReshapeSplitNoShuffleZeroElements) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(Array2D<float>(6, 0));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1},
          /*new_sizes=*/{2, 3, 0, 0});
  auto expected_literal =
      LiteralUtil::CreateFromArray(Array4D<float>(2, 3, 0, 0));
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, ReshapeR4ToR2ZeroElements) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(Array4D<float>(2, 3, 4, 0));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1, 2, 3},
          /*new_sizes=*/{24, 0});
  auto expected_literal = LiteralUtil::CreateFromArray(Array2D<float>(24, 0));
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Reshapes a 2-dimensional array with dimensions that are not just a
// rearrangement of the originals (split), but no reordering (no shuffle).
XLA_TEST_P(ReshapeTest, ReshapeSplitNoShuffle) {
  XlaBuilder builder(TestName());
  auto a4x3 = MakeLinspaceArray2D(1.0f, 12.0f, 4, 3);
  auto input_literal = LiteralUtil::CreateFromArray(*a4x3);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1},
          /*new_sizes=*/{2, 6});

  auto expected = MakeLinspaceArray2D(1.0f, 12.0f, 2, 6);
  auto expected_literal = LiteralUtil::CreateFromArray(*expected);
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, ReshapeSplitAndShuffleZeroElements) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(Array2D<float>(0, 6));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{1, 0},
          /*new_sizes=*/{3, 0});
  auto expected_literal = LiteralUtil::CreateFromArray(Array2D<float>(3, 0));
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Reshapes a 2-dimensional array with dimensions that are not just a
// rearrangement of the originals (split), and reorder the input (shuffle).
XLA_TEST_P(ReshapeTest, ReshapeSplitAndShuffle) {
  XlaBuilder builder(TestName());
  auto a4x3 = MakeLinspaceArray2D(1.0f, 12.0f, 4, 3);
  auto input_literal = LiteralUtil::CreateFromArray(*a4x3);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{1, 0},
          /*new_sizes=*/{2, 6});
  Array2D<float> expected({{1.0f, 4.0f, 7.0f, 10.0f, 2.0f, 5.0f},
                           {8.0f, 11.0f, 3.0f, 6.0f, 9.0f, 12.0f}});
  auto expected_literal = LiteralUtil::CreateFromArray(expected);
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// The following tests use the same input 3D array; they test the examples we
// show for the Reshape operation in the operation_semantics document.
// TODO(b/34503277): find a way to show this code in the documentation without
// duplication on the TF documentation server.
static Array3D<float> ArrayForDocR3Tests() {
  return Array3D<float>({{{10, 11, 12}, {15, 16, 17}},
                         {{20, 21, 22}, {25, 26, 27}},
                         {{30, 31, 32}, {35, 36, 37}},
                         {{40, 41, 42}, {45, 46, 47}}});
}

XLA_TEST_P(ReshapeTest, DocR3_R1_Collapse_012) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(ArrayForDocR3Tests());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1, 2},
          /*new_sizes=*/{24});
  auto expected_literal = LiteralUtil::CreateR1<float>(
      {10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26, 27,
       30, 31, 32, 35, 36, 37, 40, 41, 42, 45, 46, 47});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, DocR3_R2_Collapse_012_Refine_83) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(ArrayForDocR3Tests());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1, 2},
          /*new_sizes=*/{8, 3});
  auto expected_literal = LiteralUtil::CreateR2<float>({{10, 11, 12},
                                                        {15, 16, 17},
                                                        {20, 21, 22},
                                                        {25, 26, 27},
                                                        {30, 31, 32},
                                                        {35, 36, 37},
                                                        {40, 41, 42},
                                                        {45, 46, 47}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, DocR3_R1_Collapse_120) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(ArrayForDocR3Tests());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{1, 2, 0},
          /*new_sizes=*/{24});
  auto expected_literal = LiteralUtil::CreateR1<float>(
      {10, 20, 30, 40, 11, 21, 31, 41, 12, 22, 32, 42,
       15, 25, 35, 45, 16, 26, 36, 46, 17, 27, 37, 47});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, DocR3_R2_Collapse_120_Refine_83) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(ArrayForDocR3Tests());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{1, 2, 0},
          /*new_sizes=*/{8, 3});
  auto expected_literal = LiteralUtil::CreateR2<float>({{10, 20, 30},
                                                        {40, 11, 21},
                                                        {31, 41, 12},
                                                        {22, 32, 42},
                                                        {15, 25, 35},
                                                        {45, 16, 26},
                                                        {36, 46, 17},
                                                        {27, 37, 47}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, DocR3_R3_Collapse_120_Refine_262) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::CreateFromArray(ArrayForDocR3Tests());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{1, 2, 0},
          /*new_sizes=*/{2, 6, 2});
  auto expected_literal = LiteralUtil::CreateR3<float>(
      {{{10, 20}, {30, 40}, {11, 21}, {31, 41}, {12, 22}, {32, 42}},
       {{15, 25}, {35, 45}, {16, 26}, {36, 46}, {17, 27}, {37, 47}}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Collapses the low dimensions of a 4D tensor to get a 2D matrix, without
// reordering dimensions (for NeuralNet::FullyConnected).
//
// First we create a tesseract raster-face like:
//
// 1 2 3
// 4 5 6
//
// First we collapse Y and X within the raster space yielding:
//
// 1 2 3 4 5 6
//
// Then we collapse Z be collapsed so we just end up with planes:
//
// 1 2 3 4 5 6 1 2 3 4 5 6
XLA_TEST_P(ReshapeTest, FullyConnectedCollapse) {
  XlaBuilder builder(TestName());
  Array4D<float> t2x2x2x3(2, 2, 2, 3);
  auto filler2x3 = MakeLinspaceArray2D(1.0f, 6.0f, 2, 3);
  t2x2x2x3.FillWithYX(*filler2x3);
  auto input_literal = LiteralUtil::CreateFromArray(t2x2x2x3);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Collapse(/*operand=*/parameter, /*dimensions=*/{1, 2, 3});
  auto expected_literal = LiteralUtil::CreateR2<float>(
      {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
       {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        6.0f}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// As above, but uses reshape directly.
XLA_TEST_P(ReshapeTest, FullyConnectedCollapseDesugared) {
  XlaBuilder builder(TestName());
  Array4D<float> t(2, 1, 2, 2);
  t(0, 0, 0, 0) = 0;
  t(0, 0, 0, 1) = 1;
  t(0, 0, 1, 0) = 2;
  t(0, 0, 1, 1) = 3;
  t(1, 0, 0, 0) = 4;
  t(1, 0, 0, 1) = 5;
  t(1, 0, 1, 0) = 6;
  t(1, 0, 1, 1) = 7;
  auto input_literal = LiteralUtil::CreateFromArray(t);
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(/*operand=*/parameter, /*dimensions=*/{0, 1, 2, 3},
          /*new_sizes=*/{2, 4});

  auto expected_literal =
      LiteralUtil::CreateR2<float>({{0, 1, 2, 3}, {4, 5, 6, 7}});
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Reshape various ranks to a scalar.
XLA_TEST_P(ReshapeTest, ToScalar) {
  for (int rank = 0; rank < 8; ++rank) {
    XlaBuilder b(TestName());
    std::vector<int64_t> ones(rank, 1);  // this is {1, ..., 1}.
    std::vector<int64_t> dimensions(rank);
    std::iota(dimensions.begin(), dimensions.end(), 0);
    Literal input_literal(ShapeUtil::MakeShape(F32, ones));
    std::vector<int64_t> zeros(rank, 0);  // this is {0, ..., 0}.
    input_literal.Set<float>(zeros, 83.0f);

    XlaOp parameter;
    TF_ASSERT_OK_AND_ASSIGN(
        auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                      &b, &parameter));
    Reshape(parameter, dimensions, {});

    auto expected_literal = LiteralUtil::CreateR0<float>(83.0f);
    ComputeAndCompareLiteral(&b, expected_literal, {input.get()},
                             zero_error_spec_);
  }
}

XLA_TEST_P(ReshapeTest, BadDimensions) {
  XlaBuilder b(TestName());
  auto input_literal = LiteralUtil::CreateR1<float>({1.0f});
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &b, &parameter));
  Reshape(parameter, {}, {});
  EXPECT_THAT(
      ExecuteToString(&b, {}),
      ::testing::HasSubstr("not a permutation of the operand dimensions"));
}

XLA_TEST_P(ReshapeTest, BadNewSizes) {
  XlaBuilder b(TestName());
  auto input_literal = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &b, &parameter));
  Reshape(parameter, {1}, {});
  EXPECT_THAT(ExecuteToString(&b, {}),
              ::testing::HasSubstr("mismatched element counts"));
}

XLA_TEST_P(ReshapeTest, R4Dim0MinorLayoutToR2Dim0MajorLayout) {
  XlaBuilder builder(TestName());
  // clang-format off
  auto input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      Array4D<float>{
    {
      {
        {0, 1},
        {2, 3},
      },
      {
        {100, 101},
        {102, 103},
      },
    },
    {
      {
        {222, 333},
        {444, 555},
      },
      {
        {666, 777},
        {888, 999},
      },
    },
  },
       LayoutUtil::MakeLayout({0, 1, 2, 3}));
  // clang-format on
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));

  Reshape(parameter, /*dimensions=*/{0, 1, 2, 3}, /*new_sizes=*/{2, 8});

  Array2D<float> expected_array({
      {0, 1, 2, 3, 100, 101, 102, 103},
      {222, 333, 444, 555, 666, 777, 888, 999},
  });

  XlaComputation computation = builder.Build().ConsumeValueOrDie();
  ExecutionOptions execution_options = execution_options_;
  *execution_options.mutable_shape_with_output_layout() =
      ShapeUtil::MakeShapeWithLayout(use_bfloat16() ? BF16 : F32, {2, 8},
                                     {1, 0})
          .ToProto();
  Literal actual =
      client_
          ->ExecuteAndTransfer(computation, {input.get()}, &execution_options)
          .ConsumeValueOrDie();
  Literal expected = LiteralUtil::CreateR2FromArray2D<float>(expected_array);
  if (use_bfloat16()) {
    expected = LiteralUtil::ConvertF32ToBF16(expected);
  }
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
}

XLA_TEST_P(ReshapeTest, R2ToR4_3x8_To_3x2x1x4) {
  XlaBuilder builder(TestName());
  Literal input_literal = LiteralUtil::CreateR2<float>({
      {0, 1, 2, 3, 4, 5, 6, 7},
      {100, 101, 102, 103, 104, 105, 106, 107},
      {200, 201, 202, 203, 204, 205, 206, 207},
  });
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 1}, /*new_sizes=*/{3, 2, 1, 4});

  // clang-format off
  auto expected_literal = LiteralUtil::CreateR4<float>({
    {{{0, 1, 2, 3}},
     {{4, 5, 6, 7}}},
    {{{100, 101, 102, 103}},
     {{104, 105, 106, 107}}},
    {{{200, 201, 202, 203}},
     {{204, 205, 206, 207}}}
  });
  // clang-format on
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

// Tests R2->R4 reshape with the reshape dimensions {1, 0}.
XLA_TEST_P(ReshapeTest, R2ToR4_3x8_To_3x2x1x4_Dimensions_10) {
  XlaBuilder builder(TestName());
  Literal input_literal = LiteralUtil::CreateR2<float>({
      {0, 1, 2, 3, 4, 5, 6, 7},
      {100, 101, 102, 103, 104, 105, 106, 107},
      {200, 201, 202, 203, 204, 205, 206, 207},
  });
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, input_literal, "input",
                                                    &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{1, 0}, /*new_sizes=*/{3, 2, 1, 4});

  // clang-format off
  auto expected_literal = LiteralUtil::CreateR4<float>({
    {{{0, 100, 200, 1}},
     {{101, 201, 2, 102}}},
    {{{202, 3, 103, 203}},
     {{4, 104, 204, 5}}},
    {{{105, 205, 6, 106}},
     {{206, 7, 107, 207}}}
  });
  // clang-format on
  ComputeAndCompareLiteral(&builder, expected_literal, {input.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, R4ToR2_2x1x1x1_To_2x1) {
  XlaBuilder builder(TestName());
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  Array4D<float> input(2, 1, 1, 1);
  input.Each([&rng, &distribution](absl::Span<const int64_t> /* indices */,
                                   float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 1, 2, 3}, /*new_sizes=*/{2, 1});

  Literal expected = LiteralUtil::ReshapeSlice({2, 1}, {1, 0}, input_literal);
  ComputeAndCompareLiteral(&builder, expected, {input_data.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, R4ToR2_2x1x4x1_To_4x2) {
  XlaBuilder builder(TestName());
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  Array4D<float> input(2, 1, 4, 1);
  input.Each([&rng, &distribution](absl::Span<const int64_t> /* indices */,
                                   float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 1, 2, 3}, /*new_sizes=*/{4, 2});

  Literal expected = LiteralUtil::ReshapeSlice({4, 2}, {1, 0}, input_literal);
  ComputeAndCompareLiteral(&builder, expected, {input_data.get()},
                           zero_error_spec_);
}

// Tests R4->R2 reshape with the reshape dimensions {0, 2, 1, 3}.
XLA_TEST_P(ReshapeTest, R4ToR2_5x10x2x3_To_5x60_Dimensions_0213) {
  XlaBuilder builder(TestName());
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  Array4D<float> input(5, 10, 2, 3);
  input.Each([&rng, &distribution](absl::Span<const int64_t> /* indices */,
                                   float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 2, 1, 3},
          /*new_sizes=*/{5, 60});

  Array2D<float> expected_array(5, 60);
  input.Each([&](absl::Span<const int64_t> indices, float* cell) {
    expected_array(indices[0], indices[2] * 30 + indices[1] * 3 + indices[3]) =
        *cell;
  });
  auto expected = LiteralUtil::CreateR2FromArray2D(expected_array);
  ComputeAndCompareLiteral(&builder, expected, {input_data.get()},
                           zero_error_spec_);
}

XLA_TEST_P(ReshapeTest, NoopReshape) {
  XlaBuilder builder(TestName());
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  Array4D<float> input_array(2, 3, 5, 7);
  input_array.Each(
      [&rng, &distribution](absl::Span<const int64_t> /* indices */,
                            float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input_array, LayoutUtil::MakeLayout({1, 2, 3, 0}));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{3, 0, 1, 2},
          /*new_sizes=*/{7, 2, 3, 5});
  XlaComputation computation = builder.Build().ConsumeValueOrDie();

  ExecutionOptions execution_options = execution_options_;
  *execution_options.mutable_shape_with_output_layout() =
      ShapeUtil::MakeShapeWithLayout(use_bfloat16() ? BF16 : F32, {7, 2, 3, 5},
                                     {2, 3, 0, 1})
          .ToProto();
  Literal output_literal =
      client_
          ->ExecuteAndTransfer(computation, {input_data.get()},
                               &execution_options)
          .ConsumeValueOrDie();

  // Since the reshape is a no-op, verify that it does not change the underlying
  // data.
  if (use_bfloat16()) {
    auto expected = LiteralUtil::ConvertF32ToBF16(input_literal);
    EXPECT_EQ(expected.data<bfloat16>(), output_literal.data<bfloat16>());
  } else {
    EXPECT_EQ(input_literal.data<float>(), output_literal.data<float>());
  }
}

XLA_TEST_P(ReshapeTest, R4ToR4Reshape_Trivial) {
  XlaBuilder builder(TestName());
  auto literal_1x2x3x4 = LiteralUtil::CreateR4<float>(
      {{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
        {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}});

  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, literal_1x2x3x4, "input",
                                                    &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 1, 2, 3},
          /*new_sizes=*/{1, 2, 3, 4});

  ComputeAndCompareLiteral(&builder, literal_1x2x3x4, {input.get()});
}

XLA_TEST_P(ReshapeTest, R4ToR4Reshape) {
  auto literal_1x2x3x4 = LiteralUtil::CreateR4<float>(
      {{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
        {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}});

  XlaBuilder builder(TestName());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input, CreateParameterAndTransferLiteral(0, literal_1x2x3x4, "input",
                                                    &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{1, 3, 2, 0},
          /*new_sizes=*/{2, 4, 3, 1});

  // clang-format off
  auto expected_2x4x3x1 = LiteralUtil::CreateR4<float>(
      {{{{1}, {5}, {9}},
        {{2}, {6}, {10}},
        {{3}, {7}, {11}},
        {{4}, {8}, {12}}},
       {{{13}, {17}, {21}},
        {{14}, {18}, {22}},
        {{15}, {19}, {23}},
        {{16}, {20}, {24}}}});
  // clang-format on

  ComputeAndCompareLiteral(&builder, expected_2x4x3x1, {input.get()});
}

XLA_TEST_P(ReshapeTest, R4TwoMinorTransposeSimple) {
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<int64_t> bounds = {2, 2, 2, 2};
  std::vector<int64_t> new_bounds = {bounds[0], bounds[1], bounds[3],
                                     bounds[2]};
  Array4D<float> input(bounds[0], bounds[1], bounds[2], bounds[3]);
  input.Each([&rng, &distribution](absl::Span<const int64_t> /* indices */,
                                   float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaBuilder builder(TestName());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 1, 3, 2},
          /*new_sizes=*/new_bounds);

  Literal expected =
      LiteralUtil::ReshapeSlice(new_bounds, {2, 3, 1, 0}, input_literal)
          .Relayout(LayoutUtil::MakeLayout({3, 2, 1, 0}));

  // Specify the requested output shape explicitly to ensure that this reshape
  // actually corresponds to a two minor transpose.
  ComputeAndCompareLiteral(&builder, expected, {input_data.get()},
                           zero_error_spec_, &expected.shape());
}

XLA_TEST_P(ReshapeTest, R4TwoMinorTransposeMajorFirstEffectiveR2) {
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<int64_t> bounds = {1, 1, 250, 300};
  std::vector<int64_t> new_bounds = {bounds[0], bounds[1], bounds[3],
                                     bounds[2]};
  Array4D<float> input(bounds[0], bounds[1], bounds[2], bounds[3]);
  input.Each([&rng, &distribution](absl::Span<const int64_t> /* indices */,
                                   float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaBuilder builder(TestName());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 1, 3, 2},
          /*new_sizes=*/new_bounds);

  Literal expected =
      LiteralUtil::ReshapeSlice(new_bounds, {2, 3, 1, 0}, input_literal)
          .Relayout(LayoutUtil::MakeLayout({3, 2, 1, 0}));

  // Specify the requested output shape explicitly to ensure that this reshape
  // actually corresponds to a two minor transpose.
  ComputeAndCompareLiteral(&builder, expected, {input_data.get()},
                           zero_error_spec_, &expected.shape());
}

XLA_TEST_P(ReshapeTest, R4TwoMinorTransposeMajorFirstMinorEffectiveR1) {
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<int64_t> bounds = {5, 5, 1, 10};
  std::vector<int64_t> new_bounds = {bounds[0], bounds[1], bounds[3],
                                     bounds[2]};
  Array4D<float> input(bounds[0], bounds[1], bounds[2], bounds[3]);
  input.Each([&rng, &distribution](absl::Span<const int64_t> /* indices */,
                                   float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaBuilder builder(TestName());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 1, 3, 2},
          /*new_sizes=*/new_bounds);

  Literal expected =
      LiteralUtil::ReshapeSlice(new_bounds, {2, 3, 1, 0}, input_literal)
          .Relayout(LayoutUtil::MakeLayout({3, 2, 1, 0}));

  // Specify the requested output shape explicitly to ensure that this reshape
  // actually corresponds to a two minor transpose.
  ComputeAndCompareLiteral(&builder, expected, {input_data.get()},
                           zero_error_spec_, &expected.shape());
}

XLA_TEST_P(ReshapeTest, R4TwoMinorTransposeMajorFirstMinorEffectiveR1InR2) {
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  // This happens in NN-Builder MNIST.
  std::vector<int64_t> bounds = {5, 5, 10, 1};
  std::vector<int64_t> new_bounds = {bounds[0], bounds[1], bounds[3],
                                     bounds[2]};
  Array4D<float> input(bounds[0], bounds[1], bounds[2], bounds[3]);
  input.Each([&rng, &distribution](absl::Span<const int64_t> /* indices */,
                                   float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaBuilder builder(TestName());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{0, 1, 3, 2},
          /*new_sizes=*/new_bounds);

  Literal expected =
      LiteralUtil::ReshapeSlice(new_bounds, {2, 3, 1, 0}, input_literal)
          .Relayout(LayoutUtil::MakeLayout({3, 2, 1, 0}));

  // Specify the requested output shape explicitly to ensure that this reshape
  // actually corresponds to a two minor transpose.
  ComputeAndCompareLiteral(&builder, expected, {input_data.get()},
                           zero_error_spec_, &expected.shape());
}

XLA_TEST_P(ReshapeTest, R4TwoMinorTransposeTrivialR2) {
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<int64_t> bounds = {3, 3, 1, 3};
  std::vector<int64_t> new_bounds = {bounds[1], bounds[0], bounds[2],
                                     bounds[3]};
  Array4D<float> input(bounds[0], bounds[1], bounds[2], bounds[3]);
  input.Each([&rng, &distribution](absl::Span<const int64_t> /* indices */,
                                   float* cell) { *cell = distribution(rng); });
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input, LayoutUtil::MakeLayout({0, 1, 2, 3}));
  XlaBuilder builder(TestName());
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(auto input_data,
                          CreateParameterAndTransferLiteral(
                              0, input_literal, "input", &builder, &parameter));
  Reshape(parameter, /*dimensions=*/{1, 0, 2, 3},
          /*new_sizes=*/new_bounds);

  Literal expected =
      LiteralUtil::ReshapeSlice(new_bounds, {1, 0, 2, 3}, input_literal)
          .Relayout(input_literal.shape().layout());

  // Specify the requested output shape explicitly to ensure that this reshape
  // actually corresponds to a two minor transpose.
  ComputeAndCompareLiteral(&builder, expected, {input_data.get()},
                           zero_error_spec_, &expected.shape());
}

#ifdef XLA_BACKEND_SUPPORTS_BFLOAT16
INSTANTIATE_TEST_CASE_P(ReshapeTestInstance, ReshapeTest, ::testing::Bool());
#else
INSTANTIATE_TEST_CASE_P(ReshapeTestInstance, ReshapeTest,
                        ::testing::ValuesIn(std::vector<bool>{false}));
#endif

}  // namespace
}  // namespace xla
