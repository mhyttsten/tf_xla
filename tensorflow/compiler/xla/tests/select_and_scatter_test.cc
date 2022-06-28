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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSselect_and_scatter_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSselect_and_scatter_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSselect_and_scatter_testDTcc() {
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

// Tests the select-and-scatter XLA operation.

// b/194424657: On macs, the compiler hangs when trying to compile this file
#if !defined(__APPLE__)

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

struct SelectAndScatterTestParam {
  std::vector<int64_t> operand_shape;
  std::vector<int64_t> source_shape;
  Padding padding_type;
  std::vector<int64_t> window_dimensions;
  std::vector<int64_t> window_strides;
};

class SelectAndScatterTest
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<SelectAndScatterTestParam> {
 public:
  SelectAndScatterTest() : builder_(TestName()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSselect_and_scatter_testDTcc mht_0(mht_0_v, 224, "", "./tensorflow/compiler/xla/tests/select_and_scatter_test.cc", "SelectAndScatterTest");

    // Create S32 GE and ADD computations for select and scatter respectively.
    ge_s32_ = CreateScalarGeComputation(S32, &builder_);
    add_s32_ = CreateScalarAddComputation(S32, &builder_);
    ge_f32_ = CreateScalarGeComputation(F32, &builder_);
    add_f32_ = CreateScalarAddComputation(F32, &builder_);
    max_f32_ = CreateScalarMaxComputation(F32, &builder_);
    min_f32_ = CreateScalarMinComputation(F32, &builder_);
  }

  XlaBuilder builder_;
  XlaComputation ge_s32_;
  XlaComputation add_s32_;
  XlaComputation ge_f32_;
  XlaComputation add_f32_;
  XlaComputation max_f32_;
  XlaComputation min_f32_;
};

XLA_TEST_P(SelectAndScatterTest, ParamTest) {
  auto operand_shape = GetParam().operand_shape;
  Array<float> o(operand_shape);
  o.FillRandom(1.5f);
  auto operand = ConstantFromArray(&builder_, o);

  auto source_shape = GetParam().source_shape;
  Array<float> s(source_shape);
  s.FillRandom(12.0f);
  auto source = ConstantFromArray(&builder_, s);

  SelectAndScatter(operand, ge_f32_, GetParam().window_dimensions,
                   GetParam().window_strides, GetParam().padding_type, source,
                   ConstantR0<float>(&builder_, 0.0f), add_f32_);

  ComputeAndCompare(&builder_, {}, ErrorSpec(1e-5, 1e-5));
}

INSTANTIATE_TEST_CASE_P(
    SelectAndScatterTest_Instantiation, SelectAndScatterTest,
    ::testing::Values(
        SelectAndScatterTestParam{{6, 6, 6, 4, 4},
                                  {3, 3, 3, 4, 4},
                                  Padding::kSame,
                                  {3, 3, 3, 1, 1},
                                  {2, 2, 2, 1, 1}},
        SelectAndScatterTestParam{{7, 7, 7, 4, 4},
                                  {3, 3, 3, 4, 4},
                                  Padding::kValid,
                                  {3, 3, 3, 1, 1},
                                  {2, 2, 2, 1, 1}},

        SelectAndScatterTestParam{{8, 8, 8, 4, 4},
                                  {1, 3, 3, 4, 4},
                                  Padding::kValid,
                                  {8, 4, 4, 1, 1},
                                  {1, 2, 2, 1, 1}},
        SelectAndScatterTestParam{{6, 6, 256, 128},
                                  {3, 3, 256, 128},
                                  Padding::kSame,
                                  {3, 3, 1, 1},
                                  {2, 2, 1, 1}},
        SelectAndScatterTestParam{{7, 7, 256, 128},
                                  {3, 3, 256, 128},
                                  Padding::kValid,
                                  {3, 3, 1, 1},
                                  {2, 2, 1, 1}},
        SelectAndScatterTestParam{{6, 7, 256, 128},
                                  {3, 3, 256, 128},
                                  Padding::kValid,
                                  {2, 3, 1, 1},
                                  {2, 2, 1, 1}},
        SelectAndScatterTestParam{{6, 7, 256, 128},
                                  {2, 3, 256, 128},
                                  Padding::kValid,
                                  {2, 3, 1, 1},
                                  {3, 2, 1, 1}},
        SelectAndScatterTestParam{{9, 9, 16, 128},
                                  {3, 3, 16, 128},
                                  Padding::kValid,
                                  {3, 3, 1, 1},
                                  {3, 3, 1, 1}},
        SelectAndScatterTestParam{{3, 3, 4, 4},
                                  {1, 1, 4, 4},
                                  Padding::kValid,
                                  {3, 3, 1, 1},
                                  {3, 3, 1, 1}},
        SelectAndScatterTestParam{{3, 3, 4, 4},
                                  {1, 1, 4, 4},
                                  Padding::kValid,
                                  {3, 3, 1, 1},
                                  {3, 3, 1, 1}},
        SelectAndScatterTestParam{{9, 3, 4, 4},
                                  {3, 1, 4, 4},
                                  Padding::kValid,
                                  {3, 3, 1, 1},
                                  {3, 3, 1, 1}},
        // Uncovered by b/126212776.
        SelectAndScatterTestParam{{15, 1, 1, 1},
                                  {2, 1, 1, 1},
                                  Padding::kValid,
                                  {14, 1, 1, 1},
                                  {1, 1, 1, 1}},
        SelectAndScatterTestParam{{7, 3, 4, 4},
                                  {3, 1, 4, 4},
                                  Padding::kValid,
                                  {3, 3, 1, 1},
                                  {2, 3, 1, 1}},
        SelectAndScatterTestParam{{1, 1, 5, 5},
                                  {1, 1, 5, 5},
                                  Padding::kSame,
                                  {3, 3, 1, 1},
                                  {3, 3, 1, 1}},
        SelectAndScatterTestParam{{7, 7, 8, 256},
                                  {4, 4, 8, 256},
                                  Padding::kSame,
                                  {2, 2, 1, 1},
                                  {2, 2, 1, 1}},
        SelectAndScatterTestParam{
            {6, 4, 4}, {3, 4, 4}, Padding::kSame, {3, 1, 1}, {2, 1, 1}},
        SelectAndScatterTestParam{
            {6, 256, 128}, {3, 256, 128}, Padding::kSame, {3, 1, 1}, {2, 1, 1}},
        SelectAndScatterTestParam{{7, 256, 128},
                                  {3, 256, 128},
                                  Padding::kValid,
                                  {3, 1, 1},
                                  {2, 1, 1}},
        SelectAndScatterTestParam{{6, 256, 128},
                                  {3, 256, 128},
                                  Padding::kValid,
                                  {2, 1, 1},
                                  {2, 1, 1}},
        SelectAndScatterTestParam{{6, 256, 128},
                                  {2, 256, 128},
                                  Padding::kValid,
                                  {2, 1, 1},
                                  {3, 1, 1}},
        SelectAndScatterTestParam{{10, 10, 8, 256},
                                  {5, 5, 8, 256},
                                  Padding::kSame,
                                  {2, 2, 1, 1},
                                  {2, 2, 1, 1}},
        SelectAndScatterTestParam{
            {9, 16, 128}, {3, 16, 128}, Padding::kValid, {3, 1, 1}, {3, 1, 1}},
        SelectAndScatterTestParam{
            {3, 4, 4}, {1, 4, 4}, Padding::kValid, {3, 1, 1}, {3, 1, 1}},
        SelectAndScatterTestParam{
            {3, 4, 4}, {1, 4, 4}, Padding::kValid, {3, 1, 1}, {3, 1, 1}},
        SelectAndScatterTestParam{
            {9, 4, 4}, {3, 4, 4}, Padding::kValid, {3, 1, 1}, {3, 1, 1}},
        SelectAndScatterTestParam{
            {7, 4, 4}, {3, 4, 4}, Padding::kValid, {3, 1, 1}, {2, 1, 1}},
        SelectAndScatterTestParam{
            {1, 5, 5}, {1, 5, 5}, Padding::kSame, {3, 1, 1}, {3, 1, 1}},
        SelectAndScatterTestParam{
            {7, 8, 256}, {4, 8, 256}, Padding::kSame, {2, 1, 1}, {2, 1, 1}},
        SelectAndScatterTestParam{{1104}, {551}, Padding::kValid, {3}, {2}},
        SelectAndScatterTestParam{{1300}, {1171}, Padding::kValid, {130}, {1}},
        SelectAndScatterTestParam{{3000}, {1701}, Padding::kValid, {1300}, {1}},
        SelectAndScatterTestParam{{6500}, {5}, Padding::kValid, {1300}, {1300}},
        SelectAndScatterTestParam{
            {3000}, {401}, Padding::kValid, {2600}, {1}}));

// Test for F32 1D array, with a zero-element input.
XLA_TEST_F(SelectAndScatterTest, R1S0F32) {
  const auto operand = ConstantR1<float>(&builder_, {});
  const auto source = ConstantR1<float>(&builder_, {});
  SelectAndScatter(operand, ge_f32_, /*window_dimensions=*/{3},
                   /*window_strides=*/{3}, Padding::kValid, source,
                   ConstantR0<float>(&builder_, 0.0f), add_f32_);
  ComputeAndCompareR1<float>(&builder_, {}, {}, ErrorSpec(1e-7));
}

// Test for F32 1D array, when windows do not overlap.
XLA_TEST_F(SelectAndScatterTest, R1F32) {
  const auto operand =
      ConstantR1<float>(&builder_, {1.f, 9.f, 3.f, 7.f, 5.f, 6.f});
  const auto source = ConstantR1<float>(&builder_, {34.f, 42.f});
  const std::vector<float> expected = {0.f, 34.f, 0.f, 42.f, 0.f, 0.f};
  SelectAndScatter(operand, ge_f32_, /*window_dimensions=*/{3},
                   /*window_strides=*/{3}, Padding::kValid, source,
                   ConstantR0<float>(&builder_, 0.0f), add_f32_);
  ComputeAndCompareR1<float>(&builder_, expected, {}, ErrorSpec(1e-7));
}

// Test for S32 1D array, when windows do not overlap and the init value is 1.
XLA_TEST_F(SelectAndScatterTest, R1S32) {
  const auto operand = ConstantR1<int32_t>(&builder_, {-1, 0, 6, 4, -4, 10});
  const auto source = ConstantR1<int32_t>(&builder_, {-10, 20});
  const std::vector<int32_t> expected = {1, 1, -9, 1, 1, 21};
  SelectAndScatter(operand, ge_s32_, /*window_dimensions=*/{3},
                   /*window_strides=*/{3}, Padding::kValid, source,
                   ConstantR0<int32_t>(&builder_, 1), add_s32_);
  ComputeAndCompareR1<int32_t>(&builder_, expected, {});
}

// Test for S32 1D array, when windows overlap with each other.
XLA_TEST_F(SelectAndScatterTest, R1S32OverlappingWindow) {
  const auto operand = ConstantR1<int32_t>(&builder_, {1, 9, 3, 7, 5, 6});
  const auto source = ConstantR1<int32_t>(&builder_, {34, 42, 53, 19});
  const std::vector<int32_t> expected = {0, 76, 0, 72, 0, 0};
  SelectAndScatter(operand, ge_s32_, /*window_dimensions=*/{3},
                   /*window_strides=*/{1}, Padding::kValid, source,
                   ConstantR0<int32_t>(&builder_, 0), add_s32_);
  ComputeAndCompareR1<int32_t>(&builder_, expected, {});
}

// Test for S32 2D array, when windows do not overlap.
XLA_TEST_F(SelectAndScatterTest, R2S32) {
  const auto operand =
      ConstantR2<int32_t>(&builder_, {{7, 2, 5, 3, 10, 2}, {3, 8, 9, 3, 4, 2}});
  const auto source = ConstantR2<int32_t>(&builder_, {{2, 6}});
  Array2D<int32_t> expected({{0, 0, 0, 0, 6, 0}, {0, 0, 2, 0, 0, 0}});
  SelectAndScatter(operand, ge_s32_, /*window_dimensions=*/{2, 3},
                   /*window_strides=*/{2, 3}, Padding::kValid, source,
                   ConstantR0<int32_t>(&builder_, 0), add_s32_);
  ComputeAndCompareR2<int32_t>(&builder_, expected, {});
}

// Test for tie breaking rule in ge_f32_. When a tie is present, the operand
// that has the lower lexicographical order (smaller index) should be chosen.
XLA_TEST_F(SelectAndScatterTest, R2F32Tie) {
  const auto operand = ConstantR2<float>(
      &builder_, {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}});
  const auto source = ConstantR2<float>(
      &builder_, {{1.0f, 2.0f, 3.0f}, {4.f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}});
  Array2D<float> expected(
      {{12.f, 9.f, 0.f}, {15.f, 9.f, 0.f}, {0.f, 0.f, 0.f}});
  SelectAndScatter(operand, ge_f32_, /*window_dimensions=*/{3, 3},
                   /*window_strides=*/{1, 1}, Padding::kSame, source,
                   ConstantR0<float>(&builder_, 0.0f), add_f32_);
  ComputeAndCompareR2<float>(&builder_, expected, {}, ErrorSpec(1e-7));
}

// Similar to SelectAndScatterTest.R2S32 but the input is transposed.
XLA_TEST_F(SelectAndScatterTest, ReshapeR2S32) {
  const auto operand = ConstantR2<int32_t>(
      &builder_, {{7, 3}, {2, 8}, {5, 9}, {3, 3}, {10, 4}, {2, 2}});
  const auto reshape =
      Reshape(operand, /*dimensions=*/{1, 0}, /*new_sizes=*/{2, 6});
  const auto source = ConstantR2<int32_t>(&builder_, {{2, 6}});
  Array2D<int32_t> expected({{0, 0, 0, 0, 6, 0}, {0, 0, 2, 0, 0, 0}});
  SelectAndScatter(reshape, ge_s32_, /*window_dimensions=*/{2, 3},
                   /*window_strides=*/{2, 3}, Padding::kValid, source,
                   ConstantR0<int32_t>(&builder_, 0), add_s32_);
  ComputeAndCompareR2<int32_t>(&builder_, expected, {});
}

// Test for S32 2D array, when windows overlap with each other.
XLA_TEST_F(SelectAndScatterTest, R2S32OverlappingWindow) {
  const auto operand =
      ConstantR2<int32_t>(&builder_, {{7, 2, 5, 3, 8}, {3, 8, 9, 3, 4}});
  const auto source = ConstantR2<int32_t>(&builder_, {{2, 6, 4}});
  Array2D<int32_t> expected({{0, 0, 0, 0, 0}, {0, 0, 12, 0, 0}});
  SelectAndScatter(operand, ge_s32_, /*window_dimensions=*/{2, 3},
                   /*window_strides=*/{1, 1}, Padding::kValid, source,
                   ConstantR0<int32_t>(&builder_, 0), add_s32_);
  ComputeAndCompareR2<int32_t>(&builder_, expected, {});
}

// Test for S32 2D array, when the padding is Padding::kSAME.
XLA_TEST_F(SelectAndScatterTest, R2S32SamePadding) {
  const auto operand =
      ConstantR2<int32_t>(&builder_, {{7, 2, 5, 3, 8}, {3, 8, 9, 3, 4}});
  const auto source = ConstantR2<int32_t>(&builder_, {{2, 6, 4}});
  Array2D<int32_t> expected({{0, 0, 0, 0, 4}, {0, 2, 6, 0, 0}});
  SelectAndScatter(operand, ge_s32_, /*window_dimensions=*/{2, 2},
                   /*window_strides=*/{2, 2}, Padding::kSame, source,
                   ConstantR0<int32_t>(&builder_, 0), add_s32_);
  ComputeAndCompareR2<int32_t>(&builder_, expected, {});
}

// Test for S32 2D array, when the padding is Padding::kSAME and windows overlap
// with each other.
XLA_TEST_F(SelectAndScatterTest, R2S32SamePaddingOverlappingWindow) {
  const auto operand =
      ConstantR2<int32_t>(&builder_, {{7, 2, 5, 3, 8}, {3, 8, 9, 3, 4}});
  const auto source =
      ConstantR2<int32_t>(&builder_, {{2, 6, 4, 7, 1}, {3, 5, 8, 9, 10}});
  Array2D<int32_t> expected({{0, 0, 0, 0, 8}, {0, 5, 23, 0, 19}});
  SelectAndScatter(operand, ge_s32_, /*window_dimensions=*/{2, 2},
                   /*window_strides=*/{1, 1}, Padding::kSame, source,
                   ConstantR0<int32_t>(&builder_, 0), add_s32_);
  ComputeAndCompareR2<int32_t>(&builder_, expected, {});
}

XLA_TEST_F(SelectAndScatterTest, R2F32OverlappingR2Source) {
  const auto operand = ConstantR2<float>(
      &builder_, {{1.5f, 2.5f, 1.5f}, {3.5f, 1.5f, 3.5f}, {4.5f, 2.5f, 4.5f}});
  const auto source =
      ConstantR2<float>(&builder_, {{1.0f, 2.0f}, {3.0f, 4.0f}});
  Array2D<float> expected(
      {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 2.0f}, {3.0f, 0.0f, 4.0f}});
  SelectAndScatter(operand, ge_f32_, /*window_dimensions=*/{2, 2},
                   /*window_strides=*/{1, 1}, Padding::kValid, source,
                   ConstantR0<float>(&builder_, 0.0f), add_f32_);
  ComputeAndCompareR2<float>(&builder_, expected, {}, ErrorSpec(1e-7));
}

TEST_F(SelectAndScatterTest, R4F32Valid) {
  Array2D<float> pzo = {{7.0f, 2.0f, 5.0f, 3.0f, 10.0f, 2.0f},
                        {3.0f, 8.0f, 9.0f, 3.0f, 4.00f, 2.0f},
                        {1.0f, 5.0f, 7.0f, 5.0f, 6.00f, 1.0f},
                        {0.0f, 6.0f, 2.0f, 7.0f, 2.00f, 8.0f}};
  Array2D<float> pzs = {{2.0f, 6.0f}, {3.0f, 1.0f}};
  Array2D<float> pze = {{0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 0.0f},
                        {0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}};
  Array4D<float> o(4, 6, 15, 220);
  o.FillWithPZ(pzo);
  auto operand = ConstantR4FromArray4D(&builder_, o);
  Array4D<float> e(4, 6, 15, 220);
  e.FillWithPZ(pze);
  Array4D<float> s(2, 2, 15, 220);
  s.FillWithPZ(pzs);
  auto source = ConstantR4FromArray4D(&builder_, s);
  s.FillWithPZ(pzs);
  SelectAndScatter(operand, ge_f32_, {2, 3, 1, 1}, {2, 3, 1, 1},
                   Padding::kValid, source, ConstantR0<float>(&builder_, 0.0f),
                   add_f32_);
  ComputeAndCompareR4<float>(&builder_, e, {}, ErrorSpec(1e-7));
}

TEST_F(SelectAndScatterTest, R4F32Overlap) {
  Array2D<float> pzo = {{7.0f, 2.0f, 5.0f, 3.0f, 8.0f},
                        {3.0f, 8.0f, 9.0f, 3.0f, 4.0f},
                        {1.0f, 5.0f, 7.0f, 5.0f, 6.0f},
                        {0.0f, 6.0f, 2.0f, 10.0f, 2.0f}};
  Array2D<float> pzs = {{2.0f, 6.0f}, {3.0f, 1.0f}};
  Array2D<float> pze = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 8.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 3.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 0.0f, 1.0f, 0.0f}};
  Array4D<float> o(4, 5, 17, 128);
  o.FillWithPZ(pzo);
  auto operand = ConstantR4FromArray4D(&builder_, o);
  Array4D<float> e(4, 5, 17, 128);
  e.FillWithPZ(pze);
  Array4D<float> s(2, 2, 17, 128);
  s.FillWithPZ(pzs);
  auto source = ConstantR4FromArray4D(&builder_, s);
  s.FillWithPZ(pzs);
  SelectAndScatter(operand, ge_f32_, {2, 3, 1, 1}, {2, 2, 1, 1},
                   Padding::kValid, source, ConstantR0<float>(&builder_, 0.0f),
                   add_f32_);
  ComputeAndCompareR4<float>(&builder_, e, {}, ErrorSpec(1e-7));
}

TEST_F(SelectAndScatterTest, R4F32OverlapSmall) {
  Array2D<float> pzo = {{7.0f, 2.0f, 5.0f, 3.0f, 8.0f},
                        {3.0f, 8.0f, 9.0f, 3.0f, 4.0f},
                        {1.0f, 5.0f, 7.0f, 5.0f, 6.0f},
                        {0.0f, 6.0f, 2.0f, 10.0f, 2.0f}};
  Array2D<float> pzs = {{2.0f, 6.0f}, {3.0f, 1.0f}};
  Array2D<float> pze = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 8.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 3.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 0.0f, 1.0f, 0.0f}};
  Array4D<float> o(4, 5, 1, 1);
  o.FillWithPZ(pzo);
  auto operand = ConstantR4FromArray4D(&builder_, o);
  Array4D<float> e(4, 5, 1, 1);
  e.FillWithPZ(pze);
  Array4D<float> s(2, 2, 1, 1);
  s.FillWithPZ(pzs);
  auto source = ConstantR4FromArray4D(&builder_, s);
  s.FillWithPZ(pzs);
  SelectAndScatter(operand, ge_f32_, {2, 3, 1, 1}, {2, 2, 1, 1},
                   Padding::kValid, source, ConstantR0<float>(&builder_, 0.0f),
                   add_f32_);
  ComputeAndCompareR4<float>(&builder_, e, {}, ErrorSpec(1e-7));
}

TEST_F(SelectAndScatterTest, R4F32RefValidFixedSmall) {
  // This test is testing the Reference Util
  Array2D<float> pzo = {{7.0f, 2.0f, 5.0f, 3.0f, 10.0f, 2.0f},
                        {3.0f, 8.0f, 9.0f, 3.0f, 4.00f, 2.0f},
                        {1.0f, 5.0f, 7.0f, 5.0f, 6.00f, 1.0f},
                        {0.0f, 6.0f, 2.0f, 7.0f, 2.00f, 8.0f}};
  Array2D<float> pzs = {{2.0f, 6.0f}, {3.0f, 1.0f}};
  Array4D<float> o(4, 6, 4, 4);
  o.FillWithPZ(pzo);
  auto operand = ConstantR4FromArray4D(&builder_, o);
  Array4D<float> s(2, 2, 4, 4);
  s.FillWithPZ(pzs);

  auto source = ConstantR4FromArray4D(&builder_, s);
  s.FillWithPZ(pzs);
  SelectAndScatter(operand, ge_f32_, {2, 3, 1, 1}, {2, 3, 1, 1},
                   Padding::kValid, source, ConstantR0<float>(&builder_, 0.0f),
                   add_f32_);
  auto e = ReferenceUtil::SelectAndScatter4DGePlus(o, s, 0.0f, {2, 3, 1, 1},
                                                   {2, 3, 1, 1}, false);
  ComputeAndCompareR4<float>(&builder_, *e, {}, ErrorSpec(1e-7));
}

// Test for F32 4D array with negative padding on both ends.
XLA_TEST_F(SelectAndScatterTest, R4NegativePaddingOnBothEnds) {
  Array2D<float> pzo = {{7.0f, 2.0f, 5.0f, 3.0f, 10.0f, 3.0f},
                        {3.0f, 8.0f, 9.0f, 3.0f, 4.00f, 2.0f},
                        {1.0f, 5.0f, 7.0f, 5.0f, 6.00f, 1.0f},
                        {0.0f, 6.0f, 2.0f, 7.0f, 2.00f, 8.0f}};
  Array2D<float> pzs = {{2.0f, 6.0f}, {3.0f, 1.0f}};
  Array2D<float> pze = {{0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 0.0f},
                        {0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}};
  Array4D<float> o(4, 6, 4, 4);
  o.FillWithPZ(pzo);
  auto operand = ConstantR4FromArray4D(&builder_, o);
  Array4D<float> e(4, 6, 4, 4);
  e.FillWithPZ(pze);
  Array4D<float> s(2, 2, 4, 4);
  s.FillWithPZ(pzs);
  auto source = ConstantR4FromArray4D(&builder_, s);
  s.FillWithPZ(pzs);
  SelectAndScatterWithGeneralPadding(
      operand, ge_f32_, {2, 2, 1, 1}, {2, 2, 1, 1},
      {{0, 0}, {-1, -1}, {0, 0}, {0, 0}}, source,
      ConstantR0<float>(&builder_, 0.0f), add_f32_);
  ComputeAndCompareR4<float>(&builder_, e, {}, ErrorSpec(1e-7));
}

// Test for F32 4D array with positive low padding and negative high padding.
XLA_TEST_F(SelectAndScatterTest, R4PositivePaddingLowAndNegativePaddingHigh) {
  Array2D<float> pzo = {{7.0f, 2.0f, 5.0f, 3.0f, 10.0f, 3.0f},
                        {3.0f, 8.0f, 9.0f, 3.0f, 4.00f, 2.0f},
                        {1.0f, 5.0f, 7.0f, 5.0f, 6.00f, 1.0f},
                        {0.0f, 6.0f, 2.0f, 7.0f, 2.00f, 8.0f}};
  Array2D<float> pzs = {{2.0f, 6.0f, 4.0f}, {3.0f, 1.0f, 5.0f}};
  Array2D<float> pze = {{2.0f, 0.0f, 0.0f, 0.0f, 4.0f, 0.0f},
                        {0.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f},
                        {3.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f}};
  Array4D<float> o(4, 6, 4, 4);
  o.FillWithPZ(pzo);
  auto operand = ConstantR4FromArray4D(&builder_, o);
  Array4D<float> e(4, 6, 4, 4);
  e.FillWithPZ(pze);
  Array4D<float> s(2, 3, 4, 4);
  s.FillWithPZ(pzs);
  auto source = ConstantR4FromArray4D(&builder_, s);
  s.FillWithPZ(pzs);
  SelectAndScatterWithGeneralPadding(
      operand, ge_f32_, {2, 2, 1, 1}, {2, 2, 1, 1},
      {{0, 0}, {1, -1}, {0, 0}, {0, 0}}, source,
      ConstantR0<float>(&builder_, 0.0f), add_f32_);
  ComputeAndCompareR4<float>(&builder_, e, {}, ErrorSpec(1e-7));
}

// Test for F32 4D array with negative low padding and positive high padding.
XLA_TEST_F(SelectAndScatterTest, R4NegativePaddingLowAndPositivePaddingHigh) {
  Array2D<float> pzo = {{7.0f, 2.0f, 5.0f, 3.0f, 10.0f, 3.0f},
                        {3.0f, 8.0f, 9.0f, 3.0f, 4.00f, 2.0f},
                        {1.0f, 5.0f, 7.0f, 5.0f, 6.00f, 1.0f},
                        {0.0f, 6.0f, 2.0f, 7.0f, 2.00f, 8.0f}};
  Array2D<float> pzs = {{2.0f, 6.0f, 4.0f}, {3.0f, 1.0f, 5.0f}};
  Array2D<float> pze = {{0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 4.0f},
                        {0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 5.0f}};
  Array4D<float> o(4, 6, 4, 4);
  o.FillWithPZ(pzo);
  auto operand = ConstantR4FromArray4D(&builder_, o);
  Array4D<float> e(4, 6, 4, 4);
  e.FillWithPZ(pze);
  Array4D<float> s(2, 3, 4, 4);
  s.FillWithPZ(pzs);
  auto source = ConstantR4FromArray4D(&builder_, s);
  s.FillWithPZ(pzs);
  SelectAndScatterWithGeneralPadding(
      operand, ge_f32_, {2, 2, 1, 1}, {2, 2, 1, 1},
      {{0, 0}, {-1, 1}, {0, 0}, {0, 0}}, source,
      ConstantR0<float>(&builder_, 0.0f), add_f32_);
  ComputeAndCompareR4<float>(&builder_, e, {}, ErrorSpec(1e-7));
}

XLA_TEST_F(SelectAndScatterTest, R1F32OverlappingWindowMaxScatter) {
  const auto operand = ConstantR1<float>(&builder_, {1, 2, 3, 100, 3, 2, 1});
  const auto source = ConstantR1<float>(&builder_, {34, 42, 53, 19});
  const std::vector<float> expected = {0, 0, 0, 53, 0, 0, 0};
  SelectAndScatter(operand, ge_f32_, /*window_dimensions=*/{4},
                   /*window_strides=*/{1}, Padding::kValid, source,
                   ConstantR0<float>(&builder_, 0), max_f32_);
  ComputeAndCompareR1<float>(&builder_, expected, {}, ErrorSpec(1e-7));
}

XLA_TEST_F(SelectAndScatterTest, R1F32OverlappingWindowMinScatter) {
  const auto operand = ConstantR1<float>(&builder_, {1, 2, 3, 100, 3, 2, 1});
  const auto source = ConstantR1<float>(&builder_, {34, 42, 53, 19});
  const float max_float = std::numeric_limits<float>::max();
  const std::vector<float> expected = {max_float, max_float, max_float, 19,
                                       max_float, max_float, max_float};
  SelectAndScatter(operand, ge_f32_, /*window_dimensions=*/{4},
                   /*window_strides=*/{1}, Padding::kValid, source,
                   ConstantR0<float>(&builder_, max_float), min_f32_);
  ComputeAndCompareR1<float>(&builder_, expected, {}, ErrorSpec(1e-7));
}

}  // namespace
}  // namespace xla

#endif  // !defined(__APPLE__)
