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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStranspose_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStranspose_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStranspose_testDTcc() {
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

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class TransposeTest : public ClientLibraryTestBase {
 public:
  ErrorSpec error_spec_{0.0001};

 protected:
  void TestTransposeConstant021(size_t n1, size_t n2, size_t n3);
};

XLA_TEST_F(TransposeTest, Transpose0x0) {
  XlaBuilder builder("Transpose");
  auto lhs = ConstantR2FromArray2D<float>(&builder, Array2D<float>(0, 0));
  Transpose(lhs, {1, 0});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 0), {}, error_spec_);
}

XLA_TEST_F(TransposeTest, Transpose0x42) {
  XlaBuilder builder("Transpose");
  auto lhs = ConstantR2FromArray2D<float>(&builder, Array2D<float>(0, 42));
  Transpose(lhs, {1, 0});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(42, 0), {}, error_spec_);
}

XLA_TEST_F(TransposeTest, Transpose7x0) {
  XlaBuilder builder("Transpose");
  auto lhs = ConstantR2FromArray2D<float>(&builder, Array2D<float>(7, 0));
  Transpose(lhs, {1, 0});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 7), {}, error_spec_);
}

TEST_F(TransposeTest, Transpose2x2) {
  XlaBuilder builder("Transpose");
  auto lhs = ConstantR2<float>(&builder, {
                                             {1.0, 2.0},
                                             {3.0, 4.0},
                                         });
  Transpose(lhs, {1, 0});

  Array2D<float> expected({{1.0f, 3.0f}, {2.0f, 4.0f}});

  ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(TransposeTest, Transpose0x2x3_2x3x0) {
  XlaBuilder builder("Transpose");
  auto operand =
      ConstantR3FromArray3D<int32_t>(&builder, Array3D<int32_t>(0, 2, 3));
  Transpose(operand, {1, 2, 0});

  ComputeAndCompareR3<int32_t>(&builder, Array3D<int32_t>(2, 3, 0), {});
}

TEST_F(TransposeTest, Transpose1x2x3_2x3x1) {
  XlaBuilder builder("Transpose");
  auto operand =
      ConstantR3FromArray3D<int32_t>(&builder, {{{1, 2, 3}, {4, 5, 6}}});
  Transpose(operand, {1, 2, 0});

  Array3D<int32_t> expected({{{1}, {2}, {3}}, {{4}, {5}, {6}}});

  ComputeAndCompareR3<int32_t>(&builder, expected, {});
}

TEST_F(TransposeTest, Transpose1x2x3_3x2x1) {
  XlaBuilder builder("Transpose");
  auto operand =
      ConstantR3FromArray3D<int32_t>(&builder, {{{1, 2, 3}, {4, 5, 6}}});
  Transpose(operand, {2, 1, 0});

  Array3D<int32_t> expected({{{1}, {4}}, {{2}, {5}}, {{3}, {6}}});

  ComputeAndCompareR3<int32_t>(&builder, expected, {});
}

TEST_F(TransposeTest, Transpose1x2x3_1x2x3) {
  XlaBuilder builder("Transpose");
  auto operand =
      ConstantR3FromArray3D<int32_t>(&builder, {{{1, 2, 3}, {4, 5, 6}}});
  Transpose(operand, {0, 1, 2});

  Array3D<int32_t> expected({{{1, 2, 3}, {4, 5, 6}}});

  ComputeAndCompareR3<int32_t>(&builder, expected, {});
}

TEST_F(TransposeTest, MultiTranspose3x2) {
  Array2D<float> input({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
  Array2D<float> transposed({{1.0f, 3.0f, 5.0f}, {2.0f, 4.0f, 6.0f}});

  for (int transposes = 0; transposes <= 10; ++transposes) {
    XlaBuilder builder("Transpose");
    auto computed = ConstantR2FromArray2D<float>(&builder, input);
    for (int i = 0; i < transposes; ++i) {
      computed = Transpose(computed, {1, 0});
    }
    const Array2D<float>& expected = transposes % 2 == 0 ? input : transposed;
    ComputeAndCompareR2<float>(&builder, expected, {}, error_spec_);
  }
}

// Test for transposing [1x1] matrix.
TEST_F(TransposeTest, Small_1x1) {
  auto aoperand = MakeLinspaceArray2D(0.0, 1.0, 1, 1);

  XlaBuilder builder("transpose_1x1");
  auto operand = ConstantR2FromArray2D<float>(&builder, *aoperand);
  Transpose(operand, {1, 0});

  auto expected = ReferenceUtil::TransposeArray2D(*aoperand);
  ComputeAndCompareR2<float>(&builder, *expected, {}, ErrorSpec(1e-4));
}

// Test for transposing [2x2] matrix.
TEST_F(TransposeTest, Small_2x2) {
  auto aoperand = MakeLinspaceArray2D(0.0, 4.0, 2, 2);

  XlaBuilder builder("transpose_2x2");
  auto operand = ConstantR2FromArray2D<float>(&builder, *aoperand);
  Transpose(operand, {1, 0});

  auto expected = ReferenceUtil::TransposeArray2D(*aoperand);
  ComputeAndCompareR2<float>(&builder, *expected, {}, ErrorSpec(1e-4));
}

void TransposeTest::TestTransposeConstant021(size_t n1, size_t n2, size_t n3) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPStranspose_testDTcc mht_0(mht_0_v, 326, "", "./tensorflow/compiler/xla/tests/transpose_test.cc", "TransposeTest::TestTransposeConstant021");

  Array3D<int32_t> aoperand(n1, n2, n3);
  Array3D<int32_t> expected(n1, n3, n2);
  for (size_t i = 0; i < n1; ++i) {
    for (size_t j = 0; j < n2; ++j) {
      for (size_t k = 0; k < n3; ++k) {
        aoperand(i, j, k) = i * n3 * n2 + j * n3 + k;
        expected(i, k, j) = aoperand(i, j, k);
      }
    }
  }

  XlaBuilder builder(TestName());
  auto operand = ConstantR3FromArray3D(&builder, aoperand);
  Transpose(operand, {0, 2, 1});

  ComputeAndCompareR3<int32_t>(&builder, expected, {});
}

TEST_F(TransposeTest, TransposeConstant021_SingleIncompleteTilePerLayer) {
  TestTransposeConstant021(2, 2, 3);
}

TEST_F(TransposeTest, TransposeConstant021_SingleCompleteTilePerLayer) {
  TestTransposeConstant021(2, 32, 32);
}

TEST_F(TransposeTest, TransposeConstant021_MultipleTilesPerLayer) {
  TestTransposeConstant021(2, 70, 35);
}

}  // namespace
}  // namespace xla
