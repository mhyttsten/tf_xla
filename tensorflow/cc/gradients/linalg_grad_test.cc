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
class MHTracer_DTPStensorflowPSccPSgradientsPSlinalg_grad_testDTcc {
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
   MHTracer_DTPStensorflowPSccPSgradientsPSlinalg_grad_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSgradientsPSlinalg_grad_testDTcc() {
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

#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradient_checker.h"
#include "tensorflow/cc/framework/testutil.h"
#include "tensorflow/cc/gradients/grad_testutil.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

using tensorflow::ops::Einsum;
using tensorflow::ops::Placeholder;

class LinalgGradTest : public ::testing::Test {
 protected:
  LinalgGradTest() : scope_(Scope::NewRootScope()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSlinalg_grad_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/cc/gradients/linalg_grad_test.cc", "LinalgGradTest");
}

  void RunTest(const Output& x, const TensorShape& x_shape, const Output& y,
               const TensorShape& y_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSlinalg_grad_testDTcc mht_1(mht_1_v, 208, "", "./tensorflow/cc/gradients/linalg_grad_test.cc", "RunTest");

    TF_ASSERT_OK(scope_.status());
    float max_error;
    TF_ASSERT_OK((ComputeGradientError<float, float, float>(
        scope_, {x}, {x_shape}, {y}, {y_shape}, &max_error)));
    EXPECT_LT(max_error, 1e-3);
  }

  void RunTest(const OutputList& xs, const std::vector<TensorShape>& x_shapes,
               const OutputList& ys, const std::vector<TensorShape>& y_shapes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSlinalg_grad_testDTcc mht_2(mht_2_v, 220, "", "./tensorflow/cc/gradients/linalg_grad_test.cc", "RunTest");

    TF_ASSERT_OK(scope_.status());
    float max_error;
    TF_ASSERT_OK((ComputeGradientError<float, float, float>(
        scope_, xs, x_shapes, ys, y_shapes, &max_error)));
    EXPECT_LT(max_error, 1e-3);
  }

  Scope scope_;
};

TEST_F(LinalgGradTest, Einsum_Transpose) {
  TensorShape x_shape({2, 3});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = Einsum(scope_, {x}, "ij->ji");
  TensorShape y_shape({3, 2});
  RunTest({x}, {x_shape}, {y}, {y_shape});
}

TEST_F(LinalgGradTest, Einsum_TransposeBroadcast) {
  TensorShape x_shape({3, 2, 3});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = Einsum(scope_, {x}, "...ij->...ji");
  TensorShape y_shape({3, 3, 2});
  RunTest({x}, {x_shape}, {y}, {y_shape});
}

TEST_F(LinalgGradTest, Einsum_MatMul) {
  TensorShape x_shape({2, 3});
  TensorShape y_shape({3, 3});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  Output y = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(y_shape));
  auto z = Einsum(scope_, {x, y}, "ij,jk->ik");
  TensorShape z_shape({2, 3});
  RunTest({x, y}, {x_shape, y_shape}, {z}, {z_shape});
}

TEST_F(LinalgGradTest, Einsum_MatMulComplex) {
  TensorShape x_shape({2, 3});
  TensorShape y_shape({3, 3});
  Output x = Placeholder(scope_, DT_COMPLEX64, Placeholder::Shape(x_shape));
  Output y = Placeholder(scope_, DT_COMPLEX64, Placeholder::Shape(y_shape));
  auto z = Einsum(scope_, {x, y}, "ij,jk->ik");
  TensorShape z_shape({2, 3});
  TF_ASSERT_OK(scope_.status());
  float max_error;
  TF_ASSERT_OK((ComputeGradientError<complex64, complex64, float>(
      scope_, {x, y}, {x_shape, y_shape}, {z}, {z_shape}, &max_error)));
  EXPECT_LT(max_error, 1e-3);
}

TEST_F(LinalgGradTest, Einsum_MatMulBroadcast) {
  TensorShape x_shape({3, 2, 3});
  TensorShape y_shape({3, 3});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  Output y = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(y_shape));
  auto z = Einsum(scope_, {x, y}, "...ij,...jk->...ik");
  TensorShape z_shape({3, 2, 3});
  RunTest({x, y}, {x_shape, y_shape}, {z}, {z_shape});
}

TEST_F(LinalgGradTest, Einsum_Trace) {
  TensorShape x_shape({3, 3});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  // Note: In Python this could just be "ii" becuase tf.einsum normalizes the
  // equation, but c++ doesn't do that.
  auto z = Einsum(scope_, {x}, "ii->");
  TensorShape z_shape({});
  RunTest({x}, {x_shape}, {z}, {z_shape});
}

TEST_F(LinalgGradTest, Einsum_TraceBroadcast) {
  TensorShape x_shape({4, 3, 3});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  // Note: In Python this could just be "ii" becuase tf.einsum normalizes the
  // equation, but c++ doesn't do that.
  auto z = Einsum(scope_, {x}, "...ii->...");
  TensorShape z_shape({4});
  RunTest({x}, {x_shape}, {z}, {z_shape});
}

TEST_F(LinalgGradTest, Einsum_DotProduct) {
  TensorShape x_shape({3});
  TensorShape y_shape({3});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  Output y = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(y_shape));
  auto z = Einsum(scope_, {x, y}, "i,i->");
  TensorShape z_shape({});
  RunTest({x, y}, {x_shape, y_shape}, {z}, {z_shape});
}

TEST_F(LinalgGradTest, Einsum_OuterProduct) {
  TensorShape x_shape({3});
  TensorShape y_shape({5});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  Output y = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(y_shape));
  auto z = Einsum(scope_, {x, y}, "i,j->ij");
  TensorShape z_shape({3, 5});
  RunTest({x, y}, {x_shape, y_shape}, {z}, {z_shape});
}

TEST_F(LinalgGradTest, Einsum_TwoInputReduction) {
  TensorShape x_shape({3, 2, 4});
  TensorShape y_shape({4, 5});
  Output x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  Output y = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(y_shape));
  auto z = Einsum(scope_, {x, y}, "abc,cd->ad");
  TensorShape z_shape({3, 5});
  RunTest({x, y}, {x_shape, y_shape}, {z}, {z_shape});
}

}  // namespace
}  // namespace tensorflow
