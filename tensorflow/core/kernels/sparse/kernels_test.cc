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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSkernels_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSkernels_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSkernels_testDTcc() {
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

#include "tensorflow/core/kernels/sparse/kernels.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(SparseTensorToCSRSparseMatrix, SingleBatchConversion) {
  const auto indices =
      test::AsTensor<int64_t>({0, 0, 2, 3, 2, 4, 3, 0}, TensorShape({4, 2}));
  Tensor batch_ptr(DT_INT32, {2});
  Tensor csr_col_ind(DT_INT32, {4});
  auto csr_row_ptr = test::AsTensor<int32>({0, 0, 0, 0, 0});

  functor::SparseTensorToCSRSparseMatrixCPUFunctor coo_to_csr;
  TF_EXPECT_OK(coo_to_csr(1 /* batch_size */, 4 /* num_rows */,
                          indices.template matrix<int64_t>(),
                          batch_ptr.vec<int32>(), csr_row_ptr.vec<int32>(),
                          csr_col_ind.vec<int32>()));

  test::ExpectTensorEqual<int32>(batch_ptr, test::AsTensor<int32>({0, 4}));
  test::ExpectTensorEqual<int32>(csr_row_ptr,
                                 test::AsTensor<int32>({0, 1, 1, 3, 4}));
  test::ExpectTensorEqual<int32>(csr_col_ind,
                                 test::AsTensor<int32>({0, 3, 4, 0}));
}

TEST(SparseTensorToCSRSparseMatrix, BatchConversion) {
  // Batch of 3 matrices, each having dimension [3, 4] with 3 non-zero elements.
  const auto indices = test::AsTensor<int64_t>({0, 0, 0,  //
                                                0, 2, 3,  //
                                                2, 0, 1},
                                               TensorShape({3, 3}));
  Tensor batch_ptr(DT_INT32, {4});
  Tensor csr_col_ind(DT_INT32, {3});
  // row pointers have size = batch_size * (num_rows + 1) = 3 * 4 = 12
  Tensor csr_row_ptr(DT_INT32, {12});
  test::FillFn<int32>(&csr_row_ptr, [](int unused) { return 0; });

  functor::SparseTensorToCSRSparseMatrixCPUFunctor coo_to_csr;
  TF_EXPECT_OK(coo_to_csr(3 /* batch_size */, 3 /* num_rows */,
                          indices.template matrix<int64_t>(),
                          batch_ptr.vec<int32>(), csr_row_ptr.vec<int32>(),
                          csr_col_ind.vec<int32>()));

  test::ExpectTensorEqual<int32>(batch_ptr,
                                 test::AsTensor<int32>({0, 2, 2, 3}));
  test::ExpectTensorEqual<int32>(csr_row_ptr,
                                 test::AsTensor<int32>({0, 1, 1, 2,  //
                                                        0, 0, 0, 0,  //
                                                        0, 1, 1, 1}));
  test::ExpectTensorEqual<int32>(csr_col_ind, test::AsTensor<int32>({0, 3, 1}));
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSkernels_testDTcc mht_0(mht_0_v, 248, "", "./tensorflow/core/kernels/sparse/kernels_test.cc", "main");

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
