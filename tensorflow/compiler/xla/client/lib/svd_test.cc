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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSsvd_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSsvd_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSsvd_testDTcc() {
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

#include "tensorflow/compiler/xla/client/lib/svd.h"

#include <utility>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"

namespace xla {

class SVDTest : public ClientLibraryTestBase {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSsvd_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/xla/client/lib/svd_test.cc", "SetUp");

    ClientLibraryTestBase::SetUp();
    batch_3d_4x5_ = Array3D<float>{
        {
            {4, 6, 8, 10, 1},
            {6, 45, 54, 63, 1},
            {8, 54, 146, 166, 1},
            {10, 63, 166, 310, 1},
        },
        {
            {16, 24, 8, 12, 6},
            {24, 61, 82, 48, 5},
            {8, 82, 100, 6, 4},
            {12, 48, 6, 62, 3},
        },
    };

    // Test fails with TensorFloat-32 enabled
    tensorflow::enable_tensor_float_32_execution(false);
  }
  void TearDown() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSsvd_testDTcc mht_1(mht_1_v, 234, "", "./tensorflow/compiler/xla/client/lib/svd_test.cc", "TearDown");
 ClientLibraryTestBase::TearDown(); }

  Array3D<float> GetUnitMatrix3D(int32_t batch_dim, int32_t mat_dim) {
    Array3D<float> result(batch_dim, mat_dim, mat_dim, 0.0);
    for (int i = 0; i < batch_dim; ++i) {
      for (int j = 0; j < mat_dim; ++j) {
        result({i, j, j}) = 1.0;
      }
    }
    return result;
  }

  XlaOp ComputeMatmulUDVT(SVDResult result, XlaBuilder* builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSsvd_testDTcc mht_2(mht_2_v, 249, "", "./tensorflow/compiler/xla/client/lib/svd_test.cc", "ComputeMatmulUDVT");

    Shape u_shape = builder->GetShape(result.u).ValueOrDie();
    Shape v_shape = builder->GetShape(result.v).ValueOrDie();

    int64_t m = ShapeUtil::GetDimension(u_shape, -1);
    int64_t n = ShapeUtil::GetDimension(v_shape, -1);

    auto v = result.v;
    auto u = result.u;
    auto d = result.d;

    if (m > n) {
      u = SliceInMinorDims(u, {0, 0}, {m, n});
    } else if (m < n) {
      v = SliceInMinorDims(v, {0, 0}, {n, m});
    }

    int num_dims = u_shape.rank();
    std::vector<int64_t> broadcast_dims(num_dims - 1);
    std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
    broadcast_dims[num_dims - 2] = num_dims - 1;
    return BatchDot(Mul(u, d, broadcast_dims), TransposeInMinorDims(v),
                    PrecisionConfig::HIGHEST);
  }

  XlaOp GetAverageAbsoluteError(XlaOp m1, XlaOp m2, XlaBuilder* builder) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSsvd_testDTcc mht_3(mht_3_v, 277, "", "./tensorflow/compiler/xla/client/lib/svd_test.cc", "GetAverageAbsoluteError");

    Shape shape = builder->GetShape(m1).ValueOrDie();
    int64_t size = 1;
    for (auto d : shape.dimensions()) {
      size *= d;
    }
    return ReduceAll(Abs(m1 - m2), ConstantR0WithType(builder, F32, 0),
                     CreateScalarAddComputation(F32, builder)) /
           ConstantR0WithType(builder, F32, size);
  }

  Array2D<float> GenerateRandomMatrix(int xsize, int ysize) {
    Array2D<float> result{xsize, ysize, 0.0};
    result.FillRandom(10 /* stddev */, 2 /* mean */);
    return result;
  }

  Array3D<float> batch_3d_4x5_;
};

XLA_TEST_F(SVDTest, Simple2D) {
  XlaBuilder builder(TestName());

  Array2D<float> simple_2d_4x4_ = Array2D<float>{
      {4, 6, 8, 10},
      {6, 45, 54, 63},
      {8, 54, 146, 166},
      {10, 63, 166, 310},
  };
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(simple_2d_4x4_, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-6);
  ComputeMatmulUDVT(result, &builder);

  ComputeAndCompareR2<float>(&builder, simple_2d_4x4_, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SVDTest, Test_VWVt_EQ_A_2x4x5) {
  XlaBuilder builder(TestName());

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(batch_3d_4x5_, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-8);
  ComputeMatmulUDVT(result, &builder);

  ComputeAndCompareR3<float>(&builder, batch_3d_4x5_, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SVDTest, Test_Orthogonality_U) {
  XlaBuilder builder(TestName());

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(batch_3d_4x5_, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-8);
  ComputeMatmulUDVT(result, &builder);
  BatchDot(result.u, TransposeInMinorDims(result.u));

  ComputeAndCompareR3<float>(&builder, GetUnitMatrix3D(2, 4), {a_data.get()},
                             ErrorSpec(1e-2, 1e-2));
}

XLA_TEST_F(SVDTest, Test_Orthogonality_V) {
  XlaBuilder builder(TestName());

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(batch_3d_4x5_, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-8);
  BatchDot(result.v, TransposeInMinorDims(result.v), PrecisionConfig::HIGHEST);

  ComputeAndCompareR3<float>(&builder, GetUnitMatrix3D(2, 5), {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SVDTest, TestSingleValuesMatchNumpy) {
  XlaBuilder builder(TestName());

  auto singular_values = Array2D<float>{
      {431.05153007, 49.88334164, 20.94464584, 3.24845468},
      {179.73128591, 68.05162245, 21.77679503, 13.94319712},
  };

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(batch_3d_4x5_, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-8);
  Add(result.d, ZerosLike(result.d));

  ComputeAndCompareR2<float>(&builder, singular_values, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

// Too slow on the interpreter backend.
XLA_TEST_F(SVDTest,
           DISABLED_ON_INTERPRETER(Various_Size_Random_Matrix_512x128)) {
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(512, 128);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SVDTest, Various_Size_Random_Matrix_128x256) {
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(128, 256);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SVDTest, Various_Size_Random_Matrix_256x128) {
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(256, 128);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

// Too slow on the interpreter backend.
XLA_TEST_F(SVDTest,
           DISABLED_ON_INTERPRETER(Various_Size_Random_Matrix_128x512)) {
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(128, 512);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

// Too slow on the interpreter and CPU backends.
XLA_TEST_F(SVDTest, DISABLED_ON_CPU(DISABLED_ON_INTERPRETER(
                        Various_Size_Random_Matrix_512x256))) {
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(512, 256);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

// Too slow on the CPU, GPU and interpreter backends.
XLA_TEST_F(SVDTest, DISABLED_ON_GPU(DISABLED_ON_CPU(DISABLED_ON_INTERPRETER(
                        Various_Size_Random_Matrix_512x512)))) {
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(512, 512);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

}  // namespace xla
