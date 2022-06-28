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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

template <typename T>
class RGBToHSVOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType data_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/kernels/image/colorspace_op_test.cc", "MakeOp");

    TF_EXPECT_OK(NodeDefBuilder("rgb_to_hsv_op", "RGBToHSV")
                     .Input(FakeInput(data_type))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }

  void CheckBlack(DataType data_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/kernels/image/colorspace_op_test.cc", "CheckBlack");

    // Black pixel should map to hsv = [0,0,0]
    AddInputFromArray<T>(TensorShape({3}), {0, 0, 0});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {0.0, 0.0, 0.0});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckGray(DataType data_type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc mht_2(mht_2_v, 226, "", "./tensorflow/core/kernels/image/colorspace_op_test.cc", "CheckGray");

    // Gray pixel should have hue = saturation = 0.0, value = r/255
    AddInputFromArray<T>(TensorShape({3}), {.5, .5, .5});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {0.0, 0.0, .5});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckWhite(DataType data_type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/kernels/image/colorspace_op_test.cc", "CheckWhite");

    // Gray pixel should have hue = saturation = 0.0, value = 1.0
    AddInputFromArray<T>(TensorShape({3}), {1, 1, 1});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {0.0, 0.0, 1.0});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckRedMax(DataType data_type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc mht_4(mht_4_v, 252, "", "./tensorflow/core/kernels/image/colorspace_op_test.cc", "CheckRedMax");

    // Test case where red channel dominates
    AddInputFromArray<T>(TensorShape({3}), {.8f, .4f, .2f});
    TF_ASSERT_OK(RunOpKernel());

    T expected_h = 1. / 6. * .2 / .6;
    T expected_s = .6 / .8;
    T expected_v = .8 / 1.;

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {expected_h, expected_s, expected_v});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckGreenMax(DataType data_type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc mht_5(mht_5_v, 269, "", "./tensorflow/core/kernels/image/colorspace_op_test.cc", "CheckGreenMax");

    // Test case where green channel dominates
    AddInputFromArray<T>(TensorShape({3}), {.2f, .8f, .4f});
    TF_ASSERT_OK(RunOpKernel());

    T expected_h = 1. / 6. * (2.0 + (.2 / .6));
    T expected_s = .6 / .8;
    T expected_v = .8 / 1.;

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {expected_h, expected_s, expected_v});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckBlueMax(DataType data_type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc mht_6(mht_6_v, 286, "", "./tensorflow/core/kernels/image/colorspace_op_test.cc", "CheckBlueMax");

    // Test case where blue channel dominates
    AddInputFromArray<T>(TensorShape({3}), {.4f, .2f, .8f});
    TF_ASSERT_OK(RunOpKernel());

    T expected_h = 1. / 6. * (4.0 + (.2 / .6));
    T expected_s = .6 / .8;
    T expected_v = .8 / 1.;

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {expected_h, expected_s, expected_v});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckNegativeDifference(DataType data_type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc mht_7(mht_7_v, 303, "", "./tensorflow/core/kernels/image/colorspace_op_test.cc", "CheckNegativeDifference");

    AddInputFromArray<T>(TensorShape({3}), {0, .1f, .2f});
    TF_ASSERT_OK(RunOpKernel());

    T expected_h = 1. / 6. * (4.0 + (-.1 / .2));
    T expected_s = .2 / .2;
    T expected_v = .2 / 1.;

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {expected_h, expected_s, expected_v});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }
};

template <typename T>
class HSVToRGBOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType data_type) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc mht_8(mht_8_v, 323, "", "./tensorflow/core/kernels/image/colorspace_op_test.cc", "MakeOp");

    TF_EXPECT_OK(NodeDefBuilder("hsv_to_rgb_op", "HSVToRGB")
                     .Input(FakeInput(data_type))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }

  void CheckBlack(DataType data_type) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc mht_9(mht_9_v, 333, "", "./tensorflow/core/kernels/image/colorspace_op_test.cc", "CheckBlack");

    // Black pixel should map to rgb = [0,0,0]
    AddInputFromArray<T>(TensorShape({3}), {0.0, 0.0, 0.0});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {0, 0, 0});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckGray(DataType data_type) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc mht_10(mht_10_v, 346, "", "./tensorflow/core/kernels/image/colorspace_op_test.cc", "CheckGray");

    // Gray pixel should have hue = saturation = 0.0, value = r/255
    AddInputFromArray<T>(TensorShape({3}), {0.0, 0.0, .5});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {.5, .5, .5});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckWhite(DataType data_type) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc mht_11(mht_11_v, 359, "", "./tensorflow/core/kernels/image/colorspace_op_test.cc", "CheckWhite");

    // Gray pixel should have hue = saturation = 0.0, value = 1.0
    AddInputFromArray<T>(TensorShape({3}), {0.0, 0.0, 1.0});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {1, 1, 1});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckRedMax(DataType data_type) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc mht_12(mht_12_v, 372, "", "./tensorflow/core/kernels/image/colorspace_op_test.cc", "CheckRedMax");

    // Test case where red channel dominates
    T expected_h = 1. / 6. * .2 / .6;
    T expected_s = .6 / .8;
    T expected_v = .8 / 1.;

    AddInputFromArray<T>(TensorShape({3}),
                         {expected_h, expected_s, expected_v});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {.8, .4, .2});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckGreenMax(DataType data_type) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc mht_13(mht_13_v, 390, "", "./tensorflow/core/kernels/image/colorspace_op_test.cc", "CheckGreenMax");

    // Test case where green channel dominates
    T expected_h = 1. / 6. * (2.0 + (.2 / .6));
    T expected_s = .6 / .8;
    T expected_v = .8 / 1.;

    AddInputFromArray<T>(TensorShape({3}),
                         {expected_h, expected_s, expected_v});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {.2, .8, .4});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckBlueMax(DataType data_type) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc mht_14(mht_14_v, 408, "", "./tensorflow/core/kernels/image/colorspace_op_test.cc", "CheckBlueMax");

    // Test case where blue channel dominates
    T expected_h = 1. / 6. * (4.0 + (.2 / .6));
    T expected_s = .6 / .8;
    T expected_v = .8 / 1.0;

    AddInputFromArray<T>(TensorShape({3}),
                         {expected_h, expected_s, expected_v});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {.4, .2, .8});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckNegativeDifference(DataType data_type) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePScolorspace_op_testDTcc mht_15(mht_15_v, 426, "", "./tensorflow/core/kernels/image/colorspace_op_test.cc", "CheckNegativeDifference");

    T expected_h = 1. / 6. * (4.0 + (-.1 / .2));
    T expected_s = .2 / .2;
    T expected_v = .2 / 1.;

    AddInputFromArray<T>(TensorShape({3}),
                         {expected_h, expected_s, expected_v});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), data_type, TensorShape({3}));
    test::FillValues<T>(&expected, {0, .1f, .2f});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }
};

#define TEST_COLORSPACE(test, dt)         \
  TEST_F(test, CheckBlack) {              \
    MakeOp(dt);                           \
    CheckBlack(dt);                       \
  }                                       \
  TEST_F(test, CheckGray) {               \
    MakeOp(dt);                           \
    CheckGray(dt);                        \
  }                                       \
  TEST_F(test, CheckWhite) {              \
    MakeOp(dt);                           \
    CheckWhite(dt);                       \
  }                                       \
  TEST_F(test, CheckRedMax) {             \
    MakeOp(dt);                           \
    CheckRedMax(dt);                      \
  }                                       \
  TEST_F(test, CheckGreenMax) {           \
    MakeOp(dt);                           \
    CheckGreenMax(dt);                    \
  }                                       \
  TEST_F(test, CheckBlueMax) {            \
    MakeOp(dt);                           \
    CheckBlueMax(dt);                     \
  }                                       \
  TEST_F(test, CheckNegativeDifference) { \
    MakeOp(dt);                           \
    CheckNegativeDifference(dt);          \
  }

typedef RGBToHSVOpTest<float> rgb_to_hsv_float;
typedef RGBToHSVOpTest<double> rgb_to_hsv_double;

TEST_COLORSPACE(rgb_to_hsv_float, DT_FLOAT);
TEST_COLORSPACE(rgb_to_hsv_double, DT_DOUBLE);

typedef HSVToRGBOpTest<float> hsv_to_rgb_float;
typedef HSVToRGBOpTest<double> hsv_to_rgb_double;

TEST_COLORSPACE(hsv_to_rgb_float, DT_FLOAT);
TEST_COLORSPACE(hsv_to_rgb_double, DT_DOUBLE);
}  // namespace tensorflow
