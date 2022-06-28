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
class MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2d_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2d_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2d_testDTcc() {
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

#include <stdint.h>

#include <complex>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace ops {
namespace custom {

namespace {

using ::testing::ElementsAreArray;

class Irfft2dOpModel : public SingleOpModel {
 public:
  Irfft2dOpModel(const TensorData& input, const TensorData& fft_lengths) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2d_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/kernels/irfft2d_test.cc", "Irfft2dOpModel");

    input_ = AddInput(input);
    fft_lengths_ = AddInput(fft_lengths);
    TensorType output_type = TensorType_FLOAT32;
    output_ = AddOutput({output_type, {}});

    const std::vector<uint8_t> custom_option;
    SetCustomOp("Irfft2d", custom_option, Register_IRFFT2D);
    BuildInterpreter({GetShape(input_)});
  }

  int input() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2d_testDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/kernels/irfft2d_test.cc", "input");
 return input_; }
  int fft_lengths() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSirfft2d_testDTcc mht_2(mht_2_v, 225, "", "./tensorflow/lite/kernels/irfft2d_test.cc", "fft_lengths");
 return fft_lengths_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int fft_lengths_;
  int output_;
};

TEST(Irfft2dOpTest, FftLengthMatchesInputSize) {
  Irfft2dOpModel model({TensorType_COMPLEX64, {4, 3}}, {TensorType_INT32, {2}});
  // clang-format off
  model.PopulateTensor<std::complex<float>>(model.input(), {
    {75, 0},  {-6, -1}, {9, 0},  {-10, 5},  {-3, 2}, {-6, 11},
    {-15, 0}, {-2, 13}, {-5, 0}, {-10, -5}, {3, -6}, {-6, -11}
  });
  // clang-format on
  model.PopulateTensor<int32_t>(model.fft_lengths(), {4, 4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  float expected_result[16] = {1, 2, 3, 4, 3, 8, 6, 3, 5, 2, 7, 6, 9, 5, 8, 3};
  EXPECT_THAT(model.GetOutput(), ElementsAreArray(expected_result));
}

TEST(Irfft2dOpTest, FftLengthSmallerThanInputSize) {
  Irfft2dOpModel model({TensorType_COMPLEX64, {4, 3}}, {TensorType_INT32, {2}});
  // clang-format off
  model.PopulateTensor<std::complex<float>>(model.input(), {
    {75, 0},  {-6, -1}, {9, 0},  {-10, 5},  {-3, 2}, {-6, 11},
    {-15, 0}, {-2, 13}, {-5, 0}, {-10, -5}, {3, -6}, {-6, -11}
  });
  // clang-format on
  model.PopulateTensor<int32_t>(model.fft_lengths(), {2, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  float expected_result[4] = {14, 18.5, 20.5, 22};
  EXPECT_THAT(model.GetOutput(), ElementsAreArray(expected_result));
}

TEST(Irfft2dOpTest, FftLengthGreaterThanInputSize) {
  Irfft2dOpModel model({TensorType_COMPLEX64, {4, 3}}, {TensorType_INT32, {2}});
  // clang-format off
  model.PopulateTensor<std::complex<float>>(model.input(), {
    {75, 0},  {-6, -1}, {9, 0},  {-10, 5},  {-3, 2}, {-6, 11},
    {-15, 0}, {-2, 13}, {-5, 0}, {-10, -5}, {3, -6}, {-6, -11}
  });
  // clang-format on
  model.PopulateTensor<int32_t>(model.fft_lengths(), {4, 8});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  // clang-format off
  float expected_result[32] = {
    0.25, 0.54289322, 1.25, 1.25, 1.25, 1.95710678, 2.25, 1.25,
    1.25, 2.85355339, 4.25, 3.91421356, 2.75, 2.14644661, 1.75, 1.08578644,
    3., 1.43933983, 0.5, 2.14644661, 4., 3.56066017, 2.5, 2.85355339,
    5.625, 3.65533009, 1.375, 3.3017767, 5.125, 2.59466991, 0.375, 2.9482233
  };
  // clang-format on
  EXPECT_THAT(model.GetOutput(), ElementsAreArray(expected_result));
}

TEST(Irfft2dOpTest, InputDimsGreaterThan2) {
  Irfft2dOpModel model({TensorType_COMPLEX64, {2, 2, 3}},
                       {TensorType_INT32, {2}});
  // clang-format off
  model.PopulateTensor<std::complex<float>>(model.input(), {
    {30., 0.}, {-5, -3.}, { -4., 0.},
    {-10., 0.}, {1., 7.}, {  0., 0.},
    {58., 0.}, {-18., 6.}, { 26., 0.},
    {-18., 0.}, { 14., 2.}, {-18., 0.}
  });
  // clang-format on
  model.PopulateTensor<int32_t>(model.fft_lengths(), {2, 4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  float expected_result[16] = {1., 2., 3., 4., 3., 8., 6.,  3.,
                               5., 2., 7., 6., 7., 3., 23., 5.};
  EXPECT_THAT(model.GetOutput(), ElementsAreArray(expected_result));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
