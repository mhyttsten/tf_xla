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
class MHTracer_DTPStensorflowPSlitePSkernelsPSsoftmax_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSsoftmax_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSsoftmax_testDTcc() {
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
// Unit test for TFLite SOFTMAX op.

#include "tensorflow/lite/kernels/internal/reference/softmax.h"

#include <initializer_list>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

class SoftmaxOpModel : public SingleOpModel {
 public:
  SoftmaxOpModel(int batches, int size, float beta)
      : batches_(batches), input_size_(size), beta_(beta) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsoftmax_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/lite/kernels/softmax_test.cc", "SoftmaxOpModel");

    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_SOFTMAX, BuiltinOptions_SoftmaxOptions,
                 CreateSoftmaxOptions(builder_, beta_).Union());
    BuildInterpreter({{batches_, input_size_}});
  }

  void SetInput(std::initializer_list<float> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsoftmax_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/lite/kernels/softmax_test.cc", "SetInput");

    PopulateTensor(input_, data);
  }

  void SetInput(int offset, float* begin, float* end) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsoftmax_testDTcc mht_2(mht_2_v, 223, "", "./tensorflow/lite/kernels/softmax_test.cc", "SetInput");

    PopulateTensor(input_, offset, begin, end);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;

  int batches_;
  int input_size_;
  float beta_;
};

TEST(SoftmaxOpTest, SimpleTest) {
  SoftmaxOpModel m(/*batches=*/2, /*size=*/5, /*beta=*/1.0);
  m.SetInput({
      1.0, 2.0, 3.0, 4.0, 5.0,       // b = 0
      -1.0, -2.0, -3.0, -4.0, -5.0,  // b = 0
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {0.011656231, 0.031684921, 0.086128544, 0.234121657, 0.636408647,
           0.636408647, 0.234121657, 0.086128544, 0.031684921, 0.011656231},
          1e-6)));
}

TEST(SoftmaxOpTest, CompareWithTFminiBetaEq1) {
  const int batch_size = 2;
  const int input_size = 5;
  const float beta = 1.0;
  static float input_buffer[] = {
      1.0,  2.0,  3.0,  4.0,  5.0,   // b = 0
      -1.0, -2.0, -3.0, -4.0, -5.0,  // b = 1
  };

  SoftmaxOpModel m(batch_size, input_size, beta);

  m.SetInput(0, input_buffer, input_buffer + input_size * batch_size);

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  std::unique_ptr<float[]> output_buffer(new float[input_size * batch_size]);
  auto input_shape = RuntimeShape({batch_size, 1, 1, input_size});
  SoftmaxParams params;
  params.beta = beta;
  tflite::reference_ops::Softmax(params, input_shape, input_buffer, input_shape,
                                 output_buffer.get());

  std::vector<float> expected;
  expected.insert(expected.end(), output_buffer.get(),
                  output_buffer.get() + input_size * batch_size);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(expected, 1e-6)));
}

TEST(SoftmaxOpTest, CompareWithTFminiBetaNotEq1) {
  const int batch_size = 2;
  const int input_size = 5;
  const float beta = 0.5;
  static float input_buffer[] = {
      1.0,  2.0,  3.0,  4.0,  5.0,   // b = 0
      -1.0, -2.0, -3.0, -4.0, -5.0,  // b = 1
  };

  SoftmaxOpModel m(batch_size, input_size, beta);

  m.SetInput(0, input_buffer, input_buffer + input_size * batch_size);

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  std::unique_ptr<float[]> output_buffer(new float[input_size * batch_size]);
  auto input_shape = RuntimeShape({batch_size, 1, 1, input_size});
  SoftmaxParams params;
  params.beta = beta;
  tflite::reference_ops::Softmax(params, input_shape, input_buffer, input_shape,
                                 output_buffer.get());

  std::vector<float> expected;
  expected.insert(expected.end(), output_buffer.get(),
                  output_buffer.get() + input_size * batch_size);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(expected, 1e-6)));
}

}  // namespace
}  // namespace tflite
