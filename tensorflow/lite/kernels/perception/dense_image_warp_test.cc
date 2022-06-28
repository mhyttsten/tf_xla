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
class MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSdense_image_warp_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSdense_image_warp_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSdense_image_warp_testDTcc() {
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

#include <cstdint>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/perception/perception_ops.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace ops {
namespace custom {

namespace {

using testing::ElementsAreArray;

class DenseImageWarpOpModel : public SingleOpModel {
 public:
  DenseImageWarpOpModel(const TensorData& input, const TensorData& flow,
                        const TensorData& output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSdense_image_warp_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/kernels/perception/dense_image_warp_test.cc", "DenseImageWarpOpModel");

    input_ = AddInput(input);
    flow_ = AddInput(flow);
    output_ = AddOutput(output);

    std::vector<uint8_t> custom_option;
    SetCustomOp("DenseImageWarp", custom_option, RegisterDenseImageWarp);
    BuildInterpreter({GetShape(input_), GetShape(flow_)});
  }

  void SetInput(const std::vector<float>& data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSdense_image_warp_testDTcc mht_1(mht_1_v, 217, "", "./tensorflow/lite/kernels/perception/dense_image_warp_test.cc", "SetInput");

    PopulateTensor(input_, data);
  }
  void SetFlow(const std::vector<float>& data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSdense_image_warp_testDTcc mht_2(mht_2_v, 223, "", "./tensorflow/lite/kernels/perception/dense_image_warp_test.cc", "SetFlow");
 PopulateTensor(flow_, data); }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int flow_;
  int output_;
};

TEST(DenseImageWarpOpTest, MismatchedSizeTest) {
  EXPECT_DEATH_IF_SUPPORTED(
      DenseImageWarpOpModel model(
          /*input=*/{TensorType_FLOAT32, {1, 4, 4, 1}},
          /*flow=*/{TensorType_FLOAT32, {1, 4, 2, 2}},
          /*output=*/{TensorType_FLOAT32, {}});
      , "input_shape.Dims.2. != flow_shape.Dims.2. .4 != 2.");
}

TEST(DenseImageWarpOpTest, WrongFlowSizeTest) {
  EXPECT_DEATH_IF_SUPPORTED(DenseImageWarpOpModel model(
                                /*input=*/{TensorType_FLOAT32, {1, 4, 4, 1}},
                                /*flow=*/{TensorType_FLOAT32, {1, 4, 4, 1}},
                                /*output=*/{TensorType_FLOAT32, {}});
                            , "The last dimension of flow tensor must be 2.");
}

TEST(DenseImageWarpOpTest, SimpleTest) {
  DenseImageWarpOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 4, 4, 1}},
      /*flow=*/{TensorType_FLOAT32, {1, 4, 4, 2}},
      /*output=*/{TensorType_FLOAT32, {}});
  model.SetInput({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  model.SetFlow({4, 10, 6,  10, 4, 2, 6, 6,  10, -4, 2,  -2, 6,  8, 6, 0,
                 2, -2, 10, 6,  4, 4, 2, -4, -4, 10, -4, -4, -2, 6, 4, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({0, 0, 0, 0, 3, 3, 0, 3, 2, 0,
                                                   0, 3, 12, 15, 12, 0}));
}

TEST(DenseImageWarpOpTest, RoundTest) {
  DenseImageWarpOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 4, 4, 1}},
      /*flow=*/{TensorType_FLOAT32, {1, 4, 4, 2}},
      /*output=*/{TensorType_FLOAT32, {}});
  model.SetInput({0.2, 1.5, 2.4, 3.5, 4.6, 5.1, 6.3, 7.2, 8.5, 9.6, 10.9, 11.6,
                  12.8, 13.2, 14.4, 15.5});
  model.SetFlow({4, 10, 6,  10, 4, 2, 6, 6,  10, -4, 2,  -2, 6,  8, 6, 0,
                 2, -2, 10, 6,  4, 4, 2, -4, -4, 10, -4, -4, -2, 6, 4, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({0.2, 0.2, 0.2, 0.2, 3.5, 3.5, 0.2, 3.5, 2.4,
                                0.2, 0.2, 3.5, 12.8, 15.5, 12.8, 0.2}));
}

TEST(DenseImageWarpOpTest, WithBatchandChannelTest) {
  DenseImageWarpOpModel model(
      /*input=*/{TensorType_FLOAT32, {2, 4, 4, 3}},
      /*flow=*/{TensorType_FLOAT32, {2, 4, 4, 2}},
      /*output=*/{TensorType_FLOAT32, {}});

  std::vector<float> input_data;
  for (int i = 0; i < 96; ++i) input_data.push_back(i);
  model.SetInput(input_data);
  model.SetFlow({2, -2, 10, 6,  4, 4, 2, -4, -4, 10, -4, -4, -2, 6, 4, 6,
                 4, 10, 6,  10, 4, 2, 6, 6,  10, -4, 2,  -2, 6,  8, 6, 0,
                 2, -2, 10, 6,  4, 4, 2, -4, -4, 10, -4, -4, -2, 6, 4, 6,
                 4, 10, 6,  10, 4, 2, 6, 6,  10, -4, 2,  -2, 6,  8, 6, 0});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 4, 4, 3}));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({6,  7,  8,  0,  1,  2,  0,  1,  2,  9,  10, 11, 36, 37,
                        38, 45, 46, 47, 36, 37, 38, 0,  1,  2,  0,  1,  2,  0,
                        1,  2,  0,  1,  2,  0,  1,  2,  9,  10, 11, 21, 22, 23,
                        0,  1,  2,  9,  10, 11, 54, 55, 56, 48, 49, 50, 48, 49,
                        50, 57, 58, 59, 84, 85, 86, 93, 94, 95, 84, 85, 86, 48,
                        49, 50, 48, 49, 50, 48, 49, 50, 48, 49, 50, 48, 49, 50,
                        57, 58, 59, 69, 70, 71, 48, 49, 50, 57, 58, 59}));
}
}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
