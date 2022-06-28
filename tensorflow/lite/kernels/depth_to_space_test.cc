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
class MHTracer_DTPStensorflowPSlitePSkernelsPSdepth_to_space_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSdepth_to_space_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSdepth_to_space_testDTcc() {
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
#include <stdint.h>

#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class DepthToSpaceOpModel : public SingleOpModel {
 public:
  DepthToSpaceOpModel(const TensorData& tensor_data, int block_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdepth_to_space_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/kernels/depth_to_space_test.cc", "DepthToSpaceOpModel");

    input_ = AddInput(tensor_data);
    output_ = AddOutput(tensor_data);
    SetBuiltinOp(BuiltinOperator_DEPTH_TO_SPACE,
                 BuiltinOptions_DepthToSpaceOptions,
                 CreateDepthToSpaceOptions(builder_, block_size).Union());
    BuildInterpreter({GetShape(input_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdepth_to_space_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/lite/kernels/depth_to_space_test.cc", "SetInput");

    PopulateTensor<T>(input_, data);
  }
  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int output_;
};

#ifdef GTEST_HAS_DEATH_TEST
TEST(DepthToSpaceOpModel, BadBlockSize) {
  EXPECT_DEATH(DepthToSpaceOpModel({TensorType_FLOAT32, {1, 1, 1, 4}}, 4),
               "Cannot allocate tensors");
}

TEST(DepthToSpaceOpModel, NoBlockSize) {
  EXPECT_DEATH(DepthToSpaceOpModel({TensorType_FLOAT32, {1, 1, 1, 4}}, 0),
               "Cannot allocate tensors");
}
#endif

TEST(DepthToSpaceOpModel, Float32) {
  DepthToSpaceOpModel m({TensorType_FLOAT32, {1, 1, 1, 4}}, 2);
  m.SetInput<float>({1.4, 2.3, 3.2, 4.1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({1.4, 2.3, 3.2, 4.1}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 2, 2, 1));
}

TEST(DepthToSpaceOpModel, Uint8) {
  DepthToSpaceOpModel m({TensorType_UINT8, {1, 1, 2, 4}}, 2);
  m.SetInput<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({1, 2, 5, 6, 3, 4, 7, 8}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 2, 4, 1));
}

TEST(DepthToSpaceOpModel, int8) {
  DepthToSpaceOpModel m({TensorType_INT8, {1, 2, 1, 4}}, 2);
  m.SetInput<int8_t>({1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 4, 2, 1));
}

TEST(DepthToSpaceOpModel, Int32) {
  DepthToSpaceOpModel m({TensorType_INT32, {1, 2, 2, 4}}, 2);
  m.SetInput<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(),
              ElementsAreArray(
                  {1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 4, 4, 1));
}

TEST(DepthToSpaceOpModel, Int64) {
  DepthToSpaceOpModel m({TensorType_INT64, {1, 1, 1, 1}}, 1);
  m.SetInput<int64_t>({4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int64_t>(), ElementsAreArray({4}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 1, 1, 1));
}

}  // namespace
}  // namespace tflite
