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
class MHTracer_DTPStensorflowPSlitePSkernelsPSgradientPSbcast_grad_args_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSgradientPSbcast_grad_args_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSgradientPSbcast_grad_args_testDTcc() {
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

#include "tensorflow/lite/kernels/gradient/bcast_grad_args.h"

#include <cstdint>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace ops {
namespace custom {

namespace {

using testing::ElementsAreArray;

class BcastGradArgsInt32OpModel : public SingleOpModel {
 public:
  BcastGradArgsInt32OpModel(const TensorData& input1, const TensorData& input2,
                            const TensorData& output1,
                            const TensorData& output2) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgradientPSbcast_grad_args_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/kernels/gradient/bcast_grad_args_test.cc", "BcastGradArgsInt32OpModel");

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output1_ = AddOutput(output1);
    output2_ = AddOutput(output2);

    std::vector<uint8_t> custom_option;
    SetCustomOp("BroadcastGradientArgs", custom_option,
                Register_BROADCAST_GRADIENT_ARGS);
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  void SetInput1(const std::vector<int>& data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgradientPSbcast_grad_args_testDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/kernels/gradient/bcast_grad_args_test.cc", "SetInput1");

    PopulateTensor(input1_, data);
  }
  void SetInput2(const std::vector<int>& data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgradientPSbcast_grad_args_testDTcc mht_2(mht_2_v, 227, "", "./tensorflow/lite/kernels/gradient/bcast_grad_args_test.cc", "SetInput2");

    PopulateTensor(input2_, data);
  }

  std::vector<int> GetOutput1() { return ExtractVector<int>(output1_); }
  std::vector<int> GetOutput1Shape() { return GetTensorShape(output1_); }
  std::vector<int> GetOutput2() { return ExtractVector<int>(output2_); }
  std::vector<int> GetOutput2Shape() { return GetTensorShape(output2_); }

 protected:
  int input1_;
  int input2_;
  int output1_;
  int output2_;
};

TEST(BcastGradArgsInt32OpModel, AllEqualsInt32DTypes) {
  BcastGradArgsInt32OpModel model(
      /*input1=*/{TensorType_INT32, {4}},
      /*input2=*/{TensorType_INT32, {4}},
      /*output1=*/{TensorType_INT32, {}},
      /*output2=*/{TensorType_INT32, {}});
  model.SetInput1({3, 1, 2, 3});
  model.SetInput2({3, 1, 2, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput1().size(), 0);
  EXPECT_THAT(model.GetOutput2().size(), 0);
}

TEST(BcastGradArgsInt32OpModel, BroadcastableDimAtInput1Int32DTypes) {
  BcastGradArgsInt32OpModel model(
      /*input1=*/{TensorType_INT32, {4}},
      /*input2=*/{TensorType_INT32, {4}},
      /*output1=*/{TensorType_INT32, {}},
      /*output2=*/{TensorType_INT32, {}});
  model.SetInput1({3, 4, 1, 3});
  model.SetInput2({3, 4, 2, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput1(), ElementsAreArray({2}));
  EXPECT_THAT(model.GetOutput2().size(), 0);
}

TEST(BcastGradArgsInt32OpModel, BroadcastableDimAtInput2Int32DTypes) {
  BcastGradArgsInt32OpModel model(
      /*input1=*/{TensorType_INT32, {4}},
      /*input2=*/{TensorType_INT32, {4}},
      /*output1=*/{TensorType_INT32, {}},
      /*output2=*/{TensorType_INT32, {}});
  model.SetInput1({3, 4, 2, 3});
  model.SetInput2({3, 1, 2, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput1().size(), 0);
  EXPECT_THAT(model.GetOutput2(), ElementsAreArray({1}));
}

TEST(BcastGradArgsInt32OpModel, DifferentInputSizesInt32DTypes) {
  BcastGradArgsInt32OpModel model(
      /*input1=*/{TensorType_INT32, {4}},
      /*input2=*/{TensorType_INT32, {3}},
      /*output1=*/{TensorType_INT32, {}},
      /*output2=*/{TensorType_INT32, {}});
  model.SetInput1({3, 4, 2, 3});
  model.SetInput2({4, 2, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput1().size(), 0);
  EXPECT_THAT(model.GetOutput2(), ElementsAreArray({0}));
}

TEST(BcastGradArgsInt32OpModel, NonBroadcastableDimsInt32DTypes) {
  BcastGradArgsInt32OpModel model(
      /*input1=*/{TensorType_INT32, {4}},
      /*input2=*/{TensorType_INT32, {4}},
      /*output1=*/{TensorType_INT32, {}},
      /*output2=*/{TensorType_INT32, {}});
  model.SetInput1({3, 4, 2, 3});
  model.SetInput2({9, 9, 9, 9});
  EXPECT_THAT(model.InvokeUnchecked(), kTfLiteError);
}

class BcastGradArgsInt64OpModel : public SingleOpModel {
 public:
  BcastGradArgsInt64OpModel(const TensorData& input1, const TensorData& input2,
                            const TensorData& output1,
                            const TensorData& output2) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgradientPSbcast_grad_args_testDTcc mht_3(mht_3_v, 317, "", "./tensorflow/lite/kernels/gradient/bcast_grad_args_test.cc", "BcastGradArgsInt64OpModel");

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output1_ = AddOutput(output1);
    output2_ = AddOutput(output2);

    std::vector<uint8_t> custom_option;
    SetCustomOp("BroadcastGradientArgs", custom_option,
                Register_BROADCAST_GRADIENT_ARGS);
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  void SetInput1(const std::vector<int64_t>& data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgradientPSbcast_grad_args_testDTcc mht_4(mht_4_v, 332, "", "./tensorflow/lite/kernels/gradient/bcast_grad_args_test.cc", "SetInput1");

    PopulateTensor(input1_, data);
  }
  void SetInput2(const std::vector<int64_t>& data) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgradientPSbcast_grad_args_testDTcc mht_5(mht_5_v, 338, "", "./tensorflow/lite/kernels/gradient/bcast_grad_args_test.cc", "SetInput2");

    PopulateTensor(input2_, data);
  }

  std::vector<int64_t> GetOutput1() { return ExtractVector<int64_t>(output1_); }
  std::vector<int> GetOutput1Shape() { return GetTensorShape(output1_); }
  std::vector<int64_t> GetOutput2() { return ExtractVector<int64_t>(output2_); }
  std::vector<int> GetOutput2Shape() { return GetTensorShape(output2_); }

 protected:
  int input1_;
  int input2_;
  int output1_;
  int output2_;
};

TEST(BcastGradArgsInt32OpModel, AllEqualsInt64DTypes) {
  BcastGradArgsInt64OpModel model(
      /*input1=*/{TensorType_INT64, {4}},
      /*input2=*/{TensorType_INT64, {4}},
      /*output1=*/{TensorType_INT64, {}},
      /*output2=*/{TensorType_INT64, {}});
  model.SetInput1({3, 1, 2, 3});
  model.SetInput2({3, 1, 2, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput1().size(), 0);
  EXPECT_THAT(model.GetOutput2().size(), 0);
}

TEST(BcastGradArgsInt32OpModel, BroadcastableDimAtInput1Int64DTypes) {
  BcastGradArgsInt64OpModel model(
      /*input1=*/{TensorType_INT64, {4}},
      /*input2=*/{TensorType_INT64, {4}},
      /*output1=*/{TensorType_INT64, {}},
      /*output2=*/{TensorType_INT64, {}});
  model.SetInput1({3, 4, 1, 3});
  model.SetInput2({3, 4, 2, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput1(), ElementsAreArray({2}));
  EXPECT_THAT(model.GetOutput2().size(), 0);
}

TEST(BcastGradArgsInt32OpModel, BroadcastableDimAtInput2Int64DTypes) {
  BcastGradArgsInt64OpModel model(
      /*input1=*/{TensorType_INT64, {4}},
      /*input2=*/{TensorType_INT64, {4}},
      /*output1=*/{TensorType_INT64, {}},
      /*output2=*/{TensorType_INT64, {}});
  model.SetInput1({3, 4, 2, 3});
  model.SetInput2({3, 1, 2, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput1().size(), 0);
  EXPECT_THAT(model.GetOutput2(), ElementsAreArray({1}));
}

TEST(BcastGradArgsInt32OpModel, DifferentInputSizesInt64DTypes) {
  BcastGradArgsInt64OpModel model(
      /*input1=*/{TensorType_INT64, {4}},
      /*input2=*/{TensorType_INT64, {3}},
      /*output1=*/{TensorType_INT64, {}},
      /*output2=*/{TensorType_INT64, {}});
  model.SetInput1({3, 4, 2, 3});
  model.SetInput2({4, 2, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput1().size(), 0);
  EXPECT_THAT(model.GetOutput2(), ElementsAreArray({0}));
}

TEST(BcastGradArgsInt32OpModel, NonBroadcastableDimsInt64DTypes) {
  BcastGradArgsInt64OpModel model(
      /*input1=*/{TensorType_INT64, {4}},
      /*input2=*/{TensorType_INT64, {4}},
      /*output1=*/{TensorType_INT64, {}},
      /*output2=*/{TensorType_INT64, {}});
  model.SetInput1({3, 4, 2, 3});
  model.SetInput2({9, 9, 9, 9});
  EXPECT_THAT(model.InvokeUnchecked(), kTfLiteError);
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
