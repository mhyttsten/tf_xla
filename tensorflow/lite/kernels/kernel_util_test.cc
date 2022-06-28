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
class MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_util_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_util_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/kernel_util.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {
using ::testing::ElementsAre;

struct TestContext : public TfLiteContext {
  string error;
};

void ReportError(TfLiteContext* context, const char* format, ...) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("format: \"" + (format == nullptr ? std::string("nullptr") : std::string((char*)format)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_util_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/lite/kernels/kernel_util_test.cc", "ReportError");

  TestContext* c = static_cast<TestContext*>(context);
  const size_t kBufferSize = 1024;
  char temp_buffer[kBufferSize];

  va_list args;
  va_start(args, format);
  vsnprintf(temp_buffer, kBufferSize, format, args);
  va_end(args);

  c->error = temp_buffer;
}

class KernelUtilTest : public ::testing::Test {
 public:
  KernelUtilTest() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_util_testDTcc mht_1(mht_1_v, 228, "", "./tensorflow/lite/kernels/kernel_util_test.cc", "KernelUtilTest");

    context_.ReportError = ReportError;

    memset(&tensor1_, 0, sizeof(TfLiteTensor));
    memset(&tensor2_, 0, sizeof(TfLiteTensor));
    memset(&tensor3_, 0, sizeof(TfLiteTensor));
    tensor1_.dims = nullptr;
    tensor2_.dims = nullptr;
    tensor3_.dims = nullptr;
    tensor1_.allocation_type = kTfLiteMmapRo;
    tensor2_.allocation_type = kTfLiteMmapRo;
    tensor3_.allocation_type = kTfLiteMmapRo;
  }
  ~KernelUtilTest() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_util_testDTcc mht_2(mht_2_v, 244, "", "./tensorflow/lite/kernels/kernel_util_test.cc", "~KernelUtilTest");

    TfLiteTensorFree(&tensor1_);
    TfLiteTensorFree(&tensor2_);
    TfLiteTensorFree(&tensor3_);
  }

  void SetShape(TfLiteTensor* tensor, std::initializer_list<int> dims) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSkernel_util_testDTcc mht_3(mht_3_v, 253, "", "./tensorflow/lite/kernels/kernel_util_test.cc", "SetShape");

    TfLiteTensorFree(tensor);
    tensor->dims = TfLiteIntArrayCreate(dims.size());
    int i = 0;
    for (const auto& d : dims) {
      tensor->dims->data[i] = d;
      ++i;
    }
  }

  std::vector<int> GetShape(TfLiteIntArray* dims) {
    std::vector<int> result;
    for (int i = 0; i < dims->size; ++i) {
      result.push_back(dims->data[i]);
    }
    return result;
  }

 protected:
  TestContext context_;
  TfLiteTensor tensor1_;
  TfLiteTensor tensor2_;
  TfLiteTensor tensor3_;
};

TEST_F(KernelUtilTest, SameShapeEmpty) {
  EXPECT_TRUE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor1_, {1, 2, 3});
  EXPECT_FALSE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor2_, {1, 2});
  EXPECT_FALSE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor2_, {1, 2, 3, 4});
  EXPECT_FALSE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor2_, {1, 2, 3});
  EXPECT_TRUE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor2_, {});
  EXPECT_FALSE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor1_, {});
  EXPECT_TRUE(HaveSameShapes(&tensor1_, &tensor2_));
}

TEST_F(KernelUtilTest, BroadcastShapeIncompatibleDim) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {1, 3});
  EXPECT_NE(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(context_.error,
            "Given shapes, [1,2] and [1,3], are not broadcastable.");
}

TEST_F(KernelUtilTest, BroadcastShapeIncompatibleDimWithZero) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 0});
  SetShape(&tensor2_, {1, 3});
  EXPECT_NE(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(context_.error,
            "Given shapes, [1,0] and [1,3], are not broadcastable.");
}

TEST_F(KernelUtilTest, BroadcastShapeOnes) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 1});
  SetShape(&tensor2_, {1, 3});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {1, 1});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeScalars) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(1, 2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {});
  SetShape(&tensor2_, {2});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(2));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeDifferentSizes) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {3, 1, 1});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(3, 1, 2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {1, 2, 3, 4});
  SetShape(&tensor2_, {1, 3, 1});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(1, 2, 3, 4));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeWithZero) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {3, 0, 1});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(3, 0, 2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {2, 1, 0});
  SetShape(&tensor2_, {1, 3, 1});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(2, 3, 0));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeIncompatibleDimOnThreeTensors) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {1, 3});
  SetShape(&tensor3_, {1, 4});
  EXPECT_NE(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(context_.error,
            "Given shapes, [1,2], [1,3] and [1,4], are not broadcastable.");
}

TEST_F(KernelUtilTest, BroadcastShapeIncompatibleDimWithZeroOnThreeTensors) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 1});
  SetShape(&tensor2_, {1, 3});
  SetShape(&tensor3_, {1, 0});
  EXPECT_NE(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(context_.error,
            "Given shapes, [1,1], [1,3] and [1,0], are not broadcastable.");
}

TEST_F(KernelUtilTest, BroadcastShapeOnesOnThreeTensors) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 1});
  SetShape(&tensor2_, {1, 1});
  SetShape(&tensor3_, {1, 3});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {1, 1});
  SetShape(&tensor3_, {1, 1});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {1, 1});
  SetShape(&tensor2_, {1, 4});
  SetShape(&tensor3_, {1, 1});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeScalarsOnThreeTensors) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {});
  SetShape(&tensor3_, {});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(1, 2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {});
  SetShape(&tensor2_, {2});
  SetShape(&tensor3_, {});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {});
  SetShape(&tensor2_, {});
  SetShape(&tensor3_, {3, 2, 1});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(3, 2, 1));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeDifferentSizesOnThreeTensors) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {3, 1, 1});
  SetShape(&tensor3_, {3, 1});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(3, 3, 2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {3, 4});
  SetShape(&tensor2_, {1, 3, 1});
  SetShape(&tensor3_, {1, 2, 1, 1});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(1, 2, 3, 4));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeWithZeroOnThreeTensors) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {3, 1, 1});
  SetShape(&tensor3_, {0, 1});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(3, 0, 2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {1, 4});
  SetShape(&tensor2_, {1, 0, 1});
  SetShape(&tensor3_, {1, 2, 1, 1});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(1, 2, 0, 4));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, GetShapeDebugString) {
  TfLiteIntArray* dims0 = TfLiteIntArrayCreate(0);
  EXPECT_EQ("[]", GetShapeDebugString(dims0));
  TfLiteIntArrayFree(dims0);

  TfLiteIntArray* dims1 = TfLiteIntArrayCreate(1);
  dims1->data[0] = 1;
  EXPECT_EQ("[1]", GetShapeDebugString(dims1));
  TfLiteIntArrayFree(dims1);

  TfLiteIntArray* dims2 = TfLiteIntArrayCreate(2);
  dims2->data[0] = 2;
  dims2->data[1] = 3;
  EXPECT_EQ("[2,3]", GetShapeDebugString(dims2));
  TfLiteIntArrayFree(dims2);

  TfLiteIntArray* dims3 = TfLiteIntArrayCreate(3);
  dims3->data[0] = 4;
  dims3->data[1] = 5;
  dims3->data[2] = 6;
  EXPECT_EQ("[4,5,6]", GetShapeDebugString(dims3));
  TfLiteIntArrayFree(dims3);
}

TEST_F(KernelUtilTest, CheckAndPopulate) {
  // Create input.
  TfLiteTensor input = {};
  input.type = kTfLiteInt8;
  input.allocation_type = kTfLiteArenaRw;
  input.dims = TfLiteIntArrayCreate(1);
  input.dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {0.5, 5};
  input.params = input_quant;
  input.quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 0.5;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input.quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter.
  TfLiteTensor filter = {};
  filter.type = kTfLiteInt8;
  filter.allocation_type = kTfLiteArenaRw;
  filter.dims = TfLiteIntArrayCreate(4);
  filter.dims->data[0] = 3;
  filter.dims->data[1] = 4;
  filter.dims->data[2] = 5;
  filter.dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {0.25, 0};
  filter.params = filter_quant;
  filter.quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  filter_params->scale = TfLiteFloatArrayCreate(3);
  filter_params->scale->data[0] = 0.25;
  filter_params->scale->data[1] = 0.125;
  filter_params->scale->data[2] = 0.25;
  filter_params->zero_point = TfLiteIntArrayCreate(3);
  filter_params->zero_point->data[0] = 0;
  filter_params->zero_point->data[1] = 0;
  filter_params->zero_point->data[2] = 0;
  filter_params->quantized_dimension = 0;
  filter.quantization.params = reinterpret_cast<void*>(filter_params);

  // Create bias.
  TfLiteTensor bias = {};
  bias.type = kTfLiteInt32;
  bias.allocation_type = kTfLiteArenaRw;
  bias.dims = TfLiteIntArrayCreate(4);
  TfLiteQuantizationParams bias_quant = {0.125, 9};
  bias.params = bias_quant;
  bias.quantization.type = kTfLiteAffineQuantization;
  auto* bias_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  bias_params->scale = TfLiteFloatArrayCreate(3);
  bias_params->scale->data[0] = 0.125;
  bias_params->scale->data[1] = 0.0625;
  bias_params->scale->data[2] = 0.125;
  bias_params->zero_point = TfLiteIntArrayCreate(3);
  bias_params->zero_point->data[0] = 11;
  bias_params->zero_point->data[1] = 12;
  bias_params->zero_point->data[2] = 15;
  bias.quantization.params = reinterpret_cast<void*>(bias_params);

  // Create output.
  TfLiteTensor output = {};
  output.type = kTfLiteInt8;
  output.allocation_type = kTfLiteArenaRw;
  output.dims = nullptr;
  TfLiteQuantizationParams output_quant = {0.5, -128};
  output.params = output_quant;
  output.quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 0.5;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output.quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int32_t> per_channel_shift(3);

  // Call and verify results for per channel case.
  EXPECT_EQ(
      kTfLiteOk,
      PopulateConvolutionQuantizationParams(
          &context_, &input, &filter, &bias, &output, kTfLiteActRelu,
          &multiplier, &shift, &output_activation_min, &output_activation_max,
          per_channel_multiplier.data(), per_channel_shift.data()));
  EXPECT_THAT(per_channel_multiplier,
              ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ElementsAre(-1, -2, -1));

  // Release.
  TfLiteTensorFree(&input);
  TfLiteTensorFree(&filter);
  TfLiteTensorFree(&bias);
  TfLiteTensorFree(&output);
}

TEST_F(KernelUtilTest, CheckAndPopulateShift) {
  // Create input of type kTfLiteUInt8.
  TfLiteTensor input = {};
  input.type = kTfLiteUInt8;
  input.allocation_type = kTfLiteArenaRw;
  input.dims = TfLiteIntArrayCreate(1);
  input.dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {0.5, 5};
  input.params = input_quant;
  input.quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 0.5;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input.quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter of type kTfLiteUInt8.
  TfLiteTensor filter = {};
  filter.type = kTfLiteUInt8;
  filter.allocation_type = kTfLiteArenaRw;
  filter.dims = TfLiteIntArrayCreate(4);
  filter.dims->data[0] = 3;
  filter.dims->data[1] = 4;
  filter.dims->data[2] = 5;
  filter.dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {0.25, 0};
  filter.params = filter_quant;
  filter.quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  // Create scale of size one.
  filter_params->scale = TfLiteFloatArrayCreate(1);
  filter_params->scale->data[0] = 0.25;
  filter_params->zero_point = TfLiteIntArrayCreate(1);
  filter_params->zero_point->data[0] = 0;
  filter_params->quantized_dimension = 0;
  filter.quantization.params = reinterpret_cast<void*>(filter_params);

  // Create bias for kTfLiteUInt8.
  TfLiteTensor bias = {};
  bias.type = kTfLiteUInt8;
  bias.allocation_type = kTfLiteArenaRw;
  bias.dims = TfLiteIntArrayCreate(4);
  TfLiteQuantizationParams bias_quant = {0.125, 9};
  bias.params = bias_quant;
  bias.quantization.type = kTfLiteAffineQuantization;
  auto* bias_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  bias_params->scale = TfLiteFloatArrayCreate(3);
  bias_params->scale->data[0] = 0.125;
  bias_params->scale->data[1] = 0.0625;
  bias_params->scale->data[2] = 0.125;
  bias_params->zero_point = TfLiteIntArrayCreate(3);
  bias_params->zero_point->data[0] = 11;
  bias_params->zero_point->data[1] = 12;
  bias_params->zero_point->data[2] = 15;
  bias.quantization.params = reinterpret_cast<void*>(bias_params);

  // Create output for kTfLiteUInt8.
  TfLiteTensor output = {};
  output.type = kTfLiteUInt8;
  output.allocation_type = kTfLiteArenaRw;
  output.dims = nullptr;
  TfLiteQuantizationParams output_quant = {0.5, 128};
  output.params = output_quant;
  output.quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 0.5;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = 128;
  output.quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int> per_channel_shift(3);

  // Call and verify results for per channel case.
  EXPECT_EQ(
      kTfLiteOk,
      PopulateConvolutionQuantizationParams(
          &context_, &input, &filter, &bias, &output, kTfLiteActRelu,
          &multiplier, &shift, &output_activation_min, &output_activation_max,
          per_channel_multiplier.data(), per_channel_shift.data(), 3));
  // Since the filter scale has a size of one but the number of channels is
  // three, in our TC we expect three 1073741824 as output
  EXPECT_THAT(per_channel_multiplier,
              ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ElementsAre(-1, -1, -1));
  EXPECT_EQ(shift, 1);
  EXPECT_EQ(multiplier, 1073741824);

  // Release.
  TfLiteTensorFree(&input);
  TfLiteTensorFree(&filter);
  TfLiteTensorFree(&bias);
  TfLiteTensorFree(&output);
}

#ifndef __APPLE__  // Some Apple toolchains don't support std::ldexp
TEST_F(KernelUtilTest, CheckAndPopulateZeroValue) {
  // Create input.
  TfLiteTensor input = {};
  input.type = kTfLiteInt8;
  input.allocation_type = kTfLiteArenaRw;
  input.dims = TfLiteIntArrayCreate(1);
  input.dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {1, 5};
  input.params = input_quant;
  input.quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 1;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input.quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter.
  TfLiteTensor filter = {};
  filter.type = kTfLiteInt8;
  filter.allocation_type = kTfLiteArenaRw;
  filter.dims = TfLiteIntArrayCreate(4);
  filter.dims->data[0] = 3;
  filter.dims->data[1] = 4;
  filter.dims->data[2] = 5;
  filter.dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {4.6566129e-10, 0};
  filter.params = filter_quant;
  filter.quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  filter_params->scale = TfLiteFloatArrayCreate(3);
  filter_params->scale->data[0] = std::ldexp(1.0f, -31);
  filter_params->scale->data[1] = std::ldexp(1.0f, -32);
  filter_params->scale->data[2] = std::ldexp(1.0f, -33);
  filter_params->zero_point = TfLiteIntArrayCreate(3);
  filter_params->zero_point->data[0] = 0;
  filter_params->zero_point->data[1] = 0;
  filter_params->zero_point->data[2] = 0;
  filter_params->quantized_dimension = 0;
  filter.quantization.params = reinterpret_cast<void*>(filter_params);

  // Create bias.
  TfLiteTensor bias = {};
  bias.type = kTfLiteInt32;
  bias.allocation_type = kTfLiteArenaRw;
  bias.dims = TfLiteIntArrayCreate(4);
  TfLiteQuantizationParams bias_quant = {4.6566129e-10, 9};
  bias.params = bias_quant;
  bias.quantization.type = kTfLiteAffineQuantization;
  auto* bias_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  bias_params->scale = TfLiteFloatArrayCreate(3);
  bias_params->scale->data[0] = std::ldexp(1.0f, -31);
  bias_params->scale->data[1] = std::ldexp(1.0f, -32);
  bias_params->scale->data[2] = std::ldexp(1.0f, -33);
  bias_params->zero_point = TfLiteIntArrayCreate(3);
  bias_params->zero_point->data[0] = 11;
  bias_params->zero_point->data[1] = 12;
  bias_params->zero_point->data[2] = 15;
  bias.quantization.params = reinterpret_cast<void*>(bias_params);

  // Create output.
  TfLiteTensor output = {};
  output.type = kTfLiteInt8;
  output.allocation_type = kTfLiteArenaRw;
  output.dims = nullptr;
  TfLiteQuantizationParams output_quant = {1, -128};
  output.params = output_quant;
  output.quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 1;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output.quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int> per_channel_shift(3);

  // Call and verify results for per channel case.
  EXPECT_EQ(
      kTfLiteOk,
      PopulateConvolutionQuantizationParams(
          &context_, &input, &filter, &bias, &output, kTfLiteActRelu,
          &multiplier, &shift, &output_activation_min, &output_activation_max,
          per_channel_multiplier.data(), per_channel_shift.data(), 3));
  EXPECT_THAT(per_channel_multiplier, ElementsAre(1073741824, 1073741824, 0));
  EXPECT_THAT(per_channel_shift, ElementsAre(-30, -31, 0));

  // Release.
  TfLiteTensorFree(&input);
  TfLiteTensorFree(&filter);
  TfLiteTensorFree(&bias);
  TfLiteTensorFree(&output);
}
#endif

TEST_F(KernelUtilTest, CheckAndPopulateUint8) {
  // Create input.
  TfLiteTensor input = {};
  input.type = kTfLiteUInt8;
  input.allocation_type = kTfLiteArenaRw;
  input.dims = TfLiteIntArrayCreate(1);
  input.dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {1, 5};
  input.params = input_quant;
  input.quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 1;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input.quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter.
  TfLiteTensor filter = {};
  filter.type = kTfLiteUInt8;
  filter.allocation_type = kTfLiteArenaRw;
  filter.dims = TfLiteIntArrayCreate(4);
  filter.dims->data[0] = 3;
  filter.dims->data[1] = 4;
  filter.dims->data[2] = 5;
  filter.dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {4.6566129e-10, 0};
  filter.params = filter_quant;
  filter.quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  filter_params->scale = TfLiteFloatArrayCreate(1);
  int32_t two_pow_neg_31 = 0x30000000;  // 2^-31 so shift = -30.
  filter_params->scale->data[0] = *reinterpret_cast<float*>(&two_pow_neg_31);
  filter_params->zero_point = TfLiteIntArrayCreate(1);
  filter_params->zero_point->data[0] = 0;
  filter_params->quantized_dimension = 0;
  filter.quantization.params = reinterpret_cast<void*>(filter_params);

  // Create bias.
  TfLiteTensor bias = {};
  bias.type = kTfLiteInt32;
  bias.allocation_type = kTfLiteArenaRw;
  bias.dims = TfLiteIntArrayCreate(4);
  TfLiteQuantizationParams bias_quant = {4.6566129e-10, 9};
  bias.params = bias_quant;
  bias.quantization.type = kTfLiteAffineQuantization;
  auto* bias_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  bias_params->scale = TfLiteFloatArrayCreate(1);
  bias_params->scale->data[0] = 4.6566129e-10;  // 2^-31
  bias_params->zero_point = TfLiteIntArrayCreate(1);
  bias_params->zero_point->data[0] = 11;
  bias.quantization.params = reinterpret_cast<void*>(bias_params);

  // Create output.
  TfLiteTensor output = {};
  output.type = kTfLiteUInt8;
  output.allocation_type = kTfLiteArenaRw;
  output.dims = nullptr;
  TfLiteQuantizationParams output_quant = {1, -128};
  output.params = output_quant;
  output.quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 1;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output.quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int> per_channel_shift(3);

  // Call and verify results for per channel case.
  EXPECT_EQ(
      kTfLiteOk,
      PopulateConvolutionQuantizationParams(
          &context_, &input, &filter, &bias, &output, kTfLiteActRelu,
          &multiplier, &shift, &output_activation_min, &output_activation_max,
          per_channel_multiplier.data(), per_channel_shift.data(), 3));
  EXPECT_THAT(per_channel_multiplier,
              ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ElementsAre(-30, -30, -30));

  // Release.
  TfLiteTensorFree(&input);
  TfLiteTensorFree(&filter);
  TfLiteTensorFree(&bias);
  TfLiteTensorFree(&output);
}

TEST_F(KernelUtilTest, CheckAndPopulateWithoutBias) {
  // Create input.
  TfLiteTensor input = {};
  input.type = kTfLiteUInt8;
  input.allocation_type = kTfLiteArenaRw;
  input.dims = TfLiteIntArrayCreate(1);
  input.dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {1, 5};
  input.params = input_quant;
  input.quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 1;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input.quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter.
  TfLiteTensor filter = {};
  filter.type = kTfLiteUInt8;
  filter.allocation_type = kTfLiteArenaRw;
  filter.dims = TfLiteIntArrayCreate(4);
  filter.dims->data[0] = 3;
  filter.dims->data[1] = 4;
  filter.dims->data[2] = 5;
  filter.dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {4.6566129e-10, 0};
  filter.params = filter_quant;
  filter.quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  filter_params->scale = TfLiteFloatArrayCreate(1);
  int32_t two_pow_neg_31 = 0x30000000;  // 2^-31 so shift = -30.
  filter_params->scale->data[0] = *reinterpret_cast<float*>(&two_pow_neg_31);
  filter_params->zero_point = TfLiteIntArrayCreate(1);
  filter_params->zero_point->data[0] = 0;
  filter_params->quantized_dimension = 0;
  filter.quantization.params = reinterpret_cast<void*>(filter_params);

  // Create output.
  TfLiteTensor output = {};
  output.type = kTfLiteUInt8;
  output.allocation_type = kTfLiteArenaRw;
  output.dims = nullptr;
  TfLiteQuantizationParams output_quant = {1, -128};
  output.params = output_quant;
  output.quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 1;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output.quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int> per_channel_shift(3);

  // Call and verify results for per channel case.
  EXPECT_EQ(
      kTfLiteOk,
      PopulateConvolutionQuantizationParams(
          &context_, &input, &filter, nullptr, &output, kTfLiteActRelu,
          &multiplier, &shift, &output_activation_min, &output_activation_max,
          per_channel_multiplier.data(), per_channel_shift.data(), 3));
  EXPECT_THAT(per_channel_multiplier,
              ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ElementsAre(-30, -30, -30));

  // Release.
  TfLiteTensorFree(&input);
  TfLiteTensorFree(&filter);
  TfLiteTensorFree(&output);
}

TEST_F(KernelUtilTest, ActivationRangeQuantizedOverflow) {
  // Create output.
  TfLiteTensor output = {};
  output.type = kTfLiteUInt8;
  output.allocation_type = kTfLiteArenaRw;
  output.dims = nullptr;
  TfLiteQuantizationParams output_quant = {1e-10, -128};
  output.params = output_quant;
  output.quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 1;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output.quantization.params = reinterpret_cast<void*>(output_params);

  // For bounded activation, a too small scale value may cause overflow.
  // Make sure overflow error is handled gracefully.
  int32_t act_min, act_max;
  ASSERT_EQ(kTfLiteOk,
            CalculateActivationRangeQuantized(&context_, kTfLiteActRelu,
                                              &output, &act_min, &act_max));
  ASSERT_NE(kTfLiteOk,
            CalculateActivationRangeQuantized(&context_, kTfLiteActRelu6,
                                              &output, &act_min, &act_max));
  EXPECT_TRUE(absl::StrContains(
      context_.error, "no_integer_overflow_from_quantization was not true"));
  ASSERT_NE(kTfLiteOk,
            CalculateActivationRangeQuantized(&context_, kTfLiteActReluN1To1,
                                              &output, &act_min, &act_max));
  EXPECT_TRUE(absl::StrContains(
      context_.error, "no_integer_overflow_from_quantization was not true"));

  // Release.
  TfLiteTensorFree(&output);
}

TEST_F(KernelUtilTest, IsMobilePlatform) {
  // Note: This isn't meant to be exhaustive, as that would require replicating
  // the method's implementation, but it is a basic smoke check.
#if defined(__ANDROID__)
  EXPECT_TRUE(IsMobilePlatform());
#elif defined(__linux__)
  EXPECT_FALSE(IsMobilePlatform());
#elif defined(_WIN32)
  EXPECT_FALSE(IsMobilePlatform());
#endif
}

TEST_F(KernelUtilTest, HasUnspecifiedDimension) {
  TfLiteTensor tensor;
  TfLiteIntArray* shape_sig = TfLiteIntArrayCreate(3);
  shape_sig->data[0] = 1;
  shape_sig->data[1] = -1;
  shape_sig->data[2] = 3;
  tensor.dims_signature = shape_sig;

  EXPECT_TRUE(HasUnspecifiedDimension(&tensor));

  shape_sig->data[1] = 2;
  EXPECT_FALSE(HasUnspecifiedDimension(&tensor));

  TfLiteIntArrayFree(shape_sig);
}

}  // namespace
}  // namespace tflite
