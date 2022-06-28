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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSutil_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSutil_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSutil_testDTcc() {
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
#include "tensorflow/lite/delegates/flex/util.h"

#include <cstdarg>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace flex {
namespace {

using tensorflow::DT_FLOAT;
using tensorflow::DT_INT32;
using tensorflow::Tensor;
using ::testing::ElementsAre;

struct TestContext : public TfLiteContext {
  string error;
  std::vector<int> new_size;
};

void ReportError(TfLiteContext* context, const char* format, ...) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("format: \"" + (format == nullptr ? std::string("nullptr") : std::string((char*)format)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSutil_testDTcc mht_0(mht_0_v, 214, "", "./tensorflow/lite/delegates/flex/util_test.cc", "ReportError");

  TestContext* c = static_cast<TestContext*>(context);
  const size_t kBufferSize = 1024;
  char temp_buffer[kBufferSize];

  va_list args;
  va_start(args, format);
  vsnprintf(temp_buffer, kBufferSize, format, args);
  va_end(args);

  c->error = temp_buffer;
}

TfLiteStatus ResizeTensor(TfLiteContext* context, TfLiteTensor* tensor,
                          TfLiteIntArray* new_size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSutil_testDTcc mht_1(mht_1_v, 231, "", "./tensorflow/lite/delegates/flex/util_test.cc", "ResizeTensor");

  TestContext* c = static_cast<TestContext*>(context);
  c->new_size.clear();
  for (int i = 0; i < new_size->size; ++i) {
    c->new_size.push_back(new_size->data[i]);
  }
  TfLiteIntArrayFree(new_size);
  return kTfLiteOk;
}

TEST(UtilTest, ConvertStatus) {
  TestContext context;
  context.ReportError = ReportError;

  EXPECT_EQ(ConvertStatus(&context, tensorflow::errors::Internal("Some Error")),
            kTfLiteError);
  EXPECT_EQ(context.error, "Some Error");

  context.error.clear();
  EXPECT_EQ(ConvertStatus(&context, tensorflow::Status()), kTfLiteOk);
  EXPECT_TRUE(context.error.empty());
}

TEST(UtilTest, CopyShapeAndType) {
  TestContext context;
  context.ReportError = ReportError;
  context.ResizeTensor = ResizeTensor;

  TfLiteTensor dst;

  EXPECT_EQ(CopyShapeAndType(&context, Tensor(), &dst), kTfLiteOk);
  EXPECT_THAT(context.new_size, ElementsAre(0));
  EXPECT_EQ(dst.type, kTfLiteFloat32);

  EXPECT_EQ(CopyShapeAndType(&context, Tensor(DT_FLOAT, {1, 2}), &dst),
            kTfLiteOk);
  EXPECT_THAT(context.new_size, ElementsAre(1, 2));
  EXPECT_EQ(dst.type, kTfLiteFloat32);

  EXPECT_EQ(CopyShapeAndType(&context, Tensor(DT_INT32, {1, 2}), &dst),
            kTfLiteOk);
  EXPECT_THAT(context.new_size, ElementsAre(1, 2));
  EXPECT_EQ(dst.type, kTfLiteInt32);

  EXPECT_EQ(CopyShapeAndType(&context, Tensor(DT_FLOAT, {1LL << 44, 2}), &dst),
            kTfLiteError);
  EXPECT_EQ(context.error,
            "Dimension value in TensorFlow shape is larger than supported by "
            "TF Lite");

  EXPECT_EQ(
      CopyShapeAndType(&context, Tensor(tensorflow::DT_HALF, {1, 2}), &dst),
      kTfLiteOk);
  EXPECT_THAT(context.new_size, ElementsAre(1, 2));
  EXPECT_EQ(dst.type, kTfLiteFloat16);
}

TEST(UtilTest, TypeConversionsFromTFLite) {
  EXPECT_EQ(TF_FLOAT, GetTensorFlowDataType(kTfLiteNoType));
  EXPECT_EQ(TF_FLOAT, GetTensorFlowDataType(kTfLiteFloat32));
  EXPECT_EQ(TF_HALF, GetTensorFlowDataType(kTfLiteFloat16));
  EXPECT_EQ(TF_DOUBLE, GetTensorFlowDataType(kTfLiteFloat64));
  EXPECT_EQ(TF_INT16, GetTensorFlowDataType(kTfLiteInt16));
  EXPECT_EQ(TF_INT32, GetTensorFlowDataType(kTfLiteInt32));
  EXPECT_EQ(TF_UINT8, GetTensorFlowDataType(kTfLiteUInt8));
  EXPECT_EQ(TF_INT64, GetTensorFlowDataType(kTfLiteInt64));
  EXPECT_EQ(TF_UINT64, GetTensorFlowDataType(kTfLiteUInt64));
  EXPECT_EQ(TF_COMPLEX64, GetTensorFlowDataType(kTfLiteComplex64));
  EXPECT_EQ(TF_COMPLEX128, GetTensorFlowDataType(kTfLiteComplex128));
  EXPECT_EQ(TF_STRING, GetTensorFlowDataType(kTfLiteString));
  EXPECT_EQ(TF_BOOL, GetTensorFlowDataType(kTfLiteBool));
  EXPECT_EQ(TF_RESOURCE, GetTensorFlowDataType(kTfLiteResource));
  EXPECT_EQ(TF_VARIANT, GetTensorFlowDataType(kTfLiteVariant));
}

TEST(UtilTest, TypeConversionsFromTensorFlow) {
  EXPECT_EQ(kTfLiteFloat16, GetTensorFlowLiteType(TF_HALF));
  EXPECT_EQ(kTfLiteFloat32, GetTensorFlowLiteType(TF_FLOAT));
  EXPECT_EQ(kTfLiteFloat64, GetTensorFlowLiteType(TF_DOUBLE));
  EXPECT_EQ(kTfLiteInt16, GetTensorFlowLiteType(TF_INT16));
  EXPECT_EQ(kTfLiteInt32, GetTensorFlowLiteType(TF_INT32));
  EXPECT_EQ(kTfLiteUInt8, GetTensorFlowLiteType(TF_UINT8));
  EXPECT_EQ(kTfLiteInt64, GetTensorFlowLiteType(TF_INT64));
  EXPECT_EQ(kTfLiteUInt64, GetTensorFlowLiteType(TF_UINT64));
  EXPECT_EQ(kTfLiteComplex64, GetTensorFlowLiteType(TF_COMPLEX64));
  EXPECT_EQ(kTfLiteComplex128, GetTensorFlowLiteType(TF_COMPLEX128));
  EXPECT_EQ(kTfLiteString, GetTensorFlowLiteType(TF_STRING));
  EXPECT_EQ(kTfLiteBool, GetTensorFlowLiteType(TF_BOOL));
  EXPECT_EQ(kTfLiteResource, GetTensorFlowLiteType(TF_RESOURCE));
  EXPECT_EQ(kTfLiteVariant, GetTensorFlowLiteType(TF_VARIANT));
}

TEST(UtilTest, GetTfLiteResourceIdentifier) {
  // Constructs a fake resource tensor.
  TfLiteTensor tensor;
  tensor.allocation_type = kTfLiteDynamic;
  tensor.type = kTfLiteResource;
  std::vector<int> dims = {1};
  tensor.dims = ConvertVectorToTfLiteIntArray(dims);
  tensor.data.raw = nullptr;
  TfLiteTensorRealloc(sizeof(int32_t), &tensor);
  tensor.delegate = nullptr;
  tensor.data.i32[0] = 1;

  EXPECT_EQ(TfLiteResourceIdentifier(&tensor), "tflite_resource_variable:1");
  TfLiteIntArrayFree(tensor.dims);
  TfLiteTensorDataFree(&tensor);
}

TEST(UtilTest, GetTfLiteResourceTensorFromResourceHandle) {
  tensorflow::ResourceHandle handle;
  handle.set_name("tflite_resource_variable:1");

  TfLiteTensor tensor;
  tensor.allocation_type = kTfLiteDynamic;
  tensor.type = kTfLiteResource;
  tensor.data.raw = nullptr;
  std::vector<int> dims = {1};
  tensor.dims = ConvertVectorToTfLiteIntArray(dims);
  EXPECT_TRUE(GetTfLiteResourceTensorFromResourceHandle(handle, &tensor));
  EXPECT_EQ(tensor.data.i32[0], 1);

  TfLiteIntArrayFree(tensor.dims);
  TfLiteTensorDataFree(&tensor);
}

TEST(UtilTest, CreateTfTensorFromTfLiteTensorResourceOrVariant) {
  TfLiteTensor tensor;
  tensor.type = kTfLiteResource;
  EXPECT_EQ(CreateTfTensorFromTfLiteTensor(&tensor).status().code(),
            tensorflow::error::INVALID_ARGUMENT);
  tensor.type = kTfLiteVariant;
  EXPECT_EQ(CreateTfTensorFromTfLiteTensor(&tensor).status().code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST(UtilTest, CreateTfTensorFromTfLiteTensorFloat) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.type = kTfLiteFloat32;
  tflite_tensor.allocation_type = kTfLiteDynamic;
  tflite_tensor.sparsity = nullptr;
  tflite_tensor.dims_signature = nullptr;

  TfLiteQuantization quant;
  quant.type = kTfLiteNoQuantization;
  quant.params = nullptr;
  tflite_tensor.quantization = quant;

  TfLiteIntArray* dims = TfLiteIntArrayCreate(2);
  dims->data[0] = 1;
  dims->data[1] = 3;
  tflite_tensor.dims = dims;
  float data_arr[] = {1.1, 0.456, 0.322};
  std::vector<float_t> data(std::begin(data_arr), std::end(data_arr));
  size_t num_bytes = data.size() * sizeof(float_t);
  tflite_tensor.data.raw = static_cast<char*>(malloc(num_bytes));
  memcpy(tflite_tensor.data.raw, data.data(), num_bytes);
  tflite_tensor.bytes = num_bytes;

  auto tf_tensor_or = CreateTfTensorFromTfLiteTensor(&tflite_tensor);
  EXPECT_TRUE(tf_tensor_or.ok());
  tensorflow::Tensor tf_tensor = tf_tensor_or.ValueOrDie();
  EXPECT_EQ(tf_tensor.NumElements(), 3);
  auto* tf_data = static_cast<float_t*>(tf_tensor.data());
  for (float weight : data_arr) {
    EXPECT_EQ(*tf_data, weight);
    tf_data++;
  }

  TfLiteTensorFree(&tflite_tensor);
}

TEST(UtilTest, CreateTfTensorFromTfLiteTensorString) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.type = kTfLiteString;
  tflite_tensor.is_variable = false;
  tflite_tensor.sparsity = nullptr;
  tflite_tensor.data.raw = nullptr;
  tflite_tensor.dims_signature = nullptr;
  tflite_tensor.allocation_type = kTfLiteArenaRw;

  TfLiteQuantization quant;
  quant.type = kTfLiteNoQuantization;
  quant.params = nullptr;
  tflite_tensor.quantization = quant;

  TfLiteIntArray* dims = TfLiteIntArrayCreate(2);
  dims->data[0] = 1;
  dims->data[1] = 2;
  tflite_tensor.dims = dims;
  std::string data_arr[] = {std::string("a_str\0ing", 9), "b_string"};
  tflite::DynamicBuffer buf;
  for (const auto& value : data_arr) {
    buf.AddString(value.data(), value.length());
  }
  buf.WriteToTensor(&tflite_tensor, nullptr);

  auto tf_tensor_or = CreateTfTensorFromTfLiteTensor(&tflite_tensor);
  EXPECT_TRUE(tf_tensor_or.ok());
  tensorflow::Tensor tf_tensor = tf_tensor_or.ValueOrDie();
  EXPECT_EQ(tf_tensor.NumElements(), 2);
  auto* tf_data = static_cast<tensorflow::tstring*>(tf_tensor.data());
  for (const auto& str : data_arr) {
    EXPECT_EQ(*tf_data, str);
    tf_data++;
  }
  TfLiteTensorFree(&tflite_tensor);
}

}  // namespace
}  // namespace flex
}  // namespace tflite
