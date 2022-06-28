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
class MHTracer_DTPStensorflowPSlitePStocoPStflitePStypes_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPStflitePStypes_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStflitePStypes_testDTcc() {
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
#include "tensorflow/lite/toco/tflite/types.h"

#include <complex>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace toco {

namespace tflite {
namespace {

using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;
using flatbuffers::Vector;

// These are types that exist in TF Mini but don't have a correspondence
// in TF Lite.
static const ArrayDataType kUnsupportedTocoTypes[] = {ArrayDataType::kNone};

// These are TF Lite types for which there is no correspondence in TF Mini.
static const ::tflite::TensorType kUnsupportedTfLiteTypes[] = {
    ::tflite::TensorType_FLOAT16};

// A little helper to match flatbuffer offsets.
MATCHER_P(HasOffset, value, "") { return arg.o == value; }

// Helper function that creates an array, writes it into a flatbuffer, and then
// reads it back in.
template <ArrayDataType T>
Array ToFlatBufferAndBack(std::initializer_list<::toco::DataType<T>> items) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePStypes_testDTcc mht_0(mht_0_v, 214, "", "./tensorflow/lite/toco/tflite/types_test.cc", "ToFlatBufferAndBack");

  // NOTE: This test does not construct the full buffers list. Since
  // Deserialize normally takes a buffer, we need to synthesize one and provide
  // an index that is non-zero so the buffer is not assumed to be empty.
  Array src;
  src.data_type = T;
  src.GetMutableBuffer<T>().data = items;

  Array result;
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(CreateTensor(builder, 0, DataType::Serialize(T),
                              /*buffer*/ 1));  // Can't use 0 which means empty.
  flatbuffers::FlatBufferBuilder buffer_builder;
  Offset<Vector<uint8_t>> data_buffer =
      DataBuffer::Serialize(src, &buffer_builder);
  buffer_builder.Finish(::tflite::CreateBuffer(buffer_builder, data_buffer));

  auto* tensor =
      flatbuffers::GetRoot<::tflite::Tensor>(builder.GetBufferPointer());
  auto* buffer =
      flatbuffers::GetRoot<::tflite::Buffer>(buffer_builder.GetBufferPointer());
  DataBuffer::Deserialize(*tensor, *buffer, &result);
  return result;
}

TEST(DataType, SupportedTypes) {
  std::vector<std::pair<ArrayDataType, ::tflite::TensorType>> testdata = {
      {ArrayDataType::kUint8, ::tflite::TensorType_UINT8},
      {ArrayDataType::kInt32, ::tflite::TensorType_INT32},
      {ArrayDataType::kUint32, ::tflite::TensorType_UINT32},
      {ArrayDataType::kInt64, ::tflite::TensorType_INT64},
      {ArrayDataType::kFloat, ::tflite::TensorType_FLOAT32},
      {ArrayDataType::kBool, ::tflite::TensorType_BOOL},
      {ArrayDataType::kComplex64, ::tflite::TensorType_COMPLEX64}};
  for (auto x : testdata) {
    EXPECT_EQ(x.second, DataType::Serialize(x.first));
    EXPECT_EQ(x.first, DataType::Deserialize(x.second));
  }
}

TEST(DataType, UnsupportedTypes) {
  for (::tflite::TensorType t : kUnsupportedTfLiteTypes) {
    EXPECT_DEATH(DataType::Deserialize(t), "Unhandled tensor type.");
  }

  // Unsupported types are all serialized as FLOAT32 currently.
  for (ArrayDataType t : kUnsupportedTocoTypes) {
    EXPECT_EQ(::tflite::TensorType_FLOAT32, DataType::Serialize(t));
  }
}

TEST(DataBuffer, EmptyBuffers) {
  flatbuffers::FlatBufferBuilder builder;
  Array array;
  EXPECT_THAT(DataBuffer::Serialize(array, &builder), HasOffset(0));

  builder.Finish(::tflite::CreateTensor(builder));
  auto* tensor =
      flatbuffers::GetRoot<::tflite::Tensor>(builder.GetBufferPointer());
  flatbuffers::FlatBufferBuilder buffer_builder;
  Offset<Vector<uint8_t>> v = buffer_builder.CreateVector<uint8_t>({});
  buffer_builder.Finish(::tflite::CreateBuffer(buffer_builder, v));
  auto* buffer =
      flatbuffers::GetRoot<::tflite::Buffer>(buffer_builder.GetBufferPointer());

  DataBuffer::Deserialize(*tensor, *buffer, &array);
  EXPECT_EQ(nullptr, array.buffer);
}

TEST(DataBuffer, UnsupportedTypes) {
  for (ArrayDataType t : kUnsupportedTocoTypes) {
    flatbuffers::FlatBufferBuilder builder;
    Array array;
    array.data_type = t;
    array.GetMutableBuffer<ArrayDataType::kFloat>();  // This is OK.
    EXPECT_DEATH(DataBuffer::Serialize(array, &builder),
                 "Unhandled array data type.");
  }

  for (::tflite::TensorType t : kUnsupportedTfLiteTypes) {
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(::tflite::CreateTensor(builder, 0, t, /*buffer*/ 1));
    flatbuffers::FlatBufferBuilder buffer_builder;
    Offset<Vector<uint8_t>> v = buffer_builder.CreateVector<uint8_t>({1});
    buffer_builder.Finish(::tflite::CreateBuffer(buffer_builder, v));
    auto* buffer = flatbuffers::GetRoot<::tflite::Buffer>(
        buffer_builder.GetBufferPointer());
    auto* tensor =
        flatbuffers::GetRoot<::tflite::Tensor>(builder.GetBufferPointer());
    Array array;
    EXPECT_DEATH(DataBuffer::Deserialize(*tensor, *buffer, &array),
                 "Unhandled tensor type.");
  }
}

TEST(DataBuffer, Float) {
  Array recovered = ToFlatBufferAndBack<ArrayDataType::kFloat>({1.0f, 2.0f});
  EXPECT_THAT(recovered.GetBuffer<ArrayDataType::kFloat>().data,
              ::testing::ElementsAre(1.0f, 2.0f));
}

TEST(DataBuffer, Uint8) {
  Array recovered = ToFlatBufferAndBack<ArrayDataType::kUint8>({127, 244});
  EXPECT_THAT(recovered.GetBuffer<ArrayDataType::kUint8>().data,
              ::testing::ElementsAre(127, 244));
}

TEST(DataBuffer, Int32) {
  Array recovered = ToFlatBufferAndBack<ArrayDataType::kInt32>({1, 1 << 30});
  EXPECT_THAT(recovered.GetBuffer<ArrayDataType::kInt32>().data,
              ::testing::ElementsAre(1, 1 << 30));
}

TEST(DataBuffer, Uint32) {
  Array recovered = ToFlatBufferAndBack<ArrayDataType::kUint32>({1, 1U << 31});
  EXPECT_THAT(recovered.GetBuffer<ArrayDataType::kUint32>().data,
              ::testing::ElementsAre(1, 1U << 31));
}

TEST(DataBuffer, Int16) {
  Array recovered = ToFlatBufferAndBack<ArrayDataType::kInt16>({1, 1 << 14});
  EXPECT_THAT(recovered.GetBuffer<ArrayDataType::kInt16>().data,
              ::testing::ElementsAre(1, 1 << 14));
}

TEST(DataBuffer, String) {
  Array recovered = ToFlatBufferAndBack<ArrayDataType::kString>(
      {"AA", "BBB", "Best. String. Ever."});
  EXPECT_THAT(recovered.GetBuffer<ArrayDataType::kString>().data,
              ::testing::ElementsAre("AA", "BBB", "Best. String. Ever."));
}

TEST(DataBuffer, Bool) {
  Array recovered =
      ToFlatBufferAndBack<ArrayDataType::kBool>({true, false, true});
  EXPECT_THAT(recovered.GetBuffer<ArrayDataType::kBool>().data,
              ::testing::ElementsAre(true, false, true));
}

TEST(DataBuffer, Complex64) {
  Array recovered = ToFlatBufferAndBack<ArrayDataType::kComplex64>(
      {std::complex<float>(1.0f, 2.0f), std::complex<float>(3.0f, 4.0f)});
  EXPECT_THAT(recovered.GetBuffer<ArrayDataType::kComplex64>().data,
              ::testing::ElementsAre(std::complex<float>(1.0f, 2.0f),
                                     std::complex<float>(3.0f, 4.0f)));
}

TEST(Padding, All) {
  EXPECT_EQ(::tflite::Padding_SAME, Padding::Serialize(PaddingType::kSame));
  EXPECT_EQ(PaddingType::kSame, Padding::Deserialize(::tflite::Padding_SAME));

  EXPECT_EQ(::tflite::Padding_VALID, Padding::Serialize(PaddingType::kValid));
  EXPECT_EQ(PaddingType::kValid, Padding::Deserialize(::tflite::Padding_VALID));

  EXPECT_DEATH(Padding::Serialize(static_cast<PaddingType>(10000)),
               "Unhandled padding type.");
  EXPECT_DEATH(Padding::Deserialize(10000), "Unhandled padding.");
}

TEST(ActivationFunction, All) {
  std::vector<
      std::pair<FusedActivationFunctionType, ::tflite::ActivationFunctionType>>
      testdata = {{FusedActivationFunctionType::kNone,
                   ::tflite::ActivationFunctionType_NONE},
                  {FusedActivationFunctionType::kRelu,
                   ::tflite::ActivationFunctionType_RELU},
                  {FusedActivationFunctionType::kRelu6,
                   ::tflite::ActivationFunctionType_RELU6},
                  {FusedActivationFunctionType::kRelu1,
                   ::tflite::ActivationFunctionType_RELU_N1_TO_1}};
  for (auto x : testdata) {
    EXPECT_EQ(x.second, ActivationFunction::Serialize(x.first));
    EXPECT_EQ(x.first, ActivationFunction::Deserialize(x.second));
  }

  EXPECT_DEATH(ActivationFunction::Serialize(
                   static_cast<FusedActivationFunctionType>(10000)),
               "Unhandled fused activation function type.");
  EXPECT_DEATH(ActivationFunction::Deserialize(10000),
               "Unhandled fused activation function type.");
}

}  // namespace
}  // namespace tflite

}  // namespace toco
