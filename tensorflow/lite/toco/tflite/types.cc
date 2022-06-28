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
class MHTracer_DTPStensorflowPSlitePStocoPStflitePStypesDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPStflitePStypesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStflitePStypesDTcc() {
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
#include "tensorflow/lite/string_util.h"

namespace toco {

namespace tflite {

namespace {

DataBuffer::FlatBufferOffset CopyStringToBuffer(
    const Array& array, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePStypesDTcc mht_0(mht_0_v, 194, "", "./tensorflow/lite/toco/tflite/types.cc", "CopyStringToBuffer");

  const auto& src_data = array.GetBuffer<ArrayDataType::kString>().data;
  ::tflite::DynamicBuffer dyn_buffer;
  for (const std::string& str : src_data) {
    dyn_buffer.AddString(str.c_str(), str.length());
  }
  char* tensor_buffer;
  int bytes = dyn_buffer.WriteToBuffer(&tensor_buffer);
  std::vector<uint8_t> dst_data(bytes);
  memcpy(dst_data.data(), tensor_buffer, bytes);
  free(tensor_buffer);
  return builder->CreateVector(dst_data.data(), bytes);
}

// vector<bool> may be implemented using a bit-set, so we can't just
// reinterpret_cast, accessing its data as vector<bool> and let flatbuffer
// CreateVector handle it.
// Background: https://isocpp.org/blog/2012/11/on-vectorbool
DataBuffer::FlatBufferOffset CopyBoolToBuffer(
    const Array& array, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePStypesDTcc mht_1(mht_1_v, 216, "", "./tensorflow/lite/toco/tflite/types.cc", "CopyBoolToBuffer");

  const auto& src_data = array.GetBuffer<ArrayDataType::kBool>().data;
  return builder->CreateVector(src_data);
}

template <ArrayDataType T>
DataBuffer::FlatBufferOffset CopyBuffer(
    const Array& array, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePStypesDTcc mht_2(mht_2_v, 226, "", "./tensorflow/lite/toco/tflite/types.cc", "CopyBuffer");

  using NativeT = ::toco::DataType<T>;
  const auto& src_data = array.GetBuffer<T>().data;
  const uint8_t* dst_data = reinterpret_cast<const uint8_t*>(src_data.data());
  auto size = src_data.size() * sizeof(NativeT);
  return builder->CreateVector(dst_data, size);
}

void CopyStringFromBuffer(const ::tflite::Buffer& buffer, Array* array) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePStypesDTcc mht_3(mht_3_v, 237, "", "./tensorflow/lite/toco/tflite/types.cc", "CopyStringFromBuffer");

  auto* src_data = reinterpret_cast<const char*>(buffer.data()->data());
  std::vector<std::string>* dst_data =
      &array->GetMutableBuffer<ArrayDataType::kString>().data;
  int32_t num_strings = ::tflite::GetStringCount(src_data);
  for (int i = 0; i < num_strings; i++) {
    ::tflite::StringRef str_ref = ::tflite::GetString(src_data, i);
    std::string this_str(str_ref.str, str_ref.len);
    dst_data->push_back(this_str);
  }
}

template <ArrayDataType T>
void CopyBuffer(const ::tflite::Buffer& buffer, Array* array) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePStypesDTcc mht_4(mht_4_v, 253, "", "./tensorflow/lite/toco/tflite/types.cc", "CopyBuffer");

  using NativeT = ::toco::DataType<T>;
  auto* src_buffer = buffer.data();
  const NativeT* src_data =
      reinterpret_cast<const NativeT*>(src_buffer->data());
  int num_items = src_buffer->size() / sizeof(NativeT);

  std::vector<NativeT>* dst_data = &array->GetMutableBuffer<T>().data;
  for (int i = 0; i < num_items; ++i) {
    dst_data->push_back(*src_data);
    ++src_data;
  }
}
}  // namespace

::tflite::TensorType DataType::Serialize(ArrayDataType array_data_type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePStypesDTcc mht_5(mht_5_v, 271, "", "./tensorflow/lite/toco/tflite/types.cc", "DataType::Serialize");

  switch (array_data_type) {
    case ArrayDataType::kFloat:
      return ::tflite::TensorType_FLOAT32;
    case ArrayDataType::kInt16:
      return ::tflite::TensorType_INT16;
    case ArrayDataType::kInt32:
      return ::tflite::TensorType_INT32;
    case ArrayDataType::kUint32:
      return ::tflite::TensorType_UINT32;
    case ArrayDataType::kInt64:
      return ::tflite::TensorType_INT64;
    case ArrayDataType::kUint8:
      return ::tflite::TensorType_UINT8;
    case ArrayDataType::kUint16:
      return ::tflite::TensorType_UINT16;
    case ArrayDataType::kString:
      return ::tflite::TensorType_STRING;
    case ArrayDataType::kBool:
      return ::tflite::TensorType_BOOL;
    case ArrayDataType::kComplex64:
      return ::tflite::TensorType_COMPLEX64;
    default:
      // FLOAT32 is filled for unknown data types.
      // TODO(ycling): Implement type inference in TF Lite interpreter.
      return ::tflite::TensorType_FLOAT32;
  }
}

ArrayDataType DataType::Deserialize(int tensor_type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePStypesDTcc mht_6(mht_6_v, 303, "", "./tensorflow/lite/toco/tflite/types.cc", "DataType::Deserialize");

  switch (::tflite::TensorType(tensor_type)) {
    case ::tflite::TensorType_FLOAT32:
      return ArrayDataType::kFloat;
    case ::tflite::TensorType_INT16:
      return ArrayDataType::kInt16;
    case ::tflite::TensorType_INT32:
      return ArrayDataType::kInt32;
    case ::tflite::TensorType_UINT32:
      return ArrayDataType::kUint32;
    case ::tflite::TensorType_INT64:
      return ArrayDataType::kInt64;
    case ::tflite::TensorType_STRING:
      return ArrayDataType::kString;
    case ::tflite::TensorType_UINT8:
      return ArrayDataType::kUint8;
    case ::tflite::TensorType_UINT16:
      return ArrayDataType::kUint16;
    case ::tflite::TensorType_BOOL:
      return ArrayDataType::kBool;
    case ::tflite::TensorType_COMPLEX64:
      return ArrayDataType::kComplex64;
    default:
      LOG(FATAL) << "Unhandled tensor type '" << tensor_type << "'.";
  }
}

flatbuffers::Offset<flatbuffers::Vector<uint8_t>> DataBuffer::Serialize(
    const Array& array, flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePStypesDTcc mht_7(mht_7_v, 334, "", "./tensorflow/lite/toco/tflite/types.cc", "DataBuffer::Serialize");

  if (!array.buffer) return 0;  // an empty buffer, usually an output.

  switch (array.data_type) {
    case ArrayDataType::kFloat:
      return CopyBuffer<ArrayDataType::kFloat>(array, builder);
    case ArrayDataType::kInt16:
      return CopyBuffer<ArrayDataType::kInt16>(array, builder);
    case ArrayDataType::kInt32:
      return CopyBuffer<ArrayDataType::kInt32>(array, builder);
    case ArrayDataType::kUint32:
      return CopyBuffer<ArrayDataType::kUint32>(array, builder);
    case ArrayDataType::kInt64:
      return CopyBuffer<ArrayDataType::kInt64>(array, builder);
    case ArrayDataType::kString:
      return CopyStringToBuffer(array, builder);
    case ArrayDataType::kUint8:
      return CopyBuffer<ArrayDataType::kUint8>(array, builder);
    case ArrayDataType::kBool:
      return CopyBoolToBuffer(array, builder);
    case ArrayDataType::kComplex64:
      return CopyBuffer<ArrayDataType::kComplex64>(array, builder);
    default:
      LOG(FATAL) << "Unhandled array data type.";
  }
}

void DataBuffer::Deserialize(const ::tflite::Tensor& tensor,
                             const ::tflite::Buffer& buffer, Array* array) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePStypesDTcc mht_8(mht_8_v, 365, "", "./tensorflow/lite/toco/tflite/types.cc", "DataBuffer::Deserialize");

  if (tensor.buffer() == 0) return;      // an empty buffer, usually an output.
  if (buffer.data() == nullptr) return;  // a non-defined buffer.

  switch (tensor.type()) {
    case ::tflite::TensorType_FLOAT32:
      return CopyBuffer<ArrayDataType::kFloat>(buffer, array);
    case ::tflite::TensorType_INT16:
      return CopyBuffer<ArrayDataType::kInt16>(buffer, array);
    case ::tflite::TensorType_INT32:
      return CopyBuffer<ArrayDataType::kInt32>(buffer, array);
    case ::tflite::TensorType_UINT32:
      return CopyBuffer<ArrayDataType::kUint32>(buffer, array);
    case ::tflite::TensorType_INT64:
      return CopyBuffer<ArrayDataType::kInt64>(buffer, array);
    case ::tflite::TensorType_STRING:
      return CopyStringFromBuffer(buffer, array);
    case ::tflite::TensorType_UINT8:
      return CopyBuffer<ArrayDataType::kUint8>(buffer, array);
    case ::tflite::TensorType_BOOL:
      return CopyBuffer<ArrayDataType::kBool>(buffer, array);
    case ::tflite::TensorType_COMPLEX64:
      return CopyBuffer<ArrayDataType::kComplex64>(buffer, array);
    default:
      LOG(FATAL) << "Unhandled tensor type.";
  }
}

::tflite::Padding Padding::Serialize(PaddingType padding_type) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePStypesDTcc mht_9(mht_9_v, 396, "", "./tensorflow/lite/toco/tflite/types.cc", "Padding::Serialize");

  switch (padding_type) {
    case PaddingType::kSame:
      return ::tflite::Padding_SAME;
    case PaddingType::kValid:
      return ::tflite::Padding_VALID;
    default:
      LOG(FATAL) << "Unhandled padding type.";
  }
}

PaddingType Padding::Deserialize(int padding) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePStypesDTcc mht_10(mht_10_v, 410, "", "./tensorflow/lite/toco/tflite/types.cc", "Padding::Deserialize");

  switch (::tflite::Padding(padding)) {
    case ::tflite::Padding_SAME:
      return PaddingType::kSame;
    case ::tflite::Padding_VALID:
      return PaddingType::kValid;
    default:
      LOG(FATAL) << "Unhandled padding.";
  }
}

::tflite::ActivationFunctionType ActivationFunction::Serialize(
    FusedActivationFunctionType faf_type) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePStypesDTcc mht_11(mht_11_v, 425, "", "./tensorflow/lite/toco/tflite/types.cc", "ActivationFunction::Serialize");

  switch (faf_type) {
    case FusedActivationFunctionType::kNone:
      return ::tflite::ActivationFunctionType_NONE;
    case FusedActivationFunctionType::kRelu:
      return ::tflite::ActivationFunctionType_RELU;
    case FusedActivationFunctionType::kRelu6:
      return ::tflite::ActivationFunctionType_RELU6;
    case FusedActivationFunctionType::kRelu1:
      return ::tflite::ActivationFunctionType_RELU_N1_TO_1;
    default:
      LOG(FATAL) << "Unhandled fused activation function type.";
  }
}

FusedActivationFunctionType ActivationFunction::Deserialize(
    int activation_function) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePStypesDTcc mht_12(mht_12_v, 444, "", "./tensorflow/lite/toco/tflite/types.cc", "ActivationFunction::Deserialize");

  switch (::tflite::ActivationFunctionType(activation_function)) {
    case ::tflite::ActivationFunctionType_NONE:
      return FusedActivationFunctionType::kNone;
    case ::tflite::ActivationFunctionType_RELU:
      return FusedActivationFunctionType::kRelu;
    case ::tflite::ActivationFunctionType_RELU6:
      return FusedActivationFunctionType::kRelu6;
    case ::tflite::ActivationFunctionType_RELU_N1_TO_1:
      return FusedActivationFunctionType::kRelu1;
    default:
      LOG(FATAL) << "Unhandled fused activation function type.";
  }
}

}  // namespace tflite

}  // namespace toco
