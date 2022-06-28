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
class MHTracer_DTPStensorflowPSlitePSstring_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSstring_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSstring_utilDTcc() {
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

#include "tensorflow/lite/string_util.h"

#include <stddef.h>
#include <stdint.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#include "tensorflow/lite/c/common.h"

namespace tflite {

void DynamicBuffer::AddString(const char* str, size_t len) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPSlitePSstring_utilDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/string_util.cc", "DynamicBuffer::AddString");

  data_.resize(data_.size() + len);
  memcpy(data_.data() + offset_.back(), str, len);
  offset_.push_back(offset_.back() + len);
}

void DynamicBuffer::AddString(const StringRef& string) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSstring_utilDTcc mht_1(mht_1_v, 208, "", "./tensorflow/lite/string_util.cc", "DynamicBuffer::AddString");

  AddString(string.str, string.len);
}

void DynamicBuffer::AddJoinedString(const std::vector<StringRef>& strings,
                                    char separator) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("separator: '" + std::string(1, separator) + "'");
   MHTracer_DTPStensorflowPSlitePSstring_utilDTcc mht_2(mht_2_v, 217, "", "./tensorflow/lite/string_util.cc", "DynamicBuffer::AddJoinedString");

  StringRef ref;
  ref.str = &separator;
  ref.len = 1;
  AddJoinedString(strings, ref);
}

void DynamicBuffer::AddJoinedString(const std::vector<StringRef>& strings,
                                    StringRef separator) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSstring_utilDTcc mht_3(mht_3_v, 228, "", "./tensorflow/lite/string_util.cc", "DynamicBuffer::AddJoinedString");

  // Resize the data buffer.
  int total_len = (strings.size() - 1) * separator.len;
  for (StringRef ref : strings) {
    total_len += ref.len;
  }
  data_.resize(data_.size() + total_len);

  char* dst = data_.data() + offset_.back();
  for (size_t i = 0; i < strings.size(); ++i) {
    // Fill separator if not first string.
    if (i != 0) {
      memcpy(dst, separator.str, separator.len);
      dst += separator.len;
    }

    // Fill content of the string.
    memcpy(dst, strings[i].str, strings[i].len);
    dst += strings[i].len;
  }
  offset_.push_back(offset_.back() + total_len);
}

int DynamicBuffer::WriteToBuffer(char** buffer) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSstring_utilDTcc mht_4(mht_4_v, 254, "", "./tensorflow/lite/string_util.cc", "DynamicBuffer::WriteToBuffer");

  // Allocate sufficient memory to tensor buffer.
  int32_t num_strings = offset_.size() - 1;
  // Total bytes include:
  //   * size of content (data_.size)
  //   * offset of each tensor (sizeof(int32_t) * num_strings)
  //   * length of whole buffer (int32_t)
  //   * num of strings (int32_t).
  int32_t bytes = data_.size()                            // size of content
                  + sizeof(int32_t) * (num_strings + 2);  // size of header

  // Caller will take ownership of buffer.
  *buffer = reinterpret_cast<char*>(malloc(bytes));

  // Set num of string
  memcpy(*buffer, &num_strings, sizeof(int32_t));

  // Set offset of strings.
  int32_t start = sizeof(int32_t) * (num_strings + 2);
  for (size_t i = 0; i < offset_.size(); i++) {
    int32_t offset = start + offset_[i];
    memcpy(*buffer + sizeof(int32_t) * (i + 1), &offset, sizeof(int32_t));
  }

  // Copy data of strings.
  memcpy(*buffer + start, data_.data(), data_.size());
  return bytes;
}

#ifndef TF_LITE_STATIC_MEMORY
void DynamicBuffer::WriteToTensorAsVector(TfLiteTensor* tensor) {
  auto dims = TfLiteIntArrayCreate(1);
  dims->data[0] = offset_.size() - 1;  // Store number of strings.
  WriteToTensor(tensor, dims);
}

void DynamicBuffer::WriteToTensor(TfLiteTensor* tensor,
                                  TfLiteIntArray* new_shape) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSstring_utilDTcc mht_5(mht_5_v, 294, "", "./tensorflow/lite/string_util.cc", "DynamicBuffer::WriteToTensor");

  char* tensor_buffer;
  int bytes = WriteToBuffer(&tensor_buffer);

  if (new_shape == nullptr) {
    new_shape = TfLiteIntArrayCopy(tensor->dims);
  }

  // Set tensor content pointer to tensor_buffer, and release original data.
  TfLiteTensorReset(tensor->type, tensor->name, new_shape, tensor->params,
                    tensor_buffer, bytes, kTfLiteDynamic, tensor->allocation,
                    tensor->is_variable, tensor);
}
#endif  // TF_LITE_STATIC_MEMORY

int GetStringCount(const void* raw_buffer) {
  // The first integers in the raw buffer is the number of strings.
  return *static_cast<const int32_t*>(raw_buffer);
}

int GetStringCount(const TfLiteTensor* tensor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSstring_utilDTcc mht_6(mht_6_v, 317, "", "./tensorflow/lite/string_util.cc", "GetStringCount");

  // The first integers in the raw buffer is the number of strings.
  return GetStringCount(tensor->data.raw);
}

StringRef GetString(const void* raw_buffer, int string_index) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSstring_utilDTcc mht_7(mht_7_v, 325, "", "./tensorflow/lite/string_util.cc", "GetString");

  const int32_t* offset =
      static_cast<const int32_t*>(raw_buffer) + (string_index + 1);
  return StringRef{
      static_cast<const char*>(raw_buffer) + (*offset),
      (*(offset + 1)) - (*offset),
  };
}

StringRef GetString(const TfLiteTensor* tensor, int string_index) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSstring_utilDTcc mht_8(mht_8_v, 337, "", "./tensorflow/lite/string_util.cc", "GetString");

  return GetString(tensor->data.raw, string_index);
}

}  // namespace tflite
