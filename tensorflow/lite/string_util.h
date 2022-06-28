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

// Util methods to read and write String tensors.
// String tensors are considered to be char tensor with protocol.
//   [0, 3] 4 bytes: N, num of strings in the tensor in little endian.
//   [(i+1)*4, (i+1)*4+3] 4 bytes: offset of i-th string in little endian,
//                                 for i from 0 to N-1.
//   [(N+1)*4, (N+1)*4+3] 4 bytes: length of the whole char buffer.
//   [offset(i), offset(i+1) - 1] : content of i-th string.
// Example of a string tensor:
// [
//   2, 0, 0, 0,     # 2 strings.
//   16, 0, 0, 0,    # 0-th string starts from index 16.
//   18, 0, 0, 0,    # 1-st string starts from index 18.
//   18, 0, 0, 0,    # total length of array.
//   'A', 'B',       # 0-th string [16..17]: "AB"
// ]                 # 1-th string, empty
//
// A typical usage:
// In op.Eval(context, node):
//   DynamicBuffer buf;
//   # Add string "AB" to tensor, string is stored in dynamic buffer.
//   buf.AddString("AB", 2);
//   # Write content of DynamicBuffer to tensor in format of string tensor
//   # described above.
//   buf.WriteToTensor(tensor, nullptr)

#ifndef TENSORFLOW_LITE_STRING_UTIL_H_
#define TENSORFLOW_LITE_STRING_UTIL_H_
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
class MHTracer_DTPStensorflowPSlitePSstring_utilDTh {
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
   MHTracer_DTPStensorflowPSlitePSstring_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSstring_utilDTh() {
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


#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {

// Convenient structure to store string pointer and length.
typedef struct {
  const char* str;
  int len;
} StringRef;

// DynamicBuffer holds temporary buffer that will be used to create a dynamic
// tensor. A typical usage is to initialize a DynamicBuffer object, fill in
// content and call CreateStringTensor in op.Eval().
class DynamicBuffer {
 public:
  DynamicBuffer() : offset_({0}) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSstring_utilDTh mht_0(mht_0_v, 234, "", "./tensorflow/lite/string_util.h", "DynamicBuffer");
}

  // Add string to dynamic buffer by resizing the buffer and copying the data.
  void AddString(const StringRef& string);

  // Add string to dynamic buffer by resizing the buffer and copying the data.
  void AddString(const char* str, size_t len);

  // Join a list of string with separator, and add as a single string to the
  // buffer.
  void AddJoinedString(const std::vector<StringRef>& strings, char separator);
  void AddJoinedString(const std::vector<StringRef>& strings,
                       StringRef separator);

  // Fill content into a buffer and returns the number of bytes stored.
  // The function allocates space for the buffer but does NOT take ownership.
  int WriteToBuffer(char** buffer);

  // Fill content into a string tensor, with the given new_shape. The new shape
  // must match the number of strings in this object. Caller relinquishes
  // ownership of new_shape. If 'new_shape' is nullptr, keep the tensor's
  // existing shape.
  void WriteToTensor(TfLiteTensor* tensor, TfLiteIntArray* new_shape);

  // Fill content into a string tensor. Set shape to {num_strings}.
  void WriteToTensorAsVector(TfLiteTensor* tensor);

 private:
  // Data buffer to store contents of strings, not including headers.
  std::vector<char> data_;
  // Offset of the starting index of each string in data buffer.
  std::vector<int32_t> offset_;
};

// Return num of strings in a String tensor.
int GetStringCount(const void* raw_buffer);
int GetStringCount(const TfLiteTensor* tensor);

// Get String pointer and length of index-th string in tensor.
// NOTE: This will not create a copy of string data.
StringRef GetString(const void* raw_buffer, int string_index);
StringRef GetString(const TfLiteTensor* tensor, int string_index);
}  // namespace tflite

#endif  // TENSORFLOW_LITE_STRING_UTIL_H_
