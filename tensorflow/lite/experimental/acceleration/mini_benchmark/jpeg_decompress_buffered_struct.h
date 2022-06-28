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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_BUFFERED_STRUCT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_BUFFERED_STRUCT_H_
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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_decompress_buffered_structDTh {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_decompress_buffered_structDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_decompress_buffered_structDTh() {
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


#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

// May provide an extra buffer of characters beyond the `jpeg_decompress_struct`
// for some builds of Libjpeg Dynamic Library on Android that expect a larger
// struct than we were compiled with. Zeroes out any allocated bytes beyond
// sizeof(jpeg_decompress_struct). This class is exclusively used by
// decode_jpeg.cc to resize `jpeg_decompress_struct`. This is to fix a struct
// mismatch problem. See go/libjpeg-android for more details.
class JpegDecompressBufferedStruct {
 public:
  explicit JpegDecompressBufferedStruct(std::size_t expected_size)
      : resized_size_(std::max(sizeof(jpeg_decompress_struct), expected_size)),
        buffer_(reinterpret_cast<char*>(malloc(resized_size_))) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_decompress_buffered_structDTh mht_0(mht_0_v, 208, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_decompress_buffered_struct.h", "JpegDecompressBufferedStruct");

    // Note: Malloc guarantees alignment for 8 bytes. Hence, using malloc
    // instead of aligned_alloc.
    // https://www.gnu.org/software/libc/manual/html_node/Aligned-Memory-Blocks.html
    // alignof(jpeg_decompress_struct) is 8 bytes both on 32 and 64 bit.
    // It's safe to align the buffered struct as
    // alignof(jpeg_decompress_struct). This is because we only access the
    // `jpeg_common_fields` fields of `jpeg_decompress_struct`, all of which are
    // pointers. The alignment of these pointer fields is 8 and 4 bytes for 64
    // bit and 32 bit platforms respectively. Since
    // alignof(jpeg_decompress_struct) is 8 bytes on both platforms, accessing
    // these fields shouldn't be a problem.
    // Zero out any excess bytes. Zero-initialization is safe for the bytes
    // beyond sizeof(jpeg_decompress_struct) because both the dynamic library
    // and the implementation in decode_jpeg.cc limit their access only to
    // `jpeg_common_fields` in `jpeg_decompress_struct`.
    while (--expected_size >= sizeof(jpeg_decompress_struct)) {
      buffer_[expected_size] = 0;
    }
  }
  ~JpegDecompressBufferedStruct() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_decompress_buffered_structDTh mht_1(mht_1_v, 231, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_decompress_buffered_struct.h", "~JpegDecompressBufferedStruct");
 std::free(buffer_); }
  JpegDecompressBufferedStruct(const JpegDecompressBufferedStruct&) = delete;
  JpegDecompressBufferedStruct& operator=(const JpegDecompressBufferedStruct&) =
      delete;
  jpeg_decompress_struct* get() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_decompress_buffered_structDTh mht_2(mht_2_v, 238, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_decompress_buffered_struct.h", "get");

    return reinterpret_cast<jpeg_decompress_struct*>(buffer_);
  }
  int const size() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_decompress_buffered_structDTh mht_3(mht_3_v, 244, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_decompress_buffered_struct.h", "size");
 return resized_size_; }
  const char* buffer() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_decompress_buffered_structDTh mht_4(mht_4_v, 248, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_decompress_buffered_struct.h", "buffer");
 return buffer_; }

 private:
  int resized_size_;
  char* const buffer_;
};

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_BUFFERED_STRUCT_H_
