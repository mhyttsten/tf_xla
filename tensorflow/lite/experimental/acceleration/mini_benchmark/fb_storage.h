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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_FB_STORAGE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_FB_STORAGE_H_
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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSfb_storageDTh {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSfb_storageDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSfb_storageDTh() {
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


#include <errno.h>

#include <cstring>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/base.h"  // from @flatbuffers
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/stderr_reporter.h"
namespace tflite {
namespace acceleration {

// FileStorage wraps storage of data in a file with locking and error handling.
// Locking makes appends and reads atomic, using flock(2).
//
// The locking in this class is not meant for general purpose multiple
// reader/writer support, but primarily for the case where a previous instance
// of a program has not finished and we'd like to not corrupt the file
// unnecessarily.
class FileStorage {
 public:
  FileStorage(absl::string_view path, ErrorReporter* error_reporter);
  // Read contents into buffer_. Returns an error if file exists but cannot be
  // read.
  MinibenchmarkStatus ReadFileIntoBuffer();
  // Append data to file. Resets the in-memory items and returns an error if
  // writing fails in any way.
  //
  // This calls fsync() on the file to guarantee persistence and is hence quite
  // expensive. The assumption is that this is not done often or in a critical
  // path.
  MinibenchmarkStatus AppendDataToFile(absl::string_view data);

 protected:
  std::string path_;
  ErrorReporter* error_reporter_;
  std::string buffer_;
};

// FlatbufferStorage stores several flatbuffer objects in a file. The primary
// usage is for storing mini benchmark results.
//
// Flatbuffers are not designed for easy mutation. This class is append-only.
// The intended usage is to store a log of events like 'start benchmark with
// configuration X', 'benchmark results for X' / 'crash observed with X' that
// are then parsed to make decisions about how to configure TFLite.
//
// The data is stored as consecutive length-prefixed flatbuffers with identifier
// "STO1".
ABSL_CONST_INIT extern const char kFlatbufferStorageIdentifier[];
template <typename T>
class FlatbufferStorage : protected FileStorage {
 public:
  explicit FlatbufferStorage(
      absl::string_view path,
      ErrorReporter* error_reporter = DefaultErrorReporter())
      : FileStorage(path, error_reporter) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("path: \"" + std::string(path.data(), path.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSfb_storageDTh mht_0(mht_0_v, 250, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h", "FlatbufferStorage");
}
  // Reads current contents. Returns an error if file is inaccessible or
  // contents are corrupt. The file not existing is not an error.
  MinibenchmarkStatus Read();
  // Get count of objects stored.
  size_t Count() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSfb_storageDTh mht_1(mht_1_v, 258, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h", "Count");
 return contents_.size(); }
  // Get object at index i, i < Count();
  const T* Get(size_t i) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSfb_storageDTh mht_2(mht_2_v, 263, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h", "Get");
 return contents_[i]; }

  // Append a new object to storage and write out to disk. Returns an error if
  // disk write or re-read fails.
  MinibenchmarkStatus Append(flatbuffers::FlatBufferBuilder* fbb,
                             flatbuffers::Offset<T> object);

 private:
  std::vector<const T*> contents_;
};

template <typename T>
MinibenchmarkStatus FlatbufferStorage<T>::Read() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSfb_storageDTh mht_3(mht_3_v, 278, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h", "FlatbufferStorage<T>::Read");

  contents_.clear();
  MinibenchmarkStatus status = ReadFileIntoBuffer();
  if (status != kMinibenchmarkSuccess) {
    return status;
  }
  size_t remaining_size = buffer_.size();
  const uint8_t* current_ptr =
      reinterpret_cast<const uint8_t*>(buffer_.c_str());
  while (remaining_size != 0) {
    if (remaining_size < sizeof(flatbuffers::uoffset_t)) {
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "Corrupt size-prefixed flatbuffer file %s (remaining size less than "
          "size of uoffset_t)",
          path_.c_str());
      return kMinibenchmarkCorruptSizePrefixedFlatbufferFile;
    }
    flatbuffers::uoffset_t current_size =
        flatbuffers::ReadScalar<flatbuffers::uoffset_t>(current_ptr);
    flatbuffers::Verifier verifier(
        current_ptr, sizeof(flatbuffers::uoffset_t) + current_size);
    if (!verifier.VerifySizePrefixedBuffer<T>(kFlatbufferStorageIdentifier)) {
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "Corrupt size-prefixed flatbuffer file %s (verifier returned false)",
          path_.c_str());
      return kMinibenchmarkCorruptSizePrefixedFlatbufferFile;
    }
    contents_.push_back(flatbuffers::GetSizePrefixedRoot<T>(current_ptr));
    size_t consumed = sizeof(flatbuffers::uoffset_t) + current_size;
    if (remaining_size < consumed) {
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "Corrupt size-prefixed flatbuffer file %s (mismatched size "
          "calculation)",
          path_.c_str());
      return kMinibenchmarkCorruptSizePrefixedFlatbufferFile;
    }
    remaining_size -= consumed;
    current_ptr += consumed;
  }
  return kMinibenchmarkSuccess;
}

template <typename T>
MinibenchmarkStatus FlatbufferStorage<T>::Append(
    flatbuffers::FlatBufferBuilder* fbb, flatbuffers::Offset<T> object) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSfb_storageDTh mht_4(mht_4_v, 328, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h", "FlatbufferStorage<T>::Append");

  contents_.clear();
  fbb->FinishSizePrefixed(object, kFlatbufferStorageIdentifier);
  const char* data = reinterpret_cast<const char*>(fbb->GetBufferPointer());
  size_t size = fbb->GetSize();
  MinibenchmarkStatus status = AppendDataToFile({data, size});
  if (status != kMinibenchmarkSuccess) {
    return status;
  }
  return Read();
}

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_FB_STORAGE_H_
