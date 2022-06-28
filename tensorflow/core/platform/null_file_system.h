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

#ifndef TENSORFLOW_CORE_PLATFORM_NULL_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_PLATFORM_NULL_FILE_SYSTEM_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh() {
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


#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/file_system_helper.h"

namespace tensorflow {

// START_SKIP_DOXYGEN

#ifndef SWIG
// Degenerate file system that provides no implementations.
class NullFileSystem : public FileSystem {
 public:
  NullFileSystem() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh mht_0(mht_0_v, 204, "", "./tensorflow/core/platform/null_file_system.h", "NullFileSystem");
}

  ~NullFileSystem() override = default;

  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  Status NewRandomAccessFile(
      const string& fname, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh mht_1(mht_1_v, 216, "", "./tensorflow/core/platform/null_file_system.h", "NewRandomAccessFile");

    return errors::Unimplemented("NewRandomAccessFile unimplemented");
  }

  Status NewWritableFile(const string& fname, TransactionToken* token,
                         std::unique_ptr<WritableFile>* result) override {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh mht_2(mht_2_v, 225, "", "./tensorflow/core/platform/null_file_system.h", "NewWritableFile");

    return errors::Unimplemented("NewWritableFile unimplemented");
  }

  Status NewAppendableFile(const string& fname, TransactionToken* token,
                           std::unique_ptr<WritableFile>* result) override {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh mht_3(mht_3_v, 234, "", "./tensorflow/core/platform/null_file_system.h", "NewAppendableFile");

    return errors::Unimplemented("NewAppendableFile unimplemented");
  }

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh mht_4(mht_4_v, 244, "", "./tensorflow/core/platform/null_file_system.h", "NewReadOnlyMemoryRegionFromFile");

    return errors::Unimplemented(
        "NewReadOnlyMemoryRegionFromFile unimplemented");
  }

  Status FileExists(const string& fname, TransactionToken* token) override {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh mht_5(mht_5_v, 253, "", "./tensorflow/core/platform/null_file_system.h", "FileExists");

    return errors::Unimplemented("FileExists unimplemented");
  }

  Status GetChildren(const string& dir, TransactionToken* token,
                     std::vector<string>* result) override {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh mht_6(mht_6_v, 262, "", "./tensorflow/core/platform/null_file_system.h", "GetChildren");

    return errors::Unimplemented("GetChildren unimplemented");
  }

  Status GetMatchingPaths(const string& pattern, TransactionToken* token,
                          std::vector<string>* results) override {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh mht_7(mht_7_v, 271, "", "./tensorflow/core/platform/null_file_system.h", "GetMatchingPaths");

    return internal::GetMatchingPaths(this, Env::Default(), pattern, results);
  }

  Status DeleteFile(const string& fname, TransactionToken* token) override {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh mht_8(mht_8_v, 279, "", "./tensorflow/core/platform/null_file_system.h", "DeleteFile");

    return errors::Unimplemented("DeleteFile unimplemented");
  }

  Status CreateDir(const string& dirname, TransactionToken* token) override {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh mht_9(mht_9_v, 287, "", "./tensorflow/core/platform/null_file_system.h", "CreateDir");

    return errors::Unimplemented("CreateDir unimplemented");
  }

  Status DeleteDir(const string& dirname, TransactionToken* token) override {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh mht_10(mht_10_v, 295, "", "./tensorflow/core/platform/null_file_system.h", "DeleteDir");

    return errors::Unimplemented("DeleteDir unimplemented");
  }

  Status GetFileSize(const string& fname, TransactionToken* token,
                     uint64* file_size) override {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh mht_11(mht_11_v, 304, "", "./tensorflow/core/platform/null_file_system.h", "GetFileSize");

    return errors::Unimplemented("GetFileSize unimplemented");
  }

  Status RenameFile(const string& src, const string& target,
                    TransactionToken* token) override {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("src: \"" + src + "\"");
   mht_12_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh mht_12(mht_12_v, 314, "", "./tensorflow/core/platform/null_file_system.h", "RenameFile");

    return errors::Unimplemented("RenameFile unimplemented");
  }

  Status Stat(const string& fname, TransactionToken* token,
              FileStatistics* stat) override {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSnull_file_systemDTh mht_13(mht_13_v, 323, "", "./tensorflow/core/platform/null_file_system.h", "Stat");

    return errors::Unimplemented("Stat unimplemented");
  }
};
#endif

// END_SKIP_DOXYGEN

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_NULL_FILE_SYSTEM_H_
