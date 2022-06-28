/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_MEMMAPPED_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_UTIL_MEMMAPPED_FILE_SYSTEM_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTh() {
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
#include <unordered_map>

#include "tensorflow/core/platform/env.h"

namespace tensorflow {

// A file system that uses a graph saved in memmapped format by
// MemmappedEnvWriter as a file system.
//
// The format supports saved tensors and protos. Tensors are saved at aligned
// offsets.
//
// Format specification:
// - last 8 bytes of a package is encoded offset to the directory. The encoding
// is always little endian, independently from the platform, done by functions
// EncodeUint64LittleEndian/DecodeUint64LittleEndian
// - the directory starts from the encoded offset and is saved proto
// MemmappedFileSystemDirectory with names and offsets to the regions.
// - at the offsets in the directory the file regions are stored. Tensor regions
// are aligned such way that when the package mapped to RAM they have the right
// offset to be used by ImmutableConst operator.
//
// Region naming:
// Region naming is up to the application, all of them starts from
// kMemmappedPackagePrefix. The default graph usually has name
// kMemmappedPackageDefaultGraphDef;
//
// A "frozen" GraphDef can be converted into this format using
// tensorflow/contrib/util/convert_graphdef_memmapped_format
class MemmappedFileSystem : public FileSystem {
 public:
  // Memmapped regions use this prefix to distinguish from
  // the filesystem.
  static constexpr const char kMemmappedPackagePrefix[] =
      "memmapped_package://";

  // The default graphdef in the package.
  static constexpr const char kMemmappedPackageDefaultGraphDef[] =
      "memmapped_package://.";

  MemmappedFileSystem();
  ~MemmappedFileSystem() override = default;

  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  Status FileExists(const string& fname, TransactionToken* token) override;
  Status NewRandomAccessFile(
      const string& filename, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override;
  Status NewReadOnlyMemoryRegionFromFile(
      const string& filename, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;

  // All these functions return Unimplemented error, the memmapped storage is
  // read only.
  Status NewWritableFile(const string& fname, TransactionToken* token,
                         std::unique_ptr<WritableFile>* result) override;
  Status NewAppendableFile(const string& fname, TransactionToken* token,
                           std::unique_ptr<WritableFile>* result) override;
  Status GetChildren(const string& dir, TransactionToken* token,
                     std::vector<string>* r) override;
  Status GetMatchingPaths(const string& pattern, TransactionToken* token,
                          std::vector<string>* results) override;
  Status DeleteFile(const string& f, TransactionToken* token) override;
  Status CreateDir(const string& d, TransactionToken* token) override;
  Status DeleteDir(const string& d, TransactionToken* token) override;
  Status RenameFile(const string& s, const string& t,
                    TransactionToken* token) override;

  // These functions are implemented.
  Status GetFileSize(const string& f, TransactionToken* token,
                     uint64* s) override;
  // Currently just returns size.
  Status Stat(const string& fname, TransactionToken* token,
              FileStatistics* stat) override;

  // Initializes filesystem from a file in memmapped format.
  Status InitializeFromFile(Env* env, const string& filename);

  // Checks if the filename has a correct prefix.
  static bool IsMemmappedPackageFilename(const string& filename);

  static bool IsWellFormedMemmappedPackageFilename(const string& filename);

 private:
  struct FileRegion {
    FileRegion(uint64 o, uint64 l) : offset(o), length(l) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTh mht_0(mht_0_v, 276, "", "./tensorflow/core/util/memmapped_file_system.h", "FileRegion");
}

    uint64 offset;  // Offset from the beginning of the file.
    uint64 length;  // Length of the region.
  };

  using DirectoryType = std::unordered_map<string, FileRegion>;

  const void* GetMemoryWithOffset(uint64 offset) const;

  std::unique_ptr<ReadOnlyMemoryRegion> mapped_memory_;
  DirectoryType directory_;

  TF_DISALLOW_COPY_AND_ASSIGN(MemmappedFileSystem);
};

class MemmappedEnv : public EnvWrapper {
 public:
  explicit MemmappedEnv(Env* env);
  ~MemmappedEnv() override = default;
  Status GetFileSystemForFile(const string& fname,
                              FileSystem** result) override;
  Status GetRegisteredFileSystemSchemes(std::vector<string>* schemes) override;
  Status InitializeFromFile(const string& filename);

 protected:
  std::unique_ptr<MemmappedFileSystem> memmapped_file_system_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_MEMMAPPED_FILE_SYSTEM_H_
