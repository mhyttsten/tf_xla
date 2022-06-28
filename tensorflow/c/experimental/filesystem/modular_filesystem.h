/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_MODULAR_FILESYSTEM_H_
#define TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_MODULAR_FILESYSTEM_H_
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
class MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSmodular_filesystemDTh {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSmodular_filesystemDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSmodular_filesystemDTh() {
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

#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/core/platform/file_system.h"

/// This file builds classes needed to hold a filesystem implementation in the
/// modular world. Once all TensorFlow filesystems are converted to use the
/// plugin based approach, this file will replace the one in core/platform and
/// the names will lose the `Modular` part. Until that point, the `Modular*`
/// classes here are experimental and subject to breaking changes.
/// For documentation on these methods, consult `core/platform/filesystem.h`.

namespace tensorflow {

// TODO(b/143949615): After all filesystems are converted, this file will be
// moved to core/platform, and this class can become a singleton and replace the
// need for `Env::Default()`. At that time, we might decide to remove the need
// for `Env::Default()` altogether, but that's a different project, not in
// scope for now. I'm just mentioning this here as that transition will mean
// removal of the registration part from `Env` and adding it here instead: we
// will need tables to hold for each scheme the function tables that implement
// the needed functionality instead of the current `FileSystemRegistry` code in
// `core/platform/env.cc`.
class ModularFileSystem final : public FileSystem {
 public:
  ModularFileSystem(
      std::unique_ptr<TF_Filesystem> filesystem,
      std::unique_ptr<const TF_FilesystemOps> filesystem_ops,
      std::unique_ptr<const TF_RandomAccessFileOps> random_access_file_ops,
      std::unique_ptr<const TF_WritableFileOps> writable_file_ops,
      std::unique_ptr<const TF_ReadOnlyMemoryRegionOps>
          read_only_memory_region_ops,
      std::function<void*(size_t)> plugin_memory_allocate,
      std::function<void(void*)> plugin_memory_free)
      : filesystem_(std::move(filesystem)),
        ops_(std::move(filesystem_ops)),
        random_access_file_ops_(std::move(random_access_file_ops)),
        writable_file_ops_(std::move(writable_file_ops)),
        read_only_memory_region_ops_(std::move(read_only_memory_region_ops)),
        plugin_memory_allocate_(std::move(plugin_memory_allocate)),
        plugin_memory_free_(std::move(plugin_memory_free)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSmodular_filesystemDTh mht_0(mht_0_v, 227, "", "./tensorflow/c/experimental/filesystem/modular_filesystem.h", "ModularFileSystem");
}

  ~ModularFileSystem() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSmodular_filesystemDTh mht_1(mht_1_v, 232, "", "./tensorflow/c/experimental/filesystem/modular_filesystem.h", "~ModularFileSystem");
 ops_->cleanup(filesystem_.get()); }

  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  Status NewRandomAccessFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override;
  Status NewWritableFile(const std::string& fname, TransactionToken* token,
                         std::unique_ptr<WritableFile>* result) override;
  Status NewAppendableFile(const std::string& fname, TransactionToken* token,
                           std::unique_ptr<WritableFile>* result) override;
  Status NewReadOnlyMemoryRegionFromFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;
  Status FileExists(const std::string& fname, TransactionToken* token) override;
  bool FilesExist(const std::vector<std::string>& files,
                  TransactionToken* token,
                  std::vector<Status>* status) override;
  Status GetChildren(const std::string& dir, TransactionToken* token,
                     std::vector<std::string>* result) override;
  Status GetMatchingPaths(const std::string& pattern, TransactionToken* token,
                          std::vector<std::string>* results) override;
  Status DeleteFile(const std::string& fname, TransactionToken* token) override;
  Status DeleteRecursively(const std::string& dirname, TransactionToken* token,
                           int64_t* undeleted_files,
                           int64_t* undeleted_dirs) override;
  Status DeleteDir(const std::string& dirname,
                   TransactionToken* token) override;
  Status RecursivelyCreateDir(const std::string& dirname,
                              TransactionToken* token) override;
  Status CreateDir(const std::string& dirname,
                   TransactionToken* token) override;
  Status Stat(const std::string& fname, TransactionToken* token,
              FileStatistics* stat) override;
  Status IsDirectory(const std::string& fname,
                     TransactionToken* token) override;
  Status GetFileSize(const std::string& fname, TransactionToken* token,
                     uint64* file_size) override;
  Status RenameFile(const std::string& src, const std::string& target,
                    TransactionToken* token) override;
  Status CopyFile(const std::string& src, const std::string& target,
                  TransactionToken* token) override;
  std::string TranslateName(const std::string& name) const override;
  void FlushCaches(TransactionToken* token) override;
  Status SetOption(const std::string& name,
                   const std::vector<string>& values) override;
  Status SetOption(const std::string& name,
                   const std::vector<int64_t>& values) override;
  Status SetOption(const std::string& name,
                   const std::vector<double>& values) override;

 private:
  std::unique_ptr<TF_Filesystem> filesystem_;
  std::unique_ptr<const TF_FilesystemOps> ops_;
  std::unique_ptr<const TF_RandomAccessFileOps> random_access_file_ops_;
  std::unique_ptr<const TF_WritableFileOps> writable_file_ops_;
  std::unique_ptr<const TF_ReadOnlyMemoryRegionOps>
      read_only_memory_region_ops_;
  std::function<void*(size_t)> plugin_memory_allocate_;
  std::function<void(void*)> plugin_memory_free_;
  TF_DISALLOW_COPY_AND_ASSIGN(ModularFileSystem);
};

class ModularRandomAccessFile final : public RandomAccessFile {
 public:
  ModularRandomAccessFile(const std::string& filename,
                          std::unique_ptr<TF_RandomAccessFile> file,
                          const TF_RandomAccessFileOps* ops)
      : filename_(filename), file_(std::move(file)), ops_(ops) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSmodular_filesystemDTh mht_2(mht_2_v, 304, "", "./tensorflow/c/experimental/filesystem/modular_filesystem.h", "ModularRandomAccessFile");
}

  ~ModularRandomAccessFile() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSmodular_filesystemDTh mht_3(mht_3_v, 309, "", "./tensorflow/c/experimental/filesystem/modular_filesystem.h", "~ModularRandomAccessFile");
 ops_->cleanup(file_.get()); }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override;
  Status Name(StringPiece* result) const override;

 private:
  std::string filename_;
  std::unique_ptr<TF_RandomAccessFile> file_;
  const TF_RandomAccessFileOps* ops_;  // not owned
  TF_DISALLOW_COPY_AND_ASSIGN(ModularRandomAccessFile);
};

class ModularWritableFile final : public WritableFile {
 public:
  ModularWritableFile(const std::string& filename,
                      std::unique_ptr<TF_WritableFile> file,
                      const TF_WritableFileOps* ops)
      : filename_(filename), file_(std::move(file)), ops_(ops) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSmodular_filesystemDTh mht_4(mht_4_v, 331, "", "./tensorflow/c/experimental/filesystem/modular_filesystem.h", "ModularWritableFile");
}

  ~ModularWritableFile() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSmodular_filesystemDTh mht_5(mht_5_v, 336, "", "./tensorflow/c/experimental/filesystem/modular_filesystem.h", "~ModularWritableFile");
 ops_->cleanup(file_.get()); }

  Status Append(StringPiece data) override;
  Status Close() override;
  Status Flush() override;
  Status Sync() override;
  Status Name(StringPiece* result) const override;
  Status Tell(int64_t* position) override;

 private:
  std::string filename_;
  std::unique_ptr<TF_WritableFile> file_;
  const TF_WritableFileOps* ops_;  // not owned
  TF_DISALLOW_COPY_AND_ASSIGN(ModularWritableFile);
};

class ModularReadOnlyMemoryRegion final : public ReadOnlyMemoryRegion {
 public:
  ModularReadOnlyMemoryRegion(std::unique_ptr<TF_ReadOnlyMemoryRegion> region,
                              const TF_ReadOnlyMemoryRegionOps* ops)
      : region_(std::move(region)), ops_(ops) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSmodular_filesystemDTh mht_6(mht_6_v, 359, "", "./tensorflow/c/experimental/filesystem/modular_filesystem.h", "ModularReadOnlyMemoryRegion");
}

  ~ModularReadOnlyMemoryRegion() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSmodular_filesystemDTh mht_7(mht_7_v, 364, "", "./tensorflow/c/experimental/filesystem/modular_filesystem.h", "~ModularReadOnlyMemoryRegion");
 ops_->cleanup(region_.get()); };

  const void* data() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSmodular_filesystemDTh mht_8(mht_8_v, 369, "", "./tensorflow/c/experimental/filesystem/modular_filesystem.h", "data");
 return ops_->data(region_.get()); }
  uint64 length() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSmodular_filesystemDTh mht_9(mht_9_v, 373, "", "./tensorflow/c/experimental/filesystem/modular_filesystem.h", "length");
 return ops_->length(region_.get()); }

 private:
  std::unique_ptr<TF_ReadOnlyMemoryRegion> region_;
  const TF_ReadOnlyMemoryRegionOps* ops_;  // not owned
  TF_DISALLOW_COPY_AND_ASSIGN(ModularReadOnlyMemoryRegion);
};

// Registers a filesystem plugin so that core TensorFlow can use it.
Status RegisterFilesystemPlugin(const std::string& dso_path);

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_MODULAR_FILESYSTEM_H_
