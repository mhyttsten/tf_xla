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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_RETRYING_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_RETRYING_FILE_SYSTEM_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh() {
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


#include <functional>
#include <string>
#include <vector>

#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/retrying_utils.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

/// A wrapper to add retry logic to another file system.
template <typename Underlying>
class RetryingFileSystem : public FileSystem {
 public:
  RetryingFileSystem(std::unique_ptr<Underlying> base_file_system,
                     const RetryConfig& retry_config)
      : base_file_system_(std::move(base_file_system)),
        retry_config_(retry_config) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_0(mht_0_v, 208, "", "./tensorflow/core/platform/retrying_file_system.h", "RetryingFileSystem");
}

  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  Status NewRandomAccessFile(
      const string& filename, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override;

  Status NewWritableFile(const string& filename, TransactionToken* token,
                         std::unique_ptr<WritableFile>* result) override;

  Status NewAppendableFile(const string& filename, TransactionToken* token,
                           std::unique_ptr<WritableFile>* result) override;

  Status NewReadOnlyMemoryRegionFromFile(
      const string& filename, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;

  Status FileExists(const string& fname, TransactionToken* token) override {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_1(mht_1_v, 230, "", "./tensorflow/core/platform/retrying_file_system.h", "FileExists");

    return RetryingUtils::CallWithRetries(
        [this, &fname, token]() {
          return base_file_system_->FileExists(fname, token);
        },
        retry_config_);
  }

  Status GetChildren(const string& dir, TransactionToken* token,
                     std::vector<string>* result) override {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_2(mht_2_v, 243, "", "./tensorflow/core/platform/retrying_file_system.h", "GetChildren");

    return RetryingUtils::CallWithRetries(
        [this, &dir, result, token]() {
          return base_file_system_->GetChildren(dir, token, result);
        },
        retry_config_);
  }

  Status GetMatchingPaths(const string& pattern, TransactionToken* token,
                          std::vector<string>* result) override {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_3(mht_3_v, 256, "", "./tensorflow/core/platform/retrying_file_system.h", "GetMatchingPaths");

    return RetryingUtils::CallWithRetries(
        [this, &pattern, result, token]() {
          return base_file_system_->GetMatchingPaths(pattern, token, result);
        },
        retry_config_);
  }

  Status Stat(const string& fname, TransactionToken* token,
              FileStatistics* stat) override {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_4(mht_4_v, 269, "", "./tensorflow/core/platform/retrying_file_system.h", "Stat");

    return RetryingUtils::CallWithRetries(
        [this, &fname, stat, token]() {
          return base_file_system_->Stat(fname, token, stat);
        },
        retry_config_);
  }

  Status DeleteFile(const string& fname, TransactionToken* token) override {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_5(mht_5_v, 281, "", "./tensorflow/core/platform/retrying_file_system.h", "DeleteFile");

    return RetryingUtils::DeleteWithRetries(
        [this, &fname, token]() {
          return base_file_system_->DeleteFile(fname, token);
        },
        retry_config_);
  }

  Status CreateDir(const string& dirname, TransactionToken* token) override {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_6(mht_6_v, 293, "", "./tensorflow/core/platform/retrying_file_system.h", "CreateDir");

    return RetryingUtils::CallWithRetries(
        [this, &dirname, token]() {
          return base_file_system_->CreateDir(dirname, token);
        },
        retry_config_);
  }

  Status DeleteDir(const string& dirname, TransactionToken* token) override {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_7(mht_7_v, 305, "", "./tensorflow/core/platform/retrying_file_system.h", "DeleteDir");

    return RetryingUtils::DeleteWithRetries(
        [this, &dirname, token]() {
          return base_file_system_->DeleteDir(dirname, token);
        },
        retry_config_);
  }

  Status GetFileSize(const string& fname, TransactionToken* token,
                     uint64* file_size) override {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_8(mht_8_v, 318, "", "./tensorflow/core/platform/retrying_file_system.h", "GetFileSize");

    return RetryingUtils::CallWithRetries(
        [this, &fname, file_size, token]() {
          return base_file_system_->GetFileSize(fname, token, file_size);
        },
        retry_config_);
  }

  Status RenameFile(const string& src, const string& target,
                    TransactionToken* token) override {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("src: \"" + src + "\"");
   mht_9_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_9(mht_9_v, 332, "", "./tensorflow/core/platform/retrying_file_system.h", "RenameFile");

    return RetryingUtils::CallWithRetries(
        [this, &src, &target, token]() {
          return base_file_system_->RenameFile(src, target, token);
        },
        retry_config_);
  }

  Status IsDirectory(const string& dirname, TransactionToken* token) override {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_10(mht_10_v, 344, "", "./tensorflow/core/platform/retrying_file_system.h", "IsDirectory");

    return RetryingUtils::CallWithRetries(
        [this, &dirname, token]() {
          return base_file_system_->IsDirectory(dirname, token);
        },
        retry_config_);
  }

  Status HasAtomicMove(const string& path, bool* has_atomic_move) override {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_11(mht_11_v, 356, "", "./tensorflow/core/platform/retrying_file_system.h", "HasAtomicMove");

    // this method does not need to be retried
    return base_file_system_->HasAtomicMove(path, has_atomic_move);
  }

  Status DeleteRecursively(const string& dirname, TransactionToken* token,
                           int64_t* undeleted_files,
                           int64_t* undeleted_dirs) override {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_12(mht_12_v, 367, "", "./tensorflow/core/platform/retrying_file_system.h", "DeleteRecursively");

    return RetryingUtils::DeleteWithRetries(
        [this, &dirname, token, undeleted_files, undeleted_dirs]() {
          return base_file_system_->DeleteRecursively(
              dirname, token, undeleted_files, undeleted_dirs);
        },
        retry_config_);
  }

  void FlushCaches(TransactionToken* token) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_13(mht_13_v, 379, "", "./tensorflow/core/platform/retrying_file_system.h", "FlushCaches");

    base_file_system_->FlushCaches(token);
  }

  Underlying* underlying() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_14(mht_14_v, 386, "", "./tensorflow/core/platform/retrying_file_system.h", "underlying");
 return base_file_system_.get(); }

 private:
  std::unique_ptr<Underlying> base_file_system_;
  const RetryConfig retry_config_;

  TF_DISALLOW_COPY_AND_ASSIGN(RetryingFileSystem);
};

namespace retrying_internals {

class RetryingRandomAccessFile : public RandomAccessFile {
 public:
  RetryingRandomAccessFile(std::unique_ptr<RandomAccessFile> base_file,
                           const RetryConfig& retry_config)
      : base_file_(std::move(base_file)), retry_config_(retry_config) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_15(mht_15_v, 404, "", "./tensorflow/core/platform/retrying_file_system.h", "RetryingRandomAccessFile");
}

  Status Name(StringPiece* result) const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_16(mht_16_v, 409, "", "./tensorflow/core/platform/retrying_file_system.h", "Name");

    return base_file_->Name(result);
  }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("scratch: \"" + (scratch == nullptr ? std::string("nullptr") : std::string((char*)scratch)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_17(mht_17_v, 418, "", "./tensorflow/core/platform/retrying_file_system.h", "Read");

    return RetryingUtils::CallWithRetries(
        [this, offset, n, result, scratch]() {
          return base_file_->Read(offset, n, result, scratch);
        },
        retry_config_);
  }

 private:
  std::unique_ptr<RandomAccessFile> base_file_;
  const RetryConfig retry_config_;
};

class RetryingWritableFile : public WritableFile {
 public:
  RetryingWritableFile(std::unique_ptr<WritableFile> base_file,
                       const RetryConfig& retry_config)
      : base_file_(std::move(base_file)), retry_config_(retry_config) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_18(mht_18_v, 438, "", "./tensorflow/core/platform/retrying_file_system.h", "RetryingWritableFile");
}

  ~RetryingWritableFile() override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_19(mht_19_v, 443, "", "./tensorflow/core/platform/retrying_file_system.h", "~RetryingWritableFile");

    // Makes sure the retrying version of Close() is called in the destructor.
    Close().IgnoreError();
  }

  Status Append(StringPiece data) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_20(mht_20_v, 451, "", "./tensorflow/core/platform/retrying_file_system.h", "Append");

    return RetryingUtils::CallWithRetries(
        [this, &data]() { return base_file_->Append(data); }, retry_config_);
  }
  Status Close() override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_21(mht_21_v, 458, "", "./tensorflow/core/platform/retrying_file_system.h", "Close");

    return RetryingUtils::CallWithRetries(
        [this]() { return base_file_->Close(); }, retry_config_);
  }
  Status Flush() override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_22(mht_22_v, 465, "", "./tensorflow/core/platform/retrying_file_system.h", "Flush");

    return RetryingUtils::CallWithRetries(
        [this]() { return base_file_->Flush(); }, retry_config_);
  }
  Status Name(StringPiece* result) const override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_23(mht_23_v, 472, "", "./tensorflow/core/platform/retrying_file_system.h", "Name");

    return base_file_->Name(result);
  }
  Status Sync() override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_24(mht_24_v, 478, "", "./tensorflow/core/platform/retrying_file_system.h", "Sync");

    return RetryingUtils::CallWithRetries(
        [this]() { return base_file_->Sync(); }, retry_config_);
  }
  Status Tell(int64_t* position) override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_25(mht_25_v, 485, "", "./tensorflow/core/platform/retrying_file_system.h", "Tell");

    return RetryingUtils::CallWithRetries(
        [this, &position]() { return base_file_->Tell(position); },
        retry_config_);
  }

 private:
  std::unique_ptr<WritableFile> base_file_;
  const RetryConfig retry_config_;
};

}  // namespace retrying_internals

template <typename Underlying>
Status RetryingFileSystem<Underlying>::NewRandomAccessFile(
    const string& filename, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_26(mht_26_v, 505, "", "./tensorflow/core/platform/retrying_file_system.h", "RetryingFileSystem<Underlying>::NewRandomAccessFile");

  std::unique_ptr<RandomAccessFile> base_file;
  TF_RETURN_IF_ERROR(RetryingUtils::CallWithRetries(
      [this, &filename, &base_file, token]() {
        return base_file_system_->NewRandomAccessFile(filename, token,
                                                      &base_file);
      },
      retry_config_));
  result->reset(new retrying_internals::RetryingRandomAccessFile(
      std::move(base_file), retry_config_));
  return Status::OK();
}

template <typename Underlying>
Status RetryingFileSystem<Underlying>::NewWritableFile(
    const string& filename, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_27(mht_27_v, 525, "", "./tensorflow/core/platform/retrying_file_system.h", "RetryingFileSystem<Underlying>::NewWritableFile");

  std::unique_ptr<WritableFile> base_file;
  TF_RETURN_IF_ERROR(RetryingUtils::CallWithRetries(
      [this, &filename, &base_file, token]() {
        return base_file_system_->NewWritableFile(filename, token, &base_file);
      },
      retry_config_));
  result->reset(new retrying_internals::RetryingWritableFile(
      std::move(base_file), retry_config_));
  return Status::OK();
}

template <typename Underlying>
Status RetryingFileSystem<Underlying>::NewAppendableFile(
    const string& filename, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_28(mht_28_v, 544, "", "./tensorflow/core/platform/retrying_file_system.h", "RetryingFileSystem<Underlying>::NewAppendableFile");

  std::unique_ptr<WritableFile> base_file;
  TF_RETURN_IF_ERROR(RetryingUtils::CallWithRetries(
      [this, &filename, &base_file, token]() {
        return base_file_system_->NewAppendableFile(filename, token,
                                                    &base_file);
      },
      retry_config_));
  result->reset(new retrying_internals::RetryingWritableFile(
      std::move(base_file), retry_config_));
  return Status::OK();
}

template <typename Underlying>
Status RetryingFileSystem<Underlying>::NewReadOnlyMemoryRegionFromFile(
    const string& filename, TransactionToken* token,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_systemDTh mht_29(mht_29_v, 564, "", "./tensorflow/core/platform/retrying_file_system.h", "RetryingFileSystem<Underlying>::NewReadOnlyMemoryRegionFromFile");

  return RetryingUtils::CallWithRetries(
      [this, &filename, result, token]() {
        return base_file_system_->NewReadOnlyMemoryRegionFromFile(
            filename, token, result);
      },
      retry_config_);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_RETRYING_FILE_SYSTEM_H_
