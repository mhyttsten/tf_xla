/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_RAM_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_PLATFORM_RAM_FILE_SYSTEM_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh() {
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


// Implementation of an in-memory TF filesystem for simple prototyping (e.g.
// via Colab). The TPU TF server does not have local filesystem access, which
// makes it difficult to provide Colab tutorials: users must have GCS access
// and sign-in in order to try out an example.
//
// Files are implemented on top of std::string. Directories, as with GCS or S3,
// are implicit based on the existence of child files. Multiple files may
// reference a single FS location, though no thread-safety guarantees are
// provided.

#include <string>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

#ifdef PLATFORM_WINDOWS
#undef DeleteFile
#undef CopyFile
#undef TranslateName
#endif

namespace tensorflow {

class RamRandomAccessFile : public RandomAccessFile, public WritableFile {
 public:
  RamRandomAccessFile(std::string name, std::shared_ptr<std::string> cord)
      : name_(name), data_(cord) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_0(mht_0_v, 218, "", "./tensorflow/core/platform/ram_file_system.h", "RamRandomAccessFile");
}
  ~RamRandomAccessFile() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_1(mht_1_v, 222, "", "./tensorflow/core/platform/ram_file_system.h", "~RamRandomAccessFile");
}

  Status Name(StringPiece* result) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_2(mht_2_v, 227, "", "./tensorflow/core/platform/ram_file_system.h", "Name");

    *result = name_;
    return Status::OK();
  }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("scratch: \"" + (scratch == nullptr ? std::string("nullptr") : std::string((char*)scratch)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_3(mht_3_v, 237, "", "./tensorflow/core/platform/ram_file_system.h", "Read");

    if (offset >= data_->size()) {
      return errors::OutOfRange("");
    }

    uint64 left = std::min(static_cast<uint64>(n), data_->size() - offset);
    auto start = data_->begin() + offset;
    auto end = data_->begin() + offset + left;

    std::copy(start, end, scratch);
    *result = StringPiece(scratch, left);

    // In case of a partial read, we must still fill `result`, but also return
    // OutOfRange.
    if (left < n) {
      return errors::OutOfRange("");
    }
    return Status::OK();
  }

  Status Append(StringPiece data) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_4(mht_4_v, 260, "", "./tensorflow/core/platform/ram_file_system.h", "Append");

    data_->append(data.data(), data.size());
    return Status::OK();
  }

#if defined(TF_CORD_SUPPORT)
  Status Append(const absl::Cord& cord) override {
    data_->append(cord.char_begin(), cord.char_end());
    return Status::OK();
  }
#endif

  Status Close() override { return Status::OK(); }
  Status Flush() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_5(mht_5_v, 276, "", "./tensorflow/core/platform/ram_file_system.h", "Flush");
 return Status::OK(); }
  Status Sync() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_6(mht_6_v, 280, "", "./tensorflow/core/platform/ram_file_system.h", "Sync");
 return Status::OK(); }

  Status Tell(int64_t* position) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_7(mht_7_v, 285, "", "./tensorflow/core/platform/ram_file_system.h", "Tell");

    *position = -1;
    return errors::Unimplemented("This filesystem does not support Tell()");
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RamRandomAccessFile);
  std::string name_;
  std::shared_ptr<std::string> data_;
};

class RamFileSystem : public FileSystem {
 public:
  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  Status NewRandomAccessFile(
      const std::string& fname_, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("fname_: \"" + fname_ + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_8(mht_8_v, 306, "", "./tensorflow/core/platform/ram_file_system.h", "NewRandomAccessFile");

    mutex_lock m(mu_);
    auto fname = StripRamFsPrefix(fname_);

    if (fs_.find(fname) == fs_.end()) {
      return errors::NotFound("");
    }
    if (fs_[fname] == nullptr) {
      return errors::InvalidArgument(fname_, " is a directory.");
    }
    *result = std::unique_ptr<RandomAccessFile>(
        new RamRandomAccessFile(fname, fs_[fname]));
    return Status::OK();
  }

  Status NewWritableFile(const std::string& fname_, TransactionToken* token,
                         std::unique_ptr<WritableFile>* result) override {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("fname_: \"" + fname_ + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_9(mht_9_v, 326, "", "./tensorflow/core/platform/ram_file_system.h", "NewWritableFile");

    mutex_lock m(mu_);
    auto fname = StripRamFsPrefix(fname_);

    if (fs_.find(fname) == fs_.end()) {
      fs_[fname] = std::make_shared<std::string>();
    }
    if (fs_[fname] == nullptr) {
      return errors::InvalidArgument(fname_, " is a directory.");
    }
    *result = std::unique_ptr<WritableFile>(
        new RamRandomAccessFile(fname, fs_[fname]));
    return Status::OK();
  }

  Status NewAppendableFile(const std::string& fname_, TransactionToken* token,
                           std::unique_ptr<WritableFile>* result) override {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("fname_: \"" + fname_ + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_10(mht_10_v, 346, "", "./tensorflow/core/platform/ram_file_system.h", "NewAppendableFile");

    mutex_lock m(mu_);
    auto fname = StripRamFsPrefix(fname_);

    if (fs_.find(fname) == fs_.end()) {
      fs_[fname] = std::make_shared<std::string>();
    }
    if (fs_[fname] == nullptr) {
      return errors::InvalidArgument(fname_, " is a directory.");
    }
    *result = std::unique_ptr<WritableFile>(
        new RamRandomAccessFile(fname, fs_[fname]));
    return Status::OK();
  }

  Status NewReadOnlyMemoryRegionFromFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_11(mht_11_v, 367, "", "./tensorflow/core/platform/ram_file_system.h", "NewReadOnlyMemoryRegionFromFile");

    return errors::Unimplemented("");
  }

  Status FileExists(const std::string& fname_,
                    TransactionToken* token) override {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("fname_: \"" + fname_ + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_12(mht_12_v, 376, "", "./tensorflow/core/platform/ram_file_system.h", "FileExists");

    FileStatistics stat;
    auto fname = StripRamFsPrefix(fname_);

    return Stat(fname, token, &stat);
  }

  Status GetChildren(const std::string& dir_, TransactionToken* token,
                     std::vector<std::string>* result) override {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("dir_: \"" + dir_ + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_13(mht_13_v, 388, "", "./tensorflow/core/platform/ram_file_system.h", "GetChildren");

    mutex_lock m(mu_);
    auto dir = StripRamFsPrefix(dir_);

    auto it = fs_.lower_bound(dir);
    while (it != fs_.end() && StartsWith(it->first, dir)) {
      auto filename = StripPrefix(StripPrefix(it->first, dir), "/");
      // It is not either (a) the parent directory itself or (b) a subdirectory
      if (!filename.empty() && filename.find("/") == std::string::npos) {
        result->push_back(filename);
      }
      ++it;
    }

    return Status::OK();
  }

  Status GetMatchingPaths(const std::string& pattern_, TransactionToken* token,
                          std::vector<std::string>* results) override {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("pattern_: \"" + pattern_ + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_14(mht_14_v, 410, "", "./tensorflow/core/platform/ram_file_system.h", "GetMatchingPaths");

    mutex_lock m(mu_);
    auto pattern = StripRamFsPrefix(pattern_);

    Env* env = Env::Default();
    for (auto it = fs_.begin(); it != fs_.end(); ++it) {
      if (env->MatchPath(it->first, pattern)) {
        results->push_back("ram://" + it->first);
      }
    }
    return Status::OK();
  }

  Status Stat(const std::string& fname_, TransactionToken* token,
              FileStatistics* stat) override {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("fname_: \"" + fname_ + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_15(mht_15_v, 428, "", "./tensorflow/core/platform/ram_file_system.h", "Stat");

    mutex_lock m(mu_);
    auto fname = StripRamFsPrefix(fname_);

    auto it = fs_.lower_bound(fname);
    if (it == fs_.end() || !StartsWith(it->first, fname)) {
      return errors::NotFound("");
    }

    if (it->first == fname && it->second != nullptr) {
      stat->is_directory = false;
      stat->length = fs_[fname]->size();
      stat->mtime_nsec = 0;
      return Status::OK();
    }

    stat->is_directory = true;
    stat->length = 0;
    stat->mtime_nsec = 0;
    return Status::OK();
  }

  Status DeleteFile(const std::string& fname_,
                    TransactionToken* token) override {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("fname_: \"" + fname_ + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_16(mht_16_v, 455, "", "./tensorflow/core/platform/ram_file_system.h", "DeleteFile");

    mutex_lock m(mu_);
    auto fname = StripRamFsPrefix(fname_);

    if (fs_.find(fname) != fs_.end()) {
      fs_.erase(fname);
      return Status::OK();
    }

    return errors::NotFound("");
  }

  Status CreateDir(const std::string& dirname_,
                   TransactionToken* token) override {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("dirname_: \"" + dirname_ + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_17(mht_17_v, 472, "", "./tensorflow/core/platform/ram_file_system.h", "CreateDir");

    mutex_lock m(mu_);
    auto dirname = StripRamFsPrefix(dirname_);

    auto it = fs_.find(dirname);
    if (it != fs_.end() && it->second != nullptr) {
      return errors::AlreadyExists(
          "cannot create directory with same name as an existing file");
    }

    fs_[dirname] = nullptr;
    return Status::OK();
  }

  Status RecursivelyCreateDir(const std::string& dirname_,
                              TransactionToken* token) override {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("dirname_: \"" + dirname_ + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_18(mht_18_v, 491, "", "./tensorflow/core/platform/ram_file_system.h", "RecursivelyCreateDir");

    auto dirname = StripRamFsPrefix(dirname_);

    std::vector<std::string> dirs = StrSplit(dirname, "/");
    Status last_status;
    std::string dir = dirs[0];
    last_status = CreateDir(dir, token);

    for (int i = 1; i < dirs.size(); ++i) {
      dir = dir + "/" + dirs[i];
      last_status = CreateDir(dir, token);
    }
    return last_status;
  }

  Status DeleteDir(const std::string& dirname_,
                   TransactionToken* token) override {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("dirname_: \"" + dirname_ + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_19(mht_19_v, 511, "", "./tensorflow/core/platform/ram_file_system.h", "DeleteDir");

    mutex_lock m(mu_);
    auto dirname = StripRamFsPrefix(dirname_);

    auto it = fs_.find(dirname);
    if (it == fs_.end()) {
      return errors::NotFound("");
    }
    if (it->second != nullptr) {
      return errors::InvalidArgument("Not a directory");
    }
    fs_.erase(dirname);

    return Status::OK();
  }

  Status GetFileSize(const std::string& fname_, TransactionToken* token,
                     uint64* file_size) override {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("fname_: \"" + fname_ + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_20(mht_20_v, 532, "", "./tensorflow/core/platform/ram_file_system.h", "GetFileSize");

    mutex_lock m(mu_);
    auto fname = StripRamFsPrefix(fname_);

    if (fs_.find(fname) != fs_.end()) {
      if (fs_[fname] == nullptr) {
        return errors::InvalidArgument("Not a file");
      }
      *file_size = fs_[fname]->size();
      return Status::OK();
    }
    return errors::NotFound("");
  }

  Status RenameFile(const std::string& src_, const std::string& target_,
                    TransactionToken* token) override {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("src_: \"" + src_ + "\"");
   mht_21_v.push_back("target_: \"" + target_ + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_21(mht_21_v, 552, "", "./tensorflow/core/platform/ram_file_system.h", "RenameFile");

    mutex_lock m(mu_);
    auto src = StripRamFsPrefix(src_);
    auto target = StripRamFsPrefix(target_);

    if (fs_.find(src) != fs_.end()) {
      fs_[target] = fs_[src];
      fs_.erase(fs_.find(src));
      return Status::OK();
    }
    return errors::NotFound("");
  }

  RamFileSystem() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_22(mht_22_v, 568, "", "./tensorflow/core/platform/ram_file_system.h", "RamFileSystem");
}
  ~RamFileSystem() override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_23(mht_23_v, 572, "", "./tensorflow/core/platform/ram_file_system.h", "~RamFileSystem");
}

 private:
  mutex mu_;
  std::map<std::string, std::shared_ptr<std::string>> fs_;

  std::vector<std::string> StrSplit(std::string s, std::string delim) {
    std::vector<std::string> ret;
    size_t curr_pos = 0;
    while ((curr_pos = s.find(delim)) != std::string::npos) {
      ret.push_back(s.substr(0, curr_pos));
      s.erase(0, curr_pos + delim.size());
    }
    ret.push_back(s);
    return ret;
  }

  bool StartsWith(std::string s, std::string prefix) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("s: \"" + s + "\"");
   mht_24_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_24(mht_24_v, 594, "", "./tensorflow/core/platform/ram_file_system.h", "StartsWith");

    return s.find(prefix) == 0;
  }

  string StripPrefix(std::string s, std::string prefix) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("s: \"" + s + "\"");
   mht_25_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_25(mht_25_v, 603, "", "./tensorflow/core/platform/ram_file_system.h", "StripPrefix");

    if (s.find(prefix) == 0) {
      return s.erase(0, prefix.size());
    }
    return s;
  }

  string StripRamFsPrefix(std::string name) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSram_file_systemDTh mht_26(mht_26_v, 614, "", "./tensorflow/core/platform/ram_file_system.h", "StripRamFsPrefix");

    std::string s = StripPrefix(name, "ram://");
    if (*(s.rbegin()) == '/') {
      s.pop_back();
    }
    return s;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_RAM_FILE_SYSTEM_H_
