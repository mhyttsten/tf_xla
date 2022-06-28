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
class MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc() {
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
#include "tensorflow/core/util/memmapped_file_system.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/memmapped_file_system.pb.h"

namespace tensorflow {

namespace {

uint64 DecodeUint64LittleEndian(const uint8* buffer) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/util/memmapped_file_system.cc", "DecodeUint64LittleEndian");

  uint64 result = 0;
  for (int i = 0; i < static_cast<int>(sizeof(uint64)); ++i) {
    result |= static_cast<uint64>(buffer[i]) << (8 * i);
  }
  return result;
}

}  // namespace

namespace {

class ReadOnlyMemoryRegionFromMemmapped : public ReadOnlyMemoryRegion {
 public:
  ReadOnlyMemoryRegionFromMemmapped(const void* data, uint64 length)
      : data_(data), length_(length) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/util/memmapped_file_system.cc", "ReadOnlyMemoryRegionFromMemmapped");
}
  ~ReadOnlyMemoryRegionFromMemmapped() override = default;
  const void* data() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_2(mht_2_v, 218, "", "./tensorflow/core/util/memmapped_file_system.cc", "data");
 return data_; }
  uint64 length() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_3(mht_3_v, 222, "", "./tensorflow/core/util/memmapped_file_system.cc", "length");
 return length_; }

 private:
  const void* const data_;
  const uint64 length_;
  // intentionally copyable
};

class RandomAccessFileFromMemmapped : public RandomAccessFile {
 public:
  RandomAccessFileFromMemmapped(const void* data, uint64 length)
      : data_(data), length_(length) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_4(mht_4_v, 236, "", "./tensorflow/core/util/memmapped_file_system.cc", "RandomAccessFileFromMemmapped");
}

  ~RandomAccessFileFromMemmapped() override = default;

  Status Name(StringPiece* result) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_5(mht_5_v, 243, "", "./tensorflow/core/util/memmapped_file_system.cc", "Name");

    return errors::Unimplemented(
        "RandomAccessFileFromMemmapped does not support Name()");
  }

  Status Read(uint64 offset, size_t to_read, StringPiece* result,
              char* scratch) const override {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("scratch: \"" + (scratch == nullptr ? std::string("nullptr") : std::string((char*)scratch)) + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_6(mht_6_v, 253, "", "./tensorflow/core/util/memmapped_file_system.cc", "Read");

    if (offset >= length_) {
      *result = StringPiece(scratch, 0);
      return Status(error::OUT_OF_RANGE, "Read after file end");
    }
    const uint64 region_left =
        std::min(length_ - offset, static_cast<uint64>(to_read));
    *result =
        StringPiece(reinterpret_cast<const char*>(data_) + offset, region_left);
    return (region_left == to_read)
               ? Status::OK()
               : Status(error::OUT_OF_RANGE, "Read less bytes than requested");
  }

 private:
  const void* const data_;
  const uint64 length_;
  // intentionally copyable
};

}  // namespace

MemmappedFileSystem::MemmappedFileSystem() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_7(mht_7_v, 278, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::MemmappedFileSystem");
}

Status MemmappedFileSystem::FileExists(const string& fname,
                                       TransactionToken* token) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_8(mht_8_v, 285, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::FileExists");

  if (!mapped_memory_) {
    return errors::FailedPrecondition("MemmappedEnv is not initialized");
  }
  const auto dir_element = directory_.find(fname);
  if (dir_element != directory_.end()) {
    return Status::OK();
  }
  return errors::NotFound(fname, " not found");
}

Status MemmappedFileSystem::NewRandomAccessFile(
    const string& filename, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_9(mht_9_v, 302, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::NewRandomAccessFile");

  if (!mapped_memory_) {
    return errors::FailedPrecondition("MemmappedEnv is not initialized");
  }
  const auto dir_element = directory_.find(filename);
  if (dir_element == directory_.end()) {
    return errors::NotFound("Region ", filename, " is not found");
  }
  result->reset(new RandomAccessFileFromMemmapped(
      GetMemoryWithOffset(dir_element->second.offset),
      dir_element->second.length));
  return Status::OK();
}

Status MemmappedFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& filename, TransactionToken* token,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_10(mht_10_v, 322, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::NewReadOnlyMemoryRegionFromFile");

  if (!mapped_memory_) {
    return errors::FailedPrecondition("MemmappedEnv is not initialized");
  }
  const auto dir_element = directory_.find(filename);
  if (dir_element == directory_.end()) {
    return errors::NotFound("Region ", filename, " is not found");
  }
  result->reset(new ReadOnlyMemoryRegionFromMemmapped(
      GetMemoryWithOffset(dir_element->second.offset),
      dir_element->second.length));
  return Status::OK();
}

Status MemmappedFileSystem::GetFileSize(const string& filename,
                                        TransactionToken* token, uint64* size) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_11(mht_11_v, 341, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::GetFileSize");

  if (!mapped_memory_) {
    return errors::FailedPrecondition("MemmappedEnv is not initialized");
  }
  const auto dir_element = directory_.find(filename);
  if (dir_element == directory_.end()) {
    return errors::NotFound("Region ", filename, " is not found");
  }
  *size = dir_element->second.length;
  return Status::OK();
}

Status MemmappedFileSystem::Stat(const string& fname, TransactionToken* token,
                                 FileStatistics* stat) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_12(mht_12_v, 358, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::Stat");

  uint64 size;
  auto status = GetFileSize(fname, token, &size);
  if (status.ok()) {
    stat->length = size;
  }
  return status;
}

Status MemmappedFileSystem::NewWritableFile(const string& filename,
                                            TransactionToken* token,
                                            std::unique_ptr<WritableFile>* wf) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_13(mht_13_v, 373, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::NewWritableFile");

  return errors::Unimplemented("memmapped format doesn't support writing");
}

Status MemmappedFileSystem::NewAppendableFile(
    const string& filename, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_14(mht_14_v, 383, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::NewAppendableFile");

  return errors::Unimplemented("memmapped format doesn't support writing");
}

Status MemmappedFileSystem::GetChildren(const string& filename,
                                        TransactionToken* token,
                                        std::vector<string>* strings) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_15(mht_15_v, 393, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::GetChildren");

  return errors::Unimplemented("memmapped format doesn't support GetChildren");
}

Status MemmappedFileSystem::GetMatchingPaths(const string& pattern,
                                             TransactionToken* token,
                                             std::vector<string>* results) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_16(mht_16_v, 403, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::GetMatchingPaths");

  return errors::Unimplemented(
      "memmapped format doesn't support GetMatchingPaths");
}

Status MemmappedFileSystem::DeleteFile(const string& filename,
                                       TransactionToken* token) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_17(mht_17_v, 413, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::DeleteFile");

  return errors::Unimplemented("memmapped format doesn't support DeleteFile");
}

Status MemmappedFileSystem::CreateDir(const string& dirname,
                                      TransactionToken* token) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_18(mht_18_v, 422, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::CreateDir");

  return errors::Unimplemented("memmapped format doesn't support CreateDir");
}

Status MemmappedFileSystem::DeleteDir(const string& dirname,
                                      TransactionToken* token) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_19(mht_19_v, 431, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::DeleteDir");

  return errors::Unimplemented("memmapped format doesn't support DeleteDir");
}

Status MemmappedFileSystem::RenameFile(const string& filename_from,
                                       const string& filename_to,
                                       TransactionToken* token) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("filename_from: \"" + filename_from + "\"");
   mht_20_v.push_back("filename_to: \"" + filename_to + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_20(mht_20_v, 442, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::RenameFile");

  return errors::Unimplemented("memmapped format doesn't support RenameFile");
}

const void* MemmappedFileSystem::GetMemoryWithOffset(uint64 offset) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_21(mht_21_v, 449, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::GetMemoryWithOffset");

  return reinterpret_cast<const uint8*>(mapped_memory_->data()) + offset;
}

constexpr const char MemmappedFileSystem::kMemmappedPackagePrefix[];
constexpr const char MemmappedFileSystem::kMemmappedPackageDefaultGraphDef[];

Status MemmappedFileSystem::InitializeFromFile(Env* env,
                                               const string& filename) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_22(mht_22_v, 461, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::InitializeFromFile");

  TF_RETURN_IF_ERROR(
      env->NewReadOnlyMemoryRegionFromFile(filename, &mapped_memory_));
  directory_.clear();
  if (mapped_memory_->length() <= sizeof(uint64)) {
    return errors::DataLoss("Corrupted memmapped model file: ", filename,
                            " Invalid package size");
  }
  const auto memory_start =
      reinterpret_cast<const uint8*>(mapped_memory_->data());
  const uint64 directory_offset = DecodeUint64LittleEndian(
      memory_start + mapped_memory_->length() - sizeof(uint64));
  if (directory_offset > mapped_memory_->length() - sizeof(uint64)) {
    return errors::DataLoss("Corrupted memmapped model file: ", filename,
                            " Invalid directory offset");
  }
  MemmappedFileSystemDirectory proto_directory;
  if (!ParseProtoUnlimited(
          &proto_directory, memory_start + directory_offset,
          mapped_memory_->length() - directory_offset - sizeof(uint64))) {
    return errors::DataLoss("Corrupted memmapped model file: ", filename,
                            " Can't parse its internal directory");
  }

  // Iterating in reverse order to get lengths of elements;
  uint64 prev_element_offset = directory_offset;
  for (auto element_iter = proto_directory.element().rbegin();
       element_iter != proto_directory.element().rend(); ++element_iter) {
    // Check that the element offset is in the right range.
    if (element_iter->offset() >= prev_element_offset) {
      return errors::DataLoss("Corrupted memmapped model file: ", filename,
                              " Invalid offset of internal component");
    }
    if (!directory_
             .insert(std::make_pair(
                 element_iter->name(),
                 FileRegion(element_iter->offset(), element_iter->length())))
             .second) {
      return errors::DataLoss("Corrupted memmapped model file: ", filename,
                              " Duplicate name of internal component ",
                              element_iter->name());
    }
    prev_element_offset = element_iter->offset();
  }
  return Status::OK();
}

bool MemmappedFileSystem::IsMemmappedPackageFilename(const string& filename) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_23(mht_23_v, 512, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::IsMemmappedPackageFilename");

  return absl::StartsWith(filename, kMemmappedPackagePrefix);
}

namespace {
bool IsValidRegionChar(char c) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("c: '" + std::string(1, c) + "'");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_24(mht_24_v, 521, "", "./tensorflow/core/util/memmapped_file_system.cc", "IsValidRegionChar");

  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
         (c >= '0' && c <= '9') || c == '_' || c == '.';
}
}  // namespace

bool MemmappedFileSystem::IsWellFormedMemmappedPackageFilename(
    const string& filename) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_25(mht_25_v, 532, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedFileSystem::IsWellFormedMemmappedPackageFilename");

  if (!IsMemmappedPackageFilename(filename)) {
    return false;
  }
  for (char c :
       filename.substr(strlen(kMemmappedPackagePrefix),
                       filename.length() - strlen(kMemmappedPackagePrefix))) {
    if (!IsValidRegionChar(c)) {
      return false;
    }
  }
  return true;
}

MemmappedEnv::MemmappedEnv(Env* env) : EnvWrapper(env) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_26(mht_26_v, 549, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedEnv::MemmappedEnv");
}

Status MemmappedEnv::GetFileSystemForFile(const string& fname,
                                          FileSystem** result) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_27(mht_27_v, 556, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedEnv::GetFileSystemForFile");

  if (MemmappedFileSystem::IsMemmappedPackageFilename(fname)) {
    if (!memmapped_file_system_) {
      return errors::FailedPrecondition(
          "MemmappedEnv is not initialized from a file.");
    }
    *result = memmapped_file_system_.get();
    return Status::OK();
  }
  return EnvWrapper::GetFileSystemForFile(fname, result);
}

Status MemmappedEnv::GetRegisteredFileSystemSchemes(
    std::vector<string>* schemes) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_28(mht_28_v, 572, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedEnv::GetRegisteredFileSystemSchemes");

  const auto status = EnvWrapper::GetRegisteredFileSystemSchemes(schemes);
  if (status.ok()) {
    schemes->emplace_back(MemmappedFileSystem::kMemmappedPackagePrefix);
  }
  return status;
}

Status MemmappedEnv::InitializeFromFile(const string& package_filename) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("package_filename: \"" + package_filename + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_systemDTcc mht_29(mht_29_v, 584, "", "./tensorflow/core/util/memmapped_file_system.cc", "MemmappedEnv::InitializeFromFile");

  std::unique_ptr<MemmappedFileSystem> file_system_ptr(new MemmappedFileSystem);
  const auto status =
      file_system_ptr->InitializeFromFile(target(), package_filename);
  if (status.ok()) {
    memmapped_file_system_ = std::move(file_system_ptr);
  }
  return status;
}

}  // namespace tensorflow
