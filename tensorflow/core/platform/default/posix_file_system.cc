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
class MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>

#if defined(__linux__)
#include <sys/sendfile.h>
#endif
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "tensorflow/core/platform/default/posix_file_system.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

using ::tensorflow::errors::IOError;

// 128KB of copy buffer
constexpr size_t kPosixCopyFileBufferSize = 128 * 1024;

// pread() based random-access
class PosixRandomAccessFile : public RandomAccessFile {
 private:
  string filename_;
  int fd_;

 public:
  PosixRandomAccessFile(const string& fname, int fd)
      : filename_(fname), fd_(fd) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_0(mht_0_v, 226, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixRandomAccessFile");
}
  ~PosixRandomAccessFile() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/platform/default/posix_file_system.cc", "~PosixRandomAccessFile");

    if (close(fd_) < 0) {
      LOG(ERROR) << "close() failed: " << strerror(errno);
    }
  }

  Status Name(StringPiece* result) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_2(mht_2_v, 239, "", "./tensorflow/core/platform/default/posix_file_system.cc", "Name");

    *result = filename_;
    return Status::OK();
  }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("scratch: \"" + (scratch == nullptr ? std::string("nullptr") : std::string((char*)scratch)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_3(mht_3_v, 249, "", "./tensorflow/core/platform/default/posix_file_system.cc", "Read");

    Status s;
    char* dst = scratch;
    while (n > 0 && s.ok()) {
      // Some platforms, notably macs, throw EINVAL if pread is asked to read
      // more than fits in a 32-bit integer.
      size_t requested_read_length;
      if (n > INT32_MAX) {
        requested_read_length = INT32_MAX;
      } else {
        requested_read_length = n;
      }
      ssize_t r =
          pread(fd_, dst, requested_read_length, static_cast<off_t>(offset));
      if (r > 0) {
        dst += r;
        n -= r;
        offset += r;
      } else if (r == 0) {
        s = Status(error::OUT_OF_RANGE, "Read less bytes than requested");
      } else if (errno == EINTR || errno == EAGAIN) {
        // Retry
      } else {
        s = IOError(filename_, errno);
      }
    }
    *result = StringPiece(scratch, dst - scratch);
    return s;
  }

#if defined(TF_CORD_SUPPORT)
  Status Read(uint64 offset, size_t n, absl::Cord* cord) const override {
    if (n == 0) {
      return Status::OK();
    }
    if (n < 0) {
      return errors::InvalidArgument(
          "Attempting to read ", n,
          " bytes. You cannot read a negative number of bytes.");
    }

    char* scratch = new char[n];
    if (scratch == nullptr) {
      return errors::ResourceExhausted("Unable to allocate ", n,
                                       " bytes for file reading.");
    }

    StringPiece tmp;
    Status s = Read(offset, n, &tmp, scratch);

    absl::Cord tmp_cord = absl::MakeCordFromExternal(
        absl::string_view(static_cast<char*>(scratch), tmp.size()),
        [scratch](absl::string_view) { delete[] scratch; });
    cord->Append(tmp_cord);
    return s;
  }
#endif
};

class PosixWritableFile : public WritableFile {
 private:
  string filename_;
  FILE* file_;

 public:
  PosixWritableFile(const string& fname, FILE* f)
      : filename_(fname), file_(f) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_4(mht_4_v, 319, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixWritableFile");
}

  ~PosixWritableFile() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_5(mht_5_v, 324, "", "./tensorflow/core/platform/default/posix_file_system.cc", "~PosixWritableFile");

    if (file_ != nullptr) {
      // Ignoring any potential errors
      fclose(file_);
    }
  }

  Status Append(StringPiece data) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_6(mht_6_v, 334, "", "./tensorflow/core/platform/default/posix_file_system.cc", "Append");

    size_t r = fwrite(data.data(), 1, data.size(), file_);
    if (r != data.size()) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

#if defined(TF_CORD_SUPPORT)
  // \brief Append 'cord' to the file.
  Status Append(const absl::Cord& cord) override {
    for (const auto& chunk : cord.Chunks()) {
      size_t r = fwrite(chunk.data(), 1, chunk.size(), file_);
      if (r != chunk.size()) {
        return IOError(filename_, errno);
      }
    }
    return Status::OK();
  }
#endif

  Status Close() override {
    if (file_ == nullptr) {
      return IOError(filename_, EBADF);
    }
    Status result;
    if (fclose(file_) != 0) {
      result = IOError(filename_, errno);
    }
    file_ = nullptr;
    return result;
  }

  Status Flush() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_7(mht_7_v, 370, "", "./tensorflow/core/platform/default/posix_file_system.cc", "Flush");

    if (fflush(file_) != 0) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status Name(StringPiece* result) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_8(mht_8_v, 380, "", "./tensorflow/core/platform/default/posix_file_system.cc", "Name");

    *result = filename_;
    return Status::OK();
  }

  Status Sync() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_9(mht_9_v, 388, "", "./tensorflow/core/platform/default/posix_file_system.cc", "Sync");

    Status s;
    if (fflush(file_) != 0) {
      s = IOError(filename_, errno);
    }
    return s;
  }

  Status Tell(int64_t* position) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_10(mht_10_v, 399, "", "./tensorflow/core/platform/default/posix_file_system.cc", "Tell");

    Status s;
    *position = ftell(file_);

    if (*position == -1) {
      s = IOError(filename_, errno);
    }

    return s;
  }
};

class PosixReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  PosixReadOnlyMemoryRegion(const void* address, uint64 length)
      : address_(address), length_(length) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_11(mht_11_v, 417, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixReadOnlyMemoryRegion");
}
  ~PosixReadOnlyMemoryRegion() override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_12(mht_12_v, 421, "", "./tensorflow/core/platform/default/posix_file_system.cc", "~PosixReadOnlyMemoryRegion");

    munmap(const_cast<void*>(address_), length_);
  }
  const void* data() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_13(mht_13_v, 427, "", "./tensorflow/core/platform/default/posix_file_system.cc", "data");
 return address_; }
  uint64 length() override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_14(mht_14_v, 431, "", "./tensorflow/core/platform/default/posix_file_system.cc", "length");
 return length_; }

 private:
  const void* const address_;
  const uint64 length_;
};

Status PosixFileSystem::NewRandomAccessFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_15(mht_15_v, 444, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixFileSystem::NewRandomAccessFile");

  string translated_fname = TranslateName(fname);
  Status s;
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixRandomAccessFile(translated_fname, fd));
  }
  return s;
}

Status PosixFileSystem::NewWritableFile(const string& fname,
                                        TransactionToken* token,
                                        std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_16(mht_16_v, 462, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixFileSystem::NewWritableFile");

  string translated_fname = TranslateName(fname);
  Status s;
  FILE* f = fopen(translated_fname.c_str(), "w");
  if (f == nullptr) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixWritableFile(translated_fname, f));
  }
  return s;
}

Status PosixFileSystem::NewAppendableFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_17(mht_17_v, 480, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixFileSystem::NewAppendableFile");

  string translated_fname = TranslateName(fname);
  Status s;
  FILE* f = fopen(translated_fname.c_str(), "a");
  if (f == nullptr) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixWritableFile(translated_fname, f));
  }
  return s;
}

Status PosixFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_18(mht_18_v, 498, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixFileSystem::NewReadOnlyMemoryRegionFromFile");

  string translated_fname = TranslateName(fname);
  Status s = Status::OK();
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    s = IOError(fname, errno);
  } else {
    struct stat st;
    ::fstat(fd, &st);
    const void* address =
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (address == MAP_FAILED) {
      s = IOError(fname, errno);
    } else {
      result->reset(new PosixReadOnlyMemoryRegion(address, st.st_size));
    }
    if (close(fd) < 0) {
      s = IOError(fname, errno);
    }
  }
  return s;
}

Status PosixFileSystem::FileExists(const string& fname,
                                   TransactionToken* token) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_19(mht_19_v, 526, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixFileSystem::FileExists");

  if (access(TranslateName(fname).c_str(), F_OK) == 0) {
    return Status::OK();
  }
  return errors::NotFound(fname, " not found");
}

Status PosixFileSystem::GetChildren(const string& dir, TransactionToken* token,
                                    std::vector<string>* result) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_20(mht_20_v, 538, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixFileSystem::GetChildren");

  string translated_dir = TranslateName(dir);
  result->clear();
  DIR* d = opendir(translated_dir.c_str());
  if (d == nullptr) {
    return IOError(dir, errno);
  }
  struct dirent* entry;
  while ((entry = readdir(d)) != nullptr) {
    StringPiece basename = entry->d_name;
    if ((basename != ".") && (basename != "..")) {
      result->push_back(entry->d_name);
    }
  }
  if (closedir(d) < 0) {
    return IOError(dir, errno);
  }
  return Status::OK();
}

Status PosixFileSystem::GetMatchingPaths(const string& pattern,
                                         TransactionToken* token,
                                         std::vector<string>* results) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_21(mht_21_v, 564, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixFileSystem::GetMatchingPaths");

  return internal::GetMatchingPaths(this, Env::Default(), pattern, results);
}

Status PosixFileSystem::DeleteFile(const string& fname,
                                   TransactionToken* token) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_22(mht_22_v, 573, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixFileSystem::DeleteFile");

  Status result;
  if (unlink(TranslateName(fname).c_str()) != 0) {
    result = IOError(fname, errno);
  }
  return result;
}

Status PosixFileSystem::CreateDir(const string& name, TransactionToken* token) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_23(mht_23_v, 585, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixFileSystem::CreateDir");

  string translated = TranslateName(name);
  if (translated.empty()) {
    return errors::AlreadyExists(name);
  }
  if (mkdir(translated.c_str(), 0755) != 0) {
    return IOError(name, errno);
  }
  return Status::OK();
}

Status PosixFileSystem::DeleteDir(const string& name, TransactionToken* token) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_24(mht_24_v, 600, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixFileSystem::DeleteDir");

  Status result;
  if (rmdir(TranslateName(name).c_str()) != 0) {
    result = IOError(name, errno);
  }
  return result;
}

Status PosixFileSystem::GetFileSize(const string& fname,
                                    TransactionToken* token, uint64* size) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_25(mht_25_v, 613, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixFileSystem::GetFileSize");

  Status s;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    *size = 0;
    s = IOError(fname, errno);
  } else {
    *size = sbuf.st_size;
  }
  return s;
}

Status PosixFileSystem::Stat(const string& fname, TransactionToken* token,
                             FileStatistics* stats) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_26(mht_26_v, 630, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixFileSystem::Stat");

  Status s;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    s = IOError(fname, errno);
  } else {
    stats->length = sbuf.st_size;
    stats->mtime_nsec = sbuf.st_mtime * 1e9;
    stats->is_directory = S_ISDIR(sbuf.st_mode);
  }
  return s;
}

Status PosixFileSystem::RenameFile(const string& src, const string& target,
                                   TransactionToken* token) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("src: \"" + src + "\"");
   mht_27_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_27(mht_27_v, 649, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixFileSystem::RenameFile");

  Status result;
  if (rename(TranslateName(src).c_str(), TranslateName(target).c_str()) != 0) {
    result = IOError(src, errno);
  }
  return result;
}

Status PosixFileSystem::CopyFile(const string& src, const string& target,
                                 TransactionToken* token) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("src: \"" + src + "\"");
   mht_28_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSposix_file_systemDTcc mht_28(mht_28_v, 663, "", "./tensorflow/core/platform/default/posix_file_system.cc", "PosixFileSystem::CopyFile");

  string translated_src = TranslateName(src);
  struct stat sbuf;
  if (stat(translated_src.c_str(), &sbuf) != 0) {
    return IOError(src, errno);
  }
  int src_fd = open(translated_src.c_str(), O_RDONLY);
  if (src_fd < 0) {
    return IOError(src, errno);
  }
  string translated_target = TranslateName(target);
  // O_WRONLY | O_CREAT | O_TRUNC:
  //   Open file for write and if file does not exist, create the file.
  //   If file exists, truncate its size to 0.
  // When creating file, use the same permissions as original
  mode_t mode = sbuf.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
  int target_fd =
      open(translated_target.c_str(), O_WRONLY | O_CREAT | O_TRUNC, mode);
  if (target_fd < 0) {
    close(src_fd);
    return IOError(target, errno);
  }
  int rc = 0;
  off_t offset = 0;
  std::unique_ptr<char[]> buffer(new char[kPosixCopyFileBufferSize]);
  while (offset < sbuf.st_size) {
    // Use uint64 for safe compare SSIZE_MAX
    uint64 chunk = sbuf.st_size - offset;
    if (chunk > SSIZE_MAX) {
      chunk = SSIZE_MAX;
    }
#if defined(__linux__) && !defined(__ANDROID__)
    rc = sendfile(target_fd, src_fd, &offset, static_cast<size_t>(chunk));
#else
    if (chunk > kPosixCopyFileBufferSize) {
      chunk = kPosixCopyFileBufferSize;
    }
    rc = read(src_fd, buffer.get(), static_cast<size_t>(chunk));
    if (rc <= 0) {
      break;
    }
    rc = write(target_fd, buffer.get(), static_cast<size_t>(chunk));
    offset += chunk;
#endif
    if (rc <= 0) {
      break;
    }
  }

  Status result = Status::OK();
  if (rc < 0) {
    result = IOError(target, errno);
  }

  // Keep the error code
  rc = close(target_fd);
  if (rc < 0 && result == Status::OK()) {
    result = IOError(target, errno);
  }
  rc = close(src_fd);
  if (rc < 0 && result == Status::OK()) {
    result = IOError(target, errno);
  }

  return result;
}

}  // namespace tensorflow
