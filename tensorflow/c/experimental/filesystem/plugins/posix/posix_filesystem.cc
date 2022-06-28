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
class MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc() {
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
#include "tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem_helper.h"
#include "tensorflow/c/tf_status.h"

// Implementation of a filesystem for POSIX environments.
// This filesystem will support `file://` and empty (local) URI schemes.

static void* plugin_memory_allocate(size_t size) { return calloc(1, size); }
static void plugin_memory_free(void* ptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_0(mht_0_v, 205, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "plugin_memory_free");
 free(ptr); }

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {

typedef struct PosixFile {
  const char* filename;
  int fd;
} PosixFile;

static void Cleanup(TF_RandomAccessFile* file) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_1(mht_1_v, 219, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "Cleanup");

  auto posix_file = static_cast<PosixFile*>(file->plugin_file);
  close(posix_file->fd);
  // This would be safe to free using `free` directly as it is only opaque.
  // However, it is better to be consistent everywhere.
  plugin_memory_free(const_cast<char*>(posix_file->filename));
  delete posix_file;
}

static int64_t Read(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
                    char* buffer, TF_Status* status) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_2(mht_2_v, 233, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "Read");

  auto posix_file = static_cast<PosixFile*>(file->plugin_file);
  char* dst = buffer;
  int64_t read = 0;

  while (n > 0) {
    // Some platforms, notably macs, throw `EINVAL` if `pread` is asked to read
    // more than fits in a 32-bit integer.
    size_t requested_read_length;
    if (n > INT32_MAX)
      requested_read_length = INT32_MAX;
    else
      requested_read_length = n;

    // `pread` returns a `ssize_t` on POSIX, but due to interface being
    // cross-platform, return type of `Read` is `int64_t`.
    int64_t r = int64_t{pread(posix_file->fd, dst, requested_read_length,
                              static_cast<off_t>(offset))};
    if (r > 0) {
      dst += r;
      offset += static_cast<uint64_t>(r);
      n -= r;  // safe as 0 < r <= n so n will never underflow
      read += r;
    } else if (r == 0) {
      TF_SetStatus(status, TF_OUT_OF_RANGE, "Read fewer bytes than requested");
      break;
    } else if (errno == EINTR || errno == EAGAIN) {
      // Retry
    } else {
      TF_SetStatusFromIOError(status, errno, posix_file->filename);
      break;
    }
  }

  return read;
}

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {

typedef struct PosixFile {
  const char* filename;
  FILE* handle;
} PosixFile;

static void Cleanup(TF_WritableFile* file) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_3(mht_3_v, 284, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "Cleanup");

  auto posix_file = static_cast<PosixFile*>(file->plugin_file);
  plugin_memory_free(const_cast<char*>(posix_file->filename));
  delete posix_file;
}

static void Append(const TF_WritableFile* file, const char* buffer, size_t n,
                   TF_Status* status) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_4(mht_4_v, 295, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "Append");

  auto posix_file = static_cast<PosixFile*>(file->plugin_file);

  size_t r = fwrite(buffer, 1, n, posix_file->handle);
  if (r != n)
    TF_SetStatusFromIOError(status, errno, posix_file->filename);
  else
    TF_SetStatus(status, TF_OK, "");
}

static int64_t Tell(const TF_WritableFile* file, TF_Status* status) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_5(mht_5_v, 308, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "Tell");

  auto posix_file = static_cast<PosixFile*>(file->plugin_file);

  // POSIX's `ftell` returns `long`, do a manual cast.
  int64_t position = int64_t{ftell(posix_file->handle)};
  if (position < 0)
    TF_SetStatusFromIOError(status, errno, posix_file->filename);
  else
    TF_SetStatus(status, TF_OK, "");

  return position;
}

static void Flush(const TF_WritableFile* file, TF_Status* status) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_6(mht_6_v, 324, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "Flush");

  auto posix_file = static_cast<PosixFile*>(file->plugin_file);

  TF_SetStatus(status, TF_OK, "");
  if (fflush(posix_file->handle) != 0)
    TF_SetStatusFromIOError(status, errno, posix_file->filename);
}

static void Sync(const TF_WritableFile* file, TF_Status* status) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_7(mht_7_v, 335, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "Sync");

  // For historical reasons, this does the same as `Flush` at the moment.
  // TODO(b/144055243): This should use `fsync`/`sync`.
  Flush(file, status);
}

static void Close(const TF_WritableFile* file, TF_Status* status) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_8(mht_8_v, 344, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "Close");

  auto posix_file = static_cast<PosixFile*>(file->plugin_file);

  if (fclose(posix_file->handle) != 0)
    TF_SetStatusFromIOError(status, errno, posix_file->filename);
  else
    TF_SetStatus(status, TF_OK, "");
}

}  // namespace tf_writable_file

// SECTION 3. Implementation for `TF_ReadOnlyMemoryRegion`
// ----------------------------------------------------------------------------
namespace tf_read_only_memory_region {

typedef struct PosixMemoryRegion {
  const void* const address;
  const uint64_t length;
} PosixMemoryRegion;

static void Cleanup(TF_ReadOnlyMemoryRegion* region) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_9(mht_9_v, 367, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "Cleanup");

  auto r = static_cast<PosixMemoryRegion*>(region->plugin_memory_region);
  munmap(const_cast<void*>(r->address), r->length);
  delete r;
}

static const void* Data(const TF_ReadOnlyMemoryRegion* region) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_10(mht_10_v, 376, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "Data");

  auto r = static_cast<PosixMemoryRegion*>(region->plugin_memory_region);
  return r->address;
}

static uint64_t Length(const TF_ReadOnlyMemoryRegion* region) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_11(mht_11_v, 384, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "Length");

  auto r = static_cast<PosixMemoryRegion*>(region->plugin_memory_region);
  return r->length;
}

}  // namespace tf_read_only_memory_region

// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------
namespace tf_posix_filesystem {

static void Init(TF_Filesystem* filesystem, TF_Status* status) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_12(mht_12_v, 398, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "Init");

  TF_SetStatus(status, TF_OK, "");
}

static void Cleanup(TF_Filesystem* filesystem) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_13(mht_13_v, 405, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "Cleanup");
}

static void NewRandomAccessFile(const TF_Filesystem* filesystem,
                                const char* path, TF_RandomAccessFile* file,
                                TF_Status* status) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("path: \"" + (path == nullptr ? std::string("nullptr") : std::string((char*)path)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_14(mht_14_v, 413, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "NewRandomAccessFile");

  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    TF_SetStatusFromIOError(status, errno, path);
    return;
  }

  struct stat st;
  fstat(fd, &st);
  if (S_ISDIR(st.st_mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is a directory");
    close(fd);
    return;
  }

  file->plugin_file = new tf_random_access_file::PosixFile({strdup(path), fd});
  TF_SetStatus(status, TF_OK, "");
}

static void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                            TF_WritableFile* file, TF_Status* status) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("path: \"" + (path == nullptr ? std::string("nullptr") : std::string((char*)path)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_15(mht_15_v, 437, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "NewWritableFile");

  FILE* f = fopen(path, "w");
  if (f == nullptr) {
    TF_SetStatusFromIOError(status, errno, path);
    return;
  }

  file->plugin_file = new tf_writable_file::PosixFile({strdup(path), f});
  TF_SetStatus(status, TF_OK, "");
}

static void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                              TF_WritableFile* file, TF_Status* status) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("path: \"" + (path == nullptr ? std::string("nullptr") : std::string((char*)path)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_16(mht_16_v, 453, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "NewAppendableFile");

  FILE* f = fopen(path, "a");
  if (f == nullptr) {
    TF_SetStatusFromIOError(status, errno, path);
    return;
  }

  file->plugin_file = new tf_writable_file::PosixFile({strdup(path), f});
  TF_SetStatus(status, TF_OK, "");
}

static void NewReadOnlyMemoryRegionFromFile(const TF_Filesystem* filesystem,
                                            const char* path,
                                            TF_ReadOnlyMemoryRegion* region,
                                            TF_Status* status) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("path: \"" + (path == nullptr ? std::string("nullptr") : std::string((char*)path)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_17(mht_17_v, 471, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "NewReadOnlyMemoryRegionFromFile");

  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    TF_SetStatusFromIOError(status, errno, path);
    return;
  }

  struct stat st;
  fstat(fd, &st);
  if (S_ISDIR(st.st_mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is a directory");
  } else {
    const void* address =
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (address == MAP_FAILED) {
      TF_SetStatusFromIOError(status, errno, path);
    } else {
      region->plugin_memory_region =
          new tf_read_only_memory_region::PosixMemoryRegion{
              address, static_cast<uint64_t>(st.st_size)};
      TF_SetStatus(status, TF_OK, "");
    }
  }

  close(fd);
}

static void CreateDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("path: \"" + (path == nullptr ? std::string("nullptr") : std::string((char*)path)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_18(mht_18_v, 503, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "CreateDir");

  if (strlen(path) == 0)
    TF_SetStatus(status, TF_ALREADY_EXISTS, "already exists");
  else if (mkdir(path, /*mode=*/0755) != 0)
    TF_SetStatusFromIOError(status, errno, path);
  else
    TF_SetStatus(status, TF_OK, "");
}

static void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("path: \"" + (path == nullptr ? std::string("nullptr") : std::string((char*)path)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_19(mht_19_v, 517, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "DeleteFile");

  if (unlink(path) != 0)
    TF_SetStatusFromIOError(status, errno, path);
  else
    TF_SetStatus(status, TF_OK, "");
}

static void DeleteDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("path: \"" + (path == nullptr ? std::string("nullptr") : std::string((char*)path)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_20(mht_20_v, 529, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "DeleteDir");

  if (rmdir(path) != 0)
    TF_SetStatusFromIOError(status, errno, path);
  else
    TF_SetStatus(status, TF_OK, "");
}

static void RenameFile(const TF_Filesystem* filesystem, const char* src,
                       const char* dst, TF_Status* status) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("src: \"" + (src == nullptr ? std::string("nullptr") : std::string((char*)src)) + "\"");
   mht_21_v.push_back("dst: \"" + (dst == nullptr ? std::string("nullptr") : std::string((char*)dst)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_21(mht_21_v, 542, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "RenameFile");

  // If target is a directory return TF_FAILED_PRECONDITION.
  // Target might be missing, so don't error in that case.
  struct stat st;
  if (stat(dst, &st) != 0) {
    if (errno != ENOENT) {
      TF_SetStatusFromIOError(status, errno, dst);
      return;
    }
  } else if (S_ISDIR(st.st_mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "target path is a directory");
    return;
  }

  // We cannot rename directories yet, so prevent this.
  if (stat(src, &st) != 0) {
    TF_SetStatusFromIOError(status, errno, src);
    return;
  } else if (S_ISDIR(st.st_mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "source path is a directory");
    return;
  }

  // Do the actual rename. Here both arguments are filenames.
  if (rename(src, dst) != 0)
    TF_SetStatusFromIOError(status, errno, dst);
  else
    TF_SetStatus(status, TF_OK, "");
}

static void CopyFile(const TF_Filesystem* filesystem, const char* src,
                     const char* dst, TF_Status* status) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("src: \"" + (src == nullptr ? std::string("nullptr") : std::string((char*)src)) + "\"");
   mht_22_v.push_back("dst: \"" + (dst == nullptr ? std::string("nullptr") : std::string((char*)dst)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_22(mht_22_v, 578, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "CopyFile");

  // If target is a directory return TF_FAILED_PRECONDITION.
  // Target might be missing, so don't error in that case.
  struct stat st;
  if (stat(dst, &st) != 0) {
    if (errno != ENOENT) {
      TF_SetStatusFromIOError(status, errno, dst);
      return;
    }
  } else if (S_ISDIR(st.st_mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "target path is a directory");
    return;
  }

  // We cannot copy directories yet, so prevent this.
  if (stat(src, &st) != 0) {
    TF_SetStatusFromIOError(status, errno, src);
    return;
  } else if (S_ISDIR(st.st_mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "source path is a directory");
    return;
  }

  // Both `src` and `dst` point to files here. Delegate to helper.
  if (TransferFileContents(src, dst, st.st_mode, st.st_size) < 0)
    TF_SetStatusFromIOError(status, errno, dst);
  else
    TF_SetStatus(status, TF_OK, "");
}

static void PathExists(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("path: \"" + (path == nullptr ? std::string("nullptr") : std::string((char*)path)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_23(mht_23_v, 613, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "PathExists");

  if (access(path, F_OK) != 0)
    TF_SetStatusFromIOError(status, errno, path);
  else
    TF_SetStatus(status, TF_OK, "");
}

static void Stat(const TF_Filesystem* filesystem, const char* path,
                 TF_FileStatistics* stats, TF_Status* status) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("path: \"" + (path == nullptr ? std::string("nullptr") : std::string((char*)path)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_24(mht_24_v, 625, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "Stat");

  struct stat sbuf;
  if (stat(path, &sbuf) != 0) {
    TF_SetStatusFromIOError(status, errno, path);
  } else {
    stats->length = sbuf.st_size;
    stats->mtime_nsec = sbuf.st_mtime * (1000 * 1000 * 1000);
    stats->is_directory = S_ISDIR(sbuf.st_mode);
    TF_SetStatus(status, TF_OK, "");
  }
}

static int GetChildren(const TF_Filesystem* filesystem, const char* path,
                       char*** entries, TF_Status* status) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("path: \"" + (path == nullptr ? std::string("nullptr") : std::string((char*)path)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_25(mht_25_v, 642, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "GetChildren");

  struct dirent** dir_entries = nullptr;
  /* we don't promise entries would be sorted */
  int num_entries =
      scandir(path, &dir_entries, RemoveSpecialDirectoryEntries, nullptr);
  if (num_entries < 0) {
    TF_SetStatusFromIOError(status, errno, path);
  } else {
    *entries = static_cast<char**>(
        plugin_memory_allocate(num_entries * sizeof((*entries)[0])));
    for (int i = 0; i < num_entries; i++) {
      (*entries)[i] = strdup(dir_entries[i]->d_name);
      plugin_memory_free(dir_entries[i]);
    }
    plugin_memory_free(dir_entries);
  }

  return num_entries;
}

}  // namespace tf_posix_filesystem

static void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops,
                                        const char* uri) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("uri: \"" + (uri == nullptr ? std::string("nullptr") : std::string((char*)uri)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_26(mht_26_v, 669, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "ProvideFilesystemSupportFor");

  TF_SetFilesystemVersionMetadata(ops);
  ops->scheme = strdup(uri);

  ops->random_access_file_ops = static_cast<TF_RandomAccessFileOps*>(
      plugin_memory_allocate(TF_RANDOM_ACCESS_FILE_OPS_SIZE));
  ops->random_access_file_ops->cleanup = tf_random_access_file::Cleanup;
  ops->random_access_file_ops->read = tf_random_access_file::Read;

  ops->writable_file_ops = static_cast<TF_WritableFileOps*>(
      plugin_memory_allocate(TF_WRITABLE_FILE_OPS_SIZE));
  ops->writable_file_ops->cleanup = tf_writable_file::Cleanup;
  ops->writable_file_ops->append = tf_writable_file::Append;
  ops->writable_file_ops->tell = tf_writable_file::Tell;
  ops->writable_file_ops->flush = tf_writable_file::Flush;
  ops->writable_file_ops->sync = tf_writable_file::Sync;
  ops->writable_file_ops->close = tf_writable_file::Close;

  ops->read_only_memory_region_ops = static_cast<TF_ReadOnlyMemoryRegionOps*>(
      plugin_memory_allocate(TF_READ_ONLY_MEMORY_REGION_OPS_SIZE));
  ops->read_only_memory_region_ops->cleanup =
      tf_read_only_memory_region::Cleanup;
  ops->read_only_memory_region_ops->data = tf_read_only_memory_region::Data;
  ops->read_only_memory_region_ops->length = tf_read_only_memory_region::Length;

  ops->filesystem_ops = static_cast<TF_FilesystemOps*>(
      plugin_memory_allocate(TF_FILESYSTEM_OPS_SIZE));
  ops->filesystem_ops->init = tf_posix_filesystem::Init;
  ops->filesystem_ops->cleanup = tf_posix_filesystem::Cleanup;
  ops->filesystem_ops->new_random_access_file =
      tf_posix_filesystem::NewRandomAccessFile;
  ops->filesystem_ops->new_writable_file = tf_posix_filesystem::NewWritableFile;
  ops->filesystem_ops->new_appendable_file =
      tf_posix_filesystem::NewAppendableFile;
  ops->filesystem_ops->new_read_only_memory_region_from_file =
      tf_posix_filesystem::NewReadOnlyMemoryRegionFromFile;
  ops->filesystem_ops->create_dir = tf_posix_filesystem::CreateDir;
  ops->filesystem_ops->delete_file = tf_posix_filesystem::DeleteFile;
  ops->filesystem_ops->delete_dir = tf_posix_filesystem::DeleteDir;
  ops->filesystem_ops->rename_file = tf_posix_filesystem::RenameFile;
  ops->filesystem_ops->copy_file = tf_posix_filesystem::CopyFile;
  ops->filesystem_ops->path_exists = tf_posix_filesystem::PathExists;
  ops->filesystem_ops->stat = tf_posix_filesystem::Stat;
  ops->filesystem_ops->get_children = tf_posix_filesystem::GetChildren;
}

void TF_InitPlugin(TF_FilesystemPluginInfo* info) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSposixPSposix_filesystemDTcc mht_27(mht_27_v, 718, "", "./tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.cc", "TF_InitPlugin");

  info->plugin_memory_allocate = plugin_memory_allocate;
  info->plugin_memory_free = plugin_memory_free;
  info->num_schemes = 2;
  info->ops = static_cast<TF_FilesystemPluginOps*>(
      plugin_memory_allocate(info->num_schemes * sizeof(info->ops[0])));
  ProvideFilesystemSupportFor(&info->ops[0], "");
  ProvideFilesystemSupportFor(&info->ops[1], "file");
}
