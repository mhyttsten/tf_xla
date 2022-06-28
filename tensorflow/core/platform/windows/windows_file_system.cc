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
class MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc() {
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

/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/platform/windows/windows_file_system.h"

#include <Shlwapi.h>
#include <Windows.h>
#include <direct.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#undef StrCat
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/windows/error_windows.h"
#include "tensorflow/core/platform/windows/wide_char.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

// TODO(mrry): Prevent this Windows.h #define from leaking out of our headers.
#undef DeleteFile

namespace tensorflow {

using ::tensorflow::errors::IOError;

namespace {

// RAII helpers for HANDLEs
const auto CloseHandleFunc = [](HANDLE h) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_0(mht_0_v, 218, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "lambda");
 ::CloseHandle(h); };
typedef std::unique_ptr<void, decltype(CloseHandleFunc)> UniqueCloseHandlePtr;

inline Status IOErrorFromWindowsError(const string& context) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("context: \"" + context + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "IOErrorFromWindowsError");

  auto last_error = ::GetLastError();
  return IOError(
      context + string(" : ") + internal::WindowsGetLastErrorMessage(),
      last_error);
}

// PLEASE NOTE: hfile is expected to be an async handle
// (i.e. opened with FILE_FLAG_OVERLAPPED)
SSIZE_T pread(HANDLE hfile, char* src, size_t num_bytes, uint64_t offset) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("src: \"" + (src == nullptr ? std::string("nullptr") : std::string((char*)src)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_2(mht_2_v, 238, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "pread");

  assert(num_bytes <= std::numeric_limits<DWORD>::max());
  OVERLAPPED overlapped = {0};
  ULARGE_INTEGER offset_union;
  offset_union.QuadPart = offset;

  overlapped.Offset = offset_union.LowPart;
  overlapped.OffsetHigh = offset_union.HighPart;
  overlapped.hEvent = ::CreateEvent(NULL, TRUE, FALSE, NULL);

  if (NULL == overlapped.hEvent) {
    return -1;
  }

  SSIZE_T result = 0;

  unsigned long bytes_read = 0;
  DWORD last_error = ERROR_SUCCESS;

  BOOL read_result = ::ReadFile(hfile, src, static_cast<DWORD>(num_bytes),
                                &bytes_read, &overlapped);
  if (TRUE == read_result) {
    result = bytes_read;
  } else if ((FALSE == read_result) &&
             ((last_error = GetLastError()) != ERROR_IO_PENDING)) {
    result = (last_error == ERROR_HANDLE_EOF) ? 0 : -1;
  } else {
    if (ERROR_IO_PENDING ==
        last_error) {  // Otherwise bytes_read already has the result.
      BOOL overlapped_result =
          ::GetOverlappedResult(hfile, &overlapped, &bytes_read, TRUE);
      if (FALSE == overlapped_result) {
        result = (::GetLastError() == ERROR_HANDLE_EOF) ? 0 : -1;
      } else {
        result = bytes_read;
      }
    }
  }

  ::CloseHandle(overlapped.hEvent);

  return result;
}

// read() based random-access
class WindowsRandomAccessFile : public RandomAccessFile {
 private:
  string filename_;
  HANDLE hfile_;

 public:
  WindowsRandomAccessFile(const string& fname, HANDLE hfile)
      : filename_(fname), hfile_(hfile) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_3(mht_3_v, 294, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsRandomAccessFile");
}
  ~WindowsRandomAccessFile() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_4(mht_4_v, 298, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "~WindowsRandomAccessFile");

    if (hfile_ != NULL && hfile_ != INVALID_HANDLE_VALUE) {
      ::CloseHandle(hfile_);
    }
  }

  Status Name(StringPiece* result) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_5(mht_5_v, 307, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "Name");

    *result = filename_;
    return Status::OK();
  }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("scratch: \"" + (scratch == nullptr ? std::string("nullptr") : std::string((char*)scratch)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_6(mht_6_v, 317, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "Read");

    Status s;
    char* dst = scratch;
    while (n > 0 && s.ok()) {
      size_t requested_read_length;
      if (n > std::numeric_limits<DWORD>::max()) {
        requested_read_length = std::numeric_limits<DWORD>::max();
      } else {
        requested_read_length = n;
      }
      SSIZE_T r = pread(hfile_, dst, requested_read_length, offset);
      if (r > 0) {
        offset += r;
        dst += r;
        n -= r;
      } else if (r == 0) {
        s = Status(error::OUT_OF_RANGE, "Read fewer bytes than requested");
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

class WindowsWritableFile : public WritableFile {
 private:
  string filename_;
  HANDLE hfile_;

 public:
  WindowsWritableFile(const string& fname, HANDLE hFile)
      : filename_(fname), hfile_(hFile) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_7(mht_7_v, 384, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsWritableFile");
}

  ~WindowsWritableFile() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_8(mht_8_v, 389, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "~WindowsWritableFile");

    if (hfile_ != NULL && hfile_ != INVALID_HANDLE_VALUE) {
      WindowsWritableFile::Close();
    }
  }

  Status Append(StringPiece data) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_9(mht_9_v, 398, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "Append");

    DWORD bytes_written = 0;
    DWORD data_size = static_cast<DWORD>(data.size());
    BOOL write_result =
        ::WriteFile(hfile_, data.data(), data_size, &bytes_written, NULL);
    if (FALSE == write_result) {
      return IOErrorFromWindowsError("Failed to WriteFile: " + filename_);
    }

    assert(size_t(bytes_written) == data.size());
    return Status::OK();
  }

#if defined(TF_CORD_SUPPORT)
  // \brief Append 'data' to the file.
  Status Append(const absl::Cord& cord) override {
    for (const auto& chunk : cord.Chunks()) {
      DWORD bytes_written = 0;
      DWORD data_size = static_cast<DWORD>(chunk.size());
      BOOL write_result =
          ::WriteFile(hfile_, chunk.data(), data_size, &bytes_written, NULL);
      if (FALSE == write_result) {
        return IOErrorFromWindowsError("Failed to WriteFile: " + filename_);
      }

      assert(size_t(bytes_written) == chunk.size());
    }
    return Status::OK();
  }
#endif

  Status Tell(int64* position) override {
    Status result = Flush();
    if (!result.ok()) {
      return result;
    }

    *position = SetFilePointer(hfile_, 0, NULL, FILE_CURRENT);

    if (*position == INVALID_SET_FILE_POINTER) {
      return IOErrorFromWindowsError("Tell(SetFilePointer) failed for: " +
                                     filename_);
    }

    return Status::OK();
  }

  Status Close() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_10(mht_10_v, 448, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "Close");

    assert(INVALID_HANDLE_VALUE != hfile_);

    Status result = Flush();
    if (!result.ok()) {
      return result;
    }

    if (FALSE == ::CloseHandle(hfile_)) {
      return IOErrorFromWindowsError("CloseHandle failed for: " + filename_);
    }

    hfile_ = INVALID_HANDLE_VALUE;
    return Status::OK();
  }

  Status Flush() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_11(mht_11_v, 467, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "Flush");

    if (FALSE == ::FlushFileBuffers(hfile_)) {
      return IOErrorFromWindowsError("FlushFileBuffers failed for: " +
                                     filename_);
    }
    return Status::OK();
  }

  Status Name(StringPiece* result) const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_12(mht_12_v, 478, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "Name");

    *result = filename_;
    return Status::OK();
  }

  Status Sync() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_13(mht_13_v, 486, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "Sync");
 return Flush(); }
};

class WinReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 private:
  const std::string filename_;
  HANDLE hfile_;
  HANDLE hmap_;

  const void* const address_;
  const uint64 length_;

 public:
  WinReadOnlyMemoryRegion(const std::string& filename, HANDLE hfile,
                          HANDLE hmap, const void* address, uint64 length)
      : filename_(filename),
        hfile_(hfile),
        hmap_(hmap),
        address_(address),
        length_(length) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_14(mht_14_v, 509, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WinReadOnlyMemoryRegion");
}

  ~WinReadOnlyMemoryRegion() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_15(mht_15_v, 514, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "~WinReadOnlyMemoryRegion");

    BOOL ret = ::UnmapViewOfFile(address_);
    assert(ret);

    ret = ::CloseHandle(hmap_);
    assert(ret);

    ret = ::CloseHandle(hfile_);
    assert(ret);
  }

  const void* data() override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_16(mht_16_v, 528, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "data");
 return address_; }
  uint64 length() override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_17(mht_17_v, 532, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "length");
 return length_; }
};

}  // namespace

Status WindowsFileSystem::NewRandomAccessFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_18(mht_18_v, 543, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsFileSystem::NewRandomAccessFile");

  string translated_fname = TranslateName(fname);
  std::wstring ws_translated_fname = Utf8ToWideChar(translated_fname);
  result->reset();

  // Open the file for read-only random access
  // Open in async mode which makes Windows allow more parallelism even
  // if we need to do sync I/O on top of it.
  DWORD file_flags = FILE_ATTRIBUTE_READONLY | FILE_FLAG_OVERLAPPED;
  // Shared access is necessary for tests to pass
  // almost all tests would work with a possible exception of fault_injection.
  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;

  HANDLE hfile =
      ::CreateFileW(ws_translated_fname.c_str(), GENERIC_READ, share_mode, NULL,
                    OPEN_EXISTING, file_flags, NULL);

  if (INVALID_HANDLE_VALUE == hfile) {
    string context = "NewRandomAccessFile failed to Create/Open: " + fname;
    return IOErrorFromWindowsError(context);
  }

  result->reset(new WindowsRandomAccessFile(translated_fname, hfile));
  return Status::OK();
}

Status WindowsFileSystem::NewWritableFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_19(mht_19_v, 575, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsFileSystem::NewWritableFile");

  string translated_fname = TranslateName(fname);
  std::wstring ws_translated_fname = Utf8ToWideChar(translated_fname);
  result->reset();

  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  HANDLE hfile =
      ::CreateFileW(ws_translated_fname.c_str(), GENERIC_WRITE, share_mode,
                    NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

  if (INVALID_HANDLE_VALUE == hfile) {
    string context = "Failed to create a NewWriteableFile: " + fname;
    return IOErrorFromWindowsError(context);
  }

  result->reset(new WindowsWritableFile(translated_fname, hfile));
  return Status::OK();
}

Status WindowsFileSystem::NewAppendableFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_20(mht_20_v, 600, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsFileSystem::NewAppendableFile");

  string translated_fname = TranslateName(fname);
  std::wstring ws_translated_fname = Utf8ToWideChar(translated_fname);
  result->reset();

  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  HANDLE hfile =
      ::CreateFileW(ws_translated_fname.c_str(), GENERIC_WRITE, share_mode,
                    NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

  if (INVALID_HANDLE_VALUE == hfile) {
    string context = "Failed to create a NewAppendableFile: " + fname;
    return IOErrorFromWindowsError(context);
  }

  UniqueCloseHandlePtr file_guard(hfile, CloseHandleFunc);

  DWORD file_ptr = ::SetFilePointer(hfile, NULL, NULL, FILE_END);
  if (INVALID_SET_FILE_POINTER == file_ptr) {
    string context = "Failed to create a NewAppendableFile: " + fname;
    return IOErrorFromWindowsError(context);
  }

  result->reset(new WindowsWritableFile(translated_fname, hfile));
  file_guard.release();

  return Status::OK();
}

Status WindowsFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_21(mht_21_v, 635, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsFileSystem::NewReadOnlyMemoryRegionFromFile");

  string translated_fname = TranslateName(fname);
  std::wstring ws_translated_fname = Utf8ToWideChar(translated_fname);
  result->reset();
  Status s = Status::OK();

  // Open the file for read-only
  DWORD file_flags = FILE_ATTRIBUTE_READONLY;

  // Open in async mode which makes Windows allow more parallelism even
  // if we need to do sync I/O on top of it.
  file_flags |= FILE_FLAG_OVERLAPPED;

  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  HANDLE hfile =
      ::CreateFileW(ws_translated_fname.c_str(), GENERIC_READ, share_mode, NULL,
                    OPEN_EXISTING, file_flags, NULL);

  if (INVALID_HANDLE_VALUE == hfile) {
    return IOErrorFromWindowsError(
        "NewReadOnlyMemoryRegionFromFile failed to Create/Open: " + fname);
  }

  UniqueCloseHandlePtr file_guard(hfile, CloseHandleFunc);

  // Use mmap when virtual address-space is plentiful.
  uint64_t file_size;
  s = GetFileSize(translated_fname, &file_size);
  if (s.ok()) {
    // Will not map empty files
    if (file_size == 0) {
      return IOError(
          "NewReadOnlyMemoryRegionFromFile failed to map empty file: " + fname,
          EINVAL);
    }

    HANDLE hmap = ::CreateFileMappingA(hfile, NULL, PAGE_READONLY,
                                       0,  // Whole file at its present length
                                       0,
                                       NULL);  // Mapping name

    if (!hmap) {
      string context =
          "Failed to create file mapping for "
          "NewReadOnlyMemoryRegionFromFile: " +
          fname;
      return IOErrorFromWindowsError(context);
    }

    UniqueCloseHandlePtr map_guard(hmap, CloseHandleFunc);

    const void* mapped_region =
        ::MapViewOfFileEx(hmap, FILE_MAP_READ,
                          0,  // High DWORD of access start
                          0,  // Low DWORD
                          file_size,
                          NULL);  // Let the OS choose the mapping

    if (!mapped_region) {
      string context =
          "Failed to MapViewOfFile for "
          "NewReadOnlyMemoryRegionFromFile: " +
          fname;
      return IOErrorFromWindowsError(context);
    }

    result->reset(new WinReadOnlyMemoryRegion(fname, hfile, hmap, mapped_region,
                                              file_size));

    map_guard.release();
    file_guard.release();
  }

  return s;
}

Status WindowsFileSystem::FileExists(const string& fname,
                                     TransactionToken* token) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_22(mht_22_v, 716, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsFileSystem::FileExists");

  constexpr int kOk = 0;
  std::wstring ws_translated_fname = Utf8ToWideChar(TranslateName(fname));
  if (_waccess(ws_translated_fname.c_str(), kOk) == 0) {
    return Status::OK();
  }
  return errors::NotFound(fname, " not found");
}

Status WindowsFileSystem::GetChildren(const string& dir,
                                      TransactionToken* token,
                                      std::vector<string>* result) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_23(mht_23_v, 731, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsFileSystem::GetChildren");

  string translated_dir = TranslateName(dir);
  std::wstring ws_translated_dir = Utf8ToWideChar(translated_dir);
  result->clear();

  std::wstring pattern = ws_translated_dir;
  if (!pattern.empty() && pattern.back() != '\\' && pattern.back() != '/') {
    pattern += L"\\*";
  } else {
    pattern += L'*';
  }

  WIN32_FIND_DATAW find_data;
  HANDLE find_handle = ::FindFirstFileW(pattern.c_str(), &find_data);
  if (find_handle == INVALID_HANDLE_VALUE) {
    string context = "FindFirstFile failed for: " + translated_dir;
    return IOErrorFromWindowsError(context);
  }

  do {
    string file_name = WideCharToUtf8(find_data.cFileName);
    const StringPiece basename = file_name;
    if (basename != "." && basename != "..") {
      result->push_back(file_name);
    }
  } while (::FindNextFileW(find_handle, &find_data));

  if (!::FindClose(find_handle)) {
    string context = "FindClose failed for: " + translated_dir;
    return IOErrorFromWindowsError(context);
  }

  return Status::OK();
}

Status WindowsFileSystem::DeleteFile(const string& fname,
                                     TransactionToken* token) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_24(mht_24_v, 771, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsFileSystem::DeleteFile");

  Status result;
  std::wstring file_name = Utf8ToWideChar(fname);
  if (_wunlink(file_name.c_str()) != 0) {
    result = IOError("Failed to delete a file: " + fname, errno);
  }
  return result;
}

Status WindowsFileSystem::CreateDir(const string& name,
                                    TransactionToken* token) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_25(mht_25_v, 785, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsFileSystem::CreateDir");

  Status result;
  std::wstring ws_name = Utf8ToWideChar(name);
  if (ws_name.empty()) {
    return errors::AlreadyExists(name);
  }
  if (_wmkdir(ws_name.c_str()) != 0) {
    result = IOError("Failed to create a directory: " + name, errno);
  }
  return result;
}

Status WindowsFileSystem::DeleteDir(const string& name,
                                    TransactionToken* token) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_26(mht_26_v, 802, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsFileSystem::DeleteDir");

  Status result;
  std::wstring ws_name = Utf8ToWideChar(name);
  if (_wrmdir(ws_name.c_str()) != 0) {
    result = IOError("Failed to remove a directory: " + name, errno);
  }
  return result;
}

Status WindowsFileSystem::GetFileSize(const string& fname,
                                      TransactionToken* token, uint64* size) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_27(mht_27_v, 816, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsFileSystem::GetFileSize");

  string translated_fname = TranslateName(fname);
  std::wstring ws_translated_dir = Utf8ToWideChar(translated_fname);
  Status result;
  WIN32_FILE_ATTRIBUTE_DATA attrs;
  if (TRUE == ::GetFileAttributesExW(ws_translated_dir.c_str(),
                                     GetFileExInfoStandard, &attrs)) {
    ULARGE_INTEGER file_size;
    file_size.HighPart = attrs.nFileSizeHigh;
    file_size.LowPart = attrs.nFileSizeLow;
    *size = file_size.QuadPart;
  } else {
    string context = "Can not get size for: " + fname;
    result = IOErrorFromWindowsError(context);
  }
  return result;
}

Status WindowsFileSystem::IsDirectory(const string& fname,
                                      TransactionToken* token) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_28(mht_28_v, 839, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsFileSystem::IsDirectory");

  TF_RETURN_IF_ERROR(FileExists(fname));
  std::wstring ws_translated_fname = Utf8ToWideChar(TranslateName(fname));
  if (PathIsDirectoryW(ws_translated_fname.c_str())) {
    return Status::OK();
  }
  return Status(tensorflow::error::FAILED_PRECONDITION, "Not a directory");
}

Status WindowsFileSystem::RenameFile(const string& src, const string& target,
                                     TransactionToken* token) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("src: \"" + src + "\"");
   mht_29_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_29(mht_29_v, 854, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsFileSystem::RenameFile");

  // rename() is not capable of replacing the existing file as on Linux
  // so use OS API directly
  std::wstring ws_translated_src = Utf8ToWideChar(TranslateName(src));
  std::wstring ws_translated_target = Utf8ToWideChar(TranslateName(target));

  // Calling MoveFileExW with the MOVEFILE_REPLACE_EXISTING flag can fail if
  // another process has a handle to the file that it didn't close yet. On the
  // other hand, calling DeleteFileW + MoveFileExW will work in that scenario
  // because it allows the process to keep using the old handle while also
  // creating a new handle for the new file.
  WIN32_FIND_DATAW find_file_data;
  HANDLE target_file_handle =
      ::FindFirstFileW(ws_translated_target.c_str(), &find_file_data);
  if (target_file_handle != INVALID_HANDLE_VALUE) {
    if (!::DeleteFileW(ws_translated_target.c_str())) {
      ::FindClose(target_file_handle);
      return IOErrorFromWindowsError(
          strings::StrCat("Failed to rename: ", src, " to: ", target));
    }
    ::FindClose(target_file_handle);
  }

  if (!::MoveFileExW(ws_translated_src.c_str(), ws_translated_target.c_str(),
                     0)) {
    return IOErrorFromWindowsError(
        strings::StrCat("Failed to rename: ", src, " to: ", target));
  }

  return Status::OK();
}

Status WindowsFileSystem::GetMatchingPaths(const string& pattern,
                                           TransactionToken* token,
                                           std::vector<string>* results) {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_30(mht_30_v, 892, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsFileSystem::GetMatchingPaths");

  // NOTE(mrry): The existing implementation of FileSystem::GetMatchingPaths()
  // does not handle Windows paths containing backslashes correctly. Since
  // Windows APIs will accept forward and backslashes equivalently, we
  // convert the pattern to use forward slashes exclusively. Note that this
  // is not ideal, since the API expects backslash as an escape character,
  // but no code appears to rely on this behavior.
  string converted_pattern(pattern);
  std::replace(converted_pattern.begin(), converted_pattern.end(), '\\', '/');
  TF_RETURN_IF_ERROR(internal::GetMatchingPaths(this, Env::Default(),
                                                converted_pattern, results));
  for (string& result : *results) {
    std::replace(result.begin(), result.end(), '/', '\\');
  }
  return Status::OK();
}

bool WindowsFileSystem::Match(const string& filename, const string& pattern) {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("filename: \"" + filename + "\"");
   mht_31_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_31(mht_31_v, 914, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsFileSystem::Match");

  std::wstring ws_path(Utf8ToWideChar(filename));
  std::wstring ws_pattern(Utf8ToWideChar(pattern));
  return PathMatchSpecW(ws_path.c_str(), ws_pattern.c_str()) == TRUE;
}

Status WindowsFileSystem::Stat(const string& fname, TransactionToken* token,
                               FileStatistics* stat) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSwindows_file_systemDTcc mht_32(mht_32_v, 925, "", "./tensorflow/core/platform/windows/windows_file_system.cc", "WindowsFileSystem::Stat");

  Status result;
  struct _stat64 sbuf;
  std::wstring ws_translated_fname = Utf8ToWideChar(TranslateName(fname));
  if (_wstat64(ws_translated_fname.c_str(), &sbuf) != 0) {
    result = IOError(fname, errno);
  } else {
    stat->mtime_nsec = sbuf.st_mtime * 1e9;
    stat->length = sbuf.st_size;
    stat->is_directory = IsDirectory(fname).ok();
  }
  return result;
}

}  // namespace tensorflow
