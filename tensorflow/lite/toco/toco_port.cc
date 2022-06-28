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
class MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/toco/toco_port.h"

#include <cstring>

#include "absl/status/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/toco_types.h"

#if defined(__ANDROID__) && defined(__ARM_ARCH_7A__)
namespace std {
double round(double x) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/toco/toco_port.cc", "round");
 return ::round(x); }
}  // namespace std
#endif

namespace toco {
namespace port {
void CopyToBuffer(const std::string& src, char* dest) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("src: \"" + src + "\"");
   mht_1_v.push_back("dest: \"" + (dest == nullptr ? std::string("nullptr") : std::string((char*)dest)) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_1(mht_1_v, 207, "", "./tensorflow/lite/toco/toco_port.cc", "CopyToBuffer");

  memcpy(dest, src.data(), src.size());
}

#ifdef PLATFORM_GOOGLE
void CopyToBuffer(const absl::Cord& src, char* dest) { src.CopyToArray(dest); }
#endif
}  // namespace port
}  // namespace toco

#if defined(PLATFORM_GOOGLE) && !defined(__APPLE__) && \
    !defined(__ANDROID__) && !defined(_WIN32)

// Wrap Google file operations.

#include "base/init_google.h"
#include "file/base/file.h"
#include "file/base/filesystem.h"
#include "file/base/helpers.h"
#include "file/base/options.h"
#include "file/base/path.h"

namespace toco {
namespace port {

void InitGoogle(const char* usage, int* argc, char*** argv, bool remove_flags) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("usage: \"" + (usage == nullptr ? std::string("nullptr") : std::string((char*)usage)) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_2(mht_2_v, 236, "", "./tensorflow/lite/toco/toco_port.cc", "InitGoogle");

  ::InitGoogle(usage, argc, argv, remove_flags);
}

void InitGoogleWasDoneElsewhere() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_3(mht_3_v, 243, "", "./tensorflow/lite/toco/toco_port.cc", "InitGoogleWasDoneElsewhere");

  // Nothing need be done since ::CheckInitGoogleIsDone() is aware of other
  // possible initialization entry points.
}

void CheckInitGoogleIsDone(const char* message) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("message: \"" + (message == nullptr ? std::string("nullptr") : std::string((char*)message)) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_4(mht_4_v, 252, "", "./tensorflow/lite/toco/toco_port.cc", "CheckInitGoogleIsDone");

  ::CheckInitGoogleIsDone(message);
}

namespace file {

// Conversion to our wrapper Status.
tensorflow::Status ToStatus(const absl::Status& uts) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_5(mht_5_v, 262, "", "./tensorflow/lite/toco/toco_port.cc", "ToStatus");

  if (!uts.ok()) {
    return tensorflow::Status(
        tensorflow::errors::Code(::util::RetrieveErrorCode(uts)),
        uts.error_message());
  }
  return tensorflow::Status::OK();
}

// Conversion to our wrapper Options.
toco::port::file::Options ToOptions(const ::file::Options& options) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_6(mht_6_v, 275, "", "./tensorflow/lite/toco/toco_port.cc", "ToOptions");

  CHECK_EQ(&options, &::file::Defaults());
  return Options();
}

tensorflow::Status Writable(const std::string& filename) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_7(mht_7_v, 284, "", "./tensorflow/lite/toco/toco_port.cc", "Writable");

  File* f = nullptr;
  const auto status = ::file::Open(filename, "w", &f, ::file::Defaults());
  if (f) {
    QCHECK_OK(f->Close(::file::Defaults()));
  }
  return ToStatus(status);
}

tensorflow::Status Readable(const std::string& filename,
                            const file::Options& options) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_8(mht_8_v, 298, "", "./tensorflow/lite/toco/toco_port.cc", "Readable");

  return ToStatus(::file::Readable(filename, ::file::Defaults()));
}

tensorflow::Status Exists(const std::string& filename,
                          const file::Options& options) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_9(mht_9_v, 307, "", "./tensorflow/lite/toco/toco_port.cc", "Exists");

  auto status = ::file::Exists(filename, ::file::Defaults());
  return ToStatus(status);
}

tensorflow::Status GetContents(const std::string& filename,
                               std::string* contents,
                               const file::Options& options) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_10(mht_10_v, 318, "", "./tensorflow/lite/toco/toco_port.cc", "GetContents");

  return ToStatus(::file::GetContents(filename, contents, ::file::Defaults()));
}

tensorflow::Status SetContents(const std::string& filename,
                               const std::string& contents,
                               const file::Options& options) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("filename: \"" + filename + "\"");
   mht_11_v.push_back("contents: \"" + contents + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_11(mht_11_v, 329, "", "./tensorflow/lite/toco/toco_port.cc", "SetContents");

  return ToStatus(::file::SetContents(filename, contents, ::file::Defaults()));
}

std::string JoinPath(const std::string& a, const std::string& b) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("a: \"" + a + "\"");
   mht_12_v.push_back("b: \"" + b + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_12(mht_12_v, 338, "", "./tensorflow/lite/toco/toco_port.cc", "JoinPath");

  return ::file::JoinPath(a, b);
}

}  // namespace file
}  // namespace port
}  // namespace toco

#else  // !PLATFORM_GOOGLE || __APPLE__ || __ANDROID__ || _WIN32

#include <fcntl.h>
#if defined(_WIN32)
#include <io.h>  // for _close, _open, _read
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdio>

#if defined(PLATFORM_GOOGLE)
#include "base/commandlineflags.h"
#endif

namespace toco {
namespace port {

#if defined(_WIN32)
#define close _close
#define open _open
#define read _read
// Windows does not support the same set of file permissions as other platforms,
// and also requires an explicit flag for binary file read/write support.
constexpr int kFileCreateMode = _S_IREAD | _S_IWRITE;
constexpr int kFileReadFlags = _O_RDONLY | _O_BINARY;
constexpr int kFileWriteFlags = _O_WRONLY | _O_BINARY | _O_CREAT;
#else
constexpr int kFileCreateMode = 0664;
constexpr int kFileReadFlags = O_RDONLY;
constexpr int kFileWriteFlags = O_CREAT | O_WRONLY;
#endif  // _WIN32

static bool port_initialized = false;

void InitGoogleWasDoneElsewhere() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_13(mht_13_v, 384, "", "./tensorflow/lite/toco/toco_port.cc", "InitGoogleWasDoneElsewhere");
 port_initialized = true; }

void InitGoogle(const char* usage, int* argc, char*** argv, bool remove_flags) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("usage: \"" + (usage == nullptr ? std::string("nullptr") : std::string((char*)usage)) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_14(mht_14_v, 390, "", "./tensorflow/lite/toco/toco_port.cc", "InitGoogle");

  if (!port_initialized) {
#if defined(PLATFORM_GOOGLE)
    ParseCommandLineFlags(argc, argv, remove_flags);
#endif
    port_initialized = true;
  }
}

void CheckInitGoogleIsDone(const char* message) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("message: \"" + (message == nullptr ? std::string("nullptr") : std::string((char*)message)) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_15(mht_15_v, 403, "", "./tensorflow/lite/toco/toco_port.cc", "CheckInitGoogleIsDone");

  CHECK(port_initialized) << message;
}

namespace file {

tensorflow::Status Writable(const string& filename) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_16(mht_16_v, 413, "", "./tensorflow/lite/toco/toco_port.cc", "Writable");

  FILE* f = fopen(filename.c_str(), "w");
  if (f) {
    fclose(f);
    return tensorflow::Status::OK();
  }
  return tensorflow::errors::NotFound("not writable");
}

tensorflow::Status Readable(const string& filename,
                            const file::Options& options) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_17(mht_17_v, 427, "", "./tensorflow/lite/toco/toco_port.cc", "Readable");

  FILE* f = fopen(filename.c_str(), "r");
  if (f) {
    fclose(f);
    return tensorflow::Status::OK();
  }
  return tensorflow::errors::NotFound("not readable");
}

tensorflow::Status Exists(const string& filename,
                          const file::Options& options) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_18(mht_18_v, 441, "", "./tensorflow/lite/toco/toco_port.cc", "Exists");

  struct stat statbuf;
  int ret = stat(filename.c_str(), &statbuf);
  if (ret == -1) {
    return tensorflow::errors::NotFound("file doesn't exist");
  }
  return tensorflow::Status::OK();
}

tensorflow::Status GetContents(const string& path, string* output,
                               const file::Options& options) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_19(mht_19_v, 455, "", "./tensorflow/lite/toco/toco_port.cc", "GetContents");

  output->clear();

  int fd = open(path.c_str(), kFileReadFlags);
  if (fd == -1) {
    return tensorflow::errors::NotFound("can't open() for read");
  }

  // Direct read, for speed.
  const int kBufSize = 1 << 16;
  char buffer[kBufSize];
  while (true) {
    int size = read(fd, buffer, kBufSize);
    if (size == 0) {
      // Done.
      close(fd);
      return tensorflow::Status::OK();
    } else if (size == -1) {
      // Error.
      close(fd);
      return tensorflow::errors::Internal("error during read()");
    } else {
      output->append(buffer, size);
    }
  }

  CHECK(0);
  return tensorflow::errors::Internal("internal error");
}

tensorflow::Status SetContents(const string& filename, const string& contents,
                               const file::Options& options) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("filename: \"" + filename + "\"");
   mht_20_v.push_back("contents: \"" + contents + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_20(mht_20_v, 491, "", "./tensorflow/lite/toco/toco_port.cc", "SetContents");

  int fd = open(filename.c_str(), kFileWriteFlags, kFileCreateMode);
  if (fd == -1) {
    return tensorflow::errors::Internal("can't open() for write");
  }

  size_t i = 0;
  while (i < contents.size()) {
    size_t to_write = contents.size() - i;
    ssize_t written = write(fd, &contents[i], to_write);
    if (written == -1) {
      close(fd);
      return tensorflow::errors::Internal("write() error");
    }
    i += written;
  }
  close(fd);

  return tensorflow::Status::OK();
}

string JoinPath(const string& base, const string& filename) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("base: \"" + base + "\"");
   mht_21_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_portDTcc mht_21(mht_21_v, 517, "", "./tensorflow/lite/toco/toco_port.cc", "JoinPath");

  if (base.empty()) return filename;
  string base_fixed = base;
  if (!base_fixed.empty() && base_fixed.back() == '/') base_fixed.pop_back();
  string filename_fixed = filename;
  if (!filename_fixed.empty() && filename_fixed.front() == '/')
    filename_fixed.erase(0, 1);
  return base_fixed + "/" + filename_fixed;
}

}  // namespace file
}  // namespace port
}  // namespace toco

#endif  // !PLATFORM_GOOGLE || __APPLE || __ANDROID__ || _WIN32
