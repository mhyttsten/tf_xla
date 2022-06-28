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
class MHTracer_DTPStensorflowPScorePSplatformPSpathDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSpathDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSpathDTcc() {
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

#include "tensorflow/core/platform/path.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#if defined(PLATFORM_WINDOWS)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/scanner.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {
namespace io {
namespace internal {
namespace {

const char kPathSep[] = "/";

bool FixBazelEnvPath(const char* path, string* out) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("path: \"" + (path == nullptr ? std::string("nullptr") : std::string((char*)path)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSpathDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/platform/path.cc", "FixBazelEnvPath");

  if (path == nullptr) return false;
  if (out == nullptr) return true;

  *out = path;

#ifdef PLATFORM_WINDOWS
  // On Windows, paths generated by Bazel are always use `/` as the path
  // separator. This prevents normal path management. In the event there are no
  // `\` in the path, we convert all `/` to `\`.
  if (out->find('\\') != string::npos) return path;

  for (size_t pos = out->find('/'); pos != string::npos;
       pos = out->find('/', pos + 1)) {
    (*out)[pos] = kPathSep[0];
  }
#endif

  return true;
}

}  // namespace

string JoinPathImpl(std::initializer_list<StringPiece> paths) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSpathDTcc mht_1(mht_1_v, 240, "", "./tensorflow/core/platform/path.cc", "JoinPathImpl");

  string result;

  for (StringPiece path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = string(path);
      continue;
    }

    if (IsAbsolutePath(path)) path = path.substr(1);

    if (result[result.size() - 1] == kPathSep[0]) {
      strings::StrAppend(&result, path);
    } else {
      strings::StrAppend(&result, kPathSep, path);
    }
  }

  return result;
}

// Return the parts of the URI, split on the final "/" in the path. If there is
// no "/" in the path, the first part of the output is the scheme and host, and
// the second is the path. If the only "/" in the path is the first character,
// it is included in the first part of the output.
std::pair<StringPiece, StringPiece> SplitPath(StringPiece uri) {
  StringPiece scheme, host, path;
  ParseURI(uri, &scheme, &host, &path);

  auto pos = path.rfind('/');
#ifdef PLATFORM_WINDOWS
  if (pos == StringPiece::npos) pos = path.rfind('\\');
#endif
  // Handle the case with no '/' in 'path'.
  if (pos == StringPiece::npos)
    return std::make_pair(StringPiece(uri.begin(), host.end() - uri.begin()),
                          path);

  // Handle the case with a single leading '/' in 'path'.
  if (pos == 0)
    return std::make_pair(
        StringPiece(uri.begin(), path.begin() + 1 - uri.begin()),
        StringPiece(path.data() + 1, path.size() - 1));

  return std::make_pair(
      StringPiece(uri.begin(), path.begin() + pos - uri.begin()),
      StringPiece(path.data() + pos + 1, path.size() - (pos + 1)));
}

// Return the parts of the basename of path, split on the final ".".
// If there is no "." in the basename or "." is the final character in the
// basename, the second value will be empty.
std::pair<StringPiece, StringPiece> SplitBasename(StringPiece path) {
  path = Basename(path);

  auto pos = path.rfind('.');
  if (pos == StringPiece::npos)
    return std::make_pair(path, StringPiece(path.data() + path.size(), 0));
  return std::make_pair(
      StringPiece(path.data(), pos),
      StringPiece(path.data() + pos + 1, path.size() - (pos + 1)));
}

}  // namespace internal

bool IsAbsolutePath(StringPiece path) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSpathDTcc mht_2(mht_2_v, 310, "", "./tensorflow/core/platform/path.cc", "IsAbsolutePath");

  return !path.empty() && path[0] == '/';
}

StringPiece Dirname(StringPiece path) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSpathDTcc mht_3(mht_3_v, 317, "", "./tensorflow/core/platform/path.cc", "Dirname");

  return internal::SplitPath(path).first;
}

StringPiece Basename(StringPiece path) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSpathDTcc mht_4(mht_4_v, 324, "", "./tensorflow/core/platform/path.cc", "Basename");

  return internal::SplitPath(path).second;
}

StringPiece Extension(StringPiece path) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSpathDTcc mht_5(mht_5_v, 331, "", "./tensorflow/core/platform/path.cc", "Extension");

  return internal::SplitBasename(path).second;
}

string CleanPath(StringPiece unclean_path) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSpathDTcc mht_6(mht_6_v, 338, "", "./tensorflow/core/platform/path.cc", "CleanPath");

  string path(unclean_path);
  const char* src = path.c_str();
  string::iterator dst = path.begin();

  // Check for absolute path and determine initial backtrack limit.
  const bool is_absolute_path = *src == '/';
  if (is_absolute_path) {
    *dst++ = *src++;
    while (*src == '/') ++src;
  }
  string::const_iterator backtrack_limit = dst;

  // Process all parts
  while (*src) {
    bool parsed = false;

    if (src[0] == '.') {
      //  1dot ".<whateverisnext>", check for END or SEP.
      if (src[1] == '/' || !src[1]) {
        if (*++src) {
          ++src;
        }
        parsed = true;
      } else if (src[1] == '.' && (src[2] == '/' || !src[2])) {
        // 2dot END or SEP (".." | "../<whateverisnext>").
        src += 2;
        if (dst != backtrack_limit) {
          // We can backtrack the previous part
          for (--dst; dst != backtrack_limit && dst[-1] != '/'; --dst) {
            // Empty.
          }
        } else if (!is_absolute_path) {
          // Failed to backtrack and we can't skip it either. Rewind and copy.
          src -= 2;
          *dst++ = *src++;
          *dst++ = *src++;
          if (*src) {
            *dst++ = *src;
          }
          // We can never backtrack over a copied "../" part so set new limit.
          backtrack_limit = dst;
        }
        if (*src) {
          ++src;
        }
        parsed = true;
      }
    }

    // If not parsed, copy entire part until the next SEP or EOS.
    if (!parsed) {
      while (*src && *src != '/') {
        *dst++ = *src++;
      }
      if (*src) {
        *dst++ = *src++;
      }
    }

    // Skip consecutive SEP occurrences
    while (*src == '/') {
      ++src;
    }
  }

  // Calculate and check the length of the cleaned path.
  string::difference_type path_length = dst - path.begin();
  if (path_length != 0) {
    // Remove trailing '/' except if it is root path ("/" ==> path_length := 1)
    if (path_length > 1 && path[path_length - 1] == '/') {
      --path_length;
    }
    path.resize(path_length);
  } else {
    // The cleaned path is empty; assign "." as per the spec.
    path.assign(1, '.');
  }
  return path;
}

void ParseURI(StringPiece remaining, StringPiece* scheme, StringPiece* host,
              StringPiece* path) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSpathDTcc mht_7(mht_7_v, 423, "", "./tensorflow/core/platform/path.cc", "ParseURI");

  // 0. Parse scheme
  // Make sure scheme matches [a-zA-Z][0-9a-zA-Z.]*
  // TODO(keveman): Allow "+" and "-" in the scheme.
  // Keep URI pattern in TensorBoard's `_parse_event_files_spec` updated
  // accordingly
  if (!strings::Scanner(remaining)
           .One(strings::Scanner::LETTER)
           .Many(strings::Scanner::LETTER_DIGIT_DOT)
           .StopCapture()
           .OneLiteral("://")
           .GetResult(&remaining, scheme)) {
    // If there's no scheme, assume the entire string is a path.
    *scheme = StringPiece(remaining.begin(), 0);
    *host = StringPiece(remaining.begin(), 0);
    *path = remaining;
    return;
  }

  // 1. Parse host
  if (!strings::Scanner(remaining).ScanUntil('/').GetResult(&remaining, host)) {
    // No path, so the rest of the URI is the host.
    *host = remaining;
    *path = StringPiece(remaining.end(), 0);
    return;
  }

  // 2. The rest is the path
  *path = remaining;
}

string CreateURI(StringPiece scheme, StringPiece host, StringPiece path) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSpathDTcc mht_8(mht_8_v, 457, "", "./tensorflow/core/platform/path.cc", "CreateURI");

  if (scheme.empty()) {
    return string(path);
  }
  return strings::StrCat(scheme, "://", host, path);
}

// Returns a unique number every time it is called.
int64_t UniqueId() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSpathDTcc mht_9(mht_9_v, 468, "", "./tensorflow/core/platform/path.cc", "UniqueId");

  static mutex mu(LINKER_INITIALIZED);
  static int64_t id = 0;
  mutex_lock l(mu);
  return ++id;
}

string CommonPathPrefix(absl::Span<const string> paths) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSpathDTcc mht_10(mht_10_v, 478, "", "./tensorflow/core/platform/path.cc", "CommonPathPrefix");

  if (paths.empty()) return "";
  size_t min_filename_size =
      absl::c_min_element(paths, [](const string& a, const string& b) {
        return a.size() < b.size();
      })->size();
  if (min_filename_size == 0) return "";

  size_t common_prefix_size = [&] {
    for (size_t prefix_size = 0; prefix_size < min_filename_size;
         prefix_size++) {
      char c = paths[0][prefix_size];
      for (int f = 1; f < paths.size(); f++) {
        if (paths[f][prefix_size] != c) {
          return prefix_size;
        }
      }
    }
    return min_filename_size;
  }();

  size_t rpos = absl::string_view(paths[0])
                    .substr(0, common_prefix_size)
                    .rfind(internal::kPathSep);
  return rpos == std::string::npos
             ? ""
             : std::string(absl::string_view(paths[0]).substr(0, rpos + 1));
}

string GetTempFilename(const string& extension) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("extension: \"" + extension + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSpathDTcc mht_11(mht_11_v, 511, "", "./tensorflow/core/platform/path.cc", "GetTempFilename");

#if defined(__ANDROID__)
  LOG(FATAL) << "GetTempFilename is not implemented in this platform.";
#elif defined(PLATFORM_WINDOWS)
  char temp_dir[_MAX_PATH];
  DWORD retval;
  retval = GetTempPath(_MAX_PATH, temp_dir);
  if (retval > _MAX_PATH || retval == 0) {
    LOG(FATAL) << "Cannot get the directory for temporary files.";
  }

  char temp_file_name[_MAX_PATH];
  retval = GetTempFileName(temp_dir, "", UniqueId(), temp_file_name);
  if (retval > _MAX_PATH || retval == 0) {
    LOG(FATAL) << "Cannot get a temporary file in: " << temp_dir;
  }

  string full_tmp_file_name(temp_file_name);
  full_tmp_file_name.append(extension);
  return full_tmp_file_name;
#else
  for (const char* dir : std::vector<const char*>(
           {getenv("TEST_TMPDIR"), getenv("TMPDIR"), getenv("TMP"), "/tmp"})) {
    if (!dir || !dir[0]) {
      continue;
    }
    struct stat statbuf;
    if (!stat(dir, &statbuf) && S_ISDIR(statbuf.st_mode)) {
      // UniqueId is added here because mkstemps is not as thread safe as it
      // looks. https://github.com/tensorflow/tensorflow/issues/5804 shows
      // the problem.
      string tmp_filepath;
      int fd;
      if (extension.length()) {
        tmp_filepath = io::JoinPath(
            dir, strings::StrCat("tmp_file_tensorflow_", UniqueId(), "_XXXXXX.",
                                 extension));
        fd = mkstemps(&tmp_filepath[0], extension.length() + 1);
      } else {
        tmp_filepath = io::JoinPath(
            dir,
            strings::StrCat("tmp_file_tensorflow_", UniqueId(), "_XXXXXX"));
        fd = mkstemp(&tmp_filepath[0]);
      }
      if (fd < 0) {
        LOG(FATAL) << "Failed to create temp file.";
      } else {
        if (close(fd) < 0) {
          LOG(ERROR) << "close() failed: " << strerror(errno);
        }
        return tmp_filepath;
      }
    }
  }
  LOG(FATAL) << "No temp directory found.";
  std::abort();
#endif
}

bool GetTestUndeclaredOutputsDir(string* dir) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSpathDTcc mht_12(mht_12_v, 573, "", "./tensorflow/core/platform/path.cc", "GetTestUndeclaredOutputsDir");

  return internal::FixBazelEnvPath(getenv("TEST_UNDECLARED_OUTPUTS_DIR"), dir);
}

}  // namespace io
}  // namespace tensorflow
