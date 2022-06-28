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
class MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc() {
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

#include "tensorflow/core/platform/file_system.h"

#include <sys/stat.h>

#include <algorithm>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#if defined(PLATFORM_POSIX) || defined(IS_MOBILE_PLATFORM)
#include <fnmatch.h>
#else
#include "tensorflow/core/platform/regexp.h"
#endif  // defined(PLATFORM_POSIX) || defined(IS_MOBILE_PLATFORM)

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/scanner.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {

bool FileSystem::Match(const string& filename, const string& pattern) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("filename: \"" + filename + "\"");
   mht_0_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::Match");

#if defined(PLATFORM_POSIX) || defined(IS_MOBILE_PLATFORM)
  // We avoid relying on RE2 on mobile platforms, because it incurs a
  // significant binary size increase.
  // For POSIX platforms, there is no need to depend on RE2 if `fnmatch` can be
  // used safely.
  return fnmatch(pattern.c_str(), filename.c_str(), FNM_PATHNAME) == 0;
#else
  string regexp(pattern);
  regexp = str_util::StringReplace(regexp, "*", "[^/]*", true);
  regexp = str_util::StringReplace(regexp, "?", ".", true);
  regexp = str_util::StringReplace(regexp, "(", "\\(", true);
  regexp = str_util::StringReplace(regexp, ")", "\\)", true);
  return RE2::FullMatch(filename, regexp);
#endif  // defined(PLATFORM_POSIX) || defined(IS_MOBILE_PLATFORM)
}

string FileSystem::TranslateName(const string& name) const {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_1(mht_1_v, 233, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::TranslateName");

  // If the name is empty, CleanPath returns "." which is incorrect and
  // we should return the empty path instead.
  if (name.empty()) return name;

  // Otherwise, properly separate the URI components and clean the path one
  StringPiece scheme, host, path;
  this->ParseURI(name, &scheme, &host, &path);

  // If `path` becomes empty, return `/` (`file://` should be `/`), not `.`.
  if (path.empty()) return "/";

  return this->CleanPath(path);
}

Status FileSystem::IsDirectory(const string& name, TransactionToken* token) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_2(mht_2_v, 252, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::IsDirectory");

  // Check if path exists.
  // TODO(sami):Forward token to other methods once migration is complete.
  TF_RETURN_IF_ERROR(FileExists(name));
  FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(name, &stat));
  if (stat.is_directory) {
    return Status::OK();
  }
  return Status(tensorflow::error::FAILED_PRECONDITION, "Not a directory");
}

Status FileSystem::HasAtomicMove(const string& path, bool* has_atomic_move) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_3(mht_3_v, 268, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::HasAtomicMove");

  *has_atomic_move = true;
  return Status::OK();
}

void FileSystem::FlushCaches(TransactionToken* token) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_4(mht_4_v, 276, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::FlushCaches");
}

bool FileSystem::FilesExist(const std::vector<string>& files,
                            TransactionToken* token,
                            std::vector<Status>* status) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_5(mht_5_v, 283, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::FilesExist");

  bool result = true;
  for (const auto& file : files) {
    Status s = FileExists(file);
    result &= s.ok();
    if (status != nullptr) {
      status->push_back(s);
    } else if (!result) {
      // Return early since there is no need to check other files.
      return false;
    }
  }
  return result;
}

Status FileSystem::DeleteRecursively(const string& dirname,
                                     TransactionToken* token,
                                     int64_t* undeleted_files,
                                     int64_t* undeleted_dirs) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_6(mht_6_v, 305, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::DeleteRecursively");

  CHECK_NOTNULL(undeleted_files);
  CHECK_NOTNULL(undeleted_dirs);

  *undeleted_files = 0;
  *undeleted_dirs = 0;
  // Make sure that dirname exists;
  Status exists_status = FileExists(dirname);
  if (!exists_status.ok()) {
    (*undeleted_dirs)++;
    return exists_status;
  }

  // If given path to a single file, we should just delete it.
  if (!IsDirectory(dirname).ok()) {
    Status delete_root_status = DeleteFile(dirname);
    if (!delete_root_status.ok()) (*undeleted_files)++;
    return delete_root_status;
  }

  std::deque<string> dir_q;      // Queue for the BFS
  std::vector<string> dir_list;  // List of all dirs discovered
  dir_q.push_back(dirname);
  Status ret;  // Status to be returned.
  // Do a BFS on the directory to discover all the sub-directories. Remove all
  // children that are files along the way. Then cleanup and remove the
  // directories in reverse order.;
  while (!dir_q.empty()) {
    string dir = dir_q.front();
    dir_q.pop_front();
    dir_list.push_back(dir);
    std::vector<string> children;
    // GetChildren might fail if we don't have appropriate permissions.
    Status s = GetChildren(dir, &children);
    ret.Update(s);
    if (!s.ok()) {
      (*undeleted_dirs)++;
      continue;
    }
    for (const string& child : children) {
      const string child_path = this->JoinPath(dir, child);
      // If the child is a directory add it to the queue, otherwise delete it.
      if (IsDirectory(child_path).ok()) {
        dir_q.push_back(child_path);
      } else {
        // Delete file might fail because of permissions issues or might be
        // unimplemented.
        Status del_status = DeleteFile(child_path);
        ret.Update(del_status);
        if (!del_status.ok()) {
          (*undeleted_files)++;
        }
      }
    }
  }
  // Now reverse the list of directories and delete them. The BFS ensures that
  // we can delete the directories in this order.
  std::reverse(dir_list.begin(), dir_list.end());
  for (const string& dir : dir_list) {
    // Delete dir might fail because of permissions issues or might be
    // unimplemented.
    Status s = DeleteDir(dir);
    ret.Update(s);
    if (!s.ok()) {
      (*undeleted_dirs)++;
    }
  }
  return ret;
}

Status FileSystem::RecursivelyCreateDir(const string& dirname,
                                        TransactionToken* token) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_7(mht_7_v, 380, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::RecursivelyCreateDir");

  StringPiece scheme, host, remaining_dir;
  this->ParseURI(dirname, &scheme, &host, &remaining_dir);
  std::vector<StringPiece> sub_dirs;
  while (!remaining_dir.empty()) {
    std::string current_entry = this->CreateURI(scheme, host, remaining_dir);
    Status exists_status = FileExists(current_entry);
    if (exists_status.ok()) {
      // FileExists cannot differentiate between existence of a file or a
      // directory, hence we need an additional test as we must not assume that
      // a path to a file is a path to a parent directory.
      Status directory_status = IsDirectory(current_entry);
      if (directory_status.ok()) {
        break;  // We need to start creating directories from here.
      } else if (directory_status.code() == tensorflow::error::UNIMPLEMENTED) {
        return directory_status;
      } else {
        return errors::FailedPrecondition(remaining_dir, " is not a directory");
      }
    }
    if (exists_status.code() != error::Code::NOT_FOUND) {
      return exists_status;
    }
    // Basename returns "" for / ending dirs.
    if (!str_util::EndsWith(remaining_dir, "/")) {
      sub_dirs.push_back(this->Basename(remaining_dir));
    }
    remaining_dir = this->Dirname(remaining_dir);
  }

  // sub_dirs contains all the dirs to be created but in reverse order.
  std::reverse(sub_dirs.begin(), sub_dirs.end());

  // Now create the directories.
  string built_path(remaining_dir);
  for (const StringPiece sub_dir : sub_dirs) {
    built_path = this->JoinPath(built_path, sub_dir);
    Status status = CreateDir(this->CreateURI(scheme, host, built_path));
    if (!status.ok() && status.code() != tensorflow::error::ALREADY_EXISTS) {
      return status;
    }
  }
  return Status::OK();
}

Status FileSystem::CopyFile(const string& src, const string& target,
                            TransactionToken* token) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("src: \"" + src + "\"");
   mht_8_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_8(mht_8_v, 431, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::CopyFile");

  return FileSystemCopyFile(this, src, this, target);
}

char FileSystem::Separator() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_9(mht_9_v, 438, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::Separator");
 return '/'; }

string FileSystem::JoinPathImpl(std::initializer_list<StringPiece> paths) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_10(mht_10_v, 443, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::JoinPathImpl");

  string result;

  for (StringPiece path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = string(path);
      continue;
    }

    if (result[result.size() - 1] == '/') {
      if (this->IsAbsolutePath(path)) {
        strings::StrAppend(&result, path.substr(1));
      } else {
        strings::StrAppend(&result, path);
      }
    } else {
      if (this->IsAbsolutePath(path)) {
        strings::StrAppend(&result, path);
      } else {
        strings::StrAppend(&result, "/", path);
      }
    }
  }

  return result;
}

std::pair<StringPiece, StringPiece> FileSystem::SplitPath(
    StringPiece uri) const {
  StringPiece scheme, host, path;
  ParseURI(uri, &scheme, &host, &path);

  size_t pos = path.rfind(this->Separator());

  // Our code assumes it is written for linux too many times. So, for windows
  // also check for '/'
#ifdef PLATFORM_WINDOWS
  size_t pos2 = path.rfind('/');
  // Pick the max value that is not string::npos.
  if (pos == string::npos) {
    pos = pos2;
  } else {
    if (pos2 != string::npos) {
      pos = pos > pos2 ? pos : pos2;
    }
  }
#endif

  // Handle the case with no SEP in 'path'.
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

bool FileSystem::IsAbsolutePath(StringPiece path) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_11(mht_11_v, 512, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::IsAbsolutePath");

  return !path.empty() && path[0] == '/';
}

StringPiece FileSystem::Dirname(StringPiece path) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_12(mht_12_v, 519, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::Dirname");

  return this->SplitPath(path).first;
}

StringPiece FileSystem::Basename(StringPiece path) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_13(mht_13_v, 526, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::Basename");

  return this->SplitPath(path).second;
}

StringPiece FileSystem::Extension(StringPiece path) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_14(mht_14_v, 533, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::Extension");

  StringPiece basename = this->Basename(path);

  size_t pos = basename.rfind('.');
  if (pos == StringPiece::npos) {
    return StringPiece(path.data() + path.size(), 0);
  } else {
    return StringPiece(path.data() + pos + 1, path.size() - (pos + 1));
  }
}

string FileSystem::CleanPath(StringPiece unclean_path) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_15(mht_15_v, 547, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::CleanPath");

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

void FileSystem::ParseURI(StringPiece remaining, StringPiece* scheme,
                          StringPiece* host, StringPiece* path) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_16(mht_16_v, 632, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::ParseURI");

  // 0. Parse scheme
  // Make sure scheme matches [a-zA-Z][0-9a-zA-Z.]*
  // TODO(keveman): Allow "+" and "-" in the scheme.
  // Keep URI pattern in tensorboard/backend/server.py updated accordingly
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

string FileSystem::CreateURI(StringPiece scheme, StringPiece host,
                             StringPiece path) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_17(mht_17_v, 666, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::CreateURI");

  if (scheme.empty()) {
    return string(path);
  }
  return strings::StrCat(scheme, "://", host, path);
}

std::string FileSystem::DecodeTransaction(const TransactionToken* token) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTcc mht_18(mht_18_v, 676, "", "./tensorflow/core/platform/file_system.cc", "FileSystem::DecodeTransaction");

  // TODO(sami): Switch using StrCat when void* is supported
  if (token) {
    std::stringstream oss;
    oss << "Token= " << token->token << ", Owner=" << token->owner;
    return oss.str();
  }
  return "No Transaction";
}

}  // namespace tensorflow
