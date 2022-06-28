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
class MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc() {
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

#include "tensorflow/core/platform/env.h"

#include <sys/stat.h>

#include <deque>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/stringprintf.h"

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#if defined(__FreeBSD__)
#include <sys/sysctl.h>
#endif
#if defined(PLATFORM_WINDOWS)
#include <windows.h>
#undef DeleteFile
#undef CopyFile
#include "tensorflow/core/platform/windows/wide_char.h"
#define PATH_MAX MAX_PATH
#else
#include <fcntl.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace tensorflow {

// 128KB copy buffer
constexpr size_t kCopyFileBufferSize = 128 * 1024;

class FileSystemRegistryImpl : public FileSystemRegistry {
 public:
  Status Register(const std::string& scheme, Factory factory) override;
  Status Register(const std::string& scheme,
                  std::unique_ptr<FileSystem> filesystem) override;
  FileSystem* Lookup(const std::string& scheme) override;
  Status GetRegisteredFileSystemSchemes(
      std::vector<std::string>* schemes) override;

 private:
  mutable mutex mu_;
  mutable std::unordered_map<std::string, std::unique_ptr<FileSystem>> registry_
      TF_GUARDED_BY(mu_);
};

Status FileSystemRegistryImpl::Register(const std::string& scheme,
                                        FileSystemRegistry::Factory factory) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("scheme: \"" + scheme + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_0(mht_0_v, 242, "", "./tensorflow/core/platform/env.cc", "FileSystemRegistryImpl::Register");

  mutex_lock lock(mu_);
  if (!registry_.emplace(scheme, std::unique_ptr<FileSystem>(factory()))
           .second) {
    return errors::AlreadyExists("File factory for ", scheme,
                                 " already registered");
  }
  return Status::OK();
}

Status FileSystemRegistryImpl::Register(
    const std::string& scheme, std::unique_ptr<FileSystem> filesystem) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("scheme: \"" + scheme + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_1(mht_1_v, 257, "", "./tensorflow/core/platform/env.cc", "FileSystemRegistryImpl::Register");

  mutex_lock lock(mu_);
  if (!registry_.emplace(scheme, std::move(filesystem)).second) {
    return errors::AlreadyExists("File system for ", scheme,
                                 " already registered");
  }
  return Status::OK();
}

FileSystem* FileSystemRegistryImpl::Lookup(const std::string& scheme) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("scheme: \"" + scheme + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_2(mht_2_v, 270, "", "./tensorflow/core/platform/env.cc", "FileSystemRegistryImpl::Lookup");

  mutex_lock lock(mu_);
  const auto found = registry_.find(scheme);
  if (found == registry_.end()) {
    return nullptr;
  }
  return found->second.get();
}

Status FileSystemRegistryImpl::GetRegisteredFileSystemSchemes(
    std::vector<std::string>* schemes) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_3(mht_3_v, 283, "", "./tensorflow/core/platform/env.cc", "FileSystemRegistryImpl::GetRegisteredFileSystemSchemes");

  mutex_lock lock(mu_);
  for (const auto& e : registry_) {
    schemes->push_back(e.first);
  }
  return Status::OK();
}

Env::Env() : file_system_registry_(new FileSystemRegistryImpl) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_4(mht_4_v, 294, "", "./tensorflow/core/platform/env.cc", "Env::Env");
}

Status Env::GetFileSystemForFile(const std::string& fname,
                                 FileSystem** result) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_5(mht_5_v, 301, "", "./tensorflow/core/platform/env.cc", "Env::GetFileSystemForFile");

  StringPiece scheme, host, path;
  io::ParseURI(fname, &scheme, &host, &path);
  FileSystem* file_system = file_system_registry_->Lookup(std::string(scheme));
  if (!file_system) {
    if (scheme.empty()) {
      scheme = "[local]";
    }

    return errors::Unimplemented("File system scheme '", scheme,
                                 "' not implemented (file: '", fname, "')");
  }
  *result = file_system;
  return Status::OK();
}

Status Env::GetRegisteredFileSystemSchemes(std::vector<std::string>* schemes) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_6(mht_6_v, 320, "", "./tensorflow/core/platform/env.cc", "Env::GetRegisteredFileSystemSchemes");

  return file_system_registry_->GetRegisteredFileSystemSchemes(schemes);
}

Status Env::RegisterFileSystem(const std::string& scheme,
                               FileSystemRegistry::Factory factory) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("scheme: \"" + scheme + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_7(mht_7_v, 329, "", "./tensorflow/core/platform/env.cc", "Env::RegisterFileSystem");

  return file_system_registry_->Register(scheme, std::move(factory));
}

Status Env::RegisterFileSystem(const std::string& scheme,
                               std::unique_ptr<FileSystem> filesystem) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("scheme: \"" + scheme + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_8(mht_8_v, 338, "", "./tensorflow/core/platform/env.cc", "Env::RegisterFileSystem");

  return file_system_registry_->Register(scheme, std::move(filesystem));
}

Status Env::SetOption(const std::string& scheme, const std::string& key,
                      const std::string& value) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("scheme: \"" + scheme + "\"");
   mht_9_v.push_back("key: \"" + key + "\"");
   mht_9_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_9(mht_9_v, 349, "", "./tensorflow/core/platform/env.cc", "Env::SetOption");

  FileSystem* file_system = file_system_registry_->Lookup(scheme);
  if (!file_system) {
    return errors::Unimplemented("File system scheme '", scheme,
                                 "' not found to set configuration");
  }
  return file_system->SetOption(key, value);
}

Status Env::SetOption(const std::string& scheme, const std::string& key,
                      const std::vector<string>& values) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("scheme: \"" + scheme + "\"");
   mht_10_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_10(mht_10_v, 364, "", "./tensorflow/core/platform/env.cc", "Env::SetOption");

  FileSystem* file_system = file_system_registry_->Lookup(scheme);
  if (!file_system) {
    return errors::Unimplemented("File system scheme '", scheme,
                                 "' not found to set configuration");
  }
  return file_system->SetOption(key, values);
}

Status Env::SetOption(const std::string& scheme, const std::string& key,
                      const std::vector<int64_t>& values) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("scheme: \"" + scheme + "\"");
   mht_11_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_11(mht_11_v, 379, "", "./tensorflow/core/platform/env.cc", "Env::SetOption");

  FileSystem* file_system = file_system_registry_->Lookup(scheme);
  if (!file_system) {
    return errors::Unimplemented("File system scheme '", scheme,
                                 "' not found to set configuration");
  }
  return file_system->SetOption(key, values);
}

Status Env::SetOption(const std::string& scheme, const std::string& key,
                      const std::vector<double>& values) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("scheme: \"" + scheme + "\"");
   mht_12_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_12(mht_12_v, 394, "", "./tensorflow/core/platform/env.cc", "Env::SetOption");

  FileSystem* file_system = file_system_registry_->Lookup(scheme);
  if (!file_system) {
    return errors::Unimplemented("File system scheme '", scheme,
                                 "' not found to set configuration");
  }
  return file_system->SetOption(key, values);
}

Status Env::FlushFileSystemCaches() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_13(mht_13_v, 406, "", "./tensorflow/core/platform/env.cc", "Env::FlushFileSystemCaches");

  std::vector<string> schemes;
  TF_RETURN_IF_ERROR(GetRegisteredFileSystemSchemes(&schemes));
  for (const string& scheme : schemes) {
    FileSystem* fs = nullptr;
    TF_RETURN_IF_ERROR(
        GetFileSystemForFile(io::CreateURI(scheme, "", ""), &fs));
    fs->FlushCaches();
  }
  return Status::OK();
}

Status Env::NewRandomAccessFile(const string& fname,
                                std::unique_ptr<RandomAccessFile>* result) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_14(mht_14_v, 423, "", "./tensorflow/core/platform/env.cc", "Env::NewRandomAccessFile");

  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewRandomAccessFile(fname, result);
}

Status Env::NewReadOnlyMemoryRegionFromFile(
    const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_15(mht_15_v, 434, "", "./tensorflow/core/platform/env.cc", "Env::NewReadOnlyMemoryRegionFromFile");

  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewReadOnlyMemoryRegionFromFile(fname, result);
}

Status Env::NewWritableFile(const string& fname,
                            std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_16(mht_16_v, 445, "", "./tensorflow/core/platform/env.cc", "Env::NewWritableFile");

  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewWritableFile(fname, result);
}

Status Env::NewAppendableFile(const string& fname,
                              std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_17(mht_17_v, 456, "", "./tensorflow/core/platform/env.cc", "Env::NewAppendableFile");

  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewAppendableFile(fname, result);
}

Status Env::FileExists(const string& fname) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_18(mht_18_v, 466, "", "./tensorflow/core/platform/env.cc", "Env::FileExists");

  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->FileExists(fname);
}

bool Env::FilesExist(const std::vector<string>& files,
                     std::vector<Status>* status) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_19(mht_19_v, 476, "", "./tensorflow/core/platform/env.cc", "Env::FilesExist");

  std::unordered_map<string, std::vector<string>> files_per_fs;
  for (const auto& file : files) {
    StringPiece scheme, host, path;
    io::ParseURI(file, &scheme, &host, &path);
    files_per_fs[string(scheme)].push_back(file);
  }

  std::unordered_map<string, Status> per_file_status;
  bool result = true;
  for (auto itr : files_per_fs) {
    FileSystem* file_system = file_system_registry_->Lookup(itr.first);
    bool fs_result;
    std::vector<Status> local_status;
    std::vector<Status>* fs_status = status ? &local_status : nullptr;
    if (!file_system) {
      fs_result = false;
      if (fs_status) {
        Status s = errors::Unimplemented("File system scheme '", itr.first,
                                         "' not implemented");
        local_status.resize(itr.second.size(), s);
      }
    } else {
      fs_result = file_system->FilesExist(itr.second, fs_status);
    }
    if (fs_status) {
      result &= fs_result;
      for (size_t i = 0; i < itr.second.size(); ++i) {
        per_file_status[itr.second[i]] = fs_status->at(i);
      }
    } else if (!fs_result) {
      // Return early
      return false;
    }
  }

  if (status) {
    for (const auto& file : files) {
      status->push_back(per_file_status[file]);
    }
  }

  return result;
}

Status Env::GetChildren(const string& dir, std::vector<string>* result) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_20(mht_20_v, 525, "", "./tensorflow/core/platform/env.cc", "Env::GetChildren");

  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dir, &fs));
  return fs->GetChildren(dir, result);
}

Status Env::GetMatchingPaths(const string& pattern,
                             std::vector<string>* results) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_21(mht_21_v, 536, "", "./tensorflow/core/platform/env.cc", "Env::GetMatchingPaths");

  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(pattern, &fs));
  return fs->GetMatchingPaths(pattern, results);
}

Status Env::DeleteFile(const string& fname) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_22(mht_22_v, 546, "", "./tensorflow/core/platform/env.cc", "Env::DeleteFile");

  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->DeleteFile(fname);
}

Status Env::RecursivelyCreateDir(const string& dirname) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_23(mht_23_v, 556, "", "./tensorflow/core/platform/env.cc", "Env::RecursivelyCreateDir");

  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->RecursivelyCreateDir(dirname);
}

Status Env::CreateDir(const string& dirname) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_24(mht_24_v, 566, "", "./tensorflow/core/platform/env.cc", "Env::CreateDir");

  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->CreateDir(dirname);
}

Status Env::DeleteDir(const string& dirname) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_25(mht_25_v, 576, "", "./tensorflow/core/platform/env.cc", "Env::DeleteDir");

  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->DeleteDir(dirname);
}

Status Env::Stat(const string& fname, FileStatistics* stat) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_26(mht_26_v, 586, "", "./tensorflow/core/platform/env.cc", "Env::Stat");

  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->Stat(fname, stat);
}

Status Env::IsDirectory(const string& fname) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_27(mht_27_v, 596, "", "./tensorflow/core/platform/env.cc", "Env::IsDirectory");

  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->IsDirectory(fname);
}

Status Env::HasAtomicMove(const string& path, bool* has_atomic_move) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_28(mht_28_v, 606, "", "./tensorflow/core/platform/env.cc", "Env::HasAtomicMove");

  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(path, &fs));
  return fs->HasAtomicMove(path, has_atomic_move);
}

Status Env::DeleteRecursively(const string& dirname, int64_t* undeleted_files,
                              int64_t* undeleted_dirs) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_29(mht_29_v, 617, "", "./tensorflow/core/platform/env.cc", "Env::DeleteRecursively");

  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->DeleteRecursively(dirname, undeleted_files, undeleted_dirs);
}

Status Env::GetFileSize(const string& fname, uint64* file_size) {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_30(mht_30_v, 627, "", "./tensorflow/core/platform/env.cc", "Env::GetFileSize");

  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->GetFileSize(fname, file_size);
}

Status Env::RenameFile(const string& src, const string& target) {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("src: \"" + src + "\"");
   mht_31_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_31(mht_31_v, 638, "", "./tensorflow/core/platform/env.cc", "Env::RenameFile");

  FileSystem* src_fs;
  FileSystem* target_fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(src, &src_fs));
  TF_RETURN_IF_ERROR(GetFileSystemForFile(target, &target_fs));
  if (src_fs != target_fs) {
    return errors::Unimplemented("Renaming ", src, " to ", target,
                                 " not implemented");
  }
  return src_fs->RenameFile(src, target);
}

Status Env::CopyFile(const string& src, const string& target) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("src: \"" + src + "\"");
   mht_32_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_32(mht_32_v, 655, "", "./tensorflow/core/platform/env.cc", "Env::CopyFile");

  FileSystem* src_fs;
  FileSystem* target_fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(src, &src_fs));
  TF_RETURN_IF_ERROR(GetFileSystemForFile(target, &target_fs));
  if (src_fs == target_fs) {
    return src_fs->CopyFile(src, target);
  }
  return FileSystemCopyFile(src_fs, src, target_fs, target);
}

string Env::GetExecutablePath() {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_33(mht_33_v, 669, "", "./tensorflow/core/platform/env.cc", "Env::GetExecutablePath");

  char exe_path[PATH_MAX] = {0};
#ifdef __APPLE__
  uint32_t buffer_size(0U);
  _NSGetExecutablePath(nullptr, &buffer_size);
  std::vector<char> unresolved_path(buffer_size);
  _NSGetExecutablePath(unresolved_path.data(), &buffer_size);
  CHECK(realpath(unresolved_path.data(), exe_path));
#elif defined(__FreeBSD__)
  int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1};
  size_t exe_path_size = PATH_MAX;

  if (sysctl(mib, 4, exe_path, &exe_path_size, NULL, 0) != 0) {
    // Resolution of path failed
    return "";
  }
#elif defined(PLATFORM_WINDOWS)
  HMODULE hModule = GetModuleHandleW(NULL);
  WCHAR wc_file_path[MAX_PATH] = {0};
  GetModuleFileNameW(hModule, wc_file_path, MAX_PATH);
  string file_path = WideCharToUtf8(wc_file_path);
  std::copy(file_path.begin(), file_path.end(), exe_path);
#else
  char buf[PATH_MAX] = {0};
  int path_length = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  CHECK_NE(-1, path_length);

  if (strstr(buf, "python") != nullptr) {
    // Discard the path of the python binary, and any flags.
    int fd = open("/proc/self/cmdline", O_RDONLY);
    int cmd_length = read(fd, buf, PATH_MAX - 1);
    CHECK_NE(-1, cmd_length);
    int token_pos = 0;
    for (bool token_is_first_or_flag = true; token_is_first_or_flag;) {
      // Get token length, including null
      int token_len = strlen(&buf[token_pos]) + 1;
      token_is_first_or_flag = false;
      // Check if we can skip without overshooting
      if (token_pos + token_len < cmd_length) {
        token_pos += token_len;
        token_is_first_or_flag = (buf[token_pos] == '-');  // token is a flag
      }
    }
    snprintf(exe_path, sizeof(exe_path), "%s", &buf[token_pos]);
  } else {
    snprintf(exe_path, sizeof(exe_path), "%s", buf);
  }

#endif
  // Make sure it's null-terminated:
  exe_path[sizeof(exe_path) - 1] = 0;

  return exe_path;
}

bool Env::LocalTempFilename(string* filename) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_34(mht_34_v, 727, "", "./tensorflow/core/platform/env.cc", "Env::LocalTempFilename");

  std::vector<string> dirs;
  GetLocalTempDirectories(&dirs);

  // Try each directory, as they might be full, have inappropriate
  // permissions or have different problems at times.
  for (const string& dir : dirs) {
    *filename = io::JoinPath(dir, "tempfile-");
    if (CreateUniqueFileName(filename, "")) {
      return true;
    }
  }
  return false;
}

bool Env::CreateUniqueFileName(string* prefix, const string& suffix) {
   std::vector<std::string> mht_35_v;
   mht_35_v.push_back("suffix: \"" + suffix + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_35(mht_35_v, 746, "", "./tensorflow/core/platform/env.cc", "Env::CreateUniqueFileName");

  int32_t tid = GetCurrentThreadId();
  int32_t pid = GetProcessId();
  long long now_microsec = NowMicros();  // NOLINT

  *prefix += strings::Printf("%s-%x-%d-%llx", port::Hostname().c_str(), tid,
                             pid, now_microsec);

  if (!suffix.empty()) {
    *prefix += suffix;
  }
  if (FileExists(*prefix).ok()) {
    prefix->clear();
    return false;
  } else {
    return true;
  }
}

int32 Env::GetProcessId() {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_36(mht_36_v, 768, "", "./tensorflow/core/platform/env.cc", "Env::GetProcessId");

#ifdef PLATFORM_WINDOWS
  return static_cast<int32>(GetCurrentProcessId());
#else
  return static_cast<int32>(getpid());
#endif
}

Thread::~Thread() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_37(mht_37_v, 779, "", "./tensorflow/core/platform/env.cc", "Thread::~Thread");
}

EnvWrapper::~EnvWrapper() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_38(mht_38_v, 784, "", "./tensorflow/core/platform/env.cc", "EnvWrapper::~EnvWrapper");
}

Status ReadFileToString(Env* env, const string& fname, string* data) {
   std::vector<std::string> mht_39_v;
   mht_39_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_39(mht_39_v, 790, "", "./tensorflow/core/platform/env.cc", "ReadFileToString");

  uint64 file_size;
  Status s = env->GetFileSize(fname, &file_size);
  if (!s.ok()) {
    return s;
  }
  std::unique_ptr<RandomAccessFile> file;
  s = env->NewRandomAccessFile(fname, &file);
  if (!s.ok()) {
    return s;
  }
  data->resize(file_size);
  char* p = &*data->begin();
  StringPiece result;
  s = file->Read(0, file_size, &result, p);
  if (!s.ok()) {
    data->clear();
  } else if (result.size() != file_size) {
    s = errors::Aborted("File ", fname, " changed while reading: ", file_size,
                        " vs. ", result.size());
    data->clear();
  } else if (result.data() == p) {
    // Data is already in the correct location
  } else {
    memmove(p, result.data(), result.size());
  }
  return s;
}

Status WriteStringToFile(Env* env, const string& fname,
                         const StringPiece& data) {
   std::vector<std::string> mht_40_v;
   mht_40_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_40(mht_40_v, 824, "", "./tensorflow/core/platform/env.cc", "WriteStringToFile");

  std::unique_ptr<WritableFile> file;
  Status s = env->NewWritableFile(fname, &file);
  if (!s.ok()) {
    return s;
  }
  s = file->Append(data);
  if (s.ok()) {
    s = file->Close();
  }
  return s;
}

Status FileSystemCopyFile(FileSystem* src_fs, const string& src,
                          FileSystem* target_fs, const string& target) {
   std::vector<std::string> mht_41_v;
   mht_41_v.push_back("src: \"" + src + "\"");
   mht_41_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_41(mht_41_v, 843, "", "./tensorflow/core/platform/env.cc", "FileSystemCopyFile");

  std::unique_ptr<RandomAccessFile> src_file;
  TF_RETURN_IF_ERROR(src_fs->NewRandomAccessFile(src, &src_file));

  // When `target` points to a directory, we need to create a file within.
  string target_name;
  if (target_fs->IsDirectory(target).ok()) {
    target_name = io::JoinPath(target, io::Basename(src));
  } else {
    target_name = target;
  }

  std::unique_ptr<WritableFile> target_file;
  TF_RETURN_IF_ERROR(target_fs->NewWritableFile(target_name, &target_file));

  uint64 offset = 0;
  std::unique_ptr<char[]> scratch(new char[kCopyFileBufferSize]);
  Status s = Status::OK();
  while (s.ok()) {
    StringPiece result;
    s = src_file->Read(offset, kCopyFileBufferSize, &result, scratch.get());
    if (!(s.ok() || s.code() == error::OUT_OF_RANGE)) {
      return s;
    }
    TF_RETURN_IF_ERROR(target_file->Append(result));
    offset += result.size();
  }
  return target_file->Close();
}

// A ZeroCopyInputStream on a RandomAccessFile.
namespace {
class FileStream : public protobuf::io::ZeroCopyInputStream {
 public:
  explicit FileStream(RandomAccessFile* file) : file_(file), pos_(0) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_42(mht_42_v, 880, "", "./tensorflow/core/platform/env.cc", "FileStream");
}

  void BackUp(int count) override {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_43(mht_43_v, 885, "", "./tensorflow/core/platform/env.cc", "BackUp");
 pos_ -= count; }
  bool Skip(int count) override {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_44(mht_44_v, 889, "", "./tensorflow/core/platform/env.cc", "Skip");

    pos_ += count;
    return true;
  }
  int64_t ByteCount() const override {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_45(mht_45_v, 896, "", "./tensorflow/core/platform/env.cc", "ByteCount");
 return pos_; }
  Status status() const {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_46(mht_46_v, 900, "", "./tensorflow/core/platform/env.cc", "status");
 return status_; }

  bool Next(const void** data, int* size) override {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_47(mht_47_v, 905, "", "./tensorflow/core/platform/env.cc", "Next");

    StringPiece result;
    Status s = file_->Read(pos_, kBufSize, &result, scratch_);
    if (result.empty()) {
      status_ = s;
      return false;
    }
    pos_ += result.size();
    *data = result.data();
    *size = result.size();
    return true;
  }

 private:
  static constexpr int kBufSize = 512 << 10;

  RandomAccessFile* file_;
  int64_t pos_;
  Status status_;
  char scratch_[kBufSize];
};

}  // namespace

Status WriteBinaryProto(Env* env, const string& fname,
                        const protobuf::MessageLite& proto) {
   std::vector<std::string> mht_48_v;
   mht_48_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_48(mht_48_v, 934, "", "./tensorflow/core/platform/env.cc", "WriteBinaryProto");

  string serialized;
  proto.AppendToString(&serialized);
  return WriteStringToFile(env, fname, serialized);
}

Status ReadBinaryProto(Env* env, const string& fname,
                       protobuf::MessageLite* proto) {
   std::vector<std::string> mht_49_v;
   mht_49_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_49(mht_49_v, 945, "", "./tensorflow/core/platform/env.cc", "ReadBinaryProto");

  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &file));
  std::unique_ptr<FileStream> stream(new FileStream(file.get()));
  protobuf::io::CodedInputStream coded_stream(stream.get());

  if (!proto->ParseFromCodedStream(&coded_stream) ||
      !coded_stream.ConsumedEntireMessage()) {
    TF_RETURN_IF_ERROR(stream->status());
    return errors::DataLoss("Can't parse ", fname, " as binary proto");
  }
  return Status::OK();
}

Status WriteTextProto(Env* env, const string& fname,
                      const protobuf::Message& proto) {
   std::vector<std::string> mht_50_v;
   mht_50_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_50(mht_50_v, 964, "", "./tensorflow/core/platform/env.cc", "WriteTextProto");

  string serialized;
  if (!protobuf::TextFormat::PrintToString(proto, &serialized)) {
    return errors::FailedPrecondition("Unable to convert proto to text.");
  }
  return WriteStringToFile(env, fname, serialized);
}

Status ReadTextProto(Env* env, const string& fname, protobuf::Message* proto) {
   std::vector<std::string> mht_51_v;
   mht_51_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_51(mht_51_v, 976, "", "./tensorflow/core/platform/env.cc", "ReadTextProto");

  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &file));
  std::unique_ptr<FileStream> stream(new FileStream(file.get()));

  if (!protobuf::TextFormat::Parse(stream.get(), proto)) {
    TF_RETURN_IF_ERROR(stream->status());
    return errors::DataLoss("Can't parse ", fname, " as text proto");
  }
  return Status::OK();
}

Status ReadTextOrBinaryProto(Env* env, const string& fname,
                             protobuf::Message* proto) {
   std::vector<std::string> mht_52_v;
   mht_52_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_52(mht_52_v, 993, "", "./tensorflow/core/platform/env.cc", "ReadTextOrBinaryProto");

  if (ReadTextProto(env, fname, proto).ok()) {
    return Status::OK();
  }
  return ReadBinaryProto(env, fname, proto);
}

Status ReadTextOrBinaryProto(Env* env, const string& fname,
                             protobuf::MessageLite* proto) {
   std::vector<std::string> mht_53_v;
   mht_53_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTcc mht_53(mht_53_v, 1005, "", "./tensorflow/core/platform/env.cc", "ReadTextOrBinaryProto");

  return ReadBinaryProto(env, fname, proto);
}

}  // namespace tensorflow
