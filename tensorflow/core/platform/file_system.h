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

#ifndef TENSORFLOW_CORE_PLATFORM_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_PLATFORM_FILE_SYSTEM_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh() {
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


#include <stdint.h>

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/cord.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_statistics.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

#ifdef PLATFORM_WINDOWS
#undef DeleteFile
#undef CopyFile
#undef TranslateName
#endif

namespace tensorflow {

class RandomAccessFile;
class ReadOnlyMemoryRegion;
class WritableFile;

class FileSystem;
struct TransactionToken {
  FileSystem* owner;
  void* token;
};

/// A generic interface for accessing a file system.  Implementations
/// of custom filesystem adapters must implement this interface,
/// RandomAccessFile, WritableFile, and ReadOnlyMemoryRegion classes.
class FileSystem {
 public:
  /// \brief Creates a brand new random access read-only file with the
  /// specified name.
  ///
  /// On success, stores a pointer to the new file in
  /// *result and returns OK.  On failure stores NULL in *result and
  /// returns non-OK.  If the file does not exist, returns a non-OK
  /// status.
  ///
  /// The returned file may be concurrently accessed by multiple threads.
  ///
  /// The ownership of the returned RandomAccessFile is passed to the caller
  /// and the object should be deleted when is not used.
  virtual tensorflow::Status NewRandomAccessFile(
      const std::string& fname, std::unique_ptr<RandomAccessFile>* result) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_0(mht_0_v, 241, "", "./tensorflow/core/platform/file_system.h", "NewRandomAccessFile");

    return NewRandomAccessFile(fname, nullptr, result);
  }

  virtual tensorflow::Status NewRandomAccessFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_1(mht_1_v, 251, "", "./tensorflow/core/platform/file_system.h", "NewRandomAccessFile");

    // We duplicate these methods due to Google internal coding style prevents
    // virtual functions with default arguments. See PR #41615.
    return Status::OK();
  }

  /// \brief Creates an object that writes to a new file with the specified
  /// name.
  ///
  /// Deletes any existing file with the same name and creates a
  /// new file.  On success, stores a pointer to the new file in
  /// *result and returns OK.  On failure stores NULL in *result and
  /// returns non-OK.
  ///
  /// The returned file will only be accessed by one thread at a time.
  ///
  /// The ownership of the returned WritableFile is passed to the caller
  /// and the object should be deleted when is not used.
  virtual tensorflow::Status NewWritableFile(
      const std::string& fname, std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_2(mht_2_v, 274, "", "./tensorflow/core/platform/file_system.h", "NewWritableFile");

    return NewWritableFile(fname, nullptr, result);
  }

  virtual tensorflow::Status NewWritableFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_3(mht_3_v, 284, "", "./tensorflow/core/platform/file_system.h", "NewWritableFile");

    return Status::OK();
  }

  /// \brief Creates an object that either appends to an existing file, or
  /// writes to a new file (if the file does not exist to begin with).
  ///
  /// On success, stores a pointer to the new file in *result and
  /// returns OK.  On failure stores NULL in *result and returns
  /// non-OK.
  ///
  /// The returned file will only be accessed by one thread at a time.
  ///
  /// The ownership of the returned WritableFile is passed to the caller
  /// and the object should be deleted when is not used.
  virtual tensorflow::Status NewAppendableFile(
      const std::string& fname, std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_4(mht_4_v, 304, "", "./tensorflow/core/platform/file_system.h", "NewAppendableFile");

    return NewAppendableFile(fname, nullptr, result);
  }

  virtual tensorflow::Status NewAppendableFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_5(mht_5_v, 314, "", "./tensorflow/core/platform/file_system.h", "NewAppendableFile");

    return Status::OK();
  }

  /// \brief Creates a readonly region of memory with the file context.
  ///
  /// On success, it returns a pointer to read-only memory region
  /// from the content of file fname. The ownership of the region is passed to
  /// the caller. On failure stores nullptr in *result and returns non-OK.
  ///
  /// The returned memory region can be accessed from many threads in parallel.
  ///
  /// The ownership of the returned ReadOnlyMemoryRegion is passed to the caller
  /// and the object should be deleted when is not used.
  virtual tensorflow::Status NewReadOnlyMemoryRegionFromFile(
      const std::string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_6(mht_6_v, 333, "", "./tensorflow/core/platform/file_system.h", "NewReadOnlyMemoryRegionFromFile");

    return NewReadOnlyMemoryRegionFromFile(fname, nullptr, result);
  }

  virtual tensorflow::Status NewReadOnlyMemoryRegionFromFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_7(mht_7_v, 343, "", "./tensorflow/core/platform/file_system.h", "NewReadOnlyMemoryRegionFromFile");

    return Status::OK();
  }

  /// Returns OK if the named path exists and NOT_FOUND otherwise.
  virtual tensorflow::Status FileExists(const std::string& fname) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_8(mht_8_v, 352, "", "./tensorflow/core/platform/file_system.h", "FileExists");

    return FileExists(fname, nullptr);
  }

  virtual tensorflow::Status FileExists(const std::string& fname,
                                        TransactionToken* token) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_9(mht_9_v, 361, "", "./tensorflow/core/platform/file_system.h", "FileExists");

    return Status::OK();
  }

  /// Returns true if all the listed files exist, false otherwise.
  /// if status is not null, populate the vector with a detailed status
  /// for each file.
  virtual bool FilesExist(const std::vector<string>& files,
                          std::vector<Status>* status) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_10(mht_10_v, 372, "", "./tensorflow/core/platform/file_system.h", "FilesExist");

    return FilesExist(files, nullptr, status);
  }

  virtual bool FilesExist(const std::vector<string>& files,
                          TransactionToken* token, std::vector<Status>* status);

  /// \brief Returns the immediate children in the given directory.
  ///
  /// The returned paths are relative to 'dir'.
  virtual tensorflow::Status GetChildren(const std::string& dir,
                                         std::vector<string>* result) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_11(mht_11_v, 387, "", "./tensorflow/core/platform/file_system.h", "GetChildren");

    return GetChildren(dir, nullptr, result);
  }

  virtual tensorflow::Status GetChildren(const std::string& dir,
                                         TransactionToken* token,
                                         std::vector<string>* result) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_12(mht_12_v, 397, "", "./tensorflow/core/platform/file_system.h", "GetChildren");

    return Status::OK();
  }

  /// \brief Given a pattern, stores in *results the set of paths that matches
  /// that pattern. *results is cleared.
  ///
  /// pattern must match all of a name, not just a substring.
  ///
  /// pattern: { term }
  /// term:
  ///   '*': matches any sequence of non-'/' characters
  ///   '?': matches a single non-'/' character
  ///   '[' [ '^' ] { match-list } ']':
  ///        matches any single character (not) on the list
  ///   c: matches character c (c != '*', '?', '\\', '[')
  ///   '\\' c: matches character c
  /// character-range:
  ///   c: matches character c (c != '\\', '-', ']')
  ///   '\\' c: matches character c
  ///   lo '-' hi: matches character c for lo <= c <= hi
  ///
  /// Typical return codes:
  ///  * OK - no errors
  ///  * UNIMPLEMENTED - Some underlying functions (like GetChildren) are not
  ///                    implemented
  virtual tensorflow::Status GetMatchingPaths(const std::string& pattern,
                                              std::vector<string>* results) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_13(mht_13_v, 428, "", "./tensorflow/core/platform/file_system.h", "GetMatchingPaths");

    return GetMatchingPaths(pattern, nullptr, results);
  }

  virtual tensorflow::Status GetMatchingPaths(const std::string& pattern,
                                              TransactionToken* token,
                                              std::vector<string>* results) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_14(mht_14_v, 438, "", "./tensorflow/core/platform/file_system.h", "GetMatchingPaths");

    return Status::OK();
  }

  /// \brief Checks if the given filename matches the pattern.
  ///
  /// This function provides the equivalent of posix fnmatch, however it is
  /// implemented without fnmatch to ensure that this can be used for cloud
  /// filesystems on windows. For windows filesystems, it uses PathMatchSpec.
  virtual bool Match(const std::string& filename, const std::string& pattern);

  /// \brief Obtains statistics for the given path.
  virtual tensorflow::Status Stat(const std::string& fname,
                                  FileStatistics* stat) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_15(mht_15_v, 455, "", "./tensorflow/core/platform/file_system.h", "Stat");

    return Stat(fname, nullptr, stat);
  }

  virtual tensorflow::Status Stat(const std::string& fname,
                                  TransactionToken* token,
                                  FileStatistics* stat) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_16(mht_16_v, 465, "", "./tensorflow/core/platform/file_system.h", "Stat");

    return Status::OK();
  }

  /// \brief Deletes the named file.
  virtual tensorflow::Status DeleteFile(const std::string& fname) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_17(mht_17_v, 474, "", "./tensorflow/core/platform/file_system.h", "DeleteFile");

    return DeleteFile(fname, nullptr);
  }

  virtual tensorflow::Status DeleteFile(const std::string& fname,
                                        TransactionToken* token) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_18(mht_18_v, 483, "", "./tensorflow/core/platform/file_system.h", "DeleteFile");

    return Status::OK();
  }

  /// \brief Creates the specified directory.
  /// Typical return codes:
  ///  * OK - successfully created the directory.
  ///  * ALREADY_EXISTS - directory with name dirname already exists.
  ///  * PERMISSION_DENIED - dirname is not writable.
  virtual tensorflow::Status CreateDir(const std::string& dirname) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_19(mht_19_v, 496, "", "./tensorflow/core/platform/file_system.h", "CreateDir");

    return CreateDir(dirname, nullptr);
  }

  virtual tensorflow::Status CreateDir(const std::string& dirname,
                                       TransactionToken* token) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_20(mht_20_v, 505, "", "./tensorflow/core/platform/file_system.h", "CreateDir");

    return Status::OK();
  }

  /// \brief Creates the specified directory and all the necessary
  /// subdirectories.
  /// Typical return codes:
  ///  * OK - successfully created the directory and sub directories, even if
  ///         they were already created.
  ///  * PERMISSION_DENIED - dirname or some subdirectory is not writable.
  virtual tensorflow::Status RecursivelyCreateDir(const std::string& dirname) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_21(mht_21_v, 519, "", "./tensorflow/core/platform/file_system.h", "RecursivelyCreateDir");

    return RecursivelyCreateDir(dirname, nullptr);
  }

  virtual tensorflow::Status RecursivelyCreateDir(const std::string& dirname,
                                                  TransactionToken* token);

  /// \brief Deletes the specified directory.
  virtual tensorflow::Status DeleteDir(const std::string& dirname) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_22(mht_22_v, 531, "", "./tensorflow/core/platform/file_system.h", "DeleteDir");

    return DeleteDir(dirname, nullptr);
  }

  virtual tensorflow::Status DeleteDir(const std::string& dirname,
                                       TransactionToken* token) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_23(mht_23_v, 540, "", "./tensorflow/core/platform/file_system.h", "DeleteDir");

    return Status::OK();
  }

  /// \brief Deletes the specified directory and all subdirectories and files
  /// underneath it. This is accomplished by traversing the directory tree
  /// rooted at dirname and deleting entries as they are encountered.
  ///
  /// If dirname itself is not readable or does not exist, *undeleted_dir_count
  /// is set to 1, *undeleted_file_count is set to 0 and an appropriate status
  /// (e.g. NOT_FOUND) is returned.
  ///
  /// If dirname and all its descendants were successfully deleted, TF_OK is
  /// returned and both error counters are set to zero.
  ///
  /// Otherwise, while traversing the tree, undeleted_file_count and
  /// undeleted_dir_count are updated if an entry of the corresponding type
  /// could not be deleted. The returned error status represents the reason that
  /// any one of these entries could not be deleted.
  ///
  /// REQUIRES: undeleted_files, undeleted_dirs to be not null.
  ///
  /// Typical return codes:
  ///  * OK - dirname exists and we were able to delete everything underneath.
  ///  * NOT_FOUND - dirname doesn't exist
  ///  * PERMISSION_DENIED - dirname or some descendant is not writable
  ///  * UNIMPLEMENTED - Some underlying functions (like Delete) are not
  ///                    implemented
  virtual tensorflow::Status DeleteRecursively(const std::string& dirname,
                                               int64_t* undeleted_files,
                                               int64_t* undeleted_dirs) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_24(mht_24_v, 574, "", "./tensorflow/core/platform/file_system.h", "DeleteRecursively");

    return DeleteRecursively(dirname, nullptr, undeleted_files, undeleted_dirs);
  }

  virtual tensorflow::Status DeleteRecursively(const std::string& dirname,
                                               TransactionToken* token,
                                               int64_t* undeleted_files,
                                               int64_t* undeleted_dirs);

  /// \brief Stores the size of `fname` in `*file_size`.
  virtual tensorflow::Status GetFileSize(const std::string& fname,
                                         uint64* file_size) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_25(mht_25_v, 589, "", "./tensorflow/core/platform/file_system.h", "GetFileSize");

    return GetFileSize(fname, nullptr, file_size);
  }

  virtual tensorflow::Status GetFileSize(const std::string& fname,
                                         TransactionToken* token,
                                         uint64* file_size) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_26(mht_26_v, 599, "", "./tensorflow/core/platform/file_system.h", "GetFileSize");

    return Status::OK();
  }

  /// \brief Overwrites the target if it exists.
  virtual tensorflow::Status RenameFile(const std::string& src,
                                        const std::string& target) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("src: \"" + src + "\"");
   mht_27_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_27(mht_27_v, 610, "", "./tensorflow/core/platform/file_system.h", "RenameFile");

    return RenameFile(src, target, nullptr);
  }

  virtual tensorflow::Status RenameFile(const std::string& src,
                                        const std::string& target,
                                        TransactionToken* token) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("src: \"" + src + "\"");
   mht_28_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_28(mht_28_v, 621, "", "./tensorflow/core/platform/file_system.h", "RenameFile");

    return Status::OK();
  }

  /// \brief Copy the src to target.
  virtual tensorflow::Status CopyFile(const std::string& src,
                                      const std::string& target) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("src: \"" + src + "\"");
   mht_29_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_29(mht_29_v, 632, "", "./tensorflow/core/platform/file_system.h", "CopyFile");

    return CopyFile(src, target, nullptr);
  }

  virtual tensorflow::Status CopyFile(const std::string& src,
                                      const std::string& target,
                                      TransactionToken* token);

  /// \brief Translate an URI to a filename for the FileSystem implementation.
  ///
  /// The implementation in this class cleans up the path, removing
  /// duplicate /'s, resolving .. and removing trailing '/'.
  /// This respects relative vs. absolute paths, but does not
  /// invoke any system calls (getcwd(2)) in order to resolve relative
  /// paths with respect to the actual working directory.  That is, this is
  /// purely string manipulation, completely independent of process state.
  virtual std::string TranslateName(const std::string& name) const;

  /// \brief Returns whether the given path is a directory or not.
  ///
  /// Typical return codes (not guaranteed exhaustive):
  ///  * OK - The path exists and is a directory.
  ///  * FAILED_PRECONDITION - The path exists and is not a directory.
  ///  * NOT_FOUND - The path entry does not exist.
  ///  * PERMISSION_DENIED - Insufficient permissions.
  ///  * UNIMPLEMENTED - The file factory doesn't support directories.
  virtual tensorflow::Status IsDirectory(const std::string& fname) {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_30(mht_30_v, 662, "", "./tensorflow/core/platform/file_system.h", "IsDirectory");

    return IsDirectory(fname, nullptr);
  }

  virtual tensorflow::Status IsDirectory(const std::string& fname,
                                         TransactionToken* token);

  /// \brief Returns whether the given path is on a file system
  /// that has atomic move capabilities. This can be used
  /// to determine if there needs to be a temp location to safely write objects.
  /// The second boolean argument has_atomic_move contains this information.
  ///
  /// Returns one of the following status codes (not guaranteed exhaustive):
  ///  * OK - The path is on a recognized file system,
  ///         so has_atomic_move holds the above information.
  ///  * UNIMPLEMENTED - The file system of the path hasn't been implemented in
  ///  TF
  virtual Status HasAtomicMove(const std::string& path, bool* has_atomic_move);

  /// \brief Flushes any cached filesystem objects from memory.
  virtual void FlushCaches() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_31(mht_31_v, 685, "", "./tensorflow/core/platform/file_system.h", "FlushCaches");
 FlushCaches(nullptr); }

  virtual void FlushCaches(TransactionToken* token);

  /// \brief The separator this filesystem uses.
  ///
  /// This is implemented as a part of the filesystem, because even on windows,
  /// a user may need access to filesystems with '/' separators, such as cloud
  /// filesystems.
  virtual char Separator() const;

  /// \brief Split a path to its basename and dirname.
  ///
  /// Helper function for Basename and Dirname.
  std::pair<StringPiece, StringPiece> SplitPath(StringPiece uri) const;

  /// \brief returns the final file name in the given path.
  ///
  /// Returns the part of the path after the final "/".  If there is no
  /// "/" in the path, the result is the same as the input.
  virtual StringPiece Basename(StringPiece path) const;

  /// \brief Returns the part of the path before the final "/".
  ///
  /// If there is a single leading "/" in the path, the result will be the
  /// leading "/".  If there is no "/" in the path, the result is the empty
  /// prefix of the input.
  StringPiece Dirname(StringPiece path) const;

  /// \brief Returns the part of the basename of path after the final ".".
  ///
  /// If there is no "." in the basename, the result is empty.
  StringPiece Extension(StringPiece path) const;

  /// \brief Clean duplicate and trailing, "/"s, and resolve ".." and ".".
  ///
  /// NOTE: This respects relative vs. absolute paths, but does not
  /// invoke any system calls (getcwd(2)) in order to resolve relative
  /// paths with respect to the actual working directory.  That is, this is
  /// purely string manipulation, completely independent of process state.
  std::string CleanPath(StringPiece path) const;

  /// \brief Creates a URI from a scheme, host, and path.
  ///
  /// If the scheme is empty, we just return the path.
  std::string CreateURI(StringPiece scheme, StringPiece host,
                        StringPiece path) const;

  ///  \brief Creates a temporary file name with an extension.
  std::string GetTempFilename(const std::string& extension) const;

  /// \brief Return true if path is absolute.
  bool IsAbsolutePath(tensorflow::StringPiece path) const;

#ifndef SWIG  // variadic templates
  /// \brief Join multiple paths together.
  ///
  /// This function also removes the unnecessary path separators.
  /// For example:
  ///
  ///  Arguments                  | JoinPath
  ///  ---------------------------+----------
  ///  '/foo', 'bar'              | /foo/bar
  ///  '/foo/', 'bar'             | /foo/bar
  ///  '/foo', '/bar'             | /foo/bar
  ///
  /// Usage:
  /// string path = io::JoinPath("/mydir", filename);
  /// string path = io::JoinPath(FLAGS_test_srcdir, filename);
  /// string path = io::JoinPath("/full", "path", "to", "filename");
  template <typename... T>
  std::string JoinPath(const T&... args) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_32(mht_32_v, 759, "", "./tensorflow/core/platform/file_system.h", "JoinPath");

    return JoinPathImpl({args...});
  }
#endif /* SWIG */

  std::string JoinPathImpl(
      std::initializer_list<tensorflow::StringPiece> paths);

  /// \brief Populates the scheme, host, and path from a URI.
  ///
  /// scheme, host, and path are guaranteed by this function to point into the
  /// contents of uri, even if empty.
  ///
  /// Corner cases:
  /// - If the URI is invalid, scheme and host are set to empty strings and the
  ///  passed string is assumed to be a path
  /// - If the URI omits the path (e.g. file://host), then the path is left
  /// empty.
  void ParseURI(StringPiece remaining, StringPiece* scheme, StringPiece* host,
                StringPiece* path) const;

  // Transaction related API

  /// \brief Starts a new transaction
  virtual tensorflow::Status StartTransaction(TransactionToken** token) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_33(mht_33_v, 786, "", "./tensorflow/core/platform/file_system.h", "StartTransaction");

    *token = nullptr;
    return Status::OK();
  }

  /// \brief Adds `path` to transaction in `token`
  virtual tensorflow::Status AddToTransaction(const std::string& path,
                                              TransactionToken* token) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_34(mht_34_v, 797, "", "./tensorflow/core/platform/file_system.h", "AddToTransaction");

    return Status::OK();
  }

  /// \brief Ends transaction
  virtual tensorflow::Status EndTransaction(TransactionToken* token) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_35(mht_35_v, 805, "", "./tensorflow/core/platform/file_system.h", "EndTransaction");

    return Status::OK();
  }

  /// \brief Get token for `path` or start a new transaction and add `path` to
  /// it.
  virtual tensorflow::Status GetTokenOrStartTransaction(
      const std::string& path, TransactionToken** token) {
   std::vector<std::string> mht_36_v;
   mht_36_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_36(mht_36_v, 816, "", "./tensorflow/core/platform/file_system.h", "GetTokenOrStartTransaction");

    *token = nullptr;
    return Status::OK();
  }

  /// \brief Return transaction for `path` or nullptr in `token`
  virtual tensorflow::Status GetTransactionForPath(const std::string& path,
                                                   TransactionToken** token) {
   std::vector<std::string> mht_37_v;
   mht_37_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_37(mht_37_v, 827, "", "./tensorflow/core/platform/file_system.h", "GetTransactionForPath");

    *token = nullptr;
    return Status::OK();
  }

  /// \brief Decode transaction to human readable string.
  virtual std::string DecodeTransaction(const TransactionToken* token);

  /// \brief Set File System Configuration Options
  virtual Status SetOption(const string& key, const string& value) {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("key: \"" + key + "\"");
   mht_38_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_38(mht_38_v, 841, "", "./tensorflow/core/platform/file_system.h", "SetOption");

    return errors::Unimplemented("SetOption");
  }

  /// \brief Set File System Configuration Option
  virtual tensorflow::Status SetOption(const std::string& name,
                                       const std::vector<string>& values) {
   std::vector<std::string> mht_39_v;
   mht_39_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_39(mht_39_v, 851, "", "./tensorflow/core/platform/file_system.h", "SetOption");

    return errors::Unimplemented("SetOption");
  }

  /// \brief Set File System Configuration Option
  virtual tensorflow::Status SetOption(const std::string& name,
                                       const std::vector<int64_t>& values) {
   std::vector<std::string> mht_40_v;
   mht_40_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_40(mht_40_v, 861, "", "./tensorflow/core/platform/file_system.h", "SetOption");

    return errors::Unimplemented("SetOption");
  }

  /// \brief Set File System Configuration Option
  virtual tensorflow::Status SetOption(const std::string& name,
                                       const std::vector<double>& values) {
   std::vector<std::string> mht_41_v;
   mht_41_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_41(mht_41_v, 871, "", "./tensorflow/core/platform/file_system.h", "SetOption");

    return errors::Unimplemented("SetOption");
  }

  FileSystem() {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_42(mht_42_v, 878, "", "./tensorflow/core/platform/file_system.h", "FileSystem");
}

  virtual ~FileSystem() = default;
};
/// This macro adds forwarding methods from FileSystem class to
/// used class since name hiding will prevent these to be accessed from
/// derived classes and would require all use locations to migrate to
/// Transactional API. This is an interim solution until ModularFileSystem class
/// becomes a singleton.
// TODO(sami): Remove this macro when filesystem plugins migration is complete.
#define TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT \
  using FileSystem::NewRandomAccessFile;                      \
  using FileSystem::NewWritableFile;                          \
  using FileSystem::NewAppendableFile;                        \
  using FileSystem::NewReadOnlyMemoryRegionFromFile;          \
  using FileSystem::FileExists;                               \
  using FileSystem::GetChildren;                              \
  using FileSystem::GetMatchingPaths;                         \
  using FileSystem::Stat;                                     \
  using FileSystem::DeleteFile;                               \
  using FileSystem::RecursivelyCreateDir;                     \
  using FileSystem::DeleteDir;                                \
  using FileSystem::DeleteRecursively;                        \
  using FileSystem::GetFileSize;                              \
  using FileSystem::RenameFile;                               \
  using FileSystem::CopyFile;                                 \
  using FileSystem::IsDirectory;                              \
  using FileSystem::FlushCaches

/// A Wrapper class for Transactional FileSystem support.
/// This provides means to make use of the transactions with minimal code change
/// Any operations that are done through this interface will be through the
/// transaction created at the time of construction of this instance.
/// See FileSystem documentation for method descriptions.
/// This class simply forwards all calls to wrapped filesystem either with given
/// transaction token or with token used in its construction. This allows doing
/// transactional filesystem access with minimal code change.
class WrappedFileSystem : public FileSystem {
 public:
  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  tensorflow::Status NewRandomAccessFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override {
   std::vector<std::string> mht_43_v;
   mht_43_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_43(mht_43_v, 925, "", "./tensorflow/core/platform/file_system.h", "NewRandomAccessFile");

    return fs_->NewRandomAccessFile(fname, (token ? token : token_), result);
  }

  tensorflow::Status NewWritableFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<WritableFile>* result) override {
   std::vector<std::string> mht_44_v;
   mht_44_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_44(mht_44_v, 935, "", "./tensorflow/core/platform/file_system.h", "NewWritableFile");

    return fs_->NewWritableFile(fname, (token ? token : token_), result);
  }

  tensorflow::Status NewAppendableFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<WritableFile>* result) override {
   std::vector<std::string> mht_45_v;
   mht_45_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_45(mht_45_v, 945, "", "./tensorflow/core/platform/file_system.h", "NewAppendableFile");

    return fs_->NewAppendableFile(fname, (token ? token : token_), result);
  }

  tensorflow::Status NewReadOnlyMemoryRegionFromFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
   std::vector<std::string> mht_46_v;
   mht_46_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_46(mht_46_v, 955, "", "./tensorflow/core/platform/file_system.h", "NewReadOnlyMemoryRegionFromFile");

    return fs_->NewReadOnlyMemoryRegionFromFile(fname, (token ? token : token_),
                                                result);
  }

  tensorflow::Status FileExists(const std::string& fname,
                                TransactionToken* token) override {
   std::vector<std::string> mht_47_v;
   mht_47_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_47(mht_47_v, 965, "", "./tensorflow/core/platform/file_system.h", "FileExists");

    return fs_->FileExists(fname, (token ? token : token_));
  }

  bool FilesExist(const std::vector<string>& files, TransactionToken* token,
                  std::vector<Status>* status) override {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_48(mht_48_v, 973, "", "./tensorflow/core/platform/file_system.h", "FilesExist");

    return fs_->FilesExist(files, (token ? token : token_), status);
  }

  tensorflow::Status GetChildren(const std::string& dir,
                                 TransactionToken* token,
                                 std::vector<string>* result) override {
   std::vector<std::string> mht_49_v;
   mht_49_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_49(mht_49_v, 983, "", "./tensorflow/core/platform/file_system.h", "GetChildren");

    return fs_->GetChildren(dir, (token ? token : token_), result);
  }

  tensorflow::Status GetMatchingPaths(const std::string& pattern,
                                      TransactionToken* token,
                                      std::vector<string>* results) override {
   std::vector<std::string> mht_50_v;
   mht_50_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_50(mht_50_v, 993, "", "./tensorflow/core/platform/file_system.h", "GetMatchingPaths");

    return fs_->GetMatchingPaths(pattern, (token ? token : token_), results);
  }

  bool Match(const std::string& filename, const std::string& pattern) override {
   std::vector<std::string> mht_51_v;
   mht_51_v.push_back("filename: \"" + filename + "\"");
   mht_51_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_51(mht_51_v, 1002, "", "./tensorflow/core/platform/file_system.h", "Match");

    return fs_->Match(filename, pattern);
  }

  tensorflow::Status Stat(const std::string& fname, TransactionToken* token,
                          FileStatistics* stat) override {
   std::vector<std::string> mht_52_v;
   mht_52_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_52(mht_52_v, 1011, "", "./tensorflow/core/platform/file_system.h", "Stat");

    return fs_->Stat(fname, (token ? token : token_), stat);
  }

  tensorflow::Status DeleteFile(const std::string& fname,
                                TransactionToken* token) override {
   std::vector<std::string> mht_53_v;
   mht_53_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_53(mht_53_v, 1020, "", "./tensorflow/core/platform/file_system.h", "DeleteFile");

    return fs_->DeleteFile(fname, (token ? token : token_));
  }

  tensorflow::Status CreateDir(const std::string& dirname,
                               TransactionToken* token) override {
   std::vector<std::string> mht_54_v;
   mht_54_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_54(mht_54_v, 1029, "", "./tensorflow/core/platform/file_system.h", "CreateDir");

    return fs_->CreateDir(dirname, (token ? token : token_));
  }

  tensorflow::Status RecursivelyCreateDir(const std::string& dirname,
                                          TransactionToken* token) override {
   std::vector<std::string> mht_55_v;
   mht_55_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_55(mht_55_v, 1038, "", "./tensorflow/core/platform/file_system.h", "RecursivelyCreateDir");

    return fs_->RecursivelyCreateDir(dirname, (token ? token : token_));
  }

  tensorflow::Status DeleteDir(const std::string& dirname,
                               TransactionToken* token) override {
   std::vector<std::string> mht_56_v;
   mht_56_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_56(mht_56_v, 1047, "", "./tensorflow/core/platform/file_system.h", "DeleteDir");

    return fs_->DeleteDir(dirname, (token ? token : token_));
  }

  tensorflow::Status DeleteRecursively(const std::string& dirname,
                                       TransactionToken* token,
                                       int64_t* undeleted_files,
                                       int64_t* undeleted_dirs) override {
   std::vector<std::string> mht_57_v;
   mht_57_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_57(mht_57_v, 1058, "", "./tensorflow/core/platform/file_system.h", "DeleteRecursively");

    return fs_->DeleteRecursively(dirname, (token ? token : token_),
                                  undeleted_files, undeleted_dirs);
  }

  tensorflow::Status GetFileSize(const std::string& fname,
                                 TransactionToken* token,
                                 uint64* file_size) override {
   std::vector<std::string> mht_58_v;
   mht_58_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_58(mht_58_v, 1069, "", "./tensorflow/core/platform/file_system.h", "GetFileSize");

    return fs_->GetFileSize(fname, (token ? token : token_), file_size);
  }

  tensorflow::Status RenameFile(const std::string& src,
                                const std::string& target,
                                TransactionToken* token) override {
   std::vector<std::string> mht_59_v;
   mht_59_v.push_back("src: \"" + src + "\"");
   mht_59_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_59(mht_59_v, 1080, "", "./tensorflow/core/platform/file_system.h", "RenameFile");

    return fs_->RenameFile(src, target, (token ? token : token_));
  }

  tensorflow::Status CopyFile(const std::string& src, const std::string& target,
                              TransactionToken* token) override {
   std::vector<std::string> mht_60_v;
   mht_60_v.push_back("src: \"" + src + "\"");
   mht_60_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_60(mht_60_v, 1090, "", "./tensorflow/core/platform/file_system.h", "CopyFile");

    return fs_->CopyFile(src, target, (token ? token : token_));
  }

  std::string TranslateName(const std::string& name) const override {
   std::vector<std::string> mht_61_v;
   mht_61_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_61(mht_61_v, 1098, "", "./tensorflow/core/platform/file_system.h", "TranslateName");

    return fs_->TranslateName(name);
  }

  tensorflow::Status IsDirectory(const std::string& fname,
                                 TransactionToken* token) override {
   std::vector<std::string> mht_62_v;
   mht_62_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_62(mht_62_v, 1107, "", "./tensorflow/core/platform/file_system.h", "IsDirectory");

    return fs_->IsDirectory(fname, (token ? token : token_));
  }

  Status HasAtomicMove(const std::string& path,
                       bool* has_atomic_move) override {
   std::vector<std::string> mht_63_v;
   mht_63_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_63(mht_63_v, 1116, "", "./tensorflow/core/platform/file_system.h", "HasAtomicMove");

    return fs_->HasAtomicMove(path, has_atomic_move);
  }

  void FlushCaches(TransactionToken* token) override {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_64(mht_64_v, 1123, "", "./tensorflow/core/platform/file_system.h", "FlushCaches");

    return fs_->FlushCaches((token ? token : token_));
  }

  char Separator() const override {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_65(mht_65_v, 1130, "", "./tensorflow/core/platform/file_system.h", "Separator");
 return fs_->Separator(); }

  StringPiece Basename(StringPiece path) const override {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_66(mht_66_v, 1135, "", "./tensorflow/core/platform/file_system.h", "Basename");

    return fs_->Basename(path);
  }

  tensorflow::Status StartTransaction(TransactionToken** token) override {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_67(mht_67_v, 1142, "", "./tensorflow/core/platform/file_system.h", "StartTransaction");

    return fs_->StartTransaction(token);
  }

  tensorflow::Status AddToTransaction(const std::string& path,
                                      TransactionToken* token) override {
   std::vector<std::string> mht_68_v;
   mht_68_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_68(mht_68_v, 1151, "", "./tensorflow/core/platform/file_system.h", "AddToTransaction");

    return fs_->AddToTransaction(path, (token ? token : token_));
  }

  tensorflow::Status EndTransaction(TransactionToken* token) override {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_69(mht_69_v, 1158, "", "./tensorflow/core/platform/file_system.h", "EndTransaction");

    return fs_->EndTransaction(token);
  }

  tensorflow::Status GetTransactionForPath(const std::string& path,
                                           TransactionToken** token) override {
   std::vector<std::string> mht_70_v;
   mht_70_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_70(mht_70_v, 1167, "", "./tensorflow/core/platform/file_system.h", "GetTransactionForPath");

    return fs_->GetTransactionForPath(path, token);
  }

  tensorflow::Status GetTokenOrStartTransaction(
      const std::string& path, TransactionToken** token) override {
   std::vector<std::string> mht_71_v;
   mht_71_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_71(mht_71_v, 1176, "", "./tensorflow/core/platform/file_system.h", "GetTokenOrStartTransaction");

    return fs_->GetTokenOrStartTransaction(path, token);
  }

  std::string DecodeTransaction(const TransactionToken* token) override {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_72(mht_72_v, 1183, "", "./tensorflow/core/platform/file_system.h", "DecodeTransaction");

    return fs_->DecodeTransaction((token ? token : token_));
  }

  WrappedFileSystem(FileSystem* file_system, TransactionToken* token)
      : fs_(file_system), token_(token) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_73(mht_73_v, 1191, "", "./tensorflow/core/platform/file_system.h", "WrappedFileSystem");
}

  ~WrappedFileSystem() override = default;

 private:
  FileSystem* fs_;
  TransactionToken* token_;
};

/// A file abstraction for randomly reading the contents of a file.
class RandomAccessFile {
 public:
  RandomAccessFile() {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_74(mht_74_v, 1206, "", "./tensorflow/core/platform/file_system.h", "RandomAccessFile");
}
  virtual ~RandomAccessFile() = default;

  /// \brief Returns the name of the file.
  ///
  /// This is an optional operation that may not be implemented by every
  /// filesystem.
  virtual tensorflow::Status Name(StringPiece* result) const {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_75(mht_75_v, 1216, "", "./tensorflow/core/platform/file_system.h", "Name");

    return errors::Unimplemented("This filesystem does not support Name()");
  }

  /// \brief Reads up to `n` bytes from the file starting at `offset`.
  ///
  /// `scratch[0..n-1]` may be written by this routine.  Sets `*result`
  /// to the data that was read (including if fewer than `n` bytes were
  /// successfully read).  May set `*result` to point at data in
  /// `scratch[0..n-1]`, so `scratch[0..n-1]` must be live when
  /// `*result` is used.
  ///
  /// On OK returned status: `n` bytes have been stored in `*result`.
  /// On non-OK returned status: `[0..n]` bytes have been stored in `*result`.
  ///
  /// Returns `OUT_OF_RANGE` if fewer than n bytes were stored in `*result`
  /// because of EOF.
  ///
  /// Safe for concurrent use by multiple threads.
  virtual tensorflow::Status Read(uint64 offset, size_t n, StringPiece* result,
                                  char* scratch) const = 0;

#if defined(TF_CORD_SUPPORT)
  /// \brief Read up to `n` bytes from the file starting at `offset`.
  virtual tensorflow::Status Read(uint64 offset, size_t n,
                                  absl::Cord* cord) const {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_76(mht_76_v, 1244, "", "./tensorflow/core/platform/file_system.h", "Read");

    return errors::Unimplemented(
        "Read(uint64, size_t, absl::Cord*) is not "
        "implemented");
  }
#endif

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomAccessFile);
};

/// \brief A file abstraction for sequential writing.
///
/// The implementation must provide buffering since callers may append
/// small fragments at a time to the file.
class WritableFile {
 public:
  WritableFile() {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_77(mht_77_v, 1264, "", "./tensorflow/core/platform/file_system.h", "WritableFile");
}
  virtual ~WritableFile() = default;

  /// \brief Append 'data' to the file.
  virtual tensorflow::Status Append(StringPiece data) = 0;

#if defined(TF_CORD_SUPPORT)
  // \brief Append 'data' to the file.
  virtual tensorflow::Status Append(const absl::Cord& cord) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_78(mht_78_v, 1275, "", "./tensorflow/core/platform/file_system.h", "Append");

    for (StringPiece chunk : cord.Chunks()) {
      TF_RETURN_IF_ERROR(Append(chunk));
    }
    return tensorflow::Status::OK();
  }
#endif

  /// \brief Close the file.
  ///
  /// Flush() and de-allocate resources associated with this file
  ///
  /// Typical return codes (not guaranteed to be exhaustive):
  ///  * OK
  ///  * Other codes, as returned from Flush()
  virtual tensorflow::Status Close() = 0;

  /// \brief Flushes the file and optionally syncs contents to filesystem.
  ///
  /// This should flush any local buffers whose contents have not been
  /// delivered to the filesystem.
  ///
  /// If the process terminates after a successful flush, the contents
  /// may still be persisted, since the underlying filesystem may
  /// eventually flush the contents.  If the OS or machine crashes
  /// after a successful flush, the contents may or may not be
  /// persisted, depending on the implementation.
  virtual tensorflow::Status Flush() = 0;

  // \brief Returns the name of the file.
  ///
  /// This is an optional operation that may not be implemented by every
  /// filesystem.
  virtual tensorflow::Status Name(StringPiece* result) const {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_79(mht_79_v, 1311, "", "./tensorflow/core/platform/file_system.h", "Name");

    return errors::Unimplemented("This filesystem does not support Name()");
  }

  /// \brief Syncs contents of file to filesystem.
  ///
  /// This waits for confirmation from the filesystem that the contents
  /// of the file have been persisted to the filesystem; if the OS
  /// or machine crashes after a successful Sync, the contents should
  /// be properly saved.
  virtual tensorflow::Status Sync() = 0;

  /// \brief Retrieves the current write position in the file, or -1 on
  /// error.
  ///
  /// This is an optional operation, subclasses may choose to return
  /// errors::Unimplemented.
  virtual tensorflow::Status Tell(int64_t* position) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_80(mht_80_v, 1331, "", "./tensorflow/core/platform/file_system.h", "Tell");

    *position = -1;
    return errors::Unimplemented("This filesystem does not support Tell()");
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(WritableFile);
};

/// \brief A readonly memmapped file abstraction.
///
/// The implementation must guarantee that all memory is accessible when the
/// object exists, independently from the Env that created it.
class ReadOnlyMemoryRegion {
 public:
  ReadOnlyMemoryRegion() {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_systemDTh mht_81(mht_81_v, 1349, "", "./tensorflow/core/platform/file_system.h", "ReadOnlyMemoryRegion");
}
  virtual ~ReadOnlyMemoryRegion() = default;

  /// \brief Returns a pointer to the memory region.
  virtual const void* data() = 0;

  /// \brief Returns the length of the memory region in bytes.
  virtual uint64 length() = 0;
};

/// \brief A registry for file system implementations.
///
/// Filenames are specified as an URI, which is of the form
/// [scheme://]<filename>.
/// File system implementations are registered using the REGISTER_FILE_SYSTEM
/// macro, providing the 'scheme' as the key.
///
/// There are two `Register` methods: one using `Factory` for legacy filesystems
/// (deprecated mechanism of subclassing `FileSystem` and using
/// `REGISTER_FILE_SYSTEM` macro), and one using `std::unique_ptr<FileSystem>`
/// for the new modular approach.
///
/// Note that the new API expects a pointer to `ModularFileSystem` but this is
/// not checked as there should be exactly one caller to the API and doing the
/// check results in a circular dependency between `BUILD` targets.
///
/// Plan is to completely remove the filesystem registration from `Env` and
/// incorporate it into `ModularFileSystem` class (which will be renamed to be
/// the only `FileSystem` class and marked as `final`). But this will happen at
/// a later time, after we convert all filesystems to the new API.
///
/// TODO(mihaimaruseac): After all filesystems are converted, remove old
/// registration and update comment.
class FileSystemRegistry {
 public:
  typedef std::function<FileSystem*()> Factory;

  virtual ~FileSystemRegistry() = default;
  virtual tensorflow::Status Register(const std::string& scheme,
                                      Factory factory) = 0;
  virtual tensorflow::Status Register(
      const std::string& scheme, std::unique_ptr<FileSystem> filesystem) = 0;
  virtual FileSystem* Lookup(const std::string& scheme) = 0;
  virtual tensorflow::Status GetRegisteredFileSystemSchemes(
      std::vector<std::string>* schemes) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_FILE_SYSTEM_H_
