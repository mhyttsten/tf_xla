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

#ifndef TENSORFLOW_CORE_PLATFORM_ENV_H_
#define TENSORFLOW_CORE_PLATFORM_ENV_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSenvDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSenvDTh() {
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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

// Delete leaked Windows definitions.
#ifdef PLATFORM_WINDOWS
#undef CopyFile
#undef DeleteFile
#endif

namespace tensorflow {

class Thread;
struct ThreadOptions;

/// \brief An interface used by the tensorflow implementation to
/// access operating system functionality like the filesystem etc.
///
/// Callers may wish to provide a custom Env object to get fine grain
/// control.
///
/// All Env implementations are safe for concurrent access from
/// multiple threads without any external synchronization.
class Env {
 public:
  Env();
  virtual ~Env() = default;

  /// \brief Returns a default environment suitable for the current operating
  /// system.
  ///
  /// Sophisticated users may wish to provide their own Env
  /// implementation instead of relying on this default environment.
  ///
  /// The result of Default() belongs to this library and must never be deleted.
  static Env* Default();

  /// \brief Returns the FileSystem object to handle operations on the file
  /// specified by 'fname'. The FileSystem object is used as the implementation
  /// for the file system related (non-virtual) functions that follow.
  /// Returned FileSystem object is still owned by the Env object and will
  // (might) be destroyed when the environment is destroyed.
  virtual Status GetFileSystemForFile(const std::string& fname,
                                      FileSystem** result);

  /// \brief Returns the file system schemes registered for this Env.
  virtual Status GetRegisteredFileSystemSchemes(
      std::vector<std::string>* schemes);

  /// \brief Register a file system for a scheme.
  virtual Status RegisterFileSystem(const std::string& scheme,
                                    FileSystemRegistry::Factory factory);

  /// \brief Register a modular file system for a scheme.
  ///
  /// Same as `RegisterFileSystem` but for filesystems provided by plugins.
  ///
  /// TODO(mihaimaruseac): After all filesystems are converted, make this be the
  /// canonical registration function.
  virtual Status RegisterFileSystem(const std::string& scheme,
                                    std::unique_ptr<FileSystem> filesystem);

  Status SetOption(const std::string& scheme, const std::string& key,
                   const std::string& value);

  Status SetOption(const std::string& scheme, const std::string& key,
                   const std::vector<string>& values);

  Status SetOption(const std::string& scheme, const std::string& key,
                   const std::vector<int64_t>& values);

  Status SetOption(const std::string& scheme, const std::string& key,
                   const std::vector<double>& values);

  /// \brief Flush filesystem caches for all registered filesystems.
  Status FlushFileSystemCaches();

  /// \brief Creates a brand new random access read-only file with the
  /// specified name.

  /// On success, stores a pointer to the new file in
  /// *result and returns OK.  On failure stores NULL in *result and
  /// returns non-OK.  If the file does not exist, returns a non-OK
  /// status.
  ///
  /// The returned file may be concurrently accessed by multiple threads.
  ///
  /// The ownership of the returned RandomAccessFile is passed to the caller
  /// and the object should be deleted when is not used. The file object
  /// shouldn't live longer than the Env object.
  Status NewRandomAccessFile(const std::string& fname,
                             std::unique_ptr<RandomAccessFile>* result);

  Status NewRandomAccessFile(const std::string& fname, TransactionToken* token,
                             std::unique_ptr<RandomAccessFile>* result) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_0(mht_0_v, 298, "", "./tensorflow/core/platform/env.h", "NewRandomAccessFile");

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
  /// and the object should be deleted when is not used. The file object
  /// shouldn't live longer than the Env object.
  Status NewWritableFile(const std::string& fname,
                         std::unique_ptr<WritableFile>* result);

  Status NewWritableFile(const std::string& fname, TransactionToken* token,
                         std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_1(mht_1_v, 325, "", "./tensorflow/core/platform/env.h", "NewWritableFile");

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
  /// and the object should be deleted when is not used. The file object
  /// shouldn't live longer than the Env object.
  Status NewAppendableFile(const std::string& fname,
                           std::unique_ptr<WritableFile>* result);

  Status NewAppendableFile(const std::string& fname, TransactionToken* token,
                           std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_2(mht_2_v, 349, "", "./tensorflow/core/platform/env.h", "NewAppendableFile");

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
  /// and the object should be deleted when is not used. The memory region
  /// object shouldn't live longer than the Env object.
  Status NewReadOnlyMemoryRegionFromFile(
      const std::string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result);

  Status NewReadOnlyMemoryRegionFromFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_3(mht_3_v, 372, "", "./tensorflow/core/platform/env.h", "NewReadOnlyMemoryRegionFromFile");

    return Status::OK();
  }

  /// Returns OK if the named path exists and NOT_FOUND otherwise.
  Status FileExists(const std::string& fname);

  Status FileExists(const std::string& fname, TransactionToken* token) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_4(mht_4_v, 383, "", "./tensorflow/core/platform/env.h", "FileExists");

    return Status::OK();
  }

  /// Returns true if all the listed files exist, false otherwise.
  /// if status is not null, populate the vector with a detailed status
  /// for each file.
  bool FilesExist(const std::vector<string>& files,
                  std::vector<Status>* status);

  bool FilesExist(const std::vector<string>& files, TransactionToken* token,
                  std::vector<Status>* status) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_5(mht_5_v, 397, "", "./tensorflow/core/platform/env.h", "FilesExist");

    return true;
  }

  /// \brief Stores in *result the names of the children of the specified
  /// directory. The names are relative to "dir".
  ///
  /// Original contents of *results are dropped.
  Status GetChildren(const std::string& dir, std::vector<string>* result);

  Status GetChildren(const std::string& dir, TransactionToken* token,
                     std::vector<string>* result) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_6(mht_6_v, 412, "", "./tensorflow/core/platform/env.h", "GetChildren");

    return Status::OK();
  }

  /// \brief Returns true if the path matches the given pattern. The wildcards
  /// allowed in pattern are described in FileSystem::GetMatchingPaths.
  virtual bool MatchPath(const std::string& path,
                         const std::string& pattern) = 0;

  /// \brief Given a pattern, stores in *results the set of paths that matches
  /// that pattern. *results is cleared.
  ///
  /// More details about `pattern` in FileSystem::GetMatchingPaths.
  virtual Status GetMatchingPaths(const std::string& pattern,
                                  std::vector<string>* results);

  Status GetMatchingPaths(const std::string& pattern, TransactionToken* token,
                          std::vector<string>* results) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_7(mht_7_v, 433, "", "./tensorflow/core/platform/env.h", "GetMatchingPaths");

    return Status::OK();
  }

  /// Deletes the named file.
  Status DeleteFile(const std::string& fname);

  Status DeleteFile(const std::string& fname, TransactionToken* token) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_8(mht_8_v, 444, "", "./tensorflow/core/platform/env.h", "DeleteFile");

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
  Status DeleteRecursively(const std::string& dirname, int64_t* undeleted_files,
                           int64_t* undeleted_dirs);

  Status DeleteRecursively(const std::string& dirname, TransactionToken* token,
                           int64_t* undeleted_files, int64_t* undeleted_dirs) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_9(mht_9_v, 480, "", "./tensorflow/core/platform/env.h", "DeleteRecursively");

    return Status::OK();
  }

  /// \brief Creates the specified directory and all the necessary
  /// subdirectories. Typical return codes.
  ///  * OK - successfully created the directory and sub directories, even if
  ///         they were already created.
  ///  * PERMISSION_DENIED - dirname or some subdirectory is not writable.
  Status RecursivelyCreateDir(const std::string& dirname);

  Status RecursivelyCreateDir(const std::string& dirname,
                              TransactionToken* token) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_10(mht_10_v, 496, "", "./tensorflow/core/platform/env.h", "RecursivelyCreateDir");

    return Status::OK();
  }
  /// \brief Creates the specified directory. Typical return codes
  ///  * OK - successfully created the directory.
  ///  * ALREADY_EXISTS - directory already exists.
  ///  * PERMISSION_DENIED - dirname is not writable.
  Status CreateDir(const std::string& dirname);

  Status CreateDir(const std::string& dirname, TransactionToken* token) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_11(mht_11_v, 509, "", "./tensorflow/core/platform/env.h", "CreateDir");

    return Status::OK();
  }

  /// Deletes the specified directory.
  Status DeleteDir(const std::string& dirname);

  Status DeleteDir(const std::string& dirname, TransactionToken* token) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_12(mht_12_v, 520, "", "./tensorflow/core/platform/env.h", "DeleteDir");

    return Status::OK();
  }

  /// Obtains statistics for the given path.
  Status Stat(const std::string& fname, FileStatistics* stat);

  Status Stat(const std::string& fname, TransactionToken* token,
              FileStatistics* stat) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_13(mht_13_v, 532, "", "./tensorflow/core/platform/env.h", "Stat");

    return Status::OK();
  }

  /// \brief Returns whether the given path is a directory or not.
  /// Typical return codes (not guaranteed exhaustive):
  ///  * OK - The path exists and is a directory.
  ///  * FAILED_PRECONDITION - The path exists and is not a directory.
  ///  * NOT_FOUND - The path entry does not exist.
  ///  * PERMISSION_DENIED - Insufficient permissions.
  ///  * UNIMPLEMENTED - The file factory doesn't support directories.
  Status IsDirectory(const std::string& fname);

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
  Status HasAtomicMove(const std::string& path, bool* has_atomic_move);

  /// Stores the size of `fname` in `*file_size`.
  Status GetFileSize(const std::string& fname, uint64* file_size);

  Status GetFileSize(const std::string& fname, TransactionToken* token,
                     uint64* file_size) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_14(mht_14_v, 565, "", "./tensorflow/core/platform/env.h", "GetFileSize");

    return Status::OK();
  }

  /// \brief Renames file src to target. If target already exists, it will be
  /// replaced.
  Status RenameFile(const std::string& src, const std::string& target);

  Status RenameFile(const std::string& src, const std::string& target,
                    TransactionToken* token) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("src: \"" + src + "\"");
   mht_15_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_15(mht_15_v, 579, "", "./tensorflow/core/platform/env.h", "RenameFile");

    return Status::OK();
  }

  /// \brief Copy the src to target.
  Status CopyFile(const std::string& src, const std::string& target);

  Status CopyFile(const std::string& src, const std::string& target,
                  TransactionToken* token) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("src: \"" + src + "\"");
   mht_16_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_16(mht_16_v, 592, "", "./tensorflow/core/platform/env.h", "CopyFile");

    return Status::OK();
  }

  /// \brief starts a new transaction on the filesystem that handles filename
  Status StartTransaction(const std::string& filename,
                          TransactionToken** token) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_17(mht_17_v, 602, "", "./tensorflow/core/platform/env.h", "StartTransaction");

    *token = nullptr;
    return Status::OK();
  }

  /// \brief Adds `path` to transaction in `token` if token belongs to
  /// filesystem that handles the path.
  Status AddToTransaction(const std::string& path, TransactionToken* token) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_18(mht_18_v, 613, "", "./tensorflow/core/platform/env.h", "AddToTransaction");

    return Status::OK();
  }

  /// \brief Get token for `path` or start a new transaction and add `path` to
  /// it.
  Status GetTokenOrStartTransaction(const std::string& path,
                                    TransactionToken** token) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_19(mht_19_v, 624, "", "./tensorflow/core/platform/env.h", "GetTokenOrStartTransaction");

    *token = nullptr;
    return Status::OK();
  }

  /// \brief Returns the transaction for `path` or nullptr in `token`
  Status GetTransactionForPath(const std::string& path,
                               TransactionToken** token) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_20(mht_20_v, 635, "", "./tensorflow/core/platform/env.h", "GetTransactionForPath");

    *token = nullptr;
    return Status::OK();
  }

  /// \brief Finalizes the transaction
  Status EndTransaction(TransactionToken* token) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_21(mht_21_v, 644, "", "./tensorflow/core/platform/env.h", "EndTransaction");
 return Status::OK(); }

  /// \brief Returns the absolute path of the current executable. It resolves
  /// symlinks if there is any.
  std::string GetExecutablePath();

  /// Creates a local unique temporary file name. Returns true if success.
  bool LocalTempFilename(std::string* filename);

  /// Creates a local unique file name that starts with |prefix| and ends with
  /// |suffix|. Returns true if success.
  bool CreateUniqueFileName(std::string* prefix, const std::string& suffix);

  /// \brief Return the runfiles directory if running under bazel. Returns
  /// the directory the executable is located in if not running under bazel.
  virtual std::string GetRunfilesDir() = 0;

  // TODO(jeff,sanjay): Add back thread/thread-pool support if needed.
  // TODO(jeff,sanjay): if needed, tighten spec so relative to epoch, or
  // provide a routine to get the absolute time.

  /// \brief Returns the number of nano-seconds since the Unix epoch.
  virtual uint64 NowNanos() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_22(mht_22_v, 669, "", "./tensorflow/core/platform/env.h", "NowNanos");
 return EnvTime::NowNanos(); }

  /// \brief Returns the number of micro-seconds since the Unix epoch.
  virtual uint64 NowMicros() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_23(mht_23_v, 675, "", "./tensorflow/core/platform/env.h", "NowMicros");
 return EnvTime::NowMicros(); }

  /// \brief Returns the number of seconds since the Unix epoch.
  virtual uint64 NowSeconds() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_24(mht_24_v, 681, "", "./tensorflow/core/platform/env.h", "NowSeconds");
 return EnvTime::NowSeconds(); }

  /// Sleeps/delays the thread for the prescribed number of micro-seconds.
  virtual void SleepForMicroseconds(int64_t micros) = 0;

  /// Returns the process ID of the calling process.
  int32 GetProcessId();

  /// \brief Returns a new thread that is running fn() and is identified
  /// (for debugging/performance-analysis) by "name".
  ///
  /// Caller takes ownership of the result and must delete it eventually
  /// (the deletion will block until fn() stops running).
  virtual Thread* StartThread(const ThreadOptions& thread_options,
                              const std::string& name,
                              std::function<void()> fn) TF_MUST_USE_RESULT = 0;

  // Returns the thread id of calling thread.
  // Posix: Returns pthread id which is only guaranteed to be unique within a
  //        process.
  // Windows: Returns thread id which is unique.
  virtual int32 GetCurrentThreadId() = 0;

  // Copies current thread name to "name". Returns true if success.
  virtual bool GetCurrentThreadName(std::string* name) = 0;

  // \brief Schedules the given closure on a thread-pool.
  //
  // NOTE(mrry): This closure may block.
  virtual void SchedClosure(std::function<void()> closure) = 0;

  // \brief Schedules the given closure on a thread-pool after the given number
  // of microseconds.
  //
  // NOTE(mrry): This closure must not block.
  virtual void SchedClosureAfter(int64_t micros,
                                 std::function<void()> closure) = 0;

  // \brief Load a dynamic library.
  //
  // Pass "library_filename" to a platform-specific mechanism for dynamically
  // loading a library.  The rules for determining the exact location of the
  // library are platform-specific and are not documented here.
  //
  // On success, returns a handle to the library in "*handle" and returns
  // OK from the function.
  // Otherwise returns nullptr in "*handle" and an error status from the
  // function.
  virtual Status LoadDynamicLibrary(const char* library_filename,
                                    void** handle) = 0;

  // \brief Get a pointer to a symbol from a dynamic library.
  //
  // "handle" should be a pointer returned from a previous call to LoadLibrary.
  // On success, store a pointer to the located symbol in "*symbol" and return
  // OK from the function. Otherwise, returns nullptr in "*symbol" and an error
  // status from the function.
  virtual Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                                      void** symbol) = 0;

  // \brief build the name of dynamic library.
  //
  // "name" should be name of the library.
  // "version" should be the version of the library or NULL
  // returns the name that LoadLibrary() can use
  virtual std::string FormatLibraryFileName(const std::string& name,
                                            const std::string& version) = 0;

  // Returns a possible list of local temporary directories.
  virtual void GetLocalTempDirectories(std::vector<string>* list) = 0;

 private:
  std::unique_ptr<FileSystemRegistry> file_system_registry_;
  TF_DISALLOW_COPY_AND_ASSIGN(Env);
};

/// \brief An implementation of Env that forwards all calls to another Env.
///
/// May be useful to clients who wish to override just part of the
/// functionality of another Env.
class EnvWrapper : public Env {
 public:
  /// Initializes an EnvWrapper that delegates all calls to *t
  explicit EnvWrapper(Env* t) : target_(t) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_25(mht_25_v, 767, "", "./tensorflow/core/platform/env.h", "EnvWrapper");
}
  ~EnvWrapper() override;

  /// Returns the target to which this Env forwards all calls
  Env* target() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_26(mht_26_v, 774, "", "./tensorflow/core/platform/env.h", "target");
 return target_; }

  Status GetFileSystemForFile(const std::string& fname,
                              FileSystem** result) override {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_27(mht_27_v, 781, "", "./tensorflow/core/platform/env.h", "GetFileSystemForFile");

    return target_->GetFileSystemForFile(fname, result);
  }

  Status GetRegisteredFileSystemSchemes(std::vector<string>* schemes) override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_28(mht_28_v, 788, "", "./tensorflow/core/platform/env.h", "GetRegisteredFileSystemSchemes");

    return target_->GetRegisteredFileSystemSchemes(schemes);
  }

  Status RegisterFileSystem(const std::string& scheme,
                            FileSystemRegistry::Factory factory) override {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("scheme: \"" + scheme + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_29(mht_29_v, 797, "", "./tensorflow/core/platform/env.h", "RegisterFileSystem");

    return target_->RegisterFileSystem(scheme, factory);
  }

  bool MatchPath(const std::string& path, const std::string& pattern) override {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("path: \"" + path + "\"");
   mht_30_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_30(mht_30_v, 806, "", "./tensorflow/core/platform/env.h", "MatchPath");

    return target_->MatchPath(path, pattern);
  }

  uint64 NowMicros() const override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_31(mht_31_v, 813, "", "./tensorflow/core/platform/env.h", "NowMicros");
 return target_->NowMicros(); }
  void SleepForMicroseconds(int64_t micros) override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_32(mht_32_v, 817, "", "./tensorflow/core/platform/env.h", "SleepForMicroseconds");

    target_->SleepForMicroseconds(micros);
  }
  Thread* StartThread(const ThreadOptions& thread_options,
                      const std::string& name,
                      std::function<void()> fn) override {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_33(mht_33_v, 826, "", "./tensorflow/core/platform/env.h", "StartThread");

    return target_->StartThread(thread_options, name, fn);
  }
  int32 GetCurrentThreadId() override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_34(mht_34_v, 832, "", "./tensorflow/core/platform/env.h", "GetCurrentThreadId");
 return target_->GetCurrentThreadId(); }
  bool GetCurrentThreadName(std::string* name) override {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_35(mht_35_v, 836, "", "./tensorflow/core/platform/env.h", "GetCurrentThreadName");

    return target_->GetCurrentThreadName(name);
  }
  void SchedClosure(std::function<void()> closure) override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_36(mht_36_v, 842, "", "./tensorflow/core/platform/env.h", "SchedClosure");

    target_->SchedClosure(closure);
  }
  void SchedClosureAfter(int64_t micros,
                         std::function<void()> closure) override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_37(mht_37_v, 849, "", "./tensorflow/core/platform/env.h", "SchedClosureAfter");

    target_->SchedClosureAfter(micros, closure);
  }
  Status LoadDynamicLibrary(const char* library_filename,
                            void** handle) override {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("library_filename: \"" + (library_filename == nullptr ? std::string("nullptr") : std::string((char*)library_filename)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_38(mht_38_v, 857, "", "./tensorflow/core/platform/env.h", "LoadDynamicLibrary");

    return target_->LoadDynamicLibrary(library_filename, handle);
  }
  Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                              void** symbol) override {
   std::vector<std::string> mht_39_v;
   mht_39_v.push_back("symbol_name: \"" + (symbol_name == nullptr ? std::string("nullptr") : std::string((char*)symbol_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_39(mht_39_v, 865, "", "./tensorflow/core/platform/env.h", "GetSymbolFromLibrary");

    return target_->GetSymbolFromLibrary(handle, symbol_name, symbol);
  }
  std::string FormatLibraryFileName(const std::string& name,
                                    const std::string& version) override {
   std::vector<std::string> mht_40_v;
   mht_40_v.push_back("name: \"" + name + "\"");
   mht_40_v.push_back("version: \"" + version + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_40(mht_40_v, 874, "", "./tensorflow/core/platform/env.h", "FormatLibraryFileName");

    return target_->FormatLibraryFileName(name, version);
  }

  std::string GetRunfilesDir() override {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_41(mht_41_v, 881, "", "./tensorflow/core/platform/env.h", "GetRunfilesDir");
 return target_->GetRunfilesDir(); }

 private:
  void GetLocalTempDirectories(std::vector<string>* list) override {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_42(mht_42_v, 887, "", "./tensorflow/core/platform/env.h", "GetLocalTempDirectories");

    target_->GetLocalTempDirectories(list);
  }

  Env* target_;
};

/// Represents a thread used to run a TensorFlow function.
class Thread {
 public:
  Thread() {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_43(mht_43_v, 900, "", "./tensorflow/core/platform/env.h", "Thread");
}

  /// Blocks until the thread of control stops running.
  virtual ~Thread();

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(Thread);
};

/// \brief Cross-platform setenv.
///
/// Since setenv() is not available on windows, we provide an
/// alternative with platform specific implementations here.
int setenv(const char* name, const char* value, int overwrite);

/// Cross-platform unsetenv.
int unsetenv(const char* name);

/// \brief Options to configure a Thread.
///
/// Note that the options are all hints, and the
/// underlying implementation may choose to ignore it.
struct ThreadOptions {
  /// Thread stack size to use (in bytes).
  size_t stack_size = 0;  // 0: use system default value
  /// Guard area size to use near thread stacks to use (in bytes)
  size_t guard_size = 0;  // 0: use system default value
  int numa_node = port::kNUMANoAffinity;
};

/// A utility routine: copy contents of `src` in file system `src_fs`
/// to `target` in file system `target_fs`.
Status FileSystemCopyFile(FileSystem* src_fs, const std::string& src,
                          FileSystem* target_fs, const std::string& target);

/// A utility routine: reads contents of named file into `*data`
Status ReadFileToString(Env* env, const std::string& fname, std::string* data);

/// A utility routine: write contents of `data` to file named `fname`
/// (overwriting existing contents, if any).
Status WriteStringToFile(Env* env, const std::string& fname,
                         const StringPiece& data);

/// Write binary representation of "proto" to the named file.
Status WriteBinaryProto(Env* env, const std::string& fname,
                        const protobuf::MessageLite& proto);

/// Reads contents of named file and parse as binary encoded proto data
/// and store into `*proto`.
Status ReadBinaryProto(Env* env, const std::string& fname,
                       protobuf::MessageLite* proto);

/// Write the text representation of "proto" to the named file.
inline Status WriteTextProto(Env* /* env */, const std::string& /* fname */,
                             const protobuf::MessageLite& /* proto */) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_44(mht_44_v, 957, "", "./tensorflow/core/platform/env.h", "WriteTextProto");

  return errors::Unimplemented("Can't write text protos with protolite.");
}
Status WriteTextProto(Env* env, const std::string& fname,
                      const protobuf::Message& proto);

/// Read contents of named file and parse as text encoded proto data
/// and store into `*proto`.
inline Status ReadTextProto(Env* /* env */, const std::string& /* fname */,
                            protobuf::MessageLite* /* proto */) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_45(mht_45_v, 969, "", "./tensorflow/core/platform/env.h", "ReadTextProto");

  return errors::Unimplemented("Can't parse text protos with protolite.");
}
Status ReadTextProto(Env* env, const std::string& fname,
                     protobuf::Message* proto);

/// Read contents of named file and parse as either text or binary encoded proto
/// data and store into `*proto`.
Status ReadTextOrBinaryProto(Env* env, const std::string& fname,
                             protobuf::Message* proto);
Status ReadTextOrBinaryProto(Env* env, const std::string& fname,
                             protobuf::MessageLite* proto);

// START_SKIP_DOXYGEN

// The following approach to register filesystems is deprecated and will be
// replaced with modular filesystem plugins registration.
// TODO(mihaimaruseac): After all filesystems are converted, remove this.
namespace register_file_system {

template <typename Factory>
struct Register {
  Register(Env* env, const std::string& scheme, bool try_modular_filesystems) {
   std::vector<std::string> mht_46_v;
   mht_46_v.push_back("scheme: \"" + scheme + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenvDTh mht_46(mht_46_v, 995, "", "./tensorflow/core/platform/env.h", "Register");

    // TODO(yongtang): Remove legacy file system registration for hdfs/s3/gcs
    // after TF 2.6+.
    if (try_modular_filesystems) {
      const char* env_value = getenv("TF_USE_MODULAR_FILESYSTEM");
      string load_plugin = env_value ? absl::AsciiStrToLower(env_value) : "";
      if (load_plugin == "true" || load_plugin == "1") {
        // We don't register the static filesystem and wait for SIG IO one
        LOG(WARNING) << "Using modular file system for '" << scheme << "'."
                     << " Please switch to tensorflow-io"
                     << " (https://github.com/tensorflow/io) for file system"
                     << " support of '" << scheme << "'.";
        return;
      }
      // If the envvar is missing or not "true"/"1", then fall back to legacy
      // implementation to be backwards compatible.
    }
    // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
    env->RegisterFileSystem(scheme, []() -> FileSystem* { return new Factory; })
        .IgnoreError();
  }
};

}  // namespace register_file_system

// END_SKIP_DOXYGEN

}  // namespace tensorflow

// Register a FileSystem implementation for a scheme. Files with names that have
// "scheme://" prefixes are routed to use this implementation.
#define REGISTER_FILE_SYSTEM_ENV(env, scheme, factory, modular) \
  REGISTER_FILE_SYSTEM_UNIQ_HELPER(__COUNTER__, env, scheme, factory, modular)
#define REGISTER_FILE_SYSTEM_UNIQ_HELPER(ctr, env, scheme, factory, modular) \
  REGISTER_FILE_SYSTEM_UNIQ(ctr, env, scheme, factory, modular)
#define REGISTER_FILE_SYSTEM_UNIQ(ctr, env, scheme, factory, modular)        \
  static ::tensorflow::register_file_system::Register<factory>               \
      register_ff##ctr TF_ATTRIBUTE_UNUSED =                                 \
          ::tensorflow::register_file_system::Register<factory>(env, scheme, \
                                                                modular)

#define REGISTER_FILE_SYSTEM(scheme, factory)                             \
  REGISTER_FILE_SYSTEM_ENV(::tensorflow::Env::Default(), scheme, factory, \
                           false);

#define REGISTER_LEGACY_FILE_SYSTEM(scheme, factory) \
  REGISTER_FILE_SYSTEM_ENV(::tensorflow::Env::Default(), scheme, factory, true);

#endif  // TENSORFLOW_CORE_PLATFORM_ENV_H_
