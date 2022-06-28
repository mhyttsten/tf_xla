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
class MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc() {
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

#include "tensorflow/tools/android/inference_interface/asset_manager_filesystem.h"

#include <unistd.h>

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system_helper.h"

namespace tensorflow {
namespace {

string RemoveSuffix(const string& name, const string& suffix) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   mht_0_v.push_back("suffix: \"" + suffix + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_0(mht_0_v, 198, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "RemoveSuffix");

  string output(name);
  StringPiece piece(output);
  absl::ConsumeSuffix(&piece, suffix);
  return string(piece);
}

// Closes the given AAsset when variable is destructed.
class ScopedAsset {
 public:
  ScopedAsset(AAsset* asset) : asset_(asset) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_1(mht_1_v, 211, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "ScopedAsset");
}
  ~ScopedAsset() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_2(mht_2_v, 215, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "~ScopedAsset");

    if (asset_ != nullptr) {
      AAsset_close(asset_);
    }
  }

  AAsset* get() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_3(mht_3_v, 224, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "get");
 return asset_; }

 private:
  AAsset* asset_;
};

// Closes the given AAssetDir when variable is destructed.
class ScopedAssetDir {
 public:
  ScopedAssetDir(AAssetDir* asset_dir) : asset_dir_(asset_dir) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_4(mht_4_v, 236, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "ScopedAssetDir");
}
  ~ScopedAssetDir() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_5(mht_5_v, 240, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "~ScopedAssetDir");

    if (asset_dir_ != nullptr) {
      AAssetDir_close(asset_dir_);
    }
  }

  AAssetDir* get() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_6(mht_6_v, 249, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "get");
 return asset_dir_; }

 private:
  AAssetDir* asset_dir_;
};

class ReadOnlyMemoryRegionFromAsset : public ReadOnlyMemoryRegion {
 public:
  ReadOnlyMemoryRegionFromAsset(std::unique_ptr<char[]> data, uint64 length)
      : data_(std::move(data)), length_(length) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_7(mht_7_v, 261, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "ReadOnlyMemoryRegionFromAsset");
}
  ~ReadOnlyMemoryRegionFromAsset() override = default;

  const void* data() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_8(mht_8_v, 267, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "data");
 return reinterpret_cast<void*>(data_.get()); }
  uint64 length() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_9(mht_9_v, 271, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "length");
 return length_; }

 private:
  std::unique_ptr<char[]> data_;
  uint64 length_;
};

// Note that AAssets are not thread-safe and cannot be used across threads.
// However, AAssetManager is. Because RandomAccessFile must be thread-safe and
// used across threads, new AAssets must be created for every access.
// TODO(tylerrhodes): is there a more efficient way to do this?
class RandomAccessFileFromAsset : public RandomAccessFile {
 public:
  RandomAccessFileFromAsset(AAssetManager* asset_manager, const string& name)
      : asset_manager_(asset_manager), file_name_(name) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_10(mht_10_v, 289, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "RandomAccessFileFromAsset");
}
  ~RandomAccessFileFromAsset() override = default;

  Status Read(uint64 offset, size_t to_read, StringPiece* result,
              char* scratch) const override {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("scratch: \"" + (scratch == nullptr ? std::string("nullptr") : std::string((char*)scratch)) + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_11(mht_11_v, 297, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "Read");

    auto asset = ScopedAsset(AAssetManager_open(
        asset_manager_, file_name_.c_str(), AASSET_MODE_RANDOM));
    if (asset.get() == nullptr) {
      return errors::NotFound("File ", file_name_, " not found.");
    }

    off64_t new_offset = AAsset_seek64(asset.get(), offset, SEEK_SET);
    off64_t length = AAsset_getLength64(asset.get());
    if (new_offset < 0) {
      *result = StringPiece(scratch, 0);
      return errors::OutOfRange("Read after file end.");
    }
    const off64_t region_left =
        std::min(length - new_offset, static_cast<off64_t>(to_read));
    int read = AAsset_read(asset.get(), scratch, region_left);
    if (read < 0) {
      return errors::Internal("Error reading from asset.");
    }
    *result = StringPiece(scratch, region_left);
    return (region_left == to_read)
               ? Status::OK()
               : errors::OutOfRange("Read less bytes than requested.");
  }

 private:
  AAssetManager* asset_manager_;
  string file_name_;
};

}  // namespace

AssetManagerFileSystem::AssetManagerFileSystem(AAssetManager* asset_manager,
                                               const string& prefix)
    : asset_manager_(asset_manager), prefix_(prefix) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_12(mht_12_v, 335, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::AssetManagerFileSystem");
}

Status AssetManagerFileSystem::FileExists(const string& fname,
                                          TransactionToken* token) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_13(mht_13_v, 342, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::FileExists");

  string path = RemoveAssetPrefix(fname);
  auto asset = ScopedAsset(
      AAssetManager_open(asset_manager_, path.c_str(), AASSET_MODE_RANDOM));
  if (asset.get() == nullptr) {
    return errors::NotFound("File ", fname, " not found.");
  }
  return Status::OK();
}

Status AssetManagerFileSystem::NewRandomAccessFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_14(mht_14_v, 358, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::NewRandomAccessFile");

  string path = RemoveAssetPrefix(fname);
  auto asset = ScopedAsset(
      AAssetManager_open(asset_manager_, path.c_str(), AASSET_MODE_RANDOM));
  if (asset.get() == nullptr) {
    return errors::NotFound("File ", fname, " not found.");
  }
  result->reset(new RandomAccessFileFromAsset(asset_manager_, path));
  return Status::OK();
}

Status AssetManagerFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_15(mht_15_v, 375, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::NewReadOnlyMemoryRegionFromFile");

  string path = RemoveAssetPrefix(fname);
  auto asset = ScopedAsset(
      AAssetManager_open(asset_manager_, path.c_str(), AASSET_MODE_STREAMING));
  if (asset.get() == nullptr) {
    return errors::NotFound("File ", fname, " not found.");
  }

  off64_t start, length;
  int fd = AAsset_openFileDescriptor64(asset.get(), &start, &length);
  std::unique_ptr<char[]> data;
  if (fd >= 0) {
    data.reset(new char[length]);
    ssize_t result = pread(fd, data.get(), length, start);
    if (result < 0) {
      return errors::Internal("Error reading from file ", fname,
                              " using 'read': ", result);
    }
    if (result != length) {
      return errors::Internal("Expected size does not match size read: ",
                              "Expected ", length, " vs. read ", result);
    }
    close(fd);
  } else {
    length = AAsset_getLength64(asset.get());
    data.reset(new char[length]);
    const void* asset_buffer = AAsset_getBuffer(asset.get());
    if (asset_buffer == nullptr) {
      return errors::Internal("Error reading ", fname, " from asset manager.");
    }
    memcpy(data.get(), asset_buffer, length);
  }
  result->reset(new ReadOnlyMemoryRegionFromAsset(std::move(data), length));
  return Status::OK();
}

Status AssetManagerFileSystem::GetChildren(const string& prefixed_dir,
                                           TransactionToken* token,
                                           std::vector<string>* r) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("prefixed_dir: \"" + prefixed_dir + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_16(mht_16_v, 417, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::GetChildren");

  std::string path = NormalizeDirectoryPath(prefixed_dir);
  auto dir =
      ScopedAssetDir(AAssetManager_openDir(asset_manager_, path.c_str()));
  if (dir.get() == nullptr) {
    return errors::NotFound("Directory ", prefixed_dir, " not found.");
  }
  const char* next_file = AAssetDir_getNextFileName(dir.get());
  while (next_file != nullptr) {
    r->push_back(next_file);
    next_file = AAssetDir_getNextFileName(dir.get());
  }
  return Status::OK();
}

Status AssetManagerFileSystem::GetFileSize(const string& fname,
                                           TransactionToken* token, uint64* s) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_17(mht_17_v, 437, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::GetFileSize");

  // If fname corresponds to a directory, return early. It doesn't map to an
  // AAsset, and would otherwise return NotFound.
  if (DirectoryExists(fname)) {
    *s = 0;
    return Status::OK();
  }
  string path = RemoveAssetPrefix(fname);
  auto asset = ScopedAsset(
      AAssetManager_open(asset_manager_, path.c_str(), AASSET_MODE_RANDOM));
  if (asset.get() == nullptr) {
    return errors::NotFound("File ", fname, " not found.");
  }
  *s = AAsset_getLength64(asset.get());
  return Status::OK();
}

Status AssetManagerFileSystem::Stat(const string& fname,
                                    TransactionToken* token,
                                    FileStatistics* stat) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_18(mht_18_v, 460, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::Stat");

  uint64 size;
  stat->is_directory = DirectoryExists(fname);
  TF_RETURN_IF_ERROR(GetFileSize(fname, &size));
  stat->length = size;
  return Status::OK();
}

string AssetManagerFileSystem::NormalizeDirectoryPath(const string& fname) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_19(mht_19_v, 472, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::NormalizeDirectoryPath");

  return RemoveSuffix(RemoveAssetPrefix(fname), "/");
}

string AssetManagerFileSystem::RemoveAssetPrefix(const string& name) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_20(mht_20_v, 480, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::RemoveAssetPrefix");

  StringPiece piece(name);
  absl::ConsumePrefix(&piece, prefix_);
  return string(piece);
}

bool AssetManagerFileSystem::DirectoryExists(const std::string& fname) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_21(mht_21_v, 490, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::DirectoryExists");

  std::string path = NormalizeDirectoryPath(fname);
  auto dir =
      ScopedAssetDir(AAssetManager_openDir(asset_manager_, path.c_str()));
  // Note that openDir will return something even if the directory doesn't
  // exist. Therefore, we need to ensure one file exists in the folder.
  return AAssetDir_getNextFileName(dir.get()) != NULL;
}

Status AssetManagerFileSystem::GetMatchingPaths(const string& pattern,
                                                TransactionToken* token,
                                                std::vector<string>* results) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_22(mht_22_v, 505, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::GetMatchingPaths");

  return internal::GetMatchingPaths(this, Env::Default(), pattern, results);
}

Status AssetManagerFileSystem::NewWritableFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_23(mht_23_v, 515, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::NewWritableFile");

  return errors::Unimplemented("Asset storage is read only.");
}
Status AssetManagerFileSystem::NewAppendableFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_24(mht_24_v, 524, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::NewAppendableFile");

  return errors::Unimplemented("Asset storage is read only.");
}
Status AssetManagerFileSystem::DeleteFile(const string& f,
                                          TransactionToken* token) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("f: \"" + f + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_25(mht_25_v, 532, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::DeleteFile");

  return errors::Unimplemented("Asset storage is read only.");
}
Status AssetManagerFileSystem::CreateDir(const string& d,
                                         TransactionToken* token) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("d: \"" + d + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_26(mht_26_v, 540, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::CreateDir");

  return errors::Unimplemented("Asset storage is read only.");
}
Status AssetManagerFileSystem::DeleteDir(const string& d,
                                         TransactionToken* token) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("d: \"" + d + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_27(mht_27_v, 548, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::DeleteDir");

  return errors::Unimplemented("Asset storage is read only.");
}
Status AssetManagerFileSystem::RenameFile(const string& s, const string& t,
                                          TransactionToken* token) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("s: \"" + s + "\"");
   mht_28_v.push_back("t: \"" + t + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPSinference_interfacePSasset_manager_filesystemDTcc mht_28(mht_28_v, 557, "", "./tensorflow/tools/android/inference_interface/asset_manager_filesystem.cc", "AssetManagerFileSystem::RenameFile");

  return errors::Unimplemented("Asset storage is read only.");
}

}  // namespace tensorflow
