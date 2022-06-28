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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/serialization.h"

#if defined(_WIN32)
#include <fstream>
#include <iostream>
#else
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <cstring>
#endif  // defined(_WIN32)

#include <time.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/minimal_logging.h"
#include <farmhash.h>

namespace tflite {
namespace delegates {
namespace {

static const char kDelegatedNodesSuffix[] = "_dnodes";

// Farmhash Fingerprint
inline uint64_t CombineFingerprints(uint64_t l, uint64_t h) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTcc mht_0(mht_0_v, 217, "", "./tensorflow/lite/delegates/serialization.cc", "CombineFingerprints");

  // Murmur-inspired hashing.
  const uint64_t kMul = 0x9ddfea08eb382d69ULL;
  uint64_t a = (l ^ h) * kMul;
  a ^= (a >> 47);
  uint64_t b = (h ^ a) * kMul;
  b ^= (b >> 44);
  b *= kMul;
  b ^= (b >> 41);
  b *= kMul;
  return b;
}

inline std::string JoinPath(const std::string& path1,
                            const std::string& path2) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("path1: \"" + path1 + "\"");
   mht_1_v.push_back("path2: \"" + path2 + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTcc mht_1(mht_1_v, 236, "", "./tensorflow/lite/delegates/serialization.cc", "JoinPath");

  return (path1.back() == '/') ? (path1 + path2) : (path1 + "/" + path2);
}

inline std::string GetFilePath(const std::string& cache_dir,
                               const std::string& model_token,
                               const uint64_t fingerprint) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("cache_dir: \"" + cache_dir + "\"");
   mht_2_v.push_back("model_token: \"" + model_token + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTcc mht_2(mht_2_v, 247, "", "./tensorflow/lite/delegates/serialization.cc", "GetFilePath");

  auto file_name = (model_token + "_" + std::to_string(fingerprint) + ".bin");
  return JoinPath(cache_dir, file_name);
}

}  // namespace

std::string StrFingerprint(const void* data, const size_t num_bytes) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTcc mht_3(mht_3_v, 257, "", "./tensorflow/lite/delegates/serialization.cc", "StrFingerprint");

  return std::to_string(
      ::util::Fingerprint64(reinterpret_cast<const char*>(data), num_bytes));
}

SerializationEntry::SerializationEntry(const std::string& cache_dir,
                                       const std::string& model_token,
                                       const uint64_t fingerprint)
    : cache_dir_(cache_dir),
      model_token_(model_token),
      fingerprint_(fingerprint) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("cache_dir: \"" + cache_dir + "\"");
   mht_4_v.push_back("model_token: \"" + model_token + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTcc mht_4(mht_4_v, 272, "", "./tensorflow/lite/delegates/serialization.cc", "SerializationEntry::SerializationEntry");
}

TfLiteStatus SerializationEntry::SetData(TfLiteContext* context,
                                         const char* data,
                                         const size_t size) const {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTcc mht_5(mht_5_v, 280, "", "./tensorflow/lite/delegates/serialization.cc", "SerializationEntry::SetData");

  auto filepath = GetFilePath(cache_dir_, model_token_, fingerprint_);
  // Temporary file to write data to.
  const std::string temp_filepath =
      JoinPath(cache_dir_, (model_token_ + std::to_string(fingerprint_) +
                            std::to_string(time(nullptr))));

#if defined(_WIN32)
  std::ofstream out_file(temp_filepath.c_str());
  if (!out_file) {
    TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Could not create file: %s",
                    temp_filepath.c_str());
    return kTfLiteDelegateDataWriteError;
  }
  out_file.write(data, size);
  out_file.flush();
  out_file.close();
  // rename is an atomic operation in most systems.
  if (rename(temp_filepath.c_str(), filepath.c_str()) < 0) {
    TF_LITE_KERNEL_LOG(context, "Failed to rename to %s", filepath.c_str());
    return kTfLiteDelegateDataWriteError;
  }
#else   // !defined(_WIN32)
  // This method only works on unix/POSIX systems.
  const int fd = open(temp_filepath.c_str(),
                      O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600);
  if (fd < 0) {
    TF_LITE_KERNEL_LOG(context, "Failed to open for writing: %s",
                       temp_filepath.c_str());
    return kTfLiteDelegateDataWriteError;
  }
  // Loop until all bytes written.
  ssize_t len = 0;
  const char* buf = data;
  do {
    ssize_t ret = write(fd, buf, size);
    if (ret <= 0) {
      close(fd);
      TF_LITE_KERNEL_LOG(context, "Failed to write data to: %s, error: %s",
                         temp_filepath.c_str(), std::strerror(errno));
      return kTfLiteDelegateDataWriteError;
    }

    len += ret;
    buf += ret;
  } while (len < static_cast<ssize_t>(size));
  // Use fsync to ensure data is on disk before renaming temp file.
  if (fsync(fd) < 0) {
    close(fd);
    TF_LITE_KERNEL_LOG(context, "Could not fsync: %s, error: %s",
                       temp_filepath.c_str(), std::strerror(errno));
    return kTfLiteDelegateDataWriteError;
  }
  if (close(fd) < 0) {
    TF_LITE_KERNEL_LOG(context, "Could not close fd: %s, error: %s",
                       temp_filepath.c_str(), std::strerror(errno));
    return kTfLiteDelegateDataWriteError;
  }
  if (rename(temp_filepath.c_str(), filepath.c_str()) < 0) {
    TF_LITE_KERNEL_LOG(context, "Failed to rename to %s, error: %s",
                       filepath.c_str(), std::strerror(errno));
    return kTfLiteDelegateDataWriteError;
  }
#endif  // defined(_WIN32)

  TFLITE_LOG(TFLITE_LOG_INFO, "Wrote serialized data for model %s (%d B) to %s",
             model_token_.c_str(), size, filepath.c_str());

  return kTfLiteOk;
}

TfLiteStatus SerializationEntry::GetData(TfLiteContext* context,
                                         std::string* data) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTcc mht_6(mht_6_v, 355, "", "./tensorflow/lite/delegates/serialization.cc", "SerializationEntry::GetData");

  if (!data) return kTfLiteError;
  auto filepath = GetFilePath(cache_dir_, model_token_, fingerprint_);

#if defined(_WIN32)
  std::ifstream cache_stream(filepath,
                             std::ios_base::in | std::ios_base::binary);
  if (cache_stream.good()) {
    cache_stream.seekg(0, cache_stream.end);
    int cache_size = cache_stream.tellg();
    cache_stream.seekg(0, cache_stream.beg);

    data->resize(cache_size);
    cache_stream.read(&(*data)[0], cache_size);
    cache_stream.close();
  }
#else   // !defined(_WIN32)
  // This method only works on unix/POSIX systems, but is more optimized & has
  // lower size overhead for Android binaries.
  data->clear();
  // O_CLOEXEC is needed for correctness, as another thread may call
  // popen() and the callee inherit the lock if it's not O_CLOEXEC.
  int fd = open(filepath.c_str(), O_RDONLY | O_CLOEXEC, 0600);
  if (fd < 0) {
    TF_LITE_KERNEL_LOG(context, "File %s couldn't be opened for reading: %s",
                       filepath.c_str(), std::strerror(errno));
    return kTfLiteDelegateDataNotFound;
  }
  int lock_status = flock(fd, LOCK_EX);
  if (lock_status < 0) {
    close(fd);
    TF_LITE_KERNEL_LOG(context, "Could not flock %s: %s", filepath.c_str(),
                       std::strerror(errno));
    return kTfLiteDelegateDataReadError;
  }
  char buffer[512];
  while (true) {
    int bytes_read = read(fd, buffer, 512);
    if (bytes_read == 0) {
      // EOF
      close(fd);
      return kTfLiteOk;
    } else if (bytes_read < 0) {
      close(fd);
      TF_LITE_KERNEL_LOG(context, "Error reading %s: %s", filepath.c_str(),
                         std::strerror(errno));
      return kTfLiteDelegateDataReadError;
    } else {
      data->append(buffer, bytes_read);
    }
  }
#endif  // defined(_WIN32)

  TFLITE_LOG_PROD(TFLITE_LOG_INFO,
                  "Found serialized data for model %s (%d B) at %s",
                  model_token_.c_str(), data->size(), filepath.c_str());

  if (!data->empty()) {
    TFLITE_LOG(TFLITE_LOG_INFO, "Data found at %s: %d bytes", filepath.c_str(),
               data->size());
    return kTfLiteOk;
  } else {
    TF_LITE_KERNEL_LOG(context, "No serialized data found: %s",
                       filepath.c_str());
    return kTfLiteDelegateDataNotFound;
  }
}

SerializationEntry Serialization::GetEntryImpl(
    const std::string& custom_key, TfLiteContext* context,
    const TfLiteDelegateParams* delegate_params) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("custom_key: \"" + custom_key + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTcc mht_7(mht_7_v, 429, "", "./tensorflow/lite/delegates/serialization.cc", "Serialization::GetEntryImpl");

  // First incorporate model_token.
  // We use Fingerprint64 instead of std::hash, since the latter isn't
  // guaranteed to be stable across runs. See b/172237993.
  uint64_t fingerprint =
      ::util::Fingerprint64(model_token_.c_str(), model_token_.size());

  // Incorporate custom_key.
  const uint64_t custom_str_fingerprint =
      ::util::Fingerprint64(custom_key.c_str(), custom_key.size());
  fingerprint = CombineFingerprints(fingerprint, custom_str_fingerprint);

  // Incorporate context details, if provided.
  // A quick heuristic involving graph tensors to 'fingerprint' a
  // tflite::Subgraph. We don't consider the execution plan, since it could be
  // in flux if the delegate uses this method during
  // ReplaceNodeSubsetsWithDelegateKernels (eg in kernel Init).
  if (context) {
    std::vector<int32_t> context_data;
    // Number of tensors can be large.
    const int tensors_to_consider = std::min<int>(context->tensors_size, 100);
    context_data.reserve(1 + tensors_to_consider);
    context_data.push_back(context->tensors_size);
    for (int i = 0; i < tensors_to_consider; ++i) {
      context_data.push_back(context->tensors[i].bytes);
    }
    const uint64_t context_fingerprint =
        ::util::Fingerprint64(reinterpret_cast<char*>(context_data.data()),
                                context_data.size() * sizeof(int32_t));
    fingerprint = CombineFingerprints(fingerprint, context_fingerprint);
  }

  // Incorporate delegated partition details, if provided.
  // A quick heuristic that considers the nodes & I/O tensor sizes to
  // fingerprint TfLiteDelegateParams.
  if (delegate_params) {
    std::vector<int32_t> partition_data;
    auto* nodes = delegate_params->nodes_to_replace;
    auto* input_tensors = delegate_params->input_tensors;
    auto* output_tensors = delegate_params->output_tensors;
    partition_data.reserve(nodes->size + input_tensors->size +
                           output_tensors->size);
    partition_data.insert(partition_data.end(), nodes->data,
                          nodes->data + nodes->size);
    for (int i = 0; i < input_tensors->size; ++i) {
      auto& tensor = context->tensors[input_tensors->data[i]];
      partition_data.push_back(tensor.bytes);
    }
    for (int i = 0; i < output_tensors->size; ++i) {
      auto& tensor = context->tensors[output_tensors->data[i]];
      partition_data.push_back(tensor.bytes);
    }
    const uint64_t partition_fingerprint =
        ::util::Fingerprint64(reinterpret_cast<char*>(partition_data.data()),
                                partition_data.size() * sizeof(int32_t));
    fingerprint = CombineFingerprints(fingerprint, partition_fingerprint);
  }

  // Get a fingerprint-specific lock that is passed to the SerializationKey, to
  // ensure noone else gets access to an equivalent SerializationKey.
  return SerializationEntry(cache_dir_, model_token_, fingerprint);
}

TfLiteStatus SaveDelegatedNodes(TfLiteContext* context,
                                Serialization* serialization,
                                const std::string& delegate_id,
                                const TfLiteIntArray* node_ids) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("delegate_id: \"" + delegate_id + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTcc mht_8(mht_8_v, 499, "", "./tensorflow/lite/delegates/serialization.cc", "SaveDelegatedNodes");

  if (!node_ids) return kTfLiteError;
  std::string cache_key = delegate_id + kDelegatedNodesSuffix;
  auto entry = serialization->GetEntryForDelegate(cache_key, context);
  return entry.SetData(context, reinterpret_cast<const char*>(node_ids),
                       (1 + node_ids->size) * sizeof(int));
}

TfLiteStatus GetDelegatedNodes(TfLiteContext* context,
                               Serialization* serialization,
                               const std::string& delegate_id,
                               TfLiteIntArray** node_ids) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("delegate_id: \"" + delegate_id + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTcc mht_9(mht_9_v, 514, "", "./tensorflow/lite/delegates/serialization.cc", "GetDelegatedNodes");

  if (!node_ids) return kTfLiteError;
  std::string cache_key = delegate_id + kDelegatedNodesSuffix;
  auto entry = serialization->GetEntryForDelegate(cache_key, context);

  std::string read_buffer;
  TF_LITE_ENSURE_STATUS(entry.GetData(context, &read_buffer));
  if (read_buffer.empty()) return kTfLiteOk;
  *node_ids = TfLiteIntArrayCopy(
      reinterpret_cast<const TfLiteIntArray*>(read_buffer.data()));
  return kTfLiteOk;
}

}  // namespace delegates
}  // namespace tflite
