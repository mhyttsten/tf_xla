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
#ifndef TENSORFLOW_LITE_DELEGATES_SERIALIZATION_H_
#define TENSORFLOW_LITE_DELEGATES_SERIALIZATION_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTh() {
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


#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/common.h"

// This file implements a serialization utility that TFLite delegates can use to
// read/write initialization data.
//
// Example code:
//
// Initialization
// ==============
// SerializationParams params;
// // Acts as a namespace for all data entries for a given model.
// // See StrFingerprint().
// params.model_token = options->model_token;
// // Location where data is stored, should be private to the app using this.
// params.serialization_dir = options->serialization_dir;
// Serialization serialization(params);
//
// Writing data
// ============
// TfLiteContext* context = ...;
// TfLiteDelegateParams* params = ...;
// SerializationEntry kernels_entry = serialization->GetEntryForKernel(
//     "gpuv2_kernels", context, delegate_params);
//
// TfLiteStatus kernels_save_status = kernels_entry.SetData(
//     reinterpret_cast<char*>(data_ptr),
//     data_size);
// if (kernels_save_status == kTfLiteOk) {
//   //...serialization successful...
// } else if (kernels_save_status == kTfLiteDelegateDataWriteError) {
//   //...error in serializing data to disk...
// } else {
//   //...unexpected error...
// }
//
// Reading data
// ============
// std::string kernels_data;
// TfLiteStatus kernels_data_status = kernels_entry.GetData(&kernels_data);
// if (kernels_data_status == kTfLiteOk) {
//   //...serialized data found...
// } else if (kernels_data_status == kTfLiteDelegateDataNotFound) {
//   //...serialized data missing...
// } else {
//   //...unexpected error...
// }
namespace tflite {
namespace delegates {

// Helper to generate a unique string (converted from 64-bit farmhash) given
// some data. Intended for use by:
//
// 1. Delegates, to 'fingerprint' some custom data (like options),
//    and provide it as custom_key to Serialization::GetEntryForDelegate or
//    GetEntryForKernel.
// 2. TFLite clients, to fingerprint a model flatbuffer & get a unique
//    model_token.
std::string StrFingerprint(const void* data, const size_t num_bytes);

// Encapsulates a unique blob of data serialized by a delegate.
// Needs to be initialized with a Serialization instance.
// Any data set with this entry is 'keyed' by a 64-bit fingerprint unique to the
// parameters used during initialization via
// Serialization::GetEntryForDelegate/GetEntryForKernel.
//
// NOTE: TFLite cannot guarantee that the read data is always fully valid,
// especially if the directory is accessible to other applications/processes.
// It is the delegate's responsibility to validate the retrieved data.
class SerializationEntry {
 public:
  friend class Serialization;

  // Returns a 64-bit fingerprint unique to the parameters provided during the
  // generation of this SerializationEntry.
  // Produces same value on every run.
  uint64_t GetFingerprint() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTh mht_0(mht_0_v, 268, "", "./tensorflow/lite/delegates/serialization.h", "GetFingerprint");
 return fingerprint_; }

  // Stores `data` into a file that is unique to this SerializationKey.
  // Overwrites any existing data if present.
  //
  // Returns:
  //   kTfLiteOk if data is successfully stored
  //   kTfLiteDelegateDataWriteError for data writing issues
  //   kTfLiteError for unexpected error.
  //
  // NOTE: We use a temp file & rename it as file renaming is an atomic
  // operation in most systems.
  TfLiteStatus SetData(TfLiteContext* context, const char* data,
                       const size_t size) const;

  // Get `data` corresponding to this key, if available.
  //
  // Returns:
  //   kTfLiteOk if data is successfully stored
  //   kTfLiteDataError for data writing issues
  //   kTfLiteError for unexpected error.
  TfLiteStatus GetData(TfLiteContext* context, std::string* data) const;

  // Non-copyable.
  SerializationEntry(const SerializationEntry&) = delete;
  SerializationEntry& operator=(const SerializationEntry&) = delete;
  SerializationEntry(SerializationEntry&& src) = default;

 protected:
  SerializationEntry(const std::string& cache_dir,
                     const std::string& model_token,
                     const uint64_t fingerprint_64);

  // Caching directory.
  const std::string cache_dir_;
  // Model Token.
  const std::string model_token_;
  // For most applications, 64-bit fingerprints are enough.
  const uint64_t fingerprint_ = 0;
};

// Encapsulates all the data that clients can use to parametrize a Serialization
// interface.
typedef struct SerializationParams {
  // Acts as a 'namespace' for all SerializationEntry instances.
  // Clients should ensure that the token is unique to the model graph & data.
  // StrFingerprint() can be used with the flatbuffer data to generate a unique
  // 64-bit token.
  // TODO(b/190055017): Add 64-bit fingerprints to TFLite flatbuffers to ensure
  // different model constants automatically lead to different fingerprints.
  // Required.
  const char* model_token;
  // Denotes the directory to be used to store data.
  // It is the client's responsibility to ensure this location is valid and
  // application-specific to avoid unintended data access issues.
  // On Android, `getCodeCacheDir()` is recommended.
  // Required.
  const char* cache_dir;
} SerializationParams;

// Utility to enable caching abilities for delegates.
// See documentation at the top of the file for usage details.
//
// WARNING: Experimental interface, subject to change.
class Serialization {
 public:
  // Initialize a Serialization interface for applicable delegates.
  explicit Serialization(const SerializationParams& params)
      : cache_dir_(params.cache_dir), model_token_(params.model_token) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTh mht_1(mht_1_v, 339, "", "./tensorflow/lite/delegates/serialization.h", "Serialization");
}

  // Generate a SerializationEntry that incorporates both `custom_key` &
  // `context` into its unique fingerprint.
  //  Should be used to handle data common to all delegate kernels.
  // Delegates can incorporate versions & init arguments in custom_key using
  // StrFingerprint().
  SerializationEntry GetEntryForDelegate(const std::string& custom_key,
                                         TfLiteContext* context) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("custom_key: \"" + custom_key + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTh mht_2(mht_2_v, 351, "", "./tensorflow/lite/delegates/serialization.h", "GetEntryForDelegate");

    return GetEntryImpl(custom_key, context);
  }

  // Generate a SerializationEntry that incorporates `custom_key`, `context`,
  // and `delegate_params` into its unique fingerprint.
  // Should be used to handle data specific to a delegate kernel, since
  // the context+delegate_params combination is node-specific.
  // Delegates can incorporate versions & init arguments in custom_key using
  // StrFingerprint().
  SerializationEntry GetEntryForKernel(
      const std::string& custom_key, TfLiteContext* context,
      const TfLiteDelegateParams* partition_params) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("custom_key: \"" + custom_key + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSserializationDTh mht_3(mht_3_v, 367, "", "./tensorflow/lite/delegates/serialization.h", "GetEntryForKernel");

    return GetEntryImpl(custom_key, context, partition_params);
  }

  // Non-copyable.
  Serialization(const Serialization&) = delete;
  Serialization& operator=(const Serialization&) = delete;

 protected:
  SerializationEntry GetEntryImpl(
      const std::string& custom_key, TfLiteContext* context = nullptr,
      const TfLiteDelegateParams* delegate_params = nullptr);

  const std::string cache_dir_;
  const std::string model_token_;
};

// Helper for delegates to save their delegation decisions (which nodes to
// delegate) in TfLiteDelegate::Prepare().
// Internally, this uses a unique SerializationEntry based on the `context` &
// `delegate_id` to save the `node_ids`. It is recommended that `delegate_id` be
// unique to a backend/version to avoid reading back stale delegation decisions.
//
// NOTE: This implementation is platform-specific, so this method & the
// subsequent call to GetDelegatedNodes should happen on the same device.
TfLiteStatus SaveDelegatedNodes(TfLiteContext* context,
                                Serialization* serialization,
                                const std::string& delegate_id,
                                const TfLiteIntArray* node_ids);

// Retrieves list of delegated nodes that were saved earlier with
// SaveDelegatedNodes.
// Caller assumes ownership of data pointed by *nodes_ids.
//
// NOTE: This implementation is platform-specific, so SaveDelegatedNodes &
// corresponding GetDelegatedNodes should be called on the same device.
TfLiteStatus GetDelegatedNodes(TfLiteContext* context,
                               Serialization* serialization,
                               const std::string& delegate_id,
                               TfLiteIntArray** node_ids);

}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_SERIALIZATION_H_
