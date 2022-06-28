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

#ifndef TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_CONV_PARAMETERS_H_
#define TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_CONV_PARAMETERS_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSautotune_mapsPSconv_parametersDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSautotune_mapsPSconv_parametersDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSautotune_mapsPSconv_parametersDTh() {
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


#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "absl/types/optional.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.pb.h"

namespace tensorflow {
// Uniquely identifies a convolution operation that runs on a particular device
// model.
//
// This can serve as a hashtable key, where the value might be the autotuned
// algorithm we choose for the conv.
//
// All of the data in this class other than the device_id is stored in the
// ConvParametersProto, so it can be easily serialized (for the purposes of
// ahead-of-time autotuning).
//
// When using the cudnn frontend API, two autotuning results for two different
// GPUs of the same model are not interchangeable, because an autotuning result
// includes a cudnn execution plan, which is tied to the GPU.  As a result, we
// need to create separate ConvParameters objects for them.
class ConvParameters {
 public:
  struct FusionInfo {
    // For some implementations (e.g. cuDNN new backend) these scales are part
    // of the algorithm, not part of the parameters an algorithm take. They need
    // to be used to distinguish different algorithms.
    double conv_scale;
    double side_input_scale;
    stream_executor::dnn::ActivationMode activation_mode;
    bool is_contrib;
  };

  // LINT.IfChange(conv_parameters_version)
  // A positive number that denotes the version of this class. Should be
  // incremented everytime this class or ConvParametersProto are updated in a
  // way that may invalidate autotune results.
  static constexpr int kVersion = 1;
  // LINT.ThenChange()

  // We have three kinds of convolutions today.  Vanilla unfused convolutions,
  // fused convolutions, and fused convolutions as implemented in the `contrib`
  // directory.  The two fused convolutions ultimately correspond to the same
  // cudnn calls, but have slightly different semantics (e.g. they interpret
  // padding differently).
  ConvParameters(
      int64_t batch, int64_t in_depths, absl::Span<const int64_t> in,
      int data_format, int64_t out_depths, absl::Span<const int64_t> filter,
      absl::Span<const int64_t> dilation, absl::Span<const int64_t> stride,
      absl::Span<const int64_t> padding, DataType dtype, int device_id,
      int group_count,
      absl::optional<FusionInfo> fusion_info = absl::optional<FusionInfo>(),
      // This argument should be set only for test use.
      int version = kVersion);

  ConvParameters(int device_id, const ConvParametersProto& proto);

  bool operator==(const ConvParameters& other) const;

  bool operator!=(const ConvParameters& other) const {
    return !(*this == other);
  }
  uint64 hash() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSautotune_mapsPSconv_parametersDTh mht_0(mht_0_v, 249, "", "./tensorflow/core/util/autotune_maps/conv_parameters.h", "hash");
 return hash_code_; }

  string ToString() const;

  const ConvParametersProto& proto() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSautotune_mapsPSconv_parametersDTh mht_1(mht_1_v, 256, "", "./tensorflow/core/util/autotune_maps/conv_parameters.h", "proto");
 return proto_; }

 private:
  int device_id_;
  ConvParametersProto proto_;
  uint64 hash_code_;
};
}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_CONV_PARAMETERS_H_
