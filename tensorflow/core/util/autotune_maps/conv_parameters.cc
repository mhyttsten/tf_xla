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
class MHTracer_DTPStensorflowPScorePSutilPSautotune_mapsPSconv_parametersDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSautotune_mapsPSconv_parametersDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSautotune_mapsPSconv_parametersDTcc() {
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/autotune_maps/conv_parameters.h"

#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/core/platform/hash.h"
#include "tensorflow/core/util/autotune_maps/autotune_maps_utils.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.pb.h"

namespace tensorflow {

namespace {
using ::tensorflow::protobuf::util::MessageDifferencer;

uint64 ComputeHash(int device_id, const ConvParametersProto& proto) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSautotune_mapsPSconv_parametersDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/util/autotune_maps/conv_parameters.cc", "ComputeHash");

  return Hash64Combine(device_id, autotune_maps_utils::HashProto(proto));
}
}  // namespace

ConvParameters::ConvParameters(
    int64_t batch, int64_t in_depths, const absl::Span<const int64_t> in,
    int data_format, int64_t out_depths, const absl::Span<const int64_t> filter,
    const absl::Span<const int64_t> dilation,
    const absl::Span<const int64_t> stride,
    const absl::Span<const int64_t> padding, DataType dtype, int device_id,
    int group_count, absl::optional<ConvParameters::FusionInfo> fusion_info,
    int version)
    : device_id_(device_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSautotune_mapsPSconv_parametersDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/util/autotune_maps/conv_parameters.cc", "ConvParameters::ConvParameters");

  proto_.set_batch(batch);
  proto_.set_in_depths(in_depths);
  *proto_.mutable_in() = {in.begin(), in.end()};
  proto_.set_data_format(static_cast<int>(data_format));
  proto_.set_out_depths(out_depths);
  *proto_.mutable_filter() = {filter.begin(), filter.end()};
  *proto_.mutable_dilation() = {dilation.begin(), dilation.end()};
  *proto_.mutable_stride() = {stride.begin(), stride.end()};
  *proto_.mutable_padding() = {padding.begin(), padding.end()};
  proto_.set_dtype(dtype);
  proto_.set_group_count(group_count);
  if (fusion_info.has_value()) {
    ConvParametersProto::Fusion fusion_proto;
    fusion_proto.set_conv_scale(fusion_info.value().conv_scale);
    fusion_proto.set_side_input_scale(fusion_info.value().side_input_scale);
    fusion_proto.set_activation_mode(fusion_info.value().activation_mode);
    fusion_proto.set_is_contrib(fusion_info.value().is_contrib);
    *proto_.mutable_fusion() = fusion_proto;
  }
  proto_.set_device_identifier(
      autotune_maps_utils::DeviceIdToIdentifier(device_id));
  proto_.set_version(version);
  hash_code_ = ComputeHash(device_id_, proto_);
}

ConvParameters::ConvParameters(int device_id, const ConvParametersProto& proto)
    : device_id_(device_id),
      proto_(proto),
      hash_code_(ComputeHash(device_id, proto_)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSautotune_mapsPSconv_parametersDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/util/autotune_maps/conv_parameters.cc", "ConvParameters::ConvParameters");
}

bool ConvParameters::operator==(const ConvParameters& other) const {
  return device_id_ == other.device_id_ &&
         MessageDifferencer::Equals(this->proto_, other.proto_);
}

string ConvParameters::ToString() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSautotune_mapsPSconv_parametersDTcc mht_3(mht_3_v, 258, "", "./tensorflow/core/util/autotune_maps/conv_parameters.cc", "ConvParameters::ToString");
 return proto_.DebugString(); }

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
