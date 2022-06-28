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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/segment/union_find.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace segment {

namespace {
template <typename T>
inline bool CheckIfCompatible(const absl::optional<T>& a,
                              const absl::optional<T>& b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.cc", "CheckIfCompatible");

  if (a.has_value() && b.has_value()) {
    return *a == *b;
  }
  return true;
}

template <typename T>
inline bool UnifyValues(absl::optional<T>& a, absl::optional<T>& b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc mht_1(mht_1_v, 210, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.cc", "UnifyValues");

  if (a.has_value()) {
    b = a;
  } else {
    a = b;
  }
  return true;
}

template <typename T>
inline absl::optional<T> MergeCompatible(const absl::optional<T>& a,
                                         const absl::optional<T>& b) {
  DCHECK(CheckIfCompatible(a, b));
  return a.has_value() ? a : b;
}

}  // namespace

ClusterBatchSize::ClusterBatchSize()
    : batch_size_(absl::nullopt), max_batch_size_(absl::nullopt) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc mht_2(mht_2_v, 232, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.cc", "ClusterBatchSize::ClusterBatchSize");
}

bool ClusterBatchSize::operator==(const ClusterBatchSize& other) {
  return batch_size_ == other.batch_size_ &&
         max_batch_size_ == other.max_batch_size_;
}

ClusterBatchSize& ClusterBatchSize::SetBatchSize(int batch_size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc mht_3(mht_3_v, 242, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.cc", "ClusterBatchSize::SetBatchSize");

  SetBatchSize(static_cast<absl::optional<int>>(batch_size));
  return *this;
}

ClusterBatchSize& ClusterBatchSize::SetBatchSize(
    const absl::optional<int>& batch_size) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc mht_4(mht_4_v, 251, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.cc", "ClusterBatchSize::SetBatchSize");

  batch_size_ = MergeCompatible<int>(batch_size_, batch_size);
  if (batch_size_.has_value() && batch_size_.value() >= 0) {
    SetMaxBatchSize(batch_size_);
  }
  return *this;
}

bool ClusterBatchSize::HasBatchSize() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc mht_5(mht_5_v, 262, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.cc", "ClusterBatchSize::HasBatchSize");
 return batch_size_.has_value(); }

int ClusterBatchSize::GetBatchSize() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc mht_6(mht_6_v, 267, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.cc", "ClusterBatchSize::GetBatchSize");

  DCHECK(HasBatchSize());
  return batch_size_.value();
}

ClusterBatchSize& ClusterBatchSize::SetMaxBatchSize(int max_batch_size) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc mht_7(mht_7_v, 275, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.cc", "ClusterBatchSize::SetMaxBatchSize");

  SetBatchSize(static_cast<absl::optional<int>>(max_batch_size));
  return *this;
}

ClusterBatchSize& ClusterBatchSize::SetMaxBatchSize(
    const absl::optional<int>& max_batch_size) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc mht_8(mht_8_v, 284, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.cc", "ClusterBatchSize::SetMaxBatchSize");

  max_batch_size_ = MergeCompatible<int>(max_batch_size_, max_batch_size);
  return *this;
}

absl::optional<int> ClusterBatchSize::GetOptionalMaxBatchSize() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc mht_9(mht_9_v, 292, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.cc", "ClusterBatchSize::GetOptionalMaxBatchSize");

  return max_batch_size_;
}

bool ClusterBatchSize::MergeIfCompatible(const ClusterBatchSize& other) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc mht_10(mht_10_v, 299, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.cc", "ClusterBatchSize::MergeIfCompatible");

  if (!CheckIfCompatible(batch_size_, other.batch_size_) ||
      !CheckIfCompatible(max_batch_size_, other.max_batch_size_)) {
    return false;
  }

  SetBatchSize(other.batch_size_);
  SetMaxBatchSize(other.max_batch_size_);
  return true;
}

string ClusterBatchSize::ToString() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc mht_11(mht_11_v, 313, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.cc", "ClusterBatchSize::ToString");

  string s;
  const auto append_optional_num = [&](const absl::optional<int>& num) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc mht_12(mht_12_v, 318, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.cc", "lambda");

    if (num.has_value()) {
      absl::StrAppendFormat(&s, "%d", num.value());
    } else {
      absl::StrAppendFormat(&s, "?");
    }
  };
  absl::StrAppendFormat(&s, "batch_size=");
  append_optional_num(batch_size_);
  absl::StrAppendFormat(&s, ", max_batch_size=");
  append_optional_num(max_batch_size_);
  return s;
}

ClusterProperty::ClusterProperty(const ClusterBatchSize& batch_size,
                                 const DeviceNameUtils::ParsedName& device_name)
    : batch_size_(batch_size), device_name_(device_name) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc mht_13(mht_13_v, 337, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.cc", "ClusterProperty::ClusterProperty");
}

Status ClusterProperty::Merge(const ClusterProperty& other) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTcc mht_14(mht_14_v, 342, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.cc", "ClusterProperty::Merge");

  ClusterBatchSize merged_batch_size(batch_size_);
  if (!merged_batch_size.MergeIfCompatible(other.batch_size_)) {
    return errors::Internal(
        "trying to merge clusters with incompatible batch sizes.");
  }

  absl::optional<DeviceNameUtils::ParsedName> merged_device_name =
      MergeIfCompatible(device_name_, other.device_name_);
  if (!merged_device_name.has_value()) {
    return errors::Internal(
        "trying to merge clusters with incompatible device assignment.");
  }

  batch_size_ = std::move(merged_batch_size);
  device_name_ = std::move(merged_device_name.value());
  return Status::OK();
}

}  // namespace segment
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
