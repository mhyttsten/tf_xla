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
class MHTracer_DTPStensorflowPScorePSutilPSutilDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSutilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSutilDTcc() {
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

#include "tensorflow/core/util/util.h"

#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

StringPiece NodeNamePrefix(const StringPiece& op_name) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSutilDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/util/util.cc", "NodeNamePrefix");

  StringPiece sp(op_name);
  auto p = sp.find('/');
  if (p == StringPiece::npos || p == 0) {
    return "";
  } else {
    return StringPiece(sp.data(), p);
  }
}

StringPiece NodeNameFullPrefix(const StringPiece& op_name) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSutilDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/util/util.cc", "NodeNameFullPrefix");

  StringPiece sp(op_name);
  auto p = sp.rfind('/');
  if (p == StringPiece::npos || p == 0) {
    return "";
  } else {
    return StringPiece(sp.data(), p);
  }
}

MovingAverage::MovingAverage(int window)
    : window_(window),
      sum_(0.0),
      data_(new double[window_]),
      head_(0),
      count_(0) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSutilDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/util/util.cc", "MovingAverage::MovingAverage");

  CHECK_GE(window, 1);
}

MovingAverage::~MovingAverage() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSutilDTcc mht_3(mht_3_v, 238, "", "./tensorflow/core/util/util.cc", "MovingAverage::~MovingAverage");
 delete[] data_; }

void MovingAverage::Clear() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSutilDTcc mht_4(mht_4_v, 243, "", "./tensorflow/core/util/util.cc", "MovingAverage::Clear");

  count_ = 0;
  head_ = 0;
  sum_ = 0;
}

double MovingAverage::GetAverage() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSutilDTcc mht_5(mht_5_v, 252, "", "./tensorflow/core/util/util.cc", "MovingAverage::GetAverage");

  if (count_ == 0) {
    return 0;
  } else {
    return static_cast<double>(sum_) / count_;
  }
}

void MovingAverage::AddValue(double v) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSutilDTcc mht_6(mht_6_v, 263, "", "./tensorflow/core/util/util.cc", "MovingAverage::AddValue");

  if (count_ < window_) {
    // This is the warmup phase. We don't have a full window's worth of data.
    head_ = count_;
    data_[count_++] = v;
  } else {
    if (window_ == ++head_) {
      head_ = 0;
    }
    // Toss the oldest element
    sum_ -= data_[head_];
    // Add the newest element
    data_[head_] = v;
  }
  sum_ += v;
}

static char hex_char[] = "0123456789abcdef";

string PrintMemory(const char* ptr, size_t n) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("ptr: \"" + (ptr == nullptr ? std::string("nullptr") : std::string((char*)ptr)) + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSutilDTcc mht_7(mht_7_v, 286, "", "./tensorflow/core/util/util.cc", "PrintMemory");

  string ret;
  ret.resize(n * 3);
  for (int i = 0; i < n; ++i) {
    ret[i * 3] = ' ';
    ret[i * 3 + 1] = hex_char[ptr[i] >> 4];
    ret[i * 3 + 2] = hex_char[ptr[i] & 0xf];
  }
  return ret;
}

string SliceDebugString(const TensorShape& shape, const int64_t flat) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSutilDTcc mht_8(mht_8_v, 300, "", "./tensorflow/core/util/util.cc", "SliceDebugString");

  // Special case rank 0 and 1
  const int dims = shape.dims();
  if (dims == 0) return "";
  if (dims == 1) return strings::StrCat("[", flat, "]");

  // Compute strides
  gtl::InlinedVector<int64_t, 32> strides(dims);
  strides.back() = 1;
  for (int i = dims - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape.dim_size(i + 1);
  }

  // Unflatten index
  int64_t left = flat;
  string result;
  for (int i = 0; i < dims; i++) {
    strings::StrAppend(&result, i ? "," : "[", left / strides[i]);
    left %= strides[i];
  }
  strings::StrAppend(&result, "]");
  return result;
}

bool IsMKLEnabled() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSutilDTcc mht_9(mht_9_v, 327, "", "./tensorflow/core/util/util.cc", "IsMKLEnabled");

#ifndef INTEL_MKL
  return false;
#endif  // !INTEL_MKL
  static absl::once_flag once;
#ifdef ENABLE_MKL
  // Keeping TF_DISABLE_MKL env variable for legacy reasons.
  static bool oneDNN_disabled = false;
  absl::call_once(once, [&] {
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_DISABLE_MKL", false, &oneDNN_disabled));
    if (oneDNN_disabled) VLOG(2) << "TF-MKL: Disabling oneDNN";
  });
  return (!oneDNN_disabled);
#else
  // Linux: Turn oneDNN on by default for CPUs with neural network features.
  // Windows: oneDNN is off by default.
  // No need to guard for other platforms here because INTEL_MKL is only defined
  // for non-mobile Linux or Windows.
  static bool oneDNN_enabled =
#ifdef __linux__
      port::TestCPUFeature(port::CPUFeature::AVX512_VNNI) ||
      port::TestCPUFeature(port::CPUFeature::AVX512_BF16) ||
      port::TestCPUFeature(port::CPUFeature::AVX_VNNI) ||
      port::TestCPUFeature(port::CPUFeature::AMX_TILE) ||
      port::TestCPUFeature(port::CPUFeature::AMX_INT8) ||
      port::TestCPUFeature(port::CPUFeature::AMX_BF16);
#else
      false;
#endif  // __linux__
  absl::call_once(once, [&] {
    auto status = ReadBoolFromEnvVar("TF_ENABLE_ONEDNN_OPTS", oneDNN_enabled,
                                     &oneDNN_enabled);
    if (!status.ok()) {
      LOG(WARNING) << "TF_ENABLE_ONEDNN_OPTS is not set to either '0', 'false',"
                   << " '1', or 'true'. Using the default setting: "
                   << oneDNN_enabled;
    }
    if (oneDNN_enabled) {
#ifndef DNNL_AARCH64_USE_ACL
      LOG(INFO) << "oneDNN custom operations are on. "
                << "You may see slightly different numerical results due to "
                << "floating-point round-off errors from different computation "
                << "orders. To turn them off, set the environment variable "
                << "`TF_ENABLE_ONEDNN_OPTS=0`.";
#else
      LOG(INFO) << "Experimental oneDNN custom operations are on. "
                << "If you experience issues, please turn them off by setting "
                << "the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.";
#endif  // !DNNL_AARCH64_USE_ACL
    }
  });
  return oneDNN_enabled;
#endif  // ENABLE_MKL
}

}  // namespace tensorflow
