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
class MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc() {
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

#include "tensorflow/stream_executor/stream.h"

#include "absl/strings/str_cat.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/host_or_device_scalar.h"
#include "tensorflow/stream_executor/lib/stacktrace.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/rng.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace stream_executor {

namespace {
// Code to turn parameters to functions on stream into strings that
// will be VLOG'ed. We need overloads, instead of
// e.g. BatchDescriptorToVlogString(), as the code that calls these
// functions does not know what the type of the parameter is.
std::string ToVlogString(const dnn::BatchDescriptor &descriptor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_0(mht_0_v, 206, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  return descriptor.ToShortString();
}

std::string ToVlogString(const dnn::FilterDescriptor &descriptor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_1(mht_1_v, 213, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  return descriptor.ToShortString();
}

std::string ToVlogString(const dnn::ConvolutionDescriptor &descriptor) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_2(mht_2_v, 220, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  return descriptor.ToShortString();
}

std::string ToVlogString(const dnn::PoolingDescriptor &descriptor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_3(mht_3_v, 227, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  return descriptor.ToShortString();
}

std::string ToVlogString(const dnn::NormalizeDescriptor &descriptor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_4(mht_4_v, 234, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  return descriptor.ToShortString();
}

std::string ToVlogString(dnn::ActivationMode mode) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_5(mht_5_v, 241, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  return dnn::ActivationModeString(mode);
}

std::string ToVlogString(const dnn::AlgorithmConfig &algo_config) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_6(mht_6_v, 248, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  return algo_config.ToString();
}

std::string ToVlogString(dnn::ElementwiseOperation op) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_7(mht_7_v, 255, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  return dnn::ElementwiseOperationString(op);
}

std::string ToVlogString(dnn::QuantizedActivationMode mode) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_8(mht_8_v, 262, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  return dnn::QuantizedActivationModeString(mode);
}

std::string ToVlogString(blas::Transpose t) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_9(mht_9_v, 269, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");
 return blas::TransposeString(t); }

std::string ToVlogString(blas::UpperLower ul) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_10(mht_10_v, 274, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  return blas::UpperLowerString(ul);
}

std::string ToVlogString(blas::Diagonal d) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_11(mht_11_v, 281, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");
 return blas::DiagonalString(d); }

std::string ToVlogString(blas::Side s) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_12(mht_12_v, 286, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");
 return blas::SideString(s); }

std::string ToVlogString(blas::ComputationType ty) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_13(mht_13_v, 291, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  return blas::ComputationTypeString(ty);
}

std::string ToVlogString(const void *ptr) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_14(mht_14_v, 298, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  if (ptr == nullptr) {
    return "null";
  }

  // StrCat does not convert pointers to text.
  std::ostringstream out;
  out << ptr;
  return out.str();
}

template <class T>
std::string ToVlogString(const std::complex<T> &c) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_15(mht_15_v, 313, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  // StrCat does not convert std::complex to text.
  std::ostringstream out;
  out << c;
  return out.str();
}

template <class T>
std::string ToVlogString(const std::function<T> &f) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_16(mht_16_v, 324, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  return f == nullptr ? "null" : "<non-null function>";
}

std::string ToVlogString(const DeviceMemoryBase &memory) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_17(mht_17_v, 331, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  return ToVlogString(memory.opaque());
}

std::string ToVlogString(const DeviceMemoryBase *memory) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_18(mht_18_v, 338, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  return memory == nullptr ? "null" : ToVlogString(*memory);
}

std::string ToVlogString(const Eigen::half &h) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_19(mht_19_v, 345, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  return absl::StrCat(static_cast<float>(h));
}

std::string ToVlogString(int i) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_20(mht_20_v, 352, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");
 return absl::StrCat(i); }

std::string ToVlogString(uint32 i) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_21(mht_21_v, 357, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");
 return absl::StrCat(i); }

std::string ToVlogString(uint64_t i) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_22(mht_22_v, 362, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");
 return absl::StrCat(i); }

std::string ToVlogString(int64_t i) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_23(mht_23_v, 367, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");
 return absl::StrCat(i); }

std::string ToVlogString(float f) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_24(mht_24_v, 372, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");
 return absl::StrCat(f); }

std::string ToVlogString(double d) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_25(mht_25_v, 377, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");
 return absl::StrCat(d); }

template <typename T>
std::string ToVlogString(const HostOrDeviceScalar<T> &memory_or_constant) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_26(mht_26_v, 383, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  if (memory_or_constant.is_pointer()) {
    return ToVlogString(memory_or_constant.pointer());
  }
  return ToVlogString(memory_or_constant.value());
}

template <class T>
std::string ToVlogString(port::ArraySlice<T> elements) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_27(mht_27_v, 394, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  std::string str = absl::StrCat(
      ToVlogString(reinterpret_cast<const void *>(elements.data())), "[",
      elements.size(), "]{");
  const char *separator = "";
  size_t max_to_show = std::numeric_limits<size_t>::max();
  if (!VLOG_IS_ON(2)) {
    max_to_show = 5;
  } else if (!VLOG_IS_ON(3)) {
    max_to_show = 20;
  } else if (!VLOG_IS_ON(11)) {
    max_to_show = 1000;
  }
  for (size_t i = 0; i < elements.size(); ++i) {
    if (i == max_to_show) {
      str += ", ...";
      break;
    }
    absl::StrAppend(&str, separator, ToVlogString(elements[i]));
    separator = ", ";
  }
  str += "}";
  return str;
}

template <class T>
std::string ToVlogString(port::MutableArraySlice<T> elements) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_28(mht_28_v, 423, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  return ToVlogString(port::ArraySlice<T>(elements));
}

std::string ToVlogString(dnn::DepthToSpaceLayout depth_to_space_layout) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_29(mht_29_v, 430, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  switch (depth_to_space_layout) {
    case dnn::DepthToSpaceLayout::DepthHeightWidth:
      return "DepthToSpaceLayout::DepthHeightWidth";
  }
  return "unknown DepthToSpaceLayout";
}

std::string ToVlogString(dnn::DataType data_type) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_30(mht_30_v, 441, "", "./tensorflow/stream_executor/stream.cc", "ToVlogString");

  switch (data_type) {
    case dnn::DataType::kFloat:
      return "dnn::DataType::kFloat";
    case dnn::DataType::kDouble:
      return "dnn::DataType::kDouble";
    case dnn::DataType::kHalf:
      return "dnn::DataType::kHalf";
    case dnn::DataType::kInt8:
      return "dnn::DataType::kInt8";
    case dnn::DataType::kInt32:
      return "dnn::DataType::kInt32";
    default:
      return "unknown DataType";
  }
}

// Used together with PARAM to VLOG calls made to the stream. Intended
// to be used like this:
//
//   VLOG(1) << CallStr("MyFunction", this, {PARAM(a), PARAM(b)});
//
// where a and b are the parameters to MyFunction.
//
// See VLOG_CALL for a short-hand for this. This way of doing it saves
// a tremendous amount of boilerplate code given how many functions
// there are on Stream and how many parameters they each have.
std::string CallStr(const char *function_name, Stream *stream,
                    std::vector<std::pair<const char *, std::string>> params) {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("function_name: \"" + (function_name == nullptr ? std::string("nullptr") : std::string((char*)function_name)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_31(mht_31_v, 473, "", "./tensorflow/stream_executor/stream.cc", "CallStr");

  // Do not call this function unless VLOG is on since just
  // constructing all the strings in params is expensive.
  CHECK(VLOG_IS_ON(1));

  std::string str = absl::StrCat(stream->DebugStreamPointers(),
                                 " Called Stream::", function_name, "(");
  const char *separator = "";
  for (const auto &param : params) {
    absl::StrAppend(&str, separator, param.first, "=", param.second);
    separator = ", ";
  }
  absl::StrAppend(&str, ")");
  if (VLOG_IS_ON(10)) {
    absl::StrAppend(&str, " ", port::CurrentStackTrace(), "\n");
  }
  return str;
}

// Use this macro to avoid having to type every parameter twice to log
// it with VLOG and CallStr.
#define PARAM(parameter) \
  { #parameter, ToVlogString(parameter) }

// Use this macro to avoid having to type out the name of each
// function and to save some boilerplate. Intended to be used like this:
//
//   VLOG_CALL(PARAM(a), PARAM(b))
//
// This saves a tremendous amount of boilerplate compared to the alternative:
//
//   VLOG(1) << "Calling MyFunction(a=" << ToVlogString(a)
//           << ", b=" << ToVlogString(b);
//
// Note here that most of the parameter names are not short and that
// most of the functions take many more than 2 parameters.
#define VLOG_CALL(...) VLOG(1) << CallStr(__func__, this, {__VA_ARGS__})

}  // namespace

Stream::Stream(StreamExecutor *parent)
    : parent_(parent),
      implementation_(parent->implementation()->GetStreamImplementation()),
      allocated_(false),
      status_(port::InternalError("Uninitialized stream")),
      temporary_memory_manager_(this) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_32(mht_32_v, 521, "", "./tensorflow/stream_executor/stream.cc", "Stream::Stream");

  VLOG_CALL(PARAM(parent));
}

Stream::Stream(StreamExecutor *parent,
               internal::StreamInterface *implementation)
    : parent_(parent),
      implementation_(implementation),
      allocated_(false),
      status_(port::InternalError("Uninitialized stream")),
      temporary_memory_manager_(this) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_33(mht_33_v, 534, "", "./tensorflow/stream_executor/stream.cc", "Stream::Stream");

  VLOG_CALL(PARAM(parent), PARAM(implementation));
}

Stream::~Stream() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_34(mht_34_v, 541, "", "./tensorflow/stream_executor/stream.cc", "Stream::~Stream");

  VLOG_CALL();

  // Ensure the stream is completed.
  auto status = BlockHostUntilDone();
  if (!status.ok()) {
    LOG(WARNING) << "Error blocking host until done in stream destructor: "
                 << status;
  }
  temporary_memory_manager_.ForceDeallocateAll();
  RunAfterBlockHostUntilDoneCallbacks();

  if (allocated_) {
    parent_->DeallocateStream(this);
  }
}

port::Status Stream::RefreshStatus() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_35(mht_35_v, 561, "", "./tensorflow/stream_executor/stream.cc", "Stream::RefreshStatus");

  port::Status status = parent_->GetStatus(this);
  // We should not put the stream in an error state, just because the GetStatus
  // method is unimplemented.
  if (status != port::Status(port::error::UNIMPLEMENTED,
                             "GetStatus is not supported on this executor.")) {
    CheckStatus(status);
  }
  return status;
}

Stream &Stream::Init() {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_36(mht_36_v, 575, "", "./tensorflow/stream_executor/stream.cc", "Stream::Init");

  VLOG_CALL();

  absl::MutexLock lock(&mu_);
  CHECK_EQ(false, allocated_)
      << "stream appears to already have been initialized";
  CHECK(!status_.ok()) << "stream should be in !ok() state pre-initialization";

  if (parent_->AllocateStream(this)) {
    // Successful initialization!
    allocated_ = true;
    status_ = port::Status::OK();
  } else {
    LOG(ERROR) << "failed to allocate stream during initialization";
  }

  return *this;
}

Stream &Stream::InitTimer(Timer *timer) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_37(mht_37_v, 597, "", "./tensorflow/stream_executor/stream.cc", "Stream::InitTimer");

  VLOG_CALL(PARAM(timer));

  CheckError(parent_->AllocateTimer(timer));
  return *this;
}

Stream &Stream::InitWithTimer(Timer *timer) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_38(mht_38_v, 607, "", "./tensorflow/stream_executor/stream.cc", "Stream::InitWithTimer");

  VLOG_CALL(PARAM(timer));

  return Init().InitTimer(timer);
}

Stream &Stream::ThenRecordEvent(Event *event) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_39(mht_39_v, 616, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenRecordEvent");

  VLOG_CALL(PARAM(event));

  port::Status status = parent_->RecordEvent(this, event);
  if (!status.ok()) {
    LOG(ERROR) << "Error recording event in stream: " << status.error_message()
               << "; not marking stream as bad, as the Event object may be "
               << "at fault. Monitor for further errors.";
  }

  return *this;
}

Stream &Stream::ThenBatchNormalizationForward(
    const DeviceMemory<float> &x, const DeviceMemory<float> &scale,
    const DeviceMemory<float> &offset,
    const DeviceMemory<float> &estimated_mean,
    const DeviceMemory<float> &estimated_variance,
    const DeviceMemory<float> &side_input, const dnn::BatchDescriptor &x_desc,
    const dnn::BatchDescriptor &scale_offset_desc, const double epsilon,
    const double exponential_average_factor,
    dnn::ActivationMode activation_mode, DeviceMemory<float> *y,
    DeviceMemory<float> *batch_mean, DeviceMemory<float> *batch_var,
    DeviceMemory<float> *saved_mean, DeviceMemory<float> *saved_inv_var,
    bool is_training, ScratchAllocator *reserve_space_allocator,
    ScratchAllocator *workspace_allocator) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_40(mht_40_v, 644, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBatchNormalizationForward");

  VLOG_CALL(PARAM(x), PARAM(scale), PARAM(offset), PARAM(x_desc),
            PARAM(scale_offset_desc), PARAM(epsilon), PARAM(y));
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoBatchNormalizationForward(
        this, x, scale, offset, estimated_mean, estimated_variance, side_input,
        x_desc, scale_offset_desc, epsilon, exponential_average_factor,
        activation_mode, y, batch_mean, batch_var, saved_mean, saved_inv_var,
        is_training, reserve_space_allocator, workspace_allocator));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenBatchNormalizationBackward(
    const DeviceMemory<float> &y_backprop, const DeviceMemory<float> &x,
    const DeviceMemory<float> &scale, const DeviceMemory<float> &offset,
    const DeviceMemory<float> &mean, const DeviceMemory<float> &inv_var,
    const DeviceMemory<float> &y, const dnn::BatchDescriptor &x_desc,
    const dnn::BatchDescriptor &scale_offset_desc, const double epsilon,
    dnn::ActivationMode activation_mode, DeviceMemory<float> *x_backprop,
    DeviceMemory<float> *scale_backprop, DeviceMemory<float> *offset_backprop,
    DeviceMemory<float> *side_input_backprop,
    DeviceMemory<uint8> *reserve_space_data,
    ScratchAllocator *workspace_allocator) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_41(mht_41_v, 672, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBatchNormalizationBackward");

  VLOG_CALL(PARAM(y_backprop), PARAM(x), PARAM(scale), PARAM(x_desc),
            PARAM(scale_offset_desc), PARAM(epsilon), PARAM(x_backprop),
            PARAM(scale_backprop), PARAM(offset_backprop));
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoBatchNormalizationBackward(
        this, y_backprop, x, scale, offset, mean, inv_var, y, x_desc,
        scale_offset_desc, epsilon, activation_mode, x_backprop, scale_backprop,
        offset_backprop, side_input_backprop, reserve_space_data,
        workspace_allocator));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenBatchNormalizationForward(
    const DeviceMemory<Eigen::half> &x, const DeviceMemory<float> &scale,
    const DeviceMemory<float> &offset,
    const DeviceMemory<float> &estimated_mean,
    const DeviceMemory<float> &estimated_variance,
    const DeviceMemory<Eigen::half> &side_input,
    const dnn::BatchDescriptor &x_desc,
    const dnn::BatchDescriptor &scale_offset_desc, const double epsilon,
    const double exponential_average_factor,
    dnn::ActivationMode activation_mode, DeviceMemory<Eigen::half> *y,
    DeviceMemory<float> *batch_mean, DeviceMemory<float> *batch_var,
    DeviceMemory<float> *saved_mean, DeviceMemory<float> *saved_inv_var,
    bool is_training, ScratchAllocator *reserve_space_allocator,
    ScratchAllocator *workspace_allocator) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_42(mht_42_v, 704, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBatchNormalizationForward");

  VLOG_CALL(PARAM(x), PARAM(scale), PARAM(offset), PARAM(x_desc),
            PARAM(scale_offset_desc), PARAM(epsilon), PARAM(y));
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoBatchNormalizationForward(
        this, x, scale, offset, estimated_mean, estimated_variance, side_input,
        x_desc, scale_offset_desc, epsilon, exponential_average_factor,
        activation_mode, y, batch_mean, batch_var, saved_mean, saved_inv_var,
        is_training, reserve_space_allocator, workspace_allocator));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenBatchNormalizationBackward(
    const DeviceMemory<Eigen::half> &y_backprop,
    const DeviceMemory<Eigen::half> &x, const DeviceMemory<float> &scale,
    const DeviceMemory<float> &offset, const DeviceMemory<float> &mean,
    const DeviceMemory<float> &inv_var, const DeviceMemory<Eigen::half> &y,
    const dnn::BatchDescriptor &x_desc,
    const dnn::BatchDescriptor &scale_offset_desc, const double epsilon,
    dnn::ActivationMode activation_mode, DeviceMemory<Eigen::half> *x_backprop,
    DeviceMemory<float> *scale_backprop, DeviceMemory<float> *offset_backprop,
    DeviceMemory<Eigen::half> *side_input_backprop,
    DeviceMemory<uint8> *reserve_space_data,
    ScratchAllocator *workspace_allocator) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_43(mht_43_v, 733, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBatchNormalizationBackward");

  VLOG_CALL(PARAM(y_backprop), PARAM(x), PARAM(scale), PARAM(x_desc),
            PARAM(scale_offset_desc), PARAM(epsilon), PARAM(x_backprop),
            PARAM(scale_backprop), PARAM(offset_backprop));
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoBatchNormalizationBackward(
        this, y_backprop, x, scale, offset, mean, inv_var, y, x_desc,
        scale_offset_desc, epsilon, activation_mode, x_backprop, scale_backprop,
        offset_backprop, side_input_backprop, reserve_space_data,
        workspace_allocator));

  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenConvolve(
    const dnn::BatchDescriptor &input_descriptor,
    const DeviceMemory<float> &input_data,
    const dnn::FilterDescriptor &filter_descriptor,
    const DeviceMemory<float> &filter_data,
    const dnn::ConvolutionDescriptor &convolution_descriptor,
    const dnn::BatchDescriptor &output_descriptor,
    DeviceMemory<float> *output) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_44(mht_44_v, 760, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenConvolve");

  if (ok()) {
    CheckError(ConvolveWithAlgorithm(
                   dnn::ConvolutionKind::FORWARD, input_descriptor, input_data,
                   filter_descriptor, filter_data, output_descriptor, *output,
                   convolution_descriptor,
                   /*scratch_allocator=*/nullptr, dnn::AlgorithmConfig(),
                   /*output_profile_result=*/nullptr)
                   .ok());
  }
  return *this;
}

Stream &Stream::ThenConvolveQuantized(
    const dnn::BatchDescriptor &input_descriptor,
    const DeviceMemory<float> &input_data,
    const dnn::FilterDescriptor &filter_descriptor,
    const DeviceMemory<int8> &filter_coefficients,
    const DeviceMemory<float> &coefficient_scales,
    const dnn::ConvolutionDescriptor &convolution_descriptor,
    const dnn::BatchDescriptor &output_descriptor,
    DeviceMemory<float> *output) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_45(mht_45_v, 784, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenConvolveQuantized");

  VLOG_CALL(PARAM(input_descriptor), PARAM(input_data),
            PARAM(filter_descriptor), PARAM(filter_coefficients),
            PARAM(coefficient_scales), PARAM(convolution_descriptor),
            PARAM(output_descriptor), PARAM(output));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoConvolveQuantized(
        this, input_descriptor, input_data, filter_descriptor,
        filter_coefficients, coefficient_scales, convolution_descriptor,
        output_descriptor, output));
  } else {
    SetError();
    LOG(WARNING) << "attempting to perform DNN operation using StreamExecutor "
                    "without DNN support";
  }
  return *this;
}

Stream &Stream::ThenConvolveQuantized(
    const dnn::BatchDescriptor &input_descriptor,
    const DeviceMemory<float> &input_data,
    const dnn::FilterDescriptor &filter_descriptor,
    const DeviceMemory<int16> &filter_coefficients,
    const DeviceMemory<float> &coefficient_scales,
    const dnn::ConvolutionDescriptor &convolution_descriptor,
    const dnn::BatchDescriptor &output_descriptor,
    DeviceMemory<float> *output) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_46(mht_46_v, 814, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenConvolveQuantized");

  VLOG_CALL(PARAM(input_descriptor), PARAM(input_data),
            PARAM(filter_descriptor), PARAM(filter_coefficients),
            PARAM(coefficient_scales), PARAM(convolution_descriptor),
            PARAM(output_descriptor), PARAM(output));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoConvolveQuantized(
        this, input_descriptor, input_data, filter_descriptor,
        filter_coefficients, coefficient_scales, convolution_descriptor,
        output_descriptor, output));
  } else {
    SetError();
    LOG(WARNING) << "attempting to perform DNN operation using StreamExecutor "
                    "without DNN support";
  }
  return *this;
}

Stream &Stream::ThenSeparableConvolve(
    const dnn::BatchDescriptor &batch_descriptor,
    const DeviceMemory<float> &input_data,
    const dnn::FilterDescriptor &filter_descriptor, int depth_multiplier,
    const DeviceMemory<float> &first_weights,
    const DeviceMemory<float> &second_weights,
    const dnn::ConvolutionDescriptor &convolution_descriptor,
    const dnn::BatchDescriptor &output_descriptor,
    DeviceMemory<float> *output) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_47(mht_47_v, 844, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenSeparableConvolve");

  VLOG_CALL(
      PARAM(batch_descriptor), PARAM(input_data), PARAM(filter_descriptor),
      PARAM(depth_multiplier), PARAM(first_weights), PARAM(second_weights),
      PARAM(convolution_descriptor), PARAM(output_descriptor), PARAM(output));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoSeparableConvolve(
        this, batch_descriptor, input_data, filter_descriptor, depth_multiplier,
        first_weights, second_weights, convolution_descriptor,
        output_descriptor, output));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenMatMul(const DeviceMemory<float> &input_data,
                           const DeviceMemory<float> &weights,
                           const dnn::BatchDescriptor &input_dimensions,
                           const dnn::BatchDescriptor &output_dimensions,
                           DeviceMemory<float> *output_data) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_48(mht_48_v, 868, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenMatMul");

  VLOG_CALL(PARAM(input_data), PARAM(weights), PARAM(input_dimensions),
            PARAM(output_dimensions), PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoMatMul(this, input_data, weights, input_dimensions,
                             output_dimensions, output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenMatMulQuantized(
    const DeviceMemory<float> &input_data, const DeviceMemory<int8> &weights,
    const DeviceMemory<float> &weight_scales,
    const dnn::BatchDescriptor &input_dimensions,
    const dnn::BatchDescriptor &output_dimensions,
    DeviceMemory<float> *output_data) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_49(mht_49_v, 889, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenMatMulQuantized");

  VLOG_CALL(PARAM(input_data), PARAM(weights), PARAM(weight_scales),
            PARAM(input_dimensions), PARAM(output_dimensions),
            PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoMatMulQuantized(this, input_data, weights, weight_scales,
                                      input_dimensions, output_dimensions,
                                      output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenMatMulQuantized(
    const DeviceMemory<float> &input_data, const DeviceMemory<int16> &weights,
    const DeviceMemory<float> &weight_scales,
    const dnn::BatchDescriptor &input_dimensions,
    const dnn::BatchDescriptor &output_dimensions,
    DeviceMemory<float> *output_data) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_50(mht_50_v, 912, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenMatMulQuantized");

  VLOG_CALL(PARAM(input_data), PARAM(weights), PARAM(weight_scales),
            PARAM(input_dimensions), PARAM(output_dimensions),
            PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoMatMulQuantized(this, input_data, weights, weight_scales,
                                      input_dimensions, output_dimensions,
                                      output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenBiasAdd(const DeviceMemory<float> &input_data,
                            const DeviceMemory<float> &biases,
                            const dnn::BatchDescriptor &dimensions,
                            DeviceMemory<float> *output_data) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_51(mht_51_v, 933, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBiasAdd");

  VLOG_CALL(PARAM(input_data), PARAM(biases), PARAM(dimensions),
            PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(
        dnn->DoBiasAdd(this, input_data, biases, dimensions, output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenPoolForward(
    const dnn::PoolingDescriptor &pooling_dimensions,
    const dnn::BatchDescriptor &input_dimensions,
    const DeviceMemory<double> &input_data,
    const dnn::BatchDescriptor &output_dimensions,
    DeviceMemory<double> *output_data, ScratchAllocator *workspace_allocator) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_52(mht_52_v, 954, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenPoolForward");

  VLOG_CALL(PARAM(pooling_dimensions), PARAM(input_dimensions),
            PARAM(input_data), PARAM(output_dimensions), PARAM(output_data),
            PARAM(workspace_allocator));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoPoolForward(this, pooling_dimensions, input_dimensions,
                                  input_data, output_dimensions, output_data,
                                  workspace_allocator));
  } else {
    SetError();
    LOG(WARNING) << "attempting to perform DNN operation using StreamExecutor "
                    "without DNN support";
  }
  return *this;
}

Stream &Stream::ThenPoolForward(
    const dnn::PoolingDescriptor &pooling_dimensions,
    const dnn::BatchDescriptor &input_dimensions,
    const DeviceMemory<float> &input_data,
    const dnn::BatchDescriptor &output_dimensions,
    DeviceMemory<float> *output_data, ScratchAllocator *workspace_allocator) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_53(mht_53_v, 979, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenPoolForward");

  VLOG_CALL(PARAM(pooling_dimensions), PARAM(input_dimensions),
            PARAM(input_data), PARAM(output_dimensions), PARAM(output_data),
            PARAM(workspace_allocator));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoPoolForward(this, pooling_dimensions, input_dimensions,
                                  input_data, output_dimensions, output_data,
                                  workspace_allocator));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenPoolForward(
    const dnn::PoolingDescriptor &pooling_dimensions,
    const dnn::BatchDescriptor &input_dimensions,
    const DeviceMemory<Eigen::half> &input_data,
    const dnn::BatchDescriptor &output_dimensions,
    DeviceMemory<Eigen::half> *output_data,
    ScratchAllocator *workspace_allocator) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_54(mht_54_v, 1003, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenPoolForward");

  VLOG_CALL(PARAM(pooling_dimensions), PARAM(input_dimensions),
            PARAM(input_data), PARAM(output_dimensions), PARAM(output_data),
            PARAM(workspace_allocator));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoPoolForward(this, pooling_dimensions, input_dimensions,
                                  input_data, output_dimensions, output_data,
                                  workspace_allocator));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenPoolForward(
    const dnn::PoolingDescriptor &pooling_dimensions,
    const dnn::BatchDescriptor &input_dimensions,
    const DeviceMemory<int8> &input_data,
    const dnn::BatchDescriptor &output_dimensions,
    DeviceMemory<int8> *output_data, ScratchAllocator *workspace_allocator) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_55(mht_55_v, 1026, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenPoolForward");

  VLOG_CALL(PARAM(pooling_dimensions), PARAM(input_dimensions),
            PARAM(input_data), PARAM(output_dimensions), PARAM(output_data),
            PARAM(workspace_allocator));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoPoolForward(this, pooling_dimensions, input_dimensions,
                                  input_data, output_dimensions, output_data,
                                  workspace_allocator));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenPoolBackward(
    const dnn::PoolingDescriptor &pooling_dimensions,
    const dnn::BatchDescriptor &input_dimensions,
    const DeviceMemory<double> &input_data,
    const dnn::BatchDescriptor &output_dimensions,
    const DeviceMemory<double> &output_data,
    const DeviceMemory<double> &input_diff_data,
    DeviceMemory<double> *output_diff_data,
    ScratchAllocator *workspace_allocator) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_56(mht_56_v, 1052, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenPoolBackward");

  VLOG_CALL(PARAM(pooling_dimensions), PARAM(input_dimensions),
            PARAM(input_data), PARAM(output_dimensions), PARAM(output_data),
            PARAM(input_diff_data), PARAM(output_diff_data),
            PARAM(workspace_allocator));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoPoolBackward(this, pooling_dimensions, input_dimensions,
                                   input_data, output_dimensions, output_data,
                                   input_diff_data, output_diff_data,
                                   workspace_allocator));
  } else {
    SetError();
    LOG(WARNING) << "attempting to perform DNN operation using StreamExecutor "
                    "without DNN support";
  }
  return *this;
}

Stream &Stream::ThenPoolBackward(
    const dnn::PoolingDescriptor &pooling_dimensions,
    const dnn::BatchDescriptor &input_dimensions,
    const DeviceMemory<float> &input_data,
    const dnn::BatchDescriptor &output_dimensions,
    const DeviceMemory<float> &output_data,
    const DeviceMemory<float> &input_diff_data,
    DeviceMemory<float> *output_diff_data,
    ScratchAllocator *workspace_allocator) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_57(mht_57_v, 1082, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenPoolBackward");

  VLOG_CALL(PARAM(pooling_dimensions), PARAM(input_dimensions),
            PARAM(input_data), PARAM(output_dimensions), PARAM(output_data),
            PARAM(input_diff_data), PARAM(output_diff_data),
            PARAM(workspace_allocator));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoPoolBackward(this, pooling_dimensions, input_dimensions,
                                   input_data, output_dimensions, output_data,
                                   input_diff_data, output_diff_data,
                                   workspace_allocator));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenPoolBackward(
    const dnn::PoolingDescriptor &pooling_dimensions,
    const dnn::BatchDescriptor &input_dimensions,
    const DeviceMemory<Eigen::half> &input_data,
    const dnn::BatchDescriptor &output_dimensions,
    const DeviceMemory<Eigen::half> &output_data,
    const DeviceMemory<Eigen::half> &input_diff_data,
    DeviceMemory<Eigen::half> *output_diff_data,
    ScratchAllocator *workspace_allocator) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_58(mht_58_v, 1110, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenPoolBackward");

  VLOG_CALL(PARAM(pooling_dimensions), PARAM(input_dimensions),
            PARAM(input_data), PARAM(output_dimensions), PARAM(output_data),
            PARAM(input_diff_data), PARAM(output_diff_data),
            PARAM(workspace_allocator));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoPoolBackward(this, pooling_dimensions, input_dimensions,
                                   input_data, output_dimensions, output_data,
                                   input_diff_data, output_diff_data,
                                   workspace_allocator));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenNormalizeWithDimensions(
    const dnn::NormalizeDescriptor &normalize_descriptor,
    const dnn::BatchDescriptor &dimensions,
    const DeviceMemory<float> &input_data, DeviceMemory<float> *output_data) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_59(mht_59_v, 1133, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenNormalizeWithDimensions");

  VLOG_CALL(PARAM(normalize_descriptor), PARAM(dimensions), PARAM(input_data),
            PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoNormalizeWithDimensions(
        this, normalize_descriptor, dimensions, input_data, output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenNormalizeBackwardWithDimensions(
    const dnn::NormalizeDescriptor &normalize_descriptor,
    const dnn::BatchDescriptor &dimensions, const DeviceMemory<float> &raw_data,
    const DeviceMemory<float> &normalized_data,
    const DeviceMemory<float> &normalized_variable_gradient,
    DeviceMemory<float> *raw_variable_gradient,
    ScratchAllocator *workspace_allocator) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_60(mht_60_v, 1155, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenNormalizeBackwardWithDimensions");

  VLOG_CALL(PARAM(normalize_descriptor), PARAM(dimensions), PARAM(raw_data),
            PARAM(normalized_data), PARAM(normalized_variable_gradient),
            PARAM(raw_variable_gradient), PARAM(workspace_allocator));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoNormalizeBackwardWithDimensions(
        this, normalize_descriptor, dimensions, raw_data, normalized_data,
        normalized_variable_gradient, raw_variable_gradient,
        workspace_allocator));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenActivate(dnn::ActivationMode activation_mode,
                             const dnn::BatchDescriptor &dimensions,
                             const DeviceMemory<float> &input_data,
                             DeviceMemory<float> *output_data) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_61(mht_61_v, 1177, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenActivate");

  return ThenActivateWithOptions(activation_mode, dimensions, input_data,
                                 output_data, /*options=*/0);
}

Stream &Stream::ThenActivateWithOptions(dnn::ActivationMode activation_mode,
                                        const dnn::BatchDescriptor &dimensions,
                                        const DeviceMemory<float> &input_data,
                                        DeviceMemory<float> *output_data,
                                        uint64_t options) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_62(mht_62_v, 1189, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenActivateWithOptions");

  VLOG_CALL(PARAM(activation_mode), PARAM(dimensions), PARAM(input_data),
            PARAM(output_data), PARAM(options));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoActivate(this, activation_mode, dimensions, input_data,
                               output_data, options));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenDepthConcatenate(
    port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
    port::ArraySlice<const DeviceMemory<float> *> input_data,
    DeviceMemory<float> *output_data) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_63(mht_63_v, 1208, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenDepthConcatenate");

  VLOG_CALL(PARAM(input_dimensions), PARAM(input_data), PARAM(output_data));

  for (size_t i = 1; i < input_dimensions.size(); ++i) {
    if (input_dimensions[i].count() != input_dimensions[0].count() ||
        input_dimensions[i].height() != input_dimensions[0].height() ||
        input_dimensions[i].width() != input_dimensions[0].width()) {
      SetError();
      LOG(ERROR) << "Incompatible dimensions for depth concatenation.\n"
                 << "input_dimensions[0]: " << input_dimensions[0].ToString()
                 << "input_dimensions[" << i
                 << "]: " << input_dimensions[i].ToString();
      return *this;
    }
  }

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoDepthConcatenate(this, input_dimensions, input_data,
                                       output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenSpaceConcatenate(
    port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
    port::ArraySlice<const DeviceMemory<float> *> input_data,
    DeviceMemory<float> *output_data,
    dnn::SpaceConcatenateMode concat_direction) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_64(mht_64_v, 1240, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenSpaceConcatenate");

  VLOG_CALL(PARAM(input_dimensions), PARAM(input_data), PARAM(output_data));

  // Check that the input dimensions of all the other batches match those of the
  // first batch.
  for (size_t i = 1; i < input_dimensions.size(); ++i) {
    if ((concat_direction == dnn::SpaceConcatenateMode::XDirection) &&
        (input_dimensions[i].count() != input_dimensions[0].count() ||
         input_dimensions[i].height() != input_dimensions[0].height() ||
         input_dimensions[i].feature_map_count() !=
             input_dimensions[0].feature_map_count())) {
      SetError();
      LOG(ERROR) << "Incompatible dimensions for X concatenation.\n"
                 << "input_dimensions[0]: " << input_dimensions[0].ToString()
                 << "input_dimensions[" << i
                 << "]: " << input_dimensions[i].ToString();
      return *this;
    }

    if ((concat_direction == dnn::SpaceConcatenateMode::YDirection) &&
        (input_dimensions[i].count() != input_dimensions[0].count() ||
         input_dimensions[i].width() != input_dimensions[0].width() ||
         input_dimensions[i].feature_map_count() !=
             input_dimensions[0].feature_map_count())) {
      SetError();
      LOG(ERROR) << "Incompatible dimensions for Y concatenation.\n"
                 << "input_dimensions[0]: " << input_dimensions[0].ToString()
                 << "input_dimensions[" << i
                 << "]: " << input_dimensions[i].ToString();
      return *this;
    }
  }
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoSpaceConcatenate(this, input_dimensions, input_data,
                                       output_data, concat_direction));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenReshape(const dnn::BatchDescriptor &input_dimensions,
                            const DeviceMemory<float> &input_data,
                            const dnn::BatchDescriptor &output_dimensions,
                            DeviceMemory<float> *output_data) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_65(mht_65_v, 1287, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenReshape");

  VLOG_CALL(PARAM(input_dimensions), PARAM(input_data),
            PARAM(output_dimensions), PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoReshape(this, input_dimensions, input_data,
                              output_dimensions, output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenDepthToSpace(
    const dnn::BatchDescriptor &input_dimensions,
    const DeviceMemory<float> &input_data,
    const dnn::DepthToSpaceLayout &depth_to_space_layout,
    const int sqrt_depth_reduction, DeviceMemory<float> *output_data) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_66(mht_66_v, 1307, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenDepthToSpace");

  VLOG_CALL(PARAM(input_dimensions), PARAM(input_data),
            PARAM(depth_to_space_layout), PARAM(sqrt_depth_reduction),
            PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoDepthToSpace(this, input_dimensions, input_data,
                                   depth_to_space_layout, sqrt_depth_reduction,
                                   output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenSpaceToDepth(
    const dnn::BatchDescriptor &input_dimensions,
    const DeviceMemory<float> &input_data,
    const dnn::DepthToSpaceLayout &space_to_depth_layout,
    const int sqrt_depth_increase, DeviceMemory<float> *output_data) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_67(mht_67_v, 1329, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenSpaceToDepth");

  VLOG_CALL(PARAM(input_dimensions), PARAM(input_data),
            PARAM(space_to_depth_layout), PARAM(sqrt_depth_increase),
            PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoSpaceToDepth(this, input_dimensions, input_data,
                                   space_to_depth_layout, sqrt_depth_increase,
                                   output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenElementwiseOperate(
    dnn::ElementwiseOperation operation,
    port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
    port::ArraySlice<const DeviceMemory<float> *> input_data,
    const dnn::BatchDescriptor &output_dimensions,
    DeviceMemory<float> *output_data) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_68(mht_68_v, 1352, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenElementwiseOperate");

  VLOG_CALL(PARAM(operation), PARAM(input_dimensions), PARAM(input_data),
            PARAM(output_dimensions), PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoElementwiseOperate(this, operation, input_dimensions,
                                         input_data, output_dimensions,
                                         output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenElementwiseOperateScaledQuantized(
    dnn::ElementwiseOperation operation,
    port::ArraySlice<int> input_multiplicands, int output_divisor,
    port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
    port::ArraySlice<const DeviceMemory<float> *> input_data,
    const dnn::BatchDescriptor &output_dimensions,
    DeviceMemory<float> *output_data) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_69(mht_69_v, 1375, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenElementwiseOperateScaledQuantized");

  VLOG_CALL(PARAM(operation), PARAM(input_multiplicands), PARAM(output_divisor),
            PARAM(input_dimensions), PARAM(input_data),
            PARAM(output_dimensions), PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoElementwiseOperateScaledQuantized(
        this, operation, input_multiplicands, output_divisor, input_dimensions,
        input_data, output_dimensions, output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenXYPad(const dnn::BatchDescriptor &dimensions,
                          const DeviceMemory<float> &input_data,
                          int64_t left_pad, int64_t right_pad, int64_t top_pad,
                          int64_t bottom_pad,
                          DeviceMemory<float> *output_data) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_70(mht_70_v, 1397, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenXYPad");

  VLOG_CALL(PARAM(dimensions), PARAM(input_data), PARAM(left_pad),
            PARAM(right_pad), PARAM(top_pad), PARAM(bottom_pad),
            PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoXYPad(this, dimensions, input_data, left_pad, right_pad,
                            top_pad, bottom_pad, output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenXYSlice(const dnn::BatchDescriptor &dimensions,
                            const DeviceMemory<float> &input_data,
                            int64_t left_trim, int64_t right_trim,
                            int64_t top_trim, int64_t bottom_trim,
                            DeviceMemory<float> *output_data) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_71(mht_71_v, 1418, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenXYSlice");

  VLOG_CALL(PARAM(dimensions), PARAM(input_data), PARAM(left_trim),
            PARAM(right_trim), PARAM(top_trim), PARAM(bottom_trim),
            PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoXYSlice(this, dimensions, input_data, left_trim,
                              right_trim, top_trim, bottom_trim, output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenXYBroadcast(const dnn::BatchDescriptor &dimensions,
                                const DeviceMemory<float> &input_data,
                                int64_t replicate_x, int64_t replicate_y,
                                DeviceMemory<float> *output_data) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_72(mht_72_v, 1438, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenXYBroadcast");

  VLOG_CALL(PARAM(dimensions), PARAM(input_data), PARAM(replicate_x),
            PARAM(replicate_y), PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoXYBroadcast(this, dimensions, input_data, replicate_x,
                                  replicate_y, output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenMemcpyD2HQuantized(
    const DeviceMemory<float> &gpu_unquantized_src,
    dnn::QuantizedActivationMode mode, void *host_dst, uint64_t size) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_73(mht_73_v, 1456, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenMemcpyD2HQuantized");

  VLOG_CALL(PARAM(gpu_unquantized_src), PARAM(mode), PARAM(host_dst),
            PARAM(size));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoMemcpyD2HQuantized(this, gpu_unquantized_src, mode,
                                         host_dst, size));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenMemcpyH2DQuantized(
    const void *host_src, uint64_t size, dnn::QuantizedActivationMode mode,
    DeviceMemory<float> *gpu_unquantized_dst) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_74(mht_74_v, 1474, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenMemcpyH2DQuantized");

  VLOG_CALL(PARAM(host_src), PARAM(size), PARAM(mode),
            PARAM(gpu_unquantized_dst));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoMemcpyH2DQuantized(this, host_src, size, mode,
                                         gpu_unquantized_dst));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream *Stream::GetOrCreateSubStream() {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_75(mht_75_v, 1490, "", "./tensorflow/stream_executor/stream.cc", "Stream::GetOrCreateSubStream");

  // Do not destroy bad streams when holding mu_ because ~Stream() may
  // BlockHostUntilDone and it's host callbacks might attempt to acquire mu_.
  std::vector<std::unique_ptr<Stream>> bad_streams;

  absl::MutexLock lock(&mu_);

  // Look for the first reusable sub_stream that is ok, dropping !ok sub_streams
  // we encounter along the way.
  for (size_t index = 0; index < sub_streams_.size();) {
    std::pair<std::unique_ptr<Stream>, bool> &pair = sub_streams_[index];
    if (pair.second) {
      // The sub_stream is reusable.
      Stream *sub_stream = pair.first.get();
      if (sub_stream->ok()) {
        VLOG(1) << DebugStreamPointers() << " reusing sub_stream "
                << sub_stream->DebugStreamPointers();
        pair.second = false;
        return sub_stream;
      }

      // The stream is reusable and not ok. Streams have a monotonic state
      // machine; the stream will remain in !ok forever. Swap it with the last
      // stream and pop it off.
      const int64_t last = sub_streams_.size() - 1;
      if (index != last) {
        std::swap(pair, sub_streams_[last]);
      }
      bad_streams.push_back(std::move(sub_streams_.back().first));
      sub_streams_.pop_back();
      VLOG(1) << DebugStreamPointers() << " dropped !ok sub_stream "
              << sub_stream->DebugStreamPointers();
    } else {
      // The sub_stream is not reusable, move on to the next one.
      ++index;
    }
  }

  // No streams are reusable; create a new stream.
  sub_streams_.emplace_back(std::unique_ptr<Stream>{new Stream{parent_}},
                            false);
  Stream *sub_stream = sub_streams_.back().first.get();
  sub_stream->Init();
  if (!sub_stream->ok()) {
    LOG(ERROR) << "sub-stream failed to be initialized";
  }
  VLOG(1) << DebugStreamPointers() << " created new sub_stream "
          << sub_stream->DebugStreamPointers();

  return sub_stream;
}

void Stream::ReturnSubStream(Stream *sub_stream) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_76(mht_76_v, 1545, "", "./tensorflow/stream_executor/stream.cc", "Stream::ReturnSubStream");

  // Do not destroy bad streams when holding mu_ because ~Stream() may
  // BlockHostUntilDone and it's host callbacks might attempt to acquire mu_.
  std::unique_ptr<Stream> bad_stream;

  absl::MutexLock lock(&mu_);

  // Look for the sub-stream.
  for (int64_t index = 0, end = sub_streams_.size(); index < end; ++index) {
    std::pair<std::unique_ptr<Stream>, bool> &pair = sub_streams_[index];
    if (pair.first.get() != sub_stream) {
      continue;
    }

    // Found the sub_stream.
    if (sub_stream->ok()) {
      VLOG(1) << DebugStreamPointers() << " returned ok sub_stream "
              << sub_stream->DebugStreamPointers();
      pair.second = true;
    } else {
      // The returned stream is not ok. Streams have a monotonic state
      // machine; the stream will remain in !ok forever. Swap it with the last
      // stream and pop it off.
      VLOG(1) << DebugStreamPointers() << " returned !ok sub_stream "
              << sub_stream->DebugStreamPointers();
      const int64_t last = sub_streams_.size() - 1;
      if (index != last) {
        std::swap(pair, sub_streams_[last]);
      }
      std::swap(bad_stream, sub_streams_.back().first);
      sub_streams_.pop_back();
    }
    return;
  }

  LOG(FATAL) << DebugStreamPointers()
             << " did not create the returned sub-stream "
             << sub_stream->DebugStreamPointers();
}

Stream &Stream::ThenStartTimer(Timer *t) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_77(mht_77_v, 1588, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenStartTimer");

  VLOG_CALL(PARAM(t));

  CheckError(parent_->StartTimer(this, t));
  return *this;
}

Stream &Stream::ThenStopTimer(Timer *t) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_78(mht_78_v, 1598, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenStopTimer");

  VLOG_CALL(PARAM(t));

  CheckError(parent_->StopTimer(this, t));
  return *this;
}

Stream &Stream::ThenWaitFor(Stream *other) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_79(mht_79_v, 1608, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenWaitFor");

  VLOG_CALL(PARAM(other));

  CHECK(this != other) << "stream cannot wait for itself";
  if (ok() && other->ok()) {
    CheckError(parent_->CreateStreamDependency(this, other));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers() << " did not wait for "
              << other->DebugStreamPointers();
  }
  return *this;
}

Stream &Stream::ThenWaitFor(Event *event) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_80(mht_80_v, 1625, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenWaitFor");

  VLOG_CALL(PARAM(event));

  if (ok()) {
    port::Status status = parent_->WaitForEvent(this, event);
    if (!status.ok()) {
      LOG(ERROR) << "Error waiting for event in stream: "
                 << status.error_message()
                 << "; not marking stream as bad, as the Event object may be "
                 << "at fault. Monitor for further errors.";
    }
  } else {
    LOG(INFO) << DebugStreamPointers() << " did not wait for an event.";
  }
  return *this;
}

// A functor that implements ThenBlasXXX interfaces, which calls DoBlasXXX
// functions and logs for errors.
template <typename... Args>
struct ThenBlasImpl {
  // blas_func is the DoBlasXXX member function pointer, and args are its
  // arguments except the first one of Stream* type.
  Stream &operator()(Stream *stream,
                     bool (blas::BlasSupport::*blas_func)(Stream *, Args...),
                     Args... args) {
    return Run(stream, blas_func, /*record_error=*/true, args...);
  }

  // Like operator(), but only calls stream->CheckError() if record_error is
  // true.
  Stream &Run(Stream *stream,
              bool (blas::BlasSupport::*blas_func)(Stream *, Args...),
              bool record_error, Args... args);
};

template <typename... Args>
Stream &ThenBlasImpl<Args...>::Run(
    Stream *stream, bool (blas::BlasSupport::*blas_func)(Stream *, Args...),
    bool record_error, Args... args) {
  if (stream->ok()) {
    bool ok;
    if (blas::BlasSupport *blas = stream->parent_->AsBlas()) {
      ok = (blas->*blas_func)(stream, args...);
    } else {
      LOG(WARNING)
          << "attempting to perform BLAS operation using StreamExecutor "
             "without BLAS support";
      ok = false;
    }
    if (record_error) {
      stream->CheckError(ok);
    }
  }
  return *stream;
}

Stream &Stream::ThenBlasAsum(uint64_t elem_count, const DeviceMemory<float> &x,
                             int incx, DeviceMemory<float> *result) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_81(mht_81_v, 1686, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasAsum");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<float> &, int,
               DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasAsum, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasAsum(uint64_t elem_count, const DeviceMemory<double> &x,
                             int incx, DeviceMemory<double> *result) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_82(mht_82_v, 1700, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasAsum");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<double> &, int,
               DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasAsum, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasAsum(uint64_t elem_count,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, DeviceMemory<float> *result) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_83(mht_83_v, 1715, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasAsum");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasAsum, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasAsum(uint64_t elem_count,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, DeviceMemory<double> *result) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_84(mht_84_v, 1730, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasAsum");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasAsum, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasAxpy(uint64_t elem_count, float alpha,
                             const DeviceMemory<float> &x, int incx,
                             DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_85(mht_85_v, 1745, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasAxpy");

  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy));

  ThenBlasImpl<uint64_t, float, const DeviceMemory<float> &, int,
               DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasAxpy, elem_count, alpha, x, incx,
              y, incy);
}

Stream &Stream::ThenBlasAxpy(uint64_t elem_count, double alpha,
                             const DeviceMemory<double> &x, int incx,
                             DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_86(mht_86_v, 1761, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasAxpy");

  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy));

  ThenBlasImpl<uint64_t, double, const DeviceMemory<double> &, int,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasAxpy, elem_count, alpha, x, incx,
              y, incy);
}

Stream &Stream::ThenBlasAxpy(uint64_t elem_count, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, DeviceMemory<std::complex<float>> *y,
                             int incy) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_87(mht_87_v, 1778, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasAxpy");

  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy));

  ThenBlasImpl<uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasAxpy, elem_count, alpha, x, incx,
              y, incy);
}

Stream &Stream::ThenBlasAxpy(uint64_t elem_count, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, DeviceMemory<std::complex<double>> *y,
                             int incy) {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_88(mht_88_v, 1796, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasAxpy");

  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy));

  ThenBlasImpl<uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasAxpy, elem_count, alpha, x, incx,
              y, incy);
}

Stream &Stream::ThenBlasCopy(uint64_t elem_count, const DeviceMemory<float> &x,
                             int incx, DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_89(mht_89_v, 1812, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasCopy");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy));

  ThenBlasImpl<uint64_t, const DeviceMemory<float> &, int,
               DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasCopy, elem_count, x, incx, y,
              incy);
}

Stream &Stream::ThenBlasCopy(uint64_t elem_count, const DeviceMemory<double> &x,
                             int incx, DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_90(mht_90_v, 1826, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasCopy");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy));

  ThenBlasImpl<uint64_t, const DeviceMemory<double> &, int,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasCopy, elem_count, x, incx, y,
              incy);
}

Stream &Stream::ThenBlasCopy(uint64_t elem_count,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, DeviceMemory<std::complex<float>> *y,
                             int incy) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_91(mht_91_v, 1842, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasCopy");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasCopy, elem_count, x, incx, y,
              incy);
}

Stream &Stream::ThenBlasCopy(uint64_t elem_count,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, DeviceMemory<std::complex<double>> *y,
                             int incy) {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_92(mht_92_v, 1858, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasCopy");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasCopy, elem_count, x, incx, y,
              incy);
}

Stream &Stream::ThenBlasDot(uint64_t elem_count, const DeviceMemory<float> &x,
                            int incx, const DeviceMemory<float> &y, int incy,
                            DeviceMemory<float> *result) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_93(mht_93_v, 1873, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasDot");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<float> &, int,
               const DeviceMemory<float> &, int, DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasDot, elem_count, x, incx, y, incy,
              result);
}

Stream &Stream::ThenBlasDot(uint64_t elem_count, const DeviceMemory<double> &x,
                            int incx, const DeviceMemory<double> &y, int incy,
                            DeviceMemory<double> *result) {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_94(mht_94_v, 1889, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasDot");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<double> &, int,
               const DeviceMemory<double> &, int, DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasDot, elem_count, x, incx, y, incy,
              result);
}

Stream &Stream::ThenBlasDotc(uint64_t elem_count,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx,
                             const DeviceMemory<std::complex<float>> &y,
                             int incy,
                             DeviceMemory<std::complex<float>> *result) {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_95(mht_95_v, 1908, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasDotc");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasDotc, elem_count, x, incx, y,
              incy, result);
}

Stream &Stream::ThenBlasDotc(uint64_t elem_count,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx,
                             const DeviceMemory<std::complex<double>> &y,
                             int incy,
                             DeviceMemory<std::complex<double>> *result) {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_96(mht_96_v, 1928, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasDotc");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasDotc, elem_count, x, incx, y,
              incy, result);
}

Stream &Stream::ThenBlasDotu(uint64_t elem_count,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx,
                             const DeviceMemory<std::complex<float>> &y,
                             int incy,
                             DeviceMemory<std::complex<float>> *result) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_97(mht_97_v, 1948, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasDotu");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasDotu, elem_count, x, incx, y,
              incy, result);
}

Stream &Stream::ThenBlasDotu(uint64_t elem_count,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx,
                             const DeviceMemory<std::complex<double>> &y,
                             int incy,
                             DeviceMemory<std::complex<double>> *result) {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_98(mht_98_v, 1968, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasDotu");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasDotu, elem_count, x, incx, y,
              incy, result);
}

Stream &Stream::ThenBlasNrm2(uint64_t elem_count, const DeviceMemory<float> &x,
                             int incx, DeviceMemory<float> *result) {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_99(mht_99_v, 1984, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasNrm2");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<float> &, int,
               DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasNrm2, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasNrm2(uint64_t elem_count, const DeviceMemory<double> &x,
                             int incx, DeviceMemory<double> *result) {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_100(mht_100_v, 1998, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasNrm2");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<double> &, int,
               DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasNrm2, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasNrm2(uint64_t elem_count,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, DeviceMemory<float> *result) {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_101(mht_101_v, 2013, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasNrm2");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasNrm2, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasNrm2(uint64_t elem_count,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, DeviceMemory<double> *result) {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_102(mht_102_v, 2028, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasNrm2");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasNrm2, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasRot(uint64_t elem_count, DeviceMemory<float> *x,
                            int incx, DeviceMemory<float> *y, int incy, float c,
                            float s) {
   std::vector<std::string> mht_103_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_103(mht_103_v, 2043, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasRot");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(c), PARAM(s));

  ThenBlasImpl<uint64_t, DeviceMemory<float> *, int, DeviceMemory<float> *, int,
               float, float>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRot, elem_count, x, incx, y, incy,
              c, s);
}

Stream &Stream::ThenBlasRot(uint64_t elem_count, DeviceMemory<double> *x,
                            int incx, DeviceMemory<double> *y, int incy,
                            double c, double s) {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_104(mht_104_v, 2059, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasRot");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(c), PARAM(s));

  ThenBlasImpl<uint64_t, DeviceMemory<double> *, int, DeviceMemory<double> *,
               int, double, double>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRot, elem_count, x, incx, y, incy,
              c, s);
}

Stream &Stream::ThenBlasRot(uint64_t elem_count,
                            DeviceMemory<std::complex<float>> *x, int incx,
                            DeviceMemory<std::complex<float>> *y, int incy,
                            float c, float s) {
   std::vector<std::string> mht_105_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_105(mht_105_v, 2076, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasRot");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(c), PARAM(s));

  ThenBlasImpl<uint64_t, DeviceMemory<std::complex<float>> *, int,
               DeviceMemory<std::complex<float>> *, int, float, float>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRot, elem_count, x, incx, y, incy,
              c, s);
}

Stream &Stream::ThenBlasRot(uint64_t elem_count,
                            DeviceMemory<std::complex<double>> *x, int incx,
                            DeviceMemory<std::complex<double>> *y, int incy,
                            double c, double s) {
   std::vector<std::string> mht_106_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_106(mht_106_v, 2093, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasRot");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(c), PARAM(s));

  ThenBlasImpl<uint64_t, DeviceMemory<std::complex<double>> *, int,
               DeviceMemory<std::complex<double>> *, int, double, double>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRot, elem_count, x, incx, y, incy,
              c, s);
}

Stream &Stream::ThenBlasRotg(DeviceMemory<float> *a, DeviceMemory<float> *b,
                             DeviceMemory<float> *c, DeviceMemory<float> *s) {
   std::vector<std::string> mht_107_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_107(mht_107_v, 2108, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasRotg");

  VLOG_CALL(PARAM(a), PARAM(b), PARAM(c), PARAM(s));

  ThenBlasImpl<DeviceMemory<float> *, DeviceMemory<float> *,
               DeviceMemory<float> *, DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRotg, a, b, c, s);
}

Stream &Stream::ThenBlasRotg(DeviceMemory<double> *a, DeviceMemory<double> *b,
                             DeviceMemory<double> *c, DeviceMemory<double> *s) {
   std::vector<std::string> mht_108_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_108(mht_108_v, 2121, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasRotg");

  VLOG_CALL(PARAM(a), PARAM(b), PARAM(c), PARAM(s));

  ThenBlasImpl<DeviceMemory<double> *, DeviceMemory<double> *,
               DeviceMemory<double> *, DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRotg, a, b, c, s);
}

Stream &Stream::ThenBlasRotg(DeviceMemory<std::complex<float>> *a,
                             DeviceMemory<std::complex<float>> *b,
                             DeviceMemory<float> *c,
                             DeviceMemory<std::complex<float>> *s) {
   std::vector<std::string> mht_109_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_109(mht_109_v, 2136, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasRotg");

  VLOG_CALL(PARAM(a), PARAM(b), PARAM(c), PARAM(s));

  ThenBlasImpl<DeviceMemory<std::complex<float>> *,
               DeviceMemory<std::complex<float>> *, DeviceMemory<float> *,
               DeviceMemory<std::complex<float>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRotg, a, b, c, s);
}

Stream &Stream::ThenBlasRotg(DeviceMemory<std::complex<double>> *a,
                             DeviceMemory<std::complex<double>> *b,
                             DeviceMemory<double> *c,
                             DeviceMemory<std::complex<double>> *s) {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_110(mht_110_v, 2152, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasRotg");

  VLOG_CALL(PARAM(a), PARAM(b), PARAM(c), PARAM(s));

  ThenBlasImpl<DeviceMemory<std::complex<double>> *,
               DeviceMemory<std::complex<double>> *, DeviceMemory<double> *,
               DeviceMemory<std::complex<double>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRotg, a, b, c, s);
}

Stream &Stream::ThenBlasRotm(uint64_t elem_count, DeviceMemory<float> *x,
                             int incx, DeviceMemory<float> *y, int incy,
                             const DeviceMemory<float> &param) {
   std::vector<std::string> mht_111_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_111(mht_111_v, 2167, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasRotm");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(param));

  ThenBlasImpl<uint64_t, DeviceMemory<float> *, int, DeviceMemory<float> *, int,
               const DeviceMemory<float> &>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRotm, elem_count, x, incx, y,
              incy, param);
}

Stream &Stream::ThenBlasRotm(uint64_t elem_count, DeviceMemory<double> *x,
                             int incx, DeviceMemory<double> *y, int incy,
                             const DeviceMemory<double> &param) {
   std::vector<std::string> mht_112_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_112(mht_112_v, 2183, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasRotm");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(param));

  ThenBlasImpl<uint64_t, DeviceMemory<double> *, int, DeviceMemory<double> *,
               int, const DeviceMemory<double> &>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRotm, elem_count, x, incx, y,
              incy, param);
}

Stream &Stream::ThenBlasRotmg(DeviceMemory<float> *d1, DeviceMemory<float> *d2,
                              DeviceMemory<float> *x1,
                              const DeviceMemory<float> &y1,
                              DeviceMemory<float> *param) {
   std::vector<std::string> mht_113_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_113(mht_113_v, 2200, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasRotmg");

  VLOG_CALL(PARAM(d1), PARAM(d2), PARAM(x1), PARAM(y1), PARAM(param));

  ThenBlasImpl<DeviceMemory<float> *, DeviceMemory<float> *,
               DeviceMemory<float> *, const DeviceMemory<float> &,
               DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRotmg, d1, d2, x1, y1, param);
}

Stream &Stream::ThenBlasRotmg(DeviceMemory<double> *d1,
                              DeviceMemory<double> *d2,
                              DeviceMemory<double> *x1,
                              const DeviceMemory<double> &y1,
                              DeviceMemory<double> *param) {
   std::vector<std::string> mht_114_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_114(mht_114_v, 2217, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasRotmg");

  VLOG_CALL(PARAM(d1), PARAM(d2), PARAM(x1), PARAM(y1), PARAM(param));

  ThenBlasImpl<DeviceMemory<double> *, DeviceMemory<double> *,
               DeviceMemory<double> *, const DeviceMemory<double> &,
               DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRotmg, d1, d2, x1, y1, param);
}

Stream &Stream::ThenBlasScal(uint64_t elem_count, float alpha,
                             DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_115_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_115(mht_115_v, 2231, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasScal");

  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx));

  ThenBlasImpl<uint64_t, float, DeviceMemory<float> *, int> impl;
  return impl(this, &blas::BlasSupport::DoBlasScal, elem_count, alpha, x, incx);
}

Stream &Stream::ThenBlasScal(uint64_t elem_count, double alpha,
                             DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_116_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_116(mht_116_v, 2242, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasScal");

  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx));

  ThenBlasImpl<uint64_t, double, DeviceMemory<double> *, int> impl;
  return impl(this, &blas::BlasSupport::DoBlasScal, elem_count, alpha, x, incx);
}

Stream &Stream::ThenBlasScal(uint64_t elem_count, float alpha,
                             DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_117_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_117(mht_117_v, 2253, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasScal");

  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx));

  ThenBlasImpl<uint64_t, float, DeviceMemory<std::complex<float>> *, int> impl;
  return impl(this, &blas::BlasSupport::DoBlasScal, elem_count, alpha, x, incx);
}

Stream &Stream::ThenBlasScal(uint64_t elem_count, double alpha,
                             DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_118_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_118(mht_118_v, 2264, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasScal");

  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx));

  ThenBlasImpl<uint64_t, double, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasScal, elem_count, alpha, x, incx);
}

Stream &Stream::ThenBlasScal(uint64_t elem_count, std::complex<float> alpha,
                             DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_119_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_119(mht_119_v, 2276, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasScal");

  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx));

  ThenBlasImpl<uint64_t, std::complex<float>,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasScal, elem_count, alpha, x, incx);
}

Stream &Stream::ThenBlasScal(uint64_t elem_count, std::complex<double> alpha,
                             DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_120_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_120(mht_120_v, 2289, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasScal");

  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx));

  ThenBlasImpl<uint64_t, std::complex<double>,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasScal, elem_count, alpha, x, incx);
}

Stream &Stream::ThenBlasSwap(uint64_t elem_count, DeviceMemory<float> *x,
                             int incx, DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_121_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_121(mht_121_v, 2302, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSwap");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy));

  ThenBlasImpl<uint64_t, DeviceMemory<float> *, int, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSwap, elem_count, x, incx, y,
              incy);
}

Stream &Stream::ThenBlasSwap(uint64_t elem_count, DeviceMemory<double> *x,
                             int incx, DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_122_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_122(mht_122_v, 2315, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSwap");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy));

  ThenBlasImpl<uint64_t, DeviceMemory<double> *, int, DeviceMemory<double> *,
               int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSwap, elem_count, x, incx, y,
              incy);
}

Stream &Stream::ThenBlasSwap(uint64_t elem_count,
                             DeviceMemory<std::complex<float>> *x, int incx,
                             DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_123_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_123(mht_123_v, 2330, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSwap");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy));

  ThenBlasImpl<uint64_t, DeviceMemory<std::complex<float>> *, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSwap, elem_count, x, incx, y,
              incy);
}

Stream &Stream::ThenBlasSwap(uint64_t elem_count,
                             DeviceMemory<std::complex<double>> *x, int incx,
                             DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_124_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_124(mht_124_v, 2345, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSwap");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy));

  ThenBlasImpl<uint64_t, DeviceMemory<std::complex<double>> *, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSwap, elem_count, x, incx, y,
              incy);
}

Stream &Stream::ThenBlasIamax(uint64_t elem_count, const DeviceMemory<float> &x,
                              int incx, DeviceMemory<int> *result) {
   std::vector<std::string> mht_125_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_125(mht_125_v, 2359, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasIamax");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<float> &, int, DeviceMemory<int> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasIamax, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasIamax(uint64_t elem_count,
                              const DeviceMemory<double> &x, int incx,
                              DeviceMemory<int> *result) {
   std::vector<std::string> mht_126_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_126(mht_126_v, 2373, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasIamax");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<double> &, int, DeviceMemory<int> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasIamax, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasIamax(uint64_t elem_count,
                              const DeviceMemory<std::complex<float>> &x,
                              int incx, DeviceMemory<int> *result) {
   std::vector<std::string> mht_127_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_127(mht_127_v, 2387, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasIamax");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<int> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasIamax, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasIamax(uint64_t elem_count,
                              const DeviceMemory<std::complex<double>> &x,
                              int incx, DeviceMemory<int> *result) {
   std::vector<std::string> mht_128_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_128(mht_128_v, 2402, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasIamax");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<int> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasIamax, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasIamin(uint64_t elem_count, const DeviceMemory<float> &x,
                              int incx, DeviceMemory<int> *result) {
   std::vector<std::string> mht_129_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_129(mht_129_v, 2416, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasIamin");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<float> &, int, DeviceMemory<int> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasIamin, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasIamin(uint64_t elem_count,
                              const DeviceMemory<double> &x, int incx,
                              DeviceMemory<int> *result) {
   std::vector<std::string> mht_130_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_130(mht_130_v, 2430, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasIamin");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<double> &, int, DeviceMemory<int> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasIamin, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasIamin(uint64_t elem_count,
                              const DeviceMemory<std::complex<float>> &x,
                              int incx, DeviceMemory<int> *result) {
   std::vector<std::string> mht_131_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_131(mht_131_v, 2444, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasIamin");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<int> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasIamin, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasIamin(uint64_t elem_count,
                              const DeviceMemory<std::complex<double>> &x,
                              int incx, DeviceMemory<int> *result) {
   std::vector<std::string> mht_132_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_132(mht_132_v, 2459, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasIamin");

  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<int> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasIamin, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasGbmv(blas::Transpose trans, uint64_t m, uint64 n,
                             uint64_t kl, uint64 ku, float alpha,
                             const DeviceMemory<float> &a, int lda,
                             const DeviceMemory<float> &x, int incx, float beta,
                             DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_133_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_133(mht_133_v, 2476, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGbmv");

  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(kl), PARAM(ku),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(x), PARAM(incx),
            PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::Transpose, uint64_t, uint64_t, uint64, uint64, float,
               const DeviceMemory<float> &, int, const DeviceMemory<float> &,
               int, float, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGbmv, trans, m, n, kl, ku, alpha,
              a, lda, x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasGbmv(blas::Transpose trans, uint64_t m, uint64 n,
                             uint64_t kl, uint64 ku, double alpha,
                             const DeviceMemory<double> &a, int lda,
                             const DeviceMemory<double> &x, int incx,
                             double beta, DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_134_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_134(mht_134_v, 2496, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGbmv");

  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(kl), PARAM(ku),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(x), PARAM(incx),
            PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::Transpose, uint64_t, uint64_t, uint64, uint64, double,
               const DeviceMemory<double> &, int, const DeviceMemory<double> &,
               int, double, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGbmv, trans, m, n, kl, ku, alpha,
              a, lda, x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasGbmv(blas::Transpose trans, uint64_t m, uint64 n,
                             uint64_t kl, uint64 ku, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, std::complex<float> beta,
                             DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_135_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_135(mht_135_v, 2518, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGbmv");

  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(kl), PARAM(ku),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(x), PARAM(incx),
            PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::Transpose, uint64_t, uint64_t, uint64, uint64,
               std::complex<float>, const DeviceMemory<std::complex<float>> &,
               int, const DeviceMemory<std::complex<float>> &, int,
               std::complex<float>, DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGbmv, trans, m, n, kl, ku, alpha,
              a, lda, x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasGbmv(blas::Transpose trans, uint64_t m, uint64 n,
                             uint64_t kl, uint64 ku, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, std::complex<double> beta,
                             DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_136_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_136(mht_136_v, 2541, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGbmv");

  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(kl), PARAM(ku),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(x), PARAM(incx),
            PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::Transpose, uint64_t, uint64_t, uint64, uint64,
               std::complex<double>, const DeviceMemory<std::complex<double>> &,
               int, const DeviceMemory<std::complex<double>> &, int,
               std::complex<double>, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGbmv, trans, m, n, kl, ku, alpha,
              a, lda, x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasGemv(blas::Transpose trans, uint64_t m, uint64 n,
                             float alpha, const DeviceMemory<float> &a, int lda,
                             const DeviceMemory<float> &x, int incx, float beta,
                             DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_137_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_137(mht_137_v, 2561, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemv");

  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(alpha), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx), PARAM(beta), PARAM(y),
            PARAM(incy));

  ThenBlasImpl<blas::Transpose, uint64_t, uint64_t, float,
               const DeviceMemory<float> &, int, const DeviceMemory<float> &,
               int, float, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemv, trans, m, n, alpha, a, lda,
              x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasGemv(blas::Transpose trans, uint64_t m, uint64 n,
                             double alpha, const DeviceMemory<double> &a,
                             int lda, const DeviceMemory<double> &x, int incx,
                             double beta, DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_138_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_138(mht_138_v, 2580, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemv");

  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(alpha), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx), PARAM(beta), PARAM(y),
            PARAM(incy));

  ThenBlasImpl<blas::Transpose, uint64_t, uint64_t, double,
               const DeviceMemory<double> &, int, const DeviceMemory<double> &,
               int, double, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemv, trans, m, n, alpha, a, lda,
              x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasGemv(blas::Transpose trans, uint64_t m, uint64 n,
                             std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, std::complex<float> beta,
                             DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_139_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_139(mht_139_v, 2602, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemv");

  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(alpha), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx), PARAM(beta), PARAM(y),
            PARAM(incy));

  ThenBlasImpl<blas::Transpose, uint64_t, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               std::complex<float>, DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemv, trans, m, n, alpha, a, lda,
              x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasGemv(blas::Transpose trans, uint64_t m, uint64 n,
                             std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, std::complex<double> beta,
                             DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_140_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_140(mht_140_v, 2625, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemv");

  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(alpha), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx), PARAM(beta), PARAM(y),
            PARAM(incy));

  ThenBlasImpl<blas::Transpose, uint64_t, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               std::complex<double>, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemv, trans, m, n, alpha, a, lda,
              x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasGer(uint64_t m, uint64 n, float alpha,
                            const DeviceMemory<float> &x, int incx,
                            const DeviceMemory<float> &y, int incy,
                            DeviceMemory<float> *a, int lda) {
   std::vector<std::string> mht_141_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_141(mht_141_v, 2645, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGer");

  VLOG_CALL(PARAM(m), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<uint64_t, uint64_t, float, const DeviceMemory<float> &, int,
               const DeviceMemory<float> &, int, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGer, m, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasGer(uint64_t m, uint64 n, double alpha,
                            const DeviceMemory<double> &x, int incx,
                            const DeviceMemory<double> &y, int incy,
                            DeviceMemory<double> *a, int lda) {
   std::vector<std::string> mht_142_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_142(mht_142_v, 2662, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGer");

  VLOG_CALL(PARAM(m), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<uint64_t, uint64_t, double, const DeviceMemory<double> &, int,
               const DeviceMemory<double> &, int, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGer, m, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasGerc(uint64_t m, uint64 n, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx,
                             const DeviceMemory<std::complex<float>> &y,
                             int incy, DeviceMemory<std::complex<float>> *a,
                             int lda) {
   std::vector<std::string> mht_143_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_143(mht_143_v, 2681, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGerc");

  VLOG_CALL(PARAM(m), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<uint64_t, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGerc, m, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasGerc(uint64_t m, uint64 n, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx,
                             const DeviceMemory<std::complex<double>> &y,
                             int incy, DeviceMemory<std::complex<double>> *a,
                             int lda) {
   std::vector<std::string> mht_144_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_144(mht_144_v, 2702, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGerc");

  VLOG_CALL(PARAM(m), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<uint64_t, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGerc, m, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasGeru(uint64_t m, uint64 n, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx,
                             const DeviceMemory<std::complex<float>> &y,
                             int incy, DeviceMemory<std::complex<float>> *a,
                             int lda) {
   std::vector<std::string> mht_145_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_145(mht_145_v, 2723, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGeru");

  VLOG_CALL(PARAM(m), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<uint64_t, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGeru, m, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasGeru(uint64_t m, uint64 n, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx,
                             const DeviceMemory<std::complex<double>> &y,
                             int incy, DeviceMemory<std::complex<double>> *a,
                             int lda) {
   std::vector<std::string> mht_146_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_146(mht_146_v, 2744, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGeru");

  VLOG_CALL(PARAM(m), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<uint64_t, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGeru, m, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasHbmv(blas::UpperLower uplo, uint64_t n, uint64 k,
                             std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, std::complex<float> beta,
                             DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_147_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_147(mht_147_v, 2766, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHbmv");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(k), PARAM(alpha), PARAM(a), PARAM(lda),
            PARAM(x), PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               std::complex<float>, DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHbmv, uplo, n, k, alpha, a, lda,
              x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasHbmv(blas::UpperLower uplo, uint64_t n, uint64 k,
                             std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, std::complex<double> beta,
                             DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_148_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_148(mht_148_v, 2788, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHbmv");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(k), PARAM(alpha), PARAM(a), PARAM(lda),
            PARAM(x), PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               std::complex<double>, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHbmv, uplo, n, k, alpha, a, lda,
              x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasHemv(blas::UpperLower uplo, uint64_t n,
                             std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, std::complex<float> beta,
                             DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_149_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_149(mht_149_v, 2810, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHemv");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(x),
            PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               std::complex<float>, DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHemv, uplo, n, alpha, a, lda, x,
              incx, beta, y, incy);
}

Stream &Stream::ThenBlasHemv(blas::UpperLower uplo, uint64_t n,
                             std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, std::complex<double> beta,
                             DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_150_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_150(mht_150_v, 2832, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHemv");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(x),
            PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               std::complex<double>, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHemv, uplo, n, alpha, a, lda, x,
              incx, beta, y, incy);
}

Stream &Stream::ThenBlasHer(blas::UpperLower uplo, uint64_t n, float alpha,
                            const DeviceMemory<std::complex<float>> &x,
                            int incx, DeviceMemory<std::complex<float>> *a,
                            int lda) {
   std::vector<std::string> mht_151_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_151(mht_151_v, 2851, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHer");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(a), PARAM(lda));

  ThenBlasImpl<blas::UpperLower, uint64_t, float,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHer, uplo, n, alpha, x, incx, a,
              lda);
}

Stream &Stream::ThenBlasHer(blas::UpperLower uplo, uint64_t n, double alpha,
                            const DeviceMemory<std::complex<double>> &x,
                            int incx, DeviceMemory<std::complex<double>> *a,
                            int lda) {
   std::vector<std::string> mht_152_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_152(mht_152_v, 2869, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHer");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(a), PARAM(lda));

  ThenBlasImpl<blas::UpperLower, uint64_t, double,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHer, uplo, n, alpha, x, incx, a,
              lda);
}

Stream &Stream::ThenBlasHer2(blas::UpperLower uplo, uint64_t n,
                             std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx,
                             const DeviceMemory<std::complex<float>> &y,
                             int incy, DeviceMemory<std::complex<float>> *a,
                             int lda) {
   std::vector<std::string> mht_153_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_153(mht_153_v, 2890, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHer2");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(y), PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<blas::UpperLower, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHer2, uplo, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasHer2(blas::UpperLower uplo, uint64_t n,
                             std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx,
                             const DeviceMemory<std::complex<double>> &y,
                             int incy, DeviceMemory<std::complex<double>> *a,
                             int lda) {
   std::vector<std::string> mht_154_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_154(mht_154_v, 2912, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHer2");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(y), PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<blas::UpperLower, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHer2, uplo, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasHpmv(blas::UpperLower uplo, uint64_t n,
                             std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &ap,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, std::complex<float> beta,
                             DeviceMemory<std::complex<float>> *y, int incy) {
   std::vector<std::string> mht_155_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_155(mht_155_v, 2933, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHpmv");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(ap), PARAM(x),
            PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &,
               const DeviceMemory<std::complex<float>> &, int,
               std::complex<float>, DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHpmv, uplo, n, alpha, ap, x, incx,
              beta, y, incy);
}

Stream &Stream::ThenBlasHpmv(blas::UpperLower uplo, uint64_t n,
                             std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &ap,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, std::complex<double> beta,
                             DeviceMemory<std::complex<double>> *y, int incy) {
   std::vector<std::string> mht_156_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_156(mht_156_v, 2954, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHpmv");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(ap), PARAM(x),
            PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &,
               const DeviceMemory<std::complex<double>> &, int,
               std::complex<double>, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHpmv, uplo, n, alpha, ap, x, incx,
              beta, y, incy);
}

Stream &Stream::ThenBlasHpr(blas::UpperLower uplo, uint64_t n, float alpha,
                            const DeviceMemory<std::complex<float>> &x,
                            int incx, DeviceMemory<std::complex<float>> *ap) {
   std::vector<std::string> mht_157_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_157(mht_157_v, 2972, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHpr");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(ap));

  ThenBlasImpl<blas::UpperLower, uint64_t, float,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHpr, uplo, n, alpha, x, incx, ap);
}

Stream &Stream::ThenBlasHpr(blas::UpperLower uplo, uint64_t n, double alpha,
                            const DeviceMemory<std::complex<double>> &x,
                            int incx, DeviceMemory<std::complex<double>> *ap) {
   std::vector<std::string> mht_158_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_158(mht_158_v, 2988, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHpr");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(ap));

  ThenBlasImpl<blas::UpperLower, uint64_t, double,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHpr, uplo, n, alpha, x, incx, ap);
}

Stream &Stream::ThenBlasHpr2(blas::UpperLower uplo, uint64_t n,
                             std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx,
                             const DeviceMemory<std::complex<float>> &y,
                             int incy, DeviceMemory<std::complex<float>> *ap) {
   std::vector<std::string> mht_159_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_159(mht_159_v, 3007, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHpr2");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(y), PARAM(incy), PARAM(ap));

  ThenBlasImpl<blas::UpperLower, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHpr2, uplo, n, alpha, x, incx, y,
              incy, ap);
}

Stream &Stream::ThenBlasHpr2(blas::UpperLower uplo, uint64_t n,
                             std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx,
                             const DeviceMemory<std::complex<double>> &y,
                             int incy, DeviceMemory<std::complex<double>> *ap) {
   std::vector<std::string> mht_160_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_160(mht_160_v, 3028, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHpr2");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(y), PARAM(incy), PARAM(ap));

  ThenBlasImpl<blas::UpperLower, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHpr2, uplo, n, alpha, x, incx, y,
              incy, ap);
}

Stream &Stream::ThenBlasSbmv(blas::UpperLower uplo, uint64_t n, uint64 k,
                             float alpha, const DeviceMemory<float> &a, int lda,
                             const DeviceMemory<float> &x, int incx, float beta,
                             DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_161_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_161(mht_161_v, 3047, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSbmv");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(k), PARAM(alpha), PARAM(a), PARAM(lda),
            PARAM(x), PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, uint64_t, float,
               const DeviceMemory<float> &, int, const DeviceMemory<float> &,
               int, float, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSbmv, uplo, n, k, alpha, a, lda,
              x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasSbmv(blas::UpperLower uplo, uint64_t n, uint64 k,
                             double alpha, const DeviceMemory<double> &a,
                             int lda, const DeviceMemory<double> &x, int incx,
                             double beta, DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_162_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_162(mht_162_v, 3065, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSbmv");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(k), PARAM(alpha), PARAM(a), PARAM(lda),
            PARAM(x), PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, uint64_t, double,
               const DeviceMemory<double> &, int, const DeviceMemory<double> &,
               int, double, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSbmv, uplo, n, k, alpha, a, lda,
              x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasSpmv(blas::UpperLower uplo, uint64_t n, float alpha,
                             const DeviceMemory<float> &ap,
                             const DeviceMemory<float> &x, int incx, float beta,
                             DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_163_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_163(mht_163_v, 3083, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSpmv");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(ap), PARAM(x),
            PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, float, const DeviceMemory<float> &,
               const DeviceMemory<float> &, int, float, DeviceMemory<float> *,
               int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSpmv, uplo, n, alpha, ap, x, incx,
              beta, y, incy);
}

Stream &Stream::ThenBlasSpmv(blas::UpperLower uplo, uint64_t n, double alpha,
                             const DeviceMemory<double> &ap,
                             const DeviceMemory<double> &x, int incx,
                             double beta, DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_164_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_164(mht_164_v, 3101, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSpmv");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(ap), PARAM(x),
            PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, double, const DeviceMemory<double> &,
               const DeviceMemory<double> &, int, double,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSpmv, uplo, n, alpha, ap, x, incx,
              beta, y, incy);
}

Stream &Stream::ThenBlasSpr(blas::UpperLower uplo, uint64_t n, float alpha,
                            const DeviceMemory<float> &x, int incx,
                            DeviceMemory<float> *ap) {
   std::vector<std::string> mht_165_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_165(mht_165_v, 3118, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSpr");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(ap));

  ThenBlasImpl<blas::UpperLower, uint64_t, float, const DeviceMemory<float> &,
               int, DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSpr, uplo, n, alpha, x, incx, ap);
}

Stream &Stream::ThenBlasSpr(blas::UpperLower uplo, uint64_t n, double alpha,
                            const DeviceMemory<double> &x, int incx,
                            DeviceMemory<double> *ap) {
   std::vector<std::string> mht_166_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_166(mht_166_v, 3133, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSpr");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(ap));

  ThenBlasImpl<blas::UpperLower, uint64_t, double, const DeviceMemory<double> &,
               int, DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSpr, uplo, n, alpha, x, incx, ap);
}

Stream &Stream::ThenBlasSpr2(blas::UpperLower uplo, uint64_t n, float alpha,
                             const DeviceMemory<float> &x, int incx,
                             const DeviceMemory<float> &y, int incy,
                             DeviceMemory<float> *ap) {
   std::vector<std::string> mht_167_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_167(mht_167_v, 3149, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSpr2");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(y), PARAM(incy), PARAM(ap));

  ThenBlasImpl<blas::UpperLower, uint64_t, float, const DeviceMemory<float> &,
               int, const DeviceMemory<float> &, int, DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSpr2, uplo, n, alpha, x, incx, y,
              incy, ap);
}

Stream &Stream::ThenBlasSpr2(blas::UpperLower uplo, uint64_t n, double alpha,
                             const DeviceMemory<double> &x, int incx,
                             const DeviceMemory<double> &y, int incy,
                             DeviceMemory<double> *ap) {
   std::vector<std::string> mht_168_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_168(mht_168_v, 3166, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSpr2");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(y), PARAM(incy), PARAM(ap));

  ThenBlasImpl<blas::UpperLower, uint64_t, double, const DeviceMemory<double> &,
               int, const DeviceMemory<double> &, int, DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSpr2, uplo, n, alpha, x, incx, y,
              incy, ap);
}

Stream &Stream::ThenBlasSymv(blas::UpperLower uplo, uint64_t n, float alpha,
                             const DeviceMemory<float> &a, int lda,
                             const DeviceMemory<float> &x, int incx, float beta,
                             DeviceMemory<float> *y, int incy) {
   std::vector<std::string> mht_169_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_169(mht_169_v, 3183, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSymv");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(x),
            PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, float, const DeviceMemory<float> &,
               int, const DeviceMemory<float> &, int, float,
               DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSymv, uplo, n, alpha, a, lda, x,
              incx, beta, y, incy);
}

Stream &Stream::ThenBlasSymv(blas::UpperLower uplo, uint64_t n, double alpha,
                             const DeviceMemory<double> &a, int lda,
                             const DeviceMemory<double> &x, int incx,
                             double beta, DeviceMemory<double> *y, int incy) {
   std::vector<std::string> mht_170_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_170(mht_170_v, 3201, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSymv");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(x),
            PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, double, const DeviceMemory<double> &,
               int, const DeviceMemory<double> &, int, double,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSymv, uplo, n, alpha, a, lda, x,
              incx, beta, y, incy);
}

Stream &Stream::ThenBlasSyr(blas::UpperLower uplo, uint64_t n, float alpha,
                            const DeviceMemory<float> &x, int incx,
                            DeviceMemory<float> *a, int lda) {
   std::vector<std::string> mht_171_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_171(mht_171_v, 3218, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSyr");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(a), PARAM(lda));

  ThenBlasImpl<blas::UpperLower, uint64_t, float, const DeviceMemory<float> &,
               int, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyr, uplo, n, alpha, x, incx, a,
              lda);
}

Stream &Stream::ThenBlasSyr(blas::UpperLower uplo, uint64_t n, double alpha,
                            const DeviceMemory<double> &x, int incx,
                            DeviceMemory<double> *a, int lda) {
   std::vector<std::string> mht_172_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_172(mht_172_v, 3234, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSyr");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(a), PARAM(lda));

  ThenBlasImpl<blas::UpperLower, uint64_t, double, const DeviceMemory<double> &,
               int, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyr, uplo, n, alpha, x, incx, a,
              lda);
}

Stream &Stream::ThenBlasSyr2(blas::UpperLower uplo, uint64_t n, float alpha,
                             const DeviceMemory<float> &x, int incx,
                             const DeviceMemory<float> &y, int incy,
                             DeviceMemory<float> *a, int lda) {
   std::vector<std::string> mht_173_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_173(mht_173_v, 3251, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSyr2");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(y), PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<blas::UpperLower, uint64_t, float, const DeviceMemory<float> &,
               int, const DeviceMemory<float> &, int, DeviceMemory<float> *,
               int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyr2, uplo, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasSyr2(blas::UpperLower uplo, uint64_t n, double alpha,
                             const DeviceMemory<double> &x, int incx,
                             const DeviceMemory<double> &y, int incy,
                             DeviceMemory<double> *a, int lda) {
   std::vector<std::string> mht_174_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_174(mht_174_v, 3269, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSyr2");

  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(y), PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<blas::UpperLower, uint64_t, double, const DeviceMemory<double> &,
               int, const DeviceMemory<double> &, int, DeviceMemory<double> *,
               int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyr2, uplo, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasTbmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n, uint64 k,
                             const DeviceMemory<float> &a, int lda,
                             DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_175_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_175(mht_175_v, 3287, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTbmv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(k),
            PARAM(a), PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               uint64_t, const DeviceMemory<float> &, int,
               DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTbmv, uplo, trans, diag, n, k, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTbmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n, uint64 k,
                             const DeviceMemory<double> &a, int lda,
                             DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_176_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_176(mht_176_v, 3305, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTbmv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(k),
            PARAM(a), PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               uint64_t, const DeviceMemory<double> &, int,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTbmv, uplo, trans, diag, n, k, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTbmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n, uint64 k,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda, DeviceMemory<std::complex<float>> *x,
                             int incx) {
   std::vector<std::string> mht_177_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_177(mht_177_v, 3324, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTbmv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(k),
            PARAM(a), PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               uint64_t, const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTbmv, uplo, trans, diag, n, k, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTbmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n, uint64 k,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda, DeviceMemory<std::complex<double>> *x,
                             int incx) {
   std::vector<std::string> mht_178_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_178(mht_178_v, 3343, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTbmv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(k),
            PARAM(a), PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               uint64_t, const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTbmv, uplo, trans, diag, n, k, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTbsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n, uint64 k,
                             const DeviceMemory<float> &a, int lda,
                             DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_179_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_179(mht_179_v, 3361, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTbsv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(k),
            PARAM(a), PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               uint64_t, const DeviceMemory<float> &, int,
               DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTbsv, uplo, trans, diag, n, k, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTbsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n, uint64 k,
                             const DeviceMemory<double> &a, int lda,
                             DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_180_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_180(mht_180_v, 3379, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTbsv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(k),
            PARAM(a), PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               uint64_t, const DeviceMemory<double> &, int,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTbsv, uplo, trans, diag, n, k, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTbsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n, uint64 k,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda, DeviceMemory<std::complex<float>> *x,
                             int incx) {
   std::vector<std::string> mht_181_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_181(mht_181_v, 3398, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTbsv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(k),
            PARAM(a), PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               uint64_t, const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTbsv, uplo, trans, diag, n, k, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTbsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n, uint64 k,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda, DeviceMemory<std::complex<double>> *x,
                             int incx) {
   std::vector<std::string> mht_182_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_182(mht_182_v, 3417, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTbsv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(k),
            PARAM(a), PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               uint64_t, const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTbsv, uplo, trans, diag, n, k, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTpmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<float> &ap,
                             DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_183_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_183(mht_183_v, 3435, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTpmv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(ap),
            PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<float> &, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTpmv, uplo, trans, diag, n, ap, x,
              incx);
}

Stream &Stream::ThenBlasTpmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<double> &ap,
                             DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_184_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_184(mht_184_v, 3452, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTpmv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(ap),
            PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<double> &, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTpmv, uplo, trans, diag, n, ap, x,
              incx);
}

Stream &Stream::ThenBlasTpmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<std::complex<float>> &ap,
                             DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_185_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_185(mht_185_v, 3469, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTpmv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(ap),
            PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<std::complex<float>> &,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTpmv, uplo, trans, diag, n, ap, x,
              incx);
}

Stream &Stream::ThenBlasTpmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<std::complex<double>> &ap,
                             DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_186_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_186(mht_186_v, 3487, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTpmv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(ap),
            PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<std::complex<double>> &,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTpmv, uplo, trans, diag, n, ap, x,
              incx);
}

Stream &Stream::ThenBlasTpsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<float> &ap,
                             DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_187_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_187(mht_187_v, 3505, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTpsv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(ap),
            PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<float> &, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTpsv, uplo, trans, diag, n, ap, x,
              incx);
}

Stream &Stream::ThenBlasTpsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<double> &ap,
                             DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_188_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_188(mht_188_v, 3522, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTpsv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(ap),
            PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<double> &, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTpsv, uplo, trans, diag, n, ap, x,
              incx);
}

Stream &Stream::ThenBlasTpsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<std::complex<float>> &ap,
                             DeviceMemory<std::complex<float>> *x, int incx) {
   std::vector<std::string> mht_189_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_189(mht_189_v, 3539, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTpsv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(ap),
            PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<std::complex<float>> &,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTpsv, uplo, trans, diag, n, ap, x,
              incx);
}

Stream &Stream::ThenBlasTpsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<std::complex<double>> &ap,
                             DeviceMemory<std::complex<double>> *x, int incx) {
   std::vector<std::string> mht_190_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_190(mht_190_v, 3557, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTpsv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(ap),
            PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<std::complex<double>> &,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTpsv, uplo, trans, diag, n, ap, x,
              incx);
}

Stream &Stream::ThenBlasTrmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<float> &a, int lda,
                             DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_191_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_191(mht_191_v, 3575, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrmv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<float> &, int, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrmv, uplo, trans, diag, n, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTrmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<double> &a, int lda,
                             DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_192_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_192(mht_192_v, 3592, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrmv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<double> &, int, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrmv, uplo, trans, diag, n, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTrmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda, DeviceMemory<std::complex<float>> *x,
                             int incx) {
   std::vector<std::string> mht_193_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_193(mht_193_v, 3610, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrmv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrmv, uplo, trans, diag, n, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTrmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda, DeviceMemory<std::complex<double>> *x,
                             int incx) {
   std::vector<std::string> mht_194_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_194(mht_194_v, 3629, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrmv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrmv, uplo, trans, diag, n, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTrsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<float> &a, int lda,
                             DeviceMemory<float> *x, int incx) {
   std::vector<std::string> mht_195_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_195(mht_195_v, 3647, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrsv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<float> &, int, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsv, uplo, trans, diag, n, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTrsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<double> &a, int lda,
                             DeviceMemory<double> *x, int incx) {
   std::vector<std::string> mht_196_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_196(mht_196_v, 3664, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrsv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<double> &, int, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsv, uplo, trans, diag, n, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTrsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda, DeviceMemory<std::complex<float>> *x,
                             int incx) {
   std::vector<std::string> mht_197_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_197(mht_197_v, 3682, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrsv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsv, uplo, trans, diag, n, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTrsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda, DeviceMemory<std::complex<double>> *x,
                             int incx) {
   std::vector<std::string> mht_198_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_198(mht_198_v, 3701, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrsv");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsv, uplo, trans, diag, n, a,
              lda, x, incx);
}

namespace {
// Like ThenBlasImpl, except this expects the last argument of blas_func to be a
// blas::ProfileResult*.  This functor doesn't put the stream into an error
// state if the op fails and the profile result is non-null.  Instead, the
// error-ness is returned in the profile result itself.
template <typename... Args>
struct ThenBlasWithProfileImpl {
  Stream &operator()(Stream *stream,
                     bool (blas::BlasSupport::*blas_func)(
                         Stream *, Args..., blas::ProfileResult *),
                     Args... args, blas::ProfileResult *profile_result) {
    ThenBlasImpl<Args..., blas::ProfileResult *> Runner;
    bool record_error = profile_result == nullptr;
    return Runner.Run(stream, blas_func, record_error, args..., profile_result);
  }
};
}  // anonymous namespace

Stream &Stream::ThenBlasGemvWithProfiling(
    blas::Transpose trans, uint64_t m, uint64 n, float alpha,
    const DeviceMemory<float> &a, int lda, const DeviceMemory<float> &x,
    int incx, float beta, DeviceMemory<float> *y, int incy,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_199_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_199(mht_199_v, 3738, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemvWithProfiling");

  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(alpha), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx), PARAM(beta), PARAM(y),
            PARAM(incy));

  ThenBlasWithProfileImpl<
      blas::Transpose, uint64_t, uint64_t, float, const DeviceMemory<float> &,
      int, const DeviceMemory<float> &, int, float, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemvWithProfiling, trans, m, n,
              alpha, a, lda, x, incx, beta, y, incy, output_profile_result);
}

Stream &Stream::ThenBlasGemvWithProfiling(
    blas::Transpose trans, uint64_t m, uint64 n, double alpha,
    const DeviceMemory<double> &a, int lda, const DeviceMemory<double> &x,
    int incx, double beta, DeviceMemory<double> *y, int incy,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_200_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_200(mht_200_v, 3758, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemvWithProfiling");

  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(alpha), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx), PARAM(beta), PARAM(y),
            PARAM(incy));

  ThenBlasWithProfileImpl<blas::Transpose, uint64_t, uint64_t, double,
                          const DeviceMemory<double> &, int,
                          const DeviceMemory<double> &, int, double,
                          DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemvWithProfiling, trans, m, n,
              alpha, a, lda, x, incx, beta, y, incy, output_profile_result);
}

Stream &Stream::ThenBlasGemvWithProfiling(
    blas::Transpose trans, uint64_t m, uint64 n, std::complex<float> alpha,
    const DeviceMemory<std::complex<float>> &a, int lda,
    const DeviceMemory<std::complex<float>> &x, int incx,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *y, int incy,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_201_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_201(mht_201_v, 3780, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemvWithProfiling");

  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(alpha), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx), PARAM(beta), PARAM(y),
            PARAM(incy));

  ThenBlasWithProfileImpl<
      blas::Transpose, uint64_t, uint64_t, std::complex<float>,
      const DeviceMemory<std::complex<float>> &, int,
      const DeviceMemory<std::complex<float>> &, int, std::complex<float>,
      DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemvWithProfiling, trans, m, n,
              alpha, a, lda, x, incx, beta, y, incy, output_profile_result);
}

Stream &Stream::ThenBlasGemvWithProfiling(
    blas::Transpose trans, uint64_t m, uint64 n, std::complex<double> alpha,
    const DeviceMemory<std::complex<double>> &a, int lda,
    const DeviceMemory<std::complex<double>> &x, int incx,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *y, int incy,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_202_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_202(mht_202_v, 3803, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemvWithProfiling");

  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(alpha), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx), PARAM(beta), PARAM(y),
            PARAM(incy));

  ThenBlasWithProfileImpl<
      blas::Transpose, uint64_t, uint64_t, std::complex<double>,
      const DeviceMemory<std::complex<double>> &, int,
      const DeviceMemory<std::complex<double>> &, int, std::complex<double>,
      DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemvWithProfiling, trans, m, n,
              alpha, a, lda, x, incx, beta, y, incy, output_profile_result);
}

Stream &Stream::ThenBlasGemmWithProfiling(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, float alpha, const DeviceMemory<Eigen::half> &a, int lda,
    const DeviceMemory<Eigen::half> &b, int ldb, float beta,
    DeviceMemory<Eigen::half> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_203_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_203(mht_203_v, 3826, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemmWithProfiling");

  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasWithProfileImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t,
                          uint64_t, float, const DeviceMemory<Eigen::half> &,
                          int, const DeviceMemory<Eigen::half> &, int, float,
                          DeviceMemory<Eigen::half> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmWithProfiling, transa, transb,
              m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
              output_profile_result);
}

Stream &Stream::ThenBlasGemmWithProfiling(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, float alpha, const DeviceMemory<float> &a, int lda,
    const DeviceMemory<float> &b, int ldb, float beta, DeviceMemory<float> *c,
    int ldc, blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_204_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_204(mht_204_v, 3848, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemmWithProfiling");

  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasWithProfileImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t,
                          uint64_t, float, const DeviceMemory<float> &, int,
                          const DeviceMemory<float> &, int, float,
                          DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmWithProfiling, transa, transb,
              m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
              output_profile_result);
}

Stream &Stream::ThenBlasGemmWithProfiling(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, double alpha, const DeviceMemory<double> &a, int lda,
    const DeviceMemory<double> &b, int ldb, double beta,
    DeviceMemory<double> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_205_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_205(mht_205_v, 3871, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemmWithProfiling");

  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasWithProfileImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t,
                          uint64_t, double, const DeviceMemory<double> &, int,
                          const DeviceMemory<double> &, int, double,
                          DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmWithProfiling, transa, transb,
              m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
              output_profile_result);
}

Stream &Stream::ThenBlasGemmWithProfiling(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, std::complex<float> alpha,
    const DeviceMemory<std::complex<float>> &a, int lda,
    const DeviceMemory<std::complex<float>> &b, int ldb,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_206_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_206(mht_206_v, 3895, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemmWithProfiling");

  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasWithProfileImpl<
      blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
      std::complex<float>, const DeviceMemory<std::complex<float>> &, int,
      const DeviceMemory<std::complex<float>> &, int, std::complex<float>,
      DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmWithProfiling, transa, transb,
              m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
              output_profile_result);
}

Stream &Stream::ThenBlasGemmWithProfiling(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, std::complex<double> alpha,
    const DeviceMemory<std::complex<double>> &a, int lda,
    const DeviceMemory<std::complex<double>> &b, int ldb,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_207_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_207(mht_207_v, 3920, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemmWithProfiling");

  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasWithProfileImpl<
      blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
      std::complex<double>, const DeviceMemory<std::complex<double>> &, int,
      const DeviceMemory<std::complex<double>> &, int, std::complex<double>,
      DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmWithProfiling, transa, transb,
              m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
              output_profile_result);
}

Stream &Stream::ThenBlasHemm(blas::Side side, blas::UpperLower uplo, uint64_t m,
                             uint64_t n, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda,
                             const DeviceMemory<std::complex<float>> &b,
                             int ldb, std::complex<float> beta,
                             DeviceMemory<std::complex<float>> *c, int ldc) {
   std::vector<std::string> mht_208_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_208(mht_208_v, 3945, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHemm");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(m), PARAM(n), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::Side, blas::UpperLower, uint64_t, uint64_t,
               std::complex<float>, const DeviceMemory<std::complex<float>> &,
               int, const DeviceMemory<std::complex<float>> &, int,
               std::complex<float>, DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHemm, side, uplo, m, n, alpha, a,
              lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasHemm(blas::Side side, blas::UpperLower uplo, uint64_t m,
                             uint64_t n, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda,
                             const DeviceMemory<std::complex<double>> &b,
                             int ldb, std::complex<double> beta,
                             DeviceMemory<std::complex<double>> *c, int ldc) {
   std::vector<std::string> mht_209_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_209(mht_209_v, 3968, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHemm");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(m), PARAM(n), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::Side, blas::UpperLower, uint64_t, uint64_t,
               std::complex<double>, const DeviceMemory<std::complex<double>> &,
               int, const DeviceMemory<std::complex<double>> &, int,
               std::complex<double>, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHemm, side, uplo, m, n, alpha, a,
              lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasHerk(blas::UpperLower uplo, blas::Transpose trans,
                             uint64_t n, uint64 k, float alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda, float beta,
                             DeviceMemory<std::complex<float>> *c, int ldc) {
   std::vector<std::string> mht_210_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_210(mht_210_v, 3989, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHerk");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t, float,
               const DeviceMemory<std::complex<float>> &, int, float,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHerk, uplo, trans, n, k, alpha, a,
              lda, beta, c, ldc);
}

Stream &Stream::ThenBlasHerk(blas::UpperLower uplo, blas::Transpose trans,
                             uint64_t n, uint64 k, double alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda, double beta,
                             DeviceMemory<std::complex<double>> *c, int ldc) {
   std::vector<std::string> mht_211_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_211(mht_211_v, 4008, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHerk");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t, double,
               const DeviceMemory<std::complex<double>> &, int, double,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHerk, uplo, trans, n, k, alpha, a,
              lda, beta, c, ldc);
}

Stream &Stream::ThenBlasHer2k(blas::UpperLower uplo, blas::Transpose trans,
                              uint64_t n, uint64 k, std::complex<float> alpha,
                              const DeviceMemory<std::complex<float>> &a,
                              int lda,
                              const DeviceMemory<std::complex<float>> &b,
                              int ldb, float beta,
                              DeviceMemory<std::complex<float>> *c, int ldc) {
   std::vector<std::string> mht_212_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_212(mht_212_v, 4029, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHer2k");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t,
               std::complex<float>, const DeviceMemory<std::complex<float>> &,
               int, const DeviceMemory<std::complex<float>> &, int, float,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHer2k, uplo, trans, n, k, alpha,
              a, lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasHer2k(blas::UpperLower uplo, blas::Transpose trans,
                              uint64_t n, uint64 k, std::complex<double> alpha,
                              const DeviceMemory<std::complex<double>> &a,
                              int lda,
                              const DeviceMemory<std::complex<double>> &b,
                              int ldb, double beta,
                              DeviceMemory<std::complex<double>> *c, int ldc) {
   std::vector<std::string> mht_213_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_213(mht_213_v, 4052, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasHer2k");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t,
               std::complex<double>, const DeviceMemory<std::complex<double>> &,
               int, const DeviceMemory<std::complex<double>> &, int, double,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHer2k, uplo, trans, n, k, alpha,
              a, lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasSymm(blas::Side side, blas::UpperLower uplo, uint64_t m,
                             uint64_t n, float alpha,
                             const DeviceMemory<float> &a, int lda,
                             const DeviceMemory<float> &b, int ldb, float beta,
                             DeviceMemory<float> *c, int ldc) {
   std::vector<std::string> mht_214_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_214(mht_214_v, 4073, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSymm");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(m), PARAM(n), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::Side, blas::UpperLower, uint64_t, uint64_t, float,
               const DeviceMemory<float> &, int, const DeviceMemory<float> &,
               int, float, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSymm, side, uplo, m, n, alpha, a,
              lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasSymm(blas::Side side, blas::UpperLower uplo, uint64_t m,
                             uint64_t n, double alpha,
                             const DeviceMemory<double> &a, int lda,
                             const DeviceMemory<double> &b, int ldb,
                             double beta, DeviceMemory<double> *c, int ldc) {
   std::vector<std::string> mht_215_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_215(mht_215_v, 4093, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSymm");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(m), PARAM(n), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::Side, blas::UpperLower, uint64_t, uint64_t, double,
               const DeviceMemory<double> &, int, const DeviceMemory<double> &,
               int, double, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSymm, side, uplo, m, n, alpha, a,
              lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasSymm(blas::Side side, blas::UpperLower uplo, uint64_t m,
                             uint64_t n, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda,
                             const DeviceMemory<std::complex<float>> &b,
                             int ldb, std::complex<float> beta,
                             DeviceMemory<std::complex<float>> *c, int ldc) {
   std::vector<std::string> mht_216_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_216(mht_216_v, 4115, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSymm");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(m), PARAM(n), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::Side, blas::UpperLower, uint64_t, uint64_t,
               std::complex<float>, const DeviceMemory<std::complex<float>> &,
               int, const DeviceMemory<std::complex<float>> &, int,
               std::complex<float>, DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSymm, side, uplo, m, n, alpha, a,
              lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasSymm(blas::Side side, blas::UpperLower uplo, uint64_t m,
                             uint64_t n, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda,
                             const DeviceMemory<std::complex<double>> &b,
                             int ldb, std::complex<double> beta,
                             DeviceMemory<std::complex<double>> *c, int ldc) {
   std::vector<std::string> mht_217_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_217(mht_217_v, 4138, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSymm");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(m), PARAM(n), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::Side, blas::UpperLower, uint64_t, uint64_t,
               std::complex<double>, const DeviceMemory<std::complex<double>> &,
               int, const DeviceMemory<std::complex<double>> &, int,
               std::complex<double>, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSymm, side, uplo, m, n, alpha, a,
              lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasSyrk(blas::UpperLower uplo, blas::Transpose trans,
                             uint64_t n, uint64 k, float alpha,
                             const DeviceMemory<float> &a, int lda, float beta,
                             DeviceMemory<float> *c, int ldc) {
   std::vector<std::string> mht_218_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_218(mht_218_v, 4158, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSyrk");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t, float,
               const DeviceMemory<float> &, int, float, DeviceMemory<float> *,
               int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyrk, uplo, trans, n, k, alpha, a,
              lda, beta, c, ldc);
}

Stream &Stream::ThenBlasSyrk(blas::UpperLower uplo, blas::Transpose trans,
                             uint64_t n, uint64 k, double alpha,
                             const DeviceMemory<double> &a, int lda,
                             double beta, DeviceMemory<double> *c, int ldc) {
   std::vector<std::string> mht_219_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_219(mht_219_v, 4176, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSyrk");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t, double,
               const DeviceMemory<double> &, int, double,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyrk, uplo, trans, n, k, alpha, a,
              lda, beta, c, ldc);
}

Stream &Stream::ThenBlasSyrk(blas::UpperLower uplo, blas::Transpose trans,
                             uint64_t n, uint64 k, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda, std::complex<float> beta,
                             DeviceMemory<std::complex<float>> *c, int ldc) {
   std::vector<std::string> mht_220_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_220(mht_220_v, 4195, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSyrk");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t,
               std::complex<float>, const DeviceMemory<std::complex<float>> &,
               int, std::complex<float>, DeviceMemory<std::complex<float>> *,
               int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyrk, uplo, trans, n, k, alpha, a,
              lda, beta, c, ldc);
}

Stream &Stream::ThenBlasSyrk(blas::UpperLower uplo, blas::Transpose trans,
                             uint64_t n, uint64 k, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda, std::complex<double> beta,
                             DeviceMemory<std::complex<double>> *c, int ldc) {
   std::vector<std::string> mht_221_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_221(mht_221_v, 4215, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSyrk");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t,
               std::complex<double>, const DeviceMemory<std::complex<double>> &,
               int, std::complex<double>, DeviceMemory<std::complex<double>> *,
               int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyrk, uplo, trans, n, k, alpha, a,
              lda, beta, c, ldc);
}

Stream &Stream::ThenBlasSyr2k(blas::UpperLower uplo, blas::Transpose trans,
                              uint64_t n, uint64 k, float alpha,
                              const DeviceMemory<float> &a, int lda,
                              const DeviceMemory<float> &b, int ldb, float beta,
                              DeviceMemory<float> *c, int ldc) {
   std::vector<std::string> mht_222_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_222(mht_222_v, 4235, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSyr2k");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t, float,
               const DeviceMemory<float> &, int, const DeviceMemory<float> &,
               int, float, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyr2k, uplo, trans, n, k, alpha,
              a, lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasSyr2k(blas::UpperLower uplo, blas::Transpose trans,
                              uint64_t n, uint64 k, double alpha,
                              const DeviceMemory<double> &a, int lda,
                              const DeviceMemory<double> &b, int ldb,
                              double beta, DeviceMemory<double> *c, int ldc) {
   std::vector<std::string> mht_223_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_223(mht_223_v, 4255, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSyr2k");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t, double,
               const DeviceMemory<double> &, int, const DeviceMemory<double> &,
               int, double, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyr2k, uplo, trans, n, k, alpha,
              a, lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasSyr2k(blas::UpperLower uplo, blas::Transpose trans,
                              uint64_t n, uint64 k, std::complex<float> alpha,
                              const DeviceMemory<std::complex<float>> &a,
                              int lda,
                              const DeviceMemory<std::complex<float>> &b,
                              int ldb, std::complex<float> beta,
                              DeviceMemory<std::complex<float>> *c, int ldc) {
   std::vector<std::string> mht_224_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_224(mht_224_v, 4277, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSyr2k");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t,
               std::complex<float>, const DeviceMemory<std::complex<float>> &,
               int, const DeviceMemory<std::complex<float>> &, int,
               std::complex<float>, DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyr2k, uplo, trans, n, k, alpha,
              a, lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasSyr2k(blas::UpperLower uplo, blas::Transpose trans,
                              uint64_t n, uint64 k, std::complex<double> alpha,
                              const DeviceMemory<std::complex<double>> &a,
                              int lda,
                              const DeviceMemory<std::complex<double>> &b,
                              int ldb, std::complex<double> beta,
                              DeviceMemory<std::complex<double>> *c, int ldc) {
   std::vector<std::string> mht_225_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_225(mht_225_v, 4300, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasSyr2k");

  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t,
               std::complex<double>, const DeviceMemory<std::complex<double>> &,
               int, const DeviceMemory<std::complex<double>> &, int,
               std::complex<double>, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyr2k, uplo, trans, n, k, alpha,
              a, lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasTrmm(blas::Side side, blas::UpperLower uplo,
                             blas::Transpose transa, blas::Diagonal diag,
                             uint64_t m, uint64 n, float alpha,
                             const DeviceMemory<float> &a, int lda,
                             DeviceMemory<float> *b, int ldb) {
   std::vector<std::string> mht_226_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_226(mht_226_v, 4321, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrmm");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, float, const DeviceMemory<float> &, int,
               DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrmm, side, uplo, transa, diag, m,
              n, alpha, a, lda, b, ldb);
}

Stream &Stream::ThenBlasTrmm(blas::Side side, blas::UpperLower uplo,
                             blas::Transpose transa, blas::Diagonal diag,
                             uint64_t m, uint64 n, double alpha,
                             const DeviceMemory<double> &a, int lda,
                             DeviceMemory<double> *b, int ldb) {
   std::vector<std::string> mht_227_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_227(mht_227_v, 4340, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrmm");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, double, const DeviceMemory<double> &, int,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrmm, side, uplo, transa, diag, m,
              n, alpha, a, lda, b, ldb);
}

Stream &Stream::ThenBlasTrmm(blas::Side side, blas::UpperLower uplo,
                             blas::Transpose transa, blas::Diagonal diag,
                             uint64_t m, uint64 n, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda, DeviceMemory<std::complex<float>> *b,
                             int ldb) {
   std::vector<std::string> mht_228_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_228(mht_228_v, 4360, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrmm");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrmm, side, uplo, transa, diag, m,
              n, alpha, a, lda, b, ldb);
}

Stream &Stream::ThenBlasTrmm(blas::Side side, blas::UpperLower uplo,
                             blas::Transpose transa, blas::Diagonal diag,
                             uint64_t m, uint64 n, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda, DeviceMemory<std::complex<double>> *b,
                             int ldb) {
   std::vector<std::string> mht_229_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_229(mht_229_v, 4381, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrmm");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrmm, side, uplo, transa, diag, m,
              n, alpha, a, lda, b, ldb);
}

Stream &Stream::ThenBlasTrsm(blas::Side side, blas::UpperLower uplo,
                             blas::Transpose transa, blas::Diagonal diag,
                             uint64_t m, uint64 n, float alpha,
                             const DeviceMemory<float> &a, int lda,
                             DeviceMemory<float> *b, int ldb) {
   std::vector<std::string> mht_230_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_230(mht_230_v, 4401, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrsm");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, float, const DeviceMemory<float> &, int,
               DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsm, side, uplo, transa, diag, m,
              n, alpha, a, lda, b, ldb);
}

Stream &Stream::ThenBlasTrsm(blas::Side side, blas::UpperLower uplo,
                             blas::Transpose transa, blas::Diagonal diag,
                             uint64_t m, uint64 n, double alpha,
                             const DeviceMemory<double> &a, int lda,
                             DeviceMemory<double> *b, int ldb) {
   std::vector<std::string> mht_231_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_231(mht_231_v, 4420, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrsm");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, double, const DeviceMemory<double> &, int,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsm, side, uplo, transa, diag, m,
              n, alpha, a, lda, b, ldb);
}

Stream &Stream::ThenBlasTrsm(blas::Side side, blas::UpperLower uplo,
                             blas::Transpose transa, blas::Diagonal diag,
                             uint64_t m, uint64 n, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda, DeviceMemory<std::complex<float>> *b,
                             int ldb) {
   std::vector<std::string> mht_232_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_232(mht_232_v, 4440, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrsm");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsm, side, uplo, transa, diag, m,
              n, alpha, a, lda, b, ldb);
}

Stream &Stream::ThenBlasTrsm(blas::Side side, blas::UpperLower uplo,
                             blas::Transpose transa, blas::Diagonal diag,
                             uint64_t m, uint64 n, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda, DeviceMemory<std::complex<double>> *b,
                             int ldb) {
   std::vector<std::string> mht_233_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_233(mht_233_v, 4461, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrsm");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsm, side, uplo, transa, diag, m,
              n, alpha, a, lda, b, ldb);
}

Stream &Stream::ThenBlasTrsmBatched(blas::Side side, blas::UpperLower uplo,
                                    blas::Transpose transa, blas::Diagonal diag,
                                    uint64_t m, uint64 n, float alpha,
                                    const DeviceMemory<float *> &as, int lda,
                                    DeviceMemory<float *> *bs, int ldb,
                                    int batch_count) {
   std::vector<std::string> mht_234_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_234(mht_234_v, 4482, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrsmBatched");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(as), PARAM(lda), PARAM(bs),
            PARAM(ldb), PARAM(batch_count));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, float, const DeviceMemory<float *> &, int,
               DeviceMemory<float *> *, int, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsmBatched, side, uplo, transa,
              diag, m, n, alpha, as, lda, bs, ldb, batch_count);
}

Stream &Stream::ThenBlasTrsmBatched(blas::Side side, blas::UpperLower uplo,
                                    blas::Transpose transa, blas::Diagonal diag,
                                    uint64_t m, uint64 n, double alpha,
                                    const DeviceMemory<double *> &as, int lda,
                                    DeviceMemory<double *> *bs, int ldb,
                                    int batch_count) {
   std::vector<std::string> mht_235_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_235(mht_235_v, 4503, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrsmBatched");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(as), PARAM(lda), PARAM(bs),
            PARAM(ldb), PARAM(batch_count));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, double, const DeviceMemory<double *> &, int,
               DeviceMemory<double *> *, int, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsmBatched, side, uplo, transa,
              diag, m, n, alpha, as, lda, bs, ldb, batch_count);
}

Stream &Stream::ThenBlasTrsmBatched(
    blas::Side side, blas::UpperLower uplo, blas::Transpose transa,
    blas::Diagonal diag, uint64_t m, uint64 n, std::complex<float> alpha,
    const DeviceMemory<std::complex<float> *> &as, int lda,
    DeviceMemory<std::complex<float> *> *bs, int ldb, int batch_count) {
   std::vector<std::string> mht_236_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_236(mht_236_v, 4523, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrsmBatched");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(as), PARAM(lda), PARAM(bs),
            PARAM(ldb), PARAM(batch_count));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float> *> &, int,
               DeviceMemory<std::complex<float> *> *, int, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsmBatched, side, uplo, transa,
              diag, m, n, alpha, as, lda, bs, ldb, batch_count);
}

Stream &Stream::ThenBlasTrsmBatched(
    blas::Side side, blas::UpperLower uplo, blas::Transpose transa,
    blas::Diagonal diag, uint64_t m, uint64 n, std::complex<double> alpha,
    const DeviceMemory<std::complex<double> *> &as, int lda,
    DeviceMemory<std::complex<double> *> *bs, int ldb, int batch_count) {
   std::vector<std::string> mht_237_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_237(mht_237_v, 4544, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasTrsmBatched");

  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(as), PARAM(lda), PARAM(bs),
            PARAM(ldb), PARAM(batch_count));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double> *> &, int,
               DeviceMemory<std::complex<double> *> *, int, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsmBatched, side, uplo, transa,
              diag, m, n, alpha, as, lda, bs, ldb, batch_count);
}

Stream &Stream::ThenBlasGemmBatched(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, float alpha,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &a, int lda,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &b, int ldb, float beta,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &c, int ldc,
    int batch_count) {
   std::vector<std::string> mht_238_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_238(mht_238_v, 4567, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemmBatched");

  return ThenBlasGemmBatchedWithScratch(transa, transb, m, n, k, alpha, a, lda,
                                        b, ldb, beta, c, ldc, batch_count,
                                        /*scratch_allocator=*/nullptr);
}

Stream &Stream::ThenBlasGemmBatchedWithScratch(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, float alpha,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &a, int lda,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &b, int ldb, float beta,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &c, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_239_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_239(mht_239_v, 4582, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemmBatchedWithScratch");

  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc), PARAM(batch_count));

  ThenBlasImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
               float, const port::ArraySlice<DeviceMemory<Eigen::half> *> &,
               int, const port::ArraySlice<DeviceMemory<Eigen::half> *> &, int,
               float, const port::ArraySlice<DeviceMemory<Eigen::half> *> &,
               int, int, ScratchAllocator *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmBatched, transa, transb, m, n,
              k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count,
              scratch_allocator);
}

Stream &Stream::ThenBlasGemmBatched(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, float alpha, const port::ArraySlice<DeviceMemory<float> *> &a,
    int lda, const port::ArraySlice<DeviceMemory<float> *> &b, int ldb,
    float beta, const port::ArraySlice<DeviceMemory<float> *> &c, int ldc,
    int batch_count) {
   std::vector<std::string> mht_240_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_240(mht_240_v, 4606, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemmBatched");

  return ThenBlasGemmBatchedWithScratch(transa, transb, m, n, k, alpha, a, lda,
                                        b, ldb, beta, c, ldc, batch_count,
                                        /*scratch_allocator=*/nullptr);
}

Stream &Stream::ThenBlasGemmBatchedWithScratch(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, float alpha, const port::ArraySlice<DeviceMemory<float> *> &a,
    int lda, const port::ArraySlice<DeviceMemory<float> *> &b, int ldb,
    float beta, const port::ArraySlice<DeviceMemory<float> *> &c, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_241_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_241(mht_241_v, 4620, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemmBatchedWithScratch");

  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc), PARAM(batch_count));

  ThenBlasImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
               float, const port::ArraySlice<DeviceMemory<float> *> &, int,
               const port::ArraySlice<DeviceMemory<float> *> &, int, float,
               const port::ArraySlice<DeviceMemory<float> *> &, int, int,
               ScratchAllocator *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmBatched, transa, transb, m, n,
              k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count,
              scratch_allocator);
}

Stream &Stream::ThenBlasGemmBatched(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, double alpha, const port::ArraySlice<DeviceMemory<double> *> &a,
    int lda, const port::ArraySlice<DeviceMemory<double> *> &b, int ldb,
    double beta, const port::ArraySlice<DeviceMemory<double> *> &c, int ldc,
    int batch_count) {
   std::vector<std::string> mht_242_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_242(mht_242_v, 4644, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemmBatched");

  return ThenBlasGemmBatchedWithScratch(transa, transb, m, n, k, alpha, a, lda,
                                        b, ldb, beta, c, ldc, batch_count,
                                        /*scratch_allocator=*/nullptr);
}

Stream &Stream::ThenBlasGemmBatchedWithScratch(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, double alpha, const port::ArraySlice<DeviceMemory<double> *> &a,
    int lda, const port::ArraySlice<DeviceMemory<double> *> &b, int ldb,
    double beta, const port::ArraySlice<DeviceMemory<double> *> &c, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_243_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_243(mht_243_v, 4658, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemmBatchedWithScratch");

  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc), PARAM(batch_count));

  ThenBlasImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
               double, const port::ArraySlice<DeviceMemory<double> *> &, int,
               const port::ArraySlice<DeviceMemory<double> *> &, int, double,
               const port::ArraySlice<DeviceMemory<double> *> &, int, int,
               ScratchAllocator *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmBatched, transa, transb, m, n,
              k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count,
              scratch_allocator);
}

Stream &Stream::ThenBlasGemmBatched(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, std::complex<float> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &a, int lda,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &b, int ldb,
    std::complex<float> beta,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &c, int ldc,
    int batch_count) {
   std::vector<std::string> mht_244_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_244(mht_244_v, 4684, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemmBatched");

  return ThenBlasGemmBatchedWithScratch(transa, transb, m, n, k, alpha, a, lda,
                                        b, ldb, beta, c, ldc, batch_count,
                                        /*scratch_allocator=*/nullptr);
}

Stream &Stream::ThenBlasGemmBatchedWithScratch(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, std::complex<float> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &a, int lda,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &b, int ldb,
    std::complex<float> beta,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &c, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_245_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_245(mht_245_v, 4700, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemmBatchedWithScratch");

  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc), PARAM(batch_count));

  ThenBlasImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
               std::complex<float>,
               const port::ArraySlice<DeviceMemory<std::complex<float>> *> &,
               int,
               const port::ArraySlice<DeviceMemory<std::complex<float>> *> &,
               int, std::complex<float>,
               const port::ArraySlice<DeviceMemory<std::complex<float>> *> &,
               int, int, ScratchAllocator *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmBatched, transa, transb, m, n,
              k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count,
              scratch_allocator);
}

Stream &Stream::ThenBlasGemmBatched(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, std::complex<double> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &a, int lda,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &b, int ldb,
    std::complex<double> beta,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &c, int ldc,
    int batch_count) {
   std::vector<std::string> mht_246_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_246(mht_246_v, 4729, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemmBatched");

  return ThenBlasGemmBatchedWithScratch(transa, transb, m, n, k, alpha, a, lda,
                                        b, ldb, beta, c, ldc, batch_count,
                                        /*scratch_allocator=*/nullptr);
}

Stream &Stream::ThenBlasGemmBatchedWithScratch(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, std::complex<double> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &a, int lda,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &b, int ldb,
    std::complex<double> beta,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &c, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_247_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_247(mht_247_v, 4745, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenBlasGemmBatchedWithScratch");

  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc), PARAM(batch_count));

  ThenBlasImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
               std::complex<double>,
               const port::ArraySlice<DeviceMemory<std::complex<double>> *> &,
               int,
               const port::ArraySlice<DeviceMemory<std::complex<double>> *> &,
               int, std::complex<double>,
               const port::ArraySlice<DeviceMemory<std::complex<double>> *> &,
               int, int, ScratchAllocator *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmBatched, transa, transb, m, n,
              k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count,
              scratch_allocator);
}

template <typename ABType, typename CType>
Stream &Stream::ThenBlasLtMatmulImpl(
    const blas::IBlasLtMatmulPlan *plan, const HostOrDeviceScalar<CType> &alpha,
    const DeviceMemory<ABType> &a, const DeviceMemory<ABType> &b,
    const HostOrDeviceScalar<CType> &beta, DeviceMemory<CType> *c,
    ScratchAllocator *scratch_allocator,
    const blas::IBlasLtMatmulAlgorithm *algorithm,
    const DeviceMemory<CType> &bias,
    blas::ProfileResult *output_profile_result) {
  VLOG_CALL(PARAM(plan), PARAM(alpha), PARAM(a), PARAM(b), PARAM(beta),
            PARAM(c), PARAM(algorithm), PARAM(bias));

  ThenBlasWithProfileImpl<
      const blas::IBlasLtMatmulPlan *, const HostOrDeviceScalar<CType> &,
      const DeviceMemory<ABType> &, const DeviceMemory<ABType> &,
      const HostOrDeviceScalar<CType> &, DeviceMemory<CType> *,
      ScratchAllocator *, const blas::IBlasLtMatmulAlgorithm *,
      const DeviceMemory<CType> &>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasLtMatmul, plan, alpha, a, b, beta,
              c, scratch_allocator, algorithm, bias, output_profile_result);
}

// Explicit template instantiations for each supported type combination.
template Stream &Stream::ThenBlasLtMatmulImpl<int8, int32>(
    const blas::IBlasLtMatmulPlan *, const HostOrDeviceScalar<int32> &,
    const DeviceMemory<int8> &, const DeviceMemory<int8> &,
    const HostOrDeviceScalar<int32> &, DeviceMemory<int32> *,
    ScratchAllocator *, const blas::IBlasLtMatmulAlgorithm *,
    const DeviceMemory<int32> &, blas::ProfileResult *);

template Stream &Stream::ThenBlasLtMatmulImpl<Eigen::half, Eigen::half>(
    const blas::IBlasLtMatmulPlan *, const HostOrDeviceScalar<Eigen::half> &,
    const DeviceMemory<Eigen::half> &, const DeviceMemory<Eigen::half> &,
    const HostOrDeviceScalar<Eigen::half> &, DeviceMemory<Eigen::half> *,
    ScratchAllocator *, const blas::IBlasLtMatmulAlgorithm *,
    const DeviceMemory<Eigen::half> &, blas::ProfileResult *);

template Stream &Stream::ThenBlasLtMatmulImpl<float, float>(
    const blas::IBlasLtMatmulPlan *, const HostOrDeviceScalar<float> &,
    const DeviceMemory<float> &, const DeviceMemory<float> &,
    const HostOrDeviceScalar<float> &, DeviceMemory<float> *,
    ScratchAllocator *, const blas::IBlasLtMatmulAlgorithm *,
    const DeviceMemory<float> &, blas::ProfileResult *);

template Stream &Stream::ThenBlasLtMatmulImpl<double, double>(
    const blas::IBlasLtMatmulPlan *, const HostOrDeviceScalar<double> &,
    const DeviceMemory<double> &, const DeviceMemory<double> &,
    const HostOrDeviceScalar<double> &, DeviceMemory<double> *,
    ScratchAllocator *, const blas::IBlasLtMatmulAlgorithm *,
    const DeviceMemory<double> &, blas::ProfileResult *);

template Stream &
Stream::ThenBlasLtMatmulImpl<std::complex<float>, std::complex<float>>(
    const blas::IBlasLtMatmulPlan *,
    const HostOrDeviceScalar<std::complex<float>> &,
    const DeviceMemory<std::complex<float>> &,
    const DeviceMemory<std::complex<float>> &,
    const HostOrDeviceScalar<std::complex<float>> &,
    DeviceMemory<std::complex<float>> *, ScratchAllocator *,
    const blas::IBlasLtMatmulAlgorithm *,
    const DeviceMemory<std::complex<float>> &, blas::ProfileResult *);

template Stream &
Stream::ThenBlasLtMatmulImpl<std::complex<double>, std::complex<double>>(
    const blas::IBlasLtMatmulPlan *,
    const HostOrDeviceScalar<std::complex<double>> &,
    const DeviceMemory<std::complex<double>> &,
    const DeviceMemory<std::complex<double>> &,
    const HostOrDeviceScalar<std::complex<double>> &,
    DeviceMemory<std::complex<double>> *, ScratchAllocator *,
    const blas::IBlasLtMatmulAlgorithm *,
    const DeviceMemory<std::complex<double>> &, blas::ProfileResult *);

Stream &Stream::ThenSetRngSeed(const uint8 *seed, uint64_t seed_bytes) {
   std::vector<std::string> mht_248_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_248(mht_248_v, 4841, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenSetRngSeed");

  VLOG_CALL(PARAM(seed), PARAM(seed_bytes));

  if (rng::RngSupport *rng = parent_->AsRng()) {
    CheckError(rng->SetSeed(this, seed, seed_bytes));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers() << " unable to initialize RNG";
  }
  return *this;
}

Stream &Stream::ThenPopulateRandUniform(DeviceMemory<float> *values) {
   std::vector<std::string> mht_249_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_249(mht_249_v, 4856, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenPopulateRandUniform");

  VLOG_CALL(PARAM(values));

  if (rng::RngSupport *rng = parent_->AsRng()) {
    CheckError(rng->DoPopulateRandUniform(this, values));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform RNG operation using StreamExecutor"
                 " without RNG support.";
  }
  return *this;
}

Stream &Stream::ThenPopulateRandGaussian(float mean, float sd,
                                         DeviceMemory<float> *values) {
   std::vector<std::string> mht_250_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_250(mht_250_v, 4874, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenPopulateRandGaussian");

  VLOG_CALL(PARAM(mean), PARAM(sd), PARAM(values));

  if (rng::RngSupport *rng = parent_->AsRng()) {
    CheckError(rng->DoPopulateRandGaussian(this, mean, sd, values));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform RNG operation using StreamExecutor"
                 " without RNG support.";
  }
  return *this;
}

Stream &Stream::ThenPopulateRandGaussian(double mean, double sd,
                                         DeviceMemory<double> *values) {
   std::vector<std::string> mht_251_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_251(mht_251_v, 4892, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenPopulateRandGaussian");

  VLOG_CALL(PARAM(mean), PARAM(sd), PARAM(values));

  if (rng::RngSupport *rng = parent_->AsRng()) {
    CheckError(rng->DoPopulateRandGaussian(this, mean, sd, values));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform RNG operation using StreamExecutor"
                 " without RNG support.";
  }
  return *this;
}

Stream &Stream::ThenPopulateRandUniform(DeviceMemory<double> *values) {
   std::vector<std::string> mht_252_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_252(mht_252_v, 4909, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenPopulateRandUniform");

  VLOG_CALL(PARAM(values));

  if (rng::RngSupport *rng = parent_->AsRng()) {
    CheckError(rng->DoPopulateRandUniform(this, values));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform RNG operation using StreamExecutor"
                 " without RNG support.";
  }
  return *this;
}

Stream &Stream::ThenPopulateRandUniform(
    DeviceMemory<std::complex<float>> *values) {
   std::vector<std::string> mht_253_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_253(mht_253_v, 4927, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenPopulateRandUniform");

  VLOG_CALL(PARAM(values));

  if (rng::RngSupport *rng = parent_->AsRng()) {
    CheckError(rng->DoPopulateRandUniform(this, values));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform RNG operation using StreamExecutor"
                 " without RNG support.";
  }
  return *this;
}

Stream &Stream::ThenPopulateRandUniform(
    DeviceMemory<std::complex<double>> *values) {
   std::vector<std::string> mht_254_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_254(mht_254_v, 4945, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenPopulateRandUniform");

  VLOG_CALL(PARAM(values));

  if (rng::RngSupport *rng = parent_->AsRng()) {
    CheckError(rng->DoPopulateRandUniform(this, values));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform RNG operation using StreamExecutor"
                 " without RNG support.";
  }
  return *this;
}

Stream &Stream::ThenMemcpy(void *host_dst, const DeviceMemoryBase &gpu_src,
                           uint64_t size) {
   std::vector<std::string> mht_255_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_255(mht_255_v, 4963, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenMemcpy");

  VLOG_CALL(PARAM(host_dst), PARAM(gpu_src), PARAM(size));

  CheckError(parent_->Memcpy(this, host_dst, gpu_src, size));
  return *this;
}

Stream &Stream::ThenMemcpy(DeviceMemoryBase *gpu_dst, const void *host_src,
                           uint64_t size) {
   std::vector<std::string> mht_256_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_256(mht_256_v, 4974, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenMemcpy");

  VLOG_CALL(PARAM(gpu_dst), PARAM(host_src), PARAM(size));

  CheckError(parent_->Memcpy(this, gpu_dst, host_src, size));
  return *this;
}

Stream &Stream::ThenMemcpy(DeviceMemoryBase *gpu_dst,
                           const DeviceMemoryBase &gpu_src, uint64_t size) {
   std::vector<std::string> mht_257_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_257(mht_257_v, 4985, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenMemcpy");

  VLOG_CALL(PARAM(gpu_dst), PARAM(gpu_src), PARAM(size));

  CheckError(parent_->MemcpyDeviceToDevice(this, gpu_dst, gpu_src, size));
  return *this;
}

Stream &Stream::ThenMemZero(DeviceMemoryBase *location, uint64_t size) {
   std::vector<std::string> mht_258_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_258(mht_258_v, 4995, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenMemZero");

  VLOG_CALL(PARAM(location), PARAM(size));

  CheckStatus(parent_->MemZero(this, location, size));
  return *this;
}

Stream &Stream::ThenMemset32(DeviceMemoryBase *location, uint32 pattern,
                             uint64_t size) {
   std::vector<std::string> mht_259_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_259(mht_259_v, 5006, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenMemset32");

  VLOG_CALL(PARAM(location), PARAM(pattern), PARAM(size));

  CheckStatus(parent_->Memset32(this, location, pattern, size));
  return *this;
}

Stream &Stream::ThenRnnForward(
    const dnn::RnnDescriptor &rnn_desc,
    const dnn::RnnSequenceTensorDescriptor &input_desc,
    const DeviceMemory<Eigen::half> &input_data,
    const DeviceMemory<int> &seq_lengths_data,
    const dnn::RnnStateTensorDescriptor &input_h_desc,
    const DeviceMemory<Eigen::half> &input_h_data,
    const dnn::RnnStateTensorDescriptor &input_c_desc,
    const DeviceMemory<Eigen::half> &input_c_data,
    const DeviceMemory<Eigen::half> &params,
    const dnn::RnnSequenceTensorDescriptor &output_desc,
    DeviceMemory<Eigen::half> *output_data,
    const dnn::RnnStateTensorDescriptor &output_h_desc,
    DeviceMemory<Eigen::half> *output_h_data,
    const dnn::RnnStateTensorDescriptor &output_c_desc,
    DeviceMemory<Eigen::half> *output_c_data, bool is_training,
    ScratchAllocator *reserve_space_allocator,
    ScratchAllocator *workspace_allocator,
    dnn::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_260_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_260(mht_260_v, 5034, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenRnnForward");

  // TODO(zhengxq): add VLOG PARAM calls.
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    auto status = dnn->DoRnnForward(
        this, rnn_desc, input_desc, input_data, seq_lengths_data, input_h_desc,
        input_h_data, input_c_desc, input_c_data, params, output_desc,
        output_data, output_h_desc, output_h_data, output_c_desc, output_c_data,
        is_training, reserve_space_allocator, workspace_allocator,
        output_profile_result);
    if (!status && !output_profile_result) {
      SetError();
    }
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenRnnForward(
    const dnn::RnnDescriptor &rnn_desc,
    const dnn::RnnSequenceTensorDescriptor &input_desc,
    const DeviceMemory<float> &input_data,
    const DeviceMemory<int> &seq_lengths_data,
    const dnn::RnnStateTensorDescriptor &input_h_desc,
    const DeviceMemory<float> &input_h_data,
    const dnn::RnnStateTensorDescriptor &input_c_desc,
    const DeviceMemory<float> &input_c_data, const DeviceMemory<float> &params,
    const dnn::RnnSequenceTensorDescriptor &output_desc,
    DeviceMemory<float> *output_data,
    const dnn::RnnStateTensorDescriptor &output_h_desc,
    DeviceMemory<float> *output_h_data,
    const dnn::RnnStateTensorDescriptor &output_c_desc,
    DeviceMemory<float> *output_c_data, bool is_training,
    ScratchAllocator *reserve_space_allocator,
    ScratchAllocator *workspace_allocator,
    dnn::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_261_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_261(mht_261_v, 5072, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenRnnForward");

  // TODO(zhengxq): add VLOG PARAM calls.
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    auto status = dnn->DoRnnForward(
        this, rnn_desc, input_desc, input_data, seq_lengths_data, input_h_desc,
        input_h_data, input_c_desc, input_c_data, params, output_desc,
        output_data, output_h_desc, output_h_data, output_c_desc, output_c_data,
        is_training, reserve_space_allocator, workspace_allocator,
        output_profile_result);
    if (!status && !output_profile_result) {
      SetError();
    }
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenRnnForward(
    const dnn::RnnDescriptor &rnn_desc,
    const dnn::RnnSequenceTensorDescriptor &input_desc,
    const DeviceMemory<double> &input_data,
    const DeviceMemory<int> &seq_lengths_data,
    const dnn::RnnStateTensorDescriptor &input_h_desc,
    const DeviceMemory<double> &input_h_data,
    const dnn::RnnStateTensorDescriptor &input_c_desc,
    const DeviceMemory<double> &input_c_data,
    const DeviceMemory<double> &params,
    const dnn::RnnSequenceTensorDescriptor &output_desc,
    DeviceMemory<double> *output_data,
    const dnn::RnnStateTensorDescriptor &output_h_desc,
    DeviceMemory<double> *output_h_data,
    const dnn::RnnStateTensorDescriptor &output_c_desc,
    DeviceMemory<double> *output_c_data, bool is_training,
    ScratchAllocator *reserve_space_allocator,
    ScratchAllocator *workspace_allocator,
    dnn::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_262_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_262(mht_262_v, 5111, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenRnnForward");

  // TODO(zhengxq): add VLOG PARAM calls.
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    auto status = dnn->DoRnnForward(
        this, rnn_desc, input_desc, input_data, seq_lengths_data, input_h_desc,
        input_h_data, input_c_desc, input_c_data, params, output_desc,
        output_data, output_h_desc, output_h_data, output_c_desc, output_c_data,
        is_training, reserve_space_allocator, workspace_allocator,
        output_profile_result);
    if (!status && !output_profile_result) {
      SetError();
    }
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenRnnBackward(
    const dnn::RnnDescriptor &rnn_desc,
    const dnn::RnnSequenceTensorDescriptor &input_desc,
    const DeviceMemory<Eigen::half> &input_data,
    const DeviceMemory<int> &seq_lengths_data,
    const dnn::RnnStateTensorDescriptor &input_h_desc,
    const DeviceMemory<Eigen::half> &input_h_data,
    const dnn::RnnStateTensorDescriptor &input_c_desc,
    const DeviceMemory<Eigen::half> &input_c_data,
    const DeviceMemory<Eigen::half> &params,
    const dnn::RnnSequenceTensorDescriptor &output_desc,
    const DeviceMemory<Eigen::half> &output_data,
    const dnn::RnnStateTensorDescriptor &output_h_desc,
    const DeviceMemory<Eigen::half> &output_h_data,
    const dnn::RnnStateTensorDescriptor &output_c_desc,
    const DeviceMemory<Eigen::half> &output_c_data,
    const DeviceMemory<Eigen::half> &output_backprop_data,
    const DeviceMemory<Eigen::half> &output_h_backprop_data,
    const DeviceMemory<Eigen::half> &output_c_backprop_data,
    DeviceMemory<Eigen::half> *input_backprop_data,
    DeviceMemory<Eigen::half> *input_h_backprop_data,
    DeviceMemory<Eigen::half> *input_c_backprop_data,
    DeviceMemory<Eigen::half> *params_backprop_data,
    DeviceMemory<uint8> *reserve_space_data,
    ScratchAllocator *workspace_allocator,
    dnn::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_263_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_263(mht_263_v, 5157, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenRnnBackward");

  // TODO(zhengxq): add VLOG PARAM calls.
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    auto status = dnn->DoRnnBackward(
        this, rnn_desc, input_desc, input_data, seq_lengths_data, input_h_desc,
        input_h_data, input_c_desc, input_c_data, params, output_desc,
        output_data, output_h_desc, output_h_data, output_c_desc, output_c_data,
        output_backprop_data, output_h_backprop_data, output_c_backprop_data,
        input_backprop_data, input_h_backprop_data, input_c_backprop_data,
        params_backprop_data, reserve_space_data, workspace_allocator,
        output_profile_result);
    if (!status && !output_profile_result) {
      SetError();
    }
  } else {
    SetError();
    LOG(WARNING) << "Attempting to call ThenRnnBackward without DNN support";
  }
  return *this;
}

Stream &Stream::ThenRnnBackward(
    const dnn::RnnDescriptor &rnn_desc,
    const dnn::RnnSequenceTensorDescriptor &input_desc,
    const DeviceMemory<float> &input_data,
    const DeviceMemory<int> &seq_lengths_data,
    const dnn::RnnStateTensorDescriptor &input_h_desc,
    const DeviceMemory<float> &input_h_data,
    const dnn::RnnStateTensorDescriptor &input_c_desc,
    const DeviceMemory<float> &input_c_data, const DeviceMemory<float> &params,
    const dnn::RnnSequenceTensorDescriptor &output_desc,
    const DeviceMemory<float> &output_data,
    const dnn::RnnStateTensorDescriptor &output_h_desc,
    const DeviceMemory<float> &output_h_data,
    const dnn::RnnStateTensorDescriptor &output_c_desc,
    const DeviceMemory<float> &output_c_data,
    const DeviceMemory<float> &output_backprop_data,
    const DeviceMemory<float> &output_h_backprop_data,
    const DeviceMemory<float> &output_c_backprop_data,
    DeviceMemory<float> *input_backprop_data,
    DeviceMemory<float> *input_h_backprop_data,
    DeviceMemory<float> *input_c_backprop_data,
    DeviceMemory<float> *params_backprop_data,
    DeviceMemory<uint8> *reserve_space_data,
    ScratchAllocator *workspace_allocator,
    dnn::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_264_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_264(mht_264_v, 5205, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenRnnBackward");

  // TODO(zhengxq): add VLOG PARAM calls.
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    auto status = dnn->DoRnnBackward(
        this, rnn_desc, input_desc, input_data, seq_lengths_data, input_h_desc,
        input_h_data, input_c_desc, input_c_data, params, output_desc,
        output_data, output_h_desc, output_h_data, output_c_desc, output_c_data,
        output_backprop_data, output_h_backprop_data, output_c_backprop_data,
        input_backprop_data, input_h_backprop_data, input_c_backprop_data,
        params_backprop_data, reserve_space_data, workspace_allocator,
        output_profile_result);
    if (!status && !output_profile_result) {
      SetError();
    }
  } else {
    SetError();
    LOG(WARNING) << "Attempting to call ThenRnnBackward without DNN support";
  }
  return *this;
}

Stream &Stream::ThenRnnBackward(
    const dnn::RnnDescriptor &rnn_desc,
    const dnn::RnnSequenceTensorDescriptor &input_desc,
    const DeviceMemory<double> &input_data,
    const DeviceMemory<int> &seq_lengths_data,
    const dnn::RnnStateTensorDescriptor &input_h_desc,
    const DeviceMemory<double> &input_h_data,
    const dnn::RnnStateTensorDescriptor &input_c_desc,
    const DeviceMemory<double> &input_c_data,
    const DeviceMemory<double> &params,
    const dnn::RnnSequenceTensorDescriptor &output_desc,
    const DeviceMemory<double> &output_data,
    const dnn::RnnStateTensorDescriptor &output_h_desc,
    const DeviceMemory<double> &output_h_data,
    const dnn::RnnStateTensorDescriptor &output_c_desc,
    const DeviceMemory<double> &output_c_data,
    const DeviceMemory<double> &output_backprop_data,
    const DeviceMemory<double> &output_h_backprop_data,
    const DeviceMemory<double> &output_c_backprop_data,
    DeviceMemory<double> *input_backprop_data,
    DeviceMemory<double> *input_h_backprop_data,
    DeviceMemory<double> *input_c_backprop_data,
    DeviceMemory<double> *params_backprop_data,
    DeviceMemory<uint8> *reserve_space_data,
    ScratchAllocator *workspace_allocator,
    dnn::ProfileResult *output_profile_result) {
   std::vector<std::string> mht_265_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_265(mht_265_v, 5254, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenRnnBackward");

  // TODO(zhengxq): add VLOG PARAM calls.
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    auto status = dnn->DoRnnBackward(
        this, rnn_desc, input_desc, input_data, seq_lengths_data, input_h_desc,
        input_h_data, input_c_desc, input_c_data, params, output_desc,
        output_data, output_h_desc, output_h_data, output_c_desc, output_c_data,
        output_backprop_data, output_h_backprop_data, output_c_backprop_data,
        input_backprop_data, input_h_backprop_data, input_c_backprop_data,
        params_backprop_data, reserve_space_data, workspace_allocator,
        output_profile_result);
    if (!status && !output_profile_result) {
      SetError();
    }
  } else {
    SetError();
    LOG(WARNING) << "Attempting to call ThenRnnBackward without DNN support";
  }
  return *this;
}

Stream &Stream::ThenCtcLoss(const dnn::RnnStateTensorDescriptor &probs_desc,
                            const DeviceMemory<float> &probs_data,
                            absl::Span<const int> labels_data,
                            absl::Span<const int> labels_lengths_data,
                            absl::Span<const int> input_lengths_data,
                            DeviceMemory<float> *costs_data,
                            const dnn::RnnStateTensorDescriptor &grads_desc,
                            DeviceMemory<float> *grads_data,
                            ScratchAllocator *workspace_allocator) {
   std::vector<std::string> mht_266_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_266(mht_266_v, 5286, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenCtcLoss");

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    DeviceMemory<uint8> scratch_memory;
    int ctc_loss_algo_id;
    auto status =
        dnn->PrepareForCtcLoss(this, probs_desc, probs_data, grads_desc,
                               labels_data, labels_lengths_data,
                               input_lengths_data, workspace_allocator,
                               &scratch_memory, &ctc_loss_algo_id)
            .ok();
    if (status) {
      status = dnn->DoCtcLoss(this, probs_desc, probs_data, labels_data,
                              labels_lengths_data, input_lengths_data,
                              costs_data, grads_desc, grads_data,
                              &scratch_memory, ctc_loss_algo_id);
    }
    if (!status) {
      SetError();
    }
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenTransformTensor(const dnn::BatchDescriptor &input_desc,
                                    dnn::DataType input_type,
                                    const DeviceMemoryBase &input_data,
                                    const dnn::BatchDescriptor &output_desc,
                                    dnn::DataType output_type, float scale,
                                    DeviceMemoryBase *output_data) {
   std::vector<std::string> mht_267_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_267(mht_267_v, 5319, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenTransformTensor");

  VLOG_CALL(PARAM(input_desc), PARAM(input_type), PARAM(input_data),
            PARAM(output_desc), PARAM(output_type), PARAM(scale),
            PARAM(output_data));
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoTransformTensor(this, input_desc, input_type, input_data,
                                      output_desc, output_type, scale,
                                      output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenDoHostCallback(std::function<void()> callback) {
   std::vector<std::string> mht_268_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_268(mht_268_v, 5336, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenDoHostCallback");

  VLOG_CALL(PARAM(callback));

  if (!ok()) {
    LOG(INFO) << DebugStreamPointers()
              << " was in error state before adding host callback";
  }
  CheckError(parent_->HostCallback(this, std::move(callback)));
  return *this;
}

Stream &Stream::ThenDoHostCallbackWithStatus(
    std::function<port::Status()> callback) {
   std::vector<std::string> mht_269_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_269(mht_269_v, 5351, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenDoHostCallbackWithStatus");

  VLOG_CALL(PARAM(callback));

  if (!ok()) {
    LOG(INFO) << DebugStreamPointers()
              << " was in error state before adding host callback";
  }
  CheckError(parent_->HostCallback(this, std::move(callback)));
  return *this;
}

Stream &Stream::ThenRunAfterNextBlockHostUntilDone(
    std::function<void()> callback) {
   std::vector<std::string> mht_270_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_270(mht_270_v, 5366, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenRunAfterNextBlockHostUntilDone");

  VLOG_CALL(PARAM(callback));

  if (!ok()) {
    LOG(INFO) << DebugStreamPointers()
              << " was in error state before adding callback to be run after "
                 "next block-host-until-done.";
  }
  absl::MutexLock lock(&mu_);
  after_block_host_until_done_callbacks_.push_back(std::move(callback));
  return *this;
}

Stream &Stream::ThenFft(fft::Plan *plan,
                        const DeviceMemory<std::complex<float>> &input,
                        DeviceMemory<std::complex<float>> *output) {
   std::vector<std::string> mht_271_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_271(mht_271_v, 5384, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenFft");

  VLOG_CALL(PARAM(plan), PARAM(input), PARAM(output));

  if (fft::FftSupport *fft = parent_->AsFft()) {
    CheckError(fft->DoFft(this, plan, input, output));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform FFT operation using StreamExecutor"
                 " without FFT support";
  }
  return *this;
}

Stream &Stream::ThenFft(fft::Plan *plan,
                        const DeviceMemory<std::complex<double>> &input,
                        DeviceMemory<std::complex<double>> *output) {
   std::vector<std::string> mht_272_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_272(mht_272_v, 5403, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenFft");

  VLOG_CALL(PARAM(plan), PARAM(input), PARAM(output));

  if (fft::FftSupport *fft = parent_->AsFft()) {
    CheckError(fft->DoFft(this, plan, input, output));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform FFT operation using StreamExecutor"
                 " without FFT support";
  }
  return *this;
}

Stream &Stream::ThenFft(fft::Plan *plan, const DeviceMemory<float> &input,
                        DeviceMemory<std::complex<float>> *output) {
   std::vector<std::string> mht_273_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_273(mht_273_v, 5421, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenFft");

  VLOG_CALL(PARAM(plan), PARAM(input), PARAM(output));

  if (fft::FftSupport *fft = parent_->AsFft()) {
    CheckError(fft->DoFft(this, plan, input, output));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform FFT operation using StreamExecutor"
                 " without FFT support";
  }
  return *this;
}

Stream &Stream::ThenFft(fft::Plan *plan, const DeviceMemory<double> &input,
                        DeviceMemory<std::complex<double>> *output) {
   std::vector<std::string> mht_274_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_274(mht_274_v, 5439, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenFft");

  VLOG_CALL(PARAM(plan), PARAM(input), PARAM(output));

  if (fft::FftSupport *fft = parent_->AsFft()) {
    CheckError(fft->DoFft(this, plan, input, output));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform FFT operation using StreamExecutor"
                 " without FFT support";
  }
  return *this;
}

Stream &Stream::ThenFft(fft::Plan *plan,
                        const DeviceMemory<std::complex<float>> &input,
                        DeviceMemory<float> *output) {
   std::vector<std::string> mht_275_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_275(mht_275_v, 5458, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenFft");

  VLOG_CALL(PARAM(plan), PARAM(input), PARAM(output));

  if (fft::FftSupport *fft = parent_->AsFft()) {
    CheckError(fft->DoFft(this, plan, input, output));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform FFT operation using StreamExecutor"
                 " without FFT support";
  }
  return *this;
}

Stream &Stream::ThenFft(fft::Plan *plan,
                        const DeviceMemory<std::complex<double>> &input,
                        DeviceMemory<double> *output) {
   std::vector<std::string> mht_276_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_276(mht_276_v, 5477, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenFft");

  VLOG_CALL(PARAM(plan), PARAM(input), PARAM(output));

  if (fft::FftSupport *fft = parent_->AsFft()) {
    CheckError(fft->DoFft(this, plan, input, output));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform FFT operation using StreamExecutor"
                 " without FFT support";
  }
  return *this;
}

// It looks confusing, but all this is doing is inserting a callback at the
// present point in the stream to then enqueue a task on the host executor.
Stream &Stream::ThenEnqueueOnBackgroundThread(
    std::function<void(StreamExecutor *)> task) {
   std::vector<std::string> mht_277_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_277(mht_277_v, 5497, "", "./tensorflow/stream_executor/stream.cc", "Stream::ThenEnqueueOnBackgroundThread");

  VLOG_CALL(PARAM(task));

  StreamExecutor *stream_executor = this->parent_;
  std::function<void()> bound_task = std::bind(task, stream_executor);

  return ThenDoHostCallback([stream_executor, bound_task]() {
    stream_executor->EnqueueOnBackgroundThread(bound_task);
  });
}

port::Status Stream::BlockHostUntilDone() {
   std::vector<std::string> mht_278_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_278(mht_278_v, 5511, "", "./tensorflow/stream_executor/stream.cc", "Stream::BlockHostUntilDone");

  VLOG_CALL();

  if (!ok()) {
    port::Status status = port::Status(
        port::error::INTERNAL,
        "stream did not block host until done; was already in an error state");
    LOG(INFO) << DebugStreamPointers() << " " << status;
    return status;
  }

  temporary_memory_manager_.DeallocateFinalizedTemporaries();

  port::Status error = parent_->BlockHostUntilDone(this);
  CheckError(error.ok());

  RunAfterBlockHostUntilDoneCallbacks();
  return error;
}

void Stream::RunAfterBlockHostUntilDoneCallbacks() {
   std::vector<std::string> mht_279_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_279(mht_279_v, 5534, "", "./tensorflow/stream_executor/stream.cc", "Stream::RunAfterBlockHostUntilDoneCallbacks");

  std::vector<std::function<void()>> callbacks;
  {
    absl::MutexLock lock(&mu_);
    std::swap(callbacks, after_block_host_until_done_callbacks_);
  }
  for (const auto &fn : callbacks) {
    fn();
  }
}

std::string Stream::DebugStreamPointers() const {
   std::vector<std::string> mht_280_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_280(mht_280_v, 5548, "", "./tensorflow/stream_executor/stream.cc", "Stream::DebugStreamPointers");

  // Relies on the ToVlogString(const void*) overload above.
  return absl::StrCat("[stream=", ToVlogString(this),
                      ",impl=", ToVlogString(implementation_.get()), "]");
}

void Stream::CheckStatus(port::Status status) {
   std::vector<std::string> mht_281_v;
   MHTracer_DTPStensorflowPSstream_executorPSstreamDTcc mht_281(mht_281_v, 5557, "", "./tensorflow/stream_executor/stream.cc", "Stream::CheckStatus");

  if (status.ok()) {
    return;
  }
  LOG(ERROR) << status;
  absl::MutexLock lock(&mu_);
  status_ = status;
}

}  // namespace stream_executor
