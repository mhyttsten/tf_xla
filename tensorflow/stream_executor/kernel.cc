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
class MHTracer_DTPStensorflowPSstream_executorPSkernelDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSkernelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSkernelDTcc() {
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

// Implementation of the pointer-to-implementation wrapper for the data-parallel
// kernel abstraction. KernelBase just delegates to the internal
// platform-specific implementation instance.

#include "tensorflow/stream_executor/kernel.h"

#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/stream_executor/lib/demangle.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace stream_executor {

bool KernelMetadata::registers_per_thread(int *registers_per_thread) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernelDTcc mht_0(mht_0_v, 201, "", "./tensorflow/stream_executor/kernel.cc", "KernelMetadata::registers_per_thread");

  if (has_registers_per_thread_) {
    *registers_per_thread = registers_per_thread_;
    return true;
  }

  return false;
}

void KernelMetadata::set_registers_per_thread(int registers_per_thread) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernelDTcc mht_1(mht_1_v, 213, "", "./tensorflow/stream_executor/kernel.cc", "KernelMetadata::set_registers_per_thread");

  registers_per_thread_ = registers_per_thread;
  has_registers_per_thread_ = true;
}

bool KernelMetadata::shared_memory_bytes(int *shared_memory_bytes) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernelDTcc mht_2(mht_2_v, 221, "", "./tensorflow/stream_executor/kernel.cc", "KernelMetadata::shared_memory_bytes");

  if (has_shared_memory_bytes_) {
    *shared_memory_bytes = shared_memory_bytes_;
    return true;
  }

  return false;
}

void KernelMetadata::set_shared_memory_bytes(int shared_memory_bytes) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernelDTcc mht_3(mht_3_v, 233, "", "./tensorflow/stream_executor/kernel.cc", "KernelMetadata::set_shared_memory_bytes");

  shared_memory_bytes_ = shared_memory_bytes;
  has_shared_memory_bytes_ = true;
}

KernelBase::KernelBase(KernelBase &&from)
    : parent_(from.parent_),
      implementation_(std::move(from.implementation_)),
      name_(std::move(from.name_)),
      demangled_name_(std::move(from.demangled_name_)),
      metadata_(from.metadata_) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernelDTcc mht_4(mht_4_v, 246, "", "./tensorflow/stream_executor/kernel.cc", "KernelBase::KernelBase");

  from.parent_ = nullptr;
}

KernelBase::KernelBase(StreamExecutor *parent)
    : parent_(parent),
      implementation_(parent->implementation()->CreateKernelImplementation()) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernelDTcc mht_5(mht_5_v, 255, "", "./tensorflow/stream_executor/kernel.cc", "KernelBase::KernelBase");
}

KernelBase::KernelBase(StreamExecutor *parent,
                       internal::KernelInterface *implementation)
    : parent_(parent), implementation_(implementation) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernelDTcc mht_6(mht_6_v, 262, "", "./tensorflow/stream_executor/kernel.cc", "KernelBase::KernelBase");
}

KernelBase::~KernelBase() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernelDTcc mht_7(mht_7_v, 267, "", "./tensorflow/stream_executor/kernel.cc", "KernelBase::~KernelBase");

  if (parent_) {
    parent_->UnloadKernel(this);
  }
}

unsigned KernelBase::Arity() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernelDTcc mht_8(mht_8_v, 276, "", "./tensorflow/stream_executor/kernel.cc", "KernelBase::Arity");
 return implementation_->Arity(); }

void KernelBase::SetPreferredCacheConfig(KernelCacheConfig config) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernelDTcc mht_9(mht_9_v, 281, "", "./tensorflow/stream_executor/kernel.cc", "KernelBase::SetPreferredCacheConfig");

  return implementation_->SetPreferredCacheConfig(config);
}

KernelCacheConfig KernelBase::GetPreferredCacheConfig() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPSkernelDTcc mht_10(mht_10_v, 288, "", "./tensorflow/stream_executor/kernel.cc", "KernelBase::GetPreferredCacheConfig");

  return implementation_->GetPreferredCacheConfig();
}

void KernelBase::set_name(absl::string_view name) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSkernelDTcc mht_11(mht_11_v, 296, "", "./tensorflow/stream_executor/kernel.cc", "KernelBase::set_name");

  name_ = std::string(name);

  // CUDA splitter prefixes stub functions with __device_stub_.
  demangled_name_ =
      port::Demangle(absl::StripPrefix(name, "__device_stub_").data());
}

}  // namespace stream_executor
