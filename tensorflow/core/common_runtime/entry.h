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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_ENTRY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_ENTRY_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSentryDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSentryDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSentryDTh() {
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


#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"

namespace tensorflow {

class mutex;
class Tensor;

// An Entry store a single input value for an individual kernel invocation in
// an executor.
//
// Either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
struct Entry {
  enum class State {
    NO_VALUE = 0,      // The default state for a newly-created Entry.
    HAS_VALUE,         // `this->val` is valid.
    HAS_CONST_TENSOR,  // `this->const_tensor` is valid.
    HAS_REF_TENSOR,    // `this->ref_tensor` is valid.
  };

  Entry() : state(State::NO_VALUE) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSentryDTh mht_0(mht_0_v, 209, "", "./tensorflow/core/common_runtime/entry.h", "Entry");
}
  Entry(const Entry& other) : state(other.state), alloc_attr(other.alloc_attr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSentryDTh mht_1(mht_1_v, 213, "", "./tensorflow/core/common_runtime/entry.h", "Entry");

    switch (state) {
      case State::NO_VALUE:
        break;
      case State::HAS_VALUE:
        val.Init(*other.val);
        break;
      case State::HAS_CONST_TENSOR:
        const_tensor = other.const_tensor;
        break;
      case State::HAS_REF_TENSOR:
        ref_tensor = other.ref_tensor;
        break;
    }
  }

  ~Entry() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSentryDTh mht_2(mht_2_v, 232, "", "./tensorflow/core/common_runtime/entry.h", "~Entry");

    if (state == State::HAS_VALUE) val.Destroy();
  }

  Entry& operator=(const Entry& other) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSentryDTh mht_3(mht_3_v, 239, "", "./tensorflow/core/common_runtime/entry.h", "=");

    if (state == State::HAS_VALUE) {
      val.Destroy();
    }
    state = other.state;
    alloc_attr = other.alloc_attr;
    switch (state) {
      case State::NO_VALUE:
        break;
      case State::HAS_VALUE:
        val.Init(*other.val);
        break;
      case State::HAS_CONST_TENSOR:
        const_tensor = other.const_tensor;
        break;
      case State::HAS_REF_TENSOR:
        ref_tensor = other.ref_tensor;
        break;
    }
    return *this;
  }

  Entry& operator=(Entry&& other) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSentryDTh mht_4(mht_4_v, 264, "", "./tensorflow/core/common_runtime/entry.h", "=");

    if (state == State::HAS_VALUE) {
      val.Destroy();
    }
    state = other.state;
    alloc_attr = other.alloc_attr;
    switch (state) {
      case State::NO_VALUE:
        break;
      case State::HAS_VALUE:
        val.Init(std::move(*other.val));
        break;
      case State::HAS_CONST_TENSOR:
        const_tensor = other.const_tensor;
        break;
      case State::HAS_REF_TENSOR:
        ref_tensor = other.ref_tensor;
        break;
    }
    return *this;
  }

  // Clears the <val> field, and sets this entry to the `NO_VALUE` state.
  void ClearVal() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSentryDTh mht_5(mht_5_v, 290, "", "./tensorflow/core/common_runtime/entry.h", "ClearVal");

    if (state == State::HAS_VALUE) {
      val.Destroy();
    }
    state = State::NO_VALUE;
  }

  union {
    // A tensor value. Valid iff `state_ == HAS_VALUE`.
    ManualConstructor<Tensor> val;

    // A pointer to a constant tensor value. Valid iff `state_ ==
    // HAS_CONST_TENSOR`.
    const Tensor* const_tensor;

    // A tensor reference and associated mutex. Valid iff `state_ ==
    // HAS_REF_TENSOR`.
    struct {
      Tensor* tensor;
      mutex* mu;
    } ref_tensor;
  };

  // The current state of this entry, indicating which member of the above
  // union is active.
  State state;

  // The attributes of the allocator that creates the tensor.
  AllocatorAttributes alloc_attr;
};

// TODO(b/152925936): Re-evaluate this constant with current usage patterns.
typedef gtl::InlinedVector<Entry, 4> EntryVector;

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_ENTRY_H_
