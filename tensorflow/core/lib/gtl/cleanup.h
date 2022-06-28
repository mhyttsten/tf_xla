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

// MakeCleanup(f) returns an RAII cleanup object that calls 'f' in its
// destructor. The easiest way to use MakeCleanup is with a lambda argument,
// capturing the return value in an 'auto' local variable. Most users will not
// need more sophisticated syntax than that.
//
// Example:
//   void func() {
//     FILE* fp = fopen("data.txt", "r");
//     if (fp == nullptr) return;
//     auto fp_cleaner = gtl::MakeCleanup([fp] { fclose(fp); });
//     // No matter what, fclose(fp) will happen.
//     DataObject d;
//     while (ReadDataObject(fp, &d)) {
//       if (d.IsBad()) {
//         LOG(ERROR) << "Bad Data";
//         return;
//       }
//       PushGoodData(d);
//     }
//   }
//
// You can use Cleanup<F> directly, instead of using MakeCleanup and auto,
// but there's rarely a reason to do that.
//
// You can call 'release()' on a Cleanup object to cancel the cleanup.

#ifndef TENSORFLOW_CORE_LIB_GTL_CLEANUP_H_
#define TENSORFLOW_CORE_LIB_GTL_CLEANUP_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanupDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanupDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanupDTh() {
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


#include <type_traits>
#include <utility>

#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace gtl {

// A move-only RAII object that calls a stored cleanup functor when
// destroyed. Cleanup<F> is the return type of gtl::MakeCleanup(F).
template <typename F>
class Cleanup {
 public:
  Cleanup() : released_(true), f_() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanupDTh mht_0(mht_0_v, 227, "", "./tensorflow/core/lib/gtl/cleanup.h", "Cleanup");
}

  template <typename G>
  explicit Cleanup(G&& f)          // NOLINT
      : f_(std::forward<G>(f)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanupDTh mht_1(mht_1_v, 234, "", "./tensorflow/core/lib/gtl/cleanup.h", "Cleanup");
}  // NOLINT(build/c++11)

  Cleanup(Cleanup&& src)  // NOLINT
      : released_(src.is_released()), f_(src.release()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanupDTh mht_2(mht_2_v, 240, "", "./tensorflow/core/lib/gtl/cleanup.h", "Cleanup");
}

  // Implicitly move-constructible from any compatible Cleanup<G>.
  // The source will be released as if src.release() were called.
  // A moved-from Cleanup can be safely destroyed or reassigned.
  template <typename G>
  Cleanup(Cleanup<G>&& src)  // NOLINT
      : released_(src.is_released()), f_(src.release()) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanupDTh mht_3(mht_3_v, 250, "", "./tensorflow/core/lib/gtl/cleanup.h", "Cleanup");
}

  // Assignment to a Cleanup object behaves like destroying it
  // and making a new one in its place, analogous to unique_ptr
  // semantics.
  Cleanup& operator=(Cleanup&& src) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanupDTh mht_4(mht_4_v, 258, "", "./tensorflow/core/lib/gtl/cleanup.h", "=");
  // NOLINT
    if (!released_) f_();
    released_ = src.released_;
    f_ = src.release();
    return *this;
  }

  ~Cleanup() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanupDTh mht_5(mht_5_v, 268, "", "./tensorflow/core/lib/gtl/cleanup.h", "~Cleanup");

    if (!released_) f_();
  }

  // Releases the cleanup function instead of running it.
  // Hint: use c.release()() to run early.
  F release() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanupDTh mht_6(mht_6_v, 277, "", "./tensorflow/core/lib/gtl/cleanup.h", "release");

    released_ = true;
    return std::move(f_);
  }

  bool is_released() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPScleanupDTh mht_7(mht_7_v, 285, "", "./tensorflow/core/lib/gtl/cleanup.h", "is_released");
 return released_; }

 private:
  static_assert(!std::is_reference<F>::value, "F must not be a reference");

  bool released_ = false;
  F f_;
};

template <int&... ExplicitParameterBarrier, typename F,
          typename DecayF = typename std::decay<F>::type>
TF_MUST_USE_RESULT Cleanup<DecayF> MakeCleanup(F&& f) {
  return Cleanup<DecayF>(std::forward<F>(f));
}

}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_GTL_CLEANUP_H_
