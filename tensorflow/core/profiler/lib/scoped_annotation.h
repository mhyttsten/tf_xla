/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_LIB_SCOPED_ANNOTATION_H_
#define TENSORFLOW_CORE_PROFILER_LIB_SCOPED_ANNOTATION_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_annotationDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_annotationDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_annotationDTh() {
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


#include <stddef.h>

#include <atomic>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/profiler/internal/cpu/annotation_stack.h"
#endif

namespace tensorflow {
namespace profiler {

// Adds an annotation to all activities for the duration of the instance
// lifetime through the currently registered TraceCollector.
//
// Usage: {
//          ScopedAnnotation annotation("my kernels");
//          Kernel1<<<x,y>>>;
//          LaunchKernel2(); // Launches a CUDA kernel.
//        }
// This will add 'my kernels' to both kernels in the profiler UI
class ScopedAnnotation {
 public:
  explicit ScopedAnnotation(absl::string_view name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_annotationDTh mht_0(mht_0_v, 215, "", "./tensorflow/core/profiler/lib/scoped_annotation.h", "ScopedAnnotation");

#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(AnnotationStack::IsEnabled())) {
      old_length_ = AnnotationStack::PushAnnotation(name);
    }
#endif
  }

  explicit ScopedAnnotation(const char* name)
      : ScopedAnnotation(absl::string_view(name)) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_annotationDTh mht_1(mht_1_v, 228, "", "./tensorflow/core/profiler/lib/scoped_annotation.h", "ScopedAnnotation");
}

  explicit ScopedAnnotation(const string& name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_annotationDTh mht_2(mht_2_v, 234, "", "./tensorflow/core/profiler/lib/scoped_annotation.h", "ScopedAnnotation");

#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(AnnotationStack::IsEnabled())) {
      old_length_ = AnnotationStack::PushAnnotation(name);
    }
#endif
  }

  explicit ScopedAnnotation(string&& name) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_annotationDTh mht_3(mht_3_v, 245, "", "./tensorflow/core/profiler/lib/scoped_annotation.h", "ScopedAnnotation");

#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(AnnotationStack::IsEnabled())) {
      old_length_ = AnnotationStack::PushAnnotation(std::move(name));
    }
#endif
  }

  template <typename NameGeneratorT>
  explicit ScopedAnnotation(NameGeneratorT name_generator) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_annotationDTh mht_4(mht_4_v, 257, "", "./tensorflow/core/profiler/lib/scoped_annotation.h", "ScopedAnnotation");

#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(AnnotationStack::IsEnabled())) {
      old_length_ = AnnotationStack::PushAnnotation(name_generator());
    }
#endif
  }

  // Pops the name passed in the constructor from the current annotation.
  ~ScopedAnnotation() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_annotationDTh mht_5(mht_5_v, 269, "", "./tensorflow/core/profiler/lib/scoped_annotation.h", "~ScopedAnnotation");

    // TODO(b/137971921): without this memory fence, two presubmit tests will
    // fail probably due to compiler in that presubmit config.
    std::atomic_thread_fence(std::memory_order_acquire);
#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(old_length_ != kInvalidLength)) {
      AnnotationStack::PopAnnotation(old_length_);
    }
#endif
  }

  static bool IsEnabled() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSscoped_annotationDTh mht_6(mht_6_v, 283, "", "./tensorflow/core/profiler/lib/scoped_annotation.h", "IsEnabled");

#if !defined(IS_MOBILE_PLATFORM)
    return AnnotationStack::IsEnabled();
#else
    return false;
#endif
  }

 private:
  // signals that annotation is disabled at the constructor.
  static constexpr size_t kInvalidLength = static_cast<size_t>(-1);

  TF_DISALLOW_COPY_AND_ASSIGN(ScopedAnnotation);

  size_t old_length_ = kInvalidLength;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_LIB_SCOPED_ANNOTATION_H_
