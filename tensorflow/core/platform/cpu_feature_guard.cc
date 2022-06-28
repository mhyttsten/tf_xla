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
class MHTracer_DTPStensorflowPScorePSplatformPScpu_feature_guardDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPScpu_feature_guardDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScpu_feature_guardDTcc() {
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

#include "tensorflow/core/platform/cpu_feature_guard.h"

#ifndef __ANDROID__
#include <iostream>
#endif
#include <mutex>
#include <string>

#include "absl/base/call_once.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace port {
namespace {

// If the CPU feature isn't present, log a fatal error.
void CheckFeatureOrDie(CPUFeature feature, const string& feature_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("feature_name: \"" + feature_name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScpu_feature_guardDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/platform/cpu_feature_guard.cc", "CheckFeatureOrDie");

  if (!TestCPUFeature(feature)) {
    const auto error_msg =
        "The TensorFlow library was compiled to use " + feature_name +
        " instructions, but these aren't available on your machine.";
#ifdef __ANDROID__
    // Some Android emulators seem to indicate they don't support SSE, so we
    // only issue a warning to avoid crashes when testing. We use the logging
    // framework here because std::cout and std::cerr made some Android targets
    // crash.
    LOG(WARNING) << error_msg;
#else
    // Avoiding use of the logging framework here as that might trigger a SIGILL
    // by itself.
    std::cerr << error_msg << std::endl;
    std::abort();
#endif
  }
}

// Check if CPU feature is included in the TensorFlow binary.
void CheckIfFeatureUnused(CPUFeature feature, const string& feature_name,
                          string& missing_instructions) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("feature_name: \"" + feature_name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScpu_feature_guardDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/platform/cpu_feature_guard.cc", "CheckIfFeatureUnused");

  if (TestCPUFeature(feature)) {
    missing_instructions.append(" ");
    missing_instructions.append(feature_name);
  }
}

// Raises an error if the binary has been compiled for a CPU feature (like AVX)
// that isn't available on the current machine. It also warns of performance
// loss if there's a feature available that's not being used.
// Depending on the compiler and initialization order, a SIGILL exception may
// occur before this code is reached, but this at least offers a chance to give
// a more meaningful error message.
class CPUFeatureGuard {
 public:
  CPUFeatureGuard() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPScpu_feature_guardDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/platform/cpu_feature_guard.cc", "CPUFeatureGuard");

#ifdef __SSE__
    CheckFeatureOrDie(CPUFeature::SSE, "SSE");
#endif  // __SSE__
#ifdef __SSE2__
    CheckFeatureOrDie(CPUFeature::SSE2, "SSE2");
#endif  // __SSE2__
#ifdef __SSE3__
    CheckFeatureOrDie(CPUFeature::SSE3, "SSE3");
#endif  // __SSE3__
#ifdef __SSE4_1__
    CheckFeatureOrDie(CPUFeature::SSE4_1, "SSE4.1");
#endif  // __SSE4_1__
#ifdef __SSE4_2__
    CheckFeatureOrDie(CPUFeature::SSE4_2, "SSE4.2");
#endif  // __SSE4_2__
#ifdef __AVX__
    CheckFeatureOrDie(CPUFeature::AVX, "AVX");
#endif  // __AVX__
#ifdef __AVX2__
    CheckFeatureOrDie(CPUFeature::AVX2, "AVX2");
#endif  // __AVX2__
#ifdef __AVX512F__
    CheckFeatureOrDie(CPUFeature::AVX512F, "AVX512F");
#endif  // __AVX512F__
#ifdef __AVX512VNNI__
    CheckFeatureOrDie(CPUFeature::AVX512_VNNI, "AVX512_VNNI");
#endif  // __AVX512VNNI__
#ifdef __AVX512BF16__
    CheckFeatureOrDie(CPUFeature::AVX512_BF16, "AVX512_BF16");
#endif  // __AVX512BF16__
#ifdef __AVXVNNI__
    CheckFeatureOrDie(CPUFeature::AVX_VNNI, "AVX_VNNI");
#endif  // __AVXVNNI__
#ifdef __AMXTILE__
    CheckFeatureOrDie(CPUFeature::AMX_TILE, "AMX_TILE");
#endif  // __AMXTILE__
#ifdef __AMXINT8__
    CheckFeatureOrDie(CPUFeature::AMX_INT8, "AMX_INT8");
#endif  // __AMXINT8__
#ifdef __AMXBF16__
    CheckFeatureOrDie(CPUFeature::AMX_BF16, "AMX_BF16");
#endif  // __AMXBF16__
#ifdef __FMA__
    CheckFeatureOrDie(CPUFeature::FMA, "FMA");
#endif  // __FMA__
  }
};

CPUFeatureGuard g_cpu_feature_guard_singleton;

absl::once_flag g_cpu_feature_guard_warn_once_flag;

}  // namespace

void InfoAboutUnusedCPUFeatures() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPScpu_feature_guardDTcc mht_3(mht_3_v, 306, "", "./tensorflow/core/platform/cpu_feature_guard.cc", "InfoAboutUnusedCPUFeatures");

  absl::call_once(g_cpu_feature_guard_warn_once_flag, [] {
    string missing_instructions;
#if defined(_MSC_VER) && !defined(__clang__)

#ifndef __AVX__
    CheckIfFeatureUnused(CPUFeature::AVX, "AVX", missing_instructions);
#endif  // __AVX__
#ifndef __AVX2__
    CheckIfFeatureUnused(CPUFeature::AVX2, "AVX2", missing_instructions);
#endif  // __AVX2__

#else  // if defined(_MSC_VER) && !defined(__clang__)

#ifndef __SSE__
    CheckIfFeatureUnused(CPUFeature::SSE, "SSE", missing_instructions);
#endif  // __SSE__
#ifndef __SSE2__
    CheckIfFeatureUnused(CPUFeature::SSE2, "SSE2", missing_instructions);
#endif  // __SSE2__
#ifndef __SSE3__
    CheckIfFeatureUnused(CPUFeature::SSE3, "SSE3", missing_instructions);
#endif  // __SSE3__
#ifndef __SSE4_1__
    CheckIfFeatureUnused(CPUFeature::SSE4_1, "SSE4.1", missing_instructions);
#endif  // __SSE4_1__
#ifndef __SSE4_2__
    CheckIfFeatureUnused(CPUFeature::SSE4_2, "SSE4.2", missing_instructions);
#endif  // __SSE4_2__
#ifndef __AVX__
    CheckIfFeatureUnused(CPUFeature::AVX, "AVX", missing_instructions);
#endif  // __AVX__
#ifndef __AVX2__
    CheckIfFeatureUnused(CPUFeature::AVX2, "AVX2", missing_instructions);
#endif  // __AVX2__
#ifndef __AVX512F__
    CheckIfFeatureUnused(CPUFeature::AVX512F, "AVX512F", missing_instructions);
#endif  // __AVX512F__
#ifndef __AVX512VNNI__
    CheckIfFeatureUnused(CPUFeature::AVX512_VNNI, "AVX512_VNNI",
                         missing_instructions);
#endif  // __AVX512VNNI__
#ifndef __AVX512BF16__
    CheckIfFeatureUnused(CPUFeature::AVX512_BF16, "AVX512_BF16",
                         missing_instructions);
#endif  // __AVX512BF16___
#ifndef __AVXVNNI__
    CheckIfFeatureUnused(CPUFeature::AVX_VNNI, "AVX_VNNI",
                         missing_instructions);
#endif  // __AVXVNNI__
#ifndef __AMXTILE__
    CheckIfFeatureUnused(CPUFeature::AMX_TILE, "AMX_TILE",
                         missing_instructions);
#endif  // __AMXTILE__
#ifndef __AMXINT8__
    CheckIfFeatureUnused(CPUFeature::AMX_INT8, "AMX_INT8",
                         missing_instructions);
#endif  // __AMXINT8__
#ifndef __AMXBF16__
    CheckIfFeatureUnused(CPUFeature::AMX_BF16, "AMX_BF16",
                         missing_instructions);
#endif  // __AMXBF16__
#ifndef __FMA__
    CheckIfFeatureUnused(CPUFeature::FMA, "FMA", missing_instructions);
#endif  // __FMA__
#endif  // else of if defined(_MSC_VER) && !defined(__clang__)
    if (!missing_instructions.empty()) {
      LOG(INFO) << "This TensorFlow binary is optimized with "
                << "oneAPI Deep Neural Network Library (oneDNN) "
                << "to use the following CPU instructions in performance-"
                << "critical operations: " << missing_instructions << std::endl
                << "To enable them in other operations, rebuild TensorFlow "
                << "with the appropriate compiler flags.";
    }
  });
}

}  // namespace port
}  // namespace tensorflow
