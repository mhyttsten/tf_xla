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

#ifndef TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_RUNTIME_BUILDER_H_
#define TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_RUNTIME_BUILDER_H_
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
class MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPSruntime_builderDTh {
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
   MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPSruntime_builderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPSruntime_builderDTh() {
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


#include <memory>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/cc/experimental/base/public/runtime.h"
#include "tensorflow/cc/experimental/base/public/status.h"

namespace tensorflow {
namespace experimental {
namespace cc {

// RuntimeBuilder is a builder used to construct a tensorflow::cc::Runtime.
// Use this to set configuration options, like threadpool size, etc.
class RuntimeBuilder {
 public:
  RuntimeBuilder() : options_(TFE_NewContextOptions()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPSruntime_builderDTh mht_0(mht_0_v, 203, "", "./tensorflow/cc/experimental/base/public/runtime_builder.h", "RuntimeBuilder");
}

  // If `use_tfrt` is true, we will use the new Tensorflow Runtime
  // (https://blog.tensorflow.org/2020/04/tfrt-new-tensorflow-runtime.html) as
  // our runtime implementation.
  RuntimeBuilder& SetUseTFRT(bool use_tfrt);

  // Build a Tensorflow Runtime.
  //
  // Params:
  //  status - Set to OK on success and an appropriate error on failure.
  // Returns:
  //  If status is not OK, returns nullptr. Otherwise, returns a
  //  unique_ptr<tensorflow::cc::Runtime>.
  std::unique_ptr<Runtime> Build(Status* status);

  // RuntimeBuilder is movable, but not copyable.
  RuntimeBuilder(RuntimeBuilder&&) = default;
  RuntimeBuilder& operator=(RuntimeBuilder&&) = default;

 private:
  // RuntimeBuilder is not copyable
  RuntimeBuilder(const RuntimeBuilder&) = delete;
  RuntimeBuilder& operator=(const RuntimeBuilder&) = delete;

  struct TFEContextOptionsDeleter {
    void operator()(TFE_ContextOptions* p) const {
      TFE_DeleteContextOptions(p);
    }
  };
  std::unique_ptr<TFE_ContextOptions, TFEContextOptionsDeleter> options_;
};

inline RuntimeBuilder& RuntimeBuilder::SetUseTFRT(bool use_tfrt) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPSruntime_builderDTh mht_1(mht_1_v, 239, "", "./tensorflow/cc/experimental/base/public/runtime_builder.h", "RuntimeBuilder::SetUseTFRT");

  TFE_ContextOptionsSetTfrt(options_.get(), use_tfrt);
  return *this;
}

inline std::unique_ptr<Runtime> RuntimeBuilder::Build(Status* status) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSbasePSpublicPSruntime_builderDTh mht_2(mht_2_v, 247, "", "./tensorflow/cc/experimental/base/public/runtime_builder.h", "RuntimeBuilder::Build");

  TFE_Context* result = TFE_NewContext(options_.get(), status->GetTFStatus());
  if (!status->ok()) {
    return nullptr;
  }
  // We can't use std::make_unique here because of its interaction with a
  // private constructor: https://abseil.io/tips/134
  return std::unique_ptr<Runtime>(new Runtime(result));
}

}  // namespace cc
}  // namespace experimental
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_RUNTIME_BUILDER_H_
