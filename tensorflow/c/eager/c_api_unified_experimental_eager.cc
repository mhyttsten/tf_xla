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
class MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_eagerDTcc {
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
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_eagerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_eagerDTcc() {
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

#include <vector>

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/strcat.h"

// =============================================================================
// Public C API entry points
// These are only the entry points specific to the Eager API.
// =============================================================================

using tensorflow::AbstractContext;
using tensorflow::AbstractTensorHandle;
using tensorflow::dyn_cast;
using tensorflow::ImmediateExecutionContext;
using tensorflow::ImmediateExecutionTensorHandle;
using tensorflow::string;
using tensorflow::unwrap;
using tensorflow::wrap;
using tensorflow::strings::StrCat;

TF_ExecutionContext* TF_NewEagerExecutionContext(TFE_ContextOptions* options,
                                                 TF_Status* s) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_eagerDTcc mht_0(mht_0_v, 216, "", "./tensorflow/c/eager/c_api_unified_experimental_eager.cc", "TF_NewEagerExecutionContext");

  TFE_Context* c_ctx = TFE_NewContext(options, s);
  if (TF_GetCode(s) != TF_OK) {
    return nullptr;
  }
  return wrap(static_cast<AbstractContext*>(unwrap(c_ctx)));
}

TF_AbstractTensor* TF_CreateAbstractTensorFromEagerTensor(TFE_TensorHandle* t,
                                                          TF_Status* s) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_eagerDTcc mht_1(mht_1_v, 228, "", "./tensorflow/c/eager/c_api_unified_experimental_eager.cc", "TF_CreateAbstractTensorFromEagerTensor");

  return wrap(static_cast<AbstractTensorHandle*>(unwrap(t)));
}

TFE_TensorHandle* TF_AbstractTensorGetEagerTensor(TF_AbstractTensor* at,
                                                  TF_Status* s) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_eagerDTcc mht_2(mht_2_v, 236, "", "./tensorflow/c/eager/c_api_unified_experimental_eager.cc", "TF_AbstractTensorGetEagerTensor");

  auto handle = dyn_cast<ImmediateExecutionTensorHandle>(unwrap(at));
  if (!handle) {
    string msg =
        StrCat("Not an eager tensor handle.", reinterpret_cast<uintptr_t>(at));
    TF_SetStatus(s, TF_INVALID_ARGUMENT, msg.c_str());
    return nullptr;
  }
  return wrap(handle);
}

TFE_Context* TF_ExecutionContextGetTFEContext(TF_ExecutionContext* ctx,
                                              TF_Status* s) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_eagerDTcc mht_3(mht_3_v, 251, "", "./tensorflow/c/eager/c_api_unified_experimental_eager.cc", "TF_ExecutionContextGetTFEContext");

  auto imm_ctx = dyn_cast<ImmediateExecutionContext>(unwrap(ctx));
  if (!imm_ctx) {
    string msg =
        StrCat("Not an eager context.", reinterpret_cast<uintptr_t>(ctx));
    TF_SetStatus(s, TF_INVALID_ARGUMENT, msg.c_str());
    return nullptr;
  }
  return wrap(imm_ctx);
}
