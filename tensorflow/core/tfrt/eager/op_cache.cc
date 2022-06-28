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
class MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cacheDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cacheDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cacheDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/eager/op_cache.h"

#include "tensorflow/core/tfrt/eager/c_api_tfrt.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

Expected<CoreRuntimeOp*> OpCache::GetOrAddOp(
    string_view op_name, OpHandler* op_handler, string_view device_name,
    llvm::SmallVector<string_view, 4> dtypes,
    OperationInterface* const op_interface) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   mht_0_v.push_back("device_name: \"" + std::string(device_name.data(), device_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cacheDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/tfrt/eager/op_cache.cc", "OpCache::GetOrAddOp");

  CacheKey cache_key{op_name, op_handler,
                     (op_handler == nullptr ? device_name : ""), dtypes};
  {
    mutex_lock l(cache_mu_);
    auto iter = cache_.find(cache_key);
    if (iter != cache_.end()) return &iter->second;
  }

  ContextInterface* context = op_interface->context_;

  auto tfrt_op_name = StrCat("tf.", op_name);
  op_interface->MaybeInferInputAttrs();
  if (op_handler == nullptr) {
    tensorflow::Status s = context->SelectOpHandlerFromNodeDef(
        *op_interface, &op_interface->fallback_attrs_.BuildNodeDef(),
        &op_handler);
    if (!s.ok()) return MakeStringError(s.error_message());
  }
  Expected<CoreRuntimeOp> expected_op =
      context->GetCoreRuntime()->MakeOp(tfrt_op_name, op_handler);
  if (!expected_op) return MakeStringError(expected_op.takeError());

  mutex_lock l(cache_mu_);
  // Insert the new op to cache. If an entry with the same key is already
  // present in the cache at this moment due to race condition, overwrites it.
  cache_key.MakeConcrete();
  cache_[cache_key] = std::move(expected_op.get());
  return &cache_[cache_key];
}

Expected<CoreRuntimeOp*> OpCache::GetOrAddXlaOp(string_view op_name,
                                                ContextInterface* context) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cacheDTcc mht_1(mht_1_v, 235, "", "./tensorflow/core/tfrt/eager/op_cache.cc", "OpCache::GetOrAddXlaOp");

  // Device name and dtype are not meaningful to a XLA op.
  CacheKey cache_key{op_name, nullptr, "", {}};
  {
    mutex_lock l(cache_mu_);
    auto iter = cache_.find(cache_key);
    if (iter != cache_.end()) return &iter->second;
  }

  auto tfrt_op_name = StrCat("tf.", op_name);
  Expected<CoreRuntimeOp> expected_op = context->GetCoreRuntime()->MakeOp(
      tfrt_op_name, context->GetFallbackOpHandler());
  if (!expected_op) return MakeStringError(expected_op.takeError());

  mutex_lock l(cache_mu_);
  // Insert the new op to cache. If an entry with the same key is already
  // present in the cache at this moment due to race condition, overwrites it.
  cache_key.MakeConcrete();
  cache_[cache_key] = std::move(expected_op.get());
  return &cache_[cache_key];
}

}  // namespace tf
}  // namespace tfrt
