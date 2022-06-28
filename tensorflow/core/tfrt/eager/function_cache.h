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
#ifndef TENSORFLOW_CORE_TFRT_EAGER_FUNCTION_CACHE_H_
#define TENSORFLOW_CORE_TFRT_EAGER_FUNCTION_CACHE_H_
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
class MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cacheDTh {
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
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cacheDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cacheDTh() {
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


#include "tensorflow/compiler/mlir/tfrt/function/function.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/utils/utils.h"
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/aligned_buffer.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/mutex.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

// A reference counted `state` object that contains a BEF file, which represents
// a lowered FunctionDef. The CoreRuntimeOp is a callable handle to the function
// to be called.
class FunctionState : public ReferenceCounted<FunctionState> {
 public:
  static RCReference<FunctionState> CreateFunctionState(
      TfrtDataTypeSlice arg_types, tensorflow::DataTypeSlice ret_types,
      BefBuffer bef_buffer, RCReference<BEFFile> bef_file, CoreRuntimeOp fn,
      std::unique_ptr<tensorflow::tfrt_stub::OpKernelRunnerTable>
          runner_table) {
    return TakeRef(new FunctionState(arg_types, ret_types,
                                     std::move(bef_buffer), std::move(bef_file),
                                     std::move(fn), std::move(runner_table)));
  }

  const CoreRuntimeOp& GetFunc() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cacheDTh mht_0(mht_0_v, 225, "", "./tensorflow/core/tfrt/eager/function_cache.h", "GetFunc");
 return fn_; }

  const TfrtDataTypeVector& GetArgTypes() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cacheDTh mht_1(mht_1_v, 230, "", "./tensorflow/core/tfrt/eager/function_cache.h", "GetArgTypes");
 return arg_types_; }

  const tensorflow::DataTypeVector& GetRetTypes() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cacheDTh mht_2(mht_2_v, 235, "", "./tensorflow/core/tfrt/eager/function_cache.h", "GetRetTypes");
 return ret_types_; }

  tensorflow::tfrt_stub::OpKernelRunnerTable* GetRunnerTable() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cacheDTh mht_3(mht_3_v, 240, "", "./tensorflow/core/tfrt/eager/function_cache.h", "GetRunnerTable");

    return runner_table_.get();
  }

 private:
  FunctionState(
      TfrtDataTypeSlice arg_types, tensorflow::DataTypeSlice ret_types,
      BefBuffer bef_buffer, RCReference<BEFFile> bef_file, CoreRuntimeOp fn,
      std::unique_ptr<tensorflow::tfrt_stub::OpKernelRunnerTable> runner_table)
      : arg_types_(arg_types.begin(), arg_types.end()),
        ret_types_(ret_types.begin(), ret_types.end()),
        bef_buffer_(std::move(bef_buffer)),
        bef_file_(std::move(bef_file)),
        fn_(std::move(fn)),
        runner_table_(std::move(runner_table)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cacheDTh mht_4(mht_4_v, 257, "", "./tensorflow/core/tfrt/eager/function_cache.h", "FunctionState");
}

  TfrtDataTypeVector arg_types_;
  tensorflow::DataTypeVector ret_types_;
  BefBuffer bef_buffer_;
  RCReference<BEFFile> bef_file_;
  const CoreRuntimeOp fn_;

  // This is the op_kernel cache used by kernel fallback compact mode. We will
  // initialize this table right after lowering the function.
  std::unique_ptr<tensorflow::tfrt_stub::OpKernelRunnerTable> runner_table_;
};

// Cache for a single core runtime op or function (composite op). Thread safe.
class FunctionCache {
 public:
  // Iterate the cache and erase the op(s) with the specified op_name.
  void RemoveFunction(string_view op_name) TFRT_EXCLUDES(cache_mu_);

  struct FunctionCacheResult {
    RCReference<FunctionState> function_state;
    bool is_cache_miss;
  };

  typedef std::function<tensorflow::Status(
      tensorflow::tfrt_stub::OpKernelRunnerTable*,
      RCReference<RequestContext>*)>
      RequestCtxBuilder;

  // Helper function to look up the cache. If miss, insert the function to the
  // cache.
  // When the return status is OK, `result` is set.
  tensorflow::Status GetOrAddFunction(
      const std::string& op_name, const std::string& device_name,
      const tensorflow::DeviceSet& device_set,
      tensorflow::EagerContext* eager_ctx, tfrt::CoreRuntime* corert,
      RequestCtxBuilder request_ctx_fn, Location loc,
      tensorflow::TfrtFunctionCompileOptions compile_options,
      tfrt::ArrayRef<const Device*> input_devices, FunctionCacheResult* result);

  // The following helper functions are for debugging and testing only.
  size_t Size() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cacheDTh mht_5(mht_5_v, 301, "", "./tensorflow/core/tfrt/eager/function_cache.h", "Size");

    mutex_lock l(cache_mu_);
    return cache_.size();
  }

  bool Contains(string_view op_name, string_view device_name) const {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   mht_6_v.push_back("device_name: \"" + std::string(device_name.data(), device_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cacheDTh mht_6(mht_6_v, 311, "", "./tensorflow/core/tfrt/eager/function_cache.h", "Contains");

    const CacheKey& cache_key{op_name.str(), device_name.str()};
    mutex_lock l(cache_mu_);
    return cache_.find(cache_key) != cache_.end();
  }

 private:
  // Note: Currently the key is a pair of op_name and device_name. New features
  // may be added in the future.
  struct CacheKey {
    std::string op_name, device_name;

    bool operator==(const CacheKey& other) const {
      return (this->op_name == other.op_name &&
              this->device_name == other.device_name);
    }
  };

  struct CacheKeyHash {
    size_t operator()(const CacheKey& pair) const {
      return std::hash<std::string>()(pair.op_name) ^
             std::hash<std::string>()(pair.device_name);
    }
  };

  mutable mutex cache_mu_;
  std::unordered_map<CacheKey, RCReference<FunctionState>, CacheKeyHash> cache_
      TFRT_GUARDED_BY(cache_mu_);
};

}  // namespace tf
}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_EAGER_FUNCTION_CACHE_H_
