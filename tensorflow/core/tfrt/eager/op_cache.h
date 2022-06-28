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
#ifndef TENSORFLOW_CORE_TFRT_EAGER_OP_CACHE_H_
#define TENSORFLOW_CORE_TFRT_EAGER_OP_CACHE_H_
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
class MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cacheDTh {
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
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cacheDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cacheDTh() {
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


#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/utils/utils.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_handler.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/mutex.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

class ContextInterface;
class OperationInterface;

// Cache for a single core runtime op. Thread safe.
class OpCache {
 public:
  // Helper function to look up the cache. If miss, insert the CoreRuntimeOp
  // to the cache.
  Expected<CoreRuntimeOp*> GetOrAddOp(string_view op_name,
                                      OpHandler* op_handler,
                                      string_view device_name,
                                      llvm::SmallVector<string_view, 4> dtypes,
                                      OperationInterface* const op_interface)
      TFRT_EXCLUDES(cache_mu_);

  // Compile with XLA is currently supported via fallback, and the compilation
  // result is a CoreRuntimeOp.
  // TODO(tfrt-devs): Native support of compile_with_xla.
  Expected<CoreRuntimeOp*> GetOrAddXlaOp(string_view op_name,
                                         ContextInterface* context)
      TFRT_EXCLUDES(cache_mu_);

  // The following helper functions are for debugging and testing only.
  size_t Size() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cacheDTh mht_0(mht_0_v, 231, "", "./tensorflow/core/tfrt/eager/op_cache.h", "Size");

    mutex_lock l(cache_mu_);
    return cache_.size();
  }

  bool Contains(string_view op_name, OpHandler* op_handler,
                string_view device_name,
                llvm::SmallVector<string_view, 4> dtypes) const {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   mht_1_v.push_back("device_name: \"" + std::string(device_name.data(), device_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cacheDTh mht_1(mht_1_v, 243, "", "./tensorflow/core/tfrt/eager/op_cache.h", "Contains");

    const CacheKey& cache_key{op_name, op_handler,
                              (op_handler == nullptr ? device_name : ""),
                              dtypes};
    mutex_lock l(cache_mu_);
    return cache_.find(cache_key) != cache_.end();
  }

 private:
  class CacheKey {
   public:
    CacheKey(string_view op_name, OpHandler* op_handler,
             string_view device_name, llvm::SmallVector<string_view, 4> dtypes)
        : op_handler_(op_handler),
          op_name_(op_name),
          device_name_(device_name),
          dtypes_(dtypes) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   mht_2_v.push_back("device_name: \"" + std::string(device_name.data(), device_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cacheDTh mht_2(mht_2_v, 264, "", "./tensorflow/core/tfrt/eager/op_cache.h", "CacheKey");
}

    CacheKey(const CacheKey& other)
        : op_handler_(other.op_handler_),
          op_name_(other.op_name_),
          device_name_(other.device_name_),
          dtypes_(other.dtypes_) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cacheDTh mht_3(mht_3_v, 273, "", "./tensorflow/core/tfrt/eager/op_cache.h", "CacheKey");

      // Copy the concrete strings if the key is concrete, and set the
      // string_views to refer to the concrete strings.
      if (other.is_concrete_) {
        op_name_concrete_ = other.op_name_concrete_;
        op_name_ = op_name_concrete_.data();
        device_name_concrete_ = other.device_name_concrete_;
        device_name_ = device_name_concrete_.data();
        size_t n = other.dtypes_concrete_.size();
        dtypes_concrete_.reserve(n);
        dtypes_.clear();
        for (size_t i = 0; i < n; ++i) {
          dtypes_concrete_.push_back(other.dtypes_concrete_[i]);
          dtypes_.push_back(dtypes_concrete_[i].data());
        }
        is_concrete_ = true;
      }
    }

    // Make the cache key concrete by copying the key components (strings) to
    // internal storage.
    void MakeConcrete() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cacheDTh mht_4(mht_4_v, 297, "", "./tensorflow/core/tfrt/eager/op_cache.h", "MakeConcrete");

      op_name_concrete_ = op_name_.str();
      device_name_concrete_ = device_name_.str();
      dtypes_concrete_.reserve(dtypes_.size());
      for (const auto& dtype : dtypes_) dtypes_concrete_.push_back(dtype.str());
      is_concrete_ = true;
    }

    bool operator==(const CacheKey& other) const {
      // During comparing keys, self or other can be either concrete or not.
      // If a CacheKey is concrete, it's likely that the string_view fields
      // are not valid (for example the key is obtained from the cache). We
      // need to make the string_view fields refer to the concrete fields
      // by constructing copies of them.
      CacheKey lhs{*this};
      CacheKey rhs{other};

      if (lhs.op_handler_ != rhs.op_handler_) return false;
      if (lhs.dtypes_.size() != rhs.dtypes_.size()) return false;

      for (size_t i = 0, n = lhs.dtypes_.size(); i < n; ++i) {
        if (lhs.dtypes_[i] != rhs.dtypes_[i]) return false;
      }
      return (lhs.op_name_ == rhs.op_name_ &&
              lhs.device_name_ == rhs.device_name_);
    }

    string_view OpName() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cacheDTh mht_5(mht_5_v, 327, "", "./tensorflow/core/tfrt/eager/op_cache.h", "OpName");
 return op_name_; }

    string_view DeviceName() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cacheDTh mht_6(mht_6_v, 332, "", "./tensorflow/core/tfrt/eager/op_cache.h", "DeviceName");
 return device_name_; }

    const llvm::SmallVector<string_view, 4>& Dtypes() { return dtypes_; }

   private:
    class OpHandler* op_handler_;
    // friend size_t CacheKeyHash::operator()(const CacheKey& input_key);
    // string_view is used for efficient cache look up to avoid string copy.
    string_view op_name_, device_name_;
    llvm::SmallVector<string_view, 4> dtypes_;

    // Concrete string is used for storing cache key, since the lifetime
    // of the strings should be the same as the container.
    bool is_concrete_ = false;
    std::string op_name_concrete_, device_name_concrete_;
    llvm::SmallVector<std::string, 4> dtypes_concrete_;
  };

  class CacheKeyHash {
   public:
    tensorflow::Fprint128 FingerprintCat128(
        const tensorflow::Fprint128& a, const tensorflow::Fprint128& b) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cacheDTh mht_7(mht_7_v, 356, "", "./tensorflow/core/tfrt/eager/op_cache.h", "FingerprintCat128");

      return {tensorflow::FingerprintCat64(a.low64, b.low64),
              tensorflow::FingerprintCat64(a.high64, b.high64)};
    }

    size_t operator()(const CacheKey& input_key) const {
      CacheKey key{input_key};
      tensorflow::Fprint128 hash = tensorflow::Fingerprint128(
          {key.OpName().data(), key.OpName().size()});
      hash = FingerprintCat128(
          hash, tensorflow::Fingerprint128(
                    {key.DeviceName().data(), key.DeviceName().size()}));
      for (const auto& dtype : key.Dtypes())
        hash = FingerprintCat128(
            hash, tensorflow::Fingerprint128({dtype.data(), dtype.size()}));
      return hash.high64 ^ hash.low64;
    }
  };

  mutable mutex cache_mu_;
  std::unordered_map<CacheKey, CoreRuntimeOp, CacheKeyHash> cache_
      TFRT_GUARDED_BY(cache_mu_);
};

}  // namespace tf
}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_EAGER_OP_CACHE_H_
