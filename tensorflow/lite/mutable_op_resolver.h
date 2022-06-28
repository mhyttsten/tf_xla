/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MUTABLE_OP_RESOLVER_H_
#define TENSORFLOW_LITE_MUTABLE_OP_RESOLVER_H_
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
class MHTracer_DTPStensorflowPSlitePSmutable_op_resolverDTh {
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
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolverDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSmutable_op_resolverDTh() {
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

#include <string>
#include <unordered_map>
#include <utility>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"

namespace tflite {

// Some versions of gcc don't support partial specialization in class scope,
// so these are defined in a namescope.
namespace op_resolver_hasher {
template <typename V>
struct ValueHasher {
  size_t operator()(const V& v) const { return std::hash<V>()(v); }
};

template <>
struct ValueHasher<tflite::BuiltinOperator> {
  size_t operator()(const tflite::BuiltinOperator& v) const {
    return std::hash<int>()(static_cast<int>(v));
  }
};

template <typename T>
struct OperatorKeyHasher {
  size_t operator()(const T& x) const {
    size_t a = ValueHasher<typename T::first_type>()(x.first);
    size_t b = ValueHasher<typename T::second_type>()(x.second);
    return CombineHashes({a, b});
  }
};
}  // namespace op_resolver_hasher

/// An OpResolver that is mutable, also used as the op in gen_op_registration.
/// A typical usage:
///   MutableOpResolver resolver;
///   resolver.AddBuiltin(BuiltinOperator_ADD, Register_ADD());
///   resolver.AddCustom("CustomOp", Register_CUSTOM_OP());
///   InterpreterBuilder(model, resolver)(&interpreter);
class MutableOpResolver : public OpResolver {
 public:
  const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                   int version) const override;
  const TfLiteRegistration* FindOp(const char* op, int version) const override;

  /// Registers the specified `version` of the specified builtin operator `op`.
  /// Replaces any previous registration for the same operator version.
  void AddBuiltin(tflite::BuiltinOperator op,
                  const TfLiteRegistration* registration, int version = 1);

  /// Registers the specified version range (versions `min_version` to
  /// `max_version`, inclusive) of the specified builtin operator `op`.
  /// Replaces any previous registration for the same operator version.
  void AddBuiltin(tflite::BuiltinOperator op,
                  const TfLiteRegistration* registration, int min_version,
                  int max_version);

  /// Registers the specified `version` of the specified builtin operator `op`.
  /// Replaces any previous registration for the same operator version.
  void AddCustom(const char* name, const TfLiteRegistration* registration,
                 int version = 1);

  /// Registers the specified version range (versions `min_version` to
  /// `max_version`, inclusive) of the specified custom operator `name`.
  /// Replaces any previous registration for the same operator version.
  void AddCustom(const char* name, const TfLiteRegistration* registration,
                 int min_version, int max_version);

  /// Registers all operator versions supported by another MutableOpResolver.
  /// Replaces any previous registrations for the same operator versions,
  /// except that registrations made with `AddBuiltin` or `AddCustom` always
  /// take precedence over registrations made with `ChainOpResolver`.
  void AddAll(const MutableOpResolver& other);

  OpResolver::TfLiteDelegateCreators GetDelegateCreators() const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolverDTh mht_0(mht_0_v, 266, "", "./tensorflow/lite/mutable_op_resolver.h", "GetDelegateCreators");

    return delegate_creators_;
  }

 protected:
  /// Registers all operator versions supported by another OpResolver,
  /// except any already registered in this MutableOpResolver.
  /// `other` must point to an OpResolver whose lifetime is at least as long
  /// as the lifetime of the MutableOpResolver pointed to by `this`.
  /// The OpResolver pointed to by `other` should not be modified during the
  /// lifetime of this MutableOpResolver.
  void ChainOpResolver(const OpResolver* other);

  /// True if this OpResolver itself (as opposed to chained op resolvers
  /// registed with ChainOpResolver) may contain user defined ops.
  ///
  /// By "user defined" ops, we mean any op definitions other than those
  /// contained in tflite::ops::builtin::BuiltinOpResolver.
  bool may_directly_contain_user_defined_ops_ = false;

  /// A vector of delegate creators to create optional delegates for resolving
  /// and handling ops in the flatbuffer model. This may be used in addition to
  /// the standard TfLiteRegistration lookup for graph resolution.
  TfLiteDelegateCreators delegate_creators_;

 private:
  bool MayContainUserDefinedOps() const override;

  typedef std::pair<tflite::BuiltinOperator, int> BuiltinOperatorKey;
  typedef std::pair<std::string, int> CustomOperatorKey;

  std::unordered_map<BuiltinOperatorKey, TfLiteRegistration,
                     op_resolver_hasher::OperatorKeyHasher<BuiltinOperatorKey> >
      builtins_;
  std::unordered_map<CustomOperatorKey, TfLiteRegistration,
                     op_resolver_hasher::OperatorKeyHasher<CustomOperatorKey> >
      custom_ops_;
  std::vector<const OpResolver*> other_op_resolvers_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MUTABLE_OP_RESOLVER_H_
