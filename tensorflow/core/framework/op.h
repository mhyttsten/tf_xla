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

#ifndef TENSORFLOW_CORE_FRAMEWORK_OP_H_
#define TENSORFLOW_CORE_FRAMEWORK_OP_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSopDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSopDTh() {
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


#include <functional>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/full_type_inference_util.h"
#include "tensorflow/core/framework/full_type_util.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Users that want to look up an OpDef by type name should take an
// OpRegistryInterface.  Functions accepting a
// (const) OpRegistryInterface* may call LookUp() from multiple threads.
class OpRegistryInterface {
 public:
  virtual ~OpRegistryInterface();

  // Returns an error status and sets *op_reg_data to nullptr if no OpDef is
  // registered under that name, otherwise returns the registered OpDef.
  // Caller must not delete the returned pointer.
  virtual Status LookUp(const std::string& op_type_name,
                        const OpRegistrationData** op_reg_data) const = 0;

  // Shorthand for calling LookUp to get the OpDef.
  Status LookUpOpDef(const std::string& op_type_name,
                     const OpDef** op_def) const;
};

// The standard implementation of OpRegistryInterface, along with a
// global singleton used for registering ops via the REGISTER
// macros below.  Thread-safe.
//
// Example registration:
//   OpRegistry::Global()->Register(
//     [](OpRegistrationData* op_reg_data)->Status {
//       // Populate *op_reg_data here.
//       return Status::OK();
//   });
class OpRegistry : public OpRegistryInterface {
 public:
  typedef std::function<Status(OpRegistrationData*)> OpRegistrationDataFactory;

  OpRegistry();
  ~OpRegistry() override;

  void Register(const OpRegistrationDataFactory& op_data_factory);

  Status LookUp(const std::string& op_type_name,
                const OpRegistrationData** op_reg_data) const override;

  // Returns OpRegistrationData* of registered op type, else returns nullptr.
  const OpRegistrationData* LookUp(const std::string& op_type_name) const;

  // Fills *ops with all registered OpDefs (except those with names
  // starting with '_' if include_internal == false) sorted in
  // ascending alphabetical order.
  void Export(bool include_internal, OpList* ops) const;

  // Returns ASCII-format OpList for all registered OpDefs (except
  // those with names starting with '_' if include_internal == false).
  std::string DebugString(bool include_internal) const;

  // A singleton available at startup.
  static OpRegistry* Global();

  // Get all registered ops.
  void GetRegisteredOps(std::vector<OpDef>* op_defs);

  // Get all `OpRegistrationData`s.
  void GetOpRegistrationData(std::vector<OpRegistrationData>* op_data);

  // Registers a function that validates op registry.
  void RegisterValidator(
      std::function<Status(const OpRegistryInterface&)> validator) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_0(mht_0_v, 273, "", "./tensorflow/core/framework/op.h", "RegisterValidator");

    op_registry_validator_ = std::move(validator);
  }

  // Watcher, a function object.
  // The watcher, if set by SetWatcher(), is called every time an op is
  // registered via the Register function. The watcher is passed the Status
  // obtained from building and adding the OpDef to the registry, and the OpDef
  // itself if it was successfully built. A watcher returns a Status which is in
  // turn returned as the final registration status.
  typedef std::function<Status(const Status&, const OpDef&)> Watcher;

  // An OpRegistry object has only one watcher. This interface is not thread
  // safe, as different clients are free to set the watcher any time.
  // Clients are expected to atomically perform the following sequence of
  // operations :
  // SetWatcher(a_watcher);
  // Register some ops;
  // op_registry->ProcessRegistrations();
  // SetWatcher(nullptr);
  // Returns a non-OK status if a non-null watcher is over-written by another
  // non-null watcher.
  Status SetWatcher(const Watcher& watcher);

  // Process the current list of deferred registrations. Note that calls to
  // Export, LookUp and DebugString would also implicitly process the deferred
  // registrations. Returns the status of the first failed op registration or
  // Status::OK() otherwise.
  Status ProcessRegistrations() const;

  // Defer the registrations until a later call to a function that processes
  // deferred registrations are made. Normally, registrations that happen after
  // calls to Export, LookUp, ProcessRegistrations and DebugString are processed
  // immediately. Call this to defer future registrations.
  void DeferRegistrations();

  // Clear the registrations that have been deferred.
  void ClearDeferredRegistrations();

 private:
  // Ensures that all the functions in deferred_ get called, their OpDef's
  // registered, and returns with deferred_ empty.  Returns true the first
  // time it is called. Prints a fatal log if any op registration fails.
  bool MustCallDeferred() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Calls the functions in deferred_ and registers their OpDef's
  // It returns the Status of the first failed op registration or Status::OK()
  // otherwise.
  Status CallDeferred() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Add 'def' to the registry with additional data 'data'. On failure, or if
  // there is already an OpDef with that name registered, returns a non-okay
  // status.
  Status RegisterAlreadyLocked(const OpRegistrationDataFactory& op_data_factory)
      const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const OpRegistrationData* LookUpSlow(const std::string& op_type_name) const;

  mutable mutex mu_;
  // Functions in deferred_ may only be called with mu_ held.
  mutable std::vector<OpRegistrationDataFactory> deferred_ TF_GUARDED_BY(mu_);
  // Values are owned.
  mutable std::unordered_map<string, const OpRegistrationData*> registry_
      TF_GUARDED_BY(mu_);
  mutable bool initialized_ TF_GUARDED_BY(mu_);

  // Registry watcher.
  mutable Watcher watcher_ TF_GUARDED_BY(mu_);

  std::function<Status(const OpRegistryInterface&)> op_registry_validator_;
};

// An adapter to allow an OpList to be used as an OpRegistryInterface.
//
// Note that shape inference functions are not passed in to OpListOpRegistry, so
// it will return an unusable shape inference function for every op it supports;
// therefore, it should only be used in contexts where this is okay.
class OpListOpRegistry : public OpRegistryInterface {
 public:
  // Does not take ownership of op_list, *op_list must outlive *this.
  explicit OpListOpRegistry(const OpList* op_list);
  ~OpListOpRegistry() override;
  Status LookUp(const std::string& op_type_name,
                const OpRegistrationData** op_reg_data) const override;

  // Returns OpRegistrationData* of op type in list, else returns nullptr.
  const OpRegistrationData* LookUp(const std::string& op_type_name) const;

 private:
  // Values are owned.
  std::unordered_map<string, const OpRegistrationData*> index_;
};

// Support for defining the OpDef (specifying the semantics of the Op and how
// it should be created) and registering it in the OpRegistry::Global()
// registry.  Usage:
//
// REGISTER_OP("my_op_name")
//     .Attr("<name>:<type>")
//     .Attr("<name>:<type>=<default>")
//     .Input("<name>:<type-expr>")
//     .Input("<name>:Ref(<type-expr>)")
//     .Output("<name>:<type-expr>")
//     .Doc(R"(
// <1-line summary>
// <rest of the description (potentially many lines)>
// <name-of-attr-input-or-output>: <description of name>
// <name-of-attr-input-or-output>: <description of name;
//   if long, indent the description on subsequent lines>
// )");
//
// Note: .Doc() should be last.
// For details, see the OpDefBuilder class in op_def_builder.h.

namespace register_op {

class OpDefBuilderWrapper {
 public:
  explicit OpDefBuilderWrapper(const char name[]) : builder_(name) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_1(mht_1_v, 394, "", "./tensorflow/core/framework/op.h", "OpDefBuilderWrapper");
}
  OpDefBuilderWrapper& Attr(std::string spec) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("spec: \"" + spec + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_2(mht_2_v, 399, "", "./tensorflow/core/framework/op.h", "Attr");

    builder_.Attr(std::move(spec));
    return *this;
  }
  OpDefBuilderWrapper& Input(std::string spec) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("spec: \"" + spec + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_3(mht_3_v, 407, "", "./tensorflow/core/framework/op.h", "Input");

    builder_.Input(std::move(spec));
    return *this;
  }
  OpDefBuilderWrapper& Output(std::string spec) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("spec: \"" + spec + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_4(mht_4_v, 415, "", "./tensorflow/core/framework/op.h", "Output");

    builder_.Output(std::move(spec));
    return *this;
  }
  OpDefBuilderWrapper& SetIsCommutative() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_5(mht_5_v, 422, "", "./tensorflow/core/framework/op.h", "SetIsCommutative");

    builder_.SetIsCommutative();
    return *this;
  }
  OpDefBuilderWrapper& SetIsAggregate() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_6(mht_6_v, 429, "", "./tensorflow/core/framework/op.h", "SetIsAggregate");

    builder_.SetIsAggregate();
    return *this;
  }
  OpDefBuilderWrapper& SetIsStateful() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_7(mht_7_v, 436, "", "./tensorflow/core/framework/op.h", "SetIsStateful");

    builder_.SetIsStateful();
    return *this;
  }
  OpDefBuilderWrapper& SetDoNotOptimize() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_8(mht_8_v, 443, "", "./tensorflow/core/framework/op.h", "SetDoNotOptimize");

    // We don't have a separate flag to disable optimizations such as constant
    // folding and CSE so we reuse the stateful flag.
    builder_.SetIsStateful();
    return *this;
  }
  OpDefBuilderWrapper& SetAllowsUninitializedInput() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_9(mht_9_v, 452, "", "./tensorflow/core/framework/op.h", "SetAllowsUninitializedInput");

    builder_.SetAllowsUninitializedInput();
    return *this;
  }
  OpDefBuilderWrapper& Deprecated(int version, std::string explanation) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("explanation: \"" + explanation + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_10(mht_10_v, 460, "", "./tensorflow/core/framework/op.h", "Deprecated");

    builder_.Deprecated(version, std::move(explanation));
    return *this;
  }
  OpDefBuilderWrapper& Doc(std::string text) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_11(mht_11_v, 468, "", "./tensorflow/core/framework/op.h", "Doc");

    builder_.Doc(std::move(text));
    return *this;
  }
  OpDefBuilderWrapper& SetShapeFn(OpShapeInferenceFn fn) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_12(mht_12_v, 475, "", "./tensorflow/core/framework/op.h", "SetShapeFn");

    builder_.SetShapeFn(std::move(fn));
    return *this;
  }
  OpDefBuilderWrapper& SetIsDistributedCommunication() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_13(mht_13_v, 482, "", "./tensorflow/core/framework/op.h", "SetIsDistributedCommunication");

    builder_.SetIsDistributedCommunication();
    return *this;
  }

  OpDefBuilderWrapper& SetTypeConstructor(OpTypeConstructor fn) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_14(mht_14_v, 490, "", "./tensorflow/core/framework/op.h", "SetTypeConstructor");

    builder_.SetTypeConstructor(std::move(fn));
    return *this;
  }

  OpDefBuilderWrapper& SetForwardTypeFn(ForwardTypeInferenceFn fn) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_15(mht_15_v, 498, "", "./tensorflow/core/framework/op.h", "SetForwardTypeFn");

    builder_.SetForwardTypeFn(std::move(fn));
    return *this;
  }

  const ::tensorflow::OpDefBuilder& builder() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSopDTh mht_16(mht_16_v, 506, "", "./tensorflow/core/framework/op.h", "builder");
 return builder_; }

  InitOnStartupMarker operator()();

 private:
  mutable ::tensorflow::OpDefBuilder builder_;
};

}  // namespace register_op

#define REGISTER_OP_IMPL(ctr, name, is_system_op)                         \
  static ::tensorflow::InitOnStartupMarker const register_op##ctr         \
      TF_ATTRIBUTE_UNUSED =                                               \
          TF_INIT_ON_STARTUP_IF(is_system_op || SHOULD_REGISTER_OP(name)) \
          << ::tensorflow::register_op::OpDefBuilderWrapper(name)

#define REGISTER_OP(name)        \
  TF_ATTRIBUTE_ANNOTATE("tf:op") \
  TF_NEW_ID_FOR_INIT(REGISTER_OP_IMPL, name, false)

// The `REGISTER_SYSTEM_OP()` macro acts as `REGISTER_OP()` except
// that the op is registered unconditionally even when selective
// registration is used.
#define REGISTER_SYSTEM_OP(name)        \
  TF_ATTRIBUTE_ANNOTATE("tf:op")        \
  TF_ATTRIBUTE_ANNOTATE("tf:op:system") \
  TF_NEW_ID_FOR_INIT(REGISTER_OP_IMPL, name, true)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_OP_H_
