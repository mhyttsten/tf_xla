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
class MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSruntimePSruntimeDTcc {
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
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSruntimePSruntimeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSruntimePSruntimeDTcc() {
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
#include "tensorflow/cc/experimental/libtf/runtime/runtime.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/graph_function.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/cc/experimental/libexport/load.h"
#include "tensorflow/cc/experimental/libtf/function.h"
#include "tensorflow/cc/experimental/libtf/object.h"
#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"

namespace tf {
namespace libtf {
namespace runtime {

using tensorflow::AbstractContext;
using tensorflow::AbstractFunctionPtr;
using tensorflow::DataType;
using tensorflow::FunctionDef;
using tensorflow::PartialTensorShape;
using tensorflow::SavedConcreteFunction;
using tensorflow::SavedObjectGraph;
using tensorflow::Status;
using tensorflow::StructuredValue;
using tensorflow::TensorSpecProto;
using tensorflow::libexport::TFPackage;
using tensorflow::protobuf::RepeatedPtrField;
using tensorflow::tracing::graph::GraphFunction;

TaggedValue MakeCallable(const std::string& fn_name, Function fn,
                         AbstractContext* ctx) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("fn_name: \"" + fn_name + "\"");
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSruntimePSruntimeDTcc mht_0(mht_0_v, 228, "", "./tensorflow/cc/experimental/libtf/runtime/runtime.cc", "MakeCallable");

  auto CallFn = [fn_name, fn, ctx](TaggedValue args_,
                                   TaggedValue kwargs_) -> TaggedValue {
    std::cout << "Calling " << fn_name << std::endl;
    tensorflow::StatusOr<TaggedValue> v = fn.Execute(ctx, args_);
    return v.ValueOrDie();
  };
  return TaggedValue(CallFn);
}

// Import a module from a saved model.
//
// Returns a TaggedValue::Dict. All functions found on the root of the module
// will be attached as callables to this TaggedValue.
//
// `name` specifies the full path to the saved model.
//
// `ctx` should outlive the lifetime of the module.
static tensorflow::StatusOr<TaggedValue> ImportModule(String name,
                                                      AbstractContext* ctx) {
  // Load the serialized model.
  tensorflow::StatusOr<TFPackage> tf_package = TFPackage::Load(name.get());
  if (!tf_package.status().ok()) {
    return tf_package.status();
  }
  TaggedValue module = TaggedValue::Dict();

  // Initialize concrete function traces.
  const RepeatedPtrField<FunctionDef> function_defs =
      tf_package->GetFunctionDefs();
  absl::flat_hash_map<std::string, AbstractFunctionPtr> traces;
  for (auto& fdef : function_defs) {
    AbstractFunctionPtr trace(new GraphFunction(fdef), /*add_ref=*/false);
    traces[fdef.signature().name()] = trace;
  }

  // Setup polymorphic functions and wrap in Callables.
  //
  // For each child of the root, check what type it is.  If it's a
  // SavedFunction, attach that function to the module as a Callable.
  const SavedObjectGraph object_graph = tf_package->GetObjectGraph();
  auto& nodes = object_graph.nodes();
  // Get a map of the concrete functions to their input / output signatures.
  auto& concrete_functions = object_graph.concrete_functions();
  auto& root = nodes.at(0);
  for (auto& child : root.children()) {
    // The child's name describes the name of the edge that connects to the
    // parent object. This name will be the name of the object stored in the
    // generated module.
    auto& child_node = nodes.at(child.node_id());
    auto child_name = child.local_name().c_str();

    if (child_node.kind_case() == tensorflow::SavedObject::kFunction) {
      Function tf_function;
      for (const std::string& fn_name :
           child_node.function().concrete_functions()) {
        // Setup input signature.
        //
        // For now, we have to do a lot of manual digging through these and
        // assume they are tensorspecs. Once TODO(b/190203981) is done, we
        // should be able to pass along the `StructuredValue`s to an API in a
        // much cleaner way.
        //
        // TODO(b/190206621): Implement API for inspecting signatures
        SavedConcreteFunction saved_concrete_function =
            concrete_functions.at(fn_name);
        TaggedValue input_signature = TaggedValue::Tuple();
        const RepeatedPtrField<StructuredValue>& args =
            saved_concrete_function.canonicalized_input_signature()
                .tuple_value()
                .values(0)
                .tuple_value()
                .values();
        for (const StructuredValue& arg : args) {
          PartialTensorShape shape = arg.tensor_spec_value().shape();
          DataType dtype = arg.tensor_spec_value().dtype();
          TaggedValue tensor_spec(shape, dtype);
          input_signature.tuple().emplace_back(tensor_spec);
        }

        // Setup output signature.
        TensorSpecProto output_tensor_spec_proto =
            saved_concrete_function.output_signature().tensor_spec_value();
        PartialTensorShape output_shape = output_tensor_spec_proto.shape();
        DataType output_dtype = output_tensor_spec_proto.dtype();
        TaggedValue output_tensor_spec(output_shape, output_dtype);

        // Register the function trace.
        //
        // This does *not* currently register the function with the runtime.
        // Instead, we're registering JIT at call time. This is likely
        // something that we'll change in TODO(b/190070277).
        auto& trace = traces[fn_name];
        Status status = tf_function.RegisterTrace(
            std::move(trace), input_signature, output_tensor_spec);
      }
      TaggedValue callable = MakeCallable(child_name, tf_function, ctx);
      module.dict()[TaggedValue(child_name)] = callable;
    }
  }
  return module;
}

// Instantiate the Runtime, creating relevant Callables for later use.
Runtime::Runtime(AbstractContext* ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSruntimePSruntimeDTcc mht_1(mht_1_v, 335, "", "./tensorflow/cc/experimental/libtf/runtime/runtime.cc", "Runtime::Runtime");

  TaggedValue ctx_capsule =
      TaggedValue::Capsule(static_cast<void*>(ctx), [](void* p) {
        auto ctx = static_cast<AbstractContext*>(p);
        ctx->Release();
      });
  Set(String("ctx"), Handle(ctx_capsule));
  auto Load = [](Object self, String name) -> Object {
    auto ctx_capsule = self.Get<internal::Capsule>(String("ctx")).ValueOrDie();
    auto ctx = ctx_capsule.cast<AbstractContext*>();
    // TODO(b/191689645): This needs to do error handling better.
    return *Cast<Object>(Handle(*ImportModule(name, ctx)));
  };

  Set(String("Load"), Callable(TFLIB_CALLABLE_ADAPTOR(Load)));
}

tensorflow::StatusOr<Object> Runtime::Load(const String& name) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSruntimePSruntimeDTcc mht_2(mht_2_v, 355, "", "./tensorflow/cc/experimental/libtf/runtime/runtime.cc", "Runtime::Load");

  return Get<Callable>(String("Load"))->Call<Object>(*this, name);
}

}  // namespace runtime
}  // namespace libtf
}  // namespace tf
