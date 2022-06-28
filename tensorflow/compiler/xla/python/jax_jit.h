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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_JAX_JIT_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_JAX_JIT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSjax_jitDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSjax_jitDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSjax_jitDTh() {
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


#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/py_values.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/pytree.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace jax {

// Flags, such as JIT disable and the x64 mode, are controlled by:
// - a global flag value, e.g., associated to --jax_enable_x64
// - possibly a thread-local value, which initially is absl::nullopt and
//   overrides the global value if set. The thread-local state is
//   used to implement context managers that locally override the global state.
struct JitState {
  ~JitState() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSjax_jitDTh mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/python/jax_jit.h", "~JitState");

    if (extra_jit_context) {
      // We likely do not hold the GIL if this JitState is thread-local, so we
      // hand the Python object to the global reference manager to destroy.
      pybind11::object o = std::move(*extra_jit_context);
      xla::GlobalPyRefManager()->AddGarbage(absl::MakeSpan(&o, 1));
      extra_jit_context = absl::nullopt;
    }
  }

  absl::optional<bool> disable_jit;
  absl::optional<bool> enable_x64;

  // Used to manually set the default device jax should use. May be unset even
  // in global state, indicating there is no manual override.
  absl::optional<xla::ClientAndPtr<xla::PjRtDevice>> default_device;

  // Extra context that should be included in the JIT cache key. Must be
  // hashable and have an equality defined.
  absl::optional<pybind11::object> extra_jit_context;

  // A callback that, if present, is called when a JITted function is executed
  // from cache. May be unset even in global state.
  absl::optional<pybind11::function> post_hook;
};

JitState& GetGlobalState();
JitState& GetLocalState();

// Getters for JitState fields that first look in thread-local state, then
// fallback to global state.
bool GetDisableJit();
bool GetEnableX64();
absl::optional<xla::ClientAndPtr<xla::PjRtDevice>> GetDefaultDevice();
absl::optional<pybind11::function> GetPostHook();

// The signature of Python jitted function call, partitioned into:
// - dynamic positional arguments (i.e. positional args which are not static)
// - static positional arguments (i.e. the args associated to static_argnums)
// - keyword arguments
// The CallSignature should unambiguously identify a function call, thus,
// equality is based on:
// (a) Same PyTree for all dynamic positional arguments and keyword arguments
// (a) equality of the arguments and keyword arguments ArgSignature
// (a) equality (delegated to Python) of the static arguments.
struct CallSignature {
  // Not part of the signature, but we need it for error messages.
  absl::string_view function_name;

  // A PyTreeDef for each dynamic argument, positional arguments first
  // followed by keyword arguments. Keyword arguments are in the order given
  // by dynamic_arg_names.
  absl::InlinedVector<xla::PyTreeDef, 2> dynamic_arg_treedefs;
  // Dynamic keyword argument names. Interned, and sorted by the keyword
  // name.
  std::vector<pybind11::object> dynamic_arg_names;
  // Shape and dtype for both the dynamic positional arguments and the keyword
  // arguments (sorted by keyword name).
  absl::InlinedVector<xla::PyArgSignature, 2> dynamic_arg_signatures;

  // Static arguments. Contains the positional arguments sorted in argument
  // order, followed by static keyword arguments in the order given by
  // `static_arg_names`.
  std::vector<pybind11::object> static_args;
  // Static keyword argument names. Interned, and sorted by keyword name.
  std::vector<pybind11::object> static_arg_names;

  // For JIT, we need this in the key because computation follows the data, so
  // we may have multiple executables depending on the devices the data is on.
  // This is not the case for PMAP, and is set to `nullptr`.
  xla::PjRtDevice* device = nullptr;
  bool jax_enable_x64;

  // Opaque additional context that should be included as part of the cache key.
  absl::optional<pybind11::object> global_extra_jit_context;
  absl::optional<pybind11::object> thread_local_extra_jit_context;

  bool operator==(const CallSignature& other) const;
  bool operator!=(const CallSignature& other) const {
    return !(*this == other);
  }

  std::string DebugString() const;
};

template <typename H>
H AbslHashValue(H h, const CallSignature& s);

// The resulting information of the parsing and conversion of the arguments.
struct ParsedArgumentsAsBuffers {
  // The call signature will be filled during 2 steps:
  // - `ParseArguments` will fill the static arguments and the pytree
  //    structures
  // - the shapes and dtypes are filled later, by `ParseAndTransferArguments`.
  CallSignature signature;
  // The concatenation of the dynamic positional arguments and the sorted
  // keyword arguments.
  absl::InlinedVector<pybind11::object, 2> flat_dynamic_args;
  std::vector<pybind11::object> keep_alive_objects;

  // The following is only valid if the parsing succeeds.
  std::vector<xla::PjRtBuffer*> arg_buffers;
  // We may need to keep these objects around, because:
  // (a) we need to extend the lifetime of objects created within
  //    `CopyBuffersToDevice`
  // (b) `arg_buffers` do not maintain ownership
  std::vector<std::unique_ptr<xla::PjRtBuffer>> keep_alive;
};

// Filter out static arguments, flatten and concatenate other arguments (i.e.
// dynamic positional and keyword arguments), filling `arguments` in place.
xla::Status ParseArguments(pybind11::handle args,
                           const absl::optional<pybind11::kwargs>& py_kwargs,
                           absl::Span<int const> static_argnums,
                           absl::Span<pybind11::str const> static_argnames,
                           ParsedArgumentsAsBuffers& arguments);

// The function to call in `xla.cc` to add the bindings for this module.
void BuildJaxjitSubmodule(pybind11::module& m);

}  // namespace jax

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_JAX_JIT_H_
