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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEBUGGER_STATE_INTERFACE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEBUGGER_STATE_INTERFACE_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSdebugger_state_interfaceDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdebugger_state_interfaceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSdebugger_state_interfaceDTh() {
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

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/debug.pb.h"

namespace tensorflow {

// Returns a summary string for the list of debug tensor watches.
const string SummarizeDebugTensorWatches(
    const protobuf::RepeatedPtrField<DebugTensorWatch>& watches);

// An abstract interface for storing and retrieving debugging information.
class DebuggerStateInterface {
 public:
  virtual ~DebuggerStateInterface() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdebugger_state_interfaceDTh mht_0(mht_0_v, 206, "", "./tensorflow/core/common_runtime/debugger_state_interface.h", "~DebuggerStateInterface");
}

  // Publish metadata about the debugged Session::Run() call.
  //
  // Args:
  //   global_step: A global step count supplied by the caller of
  //     Session::Run().
  //   session_run_index: A chronologically sorted index for calls to the Run()
  //     method of the Session object.
  //   executor_step_index: A chronologically sorted index of invocations of the
  //     executor charged to serve this Session::Run() call.
  //   input_names: Name of the input Tensors (feed keys).
  //   output_names: Names of the fetched Tensors.
  //   target_names: Names of the target nodes.
  virtual Status PublishDebugMetadata(
      const int64_t global_step, const int64_t session_run_index,
      const int64_t executor_step_index, const std::vector<string>& input_names,
      const std::vector<string>& output_names,
      const std::vector<string>& target_nodes) = 0;
};

class DebugGraphDecoratorInterface {
 public:
  virtual ~DebugGraphDecoratorInterface() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdebugger_state_interfaceDTh mht_1(mht_1_v, 232, "", "./tensorflow/core/common_runtime/debugger_state_interface.h", "~DebugGraphDecoratorInterface");
}

  // Insert special-purpose debug nodes to graph and dump the graph for
  // record. See the documentation of DebugNodeInserter::InsertNodes() for
  // details.
  virtual Status DecorateGraph(Graph* graph, Device* device) = 0;

  // Publish Graph to debug URLs.
  virtual Status PublishGraph(const Graph& graph,
                              const string& device_name) = 0;
};

typedef std::function<std::unique_ptr<DebuggerStateInterface>(
    const DebugOptions& options)>
    DebuggerStateFactory;

// Contains only static methods for registering DebuggerStateFactory.
// We don't expect to create any instances of this class.
// Call DebuggerStateRegistry::RegisterFactory() at initialization time to
// define a global factory that creates instances of DebuggerState, then call
// DebuggerStateRegistry::CreateState() to create a single instance.
class DebuggerStateRegistry {
 public:
  // Registers a function that creates a concrete DebuggerStateInterface
  // implementation based on DebugOptions.
  static void RegisterFactory(const DebuggerStateFactory& factory);

  // If RegisterFactory() has been called, creates and supplies a concrete
  // DebuggerStateInterface implementation using the registered factory,
  // owned by the caller and return an OK Status. Otherwise returns an error
  // Status.
  static Status CreateState(const DebugOptions& debug_options,
                            std::unique_ptr<DebuggerStateInterface>* state);

 private:
  static DebuggerStateFactory* factory_;

  TF_DISALLOW_COPY_AND_ASSIGN(DebuggerStateRegistry);
};

typedef std::function<std::unique_ptr<DebugGraphDecoratorInterface>(
    const DebugOptions& options)>
    DebugGraphDecoratorFactory;

class DebugGraphDecoratorRegistry {
 public:
  static void RegisterFactory(const DebugGraphDecoratorFactory& factory);

  static Status CreateDecorator(
      const DebugOptions& options,
      std::unique_ptr<DebugGraphDecoratorInterface>* decorator);

 private:
  static DebugGraphDecoratorFactory* factory_;

  TF_DISALLOW_COPY_AND_ASSIGN(DebugGraphDecoratorRegistry);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEBUGGER_STATE_INTERFACE_H_
