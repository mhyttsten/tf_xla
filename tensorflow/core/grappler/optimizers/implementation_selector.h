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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_IMPLEMENTATION_SELECTOR_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_IMPLEMENTATION_SELECTOR_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTh() {
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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/function_api_info.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

// Motivation: To achieve the same high level functionality, the underlying
// implementations sometimes are different for various devices where the
// function runs. In order to achieve the correct result and best performance,
// the proper implementation needs to be picked dynamically.
//
// Currently there are two approaches to do this.
// (1) Utilize case op and dynamacically change the branch index.
// (2) Swap function implementation, it will be deprecated.
//
// Idea for approach 1.
// This transformation rewrites the DeviceIndex op with a Const op with value
// of the index of the device the associcated Case op runs.
// Example:
// def plus_one_gpu(x): return x + 1.0
// def plus_one_reference_implementation(x): return x + 1.0
// input = tf.constant(2.0, dtype=tf.float32)
// cpu_fn = lambda:plus_one_reference_implementation(input)
// gpu_fn = lambda:plus_one_gpu(input)
// control_flow_ops.execute_fn_for_device(
//  {"CPU": cpu_fn, "GPU":gpu_fn)}, default_fn=cpu_fn)
//
// Idea for approach 2.
// This transformation replaces function calls by the appropriate function
// definition based on properties of the runtime system. For instance,
// we may choose one implementation over another if we have a GPU with
// enough memory available.
//
// It is a way for the programmer to specify alternative implementations
// of the same functionality in the graph, and let TensorFlow pick the
// most appropriate one at runtime.
//
// For instance, the python code might specify:
// @Defun(tf.float32,
//        api_implements='plus_one',
//        api_preferred_device='GPU')
// def plus_one_gpu(x): return x + 1.0
//
// @Defun(tf.float32,
//        api_implements='plus_one')
// def plus_one_reference_implementation(x): return x + 1.0
// input = tf.constant(2.0, dtype=tf.float32)
//
// z = plus_one_reference_implementation(input)
// z = plus_one_gpu(input)
// print(sess.run(z))
//

// At runtime, we will select either `plus_one_gpu` or
// `plus_one_reference_implementation` based on the availability of the GPU.
//
// Available annotations:
//  - api_implements(string): all functions mapping to the same
//    string can be interchanged. For now, all functions must have the same
//    signature and overloads are not allowed. Defuns within defuns are
//    allowed.
//  - api_preferred_device(string): sets which device is preferred.
class ImplementationSelector : public CustomGraphOptimizer {
 public:
  ImplementationSelector() = default;
  ~ImplementationSelector() override = default;
  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTh mht_0(mht_0_v, 267, "", "./tensorflow/core/grappler/optimizers/implementation_selector.h", "Init");

    return Status::OK();
  }
  string name() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTh mht_1(mht_1_v, 273, "", "./tensorflow/core/grappler/optimizers/implementation_selector.h", "name");

    return "implementation_selector";
  }

  bool UsesFunctionLibrary() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTh mht_2(mht_2_v, 280, "", "./tensorflow/core/grappler/optimizers/implementation_selector.h", "UsesFunctionLibrary");
 return false; }

  // This call is not thread-safe.
  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

 private:
  Status LoadFunctions(const GraphDef& graph);
  Status MaybeOptimizeFunctionCall(utils::MutableNodeView* node_view) const;

  // Finds all call sites for functions, then replace with the appropriate
  // implementation.
  // There are two ways of calling functions:
  //  1. By specifying an op name as a function name, and
  //  2. Via the functional interface, where the function name appears as an
  //  Attr.
  //
  // There may be multiple call sites for a given function. The function body
  // may call into another function, so a function might have to be duplicated.
  // For simplicity, we do not change function bodies. Also, we do not change
  // gradients.
  Status SelectImplementation(GraphDef* graph) const;

  // Rewrites the DeviceIndex op with a Const op with value of the index of the
  // device the associcated Case op runs.

  // This function first looks up all the DeviceIndex ops.
  // Then for each of these ops, it finds the device of the
  // associated Case op that takes the DeviceIndex op as the input, and
  // caculates the index of the device in the device list of DeviceIndex op.
  // Lastly, it rewrites the DeviceIndex op with a Const op and sets the value
  // to be the index.
  //
  // Example input nodes:
  // node {
  //   name: "x"
  //   op: "DeviceIndex"
  //   device: "/device:CPU:0"
  //   attr {
  //     key: "device_names"
  //     value {
  //       list {
  //         s: "CPU"
  //         s: "TPU_REPLICATED_CORE"
  //         s: "GPU"
  //       }
  //     }
  //   }
  // }
  // node {
  //   name: "case"
  //   op: "Case"
  //   input: "x"
  //   device: "/device:GPU:0"
  //   ...
  // }
  // Example output nodes:
  //
  //  name: "x"
  //  op: "Const"
  //  device: "/device:CPU:0"
  //  attr {
  //    key: "dtype"
  //    value {
  //      type: DT_INT32
  //    }
  //  }
  //  attr {
  //    key: "value"
  //    value {
  //      tensor {
  //        dtype: DT_INT32
  //        int_val: 2
  //      }
  //    }
  //  }
  // node {
  //   name: "case"
  //   op: "Case"
  //   input: "x"
  //   device: "/device:GPU:0"
  //   ...
  // }
  Status SelectDeviceIndex(GraphDef* graph) const;

  std::unique_ptr<FunctionLibraryApiInfo> lib_info_;

  TF_DISALLOW_COPY_AND_ASSIGN(ImplementationSelector);
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_IMPLEMENTATION_SELECTOR_H_
