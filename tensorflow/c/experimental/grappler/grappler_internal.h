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
// Classes and utilities that work with Graph C API for internal use.
// This includes functions used for optimizer registration and interfaces needed
// for testing.

#ifndef TENSORFLOW_C_EXPERIMENTAL_GRAPPLER_GRAPPLER_INTERNAL_H_
#define TENSORFLOW_C_EXPERIMENTAL_GRAPPLER_GRAPPLER_INTERNAL_H_
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
class MHTracer_DTPStensorflowPScPSexperimentalPSgrapplerPSgrappler_internalDTh {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSgrapplerPSgrappler_internalDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSgrapplerPSgrappler_internalDTh() {
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
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

// Plugin initialization function that a device plugin
// must define.
typedef void (*TFInitGraphPluginFn)(TP_OptimizerRegistrationParams* const,
                                    TF_Status* const);

// Registers Graph optimizers.
Status InitGraphPlugin(void* dso_handle);

// Allow registering a graph optimizer using a function (used for
// testing).
Status InitGraphPlugin(TFInitGraphPluginFn init_fn);

struct GrapplerItem;
class Cluster;

struct TFStatusDeleter {
  void operator()(TF_Status* s) const { TF_DeleteStatus(s); }
};
using OwnedTFStatus = std::unique_ptr<TF_Status, TFStatusDeleter>;

struct TFBufferDeleter {
  void operator()(TF_Buffer* buf) const { TF_DeleteBuffer(buf); }
};
using OwnedTFBuffer = std::unique_ptr<TF_Buffer, TFBufferDeleter>;

class CGraphOptimizer : public CustomGraphOptimizer {
 public:
  explicit CGraphOptimizer(TP_Optimizer optimizer, const char* device_type)
      : optimizer_(optimizer), device_type_(device_type) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("device_type: \"" + (device_type == nullptr ? std::string("nullptr") : std::string((char*)device_type)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgrapplerPSgrappler_internalDTh mht_0(mht_0_v, 237, "", "./tensorflow/c/experimental/grappler/grappler_internal.h", "CGraphOptimizer");

    if (optimizer.create_func != nullptr) {
      c_optimizer_ = (*optimizer_.create_func)();
    } else {
      c_optimizer_ = nullptr;
    }
  }
  std::string name() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgrapplerPSgrappler_internalDTh mht_1(mht_1_v, 247, "", "./tensorflow/c/experimental/grappler/grappler_internal.h", "name");
 return "PluggableGraphOptimizer"; }
  bool UsesFunctionLibrary() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgrapplerPSgrappler_internalDTh mht_2(mht_2_v, 251, "", "./tensorflow/c/experimental/grappler/grappler_internal.h", "UsesFunctionLibrary");
 return false; }
  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgrapplerPSgrappler_internalDTh mht_3(mht_3_v, 256, "", "./tensorflow/c/experimental/grappler/grappler_internal.h", "Init");

    return Status::OK();
  }
  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph_def) override;

  ~CGraphOptimizer() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgrapplerPSgrappler_internalDTh mht_4(mht_4_v, 265, "", "./tensorflow/c/experimental/grappler/grappler_internal.h", "~CGraphOptimizer");

    if (optimizer_.destroy_func != nullptr) {
      (*optimizer_.destroy_func)(c_optimizer_);
    }
  }

 private:
  TP_Optimizer optimizer_;
  std::string device_type_;
  void* c_optimizer_;
};

// Registration function to register a CGraphOptimizer along with plugin configs
// and device type.
void CGraphOptimizerRegister(
    const PluginGraphOptimizerRegistry::Creator& creator,
    const TP_OptimizerConfigs tp_configs, const char* device_type);

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_GRAPPLER_GRAPPLER_INTERNAL_H_
