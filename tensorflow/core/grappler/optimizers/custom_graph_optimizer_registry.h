/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CUSTOM_GRAPH_OPTIMIZER_REGISTRY_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CUSTOM_GRAPH_OPTIMIZER_REGISTRY_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTh() {
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
#include <string>
#include <vector>

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"

namespace tensorflow {
namespace grappler {

// Contains plugin's configurations for each Grappler optimizer (on/off).
// See tensorflow/core/protobuf/rewriter_config.proto for optimizer description.
struct ConfigList {
  ConfigList() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTh mht_0(mht_0_v, 200, "", "./tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h", "ConfigList");
}
  ConfigList(bool disable_model_pruning,
             std::unordered_map<string, RewriterConfig_Toggle> config)
      : disable_model_pruning(disable_model_pruning),
        toggle_config(std::move(config)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTh mht_1(mht_1_v, 207, "", "./tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h", "ConfigList");
}

  bool operator==(const ConfigList& other) const {
    return (disable_model_pruning == other.disable_model_pruning) &&
           (toggle_config == other.toggle_config);
  }
  bool disable_model_pruning;  // Don't remove unnecessary ops from the graph.
  std::unordered_map<string, RewriterConfig_Toggle> toggle_config;
};

class CustomGraphOptimizerRegistry {
 public:
  static std::unique_ptr<CustomGraphOptimizer> CreateByNameOrNull(
      const string& name);

  static std::vector<string> GetRegisteredOptimizers();

  typedef std::function<CustomGraphOptimizer*()> Creator;
  // Register graph optimizer which can be called during program initialization.
  // This class is not thread-safe.
  static void RegisterOptimizerOrDie(const Creator& optimizer_creator,
                                     const string& name);
};

class CustomGraphOptimizerRegistrar {
 public:
  explicit CustomGraphOptimizerRegistrar(
      const CustomGraphOptimizerRegistry::Creator& creator,
      const string& name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTh mht_2(mht_2_v, 239, "", "./tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h", "CustomGraphOptimizerRegistrar");

    CustomGraphOptimizerRegistry::RegisterOptimizerOrDie(creator, name);
  }
};

#define REGISTER_GRAPH_OPTIMIZER_AS(MyCustomGraphOptimizerClass, name) \
  namespace {                                                          \
  static ::tensorflow::grappler::CustomGraphOptimizerRegistrar         \
      MyCustomGraphOptimizerClass##_registrar(                         \
          []() { return new MyCustomGraphOptimizerClass; }, (name));   \
  }  // namespace

#define REGISTER_GRAPH_OPTIMIZER(MyCustomGraphOptimizerClass) \
  REGISTER_GRAPH_OPTIMIZER_AS(MyCustomGraphOptimizerClass,    \
                              #MyCustomGraphOptimizerClass)

// A separate registry to register all plug-in CustomGraphOptimizers.
class PluginGraphOptimizerRegistry {
 public:
  // Constructs a list of plug-in CustomGraphOptimizers from the global map
  // `registered_plugin_optimizers`.
  static std::vector<std::unique_ptr<CustomGraphOptimizer>> CreateOptimizers(
      const std::set<string>& device_types);

  typedef std::function<CustomGraphOptimizer*()> Creator;

  // Returns plugin's config. If any of the config is turned off, the returned
  // config will be turned off.
  static ConfigList GetPluginConfigs(bool use_plugin_optimizers,
                                     const std::set<string>& device_types);

  // Registers plugin graph optimizer which can be called during program
  // initialization. Dies if multiple plugins with the same `device_type` are
  // registered. This class is not thread-safe.
  static void RegisterPluginOptimizerOrDie(const Creator& optimizer_creator,
                                           const std::string& device_type,
                                           ConfigList& configs);

  // Prints plugin's configs if there are some conflicts.
  static void PrintPluginConfigsIfConflict(
      const std::set<string>& device_types);

  // Returns true when `plugin_config` conflicts with `user_config`:
  // - Plugin's `disable_model_pruning` is not equal to `user_config`'s, or
  // - At least one of plugin's `toggle_config`s is on when it is set to off in
  //   `user_config`'s.
  static bool IsConfigsConflict(ConfigList& user_config,
                                ConfigList& plugin_config);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CUSTOM_GRAPH_OPTIMIZER_REGISTRY_H_
