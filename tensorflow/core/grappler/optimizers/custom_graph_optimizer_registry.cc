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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTcc() {
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
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"

#include <string>
#include <unordered_map>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace {

typedef std::unordered_map<string, CustomGraphOptimizerRegistry::Creator>
    RegistrationMap;
RegistrationMap* registered_optimizers = nullptr;
RegistrationMap* GetRegistrationMap() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc", "GetRegistrationMap");

  if (registered_optimizers == nullptr)
    registered_optimizers = new RegistrationMap;
  return registered_optimizers;
}

// This map is a global map for registered plugin optimizers. It contains the
// device_type as its key, and an optimizer creator as the value.
typedef std::unordered_map<string, PluginGraphOptimizerRegistry::Creator>
    PluginRegistrationMap;
PluginRegistrationMap* GetPluginRegistrationMap() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc", "GetPluginRegistrationMap");

  static PluginRegistrationMap* registered_plugin_optimizers =
      new PluginRegistrationMap;
  return registered_plugin_optimizers;
}

// This map is a global map for registered plugin configs. It contains the
// device_type as its key, and ConfigList as the value.
typedef std::unordered_map<string, ConfigList> PluginConfigMap;
PluginConfigMap* GetPluginConfigMap() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTcc mht_2(mht_2_v, 223, "", "./tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc", "GetPluginConfigMap");

  static PluginConfigMap* plugin_config_map = new PluginConfigMap;
  return plugin_config_map;
}

// Returns plugin's default configuration for each Grappler optimizer (on/off).
// See tensorflow/core/protobuf/rewriter_config.proto for more details about
// each optimizer.
const ConfigList& DefaultPluginConfigs() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTcc mht_3(mht_3_v, 234, "", "./tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc", "DefaultPluginConfigs");

  static ConfigList* default_plugin_configs =
      new ConfigList(/*disable_model_pruning=*/false,
                     {{"implementation_selector", RewriterConfig::ON},
                      {"function_optimization", RewriterConfig::ON},
                      {"common_subgraph_elimination", RewriterConfig::ON},
                      {"arithmetic_optimization", RewriterConfig::ON},
                      {"debug_stripper", RewriterConfig::ON},
                      {"constant_folding", RewriterConfig::ON},
                      {"shape_optimization", RewriterConfig::ON},
                      {"auto_mixed_precision", RewriterConfig::ON},
                      {"auto_mixed_precision_mkl", RewriterConfig::ON},
                      {"auto_mixed_precision_cpu", RewriterConfig::ON},
                      {"pin_to_host_optimization", RewriterConfig::ON},
                      {"layout_optimizer", RewriterConfig::ON},
                      {"remapping", RewriterConfig::ON},
                      {"loop_optimization", RewriterConfig::ON},
                      {"dependency_optimization", RewriterConfig::ON},
                      {"auto_parallel", RewriterConfig::ON},
                      {"memory_optimization", RewriterConfig::ON},
                      {"scoped_allocator_optimization", RewriterConfig::ON}});
  return *default_plugin_configs;
}

}  // namespace

std::unique_ptr<CustomGraphOptimizer>
CustomGraphOptimizerRegistry::CreateByNameOrNull(const string& name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTcc mht_4(mht_4_v, 265, "", "./tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc", "CustomGraphOptimizerRegistry::CreateByNameOrNull");

  const auto it = GetRegistrationMap()->find(name);
  if (it == GetRegistrationMap()->end()) return nullptr;
  return std::unique_ptr<CustomGraphOptimizer>(it->second());
}

std::vector<string> CustomGraphOptimizerRegistry::GetRegisteredOptimizers() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTcc mht_5(mht_5_v, 274, "", "./tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc", "CustomGraphOptimizerRegistry::GetRegisteredOptimizers");

  std::vector<string> optimizer_names;
  optimizer_names.reserve(GetRegistrationMap()->size());
  for (const auto& opt : *GetRegistrationMap())
    optimizer_names.emplace_back(opt.first);
  return optimizer_names;
}

void CustomGraphOptimizerRegistry::RegisterOptimizerOrDie(
    const Creator& optimizer_creator, const string& name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTcc mht_6(mht_6_v, 287, "", "./tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc", "CustomGraphOptimizerRegistry::RegisterOptimizerOrDie");

  const auto it = GetRegistrationMap()->find(name);
  if (it != GetRegistrationMap()->end()) {
    LOG(FATAL) << "CustomGraphOptimizer is registered twice: " << name;
  }
  GetRegistrationMap()->insert({name, optimizer_creator});
}

std::vector<std::unique_ptr<CustomGraphOptimizer>>
PluginGraphOptimizerRegistry::CreateOptimizers(
    const std::set<string>& device_types) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTcc mht_7(mht_7_v, 300, "", "./tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc", "PluginGraphOptimizerRegistry::CreateOptimizers");

  std::vector<std::unique_ptr<CustomGraphOptimizer>> optimizer_list;
  for (auto it = GetPluginRegistrationMap()->begin();
       it != GetPluginRegistrationMap()->end(); ++it) {
    if (device_types.find(it->first) == device_types.end()) continue;
    LOG(INFO) << "Plugin optimizer for device_type " << it->first
              << " is enabled.";
    optimizer_list.emplace_back(
        std::unique_ptr<CustomGraphOptimizer>(it->second()));
  }
  return optimizer_list;
}

void PluginGraphOptimizerRegistry::RegisterPluginOptimizerOrDie(
    const Creator& optimizer_creator, const std::string& device_type,
    ConfigList& configs) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("device_type: \"" + device_type + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTcc mht_8(mht_8_v, 319, "", "./tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc", "PluginGraphOptimizerRegistry::RegisterPluginOptimizerOrDie");

  auto ret = GetPluginConfigMap()->insert({device_type, configs});
  if (!ret.second) {
    LOG(FATAL) << "PluginGraphOptimizer with device_type "  // Crash OK
               << device_type << " is registered twice.";
  }
  GetPluginRegistrationMap()->insert({device_type, optimizer_creator});
}

void PluginGraphOptimizerRegistry::PrintPluginConfigsIfConflict(
    const std::set<string>& device_types) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTcc mht_9(mht_9_v, 332, "", "./tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc", "PluginGraphOptimizerRegistry::PrintPluginConfigsIfConflict");

  bool init = false, conflict = false;
  ConfigList plugin_configs;
  // Check if plugin's configs have conflict.
  for (const auto& device_type : device_types) {
    const auto it = GetPluginConfigMap()->find(device_type);
    if (it == GetPluginConfigMap()->end()) continue;
    auto cur_plugin_configs = it->second;

    if (!init) {
      plugin_configs = cur_plugin_configs;
      init = true;
    } else {
      if (!(plugin_configs == cur_plugin_configs)) {
        conflict = true;
        break;
      }
    }
  }
  if (!conflict) return;
  LOG(WARNING) << "Plugins have conflicting configs. Potential performance "
                  "regression may happen.";
  for (const auto& device_type : device_types) {
    const auto it = GetPluginConfigMap()->find(device_type);
    if (it == GetPluginConfigMap()->end()) continue;
    auto cur_plugin_configs = it->second;

    // Print logs in following style:
    // disable_model_pruning    0
    // remapping                1
    // ...
    string logs = "";
    strings::StrAppend(&logs, "disable_model_pruning\t\t",
                       cur_plugin_configs.disable_model_pruning, "\n");
    for (auto const& pair : cur_plugin_configs.toggle_config) {
      strings::StrAppend(&logs, pair.first, string(32 - pair.first.size(), ' '),
                         (pair.second != RewriterConfig::OFF), "\n");
    }
    LOG(WARNING) << "Plugin's configs for device_type " << device_type << ":\n"
                 << logs;
  }
}

ConfigList PluginGraphOptimizerRegistry::GetPluginConfigs(
    bool use_plugin_optimizers, const std::set<string>& device_types) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTcc mht_10(mht_10_v, 379, "", "./tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc", "PluginGraphOptimizerRegistry::GetPluginConfigs");

  if (!use_plugin_optimizers) return DefaultPluginConfigs();

  ConfigList ret_plugin_configs = DefaultPluginConfigs();
  for (const auto& device_type : device_types) {
    const auto it = GetPluginConfigMap()->find(device_type);
    if (it == GetPluginConfigMap()->end()) continue;
    auto cur_plugin_configs = it->second;
    // If any of the plugin turns on `disable_model_pruning`,
    // then `disable_model_pruning` should be true;
    if (cur_plugin_configs.disable_model_pruning == true)
      ret_plugin_configs.disable_model_pruning = true;

    // If any of the plugin turns off a certain optimizer,
    // then the optimizer should be turned off;
    for (auto& pair : cur_plugin_configs.toggle_config) {
      if (cur_plugin_configs.toggle_config[pair.first] == RewriterConfig::OFF)
        ret_plugin_configs.toggle_config[pair.first] = RewriterConfig::OFF;
    }
  }

  return ret_plugin_configs;
}

bool PluginGraphOptimizerRegistry::IsConfigsConflict(
    ConfigList& user_config, ConfigList& plugin_config) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPScustom_graph_optimizer_registryDTcc mht_11(mht_11_v, 407, "", "./tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc", "PluginGraphOptimizerRegistry::IsConfigsConflict");

  if (plugin_config == DefaultPluginConfigs()) return false;
  if (user_config.disable_model_pruning != plugin_config.disable_model_pruning)
    return true;
  // Returns true if user_config is turned on but plugin_config is turned off.
  for (auto& pair : user_config.toggle_config) {
    if ((user_config.toggle_config[pair.first] == RewriterConfig::ON) &&
        (plugin_config.toggle_config[pair.first] == RewriterConfig::OFF))
      return true;
  }
  return false;
}

}  // end namespace grappler
}  // end namespace tensorflow
