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
class MHTracer_DTPStensorflowPSpythonPSgrapplerPStf_optimizer_wrapperDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSgrapplerPStf_optimizer_wrapperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSgrapplerPStf_optimizer_wrapperDTcc() {
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

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "pybind11/pybind11.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/clusters/utils.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;

void DetectDevices(
    std::unordered_map<std::string, tensorflow::DeviceProperties>* device_map) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSgrapplerPStf_optimizer_wrapperDTcc mht_0(mht_0_v, 211, "", "./tensorflow/python/grappler/tf_optimizer_wrapper.cc", "DetectDevices");

  tensorflow::SessionOptions options;
  std::vector<std::unique_ptr<tensorflow::Device>> devices;
  if (!tensorflow::DeviceFactory::AddDevices(options, "", &devices).ok()) {
    return;
  }

  for (const std::unique_ptr<tensorflow::Device>& device : devices) {
    tensorflow::DeviceProperties& prop = (*device_map)[device->name()];
    prop = tensorflow::grappler::GetDeviceInfo(device->parsed_name());

    // Overwrite the memory limit since users might have requested to use only a
    // fraction of the available device memory.
    const tensorflow::DeviceAttributes& attr = device->attributes();
    prop.set_memory_size(attr.memory_limit());
  }
}

PYBIND11_MODULE(_pywrap_tf_optimizer, m) {
  m.def("TF_OptimizeGraph",
        [](tensorflow::grappler::Cluster* cluster,
           const std::string& serialized_config_proto,
           const std::string& serialized_metagraph, bool verbose,
           const std::string& graph_id,
           bool strip_default_attributes) -> py::bytes {
          std::string out_graph_bytes;
          {
            py::gil_scoped_release gil_release;
            tensorflow::ConfigProto config_proto;
            if (!config_proto.ParseFromString(serialized_config_proto)) {
              throw std::invalid_argument(
                  "The ConfigProto could not be parsed as a valid protocol "
                  "buffer");
            }
            tensorflow::MetaGraphDef metagraph;
            if (!metagraph.ParseFromString(serialized_metagraph)) {
              throw std::invalid_argument(
                  "The MetaGraphDef could not be parsed as a valid protocol "
                  "buffer");
            }

            tensorflow::grappler::ItemConfig item_config;
            // This disables graph optimizations in the older graph optimizer,
            // which tend to overlap / be redundant with those in Grappler.
            item_config.apply_optimizations = false;
            item_config.ignore_user_placement = false;
            std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
                tensorflow::grappler::GrapplerItemFromMetaGraphDef(
                    graph_id, metagraph, item_config);
            if (!grappler_item) {
              throw std::invalid_argument(
                  "Failed to import metagraph, check error log for more info.");
            }

            tensorflow::DeviceBase* cpu_device = nullptr;
            tensorflow::GraphDef out_graph;
            tensorflow::grappler::MetaOptimizer optimizer(cpu_device,
                                                          config_proto);

            MaybeRaiseRegisteredFromStatusWithGIL(
                optimizer.Optimize(cluster, *grappler_item, &out_graph));
            if (strip_default_attributes) {
              tensorflow::StripDefaultAttributes(
                  *tensorflow::OpRegistry::Global(), out_graph.mutable_node());
            }
            if (verbose) {
              optimizer.PrintResult();
            }
            out_graph_bytes = out_graph.SerializeAsString();
          }
          return py::bytes(std::move(out_graph_bytes));
        });
}
