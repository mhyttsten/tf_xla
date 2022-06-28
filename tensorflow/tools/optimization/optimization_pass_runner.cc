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
class MHTracer_DTPStensorflowPStoolsPSoptimizationPSoptimization_pass_runnerDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSoptimizationPSoptimization_pass_runnerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSoptimizationPSoptimization_pass_runnerDTcc() {
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
// This file creates a library that can run any registered optimization pass.
// The binary that uses this will be run in a form similar to:
// ./optimization_pass_runner  --input_file_path=/tmp/input.pbtxt
// --output_file_path=/tmp/output.pbtxt
// --optimization_pass=NameOfGraphOptimizationPass
#include "tensorflow/tools/optimization/optimization_pass_runner.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {
// A fake device used to populate a DeviceSet.
class FakeDevice : public Device {
 private:
  explicit FakeDevice(const DeviceAttributes& device_attributes)
      : Device(nullptr, device_attributes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSoptimizationPSoptimization_pass_runnerDTcc mht_0(mht_0_v, 219, "", "./tensorflow/tools/optimization/optimization_pass_runner.cc", "FakeDevice");
}

 public:
  Status Sync() override;
  static std::unique_ptr<Device> Make(const string& name, const string& type);
};

Status FakeDevice::Sync() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSoptimizationPSoptimization_pass_runnerDTcc mht_1(mht_1_v, 229, "", "./tensorflow/tools/optimization/optimization_pass_runner.cc", "FakeDevice::Sync");

  return errors::Unimplemented("FakeDevice::Sync()");
}

std::unique_ptr<Device> FakeDevice::Make(const string& name,
                                         const string& type) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   mht_2_v.push_back("type: \"" + type + "\"");
   MHTracer_DTPStensorflowPStoolsPSoptimizationPSoptimization_pass_runnerDTcc mht_2(mht_2_v, 239, "", "./tensorflow/tools/optimization/optimization_pass_runner.cc", "FakeDevice::Make");

  DeviceAttributes device_attributes;
  device_attributes.set_name(name);
  device_attributes.set_device_type(DeviceType(type).type());
  return std::unique_ptr<Device>(new FakeDevice(device_attributes));
}

Status FindPassWithName(absl::string_view name,
                        GraphOptimizationPass** result) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPStoolsPSoptimizationPSoptimization_pass_runnerDTcc mht_3(mht_3_v, 251, "", "./tensorflow/tools/optimization/optimization_pass_runner.cc", "FindPassWithName");

  *result = nullptr;
  // Run the optimization pass specified by the command line flag.
  for (const auto& groups_and_passes :
       OptimizationPassRegistry::Global()->groups()) {
    for (const auto& phase_and_passes : groups_and_passes.second) {
      for (const auto& pass : phase_and_passes.second) {
        if (pass->name() == name) {
          if (*result) {
            return errors::Internal("Found more than one pass with name ",
                                    name);
          }
          *result = pass.get();
        }
      }
    }
  }

  return *result == nullptr
             ? errors::Internal("Could not find pass with name ", name)
             : Status::OK();
}
}  // namespace

Status OptimizationPassRunner::Run(absl::string_view pass_to_run,
                                   GraphDef input, GraphDef* result) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("pass_to_run: \"" + std::string(pass_to_run.data(), pass_to_run.size()) + "\"");
   MHTracer_DTPStensorflowPStoolsPSoptimizationPSoptimization_pass_runnerDTcc mht_4(mht_4_v, 280, "", "./tensorflow/tools/optimization/optimization_pass_runner.cc", "OptimizationPassRunner::Run");

  auto session_options = absl::make_unique<SessionOptions>();
  session_options->config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(jit_level_);
  FunctionDefLibrary flib;
  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());

  GraphOptimizationPassOptions options;
  options.session_options = session_options.get();
  options.graph = &graph;
  std::unique_ptr<FunctionLibraryDefinition> flib_def(
      new FunctionLibraryDefinition((*options.graph)->op_registry(), flib));
  options.flib_def = flib_def.get();

  // Grab the data
  GraphConstructorOptions graph_opts;
  graph_opts.expect_device_spec = true;
  graph_opts.allow_internal_ops = true;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(graph_opts, std::move(input),
                                            options.graph->get()));

  // Add all devices that were previously configured with AddDevice.
  DeviceSet device_set;
  for (auto& device : devices_) {
    device_set.AddDevice(device.get());
  }
  options.device_set = &device_set;

  GraphOptimizationPass* pass;
  TF_RETURN_IF_ERROR(FindPassWithName(pass_to_run, &pass));
  TF_RETURN_IF_ERROR(pass->Run(options));

  options.graph->get()->ToGraphDef(result);
  return Status::OK();
}

Status OptimizationPassRunner::SetJitLevel(
    OptimizerOptions::GlobalJitLevel jit_level) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSoptimizationPSoptimization_pass_runnerDTcc mht_5(mht_5_v, 321, "", "./tensorflow/tools/optimization/optimization_pass_runner.cc", "OptimizationPassRunner::SetJitLevel");

  jit_level_ = jit_level;
  return Status::OK();
}

Status OptimizationPassRunner::AddDevices(absl::string_view type, int count) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("type: \"" + std::string(type.data(), type.size()) + "\"");
   MHTracer_DTPStensorflowPStoolsPSoptimizationPSoptimization_pass_runnerDTcc mht_6(mht_6_v, 330, "", "./tensorflow/tools/optimization/optimization_pass_runner.cc", "OptimizationPassRunner::AddDevices");

  for (int i = 0; i < count; i++) {
    devices_.push_back(FakeDevice::Make(
        absl::StrCat("/job:localhost/replica:0/task:0/device:", type, ":", i),
        absl::StrCat(type)));
    devices_.push_back(FakeDevice::Make(
        absl::StrCat("/job:localhost/replica:0/task:0/device:XLA_", type, ":",
                     i),
        absl::StrCat(type)));
  }

  return Status::OK();
}
}  // namespace tensorflow
