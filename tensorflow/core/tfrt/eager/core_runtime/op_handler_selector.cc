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
class MHTracer_DTPStensorflowPScorePStfrtPSeagerPScore_runtimePSop_handler_selectorDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScore_runtimePSop_handler_selectorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSeagerPScore_runtimePSop_handler_selectorDTcc() {
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

#include "tensorflow/core/tfrt/eager/core_runtime/op_handler_selector.h"

#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/placement_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/tfrt/eager/core_runtime/op_handler_registry.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/device.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

EagerOpHandlerSelector::EagerOpHandlerSelector(CoreRuntime* core_runtime,
                                               EagerContext* eager_context,
                                               OpHandler* fallback_op_handler,
                                               bool pin_small_ops_to_cpu)
    : core_runtime_(core_runtime),
      eager_context_(eager_context),
      cpu_device_(core_runtime->GetHostContext()->GetHostDevice()),
      cpu_op_handler_(core_runtime_->GetOpHandler(cpu_device_.name())),
      fallback_op_handler_(fallback_op_handler),
      pin_small_ops_to_cpu_(pin_small_ops_to_cpu) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScore_runtimePSop_handler_selectorDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/tfrt/eager/core_runtime/op_handler_selector.cc", "EagerOpHandlerSelector::EagerOpHandlerSelector");

  assert(cpu_op_handler_);
  assert(fallback_op_handler_);
}

EagerOpHandlerSelector::~EagerOpHandlerSelector() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScore_runtimePSop_handler_selectorDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/tfrt/eager/core_runtime/op_handler_selector.cc", "EagerOpHandlerSelector::~EagerOpHandlerSelector");
}

Status EagerOpHandlerSelector::SelectFromArguments(
    const ImmediateExecutionOperation& op, OpHandler** op_handler) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScore_runtimePSop_handler_selectorDTcc mht_2(mht_2_v, 224, "", "./tensorflow/core/tfrt/eager/core_runtime/op_handler_selector.cc", "EagerOpHandlerSelector::SelectFromArguments");

  // If the op contains resource handle, place the op on the device of the
  // resource.
  // TODO(tfrt-devs): Unify this logic with MaybePinToResourceDevice in Eager
  // runtime.
  for (int i = 0; i < op.GetInputs().size(); i++) {
    auto& handle = op.GetInputs()[i];
    Status s;
    if (handle->DataType() == tensorflow::DT_RESOURCE) {
      auto device_name = handle->DeviceName(&s);
      TF_RETURN_IF_ERROR(s);
      *op_handler = core_runtime_->GetOpHandler(device_name);
      if (*op_handler != nullptr) {
        DVLOG(1) << "Setting device of operation " << op.Name() << " to "
                 << device_name << " because input #" << i
                 << " is a resource in this device.";
        return Status::OK();
      }
    }
  }

  // Pin the op to cpu op handler if it is a small ops and all its inputs
  // are on cpu already.
  if (pin_small_ops_to_cpu_) {
    bool pin_to_cpu;
    TF_RETURN_IF_ERROR(tensorflow::eager::MaybePinSmallOpsToCpu(
        &pin_to_cpu, op.Name(), op.GetInputs(),
        {cpu_device_.name().data(), cpu_device_.name().size()}));
    if (pin_to_cpu) {
      *op_handler = cpu_op_handler_;
      return Status::OK();
    }
  }

  // Note: The output op_handler is nullptr.
  return Status::OK();
}

Status EagerOpHandlerSelector::SelectFromNodeDef(
    const ImmediateExecutionOperation& op, const NodeDef* ndef,
    OpHandler** op_handler) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScore_runtimePSop_handler_selectorDTcc mht_3(mht_3_v, 267, "", "./tensorflow/core/tfrt/eager/core_runtime/op_handler_selector.cc", "EagerOpHandlerSelector::SelectFromNodeDef");

  const auto& requested_device = op.DeviceName();

  // TODO(fishx): Use TFRT native op registry to select op handler.

  // TODO(fishx): Add a cache for following device placement using current TF.
  // Use EagerContext from current tf to select op handler for this op.
  tensorflow::DeviceNameUtils::ParsedName device_parsed_name;
  if (!tensorflow::DeviceNameUtils::ParseFullName(requested_device,
                                                  &device_parsed_name)) {
    return tensorflow::errors::InvalidArgument("Failed to parse device name: ",
                                               requested_device);
  }

  tensorflow::Device* device;
  TF_RETURN_IF_ERROR(
      eager_context_->SelectDevice(device_parsed_name, *ndef, &device));

  *op_handler = core_runtime_->GetOpHandler(device->name());

  if (!(*op_handler)) *op_handler = fallback_op_handler_;

  return tensorflow::Status::OK();
}

}  // namespace tf
}  // namespace tfrt
