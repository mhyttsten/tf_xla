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
class MHTracer_DTPStensorflowPScompilerPSjitPSxla_cpu_deviceDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cpu_deviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSxla_cpu_deviceDTcc() {
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

// Registers the XLA_CPU device, which is an XlaDevice instantiation that runs
// operators using XLA via the XLA "Host" (CPU) backend.

#include "absl/memory/memory.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/jit/xla_compile_on_demand_op.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
using tensorflow::IdentityShapeRepresentationFn;

class XlaCpuDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override;
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;
};

Status XlaCpuDeviceFactory::ListPhysicalDevices(std::vector<string>* devices) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cpu_deviceDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/jit/xla_cpu_device.cc", "XlaCpuDeviceFactory::ListPhysicalDevices");

  XlaDeviceFlags* flags = GetXlaDeviceFlags();
  if (!flags->tf_xla_enable_xla_devices && !XlaDevicesCreationRequired()) {
    VLOG(1) << "Not creating XLA devices, tf_xla_enable_xla_devices not set "
               "and XLA device creation not requested";
    return Status::OK();
  }

  devices->push_back(absl::StrCat("/physical_device:", DEVICE_XLA_CPU, ":0"));
  return Status::OK();
}

Status XlaCpuDeviceFactory::CreateDevices(
    const SessionOptions& session_options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name_prefix: \"" + name_prefix + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_cpu_deviceDTcc mht_1(mht_1_v, 228, "", "./tensorflow/compiler/jit/xla_cpu_device.cc", "XlaCpuDeviceFactory::CreateDevices");

  XlaDeviceFlags* flags = GetXlaDeviceFlags();
  if (!flags->tf_xla_enable_xla_devices && !XlaDevicesCreationRequired()) {
    VLOG(1) << "Not creating XLA devices, tf_xla_enable_xla_devices not set";
    return Status::OK();
  }
  bool compile_on_demand = flags->tf_xla_compile_on_demand;

  XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = DEVICE_CPU_XLA_JIT;
  registration.autoclustering_policy =
      compile_on_demand
          ? XlaOpRegistry::AutoclusteringPolicy::kIfExplicitlyRequested
          : XlaOpRegistry::AutoclusteringPolicy::kAlways;
  registration.cluster_resource_variable_ops_unsafely = true;
  registration.cluster_stack_ops = false;
  registration.cluster_tensor_array_ops = true;
  registration.cluster_stateful_rng_ops = true;
  registration.cluster_control_trigger = true;
  registration.elide_assert_and_checknumerics = true;
  registration.cluster_variant_ops = true;
  registration.cluster_slow_ops = true;
  registration.cluster_inaccurate_ops = true;
  XlaOpRegistry::RegisterCompilationDevice(DEVICE_XLA_CPU, registration);

  static XlaDeviceOpRegistrations* registrations =
      RegisterXlaDeviceKernels(DEVICE_XLA_CPU, DEVICE_CPU_XLA_JIT);
  (void)registrations;

  TF_ASSIGN_OR_RETURN(auto platform,
                      se::MultiPlatformManager::PlatformWithName("Host"));

  XlaDevice::Options options;
  options.platform = platform;
  options.device_name_prefix = name_prefix;
  options.device_name = DEVICE_XLA_CPU;
  options.device_ordinal = 0;
  options.compilation_device_name = DEVICE_CPU_XLA_JIT;
  options.use_multiple_streams = false;
  XlaShapeLayoutHelpers::ShapeDeterminationFns shape_representation_fns{
      UseNoPreferenceLayoutFn(), IdentityShapeRepresentationFn()};
  options.shape_determination_fns = {shape_representation_fns};
  auto device = absl::make_unique<XlaDevice>(session_options, options);

  // Setting GpuDeviceInfo because eager runtime relies on the device
  // context in tensorflow_gpu_device_info(). Also,
  // tensorflow_gpu_device_info() == nullptr is used as an IsCPU test.
  // We need XlaCpuDevice to be treated not as CPU because it allocates
  // XlaTensors, not regular Tensors.
  Status status = device->UseGpuDeviceInfo();
  if (!status.ok()) {
    errors::AppendToMessage(&status, "while setting up ", DEVICE_GPU_XLA_JIT);
    return status;
  }
  devices->push_back(std::move(device));
  return Status::OK();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_CPU, XlaCpuDeviceFactory);

// Kernel registrations

constexpr std::array<DataType, 16> kAllXlaCpuTypes = {
    {DT_UINT8, DT_QUINT8, DT_UINT16, DT_INT8, DT_QINT8, DT_INT16, DT_INT32,
     DT_QINT32, DT_INT64, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
     DT_COMPLEX128, DT_BOOL, DT_BFLOAT16}};

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_CPU, XlaLocalLaunchOp, kAllXlaCpuTypes);
REGISTER_XLA_COMPILE_KERNEL(DEVICE_XLA_CPU, XlaCompileOp, kAllXlaCpuTypes);
REGISTER_XLA_RUN_KERNEL(DEVICE_XLA_CPU, XlaRunOp, kAllXlaCpuTypes);

REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_CPU, kAllXlaCpuTypes);

}  // namespace tensorflow
