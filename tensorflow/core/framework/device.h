/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// A Device is a something that can perform computations as part of a
// model.  Devices can be local (runs computation on this machine), or
// remote (contacts a device local to another machine using an RPC to
// do the work).  Devices are registered in a DeviceSet, which is also
// responsible for the Device <-> id mapping.
//
// Device names
// * Every Device should have a unique name with the format:
//     /job:___/replica:___/task:___/(gpu|cpu):___
//   An example name would be "/job:train/replica:0/task:3/device:GPU:2".
// * Task numbers are within the specified replica, so there are as
//   many "task zeros" as replicas.

#ifndef TENSORFLOW_CORE_FRAMEWORK_DEVICE_H_
#define TENSORFLOW_CORE_FRAMEWORK_DEVICE_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh() {
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
#include <string>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

class Device : public DeviceBase {
 public:
  // Callback type that takes a Status and returns void.
  typedef std::function<void(const Status&)> DoneCallback;

  Device(Env* env, const DeviceAttributes& device_attributes);
  ~Device() override;

  // Full name of this device (see top comment).
  const std::string& name() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_0(mht_0_v, 232, "", "./tensorflow/core/framework/device.h", "name");
 return device_attributes_.name(); }

  // Parsed name of this device
  const DeviceNameUtils::ParsedName& parsed_name() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_1(mht_1_v, 238, "", "./tensorflow/core/framework/device.h", "parsed_name");

    return parsed_name_;
  }

  // Describes what kind of device this is.  This is intended to be
  // human-readable and not computer-parsed, except that two devices
  // with the same device_type() are expected to perform similarly
  // (both from a computation and communication perspective).
  const std::string& device_type() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_2(mht_2_v, 249, "", "./tensorflow/core/framework/device.h", "device_type");

    return device_attributes_.device_type();
  }

  // Returns an aggregation of device attributes.
  const DeviceAttributes& attributes() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_3(mht_3_v, 257, "", "./tensorflow/core/framework/device.h", "attributes");

    return device_attributes_;
  }

  // Performs the actual compute function.
  //
  // Subclasses may override this function if they wish to perform
  // some initialization before each compute.
  virtual void Compute(OpKernel* op_kernel, OpKernelContext* context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_4(mht_4_v, 268, "", "./tensorflow/core/framework/device.h", "Compute");

    op_kernel->Compute(context);
  }

  // Asynchronous kernel's compute.
  virtual void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                            AsyncOpKernel::DoneCallback done) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_5(mht_5_v, 277, "", "./tensorflow/core/framework/device.h", "ComputeAsync");

    op_kernel->ComputeAsync(context, std::move(done));
  }

  // Blocks until all operations queued on the device at the time of
  // the call have completed.  Returns any error pending on the device
  // at completion.
  virtual Status Sync() = 0;

  // Calls the given callback when all operations queued on the device at the
  // time of the call have completed. The callback is passed any error pending
  // on the device at completion.
  // TODO(b/112409994): Consolidate these two APIs, removing the synchronous
  // version.
  virtual void Sync(const DoneCallback& done);

  // On session completion, the executor may call Device::Sync() depending on
  // flag settings. Override this to return false for devices that don't allow
  // such calls. Instead, these devices must use other mechanisms (such as
  // num_deferred_ops) to ensure the device has finished processing necessary
  // work at session completion. In addition, for these devices, RefreshStatus
  // must be called at session completion to retrieve execution result status.
  //
  // Devices that override this function must also implement RefreshStatus.
  virtual bool AllowsSyncOnCompletion() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_6(mht_6_v, 304, "", "./tensorflow/core/framework/device.h", "AllowsSyncOnCompletion");
 return true; }

  // This is used in conjunction with AllowsSyncOnCompletion to allow the
  // executor to get execution result status at session completion.
  //
  // For supported devices, this call returns the underlying device stream's
  // current status in a non-blocking way, without using blocking calls such as
  // Stream::BlockHostUntilDone or Device::Sync. When applicable, the device
  // status is also updated with the retrieved stream status.
  virtual Status RefreshStatus() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_7(mht_7_v, 316, "", "./tensorflow/core/framework/device.h", "RefreshStatus");

    return errors::Unimplemented(
        "RefreshStatus is not supported on this device.");
  }

  // Optionally modify the device's GraphDef before execution.
  //
  // This method should be considered experimental and is supplied to enable
  // prototyping of TensorFlow device implementations that need to modify
  // the GraphDef before execution.
  //
  // 'graph' supplies the partition of the graph assigned to this
  // device.
  virtual Status MaybeRewriteGraph(std::unique_ptr<Graph>* /*graph*/) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_8(mht_8_v, 332, "", "./tensorflow/core/framework/device.h", "MaybeRewriteGraph");

    return Status::OK();
  }

  // Sets `out_context` a new DeviceContext* for executing a graph, or nullptr
  // if the device does not support contexts. Returns an error status if any
  // error occurred while trying to create a context, otherwise OK.
  //
  // The caller takes ownership of one reference on the output DeviceContext*,
  // and should call Unref().
  virtual Status TryGetDeviceContext(DeviceContext** out_context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_9(mht_9_v, 345, "", "./tensorflow/core/framework/device.h", "TryGetDeviceContext");

    *out_context = nullptr;
    return Status::OK();
  }

  // Returns the op segment of this device.  The caller can reuse op
  // kernels registered for the same session running on this device.
  OpSegment* op_segment() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_10(mht_10_v, 355, "", "./tensorflow/core/framework/device.h", "op_segment");
 return &op_seg_; }

  // Returns the resource manager associated w/ this device.
  virtual ResourceMgr* resource_manager() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_11(mht_11_v, 361, "", "./tensorflow/core/framework/device.h", "resource_manager");
 return rmgr_; }

  // Summarizes the status of this Device, for debugging.
  std::string DebugString() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_12(mht_12_v, 367, "", "./tensorflow/core/framework/device.h", "DebugString");
 return device_attributes_.DebugString(); }

  // Assembles the parameter components into a complete DeviceAttributes value.
  static DeviceAttributes BuildDeviceAttributes(
      const std::string& name, DeviceType device, Bytes memory_limit,
      const DeviceLocality& locality, const std::string& physical_device_desc);

  static DeviceAttributes BuildDeviceAttributes(
      const std::string& name, DeviceType device, Bytes memory_limit,
      const DeviceLocality& locality) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_13(mht_13_v, 380, "", "./tensorflow/core/framework/device.h", "BuildDeviceAttributes");

    // Pass in an empty string as physical device name.
    return BuildDeviceAttributes(name, device, memory_limit, locality, "");
  }

  // Updates `attributes()`, indicating the XLA global ID associated with this
  // device. This ID is unique across clients in a multi-client setup. For TPUs
  // this does not happen until the TPU system has been initialized.
  void set_xla_global_id(int64_t id) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_14(mht_14_v, 391, "", "./tensorflow/core/framework/device.h", "set_xla_global_id");

    device_attributes_.set_xla_global_id(id);
  }

  // Clears the resource manager associated with this device.
  void ClearResourceMgr() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_15(mht_15_v, 399, "", "./tensorflow/core/framework/device.h", "ClearResourceMgr");
 rmgr_->Clear(); }

  virtual bool IsLocal() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_16(mht_16_v, 404, "", "./tensorflow/core/framework/device.h", "IsLocal");
 return true; }

  // Informs if this Device can be used as a caller in RemoteCall operation.
  virtual bool IsRemoteCallAllowed() const;

 protected:
  void DeleteResourceMgr() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdeviceDTh mht_17(mht_17_v, 413, "", "./tensorflow/core/framework/device.h", "DeleteResourceMgr");

    delete rmgr_;
    rmgr_ = nullptr;
  }

 private:
  DeviceAttributes device_attributes_;
  DeviceNameUtils::ParsedName parsed_name_;

  // op_seg_ maps session handle and op name to OpKernel objects.
  OpSegment op_seg_;

  // Resources associated w/ this device. E.g., shared variables, etc.
  ResourceMgr* rmgr_ = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(Device);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_DEVICE_H_
