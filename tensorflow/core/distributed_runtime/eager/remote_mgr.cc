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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_mgrDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_mgrDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_mgrDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"

#include <memory>

#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace eager {

void RemoteMgr::AddOperationOutputs(
    const gtl::ArraySlice<tensorflow::TensorHandle*> handles,
    int64_t operation_id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_mgrDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/distributed_runtime/eager/remote_mgr.cc", "RemoteMgr::AddOperationOutputs");

  mutex_lock l(remote_tensor_handle_mu_);
  for (int i = 0, end = handles.size(); i < end; i++) {
    // TODO(nareshmodi): Correctly handle operation_id not being unique.
    remote_tensor_handle_map_.emplace(
        RemoteTensorHandleInternal(operation_id, i), handles[i]);
  }
}

void RemoteMgr::AddOperationOutput(tensorflow::TensorHandle* handle,
                                   int64_t operation_id, int32_t output_num) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_mgrDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/distributed_runtime/eager/remote_mgr.cc", "RemoteMgr::AddOperationOutput");

  mutex_lock l(remote_tensor_handle_mu_);
  remote_tensor_handle_map_.emplace(
      RemoteTensorHandleInternal(operation_id, output_num), handle);
}

Status RemoteMgr::GetTensorHandleImpl(
    const RemoteTensorHandleInternal& remote_handle,
    tensorflow::TensorHandle** handle) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_mgrDTcc mht_2(mht_2_v, 222, "", "./tensorflow/core/distributed_runtime/eager/remote_mgr.cc", "RemoteMgr::GetTensorHandleImpl");

  auto iter = remote_tensor_handle_map_.find(remote_handle);
  if (iter == remote_tensor_handle_map_.end()) {
    // TODO(b/217820532): Fix the tensor deallocation order issue.
    return errors::InvalidArgument(
        "Unable to find the relevant tensor remote_handle: Op ID: ",
        remote_handle.op_id, ", Output num: ", remote_handle.output_num,
        ". One possible cause is that the tensor was accessed after "
        "deallocation in a distributed worker setup. Try setting "
        "`os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'` in "
        "your client to disable async streaming behavior to see if it fixes "
        "the problem.");
  }

  *handle = iter->second;

  return Status::OK();
}

Status RemoteMgr::GetTensorHandle(
    const RemoteTensorHandleInternal& remote_handle,
    tensorflow::TensorHandle** handle) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_mgrDTcc mht_3(mht_3_v, 246, "", "./tensorflow/core/distributed_runtime/eager/remote_mgr.cc", "RemoteMgr::GetTensorHandle");

  tf_shared_lock l(remote_tensor_handle_mu_);
  return GetTensorHandleImpl(remote_handle, handle);
}

Status RemoteMgr::GetMirroredResourceShape(
    const RemoteTensorHandleInternal& remote_handle,
    std::vector<DtypeAndPartialTensorShape>* handle) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_mgrDTcc mht_4(mht_4_v, 256, "", "./tensorflow/core/distributed_runtime/eager/remote_mgr.cc", "RemoteMgr::GetMirroredResourceShape");

  tf_shared_lock l(mirrored_resource_shape_mu_);
  auto iter = mirrored_resource_shape_map_.find(remote_handle);
  if (iter == mirrored_resource_shape_map_.end()) {
    // TODO(b/217820532): Fix the tensor deallocation order issue.
    return errors::InvalidArgument(
        "Unable to find the relevant tensor remote_handle: Op ID: ",
        remote_handle.op_id, ", Output num: ", remote_handle.output_num,
        ". One possible cause is that the tensor was accessed after "
        "deallocation in a distributed worker setup. Try setting "
        "`os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'` in "
        "your client to disable async streaming behavior to see if it fixes "
        "the problem.");
  }

  *handle = iter->second;

  return Status::OK();
}

Status RemoteMgr::GetRemoteTensorHandle(const tensorflow::TensorHandle* handle,
                                        const bool wait_until_ready,
                                        int64_t* op_id, int32* output_num) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_mgrDTcc mht_5(mht_5_v, 281, "", "./tensorflow/core/distributed_runtime/eager/remote_mgr.cc", "RemoteMgr::GetRemoteTensorHandle");

  TF_RETURN_IF_ERROR(handle->RemoteAddress(handle->device(), wait_until_ready,
                                           op_id, output_num));
  tensorflow::TensorHandle* h;
  TF_RETURN_IF_ERROR(
      GetTensorHandleImpl(RemoteTensorHandleInternal(*op_id, *output_num), &h));
  if (handle != h) {
    return errors::Internal(
        "Found two different tensor handles with the same op_id:", *op_id,
        " and output_num:", *output_num);
  }
  return Status::OK();
}

Status RemoteMgr::DeleteTensorHandle(
    const RemoteTensorHandleInternal& remote_handle) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_mgrDTcc mht_6(mht_6_v, 299, "", "./tensorflow/core/distributed_runtime/eager/remote_mgr.cc", "RemoteMgr::DeleteTensorHandle");

  {
    mutex_lock l(remote_tensor_handle_mu_);
    auto iter = remote_tensor_handle_map_.find(remote_handle);
    if (iter != remote_tensor_handle_map_.end()) {
      iter->second->Unref();
      remote_tensor_handle_map_.erase(iter);
      return Status::OK();
    }
  }
  {
    mutex_lock l(mirrored_resource_shape_mu_);
    auto iter = mirrored_resource_shape_map_.find(remote_handle);
    if (iter != mirrored_resource_shape_map_.end()) {
      mirrored_resource_shape_map_.erase(iter);
      return Status::OK();
    }
  }
  return errors::InvalidArgument(
      "Unable to find the relevant tensor remote_handle: Op ID: ",
      remote_handle.op_id, ", Output num: ", remote_handle.output_num);
}

Status RemoteMgr::SerializeRemoteTensorHandle(
    TensorHandle* in, const bool wait_until_ready, RemoteTensorHandle* out,
    Device* device, const string& device_name,
    const bool serialize_resource_dtype_and_shape) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_mgrDTcc mht_7(mht_7_v, 329, "", "./tensorflow/core/distributed_runtime/eager/remote_mgr.cc", "RemoteMgr::SerializeRemoteTensorHandle");

  int64_t op_id;
  int32_t output_num;
  if (!in->RemoteAddress(device, wait_until_ready, &op_id, &output_num).ok()) {
    tf_shared_lock l(remote_tensor_handle_mu_);
    TF_RETURN_IF_ERROR(
        GetRemoteTensorHandle(in, wait_until_ready, &op_id, &output_num));
  }
  out->Clear();
  out->set_op_id(op_id);
  out->set_output_num(output_num);
  out->set_op_device(in->op_device() ? in->op_device()->name() : "");
  out->set_device(device_name);
  out->set_dtype(in->dtype);
  if (serialize_resource_dtype_and_shape) {
    std::vector<DtypeAndPartialTensorShape> resource_dtypes_and_shapes;
    TF_RETURN_IF_ERROR(
        in->GetResourceHandleDtypesAndShapes(&resource_dtypes_and_shapes));
    for (const auto& dtype_and_shape : resource_dtypes_and_shapes) {
      ResourceDtypeAndShape* dtype_and_shape_proto =
          out->add_resource_dtypes_and_shapes();
      dtype_and_shape_proto->set_dtype(dtype_and_shape.dtype);
      dtype_and_shape.shape.AsProto(dtype_and_shape_proto->mutable_shape());
    }
  }
  return Status::OK();
}

Status RemoteMgr::DeserializeRemoteTensorHandle(const RemoteTensorHandle& in,
                                                TensorHandle** out) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_mgrDTcc mht_8(mht_8_v, 361, "", "./tensorflow/core/distributed_runtime/eager/remote_mgr.cc", "RemoteMgr::DeserializeRemoteTensorHandle");

  Device* device;
  if (parent_->local_device_mgr()->LookupDevice(in.op_device(), &device).ok() ||
      parent_->local_device_mgr()->LookupDevice(in.device(), &device).ok()) {
    TF_RETURN_IF_ERROR(GetTensorHandle(RemoteTensorHandleInternal(in), out));
    (*out)->Ref();
  } else {
    // Create a remote TensorHandle for remote tensors which have not been
    // copied to the local worker yet (e.g. remote function inputs).
    const string& device_name =
        in.op_device().empty() ? in.device() : in.op_device();
    TF_RETURN_IF_ERROR(
        parent_->FindDeviceFromName(device_name.c_str(), &device));
    *out = TensorHandle::CreateLazyRemoteHandle(in.op_id(), in.output_num(),
                                                in.dtype(), device,
                                                /*is_ready=*/true, parent_);
    std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes;
    if (!GetMirroredResourceShape(RemoteTensorHandleInternal(in),
                                  &dtypes_and_shapes)
             .ok()) {
      for (const auto& dtype_and_shape_proto :
           in.resource_dtypes_and_shapes()) {
        dtypes_and_shapes.push_back(DtypeAndPartialTensorShape{
            dtype_and_shape_proto.dtype(),
            TensorShape(dtype_and_shape_proto.shape())});
      }
      mutex_lock l(mirrored_resource_shape_mu_);
      mirrored_resource_shape_map_.emplace(
          RemoteTensorHandleInternal(in.op_id(), in.output_num()),
          dtypes_and_shapes);
    }
    (*out)->SetResourceHandleDtypeAndShape(std::move(dtypes_and_shapes));
  }

  return Status::OK();
}

EagerExecutor& RemoteMgr::GetOrCreateExecutorForStream(uint64 stream_id) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_mgrDTcc mht_9(mht_9_v, 401, "", "./tensorflow/core/distributed_runtime/eager/remote_mgr.cc", "RemoteMgr::GetOrCreateExecutorForStream");

  mutex_lock l(executor_map_mu_);
  auto it = executor_map_.find(stream_id);
  if (it == executor_map_.end()) {
    auto it_and_bool = executor_map_.emplace(
        std::piecewise_construct, std::forward_as_tuple(stream_id),
        std::forward_as_tuple(/*async=*/true));
    DCHECK(it_and_bool.second);
    it = it_and_bool.first;
  }
  return it->second;
}

void RemoteMgr::DeleteExecutorForStream(uint64 stream_id) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_mgrDTcc mht_10(mht_10_v, 417, "", "./tensorflow/core/distributed_runtime/eager/remote_mgr.cc", "RemoteMgr::DeleteExecutorForStream");

  mutex_lock l(executor_map_mu_);
  auto it = executor_map_.find(stream_id);
  if (it == executor_map_.end()) {
    return;
  }
  Status s = it->second.ShutDown();
  if (!s.ok()) {
    LOG(ERROR) << "EagerExecutor shutdown with error " << s.error_message();
  }
  executor_map_.erase(it);
}

}  // namespace eager
}  // namespace tensorflow
