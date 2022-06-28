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
class MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTcc() {
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

#include "tensorflow/compiler/jit/xla_tensor.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace tensorflow {

/*static*/ XlaTensor* XlaTensor::FromTensor(const Tensor* tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTcc mht_0(mht_0_v, 192, "", "./tensorflow/compiler/jit/xla_tensor.cc", "XlaTensor::FromTensor");

  if (tensor->NumElements() == 0) {
    return nullptr;
  }
  XlaTensor* xla_tensor =
      FromOpaquePointer(const_cast<char*>(tensor->tensor_data().data()));
  return xla_tensor;
}

/*static*/ se::DeviceMemoryBase XlaTensor::DeviceMemoryFromTensor(
    const Tensor& tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTcc mht_1(mht_1_v, 205, "", "./tensorflow/compiler/jit/xla_tensor.cc", "XlaTensor::DeviceMemoryFromTensor");

  const XlaTensor* xla_tensor = FromTensor(&tensor);
  if (xla_tensor) {
    CHECK(xla_tensor->has_shaped_buffer());
    return xla_tensor->shaped_buffer().root_buffer();
  } else {
    return se::DeviceMemoryBase(const_cast<char*>(tensor.tensor_data().data()),
                                tensor.tensor_data().size());
  }
}

Status XlaTensor::AllocateShapedBuffer(DataType dtype,
                                       const xla::Shape& on_device_shape,
                                       xla::LocalClient* client,
                                       int device_ordinal) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTcc mht_2(mht_2_v, 222, "", "./tensorflow/compiler/jit/xla_tensor.cc", "XlaTensor::AllocateShapedBuffer");

  xla::Shape on_host_shape =
      xla::ShapeUtil::DeviceShapeToHostShape(on_device_shape);
  xla::ScopedShapedBuffer shaped_buffer(on_host_shape, on_device_shape,
                                        client->backend().memory_allocator(),
                                        device_ordinal);
  for (auto& index_to_buffer : shaped_buffer.buffers()) {
    xla::Shape subshape =
        xla::ShapeUtil::GetSubshape(on_device_shape, index_to_buffer.first);
    uint64 size =
        client->backend().transfer_manager()->GetByteSizeRequirement(subshape);
    TF_ASSIGN_OR_RETURN(se::OwningDeviceMemory buffer,
                        client->backend().memory_allocator()->Allocate(
                            device_ordinal, size, /*retry_on_failure=*/false,
                            subshape.layout().memory_space()));
    // Move our buffer into shaped_buffer, which takes ownership of it.
    index_to_buffer.second = buffer.Release();
  }

  VLOG(4) << shaped_buffer.ToString();

  set_shaped_buffer(std::move(shaped_buffer));
  return Status::OK();
}

void XlaTensor::WaitForDefinitionEventOnStream(se::Stream* stream) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTcc mht_3(mht_3_v, 250, "", "./tensorflow/compiler/jit/xla_tensor.cc", "XlaTensor::WaitForDefinitionEventOnStream");

  mutex_lock lock(mu_);
  if (!definition_event_) {
    return;
  }

  // The set of defined streams is expected to be very small indeed (usually
  // 1-2), so a simple linear scan should be fast enough.
  if (std::find(streams_defined_on_.begin(), streams_defined_on_.end(),
                stream) != streams_defined_on_.end()) {
    // stream is in streams_defined_on_; it doesn't need to be waited on.
    return;
  }

  stream->ThenWaitFor(definition_event_.get());
  streams_defined_on_.push_back(stream);
}

void XlaTensor::ResetDefinitionEvent(std::shared_ptr<se::Event> event,
                                     se::Stream* stream) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTcc mht_4(mht_4_v, 272, "", "./tensorflow/compiler/jit/xla_tensor.cc", "XlaTensor::ResetDefinitionEvent");

  mutex_lock lock(mu_);
  definition_event_ = std::move(event);
  streams_defined_on_ = {stream};
}

Status XlaTensor::RefreshStatusOfStreams() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTcc mht_5(mht_5_v, 281, "", "./tensorflow/compiler/jit/xla_tensor.cc", "XlaTensor::RefreshStatusOfStreams");

  mutex_lock lock(mu_);
  Status status;
  for (se::Stream* stream : streams_defined_on_) {
    status.Update(stream->RefreshStatus());
  }
  return status;
}

// The pointer tag, OR-ed into the XlaTensor's address to distinguish it from
// device-side tensors, which are either CPU or GPU memory pointers. This works
// because we're guaranteed that CPU and GPU pointers are aligned to > 1 bits.
namespace {
constexpr uintptr_t kTag = 0x1ULL;
}

/*static*/ XlaTensor* XlaTensor::FromOpaquePointer(void* ptr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTcc mht_6(mht_6_v, 300, "", "./tensorflow/compiler/jit/xla_tensor.cc", "XlaTensor::FromOpaquePointer");

  uintptr_t value = reinterpret_cast<uintptr_t>(ptr);
  if (value & kTag) {
    return reinterpret_cast<XlaTensor*>(value & ~kTag);
  } else {
    return nullptr;
  }
}

/*static*/ void* XlaTensor::ToOpaquePointer(XlaTensor* tensor) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTcc mht_7(mht_7_v, 312, "", "./tensorflow/compiler/jit/xla_tensor.cc", "XlaTensor::ToOpaquePointer");

  uintptr_t value = reinterpret_cast<uintptr_t>(tensor);
  CHECK_EQ(value & kTag, 0);
  value |= kTag;
  return reinterpret_cast<XlaTensor*>(value);
}

}  // namespace tensorflow
