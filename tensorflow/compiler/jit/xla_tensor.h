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

#ifndef TENSORFLOW_COMPILER_JIT_XLA_TENSOR_H_
#define TENSORFLOW_COMPILER_JIT_XLA_TENSOR_H_
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
class MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTh {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTh() {
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

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// The implementation of a Tensor for an XlaDevice. All device tensors are
// actually one of these.
//
// To distinguish between "normal" device tensors and XlaTensors, the raw
// pointer data stored in the TensorBuffer is a tagged pointer.
class XlaTensor {
 public:
  // Downcast from a Tensor to an XlaTensor. Return nullptr if the downcast
  // fails.
  static XlaTensor* FromTensor(const Tensor* tensor);

  // Create a DeviceMemoryBase from a Tensor. The Tensor can be an XlaTensor, in
  // which case the returned value is shaped_buffer()->root_buffer(), or a
  // normal Tensor in which case the returned value is
  // {tensor.tensor_data().data(), tensor.tensor_data().size}.
  static se::DeviceMemoryBase DeviceMemoryFromTensor(const Tensor& tensor);

  // Assign the internal ShapedBuffer to new memory for the given dtype and
  // shape. If a ShapedBuffer exists already (has_shaped_buffer() == true), it
  // is replaced and the managed memory deallocated.
  Status AllocateShapedBuffer(DataType dtype, const xla::Shape& on_device_shape,
                              xla::LocalClient* client, int device_ordinal);

  // Some Tensors can have complex on-device shapes, including tuple shapes. To
  // manage the memory for these tensors a ShapedBuffer may be required.

  // Return true if this XlaTensor contains a ShapedBuffer.
  bool has_shaped_buffer() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTh mht_0(mht_0_v, 227, "", "./tensorflow/compiler/jit/xla_tensor.h", "has_shaped_buffer");
 return shaped_buffer_.has_value(); }
  // Return the contained ShapedBuffer.
  // REQUIRES: has_shaped_buffer()
  const xla::ShapedBuffer& shaped_buffer() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTh mht_1(mht_1_v, 233, "", "./tensorflow/compiler/jit/xla_tensor.h", "shaped_buffer");

    CHECK(has_shaped_buffer());
    return *shaped_buffer_;
  }
  xla::ShapedBuffer& shaped_buffer() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTh mht_2(mht_2_v, 240, "", "./tensorflow/compiler/jit/xla_tensor.h", "shaped_buffer");

    CHECK(has_shaped_buffer());
    return *shaped_buffer_;
  }
  // Mutates the XlaTensor to set the ShapedBuffer.
  void set_shaped_buffer(xla::ScopedShapedBuffer shaped_buffer) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_tensorDTh mht_3(mht_3_v, 248, "", "./tensorflow/compiler/jit/xla_tensor.h", "set_shaped_buffer");

    shaped_buffer_ = std::move(shaped_buffer);
  }

  // Adds synchronization events to 'stream' that wait for this tensor to be
  // defined on 'stream'. Does nothing if the tensor is already defined on that
  // stream.
  void WaitForDefinitionEventOnStream(se::Stream* stream);

  // (Re)sets the definition event of the tensor to 'event', and promises that
  // the tensor has already been defined on stream. Removes any previous
  // definition event or any previous promises about the tensor being defined on
  // streams.
  // It is legal to reset the definition event of a tensor when overwriting the
  // tensor's value (at which point, it is effectively a new tensor once again.)
  void ResetDefinitionEvent(std::shared_ptr<se::Event> event,
                            se::Stream* stream);

  // Refresh the status of streams_defined_on_. Return the first not-OK stream's
  // status or OK.
  Status RefreshStatusOfStreams();

  // Convert from a raw pointer to an XlaTensor, removing the pointer tag.
  static XlaTensor* FromOpaquePointer(void* ptr);
  // Convert to a raw pointer from an XlaTensor, adding the pointer tag.
  static void* ToOpaquePointer(XlaTensor* tensor);

 private:
  // The optional contained ShapedBuffer.
  absl::optional<xla::ScopedShapedBuffer> shaped_buffer_;
  // An optional host tensor value.
  absl::optional<Tensor> host_tensor_;
  // An optional event that is triggered when the tensor's content has been
  // defined. If this event is nullptr, it is assumed that the tensor's content
  // is always defined.
  std::shared_ptr<se::Event> definition_event_;
  // A list of all streams for which the tensor's content is defined for any
  // newly enqueued command.
  absl::InlinedVector<se::Stream*, 2> streams_defined_on_ TF_GUARDED_BY(mu_);
  mutex mu_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_TENSOR_H_
