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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilation_deviceDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilation_deviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilation_deviceDTcc() {
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

#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"

#include <functional>
#include <memory>

#include "tensorflow/compiler/tf2xla/frontend_attributes_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/sharding_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {

// The XlaCompilationAllocator doesn't actually back any Tensors with storage
// buffers of values: instead for each Tensor it stores a
// XlaExpression which corresponds to the XLA computation
// represented by the Tensor.
class XlaCompilationAllocator : public Allocator {
 public:
  XlaCompilationAllocator() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilation_deviceDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/tf2xla/xla_compilation_device.cc", "XlaCompilationAllocator");
}
  ~XlaCompilationAllocator() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilation_deviceDTcc mht_1(mht_1_v, 212, "", "./tensorflow/compiler/tf2xla/xla_compilation_device.cc", "~XlaCompilationAllocator");
}

  string Name() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilation_deviceDTcc mht_2(mht_2_v, 217, "", "./tensorflow/compiler/tf2xla/xla_compilation_device.cc", "Name");
 return "xla_compilation"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilation_deviceDTcc mht_3(mht_3_v, 222, "", "./tensorflow/compiler/tf2xla/xla_compilation_device.cc", "AllocateRaw");

    // Regardless of the size requested, always allocates an XlaExpression.
    // Respects the alignment request because there is alignment checking even
    // for Tensors whose data is never accessed.
    void* p = port::AlignedMalloc(sizeof(XlaExpression), alignment);
    XlaExpression* expression = reinterpret_cast<XlaExpression*>(p);
    new (expression) XlaExpression();
    return expression;
  }

  void DeallocateRaw(void* ptr) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilation_deviceDTcc mht_4(mht_4_v, 235, "", "./tensorflow/compiler/tf2xla/xla_compilation_device.cc", "DeallocateRaw");

    XlaExpression* expression = reinterpret_cast<XlaExpression*>(ptr);
    expression->~XlaExpression();
    port::AlignedFree(ptr);
  }

  // Make sure that even tensors with 0 elements have allocated
  // buffers, so they get ids to track.
  //
  // NOTE: It is the caller's responsibility to track whether an allocated
  // object is a buffer or an opaque handle. In particular, when this allocator
  // is used, the caller must not run any constructors or destructors for
  // complex objects, since there is no backing store for the tensor in which to
  // place their outputs.
  bool AllocatesOpaqueHandle() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilation_deviceDTcc mht_5(mht_5_v, 252, "", "./tensorflow/compiler/tf2xla/xla_compilation_device.cc", "AllocatesOpaqueHandle");
 return true; }
};

XlaCompilationDevice::XlaCompilationDevice(const SessionOptions& options,
                                           DeviceType type)
    : LocalDevice(options, Device::BuildDeviceAttributes(
                               absl::StrCat("/device:", type.type(), ":0"),
                               type, Bytes(256 << 20), DeviceLocality(),
                               absl::StrCat("device: XLA compilation device ",
                                            type.type()))),
      allocator_(new XlaCompilationAllocator()) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilation_deviceDTcc mht_6(mht_6_v, 265, "", "./tensorflow/compiler/tf2xla/xla_compilation_device.cc", "XlaCompilationDevice::XlaCompilationDevice");
}

XlaCompilationDevice::~XlaCompilationDevice() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilation_deviceDTcc mht_7(mht_7_v, 270, "", "./tensorflow/compiler/tf2xla/xla_compilation_device.cc", "XlaCompilationDevice::~XlaCompilationDevice");
}

Allocator* XlaCompilationDevice::GetAllocator(AllocatorAttributes attr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilation_deviceDTcc mht_8(mht_8_v, 275, "", "./tensorflow/compiler/tf2xla/xla_compilation_device.cc", "XlaCompilationDevice::GetAllocator");

  return allocator_.get();
}

// Attaches location from the node stack trace to metadata. As a heuristic,
// picks the last frame which does not contain the "tensorflow/python" substring
// (making exception for frames containing "test" to allow for testing the
// feature).
static void AttachLocationToMetadata(xla::OpMetadata& metadata,
                                     OpKernel* op_kernel, XlaContext& context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilation_deviceDTcc mht_9(mht_9_v, 287, "", "./tensorflow/compiler/tf2xla/xla_compilation_device.cc", "AttachLocationToMetadata");

  if (const AbstractStackTrace* stack_trace =
          context.StackTraceForNodeName(op_kernel->def().name())) {
    if (absl::optional<StackFrame> frame = stack_trace->LastUserFrame()) {
      metadata.set_source_file(frame->file_name);
      metadata.set_source_line(frame->line_number);
    }
  }
}

void XlaCompilationDevice::Compute(OpKernel* op_kernel,
                                   OpKernelContext* context) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilation_deviceDTcc mht_10(mht_10_v, 301, "", "./tensorflow/compiler/tf2xla/xla_compilation_device.cc", "XlaCompilationDevice::Compute");

  VLOG(4) << "XlaCompilationDevice::Compute "
          << FormatNodeDefForError(op_kernel->def());
  XlaContext& xla_context = XlaContext::Get(context);
  auto* b = xla_context.builder();
  xla::OpMetadata metadata;
  metadata.set_op_type(op_kernel->type_string());
  metadata.set_op_name(op_kernel->name());
  AttachLocationToMetadata(metadata, op_kernel, xla_context);
  b->SetOpMetadata(metadata);

  auto sharding_parse_result =
      ParseShardingFromDevice(op_kernel->def(), std::numeric_limits<int>::max(),
                              /*add_metadata=*/false);
  OP_REQUIRES_OK(context, sharding_parse_result.status());
  absl::optional<xla::OpSharding> op_sharding =
      sharding_parse_result.ValueOrDie();

  auto frontend_attributes_result =
      GetFrontendAttributesFromAttrSlice(AttrSlice(op_kernel->def()));
  OP_REQUIRES_OK(context, frontend_attributes_result.status());
  absl::optional<xla::FrontendAttributes> attributes =
      frontend_attributes_result.ValueOrDie();

  xla::FrontendAttributes merged_attributes = b->frontend_attributes();
  if (attributes.has_value()) {
    merged_attributes.mutable_map()->insert(attributes.value().map().begin(),
                                            attributes.value().map().end());
  }
  xla::XlaScopedFrontendAttributesAssignment assign_frontend_attributes(
      b, std::move(merged_attributes));

  // If no sharding metadata is found, XLA is free to use whatever device it
  // wants. In practice this usually has the effect of placing things on device
  // 0.
  xla::XlaScopedShardingAssignment assign_sharding(b, op_sharding);
  op_kernel->Compute(context);

  b->ClearOpMetadata();
  VLOG(4) << "Done";
}

Status XlaCompilationDevice::Sync() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilation_deviceDTcc mht_11(mht_11_v, 346, "", "./tensorflow/compiler/tf2xla/xla_compilation_device.cc", "XlaCompilationDevice::Sync");
 return Status::OK(); }

Status XlaCompilationDevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilation_deviceDTcc mht_12(mht_12_v, 353, "", "./tensorflow/compiler/tf2xla/xla_compilation_device.cc", "XlaCompilationDevice::MakeTensorFromProto");

  return errors::InvalidArgument(
      "XLACompilationDevice::MakeTensorFromProto should not be called");
}

}  // namespace tensorflow
