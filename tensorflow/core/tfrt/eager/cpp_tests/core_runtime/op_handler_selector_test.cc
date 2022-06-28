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
class MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc() {
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

#include <memory>
#include <string>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tfrt/cpu/core_runtime/null_op_handler.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace tfrt {
namespace tf {
namespace {

using ::tensorflow::AbstractTensorHandle;
using ::tensorflow::Allocator;
using ::tensorflow::AllocatorAttributes;
using ::tensorflow::AttrBuilder;
using ::tensorflow::DataType;
using ::tensorflow::DEVICE_CPU;
using ::tensorflow::DeviceAttributes;
using ::tensorflow::DynamicDeviceMgr;
using ::tensorflow::EagerContext;
using ::tensorflow::ImmediateExecutionOperation;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::SessionOptions;
using ::tensorflow::Status;

constexpr char kFullCPU[] = "/job:a/replica:0/task:0/device:CPU:0";
constexpr char kFullGPU[] = "/job:a/replica:0/task:0/device:FakeGPU:0";

////////////////////////////////////////////////////////////////////////////////
//
// Op, kernel to set up the environment.
//
// The Placer uses information about the op (input types),
// kernel (device constraints). To avoid depending on the full runtime, we
// define dummy implementations of these, and register them with the
// runtime.
//
////////////////////////////////////////////////////////////////////////////////

// A dummy OpKernel that is used to register ops on different devices.
class DummyOp : public OpKernel {
 public:
  explicit DummyOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_0(mht_0_v, 251, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "DummyOp");
}
  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_1(mht_1_v, 255, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "Compute");
}
};

// Register the following ops so they can be added to a Graph, and
// kernels so that they can be placed on particular device types.
REGISTER_OP("InvalidOp").Output("o: Ref(float)");

REGISTER_OP("TestOp").Output("o: Ref(float)");
REGISTER_KERNEL_BUILDER(Name("TestOp").Device(DEVICE_CPU).Priority(1), DummyOp);
REGISTER_KERNEL_BUILDER(Name("TestOp").Device("FakeGPU").Priority(2), DummyOp);

static tensorflow::Device* CreateDevice(const char* type, const char* name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("type: \"" + (type == nullptr ? std::string("nullptr") : std::string((char*)type)) + "\"");
   mht_2_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_2(mht_2_v, 271, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "CreateDevice");

  class FakeDevice : public tensorflow::Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_3(mht_3_v, 277, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "FakeDevice");
}
    Status Sync() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_4(mht_4_v, 281, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "Sync");
 return Status::OK(); }
    Allocator* GetAllocator(AllocatorAttributes) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_5(mht_5_v, 285, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "GetAllocator");
 return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  return new FakeDevice(attr);
}

class FakeTensorHandle : public tensorflow::ImmediateExecutionTensorHandle {
 public:
  explicit FakeTensorHandle(string_view device_name, tensorflow::DataType dtype)
      : ImmediateExecutionTensorHandle(kTfrt),
        device_name_(device_name),
        dtype_(dtype) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("device_name: \"" + std::string(device_name.data(), device_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_6(mht_6_v, 302, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "FakeTensorHandle");
}

  void Release() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_7(mht_7_v, 307, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "Release");
 Unref(); }

  tensorflow::DataType DataType() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_8(mht_8_v, 312, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "DataType");
 return dtype_; }
  Status Shape(tensorflow::PartialTensorShape* shape) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_9(mht_9_v, 316, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "Shape");

    int64_t dim_sizes[] = {1};
    return tensorflow::PartialTensorShape::MakePartialShape(dim_sizes, 1,
                                                            shape);
  }
  Status NumDims(int* num_dims) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_10(mht_10_v, 324, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "NumDims");

    *num_dims = 1;
    return Status::OK();
  }
  Status NumElements(int64_t* num_elements) const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_11(mht_11_v, 331, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "NumElements");

    *num_elements = 1;
    return Status::OK();
  }
  Status Dim(int dim_index, int64_t* dim) const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_12(mht_12_v, 338, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "Dim");

    llvm_unreachable("unimplemented method.");
  }

  const char* DeviceName(Status* status) const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_13(mht_13_v, 345, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "DeviceName");

    return device_name_.c_str();
  }
  const char* BackingDeviceName(Status* status) const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_14(mht_14_v, 351, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "BackingDeviceName");

    llvm_unreachable("unimplemented method.");
  }
  const char* DeviceType(Status* status) const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_15(mht_15_v, 357, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "DeviceType");

    llvm_unreachable("unimplemented method.");
  }
  int DeviceId(Status* status) const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_16(mht_16_v, 363, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "DeviceId");

    llvm_unreachable("unimplemented method.");
  }
  tensorflow::AbstractTensorInterface* Resolve(Status* status) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_17(mht_17_v, 369, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "Resolve");

    llvm_unreachable("unimplemented method.");
  }
  ImmediateExecutionTensorHandle* Copy() override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_18(mht_18_v, 375, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "Copy");

    Ref();
    return this;
  }

  static bool classof(const AbstractTensorHandle* ptr) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_19(mht_19_v, 383, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "classof");
 return true; }

 private:
  std::string device_name_;
  tensorflow::DataType dtype_;
};

class FakeOperation : public ImmediateExecutionOperation {
 public:
  explicit FakeOperation() : ImmediateExecutionOperation(kTfrt) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_20(mht_20_v, 395, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "FakeOperation");
}
  ~FakeOperation() override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_21(mht_21_v, 399, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "~FakeOperation");
}

  void Release() override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_22(mht_22_v, 404, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "Release");
 delete this; }

  void Clear() override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_23(mht_23_v, 409, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "Clear");
 args_.clear(); }

  tensorflow::ImmediateExecutionContext* GetContext() const override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_24(mht_24_v, 414, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "GetContext");

    return nullptr;
  }

  bool HasCustomDeviceInput() const override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_25(mht_25_v, 421, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "HasCustomDeviceInput");
 return false; }

  Status Reset(const char* op, const char* raw_device_name) override {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   mht_26_v.push_back("raw_device_name: \"" + (raw_device_name == nullptr ? std::string("nullptr") : std::string((char*)raw_device_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_26(mht_26_v, 428, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "Reset");

    op_name_ = op;
    device_name_ = raw_device_name;
    attrs_.Reset(op);
    args_.clear();
    return Status::OK();
  }
  const std::string& Name() const override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_27(mht_27_v, 438, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "Name");
 return op_name_; }
  const std::string& DeviceName() const override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_28(mht_28_v, 442, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "DeviceName");
 return device_name_; }
  tensorflow::Status SetDeviceName(const char* name) override {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_29(mht_29_v, 447, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetDeviceName");

    device_name_ = name;
    return Status::OK();
  }

  Status AddInput(AbstractTensorHandle* input) override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_30(mht_30_v, 455, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "AddInput");

    input->Ref();
    args_.push_back(tensorflow::core::RefCountPtr<FakeTensorHandle>(
        static_cast<FakeTensorHandle*>(input)));
    attrs_.NumInputs(args_.size());
    return Status::OK();
  }
  Status SetInput(size_t index,
                  tensorflow::ImmediateExecutionTensorHandle* input) override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_31(mht_31_v, 466, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetInput");

    llvm_unreachable("unimplemented method.");
  }
  Status AddInputList(absl::Span<AbstractTensorHandle* const> inputs) override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_32(mht_32_v, 472, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "AddInputList");

    llvm_unreachable("unimplemented method.");
  }
  absl::Span<tensorflow::ImmediateExecutionTensorHandle* const> GetInputs()
      const override {
    return absl::MakeSpan(
        reinterpret_cast<tensorflow::ImmediateExecutionTensorHandle* const*>(
            args_.data()),
        args_.size());
  }
  Status Execute(absl::Span<AbstractTensorHandle*> retvals,
                 int* num_retvals) override {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_33(mht_33_v, 486, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "Execute");

    llvm_unreachable("unimplemented method.");
  }
  const tensorflow::OpDef* OpDef() const override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_34(mht_34_v, 492, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "OpDef");

    llvm_unreachable("unimplemented method.");
  }
  const tensorflow::AbstractOpAttrs* GetOpAttrs() const override {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_35(mht_35_v, 498, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "GetOpAttrs");

    llvm_unreachable("unimplemented method.");
  }
  void AddAttrs(const tensorflow::AbstractOpAttrs* op_attrs) override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_36(mht_36_v, 504, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "AddAttrs");

    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrString(const char* attr_name, const char* data,
                       size_t length) override {
   std::vector<std::string> mht_37_v;
   mht_37_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_37_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_37(mht_37_v, 513, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetAttrString");

    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrInt(const char* attr_name, int64_t value) override {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_38(mht_38_v, 520, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetAttrInt");

    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrFloat(const char* attr_name, float value) override {
   std::vector<std::string> mht_39_v;
   mht_39_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_39(mht_39_v, 527, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetAttrFloat");

    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrBool(const char* attr_name, bool value) override {
   std::vector<std::string> mht_40_v;
   mht_40_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_40(mht_40_v, 534, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetAttrBool");

    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrType(const char* attr_name,
                     tensorflow::DataType value) override {
   std::vector<std::string> mht_41_v;
   mht_41_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_41(mht_41_v, 542, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetAttrType");

    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrShape(const char* attr_name, const int64_t* dims,
                      const int num_dims) override {
   std::vector<std::string> mht_42_v;
   mht_42_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_42(mht_42_v, 550, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetAttrShape");

    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrFunction(const char* attr_name,
                         const AbstractOperation* value) override {
   std::vector<std::string> mht_43_v;
   mht_43_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_43(mht_43_v, 558, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetAttrFunction");

    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrFunctionName(const char* attr_name, const char* data,
                             size_t length) override {
   std::vector<std::string> mht_44_v;
   mht_44_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_44_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_44(mht_44_v, 567, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetAttrFunctionName");

    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrTensor(const char* attr_name,
                       tensorflow::AbstractTensorInterface* tensor) override {
   std::vector<std::string> mht_45_v;
   mht_45_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_45(mht_45_v, 575, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetAttrTensor");

    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrStringList(const char* attr_name, const void* const* values,
                           const size_t* lengths, int num_values) override {
   std::vector<std::string> mht_46_v;
   mht_46_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_46(mht_46_v, 583, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetAttrStringList");

    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrFloatList(const char* attr_name, const float* values,
                          int num_values) override {
   std::vector<std::string> mht_47_v;
   mht_47_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_47(mht_47_v, 591, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetAttrFloatList");

    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrIntList(const char* attr_name, const int64_t* values,
                        int num_values) override {
   std::vector<std::string> mht_48_v;
   mht_48_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_48(mht_48_v, 599, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetAttrIntList");

    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrTypeList(const char* attr_name,
                         const tensorflow::DataType* values,
                         int num_values) override {
   std::vector<std::string> mht_49_v;
   mht_49_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_49(mht_49_v, 608, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetAttrTypeList");

    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrBoolList(const char* attr_name, const unsigned char* values,
                         int num_values) override {
   std::vector<std::string> mht_50_v;
   mht_50_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_50_v.push_back("values: \"" + (values == nullptr ? std::string("nullptr") : std::string((char*)values)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_50(mht_50_v, 617, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetAttrBoolList");

    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrShapeList(const char* attr_name, const int64_t** dims,
                          const int* num_dims, int num_values) override {
   std::vector<std::string> mht_51_v;
   mht_51_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_51(mht_51_v, 625, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetAttrShapeList");

    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrFunctionList(
      const char* attr_name,
      absl::Span<const AbstractOperation*> values) override {
   std::vector<std::string> mht_52_v;
   mht_52_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_52(mht_52_v, 634, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetAttrFunctionList");

    llvm_unreachable("unimplemented method.");
  }

  Status InputLength(const char* input_name, int* length) override {
   std::vector<std::string> mht_53_v;
   mht_53_v.push_back("input_name: \"" + (input_name == nullptr ? std::string("nullptr") : std::string((char*)input_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_53(mht_53_v, 642, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "InputLength");

    llvm_unreachable("unimplemented method.");
  }
  Status OutputLength(const char* output_name, int* length) override {
   std::vector<std::string> mht_54_v;
   mht_54_v.push_back("output_name: \"" + (output_name == nullptr ? std::string("nullptr") : std::string((char*)output_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_54(mht_54_v, 649, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "OutputLength");

    llvm_unreachable("unimplemented method.");
  }

  void SetCancellationManager(
      tensorflow::CancellationManager* cancellation_manager) override {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_55(mht_55_v, 657, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetCancellationManager");

    llvm_unreachable("unimplemented method.");
  }

  void SetStackTrace(tensorflow::ManagedStackTrace stack_trace) override {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_56(mht_56_v, 664, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetStackTrace");

    llvm_unreachable("unimplemented method.");
  }

  absl::optional<tensorflow::ManagedStackTrace> GetStackTrace() override {
    llvm_unreachable("unimplemented method.");
  }

  void SetStepId(int64_t step_id) override {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_57(mht_57_v, 675, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SetStepId");

    llvm_unreachable("unimplemented method.");
  }

  static bool classof(const AbstractOperation* ptr) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_58(mht_58_v, 682, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "classof");
 return true; }

  AttrBuilder* GetAttrs() {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_59(mht_59_v, 687, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "GetAttrs");
 return &attrs_; }

 private:
  std::string op_name_;
  std::string device_name_;
  llvm::SmallVector<tensorflow::core::RefCountPtr<FakeTensorHandle>, 8> args_;
  AttrBuilder attrs_;
};

static std::unique_ptr<CoreRuntime> CreateCoreRuntime() {
  auto diag_handler = [](const DecodedDiagnostic& diag) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_60(mht_60_v, 700, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "lambda");

    LOG(ERROR) << "Encountered runtime error: " << diag.message << "\n";
  };
  auto corert =
      CoreRuntime::Create(diag_handler, tfrt::CreateMallocAllocator(),
                          tfrt::CreateMultiThreadedWorkQueue(
                              /*num_threads=*/4, /*num_blocking_threads=*/64),
                          kFullCPU);

  assert(corert);
  return std::move(*corert);
}

class SelectorTest : public ::testing::Test {
 public:
  SelectorTest() {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_61(mht_61_v, 718, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "SelectorTest");

    device_manager_ = new DynamicDeviceMgr();
    std::vector<std::unique_ptr<tensorflow::Device>> added_devices;
    SessionOptions opts;

    // Have to use real CPU device. Other, ctx->HostCPU() will return invalid
    // device.
    added_devices.emplace_back(CreateDevice(tensorflow::DEVICE_CPU, kFullCPU));
    added_devices.emplace_back(CreateDevice("FakeGPU", kFullGPU));

    TF_CHECK_OK(device_manager_->AddDevices(std::move(added_devices)));

    SessionOptions options;
    options.config.set_log_device_placement(true);
    options.config.set_allow_soft_placement(true);
    eager_context_ = new EagerContext(
        options,
        tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
        /* async */ false, device_manager_,
        /* device_mgr_owned */ false, /* rendezvous */ nullptr,
        /* cluster_flr */ nullptr);
    corert_ = CreateCoreRuntime();
    fallback_op_handler_ = CreateOpHandler();
    cpu_op_handler_ = CreateOpHandler();
    gpu_op_handler_ = CreateOpHandler();
    corert_->RegisterOpHandler(kFullCPU, cpu_op_handler_);
    corert_->RegisterOpHandler(kFullGPU, gpu_op_handler_);

    selector_ = std::make_unique<EagerOpHandlerSelector>(
        corert_.get(), eager_context_, fallback_op_handler_,
        /*pin_small_ops_to_cpu=*/true);
  }

  ~SelectorTest() override {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_62(mht_62_v, 754, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "~SelectorTest");

    delete device_manager_;
    if (eager_context_) {
      eager_context_->Unref();
    }
  }

  EagerOpHandlerSelector* selector() {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_63(mht_63_v, 764, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "selector");
 return selector_.get(); }

  void Init() {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_64(mht_64_v, 769, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "Init");
}

 protected:
  OpHandler* CreateOpHandler() {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPScpp_testsPScore_runtimePSop_handler_selector_testDTcc mht_65(mht_65_v, 775, "", "./tensorflow/core/tfrt/eager/cpp_tests/core_runtime/op_handler_selector_test.cc", "CreateOpHandler");

    auto expected_op_handler = tfrt::CreateNullOpHandler(corert_.get());
    assert(expected_op_handler);
    return std::move(expected_op_handler.get());
  }

  DynamicDeviceMgr* device_manager_;
  EagerContext* eager_context_;
  std::unique_ptr<CoreRuntime> corert_;
  OpHandler* fallback_op_handler_;
  OpHandler* cpu_op_handler_;
  OpHandler* gpu_op_handler_;
  std::unique_ptr<EagerOpHandlerSelector> selector_;
};

TEST_F(SelectorTest, PinSmallOpToCpuTest) {
  auto op = std::make_unique<FakeOperation>();
  tensorflow::core::RefCountPtr<FakeTensorHandle> cpu_tensor(
      new FakeTensorHandle(kFullCPU, tensorflow::DT_INT32));
  tensorflow::core::RefCountPtr<FakeTensorHandle> gpu_tensor(
      new FakeTensorHandle(kFullGPU, tensorflow::DT_INT32));

  tensorflow::Status s;
  TF_ASSERT_OK(op->Reset("TestOp", kFullGPU));
  TF_ASSERT_OK(op->AddInput(cpu_tensor.get()));
  OpHandler* op_handler = nullptr;
  s = selector()->SelectFromArguments(*op, &op_handler);
  ASSERT_EQ(s, tensorflow::Status::OK());
  ASSERT_TRUE(static_cast<bool>(op_handler));
  ASSERT_EQ(op_handler, cpu_op_handler_);

  op_handler = nullptr;
  TF_ASSERT_OK(op->Reset("TestOp", kFullGPU));
  TF_ASSERT_OK(op->AddInput(gpu_tensor.get()));
  s = selector()->SelectFromArguments(*op, &op_handler);
  ASSERT_EQ(s, tensorflow::Status::OK());
  ASSERT_FALSE(static_cast<bool>(op_handler));
  s = selector()->SelectFromNodeDef(*op, &op->GetAttrs()->BuildNodeDef(),
                                    &op_handler);
  ASSERT_EQ(s, tensorflow::Status::OK());
  ASSERT_TRUE(static_cast<bool>(op_handler));
  ASSERT_EQ(op_handler, gpu_op_handler_);
}

TEST_F(SelectorTest, PinResourceTest) {
  auto op = std::make_unique<FakeOperation>();
  tensorflow::core::RefCountPtr<FakeTensorHandle> cpu_tensor(
      new FakeTensorHandle(kFullCPU, tensorflow::DT_RESOURCE));
  tensorflow::core::RefCountPtr<FakeTensorHandle> gpu_tensor(
      new FakeTensorHandle(kFullGPU, tensorflow::DT_RESOURCE));

  tensorflow::Status s;
  TF_ASSERT_OK(op->Reset("TestOp", kFullGPU));
  TF_ASSERT_OK(op->AddInput(cpu_tensor.get()));
  OpHandler* op_handler = nullptr;
  s = selector()->SelectFromArguments(*op, &op_handler);
  ASSERT_EQ(s, tensorflow::Status::OK());
  ASSERT_TRUE(static_cast<bool>(op_handler));
  ASSERT_EQ(op_handler, cpu_op_handler_);

  op_handler = nullptr;
  TF_ASSERT_OK(op->Reset("TestOp", kFullCPU));
  TF_ASSERT_OK(op->AddInput(gpu_tensor.get()));
  s = selector()->SelectFromArguments(*op, &op_handler);
  ASSERT_EQ(s, tensorflow::Status::OK());
  ASSERT_TRUE(static_cast<bool>(op_handler));
  ASSERT_EQ(op_handler, gpu_op_handler_);
}

TEST_F(SelectorTest, InvalidDeviceNameTest) {
  auto op = std::make_unique<FakeOperation>();

  TF_ASSERT_OK(op->Reset("TestOp", "invalid_device_name"));

  tensorflow::Status s;
  OpHandler* op_handler = nullptr;
  s = selector()->SelectFromNodeDef(*op, &op->GetAttrs()->BuildNodeDef(),
                                    &op_handler);
  ASSERT_EQ(s.code(), tensorflow::error::INVALID_ARGUMENT);
  ASSERT_FALSE(static_cast<bool>(op_handler));
  EXPECT_TRUE(
      absl::StrContains(s.error_message(), "Failed to parse device name"));
}

TEST_F(SelectorTest, SoftPlacementTest) {
  auto op = std::make_unique<FakeOperation>();

  TF_ASSERT_OK(op->Reset("TestOp", "/device:FakeGPU:99"));
  tensorflow::Status s;
  OpHandler* op_handler = nullptr;
  s = selector()->SelectFromNodeDef(*op, &op->GetAttrs()->BuildNodeDef(),
                                    &op_handler);
  ASSERT_EQ(s, tensorflow::Status::OK());
  ASSERT_TRUE(static_cast<bool>(op_handler)) << StrCat(s.error_message());
  ASSERT_EQ(op_handler, gpu_op_handler_);
}

TEST_F(SelectorTest, HigherPriorityDeviceTest) {
  auto op = std::make_unique<FakeOperation>();

  tensorflow::Status s;
  TF_ASSERT_OK(op->Reset("TestOp", ""));
  OpHandler* op_handler = nullptr;
  s = selector()->SelectFromNodeDef(*op, &op->GetAttrs()->BuildNodeDef(),
                                    &op_handler);
  ASSERT_EQ(s, tensorflow::Status::OK());
  ASSERT_TRUE(static_cast<bool>(op_handler));
  ASSERT_EQ(op_handler, gpu_op_handler_);
}

}  // namespace
}  // namespace tf
}  // namespace tfrt
