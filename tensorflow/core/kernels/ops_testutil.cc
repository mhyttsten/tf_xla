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
class MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc() {
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

#include "tensorflow/core/framework/node_properties.h"
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#include "tensorflow/core/common_runtime/gpu/gpu_managed_allocator.h"
#endif

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {
namespace test {

void SetOutputAttrs(OpKernelContext::Params* params,
                    std::vector<AllocatorAttributes>* attrs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_0(mht_0_v, 226, "", "./tensorflow/core/kernels/ops_testutil.cc", "SetOutputAttrs");

  attrs->clear();
  for (int index = 0; index < params->op_kernel->num_outputs(); index++) {
    AllocatorAttributes attr;
    const bool on_host =
        (params->op_kernel->output_memory_types()[index] == HOST_MEMORY);
    attr.set_on_host(on_host);
    attrs->push_back(attr);
  }
  params->output_attr_array = attrs->data();
}

}  // namespace test

OpsTestBase::OpsTestBase() : device_type_(DEVICE_CPU) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_1(mht_1_v, 243, "", "./tensorflow/core/kernels/ops_testutil.cc", "OpsTestBase::OpsTestBase");

  auto device = DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0");
  CHECK(device) << "Could not create CPU device";

  thread_pool_ = absl::make_unique<thread::ThreadPool>(
      Env::Default(), /*name=*/"default", /*num_threads=*/1);

  device_ = device.get();
  device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(device));

  allocator_ = device_->GetAllocator(AllocatorAttributes());

  flib_def_ = absl::make_unique<FunctionLibraryDefinition>(
      OpRegistry::Global(), FunctionDefLibrary{});
  pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr_.get(), Env::Default(), /*config=*/nullptr,
      TF_GRAPH_DEF_VERSION, flib_def_.get(), OptimizerOptions());
}

OpsTestBase::~OpsTestBase() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_2(mht_2_v, 265, "", "./tensorflow/core/kernels/ops_testutil.cc", "OpsTestBase::~OpsTestBase");

  for (auto& temp : tensors_) {
    delete temp;
  }
  for (auto& temp : managed_outputs_) {
    delete temp;
  }
  tensors_.clear();
  managed_outputs_.clear();
  context_.reset(nullptr);
  params_.reset(nullptr);
}

void OpsTestBase::SetDevice(const DeviceType& device_type,
                            std::unique_ptr<Device> device) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_3(mht_3_v, 282, "", "./tensorflow/core/kernels/ops_testutil.cc", "OpsTestBase::SetDevice");

  CHECK(device_) << "No device provided";

  device_ = device.get();
  device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(device));
  pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr_.get(), Env::Default(), /*config=*/nullptr,
      TF_GRAPH_DEF_VERSION, flib_def_.get(), OptimizerOptions(),
      thread_pool_.get());

  device_type_ = device_type;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  if (device_type == DEVICE_GPU) {
    managed_allocator_.reset(new GpuManagedAllocator());
    allocator_ = managed_allocator_.get();
  } else {
    managed_allocator_.reset();
    allocator_ = device_->GetAllocator(AllocatorAttributes());
  }
#else
  CHECK_NE(device_type, DEVICE_GPU)
      << "Requesting GPU on binary compiled without GOOGLE_CUDA or "
         "TENSORFLOW_USE_ROCM.";
  allocator_ = device_->GetAllocator(AllocatorAttributes());
#endif
}

void OpsTestBase::set_node_def(const NodeDef& node_def) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_4(mht_4_v, 312, "", "./tensorflow/core/kernels/ops_testutil.cc", "OpsTestBase::set_node_def");

  node_def_.CopyFrom(node_def);
}

NodeDef* OpsTestBase::node_def() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_5(mht_5_v, 319, "", "./tensorflow/core/kernels/ops_testutil.cc", "OpsTestBase::node_def");
 return &node_def_; }

Status OpsTestBase::InitOp() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_6(mht_6_v, 324, "", "./tensorflow/core/kernels/ops_testutil.cc", "OpsTestBase::InitOp");

  return InitOpWithGraphVersion(TF_GRAPH_DEF_VERSION);
}

Status OpsTestBase::InitOpWithGraphVersion(int graph_def_version) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_7(mht_7_v, 331, "", "./tensorflow/core/kernels/ops_testutil.cc", "OpsTestBase::InitOpWithGraphVersion");

  std::shared_ptr<const NodeProperties> props;
  TF_RETURN_IF_ERROR(NodeProperties::CreateFromNodeDef(
      node_def_, OpRegistry::Global(), &props));
  OpKernel* kernel;
  TF_RETURN_IF_ERROR(CreateOpKernel(
      device_type_, device_, allocator(), /*flib=*/nullptr,
      device_->resource_manager(), props, graph_def_version, &kernel));
  kernel_.reset(kernel);
  input_types_ = kernel_->input_types();
  return Status::OK();
}

Status OpsTestBase::RunOpKernel() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_8(mht_8_v, 347, "", "./tensorflow/core/kernels/ops_testutil.cc", "OpsTestBase::RunOpKernel");

  // Make sure the old OpKernelContext is deleted before the Params
  // it was using.
  context_.reset(nullptr);

  // Delete the output copies from previous runs.
  for (auto& temp : managed_outputs_) {
    delete temp;
  }
  managed_outputs_.clear();
  managed_outputs_.resize(0);

  params_.reset(new OpKernelContext::Params);
  params_->device = device_;
  params_->frame_iter = FrameAndIter(0, 0);
  params_->inputs = &inputs_;
  params_->op_kernel = kernel_.get();
  step_container_.reset(new ScopedStepContainer(0, [](const string&) {}));
  params_->step_container = step_container_.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(params_.get(), &attrs);
  checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
  params_->slice_reader_cache = &slice_reader_cache_wrapper;
  params_->resource_manager = device_->resource_manager();
  params_->function_library = pflr_->GetFLR(device_->name());

  context_.reset(new OpKernelContext(params_.get()));
  device_->Compute(kernel_.get(), context_.get());
  return context_->status();
}

const Tensor& OpsTestBase::GetInput(int input_index) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_9(mht_9_v, 381, "", "./tensorflow/core/kernels/ops_testutil.cc", "OpsTestBase::GetInput");

  CHECK_LT(input_index, context_->num_inputs());
  CHECK(!IsRefType(context_->input_dtype(input_index)));
  return context_->input(input_index);
}

TensorValue OpsTestBase::mutable_input(int input_index) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_10(mht_10_v, 390, "", "./tensorflow/core/kernels/ops_testutil.cc", "OpsTestBase::mutable_input");

  CHECK_LT(input_index, inputs_.size());
  return inputs_[input_index];
}

Tensor* OpsTestBase::GetOutput(int output_index) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_11(mht_11_v, 398, "", "./tensorflow/core/kernels/ops_testutil.cc", "OpsTestBase::GetOutput");

  CHECK_LT(output_index, context_->num_outputs());
  Tensor* output = context_->mutable_output(output_index);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  if (device_type_ == DEVICE_GPU) {
    managed_outputs_.resize(context_->num_outputs());
    // Copy the output tensor to managed memory if we haven't done so.
    if (!managed_outputs_[output_index]) {
      Tensor* managed_output =
          new Tensor(allocator(), output->dtype(), output->shape());
      auto src = output->tensor_data();
      auto dst = managed_output->tensor_data();
      context_->eigen_gpu_device().memcpyDeviceToHost(
          const_cast<char*>(dst.data()), src.data(), src.size());
      context_->eigen_gpu_device().synchronize();
      managed_outputs_[output_index] = managed_output;
    }
    output = managed_outputs_[output_index];
  }
#endif
  return output;
}

Allocator* OpsTestBase::allocator() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_12(mht_12_v, 424, "", "./tensorflow/core/kernels/ops_testutil.cc", "OpsTestBase::allocator");
 return allocator_; }

OpKernel* OpsTestBase::op_kernel() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_13(mht_13_v, 429, "", "./tensorflow/core/kernels/ops_testutil.cc", "OpsTestBase::op_kernel");
 return kernel_.get(); }

const DataTypeVector& OpsTestBase::output_types() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_14(mht_14_v, 434, "", "./tensorflow/core/kernels/ops_testutil.cc", "OpsTestBase::output_types");

  return kernel_->output_types();
}

Tensor* OpsTestBase::AddInput(DataType dtype, const TensorShape& shape) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_15(mht_15_v, 441, "", "./tensorflow/core/kernels/ops_testutil.cc", "OpsTestBase::AddInput");

  CHECK_GT(input_types_.size(), inputs_.size())
      << "Adding more inputs than types; perhaps you need to call MakeOp";
  bool is_ref = IsRefType(input_types_[inputs_.size()]);
  Tensor* input = new Tensor(allocator(), dtype, shape);
  tensors_.push_back(input);
  if (is_ref) {
    CHECK_EQ(RemoveRefType(input_types_[inputs_.size()]), dtype);
    inputs_.push_back({&lock_for_refs_, input});
  } else {
    CHECK_EQ(input_types_[inputs_.size()], dtype);
    inputs_.push_back({nullptr, input});
  }
  return input;
}

void OpsTestBase::AddResourceInputInternal(const std::string& container_name,
                                           const std::string& name,
                                           const TypeIndex& type_index) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("container_name: \"" + container_name + "\"");
   mht_16_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTcc mht_16(mht_16_v, 464, "", "./tensorflow/core/kernels/ops_testutil.cc", "OpsTestBase::AddResourceInputInternal");

  ResourceHandle handle;
  handle.set_device(device_->name());
  handle.set_container(container_name);
  handle.set_name(name);
  handle.set_hash_code(type_index.hash_code());
  handle.set_maybe_type_name(type_index.name());
  Tensor* input = new Tensor(allocator(), DT_RESOURCE, TensorShape({}));
  input->scalar<ResourceHandle>()() = handle;
  tensors_.push_back(input);
  inputs_.push_back({nullptr, input});
}

}  // namespace tensorflow
