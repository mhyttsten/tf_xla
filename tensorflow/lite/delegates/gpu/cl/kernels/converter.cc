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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/kernels/converter.h"

#include <algorithm>
#include <array>
#include <string>

#include "tensorflow/lite/delegates/gpu/cl/cl_arguments.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_errors.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/task/arguments.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

class OpenClConverterImpl : public TensorObjectConverter {
 public:
  virtual absl::Status Init(const TensorObjectDef& input_def,
                            const TensorObjectDef& output_def,
                            Environment* environment) = 0;

  void SetGpuInfo(const GpuInfo& info) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_0(mht_0_v, 214, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "SetGpuInfo");
 gpu_info_ = info; }

 protected:
  absl::Status DispatchKernel(cl_mem buffer_mem, Tensor* tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_1(mht_1_v, 220, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "DispatchKernel");

    kernel_.ResetBindingCounter();
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(buffer_mem));
    RETURN_IF_ERROR(cl_args_.SetObjectRef("tensor", tensor));
    RETURN_IF_ERROR(
        cl_args_.Bind(kernel_.kernel(), kernel_.GetBindingCounter()));
    const int3 grid = int3(tensor->Width() * tensor->Batch(), tensor->Height(),
                           tensor->Slices());
    std::vector<int3> work_groups;
    GetPossibleWorkGroupsConv(TuningType::kFast, gpu_info_, kernel_.info_, grid,
                              &work_groups);
    const int3 work_group_size = work_groups[0];
    const int3 work_groups_count = GetWorkGroupsCount(grid, work_group_size);
    return queue_->Dispatch(kernel_, work_groups_count, work_group_size);
  }

  CLArguments cl_args_;
  BHWC shape_;
  CLKernel kernel_;
  TensorDescriptor tensor_descriptor_;
  GpuInfo gpu_info_;
  CLCommandQueue* queue_ = nullptr;
  const CLContext* context_ = nullptr;
};

bool IsSupportedDataType(DataType type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_2(mht_2_v, 248, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "IsSupportedDataType");

  return type == DataType::FLOAT16 || type == DataType::FLOAT32;
}

bool IsBHWCOpenCLBuffer(const ObjectDef& def) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_3(mht_3_v, 255, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "IsBHWCOpenCLBuffer");

  return IsSupportedDataType(def.data_type) &&
         def.object_type == ObjectType::OPENCL_BUFFER &&
         def.data_layout == DataLayout::BHWC;
}

bool IsOpenCLTensor(const ObjectDef& def) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_4(mht_4_v, 264, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "IsOpenCLTensor");

  const bool is_buffer_tensor = def.object_type == ObjectType::OPENCL_BUFFER &&
                                def.data_layout == DataLayout::DHWC4;
  const bool is_image2d_tensor =
      def.object_type == ObjectType::OPENCL_TEXTURE &&
      def.data_layout == DataLayout::HDWC4;
  const bool is_image2d_array_tensor =
      def.object_type == ObjectType::OPENCL_TEXTURE &&
      def.data_layout == DataLayout::DHWC4;
  const bool is_single_image_tensor =
      def.object_type == ObjectType::OPENCL_TEXTURE &&
      def.data_layout == DataLayout::BHWC;
  return IsSupportedDataType(def.data_type) &&
         (is_buffer_tensor || is_image2d_tensor || is_image2d_array_tensor ||
          is_single_image_tensor);
}

absl::Status GetOpenCLMemory(const TensorObject& obj, cl_mem* memory) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_5(mht_5_v, 284, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "GetOpenCLMemory");

  auto texture = absl::get_if<OpenClTexture>(&obj);
  auto buffer = absl::get_if<OpenClBuffer>(&obj);
  if (texture && texture->memobj) {
    *memory = texture->memobj;
  } else if (buffer && buffer->memobj) {
    *memory = buffer->memobj;
  } else {
    return absl::InvalidArgumentError("Missing OpenCL object.");
  }
  return absl::OkStatus();
}

// Implements conversion from OpenCL tensor to another OpenCL tensor.
class TensorToTensorConverter : public OpenClConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_6(mht_6_v, 303, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "IsSupported");

    return IsOpenCLTensor(input) && IsOpenCLTensor(output);
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_7(mht_7_v, 312, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "Init");

    src_tensor_descriptor_.layout = Layout::BHWC;
    src_tensor_descriptor_.storage_type = ToTensorStorageType(
        input_def.object_def.object_type, input_def.object_def.data_layout);
    src_tensor_descriptor_.data_type = input_def.object_def.data_type;
    Arguments args;
    args.AddObjectRef(
        "src_tensor", AccessType::READ,
        absl::make_unique<TensorDescriptor>(src_tensor_descriptor_));

    dst_tensor_descriptor_.layout = Layout::BHWC;
    dst_tensor_descriptor_.storage_type = ToTensorStorageType(
        output_def.object_def.object_type, output_def.object_def.data_layout);
    dst_tensor_descriptor_.data_type = output_def.object_def.data_type;
    args.AddObjectRef(
        "dst_tensor", AccessType::WRITE,
        absl::make_unique<TensorDescriptor>(dst_tensor_descriptor_));

    const bool need_fp16_support =
        input_def.object_def.data_type == DataType::FLOAT16 ||
        output_def.object_def.data_type == DataType::FLOAT16;
    const std::string out_data_type =
        ToCLDataType(output_def.object_def.data_type);
    std::string shader_src;
    if (need_fp16_support) {
      shader_src += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    }
    shader_src +=
        R"(__kernel void tensor_to_tensor($0) {
  int linear_id = get_global_id(0);
  int x = linear_id / args.dst_tensor.Batch();
  int b = linear_id % args.dst_tensor.Batch();
  int y = get_global_id(1);
  int d = get_global_id(2);
  if (x >= args.dst_tensor.Width() || y >= args.dst_tensor.Height() || d >= args.dst_tensor.Slices()) return;
)";
    shader_src += "  " + out_data_type + "4 input = args.src_tensor.Read<" +
                  out_data_type + ">(x, y, d, b);\n";
    shader_src += "  args.dst_tensor.Write(input, x, y, d, b);\n}";
    queue_ = environment->queue();
    context_ = &environment->context();
    shape_ = BHWC(input_def.dimensions.b, input_def.dimensions.h,
                  input_def.dimensions.w, input_def.dimensions.c);
    RETURN_IF_ERROR(
        args.Compile(environment->device().GetInfo(), {}, &shader_src));
    RETURN_IF_ERROR(cl_args_.Init(environment->device().GetInfo(), nullptr,
                                  &args, &shader_src));
    return environment->program_cache()->GetOrCreateCLKernel(
        shader_src, "tensor_to_tensor", environment->context(),
        environment->device(), &kernel_);
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_8(mht_8_v, 368, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "Convert");

    cl_mem in_memory;
    RETURN_IF_ERROR(GetOpenCLMemory(input_obj, &in_memory));
    cl_mem out_memory;
    RETURN_IF_ERROR(GetOpenCLMemory(output_obj, &out_memory));

    Tensor src_tensor;
    RETURN_IF_ERROR(CreateSharedTensor(*context_, in_memory, shape_,
                                       src_tensor_descriptor_, &src_tensor));
    Tensor dst_tensor;
    RETURN_IF_ERROR(CreateSharedTensor(*context_, out_memory, shape_,
                                       dst_tensor_descriptor_, &dst_tensor));
    RETURN_IF_ERROR(cl_args_.SetObjectRef("src_tensor", &src_tensor));
    RETURN_IF_ERROR(cl_args_.SetObjectRef("dst_tensor", &dst_tensor));
    RETURN_IF_ERROR(cl_args_.Bind(kernel_.kernel()));
    const int3 grid = int3(dst_tensor.Width() * dst_tensor.Batch(),
                           dst_tensor.Height(), dst_tensor.Slices());
    const int3 work_group_size = {16, 8, 1};
    const int3 work_groups_count = GetWorkGroupsCount(grid, work_group_size);
    return queue_->Dispatch(kernel_, work_groups_count, work_group_size);
  }

 private:
  TensorDescriptor src_tensor_descriptor_;
  TensorDescriptor dst_tensor_descriptor_;
};

// Implements conversion from OpenCL-specific tensor layout to BHWC OpenCL
// buffer.
class TensorToBHWCBufferConverter : public OpenClConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_9(mht_9_v, 402, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "IsSupported");

    return IsOpenCLTensor(input) && IsBHWCOpenCLBuffer(output);
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_10(mht_10_v, 411, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "Init");

    TensorStorageType src_tensor_type = ToTensorStorageType(
        input_def.object_def.object_type, input_def.object_def.data_layout);
    tensor_descriptor_.layout = Layout::BHWC;
    tensor_descriptor_.storage_type = src_tensor_type;
    tensor_descriptor_.data_type = input_def.object_def.data_type;
    Arguments args;
    args.AddObjectRef("tensor", AccessType::READ,
                      absl::make_unique<TensorDescriptor>(tensor_descriptor_));

    const bool need_fp16_support =
        input_def.object_def.data_type == DataType::FLOAT16 ||
        output_def.object_def.data_type == DataType::FLOAT16;
    std::string shader_src;
    if (need_fp16_support) {
      shader_src += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    }
    const std::string out_data_type =
        ToCLDataType(output_def.object_def.data_type);
    shader_src += "__kernel void tensor_to_bhwc(";
    shader_src += "__global " + out_data_type + "* dst, $0) {\n";
    shader_src += R"(  int linear_id = get_global_id(0);
  int x = linear_id / args.tensor.Batch();
  int b = linear_id % args.tensor.Batch();
  int y = get_global_id(1);
  int d = get_global_id(2);
  if (x >= args.tensor.Width() || y >= args.tensor.Height() || d >= args.tensor.Slices()) return;
)";
    shader_src += "  " + out_data_type + "4 input = args.tensor.Read<" +
                  out_data_type + ">(x, y, d, b);\n";
    shader_src += R"(  int c = d * 4;
  int index = ((b * args.tensor.Height() + y) * args.tensor.Width() + x) * args.tensor.Channels() + c;

  dst[index] = input.x;
  if (c + 1 < args.tensor.Channels()) {
    dst[index + 1] = input.y;
  }
  if (c + 2 < args.tensor.Channels()) {
    dst[index + 2] = input.z;
  }
  if (c + 3 < args.tensor.Channels()) {
    dst[index + 3] = input.w;
  }
})";
    queue_ = environment->queue();
    context_ = &environment->context();
    shape_ = BHWC(input_def.dimensions.b, input_def.dimensions.h,
                  input_def.dimensions.w, input_def.dimensions.c);
    RETURN_IF_ERROR(
        args.Compile(environment->device().GetInfo(), {}, &shader_src));
    RETURN_IF_ERROR(cl_args_.Init(environment->device().GetInfo(), nullptr,
                                  &args, &shader_src));
    return environment->program_cache()->GetOrCreateCLKernel(
        shader_src, "tensor_to_bhwc", environment->context(),
        environment->device(), &kernel_);
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_11(mht_11_v, 472, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "Convert");

    auto output = absl::get_if<OpenClBuffer>(&output_obj);
    if (!output || !output->memobj) {
      return absl::InvalidArgumentError(
          "Missing output in tensor_to_bhwc converter");
    }

    cl_mem in_memory;
    RETURN_IF_ERROR(GetOpenCLMemory(input_obj, &in_memory));
    Tensor tensor;
    RETURN_IF_ERROR(CreateSharedTensor(*context_, in_memory, shape_,
                                       tensor_descriptor_, &tensor));
    return DispatchKernel(output->memobj, &tensor);
  }
};

// Implements conversion from BHWC OpenCL buffer to OpenCL-specific tensor
// layout.
class BHWCBufferToTensorConverter : public OpenClConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_12(mht_12_v, 495, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "IsSupported");

    return IsBHWCOpenCLBuffer(input) && IsOpenCLTensor(output);
  }

  std::pair<std::string, std::string> GetFromBhwcKernel(
      const TensorObjectDef& input_def,
      const TensorObjectDef& output_def) const {
    return std::make_pair(
        "__global " + ToCLDataType(input_def.object_def.data_type) + "* src",
        R"(int c = d * 4;
  int index = ((b * args.tensor.Height() + y) * args.tensor.Width() + x) * args.tensor.Channels() + c;
  result.x = src[index];
  result.y = c + 1 < args.tensor.Channels() ? src[index + 1] : 1;
  result.z = c + 2 < args.tensor.Channels() ? src[index + 2] : 2;
  result.w = c + 3 < args.tensor.Channels() ? src[index + 3] : 3;
)");
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_13(mht_13_v, 518, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "Init");

    auto params_kernel = GetFromBhwcKernel(input_def, output_def);

    TensorStorageType dst_tensor_type = ToTensorStorageType(
        output_def.object_def.object_type, output_def.object_def.data_layout);
    tensor_descriptor_.layout = Layout::BHWC;
    tensor_descriptor_.storage_type = dst_tensor_type;
    tensor_descriptor_.data_type = output_def.object_def.data_type;
    Arguments args;
    args.AddObjectRef("tensor", AccessType::WRITE,
                      absl::make_unique<TensorDescriptor>(tensor_descriptor_));

    const bool need_fp16_support =
        input_def.object_def.data_type == DataType::FLOAT16 ||
        output_def.object_def.data_type == DataType::FLOAT16;
    std::string shader_src;
    if (need_fp16_support) {
      shader_src += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    }
    const std::string in_data_type =
        ToCLDataType(input_def.object_def.data_type);
    const std::string out_data_type =
        ToCLDataType(output_def.object_def.data_type);
    shader_src += "__kernel void bhwc_to_tensor(";
    shader_src += "__global " + in_data_type + "* src, $0) {\n";

    shader_src += R"(  int linear_id = get_global_id(0);
  int x = linear_id / args.tensor.Batch();
  int b = linear_id % args.tensor.Batch();
  int y = get_global_id(1);
  int d = get_global_id(2);

  if (x >= args.tensor.Width() || y >= args.tensor.Height() || d >= args.tensor.Slices()) return;
)";
    shader_src += "  " + out_data_type + "4 result;\n";
    shader_src += R"(  int c = d * 4;
  int index = ((b * args.tensor.Height() + y) * args.tensor.Width() + x) * args.tensor.Channels() + c;
  result.x = src[index];
  result.y = c + 1 < args.tensor.Channels() ? src[index + 1] : 1;
  result.z = c + 2 < args.tensor.Channels() ? src[index + 2] : 2;
  result.w = c + 3 < args.tensor.Channels() ? src[index + 3] : 3;
)";
    shader_src += "  args.tensor.Write(result, x, y, d, b);\n}";
    queue_ = environment->queue();
    context_ = &environment->context();
    shape_ = BHWC(output_def.dimensions.b, output_def.dimensions.h,
                  output_def.dimensions.w, output_def.dimensions.c);
    RETURN_IF_ERROR(
        args.Compile(environment->device().GetInfo(), {}, &shader_src));
    RETURN_IF_ERROR(cl_args_.Init(environment->device().GetInfo(), nullptr,
                                  &args, &shader_src));
    return environment->program_cache()->GetOrCreateCLKernel(
        shader_src, "bhwc_to_tensor", environment->context(),
        environment->device(), &kernel_);
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_14(mht_14_v, 578, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "Convert");

    auto input = absl::get_if<OpenClBuffer>(&input_obj);
    if (!input || !input->memobj) {
      return absl::InvalidArgumentError(
          "Missing input in bhwc_to_tensor converter");
    }
    cl_mem out_memory;
    RETURN_IF_ERROR(GetOpenCLMemory(output_obj, &out_memory));
    Tensor tensor;
    RETURN_IF_ERROR(CreateSharedTensor(*context_, out_memory, shape_,
                                       tensor_descriptor_, &tensor));
    return DispatchKernel(input->memobj, &tensor);
  }
};

std::array<size_t, 3> CalculateTextureRegion(const TensorObjectDef& def) {
  const auto& dims = def.dimensions;
  std::array<size_t, 3> region = {0, 0, 1};
  switch (ToTensorStorageType(def.object_def.object_type,
                              def.object_def.data_layout)) {
    case TensorStorageType::SINGLE_TEXTURE_2D:
      region[0] = static_cast<size_t>(dims.w * dims.b);
      region[1] = static_cast<size_t>(dims.h);
      break;
    case TensorStorageType::TEXTURE_2D:
      region[0] = static_cast<size_t>(dims.w * dims.b);
      region[1] = static_cast<size_t>(dims.h * dims.d());
      break;
    case TensorStorageType::TEXTURE_ARRAY:
      region[0] = static_cast<size_t>(dims.w * dims.b);
      region[1] = static_cast<size_t>(dims.h);
      region[2] = static_cast<size_t>(dims.d());
      break;
    default:
      break;
  }
  return region;
}

bool IsOpenClTextureOrBuffer(ObjectType type) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_15(mht_15_v, 620, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "IsOpenClTextureOrBuffer");

  return type == ObjectType::OPENCL_BUFFER ||
         type == ObjectType::OPENCL_TEXTURE;
}

// Copies data from one object of the same type and layout to another object.
class TrivialCopier : public OpenClConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_16(mht_16_v, 631, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "IsSupported");

    return IsOpenClTextureOrBuffer(input.object_type) &&
           input.data_type == output.data_type &&
           input.object_type == output.object_type &&
           input.data_layout == output.data_layout;
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_17(mht_17_v, 643, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "Init");

    shape_ = BHWC(input_def.dimensions.b, input_def.dimensions.h,
                  input_def.dimensions.w, input_def.dimensions.c);
    data_type_ = input_def.object_def.data_type;
    queue_ = environment->queue();
    region_ = CalculateTextureRegion(output_def);
    return absl::OkStatus();
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_18(mht_18_v, 656, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "Convert");

    auto texture_input = absl::get_if<OpenClTexture>(&input_obj);
    auto texture_output = absl::get_if<OpenClTexture>(&output_obj);
    if (texture_input && texture_output) {
      return Copy(*texture_input, *texture_output);
    }
    auto buffer_input = absl::get_if<OpenClBuffer>(&input_obj);
    auto buffer_output = absl::get_if<OpenClBuffer>(&output_obj);
    if (buffer_input && buffer_output) {
      return Copy(*buffer_input, *buffer_output);
    }
    return absl::InternalError("Unexpected object");
  }

  absl::Status Copy(const OpenClBuffer& input, const OpenClBuffer& output) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_19(mht_19_v, 673, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "Copy");

    if (input.memobj == output.memobj) {
      return absl::OkStatus();
    }
    return GetOpenCLError(
        clEnqueueCopyBuffer(queue_->queue(), input.memobj, output.memobj, 0, 0,
                            SizeOf(data_type_) * shape_.w * shape_.h *
                                AlignByN(shape_.c, 4) * shape_.b,
                            0, nullptr, nullptr));
  }

  absl::Status Copy(const OpenClTexture& input, const OpenClTexture& output) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_20(mht_20_v, 687, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "Copy");

    if (input.memobj == output.memobj) {
      return absl::OkStatus();
    }
    size_t origin[3] = {0, 0, 0};
    return GetOpenCLError(
        clEnqueueCopyImage(queue_->queue(), input.memobj, output.memobj, origin,
                           origin, region_.data(), 0, nullptr, nullptr));
  }

 private:
  DataType data_type_ = DataType::UNKNOWN;
  std::array<size_t, 3> region_;
};

// Copies data from/to CPU into a tensor.
class CpuCopier : public OpenClConverterImpl {
 public:
  explicit CpuCopier(bool asynchronous = false) : async_(asynchronous) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_21(mht_21_v, 708, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "CpuCopier");
}
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_22(mht_22_v, 712, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "IsSupported");

    return input.data_type == output.data_type &&
           input.data_layout == output.data_layout &&
           ((input.object_type == ObjectType::CPU_MEMORY &&
             IsOpenClTextureOrBuffer(output.object_type)) ||
            (output.object_type == ObjectType::CPU_MEMORY &&
             IsOpenClTextureOrBuffer(input.object_type)));
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_23(mht_23_v, 726, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "Init");

    region_ = CalculateTextureRegion(
        input_def.object_def.object_type == ObjectType::CPU_MEMORY ? output_def
                                                                   : input_def);
    queue_ = environment->queue();
    return absl::OkStatus();
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_24(mht_24_v, 738, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "Convert");

    auto cpu_input = absl::get_if<CpuMemory>(&input_obj);
    auto cpu_output = absl::get_if<CpuMemory>(&output_obj);
    if (cpu_input) {
      auto texture_output = absl::get_if<OpenClTexture>(&output_obj);
      if (texture_output) {
        return queue_->EnqueueWriteImage(
            texture_output->memobj, int3(region_[0], region_[1], region_[2]),
            cpu_input->data, async_);
      }
      auto buffer_output = absl::get_if<OpenClBuffer>(&output_obj);
      if (buffer_output) {
        return queue_->EnqueueWriteBuffer(buffer_output->memobj,
                                          cpu_input->size_bytes,
                                          cpu_input->data, async_);
      }
    } else if (cpu_output) {
      auto texture_input = absl::get_if<OpenClTexture>(&input_obj);
      if (texture_input) {
        return queue_->EnqueueReadImage(
            texture_input->memobj, int3(region_[0], region_[1], region_[2]),
            cpu_output->data, async_);
      }
      auto buffer_input = absl::get_if<OpenClBuffer>(&input_obj);
      if (buffer_input) {
        return queue_->EnqueueReadBuffer(buffer_input->memobj,
                                         cpu_output->size_bytes,
                                         cpu_output->data, async_);
      }
    }
    return absl::InternalError("Unexpected object");
  }

 private:
  std::array<size_t, 3> region_;
  bool async_;
};

class OpenClTensorConverterBuilder : public TensorObjectConverterBuilder {
 public:
  explicit OpenClTensorConverterBuilder(Environment* environment)
      : environment_(environment) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_25(mht_25_v, 782, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "OpenClTensorConverterBuilder");
}

  bool IsSupported(const TensorObjectDef& input,
                   const TensorObjectDef& output) const final {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_26(mht_26_v, 788, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "IsSupported");

    const auto& input_def = input.object_def;
    const auto& output_def = output.object_def;
    return input.dimensions == output.dimensions &&
           (TrivialCopier::IsSupported(input_def, output_def) ||
            TensorToTensorConverter::IsSupported(input_def, output_def) ||
            CpuCopier::IsSupported(input_def, output_def) ||
            TensorToBHWCBufferConverter::IsSupported(input_def, output_def) ||
            BHWCBufferToTensorConverter::IsSupported(input_def, output_def));
  }

  absl::Status MakeConverter(
      const TensorObjectDef& input, const TensorObjectDef& output,
      std::unique_ptr<TensorObjectConverter>* converter) final {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSkernelsPSconverterDTcc mht_27(mht_27_v, 804, "", "./tensorflow/lite/delegates/gpu/cl/kernels/converter.cc", "MakeConverter");

    std::unique_ptr<OpenClConverterImpl> impl;
    const auto& input_def = input.object_def;
    const auto& output_def = output.object_def;
    if (TrivialCopier::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<TrivialCopier>();
    } else if (TensorToTensorConverter::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<TensorToTensorConverter>();
    } else if (CpuCopier::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<CpuCopier>(/*asynchronous*/ true);
    } else if (TensorToBHWCBufferConverter::IsSupported(input_def,
                                                        output_def)) {
      impl = absl::make_unique<TensorToBHWCBufferConverter>();
    } else if (BHWCBufferToTensorConverter::IsSupported(input_def,
                                                        output_def)) {
      impl = absl::make_unique<BHWCBufferToTensorConverter>();
    } else {
      return absl::UnimplementedError("Unsupported conversion");
    }
    RETURN_IF_ERROR(impl->Init(input, output, environment_));
    impl->SetGpuInfo(environment_->GetDevicePtr()->GetInfo());
    *converter = std::move(impl);
    return absl::OkStatus();
  }

  Environment* environment_;
};

}  // namespace

std::unique_ptr<TensorObjectConverterBuilder> NewConverterBuilder(
    Environment* environment) {
  return absl::make_unique<OpenClTensorConverterBuilder>(environment);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
