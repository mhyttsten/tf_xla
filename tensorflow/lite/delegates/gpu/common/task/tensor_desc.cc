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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace {
std::string GetReadImageFromDataType(DataType data_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "GetReadImageFromDataType");

  if (data_type == DataType::FLOAT32) {
    return "read_imagef";
  } else if (data_type == DataType::FLOAT16) {
    return "read_imageh";
  } else if (data_type == DataType::INT8 || data_type == DataType::INT16 ||
             data_type == DataType::INT32) {
    return "read_imagei";
  } else if (data_type == DataType::UINT8 || data_type == DataType::UINT16 ||
             data_type == DataType::UINT32) {
    return "read_imageui";
  } else {
    return "error";
  }
}

DataType ToClTextureType(DataType data_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_1(mht_1_v, 219, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "ToClTextureType");

  switch (data_type) {
    case DataType::FLOAT32:
    case DataType::FLOAT16:
    case DataType::INT32:
    case DataType::UINT32:
      return data_type;
    case DataType::INT16:
    case DataType::INT8:
      return DataType::INT32;
    case DataType::UINT16:
    case DataType::UINT8:
      return DataType::UINT32;
    default:
      return DataType::UNKNOWN;
  }
}

std::string GetWriteImageFromDataType(DataType data_type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_2(mht_2_v, 240, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "GetWriteImageFromDataType");

  if (data_type == DataType::FLOAT32) {
    return "write_imagef";
  } else if (data_type == DataType::FLOAT16) {
    return "write_imageh";
  } else if (data_type == DataType::INT8 || data_type == DataType::INT16 ||
             data_type == DataType::INT32) {
    return "write_imagei";
  } else if (data_type == DataType::UINT8 || data_type == DataType::UINT16 ||
             data_type == DataType::UINT32) {
    return "write_imageui";
  } else {
    return "error";
  }
}

std::string AddressModeToCLSampler(AddressMode address_mode) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_3(mht_3_v, 259, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "AddressModeToCLSampler");

  switch (address_mode) {
    case AddressMode::kDontCare:
      return "smp_none";
    case AddressMode::kZero:
      return "smp_zero";
  }
}

std::string GetConvertionForImage(const GpuInfo& gpu_info, DataType src_type,
                                  DataType dst_type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_4(mht_4_v, 272, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "GetConvertionForImage");

  DataType interm_type = src_type;
  if (gpu_info.IsApiOpenCl()) {
    if (src_type == DataType::FLOAT16 && dst_type == DataType::FLOAT32) {
      return "";
    }
    interm_type = ToClTextureType(src_type);
  } else if (gpu_info.IsApiMetal()) {
    interm_type = ToMetalTextureType(src_type);
  }
  return GetTypeConvertion(gpu_info, interm_type, dst_type, 4);
}

std::string GetConvertion(const GpuInfo& gpu_info,
                          TensorStorageType storage_type, DataType src_type,
                          DataType dst_type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_5(mht_5_v, 290, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "GetConvertion");

  if (storage_type == TensorStorageType::BUFFER) {
    return GetTypeConvertion(gpu_info, src_type, dst_type, 4);
  } else {
    return GetConvertionForImage(gpu_info, src_type, dst_type);
  }
}

void MayBeAddConvertion(const std::string& conversion, std::string* result) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("conversion: \"" + conversion + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_6(mht_6_v, 302, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "MayBeAddConvertion");

  if (!conversion.empty()) {
    *result = conversion + "(" + *result + ")";
  }
}

}  // namespace

std::string ToString(TensorStorageType type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_7(mht_7_v, 313, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "ToString");

  switch (type) {
    case TensorStorageType::UNKNOWN:
      return "TensorStorageType::UNKNOWN";
    case TensorStorageType::BUFFER:
      return "TensorStorageType::BUFFER";
    case TensorStorageType::TEXTURE_ARRAY:
      return "TensorStorageType::TEXTURE_ARRAY";
    case TensorStorageType::TEXTURE_2D:
      return "TensorStorageType::TEXTURE_2D";
    case TensorStorageType::TEXTURE_3D:
      return "TensorStorageType::TEXTURE_3D";
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return "TensorStorageType::SINGLE_TEXTURE_2D";
    case TensorStorageType::IMAGE_BUFFER:
      return "TensorStorageType::IMAGE_BUFFER";
  }
}

TensorDescriptor::TensorDescriptor(TensorDescriptor&& desc)
    : GPUObjectDescriptor(std::move(desc)),
      data_type(desc.data_type),
      storage_type(desc.storage_type),
      layout(desc.layout),
      use_buffer_for_write_only_2d_texture(
          desc.use_buffer_for_write_only_2d_texture),
      use_buffer_for_write_only_image_buffer(
          desc.use_buffer_for_write_only_image_buffer),
      shape(desc.shape),
      data(std::move(desc.data)) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_8(mht_8_v, 345, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::TensorDescriptor");
}
TensorDescriptor& TensorDescriptor::operator=(TensorDescriptor&& desc) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_9(mht_9_v, 349, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "=");

  if (this != &desc) {
    std::swap(data_type, desc.data_type);
    std::swap(storage_type, desc.storage_type);
    std::swap(layout, desc.layout);
    std::swap(use_buffer_for_write_only_2d_texture,
              desc.use_buffer_for_write_only_2d_texture);
    std::swap(use_buffer_for_write_only_image_buffer,
              desc.use_buffer_for_write_only_image_buffer);
    std::swap(shape, desc.shape);
    data = std::move(desc.data);
    GPUObjectDescriptor::operator=(std::move(desc));
  }
  return *this;
}

GPUResources TensorDescriptor::GetGPUResources(const GpuInfo& gpu_info) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_10(mht_10_v, 368, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::GetGPUResources");

  GPUResources resources;
  resources.ints.push_back("slice_stride");
  if (HasAxis(Axis::WIDTH)) {
    resources.ints.push_back("width");
  }
  if (HasAxis(Axis::HEIGHT)) {
    resources.ints.push_back("height");
  }
  if (HasAxis(Axis::CHANNELS)) {
    resources.ints.push_back("slices");
    resources.ints.push_back("channels");
  }
  if (HasAxis(Axis::BATCH)) {
    resources.ints.push_back("batch");
  }
  if (HasAxis(Axis::DEPTH)) {
    resources.ints.push_back("depth");
  }
  if (storage_type == TensorStorageType::BUFFER) {
    GPUBufferDescriptor desc;
    desc.data_type = data_type;
    desc.access_type = access_type_;
    desc.element_size = 4;
    auto it1 = state_vars_.find("ElementsX2");
    if (it1 != state_vars_.end() && it1->second == "true") {
      desc.element_size = 8;
    }
    auto it2 = state_vars_.find("ElementsX4");
    if (it2 != state_vars_.end() && it2->second == "true") {
      desc.element_size = 16;
    }
    resources.buffers.push_back({"buffer", desc});
  } else if (storage_type == TensorStorageType::SINGLE_TEXTURE_2D ||
             storage_type == TensorStorageType::TEXTURE_2D) {
    if (access_type_ == AccessType::WRITE &&
        use_buffer_for_write_only_2d_texture) {
      resources.ints.push_back("aligned_texture_width");
      GPUBufferDescriptor desc;
      desc.data_type = data_type;
      desc.access_type = access_type_;
      desc.element_size = 4;
      resources.buffers.push_back({"buffer", desc});
    } else {
      GPUImage2DDescriptor desc;
      desc.data_type = data_type;
      desc.normalized = false;
      desc.access_type = access_type_;
      resources.images2d.push_back({"image2d", desc});
    }
  } else if (storage_type == TensorStorageType::TEXTURE_ARRAY) {
    GPUImage2DArrayDescriptor desc;
    desc.data_type = data_type;
    desc.access_type = access_type_;
    resources.image2d_arrays.push_back({"image2d_array", desc});
  } else if (storage_type == TensorStorageType::TEXTURE_3D) {
    GPUImage3DDescriptor desc;
    desc.data_type = data_type;
    desc.access_type = access_type_;
    resources.images3d.push_back({"image3d", desc});
  } else if (storage_type == TensorStorageType::IMAGE_BUFFER) {
    if (access_type_ == AccessType::WRITE &&
        use_buffer_for_write_only_image_buffer) {
      GPUBufferDescriptor desc;
      desc.data_type = data_type;
      desc.access_type = access_type_;
      desc.element_size = 4;
      resources.buffers.push_back({"buffer", desc});
    } else {
      GPUImageBufferDescriptor desc;
      desc.data_type = data_type;
      desc.access_type = access_type_;
      resources.image_buffers.push_back({"image_buffer", desc});
    }
  }
  return resources;
}

absl::Status TensorDescriptor::PerformConstExpr(const GpuInfo& gpu_info,
                                                const std::string& const_expr,
                                                std::string* result) const {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("const_expr: \"" + const_expr + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_11(mht_11_v, 452, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::PerformConstExpr");

  if (const_expr == "type" || const_expr == "scalar_type") {
    const int vec_size = const_expr == "scalar_type" ? 1 : 4;
    *result = GetTypeDeclaration(gpu_info, data_type, vec_size);
    return absl::OkStatus();
  } else if (const_expr == "zero_value" || const_expr == "scalar_zero_value") {
    const int vec_size = const_expr == "scalar_zero_value" ? 1 : 4;
    *result = GetZeroValue(gpu_info, data_type, vec_size);
    return absl::OkStatus();
  } else {
    return absl::UnimplementedError(
        absl::StrCat("Can not resolve constant expression - ", const_expr));
  }
}

absl::Status TensorDescriptor::PerformSelector(
    const GpuInfo& gpu_info, const std::string& selector,
    const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("selector: \"" + selector + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_12(mht_12_v, 474, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::PerformSelector");

  if (selector == "Width") {
    *result = "width";
    return absl::OkStatus();
  } else if (selector == "Height") {
    *result = "height";
    return absl::OkStatus();
  } else if (selector == "Slices") {
    *result = "slices";
    return absl::OkStatus();
  } else if (selector == "SliceStride") {
    *result = "slice_stride";
    return absl::OkStatus();
  } else if (selector == "Channels") {
    *result = "channels";
    return absl::OkStatus();
  } else if (selector == "Batch") {
    if (HasAxis(Axis::BATCH)) {
      *result = "batch";
    } else {
      *result = "1";
    }
    return absl::OkStatus();
  } else if (selector == "Depth") {
    *result = "depth";
    return absl::OkStatus();
  } else if (selector == "SetBatchRef") {
    if (args.size() != 1) {
      return absl::InvalidArgumentError(
          "Unsupported arguments in SetBatchRef selector");
    }
    state_vars_["batch_id"] = args[0];
    *result = "";
    return absl::OkStatus();
  } else if (selector == "Read") {
    return PerformReadSelector(gpu_info, args, template_args, result);
  } else if (selector == "ReadNearest") {
    return PerformReadNearestSelector(gpu_info, args, result);
  } else if (selector == "ReadBilinear") {
    return PerformReadBilinearSelector(gpu_info, args, result);
  } else if (selector == "ReadPerChannel") {
    return PerformReadPerChannelSelector(gpu_info, args, template_args, result);
  } else if (selector == "Write") {
    return PerformWriteSelector(gpu_info, args, template_args, result);
  } else if (selector == "WriteLinear") {
    return PerformWriteLinearSelector(gpu_info, args, template_args, result);
  } else if (selector == "Write2D") {
    return PerformWrite2DSelector(gpu_info, args, template_args, result);
  } else if (selector == "GetAddress") {
    return PerformGetAddressSelector(args, result);
  } else if (selector == "GetPtrWithSliceOffset") {
    return PerformGetPtrWithSliceOffsetSelector(args, result);
  } else if (selector == "GetWHOffset") {
    return PerformGetWHOffsetSelector(args, result);
  } else if (selector == "GetHandle") {
    return PerformGetHandleSelector(args, result);
  } else {
    return absl::NotFoundError(absl::StrCat(
        "TensorDescriptor don't have selector with name - ", selector));
  }
}

absl::Status TensorDescriptor::PerformReadSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_13(mht_13_v, 541, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::PerformReadSelector");

  DataType read_as_type = data_type;
  RETURN_IF_ERROR(
      MaybeGetDataTypeFromTemplateArgs(template_args, &read_as_type));
  if (args.size() == 1) {  // function overload for 1D linear types.
    if (storage_type == TensorStorageType::BUFFER ||
        storage_type == TensorStorageType::IMAGE_BUFFER) {
      *result = Read(gpu_info, read_as_type, {args[0]});
      return absl::OkStatus();
    } else {
      return absl::InvalidArgumentError(
          "Read selector with single argument can be used only with linear "
          "storage types(BUFFER or IMAGE_BUFFER)");
    }
  }
  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  bool parsed = ParseCoordsFromArgs(args, 0, &xc, &yc, &zc, &sc, &bc);
  if (args.size() < 2 || !parsed) {
    return absl::NotFoundError("Unrecognized Read selector");
  }

  *result = Read(gpu_info, read_as_type, GetPhysicalCoords(xc, yc, zc, sc, bc));
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformReadNearestSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_14(mht_14_v, 575, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::PerformReadNearestSelector");

  if (IsBatchedWidth()) {
    return absl::NotFoundError(
        "ReadNearest can not be used with BatchedWidth.");
  }
  // ReadNearest(result, fc_x, fc_y, {fc_z}, slice);
  if (!((args.size() == 5 && HasAxis(Axis::DEPTH)) || args.size() == 4)) {
    return absl::NotFoundError("Unrecognized ReadNearest selector");
  }
  std::vector<std::string> coord_args =
      std::vector<std::string>(args.begin() + 1, args.end());
  std::string c;
  c += "  {\n";
  c += "  int coord_x_TMP = INIT_INT(" + coord_args[0] + ");\n";
  c += "  coord_x_TMP = max(coord_x_TMP, 0);\n";
  c += "  coord_x_TMP = min(coord_x_TMP, width - 1);\n";
  coord_args[0] = "coord_x_TMP";
  c += "  int coord_y_TMP = INIT_INT(" + coord_args[1] + ");\n";
  c += "  coord_y_TMP = max(coord_y_TMP, 0);\n";
  c += "  coord_y_TMP = min(coord_y_TMP, height - 1);\n";
  coord_args[1] = "coord_y_TMP";
  if (HasAxis(Axis::DEPTH)) {
    c += "  int coord_z_TMP = INIT_INT(" + coord_args[2] + ");\n";
    c += "  coord_z_TMP = max(coord_z_TMP, 0);\n";
    c += "  coord_z_TMP = min(coord_z_TMP, depth - 1);\n";
    coord_args[2] = "coord_z_TMP";
  }
  std::string src_value;
  RETURN_IF_ERROR(PerformReadSelector(gpu_info, coord_args, {}, &src_value));
  c += "  " + args[0] + " = " + src_value + ";\n";
  c += "  }";
  *result = c;
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformReadBilinearSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_15(mht_15_v, 615, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::PerformReadBilinearSelector");

  if (IsBatchedWidth()) {
    return absl::NotFoundError(
        "ReadBilinear can not be used with BatchedWidth.");
  }
  // ReadBilinear(result, fc_x, fc_y, {fc_z}, slice);
  if (!((args.size() == 5 && HasAxis(Axis::DEPTH)) || args.size() == 4)) {
    return absl::NotFoundError("Unrecognized ReadBilinear selector");
  }
  std::vector<std::string> coord_args =
      std::vector<std::string>(args.begin() + 1, args.end());
  std::string c;
  c += "  {\n";
  c += "  float f_x_TMP = floor(" + coord_args[0] + ");\n";
  c += "  float x_scale_TMP = (" + coord_args[0] + ") - f_x_TMP;\n";
  c += "  int i_x_TMP = INIT_INT(f_x_TMP);\n";
  c += "  int start_x_TMP = max(i_x_TMP, 0);\n";
  c += "  int end_x_TMP = min(i_x_TMP + 1, width - 1);\n";
  c += "  float f_y_TMP = floor(" + coord_args[1] + ");\n";
  c += "  float y_scale_TMP = (" + coord_args[1] + ") - f_y_TMP;\n";
  c += "  int i_y_TMP = INIT_INT(f_y_TMP);\n";
  c += "  int start_y_TMP = max(i_y_TMP, 0);\n";
  c += "  int end_y_TMP = min(i_y_TMP + 1, height - 1);\n";
  if (HasAxis(Axis::DEPTH)) {
    // 3d bilinear read, x, y, z
    c += "  float f_z_TMP = floor(" + coord_args[2] + ");\n";
    c += "  float z_scale_TMP = (" + coord_args[2] + ") - f_z_TMP;\n";
    c += "  int i_z_TMP = INIT_INT(f_z_TMP);\n";
    c += "  int start_z_TMP = max(i_z_TMP, 0);\n";
    c += "  int end_z_TMP = min(i_z_TMP + 1, depth - 1);\n";
    int index = 0;
    for (const auto& src_z : {"start_z_TMP", "end_z_TMP"}) {
      for (const auto& src_y : {"start_y_TMP", "end_y_TMP"}) {
        for (const auto& src_x : {"start_x_TMP", "end_x_TMP"}) {
          coord_args[0] = src_x;
          coord_args[1] = src_y;
          coord_args[2] = src_z;
          std::string src_value;
          RETURN_IF_ERROR(
              PerformReadSelector(gpu_info, coord_args, {"float"}, &src_value));
          c += "  float4 src" + std::to_string(index) + "_TMP = " + src_value +
               ";\n";
          index++;
        }
      }
    }
    c += "  float4 t0_TMP = mix(mix(src0_TMP, src1_TMP, x_scale_TMP), "
         "mix(src2_TMP, src3_TMP, x_scale_TMP), y_scale_TMP);\n";
    c += "  float4 t1_TMP = mix(mix(src4_TMP, src5_TMP, x_scale_TMP), "
         "mix(src6_TMP, src7_TMP, x_scale_TMP), y_scale_TMP);\n";
    c += "  " + args[0] + " = TO_FLT4(mix(t0_TMP, t1_TMP, z_scale_TMP));\n";
  } else {
    // 2d bilinear read, x, y
    int index = 0;
    for (const auto& src_y : {"start_y_TMP", "end_y_TMP"}) {
      for (const auto& src_x : {"start_x_TMP", "end_x_TMP"}) {
        coord_args[0] = src_x;
        coord_args[1] = src_y;
        std::string src_value;
        RETURN_IF_ERROR(
            PerformReadSelector(gpu_info, coord_args, {"float"}, &src_value));
        c += "  float4 src" + std::to_string(index) + "_TMP = " + src_value +
             ";\n";
        index++;
      }
    }
    c += "  " + args[0] +
         " = TO_FLT4(mix(mix(src0_TMP, src1_TMP, x_scale_TMP), mix(src2_TMP, "
         "src3_TMP, x_scale_TMP), y_scale_TMP));\n";
  }
  c += "  }";
  *result = c;
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformReadPerChannelSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_16(mht_16_v, 695, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::PerformReadPerChannelSelector");

  std::vector<std::string> coord_args =
      std::vector<std::string>(args.begin() + 1, args.end());
  int channels_index = 0;
  if (HasAxis(Axis::WIDTH)) {
    channels_index++;
  }
  if (HasAxis(Axis::HEIGHT)) {
    channels_index++;
  }
  if (HasAxis(Axis::DEPTH)) {
    channels_index++;
  }
  if (channels_index >= coord_args.size()) {
    std::cout << channels_index << " " << coord_args.size() << std::endl;
    return absl::NotFoundError(
        "Wrong number of coordinates in ReadPerChannel.");
  }
  std::string c = "  {\n";
  c += "  int slice_coord_TMP = (" + coord_args[channels_index] + ") / 4;\n";
  c += "  int sub_ch_coord_TMP = (" + coord_args[channels_index] + ") % 4;\n";
  coord_args[channels_index] = "slice_coord_TMP";
  std::string src_value;
  RETURN_IF_ERROR(
      PerformReadSelector(gpu_info, coord_args, template_args, &src_value));
  if (gpu_info.IsApiOpenCl()) {
    DataType dst_type = data_type;
    RETURN_IF_ERROR(MaybeGetDataTypeFromTemplateArgs(template_args, &dst_type));
    c += "  " + GetTypeDeclaration(gpu_info, dst_type, 4) +
         " src_TMP = " + src_value + ";\n";
    c +=
        "  " + args[0] + " = (" + ToCLDataType(dst_type, 1) +
        "[4]){src_TMP.x, src_TMP.y, src_TMP.z, src_TMP.w}[sub_ch_coord_TMP];\n";
  } else {
    c += "  " + args[0] + " = " + src_value + "[sub_ch_coord_TMP];\n";
  }

  c += "  }";
  *result = c;
  return absl::OkStatus();
}

absl::Status TensorDescriptor::GetLinkingContextFromWriteSelector(
    const std::vector<std::string>& args, std::string* value_name,
    std::string* x_coord, std::string* y_coord, std::string* s_coord) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_17(mht_17_v, 742, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::GetLinkingContextFromWriteSelector");

  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  bool parsed = ParseCoordsFromArgs(args, 1, &xc, &yc, &zc, &sc, &bc);
  if (args.size() < 2 || !parsed) {
    return absl::NotFoundError("Unrecognized Write selector");
  }
  *value_name = args[0];
  if (HasAxis(Axis::BATCH) && !IsBatchedWidth()) {
    *x_coord = absl::StrCat("((", xc, ") * batch + (", bc, "))");
  } else {
    *x_coord = absl::StrCat("(", xc, ")");
  }
  *y_coord = absl::StrCat("(", yc, ")");
  *s_coord = absl::StrCat("(", sc, ")");
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformWriteSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_18(mht_18_v, 768, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::PerformWriteSelector");

  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  bool parsed = ParseCoordsFromArgs(args, 1, &xc, &yc, &zc, &sc, &bc);
  if (args.size() < 2 || !parsed) {
    return absl::NotFoundError("Unrecognized Write selector");
  }
  DataType write_type = data_type;
  RETURN_IF_ERROR(MaybeGetDataTypeFromTemplateArgs(template_args, &write_type));
  *result = Write(gpu_info, write_type, args[0],
                  GetPhysicalCoords(xc, yc, zc, sc, bc));
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformWriteLinearSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_19(mht_19_v, 790, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::PerformWriteLinearSelector");

  if (storage_type != TensorStorageType::BUFFER &&
      storage_type != TensorStorageType::IMAGE_BUFFER) {
    return absl::InvalidArgumentError(
        "WriteLinear selector can be used only with linear "
        "storages(BUFFER/IMAGE_BUFFER)");
  }
  if (args.size() != 2) {
    return absl::NotFoundError("Unrecognized WriteLinear selector");
  }
  DataType write_type = data_type;
  RETURN_IF_ERROR(MaybeGetDataTypeFromTemplateArgs(template_args, &write_type));
  *result = Write(gpu_info, write_type, args[0], {args[1]});
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformWrite2DSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_20(mht_20_v, 811, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::PerformWrite2DSelector");

  if (storage_type != TensorStorageType::TEXTURE_2D) {
    return absl::InvalidArgumentError(
        "Write2D selector can be used only with 2d "
        "storages(TEXTURE_2D)");
  }
  if (args.size() != 3) {
    return absl::NotFoundError("Unrecognized Write2D selector");
  }
  DataType write_type = data_type;
  RETURN_IF_ERROR(MaybeGetDataTypeFromTemplateArgs(template_args, &write_type));
  *result = Write(gpu_info, write_type, args[0], {args[1], args[2]});
  return absl::OkStatus();
}

std::string TensorDescriptor::Read(
    const GpuInfo& gpu_info, DataType read_as_type,
    const std::vector<std::string>& coords) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_21(mht_21_v, 831, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::Read");

  const std::string conversion =
      GetConvertion(gpu_info, storage_type, data_type, read_as_type);
  if (gpu_info.IsApiOpenCl() &&
      !(data_type == DataType::FLOAT16 && read_as_type == DataType::FLOAT32)) {
    read_as_type = data_type;
  }
  switch (storage_type) {
    case TensorStorageType::BUFFER: {
      std::string result;
      if (gpu_info.IsGlsl() && data_type == DataType::FLOAT16 &&
          !gpu_info.IsGlslSupportsExplicitFp16()) {
        result =
            absl::StrCat("vec4(unpackHalf2x16(buffer[", coords[0],
                         "].x), unpackHalf2x16(buffer[", coords[0], "].y))");
      } else {
        result = absl::StrCat("buffer[", coords[0], "]");
      }
      MayBeAddConvertion(conversion, &result);
      return result;
    }
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D: {
      std::string result;
      if (gpu_info.IsApiOpenCl()) {
        result =
            absl::Substitute("$0(image2d, $1, (int2)($2, $3))",
                             GetReadImageFromDataType(read_as_type),
                             AddressModeToCLSampler(AddressModeFromState()),
                             coords[0], coords[1]);
      } else if (gpu_info.IsApiMetal()) {
        result = absl::Substitute("image2d.read(ushort2($0, $1))", coords[0],
                                  coords[1]);
      } else if (gpu_info.IsGlsl()) {
        result = "texelFetch(image2d, ivec2(" + coords[0] + ", " + coords[1] +
                 "), 0)";
        if (data_type == DataType::FLOAT16 &&
            gpu_info.IsGlslSupportsExplicitFp16()) {
          result = "f16vec4(" + result + ")";
        }
      }
      MayBeAddConvertion(conversion, &result);
      return result;
    }
    case TensorStorageType::TEXTURE_3D: {
      std::string result;
      if (gpu_info.IsApiOpenCl()) {
        result =
            absl::Substitute("$0(image3d, $1, (int4)($2, $3, $4, 0))",
                             GetReadImageFromDataType(read_as_type),
                             AddressModeToCLSampler(AddressModeFromState()),
                             coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsApiMetal()) {
        result = absl::Substitute("image3d.read(ushort3($0, $1, $2))",
                                  coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsGlsl()) {
        result = "texelFetch(image3d, ivec3(" + coords[0] + ", " + coords[1] +
                 ", " + coords[2] + "), 0)";
        if (data_type == DataType::FLOAT16 &&
            gpu_info.IsGlslSupportsExplicitFp16()) {
          result = "f16vec4(" + result + ")";
        }
      }
      MayBeAddConvertion(conversion, &result);
      return result;
    }
    case TensorStorageType::TEXTURE_ARRAY: {
      std::string result;
      if (gpu_info.IsApiOpenCl()) {
        result =
            absl::Substitute("$0(image2d_array, $1, (int4)($2, $3, $4, 0))",
                             GetReadImageFromDataType(read_as_type),
                             AddressModeToCLSampler(AddressModeFromState()),
                             coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsApiMetal()) {
        result = absl::Substitute("image2d_array.read(ushort2($0, $1), $2)",
                                  coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsGlsl()) {
        result = "texelFetch(image2d_array, ivec3(" + coords[0] + ", " +
                 coords[1] + ", " + coords[2] + "), 0)";
        if (data_type == DataType::FLOAT16 &&
            gpu_info.IsGlslSupportsExplicitFp16()) {
          result = "f16vec4(" + result + ")";
        }
      }
      MayBeAddConvertion(conversion, &result);
      return result;
    }
    case TensorStorageType::IMAGE_BUFFER: {
      std::string result;
      if (gpu_info.IsApiOpenCl()) {
        result = absl::StrCat(GetReadImageFromDataType(read_as_type),
                              "(image_buffer, ", coords[0], ")");
      } else if (gpu_info.IsApiMetal()) {
        result = absl::Substitute("image_buffer.read(uint($0))", coords[0]);
      } else if (gpu_info.IsGlsl()) {
        result = "texelFetch(image_buffer, " + coords[0] + ")";
        if (data_type == DataType::FLOAT16 &&
            gpu_info.IsGlslSupportsExplicitFp16()) {
          result = "f16vec4(" + result + ")";
        }
      }
      MayBeAddConvertion(conversion, &result);
      return result;
    }
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string TensorDescriptor::Write(
    const GpuInfo& gpu_info, DataType write_type, const std::string& var_name,
    const std::vector<std::string>& coords) const {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("var_name: \"" + var_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_22(mht_22_v, 947, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::Write");

  bool is_texture_write = storage_type == TensorStorageType::IMAGE_BUFFER ||
                          storage_type == TensorStorageType::TEXTURE_2D ||
                          storage_type == TensorStorageType::TEXTURE_ARRAY ||
                          storage_type == TensorStorageType::TEXTURE_3D;
  if (storage_type == TensorStorageType::IMAGE_BUFFER &&
      use_buffer_for_write_only_image_buffer) {
    is_texture_write = false;
  }
  if (storage_type == TensorStorageType::TEXTURE_2D &&
      use_buffer_for_write_only_2d_texture) {
    is_texture_write = false;
  }
  DataType write_required_type = data_type;
  if (is_texture_write) {
    if (gpu_info.IsApiOpenCl()) {
      write_required_type = ToClTextureType(data_type);
    } else if (gpu_info.IsApiMetal()) {
      write_required_type = ToMetalTextureType(data_type);
    }
  }
  std::string write_expr = var_name;
  if (write_type != write_required_type) {
    const std::string conversion =
        GetTypeConvertion(gpu_info, write_type, write_required_type, 4);
    if (!conversion.empty()) {
      write_expr = conversion + "(" + write_expr + ")";
    }
  }
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      if (gpu_info.IsApiOpenCl()) {
        if (use_buffer_for_write_only_image_buffer) {
          return absl::StrCat("buffer[", coords[0], "] = ", write_expr);
        } else {
          return absl::Substitute("$0(image_buffer, $1, $2)",
                                  GetWriteImageFromDataType(data_type),
                                  coords[0], write_expr);
        }
      } else if (gpu_info.IsApiMetal()) {
        if (use_buffer_for_write_only_image_buffer) {
          return absl::StrCat("buffer[", coords[0], "] = ", write_expr);
        } else {
          return absl::Substitute("image_buffer.write($0, uint($1))",
                                  write_expr, coords[0]);
        }
      } else if (gpu_info.IsGlsl()) {
        if (data_type == DataType::FLOAT16 &&
            !gpu_info.IsGlslSupportsExplicitFp16()) {
          return absl::StrCat("buffer[", coords[0], "] = uvec2(packHalf2x16(",
                              write_expr, ".xy), packHalf2x16(", write_expr,
                              ".zw))");
        } else {
          return absl::StrCat("buffer[", coords[0], "] = ", write_expr);
        }
      } else {
        return absl::StrCat("buffer[", coords[0], "] = ", write_expr);
      }
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_2D:
      if (gpu_info.IsApiOpenCl()) {
        if (use_buffer_for_write_only_2d_texture) {
          return absl::Substitute(
              "buffer[($2) * aligned_texture_width + ($1)] = $0", write_expr,
              coords[0], coords[1]);
        } else {
          return absl::Substitute("$0(image2d, (int2)($1, $2), $3)",
                                  GetWriteImageFromDataType(data_type),
                                  coords[0], coords[1], write_expr);
        }
      } else if (gpu_info.IsApiMetal()) {
        if (use_buffer_for_write_only_2d_texture) {
          return absl::Substitute(
              "buffer[($2) * aligned_texture_width + ($1)] = $0", write_expr,
              coords[0], coords[1]);
        } else {
          return absl::Substitute("image2d.write($0, ushort2($1, $2))",
                                  write_expr, coords[0], coords[1]);
        }
      } else if (gpu_info.IsGlsl()) {
        return absl::Substitute("imageStore(image2d, ivec2($0, $1), $2)",
                                coords[0], coords[1], write_expr);
      } else {
        return "";
      }
    case TensorStorageType::TEXTURE_3D:
      if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image3d, (int4)($1, $2, $3, 0), $4)",
                                GetWriteImageFromDataType(data_type), coords[0],
                                coords[1], coords[2], write_expr);
      } else if (gpu_info.IsApiMetal()) {
        return absl::Substitute("image3d.write($0, ushort3($1, $2, $3))",
                                write_expr, coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsGlsl()) {
        return absl::Substitute("imageStore(image3d, ivec3($0, $1, $2), $3)",
                                coords[0], coords[1], coords[2], write_expr);
      } else {
        return "";
      }
    case TensorStorageType::TEXTURE_ARRAY:
      if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image2d_array, (int4)($1, $2, $3, 0), $4)",
                                GetWriteImageFromDataType(data_type), coords[0],
                                coords[1], coords[2], write_expr);
      } else if (gpu_info.IsApiMetal()) {
        return absl::Substitute("image2d_array.write($0, ushort2($1, $2), $3)",
                                write_expr, coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsGlsl()) {
        return absl::Substitute(
            "imageStore(image2d_array, ivec3($0, $1, $2), $3)", coords[0],
            coords[1], coords[2], write_expr);
      } else {
        return "";
      }
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

absl::Status TensorDescriptor::PerformGetAddressSelector(
    const std::vector<std::string>& args, std::string* result) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_23(mht_23_v, 1071, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::PerformGetAddressSelector");

  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  bool parsed = ParseCoordsFromArgs(args, 1, &xc, &yc, &zc, &sc, &bc);
  if (args.size() < 3 || !parsed) {
    return absl::NotFoundError("Unrecognized GetAddress selector");
  }

  *result = DeclareAddress(args[0],
                           GetGlobalAddressNoDeclaration(xc, yc, zc, sc, bc));
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformGetPtrWithSliceOffsetSelector(
    const std::vector<std::string>& args, std::string* result) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_24(mht_24_v, 1091, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::PerformGetPtrWithSliceOffsetSelector");

  if (storage_type != TensorStorageType::BUFFER) {
    return absl::InvalidArgumentError(
        "GetPtrWithSliceOffset selector can be used only with BUFFER");
  }
  if (args.size() != 1) {
    return absl::NotFoundError(absl::StrCat(
        "GetPtrWithSliceOffset require one argument(slice coordinate), but ",
        args.size(), " was passed"));
  }
  *result = absl::StrCat("buffer + ", args[0], " * slice_stride");
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformGetWHOffsetSelector(
    const std::vector<std::string>& args, std::string* result) const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_25(mht_25_v, 1109, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::PerformGetWHOffsetSelector");

  if (storage_type != TensorStorageType::BUFFER &&
      storage_type != TensorStorageType::IMAGE_BUFFER) {
    return absl::InvalidArgumentError(
        "GetWHOffset selector can be used only with BUFFER/IMAGE_BUFFER");
  }
  if (args.size() != 2) {
    return absl::NotFoundError(absl::StrCat(
        "GetWHOffset require two arguments(X and Y coordinates), but ",
        args.size(), " was passed"));
  }
  if (HasAxis(Axis::BATCH) && !IsBatchedWidth()) {
    auto it = state_vars_.find("batch_id");
    std::string batch_id;
    if (it == state_vars_.end()) {
      return absl::NotFoundError(
          "Not found batch_id. Should be setted up by SetBatchRef(). method");
    } else {
      batch_id = it->second;
    }
    *result = absl::StrCat("((", args[1], ") * width + (", args[0],
                           ")) * batch + (", batch_id, ")");
  } else {
    *result = absl::StrCat("(", args[1], ") * width + (", args[0], ")");
  }
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformGetHandleSelector(
    const std::vector<std::string>& args, std::string* result) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_26(mht_26_v, 1141, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::PerformGetHandleSelector");

  if (!args.empty()) {
    return absl::NotFoundError(
        absl::StrCat("GetHandle does not require arguments, but ", args.size(),
                     " was passed"));
  }
  switch (storage_type) {
    case TensorStorageType::BUFFER:
      *result = "buffer";
      return absl::OkStatus();
    case TensorStorageType::IMAGE_BUFFER:
      if (access_type_ == AccessType::READ) {
        *result = "image_buffer";
      } else {
        *result = "buffer";
      }
      return absl::OkStatus();
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      *result = "image2d";
      return absl::OkStatus();
    case TensorStorageType::TEXTURE_ARRAY:
      *result = "image2d_array";
      return absl::OkStatus();
    case TensorStorageType::TEXTURE_3D:
      *result = "image3d";
      return absl::OkStatus();
    case TensorStorageType::UNKNOWN:
      return absl::UnavailableError("Unknown type");
  }
}

std::string TensorDescriptor::DeclareAddress(const std::string& var_name,
                                             const std::string& address) const {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("var_name: \"" + var_name + "\"");
   mht_27_v.push_back("address: \"" + address + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_27(mht_27_v, 1179, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::DeclareAddress");

  return absl::StrCat(StorageTypeToAddressType(), " ", var_name, " = ", address,
                      ";");
}

std::string TensorDescriptor::StorageTypeToAddressType() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_28(mht_28_v, 1187, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::StorageTypeToAddressType");

  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return "int";
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return "int2";
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return "int4";
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsWHS(
    const std::string& x, const std::string& y, const std::string& s) const {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("x: \"" + x + "\"");
   mht_29_v.push_back("y: \"" + y + "\"");
   mht_29_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_29(mht_29_v, 1210, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::GetPhysicalCoordsWHS");

  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return {
          absl::Substitute("((($2) * height + ($1)) * width + ($0))", x, y, s)};
    case TensorStorageType::TEXTURE_2D:
      return {absl::Substitute("($0)", x),
              absl::Substitute("(($0) * slices + ($1))", y, s)};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("($0)", x), absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("($0)", x), absl::Substitute("($0)", y),
              absl::Substitute("($0)", s)};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsWHSB(
    const std::string& x, const std::string& y, const std::string& s,
    const std::string& b) const {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("x: \"" + x + "\"");
   mht_30_v.push_back("y: \"" + y + "\"");
   mht_30_v.push_back("s: \"" + s + "\"");
   mht_30_v.push_back("b: \"" + b + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_30(mht_30_v, 1241, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::GetPhysicalCoordsWHSB");

  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return {absl::Substitute(
          "(((($3) * height + $2) * width + ($1)) * batch + ($0))", b, x, y,
          s)};
    case TensorStorageType::TEXTURE_2D:
      return {absl::Substitute("(($0) * batch + ($1))", x, b),
              absl::Substitute("(($0) * slices + ($1))", y, s)};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("(($0) * batch + ($1))", x, b),
              absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("(($0) * batch + ($1))", x, b),
              absl::Substitute("($0)", y), absl::Substitute("($0)", s)};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsWHDS(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& s) const {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("x: \"" + x + "\"");
   mht_31_v.push_back("y: \"" + y + "\"");
   mht_31_v.push_back("z: \"" + z + "\"");
   mht_31_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_31(mht_31_v, 1274, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::GetPhysicalCoordsWHDS");

  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return {absl::Substitute(
          "(((($3) * slices + ($2)) * height + ($1)) * width + ($0))", x, y, s,
          z)};
    case TensorStorageType::TEXTURE_2D:
      return {absl::Substitute("(($0) * depth + ($1))", x, z),
              absl::Substitute("(($0) * slices + ($1))", y, s)};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("(($0) * depth + ($1))", x, z),
              absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("($0)", x), absl::Substitute("($0)", y),
              absl::Substitute("(($0) * slices + ($1))", z, s)};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsWHDSB(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& s, const std::string& b) const {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("x: \"" + x + "\"");
   mht_32_v.push_back("y: \"" + y + "\"");
   mht_32_v.push_back("z: \"" + z + "\"");
   mht_32_v.push_back("s: \"" + s + "\"");
   mht_32_v.push_back("b: \"" + b + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_32(mht_32_v, 1308, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::GetPhysicalCoordsWHDSB");

  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return {absl::Substitute(
          "((((($4) * slices + ($3)) * height + $2) * width + ($1)) * batch + "
          "($0))",
          b, x, y, s, z)};
    case TensorStorageType::TEXTURE_2D:
      return {absl::Substitute("((($0)*batch + ($1))*depth + ($2))", x, b, z),
              absl::Substitute("(($0) * slices + ($1))", y, s)};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("((($0)*batch + ($1))*depth + ($2))", x, b, z),
              absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("(($0) * batch + ($1))", x, b),
              absl::Substitute("($0)", y),
              absl::Substitute("(($0) * slices + ($1))", z, s)};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::string TensorDescriptor::GetGlobalAddressNoDeclaration(
    const std::string& xc, const std::string& yc, const std::string& zc,
    const std::string& sc, const std::string& bc) const {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("xc: \"" + xc + "\"");
   mht_33_v.push_back("yc: \"" + yc + "\"");
   mht_33_v.push_back("zc: \"" + zc + "\"");
   mht_33_v.push_back("sc: \"" + sc + "\"");
   mht_33_v.push_back("bc: \"" + bc + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_33(mht_33_v, 1344, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::GetGlobalAddressNoDeclaration");

  auto coords = GetPhysicalCoords(xc, yc, zc, sc, bc);
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER: {
      return coords[0];
    }
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::Substitute("(int2)($0, $1)", coords[0], coords[1]);
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return absl::Substitute("(int4)($0, $1, $2, 0)", coords[0], coords[1],
                              coords[2]);
    case TensorStorageType::UNKNOWN:
      return "error";
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoords(
    const std::string& xc, const std::string& yc, const std::string& zc,
    const std::string& sc, const std::string& bc) const {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("xc: \"" + xc + "\"");
   mht_34_v.push_back("yc: \"" + yc + "\"");
   mht_34_v.push_back("zc: \"" + zc + "\"");
   mht_34_v.push_back("sc: \"" + sc + "\"");
   mht_34_v.push_back("bc: \"" + bc + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_34(mht_34_v, 1373, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::GetPhysicalCoords");

  if (layout == Layout::HWC || (IsBatchedWidth() && layout == Layout::BHWC)) {
    return GetPhysicalCoordsWHS(xc, yc, sc);
  } else if (layout == Layout::BHWC) {
    return GetPhysicalCoordsWHSB(xc, yc, sc, bc);
  } else if (layout == Layout::HWDC ||
             (IsBatchedWidth() && layout == Layout::BHWDC)) {
    return GetPhysicalCoordsWHDS(xc, yc, zc, sc);
  } else if (layout == Layout::BHWDC) {
    return GetPhysicalCoordsWHDSB(xc, yc, zc, sc, bc);
  } else {
    return {""};
  }
}

absl::Status TensorDescriptor::MaybeGetDataTypeFromTemplateArgs(
    const std::vector<std::string>& template_args, DataType* result) const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_35(mht_35_v, 1392, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::MaybeGetDataTypeFromTemplateArgs");

  for (const auto& template_arg : template_args) {
    std::string read_type = template_arg;
    if (read_type == "FLT" || read_type == "ACCUM_FLT") {
      auto it = state_vars_.find(read_type);
      if (it == state_vars_.end()) {
        return absl::UnavailableError(
            absl::StrCat("Template argument ", read_type, " uninitialized."));
      } else {
        read_type = it->second;
      }
    }

    if (read_type == "half") {
      *result = DataType::FLOAT16;
      return absl::OkStatus();
    } else if (read_type == "float") {
      *result = DataType::FLOAT32;
      return absl::OkStatus();
    } else if (read_type == "int") {
      *result = DataType::INT32;
      return absl::OkStatus();
    } else if (read_type == "short") {
      *result = DataType::INT16;
      return absl::OkStatus();
    } else if (read_type == "char") {
      *result = DataType::INT8;
      return absl::OkStatus();
    } else if (read_type == "uint") {
      *result = DataType::UINT32;
      return absl::OkStatus();
    } else if (read_type == "ushort") {
      *result = DataType::UINT16;
      return absl::OkStatus();
    } else if (read_type == "uchar") {
      *result = DataType::UINT8;
      return absl::OkStatus();
    }
  }
  return absl::OkStatus();
}

bool TensorDescriptor::HasAxis(Axis axis) const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_36(mht_36_v, 1437, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::HasAxis");

  if (axis == Axis::WIDTH || axis == Axis::HEIGHT || axis == Axis::CHANNELS) {
    return true;
  }
  if (axis == Axis::BATCH &&
      (layout == Layout::BHWC || layout == Layout::BHWDC)) {
    return true;
  }
  if (axis == Axis::DEPTH &&
      (layout == Layout::HWDC || layout == Layout::BHWDC)) {
    return true;
  }
  return false;
}

int TensorDescriptor::GetWidthSize(BHWDC shape) const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_37(mht_37_v, 1455, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::GetWidthSize");

  int width = shape.w;
  auto it = state_vars_.find("BatchedWidth");
  if (it != state_vars_.end() && it->second == "true") {
    width *= shape.b;
  }
  auto it1 = state_vars_.find("ElementsX2");
  if (it1 != state_vars_.end() && it1->second == "true") {
    width /= 2;
  }
  auto it2 = state_vars_.find("ElementsX4");
  if (it2 != state_vars_.end() && it2->second == "true") {
    width /= 4;
  }
  return width;
}

int TensorDescriptor::GetSliceStrideSize(BHWDC shape) const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_38(mht_38_v, 1475, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::GetSliceStrideSize");

  if (IsBatchedWidth()) {
    return GetWidthSize(shape) * shape.h;
  } else {
    if (HasAxis(Axis::BATCH)) {
      return GetWidthSize(shape) * shape.h * shape.b;
    } else {
      return GetWidthSize(shape) * shape.h;
    }
  }
}

void TensorDescriptor::SetAddressMode(AddressMode mode) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_39(mht_39_v, 1490, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::SetAddressMode");

  if (mode == AddressMode::kZero) {
    state_vars_["TextureMode"] = "ZERO";
  } else {
    state_vars_["TextureMode"] = "DONT_CARE";
  }
}

bool TensorDescriptor::ParseCoordsFromArgs(const std::vector<std::string>& args,
                                           int offset, std::string* xc,
                                           std::string* yc, std::string* zc,
                                           std::string* sc,
                                           std::string* bc) const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_40(mht_40_v, 1505, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::ParseCoordsFromArgs");

  if (HasAxis(Axis::WIDTH)) {
    if (offset >= args.size()) return false;
    *xc = args[offset++];
  }
  if (HasAxis(Axis::HEIGHT)) {
    if (offset >= args.size()) return false;
    *yc = args[offset++];
  }
  if (HasAxis(Axis::DEPTH)) {
    if (offset >= args.size()) return false;
    *zc = args[offset++];
  }
  if (HasAxis(Axis::CHANNELS)) {
    if (offset >= args.size()) return false;
    *sc = args[offset++];
  }
  if (HasAxis(Axis::BATCH) && !IsBatchedWidth()) {
    if (offset >= args.size()) {
      auto it = state_vars_.find("batch_id");
      if (it == state_vars_.end()) {
        return false;
      } else {
        *bc = it->second;
      }
    } else {
      *bc = args[offset++];
    }
  }
  return true;
}

bool TensorDescriptor::IsBatchedWidth() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_41(mht_41_v, 1540, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::IsBatchedWidth");

  auto it = state_vars_.find("BatchedWidth");
  return it != state_vars_.end() && it->second == "true";
}

AddressMode TensorDescriptor::AddressModeFromState() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_42(mht_42_v, 1548, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::AddressModeFromState");

  auto it = state_vars_.find("TextureMode");
  if (it != state_vars_.end()) {
    if (it->second == "ZERO") {
      return AddressMode::kZero;
    } else {
      return AddressMode::kDontCare;
    }
  } else {
    return AddressMode::kDontCare;
  }
}

size_t TensorDescriptor::GetSizeInBytesForShape(const BHWDC& shape5d) const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_43(mht_43_v, 1564, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::GetSizeInBytesForShape");

  int aligned_channels = storage_type == TensorStorageType::SINGLE_TEXTURE_2D
                             ? shape5d.c
                             : AlignByN(shape5d.c, 4);
  int elements_count =
      shape5d.b * shape5d.w * shape5d.h * shape5d.d * aligned_channels;
  return elements_count * SizeOf(data_type);
}

int TensorDescriptor::GetLinearIndex(const BHWDC& shape5d, int b, int x, int y,
                                     int d, int s, int sub_c) const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_44(mht_44_v, 1577, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::GetLinearIndex");

  const int slices = DivideRoundUp(shape5d.c, 4);
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return ((((d * slices + s) * shape5d.h + y) * shape5d.w + x) * shape5d.b +
              b) *
                 4 +
             sub_c;  // DSHWBC4
    case TensorStorageType::TEXTURE_2D:
      return ((((y * slices + s) * shape5d.w + x) * shape5d.b + b) * shape5d.d +
              d) *
                 4 +
             sub_c;  // HSWBDC4
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return (((y * shape5d.w + x) * shape5d.b + b) * shape5d.d + d) *
                 shape5d.c +
             sub_c;  // HWBDC
    case TensorStorageType::UNKNOWN:
      return -1;
  }
}

void TensorDescriptor::UploadData(
    const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& src) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_45(mht_45_v, 1606, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::UploadData");

  shape = BHWDC(1, src.shape.h, src.shape.w, 1, src.shape.c);
  UploadData(src.data.data());
}

void TensorDescriptor::UploadData(
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_46(mht_46_v, 1615, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::UploadData");

  shape = BHWDC(1, 1, 1, 1, src.shape.v);
  UploadData(src.data.data());
}

bool TensorDescriptor::SupportsZeroClamp(const Axis& axis) const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_47(mht_47_v, 1623, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::SupportsZeroClamp");

  switch (storage_type) {
    case TensorStorageType::UNKNOWN:
      return false;
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return false;
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return axis == Axis::WIDTH || axis == Axis::HEIGHT;
    case TensorStorageType::TEXTURE_3D:
      return axis == Axis::WIDTH || axis == Axis::HEIGHT || axis == Axis::DEPTH;
  }
}

bool TensorDescriptor::CanReadOutOfBorder(const Axis& axis) const {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_48(mht_48_v, 1642, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::CanReadOutOfBorder");

  switch (storage_type) {
    case TensorStorageType::UNKNOWN:
      return false;
    case TensorStorageType::BUFFER:
      return false;
    case TensorStorageType::IMAGE_BUFFER:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_ARRAY:
      return true;
  }
}

bool TensorDescriptor::IsLinear() const {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_49(mht_49_v, 1660, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::IsLinear");

  return storage_type == TensorStorageType::BUFFER ||
         storage_type == TensorStorageType::IMAGE_BUFFER;
}

bool TensorDescriptor::ReturnsZeroForNegOneRead() const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_50(mht_50_v, 1668, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::ReturnsZeroForNegOneRead");

  return storage_type == TensorStorageType::IMAGE_BUFFER;
}

absl::Status TensorDescriptor::CanCreateTensorWithShape(
    const GpuInfo& gpu_info, const BHWDC& shape) const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_51(mht_51_v, 1676, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::CanCreateTensorWithShape");

  const int slices = DivideRoundUp(shape.c, 4);
  const uint64_t allocation_size = GetSizeInBytesForShape(shape);
  const std::string common_desc = "Shape - " + ToString(shape) +
                                  ", data type - " + ToString(data_type) + ".";
  if (allocation_size > gpu_info.GetMaxMemoryAllocationSize()) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "Requested allocation size - ", allocation_size,
        " bytes. Max allocation size for this GPU - ",
        gpu_info.GetMaxMemoryAllocationSize(), " bytes. ", common_desc));
  }
  switch (storage_type) {
    case TensorStorageType::BUFFER: {
      if (allocation_size > gpu_info.GetMaxBufferSize()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Buffer with size - ", allocation_size,
            " bytes can not be created. Max buffer size for this GPU - ",
            gpu_info.GetMaxBufferSize(), " bytes. ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::IMAGE_BUFFER: {
      const uint64_t element_size = 4 * SizeOf(data_type);
      const uint64_t image_width = allocation_size / element_size;
      if (image_width > gpu_info.GetMaxImageBufferWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image buffer with width - ", image_width,
            " can not be created. Max image buffer width for this GPU - ",
            gpu_info.GetMaxImageBufferWidth(), ". ", common_desc));
      } else if (allocation_size > gpu_info.GetMaxBufferSize()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Buffer with size - ", allocation_size,
            " bytes can not be created. Max buffer size for this GPU - ",
            gpu_info.GetMaxBufferSize(), " bytes. ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::TEXTURE_3D: {
      if (gpu_info.IsApiOpenCl() &&
          gpu_info.opencl_info.cl_version < OpenClVersion::kCl1_2 &&
          slices == 1) {
        return absl::InternalError(
            "clCreateImage3D (that used in CL 1.0/1.1) can not create image "
            "with depth = 1 by specification.");
      }
      const int image_width = shape.w * shape.b;
      const int image_height = shape.h;
      const int image_depth = slices * shape.d;
      if (image_width > gpu_info.GetMaxImage3DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image3D with width - ", image_width,
            " can not be created. Max Image3D width for this GPU - ",
            gpu_info.GetMaxImage3DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage3DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image3D with height - ", image_height,
            " can not be created. Max Image3D height for this GPU - ",
            gpu_info.GetMaxImage3DHeight(), ". ", common_desc));
      } else if (image_depth > gpu_info.GetMaxImage3DDepth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image3D with depth - ", image_depth,
            " can not be created. Max Image3D depth for this GPU - ",
            gpu_info.GetMaxImage3DDepth(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::TEXTURE_ARRAY: {
      // Bug on some Adreno. b/131099086
      if (gpu_info.IsApiOpenCl() && slices == 1 && gpu_info.IsAdreno() &&
          !gpu_info.adreno_info.support_one_layer_texture_array) {
        return absl::InternalError(
            "Image2DArray with layer = 1 works incorrect on some Adreno in "
            "OpenCL. Can not be created.");
      }
      const int image_width = shape.w * shape.b;
      const int image_height = shape.h;
      const int image_layers = slices * shape.d;
      if (image_width > gpu_info.GetMaxImage2DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2DArray with width - ", image_width,
            " can not be created. Max Image2DArray width for this GPU - ",
            gpu_info.GetMaxImage2DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage2DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2DArray with height - ", image_height,
            " can not be created. Max Image2DArray height for this GPU - ",
            gpu_info.GetMaxImage2DHeight(), ". ", common_desc));
      } else if (image_layers > gpu_info.GetMaxImage2DArrayLayers()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2DArray with layers - ", image_layers,
            " can not be created. Max Image2DArray layers for this GPU - ",
            gpu_info.GetMaxImage2DArrayLayers(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::TEXTURE_2D: {
      const int image_width = shape.w * shape.b * shape.d;
      const int image_height = shape.h * slices;
      if (image_width > gpu_info.GetMaxImage2DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with width - ", image_width,
            " can not be created. Max Image2D width for this GPU - ",
            gpu_info.GetMaxImage2DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage2DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with height - ", image_height,
            " can not be created. Max Image2D height for this GPU - ",
            gpu_info.GetMaxImage2DHeight(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::SINGLE_TEXTURE_2D: {
      const int image_width = shape.w * shape.b * shape.d;
      const int image_height = shape.h;
      if (shape.c > 4) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with channels - ", shape.c, " can not be created."));
      } else if (!gpu_info.SupportsFloatImage2D(data_type, shape.c)) {
        return absl::ResourceExhaustedError(
            "Image2D doesn't support this pixel layout.");
      } else if (image_width > gpu_info.GetMaxImage2DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with width - ", image_width,
            " can not be created. Max Image2D width for this GPU - ",
            gpu_info.GetMaxImage2DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage2DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with height - ", image_height,
            " can not be created. Max Image2D height for this GPU - ",
            gpu_info.GetMaxImage2DHeight(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    default:
      return absl::UnimplementedError(
          "Can not create resources for unknown storage type.");
  }
}

absl::Status TensorDescriptor::CanCreateTensorWithShape(
    const GpuInfo& gpu_info, const BHWC& shape) const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTcc mht_52(mht_52_v, 1825, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.cc", "TensorDescriptor::CanCreateTensorWithShape");

  const BHWDC shape5D(shape.b, shape.h, shape.w, 1, shape.c);
  return CanCreateTensorWithShape(gpu_info, shape5D);
}
}  // namespace gpu
}  // namespace tflite
