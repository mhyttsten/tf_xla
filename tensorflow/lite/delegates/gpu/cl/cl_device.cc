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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"

namespace tflite {
namespace gpu {
namespace cl {

void ParseQualcommOpenClCompilerVersion(
    const std::string& cl_driver_version,
    AdrenoInfo::OpenClCompilerVersion* result) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("cl_driver_version: \"" + cl_driver_version + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc mht_0(mht_0_v, 208, "", "./tensorflow/lite/delegates/gpu/cl/cl_device.cc", "ParseQualcommOpenClCompilerVersion");

  // Searching this part: "Compiler E031.**.**.**" where * is digit
  const std::string start = "Compiler E031.";
  size_t position = cl_driver_version.find(start);
  if (position == std::string::npos) {
    return;
  }
  const size_t main_part_length = 8;  // main part is **.**.**
  if (position + start.length() + main_part_length >
      cl_driver_version.length()) {
    return;
  }

  const std::string main_part =
      cl_driver_version.substr(position + start.length(), main_part_length);
  if (!absl::ascii_isdigit(main_part[0]) ||
      !absl::ascii_isdigit(main_part[1]) || main_part[2] != '.' ||
      !absl::ascii_isdigit(main_part[3]) ||
      !absl::ascii_isdigit(main_part[4]) || main_part[5] != '.' ||
      !absl::ascii_isdigit(main_part[6]) ||
      !absl::ascii_isdigit(main_part[7])) {
    return;
  }
  result->major = (main_part[0] - '0') * 10 + (main_part[1] - '0');
  result->minor = (main_part[3] - '0') * 10 + (main_part[4] - '0');
  result->patch = (main_part[6] - '0') * 10 + (main_part[7] - '0');
}

template <>
std::string GetDeviceInfo<std::string>(cl_device_id id, cl_device_info info) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc mht_1(mht_1_v, 240, "", "./tensorflow/lite/delegates/gpu/cl/cl_device.cc", "GetDeviceInfo<std::string>");

  size_t size;
  cl_int error = clGetDeviceInfo(id, info, 0, nullptr, &size);
  if (error != CL_SUCCESS) {
    return "";
  }

  std::string result(size - 1, 0);
  error = clGetDeviceInfo(id, info, size, &result[0], nullptr);
  if (error != CL_SUCCESS) {
    return "";
  }
  return result;
}

namespace {
template <typename T>
T GetPlatformInfo(cl_platform_id id, cl_platform_info info) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc mht_2(mht_2_v, 260, "", "./tensorflow/lite/delegates/gpu/cl/cl_device.cc", "GetPlatformInfo");

  T result;
  cl_int error = clGetPlatformInfo(id, info, sizeof(T), &result, nullptr);
  if (error != CL_SUCCESS) {
    return -1;
  }
  return result;
}

std::string GetPlatformInfo(cl_platform_id id, cl_platform_info info) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc mht_3(mht_3_v, 272, "", "./tensorflow/lite/delegates/gpu/cl/cl_device.cc", "GetPlatformInfo");

  size_t size;
  cl_int error = clGetPlatformInfo(id, info, 0, nullptr, &size);
  if (error != CL_SUCCESS) {
    return "";
  }

  std::string result(size - 1, 0);
  error = clGetPlatformInfo(id, info, size, &result[0], nullptr);
  if (error != CL_SUCCESS) {
    return "";
  }
  return result;
}

void GetDeviceWorkDimsSizes(cl_device_id id, int3* result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc mht_4(mht_4_v, 290, "", "./tensorflow/lite/delegates/gpu/cl/cl_device.cc", "GetDeviceWorkDimsSizes");

  int dims_count =
      GetDeviceInfo<cl_uint>(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
  if (dims_count < 3) {
    return;
  }
  std::vector<size_t> limits(dims_count);
  cl_int error =
      clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                      sizeof(size_t) * dims_count, limits.data(), nullptr);
  if (error != CL_SUCCESS) {
    return;
  }
  // dims_count must be at least 3 according to spec
  result->x = limits[0];
  result->y = limits[1];
  result->z = limits[2];
}

OpenClVersion ParseCLVersion(const std::string& version) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("version: \"" + version + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc mht_5(mht_5_v, 313, "", "./tensorflow/lite/delegates/gpu/cl/cl_device.cc", "ParseCLVersion");

  const auto first_dot_pos = version.find_first_of('.');
  if (first_dot_pos == std::string::npos) {
    return OpenClVersion::kCl1_0;
  }
  const int major = version[first_dot_pos - 1] - '0';
  const int minor = version[first_dot_pos + 1] - '0';

  if (major == 1) {
    if (minor == 2) {
      return OpenClVersion::kCl1_2;
    } else if (minor == 1) {
      return OpenClVersion::kCl1_1;
    } else {
      return OpenClVersion::kCl1_0;
    }
  } else if (major == 2) {
    if (minor == 2) {
      return OpenClVersion::kCl2_2;
    } else if (minor == 1) {
      return OpenClVersion::kCl2_1;
    } else {
      return OpenClVersion::kCl2_0;
    }
  } else if (major == 3) {
    return OpenClVersion::kCl3_0;
  } else {
    return OpenClVersion::kCl1_0;
  }
}

// check that gpu_version belong to range min_version-max_version
// min_version is included and max_version is excluded.
bool IsGPUVersionInRange(int gpu_version, int min_version, int max_version) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc mht_6(mht_6_v, 349, "", "./tensorflow/lite/delegates/gpu/cl/cl_device.cc", "IsGPUVersionInRange");

  return gpu_version >= min_version && gpu_version < max_version;
}

GpuInfo GpuInfoFromDeviceID(cl_device_id id, cl_platform_id platform_id) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc mht_7(mht_7_v, 356, "", "./tensorflow/lite/delegates/gpu/cl/cl_device.cc", "GpuInfoFromDeviceID");

  GpuInfo info;
  info.opencl_info.platform_version =
      GetPlatformInfo(platform_id, CL_PLATFORM_VERSION);
  info.opencl_info.device_name = GetDeviceInfo<std::string>(id, CL_DEVICE_NAME);
  info.opencl_info.vendor_name =
      GetDeviceInfo<std::string>(id, CL_DEVICE_VENDOR);
  info.opencl_info.opencl_c_version =
      GetDeviceInfo<std::string>(id, CL_DEVICE_OPENCL_C_VERSION);
  info.opencl_info.driver_version =
      GetDeviceInfo<std::string>(id, CL_DRIVER_VERSION);
  const std::string gpu_description = absl::StrCat(
      info.opencl_info.device_name, " ", info.opencl_info.vendor_name, " ",
      info.opencl_info.opencl_c_version);
  GetGpuInfoFromDeviceDescription(gpu_description, GpuApi::kOpenCl, &info);
  info.opencl_info.cl_version =
      ParseCLVersion(info.opencl_info.opencl_c_version);
  info.opencl_info.extensions =
      absl::StrSplit(GetDeviceInfo<std::string>(id, CL_DEVICE_EXTENSIONS), ' ');
  info.opencl_info.supports_fp16 = false;
  info.opencl_info.supports_image3d_writes = false;
  for (const auto& ext : info.opencl_info.extensions) {
    if (ext == "cl_khr_fp16") {
      info.opencl_info.supports_fp16 = true;
    }
    if (ext == "cl_khr_3d_image_writes") {
      info.opencl_info.supports_image3d_writes = true;
    }
  }

  info.opencl_info.supports_images =
      GetDeviceInfo<cl_bool>(id, CL_DEVICE_IMAGE_SUPPORT);

  cl_device_fp_config f32_config =
      GetDeviceInfo<cl_device_fp_config>(id, CL_DEVICE_SINGLE_FP_CONFIG);
  info.opencl_info.supports_fp32_rtn = f32_config & CL_FP_ROUND_TO_NEAREST;

  if (info.opencl_info.supports_fp16) {
    cl_device_fp_config f16_config;
    auto status = GetDeviceInfo<cl_device_fp_config>(
        id, CL_DEVICE_HALF_FP_CONFIG, &f16_config);
    // AMD supports cl_khr_fp16 but CL_DEVICE_HALF_FP_CONFIG is empty.
    if (status.ok() && !info.IsAMD()) {
      info.opencl_info.supports_fp16_rtn = f16_config & CL_FP_ROUND_TO_NEAREST;
    } else {  // happens on PowerVR
      f16_config = f32_config;
      info.opencl_info.supports_fp16_rtn = info.opencl_info.supports_fp32_rtn;
    }
  } else {
    info.opencl_info.supports_fp16_rtn = false;
  }

  if (info.IsPowerVR() && !info.opencl_info.supports_fp16) {
    // PowerVR doesn't have full support of fp16 and so doesn't list this
    // extension. But it can support fp16 in MADs and as buffers/textures types,
    // so we will use it.
    info.opencl_info.supports_fp16 = true;
    info.opencl_info.supports_fp16_rtn = info.opencl_info.supports_fp32_rtn;
  }

  if (!info.opencl_info.supports_image3d_writes &&
      ((info.IsAdreno() && info.adreno_info.IsAdreno4xx()) ||
       info.IsNvidia())) {
    // in local tests Adreno 430 can write in image 3d, at least on small sizes,
    // but it doesn't have cl_khr_3d_image_writes in list of available
    // extensions
    // The same for NVidia
    info.opencl_info.supports_image3d_writes = true;
  }
  info.opencl_info.compute_units_count =
      GetDeviceInfo<cl_uint>(id, CL_DEVICE_MAX_COMPUTE_UNITS);
  info.opencl_info.image2d_max_width =
      GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE2D_MAX_WIDTH);
  info.opencl_info.image2d_max_height =
      GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
  info.opencl_info.buffer_max_size =
      GetDeviceInfo<cl_ulong>(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
  info.opencl_info.max_allocation_size =
      GetDeviceInfo<cl_ulong>(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
  if (info.opencl_info.cl_version >= OpenClVersion::kCl1_2) {
    info.opencl_info.image_buffer_max_size =
        GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE);
    info.opencl_info.image_array_max_layers =
        GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE);
  }
  info.opencl_info.image3d_max_width =
      GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE3D_MAX_WIDTH);
  info.opencl_info.image3d_max_height =
      GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
  info.opencl_info.image3d_max_depth =
      GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE3D_MAX_DEPTH);
  int3 max_work_group_sizes;
  GetDeviceWorkDimsSizes(id, &max_work_group_sizes);
  info.opencl_info.max_work_group_size_x = max_work_group_sizes.x;
  info.opencl_info.max_work_group_size_y = max_work_group_sizes.y;
  info.opencl_info.max_work_group_size_z = max_work_group_sizes.z;
  info.opencl_info.max_work_group_total_size =
      GetDeviceInfo<size_t>(id, CL_DEVICE_MAX_WORK_GROUP_SIZE);

  info.opencl_info.base_addr_align_in_bits =
      GetDeviceInfo<cl_uint>(id, CL_DEVICE_MEM_BASE_ADDR_ALIGN);
  info.opencl_info.image_pitch_alignment = 0;
  if (info.opencl_info.cl_version == OpenClVersion::kCl2_0 ||
      info.opencl_info.cl_version == OpenClVersion::kCl2_1 ||
      info.opencl_info.cl_version == OpenClVersion::kCl2_2) {
    info.opencl_info.image_pitch_alignment =
        GetDeviceInfo<cl_uint>(id, CL_DEVICE_IMAGE_PITCH_ALIGNMENT);
    info.opencl_info.image_base_address_alignment =
        GetDeviceInfo<cl_uint>(id, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT);
  } else if (info.SupportsExtension("cl_khr_image2d_from_buffer")) {
    cl_uint result = 0;
    auto status =
        GetDeviceInfo(id, CL_DEVICE_IMAGE_PITCH_ALIGNMENT_KHR, &result);
    if (status.ok()) {
      info.opencl_info.image_pitch_alignment = result;
    }
    result = 0;
    status =
        GetDeviceInfo(id, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT_KHR, &result);
    if (status.ok()) {
      info.opencl_info.image_base_address_alignment = result;
    }
  }

  if (info.IsIntel()) {
    if (info.SupportsExtension("cl_intel_required_subgroup_size")) {
      size_t sub_groups_count;
      cl_int status =
          clGetDeviceInfo(id, 0x4108 /*CL_DEVICE_SUB_GROUP_SIZES_INTEL*/, 0,
                          nullptr, &sub_groups_count);
      if (status == CL_SUCCESS) {
        std::vector<size_t> sub_group_sizes(sub_groups_count);
        status = clGetDeviceInfo(id, 0x4108 /*CL_DEVICE_SUB_GROUP_SIZES_INTEL*/,
                                 sizeof(size_t) * sub_groups_count,
                                 sub_group_sizes.data(), nullptr);
        if (status == CL_SUCCESS) {
          for (int i = 0; i < sub_groups_count; ++i) {
            info.supported_subgroup_sizes.push_back(sub_group_sizes[i]);
          }
        }
      }
    }
  }
  if (info.IsAdreno()) {
    ParseQualcommOpenClCompilerVersion(info.opencl_info.driver_version,
                                       &info.adreno_info.cl_compiler_version);
  }
  return info;
}

}  // namespace

CLDevice::CLDevice(cl_device_id id, cl_platform_id platform_id)
    : info_(GpuInfoFromDeviceID(id, platform_id)),
      id_(id),
      platform_id_(platform_id) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc mht_8(mht_8_v, 514, "", "./tensorflow/lite/delegates/gpu/cl/cl_device.cc", "CLDevice::CLDevice");

  if (info_.IsAdreno() &&
      info_.adreno_info.adreno_gpu == AdrenoGpu::kAdreno630) {
    acceleration::AndroidInfo android_info;
    if (acceleration::RequestAndroidInfo(&android_info).ok()) {
      info_.adreno_info.compiler_bugs_in_a6xx =
          android_info.android_sdk_version == "26";
    }
  }
}

CLDevice::CLDevice(const CLDevice& device)
    : info_(device.info_), id_(device.id_), platform_id_(device.platform_id_) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc mht_9(mht_9_v, 529, "", "./tensorflow/lite/delegates/gpu/cl/cl_device.cc", "CLDevice::CLDevice");
}

CLDevice& CLDevice::operator=(const CLDevice& device) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc mht_10(mht_10_v, 534, "", "./tensorflow/lite/delegates/gpu/cl/cl_device.cc", "=");

  if (this != &device) {
    info_ = device.info_;
    id_ = device.id_;
    platform_id_ = device.platform_id_;
  }
  return *this;
}

CLDevice::CLDevice(CLDevice&& device)
    : info_(std::move(device.info_)),
      id_(device.id_),
      platform_id_(device.platform_id_) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc mht_11(mht_11_v, 549, "", "./tensorflow/lite/delegates/gpu/cl/cl_device.cc", "CLDevice::CLDevice");

  device.id_ = nullptr;
  device.platform_id_ = nullptr;
}

CLDevice& CLDevice::operator=(CLDevice&& device) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc mht_12(mht_12_v, 557, "", "./tensorflow/lite/delegates/gpu/cl/cl_device.cc", "=");

  if (this != &device) {
    id_ = nullptr;
    platform_id_ = nullptr;
    info_ = std::move(device.info_);
    std::swap(id_, device.id_);
    std::swap(platform_id_, device.platform_id_);
  }
  return *this;
}

std::string CLDevice::GetPlatformVersion() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc mht_13(mht_13_v, 571, "", "./tensorflow/lite/delegates/gpu/cl/cl_device.cc", "CLDevice::GetPlatformVersion");

  return GetPlatformInfo(platform_id_, CL_PLATFORM_VERSION);
}

void CLDevice::DisableOneLayerTextureArray() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc mht_14(mht_14_v, 578, "", "./tensorflow/lite/delegates/gpu/cl/cl_device.cc", "CLDevice::DisableOneLayerTextureArray");

  info_.adreno_info.support_one_layer_texture_array = false;
}

absl::Status CreateDefaultGPUDevice(CLDevice* result) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_deviceDTcc mht_15(mht_15_v, 585, "", "./tensorflow/lite/delegates/gpu/cl/cl_device.cc", "CreateDefaultGPUDevice");

  cl_uint num_platforms;
  cl_int status = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (status != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrFormat("clGetPlatformIDs returned %d", status));
  }
  if (num_platforms == 0) {
    return absl::UnknownError("No supported OpenCL platform.");
  }
  std::vector<cl_platform_id> platforms(num_platforms);
  status = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (status != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrFormat("clGetPlatformIDs returned %d", status));
  }

  cl_platform_id platform_id = platforms[0];
  cl_uint num_devices;
  status =
      clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
  if (status != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrFormat("clGetDeviceIDs returned %d", status));
  }
  if (num_devices == 0) {
    return absl::UnknownError("No GPU on current platform.");
  }

  std::vector<cl_device_id> devices(num_devices);
  status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, num_devices,
                          devices.data(), nullptr);
  if (status != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrFormat("clGetDeviceIDs returned %d", status));
  }

  *result = CLDevice(devices[0], platform_id);
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
