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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_GPU_INFO_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_GPU_INFO_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTh() {
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


#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"

namespace tflite {
namespace gpu {

// The VendorID returned by the GPU driver.
enum class GpuVendor {
  kApple,
  kQualcomm,
  kMali,
  kPowerVR,
  kNvidia,
  kAMD,
  kIntel,
  kUnknown
};

enum class GpuApi {
  kUnknown,
  kOpenCl,
  kMetal,
  kVulkan,
  kOpenGl,
};

enum class AdrenoGpu {
  // Adreno 7xx series
  kAdreno730,
  // Adreno 6xx series
  kAdreno685,
  kAdreno680,
  kAdreno675,
  kAdreno660,
  kAdreno650,
  kAdreno640,
  kAdreno630,
  kAdreno620,
  kAdreno618,
  kAdreno616,
  kAdreno615,
  kAdreno612,
  kAdreno610,
  kAdreno605,
  // Adreno 5xx series
  kAdreno540,
  kAdreno530,
  kAdreno512,
  kAdreno510,
  kAdreno509,
  kAdreno508,
  kAdreno506,
  kAdreno505,
  kAdreno504,
  // Adreno 4xx series
  kAdreno430,
  kAdreno420,
  kAdreno418,
  kAdreno405,
  // Adreno 3xx series
  kAdreno330,
  kAdreno320,
  kAdreno308,
  kAdreno306,
  kAdreno305,
  kAdreno304,
  // Adreno 2xx series
  kAdreno225,
  kAdreno220,
  kAdreno205,
  kAdreno203,
  kAdreno200,
  // Adreno 1xx series
  kAdreno130,
  kAdreno120,
  kUnknown
};

struct AMDInfo {
  AMDInfo() = default;
  int shader_engines;
  int compute_units_per_shader_engine;
  int GetComputeUnitsCount() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_infoDTh mht_0(mht_0_v, 273, "", "./tensorflow/lite/delegates/gpu/common/gpu_info.h", "GetComputeUnitsCount");

    return shader_engines * compute_units_per_shader_engine;
  }
};

struct AdrenoInfo {
  struct OpenClCompilerVersion {
    int major = 0;
    int minor = 0;
    int patch = 0;
  };
  AdrenoInfo() = default;
  explicit AdrenoInfo(const std::string& device_version);

  AdrenoGpu adreno_gpu;

  bool IsAdreno1xx() const;
  bool IsAdreno2xx() const;
  bool IsAdreno3xx() const;
  bool IsAdreno4xx() const;
  bool IsAdreno5xx() const;
  bool IsAdreno6xx() const;
  bool IsAdreno7xx() const;
  bool IsAdreno6xxOrHigher() const;

  // This function returns some not very documented physical parameter of
  // Adreno6xx GPU.
  // We obtained it using Snapdragon Profiler.
  int GetMaximumWavesCount() const;

  // returns amount of register memory per CU(Compute Unit) in bytes.
  int GetRegisterMemorySizePerComputeUnit() const;

  // returns maximum possible amount of waves based on register usage.
  int GetMaximumWavesCount(int register_footprint_per_tread,
                           bool full_wave = true) const;

  int GetWaveSize(bool full_wave) const;

  int GetComputeUnitsCount() const;

  // Not supported on some Adreno devices with specific driver version.
  // b/131099086
  bool support_one_layer_texture_array = true;

  bool compiler_bugs_in_a6xx = false;

  OpenClCompilerVersion cl_compiler_version;
};

enum class AppleGpu {
  kUnknown,
  kA7,
  kA8,
  kA8X,
  kA9,
  kA9X,
  kA10,
  kA10X,
  kA11,
  kA12,
  kA12X,
  kA12Z,
  kA13,
  kA14,
  kA15,
  kM1,
  kM1Pro,
  kM1Max,
};

struct AppleInfo {
  AppleInfo() = default;
  explicit AppleInfo(const std::string& gpu_description);
  AppleGpu gpu_type;

  bool IsA7GenerationGpu() const;
  bool IsA8GenerationGpu() const;
  bool IsLocalMemoryPreferredOverGlobal() const;

  bool IsBionic() const;

  bool IsSIMDMatMulSupported() const;
  // Often, fp32 alu performance is 1/2 of fp16 alu performance
  // But, on some devices, fp32 alu performance equal to fp16 alu performance,
  // at least in some scenarios.
  // This method returns true if SIMDMatMul performance in fp32 equal to fp16
  bool IsSIMDMatMulFp32Perf2x() const;

  // floating point rounding mode
  bool IsRoundToNearestSupported() const;

  int GetComputeUnitsCount() const;

  // do not use, for internal usage
  void SetComputeUnits(int compute_units_count);

 private:
  int compute_units = -1;
};

enum class MaliGpu {
  kUnknown,
  kT604,
  kT622,
  kT624,
  kT628,
  kT658,
  kT678,
  kT720,
  kT760,
  kT820,
  kT830,
  kT860,
  kT880,
  kG31,
  kG51,
  kG71,
  kG52,
  kG72,
  kG76,
  kG57,
  kG77,
  kG68,
  kG78,
  kG310,
  kG510,
  kG610,
  kG710,
};

struct MaliInfo {
  MaliInfo() = default;
  explicit MaliInfo(const std::string& gpu_description);
  MaliGpu gpu_version;

  bool IsMaliT6xx() const;
  bool IsMaliT7xx() const;
  bool IsMaliT8xx() const;
  bool IsMidgard() const;
  bool IsBifrostGen1() const;
  bool IsBifrostGen2() const;
  bool IsBifrostGen3() const;
  bool IsBifrost() const;
  bool IsValhallGen1() const;
  bool IsValhallGen2() const;
  bool IsValhallGen3() const;
  bool IsValhall() const;
};

struct OpenGlInfo {
  std::string renderer_name;
  std::string vendor_name;
  std::string version;
  int major_version = -1;
  int minor_version = -1;

  int max_image_units = 0;
  int max_ssbo_bindings = 0;
  int max_image_bindings = 0;
  int max_work_group_invocations = 0;
  int max_texture_size = 0;
  int max_array_texture_layers = 0;
  int max_fragment_image_units = 0;
  int max_fragment_uniform_vec4_count = 0;
  int max_color_atttachments = 0;
  int max_viewport_width = 0;
  int max_viewport_height = 0;
  int max_renderbuffer_size = 0;

  std::vector<std::string> extensions;
  int max_compute_work_group_size_x;
  int max_compute_work_group_size_y;
  int max_compute_work_group_size_z;

  bool SupportsExplicitFp16() const;
};

struct VulkanInfo {
  std::string vendor_name;
  uint32_t api_version = -1;
  uint32_t api_version_major = -1;
  uint32_t api_version_minor = -1;
  uint32_t api_version_patch = -1;

  int max_per_stage_descriptor_sampled_images = 0;
  uint32_t max_compute_work_group_invocations;
  uint32_t max_image_dimension_1d;
  uint32_t max_image_dimension_2d;
  uint32_t max_image_dimension_3d;
  uint32_t max_image_array_layers;
  uint64_t max_texel_buffer_elements;
  uint64_t max_uniform_buffer_range;
  uint64_t max_storage_buffer_range;
  uint64_t max_push_constants_size;

  uint32_t subgroup_size = 0;
  bool supports_subgroup_arithmetic = false;

  std::vector<std::string> extensions;
  int max_compute_work_group_size_x;
  int max_compute_work_group_size_y;
  int max_compute_work_group_size_z;

  bool SupportsExplicitFp16() const;
};

enum class OpenClVersion {
  kCl1_0,
  kCl1_1,
  kCl1_2,
  kCl2_0,
  kCl2_1,
  kCl2_2,
  kCl3_0,
  kUnknown,
};
std::string OpenClVersionToString(OpenClVersion version);

struct OpenClInfo {
  std::string device_name;
  std::string vendor_name;
  std::string opencl_c_version;
  std::string platform_version;
  std::string driver_version;

  OpenClVersion cl_version;

  std::vector<std::string> extensions;
  bool supports_fp16;
  bool supports_image3d_writes;
  bool supports_images;
  int compute_units_count;
  uint64_t buffer_max_size;
  uint64_t max_allocation_size;
  uint64_t image2d_max_width;
  uint64_t image2d_max_height;
  uint64_t image_buffer_max_size;
  uint64_t image_array_max_layers;
  uint64_t image3d_max_width;
  uint64_t image3d_max_height;
  uint64_t image3d_max_depth;
  int max_work_group_size_x;
  int max_work_group_size_y;
  int max_work_group_size_z;
  int max_work_group_total_size;

  // The row pitch alignment size in pixels for 2D images created from a buffer.
  // The value must be a power of 2.
  uint64_t image_pitch_alignment = 0;
  // The minimum alignment in pixels. The value must be a power of 2.
  uint64_t image_base_address_alignment = 0;
  uint64_t base_addr_align_in_bits;

  // rtn is ROUND_TO_NEAREST
  // with rtn precision is much better then with rtz (ROUND_TO_ZERO)
  // Adreno 3xx supports only rtz, Adreno 4xx and more support rtn
  // Mali from T6xx supports rtn
  // PowerVR supports only rtz
  bool supports_fp32_rtn;
  bool supports_fp16_rtn;

  bool supports_r_f16_tex2d = false;
  bool supports_rg_f16_tex2d = false;
  bool supports_rgb_f16_tex2d = false;
  bool supports_rgba_f16_tex2d = false;

  bool supports_r_f32_tex2d = false;
  bool supports_rg_f32_tex2d = false;
  bool supports_rgb_f32_tex2d = false;
  bool supports_rgba_f32_tex2d = false;

  bool IsImage2dFromBufferSupported() const;
};

enum class MetalLanguageVersion {
  kMetal1_0,
  kMetal1_1,
  kMetal1_2,
  kMetal2_0,
  kMetal2_1,
  kMetal2_2,
  kMetal2_3,
  kUnknown,
};

struct MetalInfo {
  MetalLanguageVersion language_version;

  int max_work_group_size_x;
  int max_work_group_size_y;
  int max_work_group_size_z;

  uint64_t buffer_max_size;

  uint64_t image2d_max_width;
  uint64_t image2d_max_height;
  uint64_t image_array_max_layers;
  uint64_t image3d_max_width;
  uint64_t image3d_max_height;
  uint64_t image3d_max_depth;
};

struct GpuInfo {
  bool IsAdreno() const;
  bool IsApple() const;
  bool IsMali() const;
  bool IsPowerVR() const;
  bool IsNvidia() const;
  bool IsAMD() const;
  bool IsIntel() const;

  bool IsGlsl() const;
  bool IsGlslSupportsExplicitFp16() const;

  // floating point rounding mode
  bool IsRoundToNearestSupported() const;

  bool SupportsFP16() const;

  bool SupportsImages() const;
  bool SupportsTextureArray() const;
  bool SupportsImageBuffer() const;
  bool SupportsImage3D() const;

  bool SupportsPointersInKernels() const;

  // returns true if device have fixed wave size equal to 32
  bool IsWaveSizeEqualTo32() const;
  bool SupportsSubGroupWithSize(int sub_group_size) const;

  bool SupportsFloatImage2D(DataType data_type, int channels) const;
  bool SupportsExtension(const std::string& extension) const;

  int GetComputeUnitsCount() const;

  int GetMaxImageArguments() const;

  int GetMaxWorkGroupSizeForX() const;
  int GetMaxWorkGroupSizeForY() const;
  int GetMaxWorkGroupSizeForZ() const;
  int GetMaxWorkGroupTotalSize() const;

  uint64_t GetMaxImage2DWidth() const;
  uint64_t GetMaxImage2DHeight() const;
  uint64_t GetMaxImage2DArrayLayers() const;
  uint64_t GetMaxImage3DWidth() const;
  uint64_t GetMaxImage3DHeight() const;
  uint64_t GetMaxImage3DDepth() const;
  uint64_t GetMaxBufferSize() const;
  uint64_t GetMaxMemoryAllocationSize() const;
  uint64_t GetMaxImageBufferWidth() const;

  GpuVendor vendor = GpuVendor::kUnknown;
  GpuApi gpu_api = GpuApi::kUnknown;

  std::vector<int> supported_subgroup_sizes;

  AdrenoInfo adreno_info;
  AMDInfo amd_info;
  AppleInfo apple_info;
  MaliInfo mali_info;

  // OpenGL specific, gpu_api should be kOpenGl
  OpenGlInfo opengl_info;
  bool IsApiOpenGl() const;
  bool IsApiOpenGl31OrAbove() const;

  // Vulkan specific, gpu_api should be kVulkan
  VulkanInfo vulkan_info;
  bool IsApiVulkan() const;

  MetalInfo metal_info;
  bool IsApiMetal() const;

  OpenClInfo opencl_info;
  bool IsApiOpenCl() const;
  bool IsCL11OrHigher() const;
  bool IsCL20OrHigher() const;
  bool IsCL30OrHigher() const;
};

// Currently it initializes:
// vendor
// AdrenoInfo if vendor is kQualcomm
// AppleInfo if vendor is kApple
// MaliInfo if vendor is kMali
void GetGpuInfoFromDeviceDescription(const std::string& gpu_description,
                                     GpuApi gpu_api, GpuInfo* gpu_info);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_GPU_INFO_H_
