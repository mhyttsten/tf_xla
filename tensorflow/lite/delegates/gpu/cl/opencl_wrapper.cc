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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSopencl_wrapperDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSopencl_wrapperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSopencl_wrapperDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"

#if defined(_WIN32)
#define __WINDOWS__
#endif

#ifdef __WINDOWS__
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

#ifdef __ANDROID__
#define LoadFunction(function)                                                 \
  if (use_wrapper) {                                                           \
    function = reinterpret_cast<PFN_##function>(loadOpenCLPointer(#function)); \
  } else {                                                                     \
    function = reinterpret_cast<PFN_##function>(dlsym(libopencl, #function));  \
  }
#elif defined(__WINDOWS__)
#define LoadFunction(function) \
  function =                   \
      reinterpret_cast<PFN_##function>(GetProcAddress(libopencl, #function));
#else
#define LoadFunction(function) \
  function = reinterpret_cast<PFN_##function>(dlsym(libopencl, #function));
#endif

#ifdef __WINDOWS__
void LoadOpenCLFunctions(HMODULE libopencl);
#else
void LoadOpenCLFunctions(void* libopencl, bool use_wrapper);
#endif

absl::Status LoadOpenCL() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSopencl_wrapperDTcc mht_0(mht_0_v, 228, "", "./tensorflow/lite/delegates/gpu/cl/opencl_wrapper.cc", "LoadOpenCL");

#ifdef __WINDOWS__
  HMODULE libopencl = LoadLibraryA("OpenCL.dll");
  if (libopencl) {
    LoadOpenCLFunctions(libopencl);
    return absl::OkStatus();
  } else {
    DWORD error_code = GetLastError();
    return absl::UnknownError(absl::StrCat(
        "Can not open OpenCL library on this device, error code - ",
        error_code));
  }
#else
  void* libopencl = nullptr;
#ifdef __ANDROID__
  // Pixel phone or auto?
  libopencl = dlopen("libOpenCL-pixel.so", RTLD_NOW | RTLD_LOCAL);
  if (!libopencl) {
    libopencl = dlopen("libOpenCL-car.so", RTLD_NOW | RTLD_LOCAL);
  }
  if (libopencl) {
    typedef void (*enableOpenCL_t)();
    enableOpenCL_t enableOpenCL =
        reinterpret_cast<enableOpenCL_t>(dlsym(libopencl, "enableOpenCL"));
    enableOpenCL();
    LoadOpenCLFunctions(libopencl, true);
    return absl::OkStatus();
  }
#endif
#ifdef __APPLE__
  static const char* kClLibName =
      "/System/Library/Frameworks/OpenCL.framework/OpenCL";
#else
  static const char* kClLibName = "libOpenCL.so";
#endif
  libopencl = dlopen(kClLibName, RTLD_NOW | RTLD_LOCAL);
  if (libopencl) {
    LoadOpenCLFunctions(libopencl, false);
    return absl::OkStatus();
  }
  // record error
  std::string error(dlerror());
  return absl::UnknownError(
      absl::StrCat("Can not open OpenCL library on this device - ", error));
#endif
}

#ifdef __WINDOWS__
void LoadOpenCLFunctions(HMODULE libopencl) {
#else
void LoadOpenCLFunctions(void* libopencl, bool use_wrapper) {
#ifdef __ANDROID__
  typedef void* (*loadOpenCLPointer_t)(const char* name);
  loadOpenCLPointer_t loadOpenCLPointer;
  if (use_wrapper) {
    loadOpenCLPointer = reinterpret_cast<loadOpenCLPointer_t>(
        dlsym(libopencl, "loadOpenCLPointer"));
  }
#endif
#endif

  LoadFunction(clGetPlatformIDs);
  LoadFunction(clGetPlatformInfo);
  LoadFunction(clGetDeviceIDs);
  LoadFunction(clGetDeviceInfo);
  LoadFunction(clCreateSubDevices);
  LoadFunction(clRetainDevice);
  LoadFunction(clReleaseDevice);
  LoadFunction(clCreateContext);
  LoadFunction(clCreateContextFromType);
  LoadFunction(clRetainContext);
  LoadFunction(clReleaseContext);
  LoadFunction(clGetContextInfo);
  LoadFunction(clCreateCommandQueueWithProperties);
  LoadFunction(clRetainCommandQueue);
  LoadFunction(clReleaseCommandQueue);
  LoadFunction(clGetCommandQueueInfo);
  LoadFunction(clCreateBuffer);
  LoadFunction(clCreateSubBuffer);
  LoadFunction(clCreateImage);
  LoadFunction(clCreatePipe);
  LoadFunction(clRetainMemObject);
  LoadFunction(clReleaseMemObject);
  LoadFunction(clGetSupportedImageFormats);
  LoadFunction(clGetMemObjectInfo);
  LoadFunction(clGetImageInfo);
  LoadFunction(clGetPipeInfo);
  LoadFunction(clSetMemObjectDestructorCallback);
  LoadFunction(clSVMAlloc);
  LoadFunction(clSVMFree);
  LoadFunction(clCreateSamplerWithProperties);
  LoadFunction(clRetainSampler);
  LoadFunction(clReleaseSampler);
  LoadFunction(clGetSamplerInfo);
  LoadFunction(clCreateProgramWithSource);
  LoadFunction(clCreateProgramWithBinary);
  LoadFunction(clCreateProgramWithBuiltInKernels);
  LoadFunction(clRetainProgram);
  LoadFunction(clReleaseProgram);
  LoadFunction(clBuildProgram);
  LoadFunction(clCompileProgram);
  LoadFunction(clLinkProgram);
  LoadFunction(clUnloadPlatformCompiler);
  LoadFunction(clGetProgramInfo);
  LoadFunction(clGetProgramBuildInfo);
  LoadFunction(clCreateKernel);
  LoadFunction(clCreateKernelsInProgram);
  LoadFunction(clRetainKernel);
  LoadFunction(clReleaseKernel);
  LoadFunction(clSetKernelArg);
  LoadFunction(clSetKernelArgSVMPointer);
  LoadFunction(clSetKernelExecInfo);
  LoadFunction(clGetKernelInfo);
  LoadFunction(clGetKernelArgInfo);
  LoadFunction(clGetKernelWorkGroupInfo);
  LoadFunction(clWaitForEvents);
  LoadFunction(clGetEventInfo);
  LoadFunction(clCreateUserEvent);
  LoadFunction(clRetainEvent);
  LoadFunction(clReleaseEvent);
  LoadFunction(clSetUserEventStatus);
  LoadFunction(clSetEventCallback);
  LoadFunction(clGetEventProfilingInfo);
  LoadFunction(clFlush);
  LoadFunction(clFinish);
  LoadFunction(clEnqueueReadBuffer);
  LoadFunction(clEnqueueReadBufferRect);
  LoadFunction(clEnqueueWriteBuffer);
  LoadFunction(clEnqueueWriteBufferRect);
  LoadFunction(clEnqueueFillBuffer);
  LoadFunction(clEnqueueCopyBuffer);
  LoadFunction(clEnqueueCopyBufferRect);
  LoadFunction(clEnqueueReadImage);
  LoadFunction(clEnqueueWriteImage);
  LoadFunction(clEnqueueFillImage);
  LoadFunction(clEnqueueCopyImage);
  LoadFunction(clEnqueueCopyImageToBuffer);
  LoadFunction(clEnqueueCopyBufferToImage);
  LoadFunction(clEnqueueMapBuffer);
  LoadFunction(clEnqueueMapImage);
  LoadFunction(clEnqueueUnmapMemObject);
  LoadFunction(clEnqueueMigrateMemObjects);
  LoadFunction(clEnqueueNDRangeKernel);
  LoadFunction(clEnqueueNativeKernel);
  LoadFunction(clEnqueueMarkerWithWaitList);
  LoadFunction(clEnqueueBarrierWithWaitList);
  LoadFunction(clEnqueueSVMFree);
  LoadFunction(clEnqueueSVMMemcpy);
  LoadFunction(clEnqueueSVMMemFill);
  LoadFunction(clEnqueueSVMMap);
  LoadFunction(clEnqueueSVMUnmap);
  LoadFunction(clGetExtensionFunctionAddressForPlatform);
  LoadFunction(clCreateImage2D);
  LoadFunction(clCreateImage3D);
  LoadFunction(clEnqueueMarker);
  LoadFunction(clEnqueueWaitForEvents);
  LoadFunction(clEnqueueBarrier);
  LoadFunction(clUnloadCompiler);
  LoadFunction(clGetExtensionFunctionAddress);
  LoadFunction(clCreateCommandQueue);
  LoadFunction(clCreateSampler);
  LoadFunction(clEnqueueTask);

  // OpenGL sharing
  LoadFunction(clCreateFromGLBuffer);
  LoadFunction(clCreateFromGLTexture);
  LoadFunction(clEnqueueAcquireGLObjects);
  LoadFunction(clEnqueueReleaseGLObjects);

  // cl_khr_egl_event extension
  LoadFunction(clCreateEventFromEGLSyncKHR);

  // EGL sharing
  LoadFunction(clCreateFromEGLImageKHR);
  LoadFunction(clEnqueueAcquireEGLObjectsKHR);
  LoadFunction(clEnqueueReleaseEGLObjectsKHR);

  LoadQcomExtensionFunctions();
}

// No OpenCL support, do not set function addresses
PFN_clGetPlatformIDs clGetPlatformIDs;
PFN_clGetPlatformInfo clGetPlatformInfo;
PFN_clGetDeviceIDs clGetDeviceIDs;
PFN_clGetDeviceInfo clGetDeviceInfo;
PFN_clCreateSubDevices clCreateSubDevices;
PFN_clRetainDevice clRetainDevice;
PFN_clReleaseDevice clReleaseDevice;
PFN_clCreateContext clCreateContext;
PFN_clCreateContextFromType clCreateContextFromType;
PFN_clRetainContext clRetainContext;
PFN_clReleaseContext clReleaseContext;
PFN_clGetContextInfo clGetContextInfo;
PFN_clCreateCommandQueueWithProperties clCreateCommandQueueWithProperties;
PFN_clRetainCommandQueue clRetainCommandQueue;
PFN_clReleaseCommandQueue clReleaseCommandQueue;
PFN_clGetCommandQueueInfo clGetCommandQueueInfo;
PFN_clCreateBuffer clCreateBuffer;
PFN_clCreateSubBuffer clCreateSubBuffer;
PFN_clCreateImage clCreateImage;
PFN_clCreatePipe clCreatePipe;
PFN_clRetainMemObject clRetainMemObject;
PFN_clReleaseMemObject clReleaseMemObject;
PFN_clGetSupportedImageFormats clGetSupportedImageFormats;
PFN_clGetMemObjectInfo clGetMemObjectInfo;
PFN_clGetImageInfo clGetImageInfo;
PFN_clGetPipeInfo clGetPipeInfo;
PFN_clSetMemObjectDestructorCallback clSetMemObjectDestructorCallback;
PFN_clSVMAlloc clSVMAlloc;
PFN_clSVMFree clSVMFree;
PFN_clCreateSamplerWithProperties clCreateSamplerWithProperties;
PFN_clRetainSampler clRetainSampler;
PFN_clReleaseSampler clReleaseSampler;
PFN_clGetSamplerInfo clGetSamplerInfo;
PFN_clCreateProgramWithSource clCreateProgramWithSource;
PFN_clCreateProgramWithBinary clCreateProgramWithBinary;
PFN_clCreateProgramWithBuiltInKernels clCreateProgramWithBuiltInKernels;
PFN_clRetainProgram clRetainProgram;
PFN_clReleaseProgram clReleaseProgram;
PFN_clBuildProgram clBuildProgram;
PFN_clCompileProgram clCompileProgram;
PFN_clLinkProgram clLinkProgram;
PFN_clUnloadPlatformCompiler clUnloadPlatformCompiler;
PFN_clGetProgramInfo clGetProgramInfo;
PFN_clGetProgramBuildInfo clGetProgramBuildInfo;
PFN_clCreateKernel clCreateKernel;
PFN_clCreateKernelsInProgram clCreateKernelsInProgram;
PFN_clRetainKernel clRetainKernel;
PFN_clReleaseKernel clReleaseKernel;
PFN_clSetKernelArg clSetKernelArg;
PFN_clSetKernelArgSVMPointer clSetKernelArgSVMPointer;
PFN_clSetKernelExecInfo clSetKernelExecInfo;
PFN_clGetKernelInfo clGetKernelInfo;
PFN_clGetKernelArgInfo clGetKernelArgInfo;
PFN_clGetKernelWorkGroupInfo clGetKernelWorkGroupInfo;
PFN_clWaitForEvents clWaitForEvents;
PFN_clGetEventInfo clGetEventInfo;
PFN_clCreateUserEvent clCreateUserEvent;
PFN_clRetainEvent clRetainEvent;
PFN_clReleaseEvent clReleaseEvent;
PFN_clSetUserEventStatus clSetUserEventStatus;
PFN_clSetEventCallback clSetEventCallback;
PFN_clGetEventProfilingInfo clGetEventProfilingInfo;
PFN_clFlush clFlush;
PFN_clFinish clFinish;
PFN_clEnqueueReadBuffer clEnqueueReadBuffer;
PFN_clEnqueueReadBufferRect clEnqueueReadBufferRect;
PFN_clEnqueueWriteBuffer clEnqueueWriteBuffer;
PFN_clEnqueueWriteBufferRect clEnqueueWriteBufferRect;
PFN_clEnqueueFillBuffer clEnqueueFillBuffer;
PFN_clEnqueueCopyBuffer clEnqueueCopyBuffer;
PFN_clEnqueueCopyBufferRect clEnqueueCopyBufferRect;
PFN_clEnqueueReadImage clEnqueueReadImage;
PFN_clEnqueueWriteImage clEnqueueWriteImage;
PFN_clEnqueueFillImage clEnqueueFillImage;
PFN_clEnqueueCopyImage clEnqueueCopyImage;
PFN_clEnqueueCopyImageToBuffer clEnqueueCopyImageToBuffer;
PFN_clEnqueueCopyBufferToImage clEnqueueCopyBufferToImage;
PFN_clEnqueueMapBuffer clEnqueueMapBuffer;
PFN_clEnqueueMapImage clEnqueueMapImage;
PFN_clEnqueueUnmapMemObject clEnqueueUnmapMemObject;
PFN_clEnqueueMigrateMemObjects clEnqueueMigrateMemObjects;
PFN_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
PFN_clEnqueueNativeKernel clEnqueueNativeKernel;
PFN_clEnqueueMarkerWithWaitList clEnqueueMarkerWithWaitList;
PFN_clEnqueueBarrierWithWaitList clEnqueueBarrierWithWaitList;
PFN_clEnqueueSVMFree clEnqueueSVMFree;
PFN_clEnqueueSVMMemcpy clEnqueueSVMMemcpy;
PFN_clEnqueueSVMMemFill clEnqueueSVMMemFill;
PFN_clEnqueueSVMMap clEnqueueSVMMap;
PFN_clEnqueueSVMUnmap clEnqueueSVMUnmap;
PFN_clGetExtensionFunctionAddressForPlatform
    clGetExtensionFunctionAddressForPlatform;
PFN_clCreateImage2D clCreateImage2D;
PFN_clCreateImage3D clCreateImage3D;
PFN_clEnqueueMarker clEnqueueMarker;
PFN_clEnqueueWaitForEvents clEnqueueWaitForEvents;
PFN_clEnqueueBarrier clEnqueueBarrier;
PFN_clUnloadCompiler clUnloadCompiler;
PFN_clGetExtensionFunctionAddress clGetExtensionFunctionAddress;
PFN_clCreateCommandQueue clCreateCommandQueue;
PFN_clCreateSampler clCreateSampler;
PFN_clEnqueueTask clEnqueueTask;

// OpenGL sharing
PFN_clCreateFromGLBuffer clCreateFromGLBuffer;
PFN_clCreateFromGLTexture clCreateFromGLTexture;
PFN_clEnqueueAcquireGLObjects clEnqueueAcquireGLObjects;
PFN_clEnqueueReleaseGLObjects clEnqueueReleaseGLObjects;

// cl_khr_egl_event extension
PFN_clCreateEventFromEGLSyncKHR clCreateEventFromEGLSyncKHR;

// EGL sharing
PFN_clCreateFromEGLImageKHR clCreateFromEGLImageKHR;
PFN_clEnqueueAcquireEGLObjectsKHR clEnqueueAcquireEGLObjectsKHR;
PFN_clEnqueueReleaseEGLObjectsKHR clEnqueueReleaseEGLObjectsKHR;

DEFINE_QCOM_FUNCTION_PTRS

cl_mem CreateImage2DLegacy(cl_context context, cl_mem_flags flags,
                           const cl_image_format* image_format,
                           const cl_image_desc* image_desc, void* host_ptr,
                           cl_int* errcode_ret) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSopencl_wrapperDTcc mht_1(mht_1_v, 534, "", "./tensorflow/lite/delegates/gpu/cl/opencl_wrapper.cc", "CreateImage2DLegacy");

  if (clCreateImage) {  // clCreateImage available since OpenCL 1.2
    return clCreateImage(context, flags, image_format, image_desc, host_ptr,
                         errcode_ret);
  } else {
    return clCreateImage2D(context, flags, image_format,
                           image_desc->image_width, image_desc->image_height,
                           image_desc->image_row_pitch, host_ptr, errcode_ret);
  }
}

cl_mem CreateImage3DLegacy(cl_context context, cl_mem_flags flags,
                           const cl_image_format* image_format,
                           const cl_image_desc* image_desc, void* host_ptr,
                           cl_int* errcode_ret) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSopencl_wrapperDTcc mht_2(mht_2_v, 551, "", "./tensorflow/lite/delegates/gpu/cl/opencl_wrapper.cc", "CreateImage3DLegacy");

  if (clCreateImage) {  // clCreateImage available since OpenCL 1.2
    return clCreateImage(context, flags, image_format, image_desc, host_ptr,
                         errcode_ret);
  } else {
    return clCreateImage3D(context, flags, image_format,
                           image_desc->image_width, image_desc->image_height,
                           image_desc->image_depth, image_desc->image_row_pitch,
                           image_desc->image_slice_pitch, host_ptr,
                           errcode_ret);
  }
}
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
