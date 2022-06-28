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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc() {
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

// Creates some GPU activity to test functionalities of gpuperfcounter/gputrace.
#include "tensorflow/core/profiler/internal/gpu/cuda_test.h"

#if GOOGLE_CUDA
#include <stdio.h>

#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#endif

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace profiler {
namespace test {

#if GOOGLE_CUDA
namespace {

// Simple printf kernel.
__global__ void simple_print() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/profiler/internal/gpu/cuda_test.cu.cc", "simple_print");
 printf("hello, world!\n"); }

// Empty kernel.
__global__ void empty() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/profiler/internal/gpu/cuda_test.cu.cc", "empty");
}

// Simple kernel accesses memory.
__global__ void access(int *addr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc mht_2(mht_2_v, 217, "", "./tensorflow/core/profiler/internal/gpu/cuda_test.cu.cc", "access");
 *addr = *addr * 2; }

unsigned *g_device_copy;

unsigned *gpu0_buf, *gpu1_buf;

}  // namespace
#endif  // GOOGLE_CUDA

void PrintfKernel(int iters) {
#if GOOGLE_CUDA
  for (int i = 0; i < iters; ++i) {
    simple_print<<<1, 1>>>();
  }
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

void EmptyKernel(int iters) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/profiler/internal/gpu/cuda_test.cu.cc", "EmptyKernel");

#if GOOGLE_CUDA
  for (int i = 0; i < iters; ++i) {
    empty<<<1, 1>>>();
  }
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

void AccessKernel(int *addr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc mht_4(mht_4_v, 252, "", "./tensorflow/core/profiler/internal/gpu/cuda_test.cu.cc", "AccessKernel");

#if GOOGLE_CUDA
  access<<<1, 1>>>(addr);
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

void Synchronize() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc mht_5(mht_5_v, 263, "", "./tensorflow/core/profiler/internal/gpu/cuda_test.cu.cc", "Synchronize");

#if GOOGLE_CUDA
  cudaDeviceSynchronize();
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

void UnifiedMemoryHtoDAndDtoH() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc mht_6(mht_6_v, 274, "", "./tensorflow/core/profiler/internal/gpu/cuda_test.cu.cc", "UnifiedMemoryHtoDAndDtoH");

#if GOOGLE_CUDA
  int *addr = nullptr;
  cudaMallocManaged(reinterpret_cast<void **>(&addr), sizeof(int));
  // The page is now in host memory.
  *addr = 1;
  // The kernel wants to access the page. HtoD transfer happens.
  AccessKernel(addr);
  Synchronize();
  // The page is now in device memory. CPU wants to access the page. DtoH
  // transfer happens.
  EXPECT_EQ(*addr, 2);
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

void MemCopyH2D() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc mht_7(mht_7_v, 294, "", "./tensorflow/core/profiler/internal/gpu/cuda_test.cu.cc", "MemCopyH2D");

#if GOOGLE_CUDA
  unsigned host_val = 0x12345678;
  cudaMalloc(reinterpret_cast<void **>(&g_device_copy), sizeof(unsigned));
  cudaMemcpy(g_device_copy, &host_val, sizeof(unsigned),
             cudaMemcpyHostToDevice);
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

void MemCopyH2D_Async() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc mht_8(mht_8_v, 308, "", "./tensorflow/core/profiler/internal/gpu/cuda_test.cu.cc", "MemCopyH2D_Async");

#if GOOGLE_CUDA
  unsigned host_val = 0x12345678;
  cudaMalloc(reinterpret_cast<void **>(&g_device_copy), sizeof(unsigned));
  cudaMemcpyAsync(g_device_copy, &host_val, sizeof(unsigned),
                  cudaMemcpyHostToDevice);
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

void MemCopyD2H() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc mht_9(mht_9_v, 322, "", "./tensorflow/core/profiler/internal/gpu/cuda_test.cu.cc", "MemCopyD2H");

#if GOOGLE_CUDA
  unsigned host_val = 0;
  cudaMalloc(reinterpret_cast<void **>(&g_device_copy), sizeof(unsigned));
  cudaMemcpy(&host_val, g_device_copy, sizeof(unsigned),
             cudaMemcpyDeviceToHost);
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

namespace {

// Helper function to set up memory buffers on two devices.
void P2PMemcpyHelper() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc mht_10(mht_10_v, 339, "", "./tensorflow/core/profiler/internal/gpu/cuda_test.cu.cc", "P2PMemcpyHelper");

#if GOOGLE_CUDA
  cudaSetDevice(0);
  cudaMalloc(reinterpret_cast<void **>(&gpu0_buf), sizeof(unsigned));
  cudaDeviceEnablePeerAccess(/*peerDevice=*/1, /*flags=*/0);
  cudaSetDevice(1);
  cudaMalloc(reinterpret_cast<void **>(&gpu1_buf), sizeof(unsigned));
  cudaDeviceEnablePeerAccess(/*peerDevice=*/0, /*flags=*/0);
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

}  // namespace

bool MemCopyP2PAvailable() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc mht_11(mht_11_v, 357, "", "./tensorflow/core/profiler/internal/gpu/cuda_test.cu.cc", "MemCopyP2PAvailable");

#if GOOGLE_CUDA
  int can_access_01 = 0;
  cudaDeviceCanAccessPeer(&can_access_01, /*device=*/0, /*peerDevice=*/1);
  int can_access_10 = 0;
  cudaDeviceCanAccessPeer(&can_access_01, /*device=*/1, /*peerDevice=*/0);
  return can_access_01 && can_access_10;
#else
  return false;
#endif
}

void MemCopyP2PImplicit() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc mht_12(mht_12_v, 372, "", "./tensorflow/core/profiler/internal/gpu/cuda_test.cu.cc", "MemCopyP2PImplicit");

#if GOOGLE_CUDA
  P2PMemcpyHelper();
  cudaMemcpy(gpu1_buf, gpu0_buf, sizeof(unsigned), cudaMemcpyDefault);
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

void MemCopyP2PExplicit() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScuda_testDTcuDTcc mht_13(mht_13_v, 384, "", "./tensorflow/core/profiler/internal/gpu/cuda_test.cu.cc", "MemCopyP2PExplicit");

#if GOOGLE_CUDA
  P2PMemcpyHelper();
  cudaMemcpyPeer(gpu1_buf, 1 /* device */, gpu0_buf, 0 /* device */,
                 sizeof(unsigned));
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

}  // namespace test
}  // namespace profiler
}  // namespace tensorflow
