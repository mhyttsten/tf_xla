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
class MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helper_testDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helper_testDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helper_testDTcuDTcc() {
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#include <time.h>

#include <numeric>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"

#define CUDA_EXPECT_SUCCESS                                 \
  {                                                         \
    gpuDeviceSynchronize();                                 \
    cudaError_t err = cudaGetLastError();                   \
    EXPECT_EQ(cudaSuccess, err) << cudaGetErrorString(err); \
  }

#define CUDA_ASSERT_SUCCESS                                 \
  {                                                         \
    gpuDeviceSynchronize();                                 \
    cudaError_t err = cudaGetLastError();                   \
    ASSERT_EQ(cudaSuccess, err) << cudaGetErrorString(err); \
  }

namespace tensorflow {

namespace {

__global__ void SetOutbufZero(GpuLaunchConfig config,
                              int* __restrict__ outbuf) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helper_testDTcuDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/util/gpu_kernel_helper_test.cu.cc", "SetOutbufZero");

  GPU_1D_KERNEL_LOOP(x, config.virtual_thread_count) { outbuf[x] = 0; }
}

// counting number of jobs by using atomic +1
__global__ void Count1D(GpuLaunchConfig config, int bufsize,
                        int* __restrict__ outbuf) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helper_testDTcuDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/util/gpu_kernel_helper_test.cu.cc", "Count1D");

  GPU_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  // x might overflow when testing extreme case
      break;
    }
    atomicAdd(&outbuf[x % bufsize], 1);
  }
}
__global__ void Count2D(Gpu2DLaunchConfig config, int bufsize,
                        int* __restrict__ outbuf) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helper_testDTcuDTcc mht_2(mht_2_v, 237, "", "./tensorflow/core/util/gpu_kernel_helper_test.cu.cc", "Count2D");

  GPU_AXIS_KERNEL_LOOP(x, config.virtual_thread_count.x, X) {
    if (x < 0) {  // x might overflow when testing extreme case
      break;
    }
    GPU_AXIS_KERNEL_LOOP(y, config.virtual_thread_count.y, Y) {
      if (y < 0) {  // y might overflow when testing extreme case
        break;
      }
      int idx = x * config.virtual_thread_count.y + y;
      atomicAdd(&outbuf[idx % bufsize], 1);
    }
  }
}
__global__ void Count3D(Gpu3DLaunchConfig config, int bufsize,
                        int* __restrict__ outbuf) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helper_testDTcuDTcc mht_3(mht_3_v, 255, "", "./tensorflow/core/util/gpu_kernel_helper_test.cu.cc", "Count3D");

  GPU_AXIS_KERNEL_LOOP(x, config.virtual_thread_count.x, X) {
    if (x < 0) {  // x might overflow when testing extreme case
      break;
    }
    GPU_AXIS_KERNEL_LOOP(y, config.virtual_thread_count.y, Y) {
      if (y < 0) {  // y might overflow when testing extreme case
        break;
      }
      GPU_AXIS_KERNEL_LOOP(z, config.virtual_thread_count.z, Z) {
        if (z < 0) {  // z might overflow when testing extreme case
          break;
        }
        int idx =
            x * config.virtual_thread_count.y * config.virtual_thread_count.z +
            y * config.virtual_thread_count.z + z;
        atomicAdd(&outbuf[idx % bufsize], 1);
      }
    }
  }
}

__global__ void GpuShuffleGetSrcLaneTest(unsigned* __restrict__ failure_count) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helper_testDTcuDTcc mht_4(mht_4_v, 280, "", "./tensorflow/core/util/gpu_kernel_helper_test.cu.cc", "GpuShuffleGetSrcLaneTest");

  unsigned lane_id = GpuLaneId();
  for (int width = warpSize; width > 1; width /= 2) {
    auto check_result = [&](const char* op_name, int param, unsigned actual,
                            unsigned expected) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helper_testDTcuDTcc mht_5(mht_5_v, 288, "", "./tensorflow/core/util/gpu_kernel_helper_test.cu.cc", "lambda");

      if (actual != expected) {
        printf("Cuda%sGetSrcLane(%d, %d) for lane %d returned %d, not %d\n",
               op_name, param, width, lane_id, actual, expected);
        GpuAtomicAdd(failure_count, 1);
      }
    };

    for (int src_lane = -warpSize; src_lane <= warpSize; ++src_lane) {
#if TENSORFLOW_USE_ROCM
      if (src_lane < 0 || src_lane >= width) continue;
#endif
      unsigned actual_lane = detail::GpuShuffleGetSrcLane(src_lane, width);
      unsigned expect_lane =
          GpuShuffleSync(kCudaWarpAll, lane_id, src_lane, width);
      check_result("Shuffle", src_lane, actual_lane, expect_lane);
    }

    for (unsigned delta = 0; delta <= warpSize; ++delta) {
      unsigned actual_lane = detail::GpuShuffleUpGetSrcLane(delta, width);
      unsigned expect_lane =
          GpuShuffleUpSync(kCudaWarpAll, lane_id, delta, width);
      check_result("ShuffleUp", delta, actual_lane, expect_lane);
    }

    for (unsigned delta = 0; delta <= warpSize; ++delta) {
      unsigned actual_lane = detail::GpuShuffleDownGetSrcLane(delta, width);
      unsigned expect_lane =
          GpuShuffleDownSync(kCudaWarpAll, lane_id, delta, width);
      check_result("ShuffleDown", delta, actual_lane, expect_lane);
    }

    for (int lane_lane = warpSize; lane_lane > 0; lane_lane /= 2) {
      unsigned actual_lane = detail::GpuShuffleXorGetSrcLane(lane_lane, width);
      unsigned expect_lane =
          GpuShuffleXorSync(kCudaWarpAll, lane_id, lane_lane, width);
      check_result("ShuffleXor", lane_lane, actual_lane, expect_lane);
    }
  }
}

}  // namespace

class GpuLaunchConfigTest : public ::testing::Test {
 protected:
  static const int bufsize = 1024;
  int* outbuf = nullptr;
  int* outbuf_host = nullptr;
  int hostbuf[bufsize];
  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice d = Eigen::GpuDevice(&stream);

  void copyToHost() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helper_testDTcuDTcc mht_6(mht_6_v, 343, "", "./tensorflow/core/util/gpu_kernel_helper_test.cu.cc", "copyToHost");

#if TENSORFLOW_USE_ROCM
    hipMemcpy(hostbuf, outbuf, sizeof(int) * bufsize, hipMemcpyDeviceToHost);
#endif
  }
  virtual void SetUp() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helper_testDTcuDTcc mht_7(mht_7_v, 351, "", "./tensorflow/core/util/gpu_kernel_helper_test.cu.cc", "SetUp");

#if GOOGLE_CUDA
    cudaError_t err = cudaMallocManaged(&outbuf, sizeof(int) * bufsize);
    outbuf_host = outbuf;
#else
    cudaError_t err = hipMalloc(&outbuf, sizeof(int) * bufsize);
    outbuf_host = hostbuf;
#endif
    ASSERT_EQ(cudaSuccess, err) << cudaGetErrorString(err);
  }

  virtual void TearDown() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSgpu_kernel_helper_testDTcuDTcc mht_8(mht_8_v, 365, "", "./tensorflow/core/util/gpu_kernel_helper_test.cu.cc", "TearDown");

    gpuDeviceSynchronize();
    gpuFree(outbuf);
    outbuf = nullptr;
  }
};

TEST_F(GpuLaunchConfigTest, GetGpuLaunchConfig) {
  GpuLaunchConfig cfg;

// test valid inputs
#define TEST_LAUNCH_PARAMETER(work_element_count)                             \
  cfg = GetGpuLaunchConfig(bufsize, d);                                       \
  TF_CHECK_OK(GpuLaunchKernel(SetOutbufZero, cfg.block_count,                 \
                              cfg.thread_per_block, 0, d.stream(), cfg,       \
                              outbuf));                                       \
  CUDA_ASSERT_SUCCESS                                                         \
  cfg = GetGpuLaunchConfig(work_element_count, d);                            \
  TF_CHECK_OK(GpuLaunchKernel(Count1D, cfg.block_count, cfg.thread_per_block, \
                              0, d.stream(), cfg, bufsize, outbuf));          \
  CUDA_EXPECT_SUCCESS                                                         \
  copyToHost();                                                               \
  EXPECT_EQ(work_element_count,                                               \
            std::accumulate(outbuf_host, outbuf_host + bufsize, 0));          \
                                                                              \
  cfg = GetGpuLaunchConfig(bufsize, d, SetOutbufZero, 0, 0);                  \
  TF_CHECK_OK(GpuLaunchKernel(SetOutbufZero, cfg.block_count,                 \
                              cfg.thread_per_block, 0, d.stream(), cfg,       \
                              outbuf));                                       \
  CUDA_ASSERT_SUCCESS                                                         \
  cfg = GetGpuLaunchConfig(work_element_count, d, Count1D, 0, 0);             \
  TF_CHECK_OK(GpuLaunchKernel(Count1D, cfg.block_count, cfg.thread_per_block, \
                              0, d.stream(), cfg, bufsize, outbuf));          \
  CUDA_EXPECT_SUCCESS                                                         \
  copyToHost();                                                               \
  EXPECT_EQ(work_element_count,                                               \
            std::accumulate(outbuf_host, outbuf_host + bufsize, 0));

  TEST_LAUNCH_PARAMETER(128);
  TEST_LAUNCH_PARAMETER(129);
  TEST_LAUNCH_PARAMETER(511);
  TEST_LAUNCH_PARAMETER(512);
  TEST_LAUNCH_PARAMETER(2048);
  TEST_LAUNCH_PARAMETER(2049);
  TEST_LAUNCH_PARAMETER(8191);
  TEST_LAUNCH_PARAMETER(8192);
  TEST_LAUNCH_PARAMETER(123456);
  TEST_LAUNCH_PARAMETER(1 << 30);
#undef TEST_LAUNCH_PARAMETER
}

bool operator==(const Gpu2DLaunchConfig& a, const Gpu2DLaunchConfig& b) {
  return a.thread_per_block.x == b.thread_per_block.x &&
         a.thread_per_block.y == b.thread_per_block.y &&
         a.thread_per_block.z == b.thread_per_block.z &&
         a.block_count.x == b.block_count.x &&
         a.block_count.y == b.block_count.y &&
         a.block_count.z == b.block_count.z &&
         a.thread_per_block.x == b.thread_per_block.x &&
         a.thread_per_block.y == b.thread_per_block.y &&
         a.thread_per_block.z == b.thread_per_block.z;
}

TEST_F(GpuLaunchConfigTest, GetGpu2DLaunchConfig) {
  Gpu2DLaunchConfig cfg;
  GpuLaunchConfig cfg1d;

// test valid inputs
#define TEST_LAUNCH_PARAMETER(dimx, dimy)                                      \
  cfg1d = GetGpuLaunchConfig(bufsize, d);                                      \
  TF_EXPECT_OK(GpuLaunchKernel(SetOutbufZero, cfg1d.block_count,               \
                               cfg1d.thread_per_block, 0, d.stream(), cfg1d,   \
                               outbuf));                                       \
  CUDA_ASSERT_SUCCESS                                                          \
  cfg = GetGpu2DLaunchConfig(dimx, dimy, d);                                   \
  TF_EXPECT_OK(GpuLaunchKernel(Count2D, cfg.block_count, cfg.thread_per_block, \
                               0, d.stream(), cfg, bufsize, outbuf));          \
  CUDA_EXPECT_SUCCESS                                                          \
  copyToHost();                                                                \
  EXPECT_EQ(dimx* dimy,                                                        \
            std::accumulate(outbuf_host, outbuf_host + bufsize, 0));           \
                                                                               \
  cfg1d = GetGpuLaunchConfig(bufsize, d, SetOutbufZero, 0, 0);                 \
  TF_EXPECT_OK(GpuLaunchKernel(SetOutbufZero, cfg1d.block_count,               \
                               cfg1d.thread_per_block, 0, d.stream(), cfg1d,   \
                               outbuf));                                       \
  CUDA_ASSERT_SUCCESS                                                          \
  cfg = GetGpu2DLaunchConfig(dimx, dimy, d, Count2D, 0, 0);                    \
  TF_EXPECT_OK(GpuLaunchKernel(Count2D, cfg.block_count, cfg.thread_per_block, \
                               0, d.stream(), cfg, bufsize, outbuf));          \
  CUDA_EXPECT_SUCCESS                                                          \
  copyToHost();                                                                \
  EXPECT_EQ(dimx* dimy, std::accumulate(outbuf_host, outbuf_host + bufsize, 0))

  TEST_LAUNCH_PARAMETER(128, 128);
  TEST_LAUNCH_PARAMETER(129, 64);
  TEST_LAUNCH_PARAMETER(511, 2048);
  TEST_LAUNCH_PARAMETER(512, 512);
  TEST_LAUNCH_PARAMETER(2048, 1024);
  TEST_LAUNCH_PARAMETER(2049, 32);
  TEST_LAUNCH_PARAMETER(8191, 1);
  TEST_LAUNCH_PARAMETER(8192, 10);
  TEST_LAUNCH_PARAMETER(123456, 12);
  TEST_LAUNCH_PARAMETER(1, 1 << 30);
  TEST_LAUNCH_PARAMETER(1 << 30, 1);
#undef TEST_LAUNCH_PARAMETER
}

TEST_F(GpuLaunchConfigTest, GetGpu3DLaunchConfig) {
  Gpu3DLaunchConfig cfg;
  GpuLaunchConfig cfg1d;

// test valid inputs
#define TEST_LAUNCH_PARAMETER(dimx, dimy, dimz)                                \
  cfg1d = GetGpuLaunchConfig(bufsize, d, SetOutbufZero, 0, 0);                 \
  TF_EXPECT_OK(GpuLaunchKernel(SetOutbufZero, cfg1d.block_count,               \
                               cfg1d.thread_per_block, 0, d.stream(), cfg1d,   \
                               outbuf));                                       \
  CUDA_ASSERT_SUCCESS                                                          \
  cfg = GetGpu3DLaunchConfig(dimx, dimy, dimz, d, Count3D, 0, 0);              \
  TF_EXPECT_OK(GpuLaunchKernel(Count3D, cfg.block_count, cfg.thread_per_block, \
                               0, d.stream(), cfg, bufsize, outbuf));          \
  CUDA_EXPECT_SUCCESS                                                          \
  copyToHost();                                                                \
  EXPECT_EQ(dimx* dimy* dimz,                                                  \
            std::accumulate(outbuf_host, outbuf_host + bufsize, 0))

  TEST_LAUNCH_PARAMETER(128, 128, 128);
  TEST_LAUNCH_PARAMETER(129, 64, 1024);
  TEST_LAUNCH_PARAMETER(511, 2048, 128);
  TEST_LAUNCH_PARAMETER(512, 512, 64);
  TEST_LAUNCH_PARAMETER(2048, 1024, 128);
  TEST_LAUNCH_PARAMETER(2049, 32, 1024);
  TEST_LAUNCH_PARAMETER(8191, 1, 1024);
  TEST_LAUNCH_PARAMETER(8192, 10, 32);
  TEST_LAUNCH_PARAMETER(123456, 12, 21);
  TEST_LAUNCH_PARAMETER(1, 1, 1 << 30);
  TEST_LAUNCH_PARAMETER(1, 1 << 30, 1);
  TEST_LAUNCH_PARAMETER(1 << 30, 1, 1);
#undef TEST_LAUNCH_PARAMETER
}

TEST(CudaDeviceFunctionsTest, ShuffleGetSrcLane) {
  unsigned* failure_count;
#if GOOGLE_CUDA
  ASSERT_EQ(cudaMallocManaged(&failure_count, sizeof(unsigned)), cudaSuccess);
#else
  ASSERT_EQ(hipHostMalloc(&failure_count, sizeof(unsigned), 0), cudaSuccess);
#endif
  *failure_count = 0;
  TF_EXPECT_OK(GpuLaunchKernel(GpuShuffleGetSrcLaneTest, 1, TF_RED_WARPSIZE, 0,
                               nullptr, failure_count));
  ASSERT_EQ(gpuDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(*failure_count, 0);
  gpuFree(failure_count);
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
