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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gatherer_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gatherer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gatherer_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/ring_gatherer.h"

#include <algorithm>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/collective_test_util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/test_collective_executor_mgr.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

class RingGathererTest : public ::testing::Test {
 protected:
  void Init(int num_workers, int num_devices, DataType dtype,
            const TensorShape& shape, const DeviceType& device_type,
            int num_subdivs, int fail_after) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gatherer_testDTcc mht_0(mht_0_v, 218, "", "./tensorflow/core/common_runtime/ring_gatherer_test.cc", "Init");

    test_env_ = CreateCollectiveTestEnv(num_workers, num_devices, device_type);
    test_env_->remote_access->set_fail_after(fail_after);
    for (int wi = 0; wi < num_workers; ++wi) {
      for (int di = 0; di < num_devices; ++di) {
        int rank = wi * num_devices + di;
        instances_.push_back(absl::make_unique<DeviceInstance>(
            rank, num_subdivs, dtype, shape, test_env_.get()));
      }
    }
  }

  void Gather(int fail_after) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gatherer_testDTcc mht_1(mht_1_v, 233, "", "./tensorflow/core/common_runtime/ring_gatherer_test.cc", "Gather");

    std::atomic<int> done(0);
    for (auto& di : instances_) {
      SchedClosure([&di, &done] {
        di->DoGather();
        ++done;
      });
      if (fail_after > 0) {
        // Stagger the op execution starts.
        Env::Default()->SleepForMicroseconds(100);
      }
    }
    while (done < static_cast<int>(instances_.size())) {
      Env::Default()->SleepForMicroseconds(1000);
    }
  }

  template <typename T>
  void RunTest(DataType dtype, const DeviceType& device_type, int num_workers,
               int num_devices, int num_subdivs, int tensor_len,
               int fail_after) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gatherer_testDTcc mht_2(mht_2_v, 256, "", "./tensorflow/core/common_runtime/ring_gatherer_test.cc", "RunTest");

    Init(num_workers, num_devices, dtype, TensorShape({tensor_len}),
         device_type, num_subdivs, fail_after);
    int32_t output_len = tensor_len * num_workers * num_devices;
    std::vector<T> expected(output_len, 0.0);
    for (int di = 0; di < static_cast<int>(instances_.size()); ++di) {
      int32_t instance_offset = di * tensor_len;
      instances_[di]->InitTensor(
          [instance_offset, &expected, dtype, di](Tensor* t) {
            for (size_t i = 0; i < t->NumElements(); ++i) {
              // The cast is necessary to prevent clang-tidy
              // from insisting that a faster non-open source
              // function be substituted.
              float value = pow(10, static_cast<double>(di)) * i;
              if (dtype == DT_INT32 || dtype == DT_INT64) {
                value = di * 10 + i;
              }
              t->flat<T>()(i) = static_cast<T>(value);
              expected[instance_offset + i] = value;
            }
          });
    }
    Gather(fail_after);
    if (fail_after > 0) {
      // Confirm that every device terminated with the expected error status.
      for (int di = 0; di < static_cast<int>(instances_.size()); ++di) {
        EXPECT_NE(
            instances_[di]->status_.error_message().find("Deliberate failure"),
            string::npos);
      }
    } else {
      // Confirm that every device accumulated the same set of correct
      // values.
      for (int di = 0; di < static_cast<int>(instances_.size()); ++di) {
        TF_EXPECT_OK(instances_[di]->status_);
        test::ExpectTensorEqual<T>(test::AsTensor<T>(expected),
                                   instances_[di]->output_tensor());
      }
    }
  }

  class DeviceInstance {
   public:
    DeviceInstance(int rank, int num_subdivs, DataType dtype,
                   const TensorShape& shape, CollectiveTestEnv* test_env)
        : test_env_(test_env), input_tensor_(dtype, shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gatherer_testDTcc mht_3(mht_3_v, 304, "", "./tensorflow/core/common_runtime/ring_gatherer_test.cc", "DeviceInstance");

      col_params_ = CreateCollectiveParams(*test_env_, rank, "RingGather",
                                           GATHER_COLLECTIVE, dtype, shape);
      if (num_subdivs > 0) {
        col_params_->instance.impl_details.subdiv_offsets =
            GenerateEvenSubdivOffsets(test_env->num_devices_per_worker,
                                      num_subdivs);
      }
      string dev_name = col_params_->group.members[rank].device.name();
      TF_CHECK_OK(test_env_->device_mgr->LookupDevice(dev_name, &device_))
          << "Couldn't find device " << dev_name
          << " existing devices: " << test_env_->device_mgr->DebugString();
      TensorShape output_shape = shape;
      output_shape.set_dim(
          0, output_shape.dim_size(0) * col_params_->group.group_size);
      output_tensor_ = Tensor(dtype, output_shape);
    }

    void InitTensor(const std::function<void(Tensor*)>& init_f) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gatherer_testDTcc mht_4(mht_4_v, 325, "", "./tensorflow/core/common_runtime/ring_gatherer_test.cc", "InitTensor");

      init_f(&input_tensor_);
    }

    void DoGather() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gatherer_testDTcc mht_5(mht_5_v, 332, "", "./tensorflow/core/common_runtime/ring_gatherer_test.cc", "DoGather");

      status_ = RunCollective(test_env_, col_params_.get(), device_,
                              &input_tensor_, &output_tensor_);
    }

    const Tensor& input_tensor() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gatherer_testDTcc mht_6(mht_6_v, 340, "", "./tensorflow/core/common_runtime/ring_gatherer_test.cc", "input_tensor");
 return input_tensor_; }
    const Tensor& output_tensor() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gatherer_testDTcc mht_7(mht_7_v, 344, "", "./tensorflow/core/common_runtime/ring_gatherer_test.cc", "output_tensor");
 return output_tensor_; }

    CollectiveTestEnv* test_env_;
    Tensor input_tensor_;
    Tensor output_tensor_;
    Device* device_;
    core::RefCountPtr<CollectiveParams> col_params_;
    Status status_;
  };

  std::unique_ptr<CollectiveTestEnv> test_env_;
  std::vector<std::unique_ptr<DeviceInstance>> instances_;
};

class RingGathererInitParamsTest : public ::testing::Test {
 protected:
  void RunSubdivPermsTest(
      CollectiveParams* cp,
      const std::vector<std::vector<int>>& expected_subdiv_perms,
      const std::vector<int>& expected_subdiv_rank) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gatherer_testDTcc mht_8(mht_8_v, 366, "", "./tensorflow/core/common_runtime/ring_gatherer_test.cc", "RunSubdivPermsTest");

    cp->instance.impl_details.subdiv_permutations.clear();
    cp->subdiv_rank.clear();
    // Create a stub ring reducer only for testing param initialization.
    core::RefCountPtr<RingGatherer> gatherer(new RingGatherer());
    TF_CHECK_OK(gatherer->InitializeCollectiveParams(cp));
    EXPECT_EQ(expected_subdiv_perms,
              cp->instance.impl_details.subdiv_permutations);
    EXPECT_EQ(expected_subdiv_rank, cp->subdiv_rank);
  }
};

TEST_F(RingGathererInitParamsTest, SpecifiedSubdivs) {
  const int kNumDevsPerWorker = 8;
  const int kNumWorkers = 3;
  auto test_env =
      CreateCollectiveTestEnv(kNumWorkers, kNumDevsPerWorker, DEVICE_CPU);
  auto cp =
      CreateCollectiveParams(*test_env, /*rank*/ 0, "RingGather",
                             GATHER_COLLECTIVE, DT_FLOAT, TensorShape({1}));

  cp->default_rank = 0;
  cp->instance.impl_details.subdiv_offsets = {};
  RunSubdivPermsTest(cp.get(),
                     {{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                       12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}},
                     {0});

  cp->instance.impl_details.subdiv_offsets = {0};
  RunSubdivPermsTest(cp.get(),
                     {{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                       12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}},
                     {0});

  cp->default_rank = 3;
  cp->instance.impl_details.subdiv_offsets = {};
  RunSubdivPermsTest(cp.get(),
                     {{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                       12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}},
                     {3});
}

// TODO(b/113171733): change to use TEST_P.
#define DEF_TEST(B, T, W, D, S, L, A)                                         \
  TEST_F(RingGathererTest,                                                    \
         DaTy##B##_DevTy##T##_Wkr##W##_Dev##D##_Sdiv##S##_Len##L##_Abrt##A) { \
    DataType dtype = DT_##B;                                                  \
    switch (dtype) {                                                          \
      case DT_FLOAT: {                                                        \
        RunTest<float>(dtype, DEVICE_##T, W, D, S, L, A);                     \
      } break;                                                                \
      case DT_DOUBLE: {                                                       \
        RunTest<double>(dtype, DEVICE_##T, W, D, S, L, A);                    \
      } break;                                                                \
      case DT_INT32: {                                                        \
        RunTest<int32>(dtype, DEVICE_##T, W, D, S, L, A);                     \
      } break;                                                                \
      case DT_INT64: {                                                        \
        RunTest<int64_t>(dtype, DEVICE_##T, W, D, S, L, A);                   \
      } break;                                                                \
      default:                                                                \
        LOG(FATAL) << "Unimplemented";                                        \
    }                                                                         \
  }

#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
// Success tests
DEF_TEST(FLOAT, CPU, 1, 2, 1, 1, 0)
DEF_TEST(FLOAT, CPU, 1, 2, 1, 2, 0)
DEF_TEST(FLOAT, CPU, 1, 2, 1, 8, 0)
DEF_TEST(FLOAT, CPU, 1, 2, 1, 16, 0)
DEF_TEST(FLOAT, CPU, 1, 2, 1, 1001, 0)
DEF_TEST(FLOAT, CPU, 2, 4, 1, 128, 0)
DEF_TEST(FLOAT, CPU, 2, 8, 1, 1001, 0)
DEF_TEST(FLOAT, CPU, 2, 8, 1, 4096, 0)
DEF_TEST(FLOAT, CPU, 2, 8, 1, 9408, 0)
DEF_TEST(FLOAT, CPU, 4, 4, 1, 32768, 0)
DEF_TEST(DOUBLE, CPU, 1, 2, 1, 1001, 0)
DEF_TEST(DOUBLE, CPU, 2, 8, 1, 4095, 0)
DEF_TEST(INT32, CPU, 1, 2, 1, 1001, 0)
DEF_TEST(INT32, CPU, 2, 8, 1, 4095, 0)
DEF_TEST(INT64, CPU, 1, 2, 1, 1001, 0)
DEF_TEST(INT64, CPU, 2, 8, 1, 4095, 0)

// Failure tests
DEF_TEST(FLOAT, CPU, 2, 8, 1, 9408, 1)
DEF_TEST(FLOAT, CPU, 2, 8, 1, 9408, 7)
DEF_TEST(FLOAT, CPU, 2, 8, 1, 9408, 11)
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// GPU tests.  So long as the device names are all in a single tasks we
// bypass inter-worker routing code and can fake multiple GPUs with a single
// GPU, from the perspective of the RingGatherer logic.  So these tests
// are all single-worker.
DEF_TEST(FLOAT, GPU, 1, 2, 1, 1, 0)
DEF_TEST(FLOAT, GPU, 1, 2, 1, 2, 0)
DEF_TEST(FLOAT, GPU, 1, 2, 1, 8, 0)
DEF_TEST(FLOAT, GPU, 1, 2, 1, 16, 0)
DEF_TEST(FLOAT, GPU, 1, 2, 1, 1001, 0)
DEF_TEST(FLOAT, GPU, 1, 8, 1, 1001, 0)
DEF_TEST(FLOAT, GPU, 1, 8, 1, 4096, 0)
DEF_TEST(FLOAT, GPU, 1, 8, 1, 4095, 0)
DEF_TEST(FLOAT, GPU, 1, 8, 1, 32768, 0)
DEF_TEST(FLOAT, GPU, 1, 4, 1, 32768, 0)
DEF_TEST(DOUBLE, GPU, 1, 2, 1, 1001, 0)
// INT32 values are never on the GPU.
// DEF_TEST(INT32, GPU, 1, 1, 1, 1001, 0)
DEF_TEST(INT64, GPU, 1, 2, 1, 1001, 0)

// Failure tests
DEF_TEST(FLOAT, GPU, 1, 8, 1, 9408, 2)
DEF_TEST(FLOAT, GPU, 1, 8, 1, 9408, 5)
#endif

}  // namespace tensorflow
