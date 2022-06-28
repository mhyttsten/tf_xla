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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_manager_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_manager_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_manager_testDTcc() {
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

#if GOOGLE_CUDA

#include "tensorflow/core/profiler/internal/gpu/cupti_error_manager.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/internal/gpu/cuda_test.h"
#include "tensorflow/core/profiler/internal/gpu/cupti_interface.h"
#include "tensorflow/core/profiler/internal/gpu/cupti_tracer.h"
#include "tensorflow/core/profiler/internal/gpu/cupti_wrapper.h"
#include "tensorflow/core/profiler/internal/gpu/mock_cupti.h"
#include "tensorflow/core/profiler/utils/time_utils.h"

namespace tensorflow {
namespace profiler {
namespace test {

using tensorflow::profiler::CuptiInterface;
using tensorflow::profiler::CuptiTracer;
using tensorflow::profiler::CuptiTracerCollectorOptions;
using tensorflow::profiler::CuptiTracerOptions;
using tensorflow::profiler::CuptiWrapper;

using ::testing::_;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::Sequence;
using ::testing::StrictMock;

// Needed to create different cupti tracer for each test cases.
class TestableCuptiTracer : public CuptiTracer {
 public:
  explicit TestableCuptiTracer(CuptiInterface* cupti_interface)
      : CuptiTracer(cupti_interface) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_manager_testDTcc mht_0(mht_0_v, 222, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager_test.cc", "TestableCuptiTracer");
}
};

// CuptiErrorManagerTest verifies that an application is not killed due to an
// unexpected error in the underlying GPU hardware during tracing.
// MockCupti is used to simulate a CUPTI call failure.
class CuptiErrorManagerTest : public ::testing::Test {
 protected:
  CuptiErrorManagerTest() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_manager_testDTcc mht_1(mht_1_v, 233, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager_test.cc", "CuptiErrorManagerTest");
}

  void SetUp() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_manager_testDTcc mht_2(mht_2_v, 238, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager_test.cc", "SetUp");

    ASSERT_GT(CuptiTracer::NumGpus(), 0) << "No devices found";
    auto mock_cupti = absl::make_unique<StrictMock<MockCupti>>();
    mock_ = mock_cupti.get();
    cupti_error_manager_ =
        absl::make_unique<CuptiErrorManager>(std::move(mock_cupti));

    cupti_tracer_ =
        absl::make_unique<TestableCuptiTracer>(cupti_error_manager_.get());
    cupti_wrapper_ = absl::make_unique<CuptiWrapper>();

    CuptiTracerCollectorOptions collector_options;
    collector_options.num_gpus = CuptiTracer::NumGpus();
    uint64_t start_gputime_ns = CuptiTracer::GetTimestamp();
    uint64_t start_walltime_ns = tensorflow::profiler::GetCurrentTimeNanos();
    cupti_collector_ = CreateCuptiCollector(
        collector_options, start_walltime_ns, start_gputime_ns);
  }

  void EnableProfiling(const CuptiTracerOptions& option) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_manager_testDTcc mht_3(mht_3_v, 260, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager_test.cc", "EnableProfiling");

    cupti_tracer_->Enable(option, cupti_collector_.get());
  }

  void DisableProfiling() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_manager_testDTcc mht_4(mht_4_v, 267, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager_test.cc", "DisableProfiling");
 cupti_tracer_->Disable(); }

  bool CuptiDisabled() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_manager_testDTcc mht_5(mht_5_v, 272, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager_test.cc", "CuptiDisabled");
 return cupti_error_manager_->Disabled(); }

  void RunGpuApp() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_manager_testDTcc mht_6(mht_6_v, 277, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager_test.cc", "RunGpuApp");

    MemCopyH2D();
    PrintfKernel(/*iters=*/10);
    Synchronize();
    MemCopyD2H();
  }

  // Pointer to MockCupti passed to CuptiBase constructor.
  // Used to inject failures to be handled by CuptiErrorManager.
  // Wrapped in StrictMock so unexpected calls cause a test failure.
  StrictMock<MockCupti>* mock_;

  // CuptiTracer instance that uses MockCupti instead of CuptiWrapper.
  std::unique_ptr<TestableCuptiTracer> cupti_tracer_ = nullptr;

  std::unique_ptr<CuptiInterface> cupti_error_manager_;

  // CuptiWrapper instance to which mock_ calls are delegated.
  std::unique_ptr<CuptiWrapper> cupti_wrapper_;

  std::unique_ptr<tensorflow::profiler::CuptiTraceCollector> cupti_collector_;
};

// Verifies that failed EnableProfiling() does not kill an application.
TEST_F(CuptiErrorManagerTest, GpuTraceActivityEnableTest) {
  // Enforces the order of execution below.
  Sequence s1;
  // CuptiBase::EnableProfiling()
  EXPECT_CALL(*mock_, Subscribe(_, _, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::Subscribe));
  EXPECT_CALL(*mock_, EnableCallback(1, _, _, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::EnableCallback));
  EXPECT_CALL(*mock_, ActivityRegisterCallbacks(_, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(),
                       &CuptiWrapper::ActivityRegisterCallbacks));
  EXPECT_CALL(*mock_, ActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_ERROR_UNKNOWN));  // injected error
  // CuptiErrorManager::ResultString()
  EXPECT_CALL(*mock_, GetResultString(CUPTI_ERROR_UNKNOWN, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::GetResultString));
  // CuptiErrorManager::UndoAndDisable()
  EXPECT_CALL(*mock_, EnableCallback(0, _, _, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::EnableCallback));
  EXPECT_CALL(*mock_, Unsubscribe(_))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::Unsubscribe));

  EXPECT_FALSE(CuptiDisabled());
  CuptiTracerOptions options;
  options.activities_selected.push_back(CUPTI_ACTIVITY_KIND_KERNEL);
  options.cbids_selected.push_back(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
  EnableProfiling(options);  // CUPTI call fails due to injected error
  EXPECT_TRUE(CuptiDisabled());

  RunGpuApp();  // Application code runs normally

  EXPECT_TRUE(CuptiDisabled());
  DisableProfiling();  // CUPTI calls are ignored
  EXPECT_TRUE(CuptiDisabled());
}

// Verifies that failed EnableProfiling() does not kill an application.
TEST_F(CuptiErrorManagerTest, GpuTraceAutoEnableTest) {
  EXPECT_FALSE(CuptiDisabled());
  // Enforces the order of execution below.
  Sequence s1;
  EXPECT_CALL(*mock_, Subscribe(_, _, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::Subscribe));
  EXPECT_CALL(*mock_, EnableDomain(1, _, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::EnableDomain));
  EXPECT_CALL(*mock_, ActivityRegisterCallbacks(_, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(),
                       &CuptiWrapper::ActivityRegisterCallbacks));
  EXPECT_CALL(*mock_, ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::ActivityEnable));
  EXPECT_CALL(*mock_, ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_ERROR_UNKNOWN));  // injected error
  // CuptiErrorManager::ResultString()
  EXPECT_CALL(*mock_, GetResultString(CUPTI_ERROR_UNKNOWN, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::GetResultString));
  // CuptiErrorManager::UndoAndDisable()
  EXPECT_CALL(*mock_, ActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::ActivityDisable));
  EXPECT_CALL(*mock_, EnableDomain(0, _, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::EnableDomain));
  EXPECT_CALL(*mock_, Unsubscribe(_))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::Unsubscribe));

  EXPECT_FALSE(CuptiDisabled());
  CuptiTracerOptions options;
  options.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMCPY);
  options.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMCPY2);
  options.activities_selected.push_back(CUPTI_ACTIVITY_KIND_KERNEL);
  // options.cbids_selected.push_back(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
  EnableProfiling(options);  // CUPTI call fails due to injected error
  EXPECT_TRUE(CuptiDisabled());

  RunGpuApp();  // Application code runs normally

  EXPECT_TRUE(CuptiDisabled());
  DisableProfiling();  // CUPTI calls are ignored
  EXPECT_TRUE(CuptiDisabled());
}

}  // namespace test
}  // namespace profiler
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
