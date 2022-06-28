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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_LOCAL_CLIENT_TEST_BASE_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_LOCAL_CLIENT_TEST_BASE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTh() {
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


#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/local_service.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/manifest_checking_test.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {

class TestAllocator : public se::StreamExecutorMemoryAllocator {
 public:
  explicit TestAllocator(se::Platform* platform)
      : se::StreamExecutorMemoryAllocator(
            platform, PlatformUtil::GetStreamExecutors(platform).ValueOrDie()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTh mht_0(mht_0_v, 217, "", "./tensorflow/compiler/xla/tests/local_client_test_base.h", "TestAllocator");

  }

  StatusOr<se::OwningDeviceMemory> Allocate(int device_ordinal, uint64_t size,
                                            bool retry_on_failure,
                                            int64_t memory_space) override;
  Status Deallocate(int device_ordinal, se::DeviceMemoryBase mem) override;

  // Return the number of allocations that have been performed.
  int64_t allocation_count() const;
  int64_t allocation_count(int device_ordinal) const;

  // Return the number of deallocations that have been performed.
  int64_t deallocation_count() const;
  int64_t deallocation_count(int device_ordinal) const;

 private:
  mutable absl::Mutex count_mutex_;

  // Global counts of allocations and deallocations.
  int64_t allocation_count_ ABSL_GUARDED_BY(count_mutex_) = 0;
  int64_t deallocation_count_ ABSL_GUARDED_BY(count_mutex_) = 0;

  // Per-device counts of allocations and deallocations.
  std::map<int, int64_t> device_allocation_count_ ABSL_GUARDED_BY(count_mutex_);
  std::map<int, int64_t> device_deallocation_count_
      ABSL_GUARDED_BY(count_mutex_);
};

// A base class for tests which exercise the LocalClient interface.
class LocalClientTestBase : public ManifestCheckingTest {
 protected:
  struct EigenThreadPoolWrapper;
  explicit LocalClientTestBase(se::Platform* platform = nullptr);
  virtual ~LocalClientTestBase();

  static TestAllocator* GetOrCreateAllocator(se::Platform* platform);

  // Copy the given literal onto the default device and return a
  // ScopedShapedBuffer. Convenience wrapper around
  // LocalClient::LiteralToShapedBuffer.
  ScopedShapedBuffer LiteralToShapedBuffer(const Literal& literal);

  // Construct and return a literal containing the array represented by
  // shaped_buffer.
  Literal ShapedBufferToLiteral(const ShapedBuffer& shaped_buffer);

  // Execute the given computation on the local client. With and without
  // options.
  StatusOr<ScopedShapedBuffer> ExecuteLocally(
      const XlaComputation& computation,
      absl::Span<const ShapedBuffer* const> arguments);
  StatusOr<ScopedShapedBuffer> ExecuteLocally(
      const XlaComputation& computation,
      absl::Span<const ShapedBuffer* const> arguments,
      const ExecutableBuildOptions& build_options,
      const ExecutableRunOptions& run_options);

  ScopedShapedBuffer ExecuteLocallyOrDie(
      const XlaComputation& computation,
      absl::Span<const ShapedBuffer* const> arguments);
  ScopedShapedBuffer ExecuteLocallyOrDie(
      const XlaComputation& computation,
      absl::Span<const ShapedBuffer* const> arguments,
      const ExecutableBuildOptions& build_options,
      const ExecutableRunOptions& run_options);

  // Parses the given string and returns module as a VerifiedHloModule.
  StatusOr<std::unique_ptr<VerifiedHloModule>> ParseAndReturnVerifiedModule(
      absl::string_view hlo_text);
  StatusOr<std::unique_ptr<VerifiedHloModule>> ParseAndReturnVerifiedModule(
      absl::string_view hlo_text, const HloModuleConfig& config);

  // Returns a default set of execute options.
  ExecutableBuildOptions DefaultExecutableBuildOptions() const;

  // Returns a default set of execute options, configured to use allocator_
  // as the allocator.
  ExecutableRunOptions DefaultExecutableRunOptions() const;

  std::string TestName() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTh mht_1(mht_1_v, 300, "", "./tensorflow/compiler/xla/tests/local_client_test_base.h", "TestName");

    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  // The allocator must live as long as the service, which lives until the end
  // of the process. So make the allocator static.
  static TestAllocator* allocator_;

  se::StreamExecutor* stream_executor_;
  TransferManager* transfer_manager_;

  LocalClient* local_client_;

  std::unique_ptr<EigenThreadPoolWrapper> thread_pool_wrapper_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_LOCAL_CLIENT_TEST_BASE_H_
