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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc() {
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
#define EIGEN_USE_THREADS

#include "tensorflow/compiler/xla/tests/local_client_test_base.h"

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

/* static */ TestAllocator* LocalClientTestBase::allocator_;

StatusOr<se::OwningDeviceMemory> TestAllocator::Allocate(int device_ordinal,
                                                         uint64_t size,
                                                         bool retry_on_failure,
                                                         int64_t memory_space) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_0(mht_0_v, 216, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "TestAllocator::Allocate");

  VLOG(2) << "Allocate(" << device_ordinal << ", " << size << ")";
  {
    absl::MutexLock lock(&count_mutex_);
    allocation_count_++;
    device_allocation_count_[device_ordinal]++;
  }
  return se::StreamExecutorMemoryAllocator::Allocate(
      device_ordinal, size, retry_on_failure, memory_space);
}

Status TestAllocator::Deallocate(int device_ordinal, se::DeviceMemoryBase mem) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_1(mht_1_v, 230, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "TestAllocator::Deallocate");

  VLOG(2) << "Deallocate(" << device_ordinal << ")";
  {
    absl::MutexLock lock(&count_mutex_);
    deallocation_count_++;
    device_deallocation_count_[device_ordinal]++;
  }
  return se::StreamExecutorMemoryAllocator::Deallocate(device_ordinal, mem);
}

int64_t TestAllocator::allocation_count() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_2(mht_2_v, 243, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "TestAllocator::allocation_count");

  absl::MutexLock lock(&count_mutex_);
  return allocation_count_;
}

int64_t TestAllocator::allocation_count(int device_ordinal) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_3(mht_3_v, 251, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "TestAllocator::allocation_count");

  absl::MutexLock lock(&count_mutex_);
  auto it = device_allocation_count_.find(device_ordinal);
  if (it == device_allocation_count_.end()) {
    return 0;
  } else {
    return it->second;
  }
}

int64_t TestAllocator::deallocation_count() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_4(mht_4_v, 264, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "TestAllocator::deallocation_count");

  absl::MutexLock lock(&count_mutex_);
  return deallocation_count_;
}

int64_t TestAllocator::deallocation_count(int device_ordinal) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_5(mht_5_v, 272, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "TestAllocator::deallocation_count");

  absl::MutexLock lock(&count_mutex_);
  auto it = device_deallocation_count_.find(device_ordinal);
  if (it == device_deallocation_count_.end()) {
    return 0;
  } else {
    return it->second;
  }
}

/* static */ TestAllocator* LocalClientTestBase::GetOrCreateAllocator(
    se::Platform* platform) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_6(mht_6_v, 286, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "LocalClientTestBase::GetOrCreateAllocator");

  static absl::Mutex mu(absl::kConstInit);
  absl::MutexLock lock(&mu);

  if (allocator_ == nullptr) {
    allocator_ = new TestAllocator(
        platform == nullptr ? PlatformUtil::GetDefaultPlatform().ValueOrDie()
                            : platform);
  }
  return allocator_;
}

// Define this in .cc file to avoid having to include eigen or forward declare
// these types in the header.
struct LocalClientTestBase::EigenThreadPoolWrapper {
  explicit EigenThreadPoolWrapper()
      : pool(new tensorflow::thread::ThreadPool(
            tensorflow::Env::Default(), "XLAEigenTest", /*num_threads=*/2)),
        device(new Eigen::ThreadPoolDevice(pool->AsEigenThreadPool(),
                                           pool->NumThreads())) {}

  std::unique_ptr<tensorflow::thread::ThreadPool> pool;
  std::unique_ptr<Eigen::ThreadPoolDevice> device;
};

LocalClientTestBase::LocalClientTestBase(se::Platform* platform)
    : local_client_(
          ClientLibrary::GetOrCreateLocalClient(platform).ValueOrDie()),
      thread_pool_wrapper_(new EigenThreadPoolWrapper()) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_7(mht_7_v, 317, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "LocalClientTestBase::LocalClientTestBase");

  // Take the first executor, since it's the default one.
  stream_executor_ = PlatformUtil::GetStreamExecutors(local_client_->platform())
                         .ValueOrDie()
                         .front();
  transfer_manager_ =
      TransferManager::GetForPlatform(local_client_->platform()).ValueOrDie();
}

LocalClientTestBase::~LocalClientTestBase() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_8(mht_8_v, 329, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "LocalClientTestBase::~LocalClientTestBase");
}

ScopedShapedBuffer LocalClientTestBase::LiteralToShapedBuffer(
    const Literal& literal) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_9(mht_9_v, 335, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "LocalClientTestBase::LiteralToShapedBuffer");

  return local_client_
      ->LiteralToShapedBuffer(literal, local_client_->default_device_ordinal())
      .ConsumeValueOrDie();
}

Literal LocalClientTestBase::ShapedBufferToLiteral(
    const ShapedBuffer& shaped_buffer) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_10(mht_10_v, 345, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "LocalClientTestBase::ShapedBufferToLiteral");

  return local_client_->ShapedBufferToLiteral(shaped_buffer)
      .ConsumeValueOrDie();
}

ExecutableBuildOptions LocalClientTestBase::DefaultExecutableBuildOptions()
    const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_11(mht_11_v, 354, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "LocalClientTestBase::DefaultExecutableBuildOptions");

  return ExecutableBuildOptions();
}

ExecutableRunOptions LocalClientTestBase::DefaultExecutableRunOptions() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_12(mht_12_v, 361, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "LocalClientTestBase::DefaultExecutableRunOptions");

  ExecutableRunOptions run_options;
  run_options.set_intra_op_thread_pool(thread_pool_wrapper_->device.get());
  run_options.set_allocator(GetOrCreateAllocator(local_client_->platform()));
  return run_options;
}

ScopedShapedBuffer LocalClientTestBase::ExecuteLocallyOrDie(
    const XlaComputation& computation,
    absl::Span<const ShapedBuffer* const> arguments) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_13(mht_13_v, 373, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "LocalClientTestBase::ExecuteLocallyOrDie");

  return ExecuteLocally(computation, arguments, DefaultExecutableBuildOptions(),
                        DefaultExecutableRunOptions())
      .ConsumeValueOrDie();
}

ScopedShapedBuffer LocalClientTestBase::ExecuteLocallyOrDie(
    const XlaComputation& computation,
    absl::Span<const ShapedBuffer* const> arguments,
    const ExecutableBuildOptions& build_options,
    const ExecutableRunOptions& run_options) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_14(mht_14_v, 386, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "LocalClientTestBase::ExecuteLocallyOrDie");

  return ExecuteLocally(computation, arguments, build_options, run_options)
      .ConsumeValueOrDie();
}

StatusOr<ScopedShapedBuffer> LocalClientTestBase::ExecuteLocally(
    const XlaComputation& computation,
    absl::Span<const ShapedBuffer* const> arguments) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_15(mht_15_v, 396, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "LocalClientTestBase::ExecuteLocally");

  return ExecuteLocally(computation, arguments, DefaultExecutableBuildOptions(),
                        DefaultExecutableRunOptions());
}

StatusOr<ScopedShapedBuffer> LocalClientTestBase::ExecuteLocally(
    const XlaComputation& computation,
    absl::Span<const ShapedBuffer* const> arguments,
    const ExecutableBuildOptions& build_options,
    const ExecutableRunOptions& run_options) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_16(mht_16_v, 408, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "LocalClientTestBase::ExecuteLocally");

  std::vector<const Shape*> argument_layouts(arguments.size());
  for (int i = 0; i < arguments.size(); ++i) {
    argument_layouts[i] = &arguments[i]->on_device_shape();
  }
  TF_ASSIGN_OR_RETURN(
      auto executables,
      local_client_->Compile(computation, argument_layouts, build_options));
  TF_RET_CHECK(executables.size() == 1);
  TF_ASSIGN_OR_RETURN(auto ret, executables[0]->Run(arguments, run_options));

  auto device_ordinal =
      build_options.device_ordinal() == -1 ? 0 : build_options.device_ordinal();
  auto* stream = run_options.stream();
  if (!stream) {
    stream = local_client_->mutable_backend()
                 ->BorrowStream(device_ordinal)
                 .ValueOrDie()
                 .get();
  }
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return std::move(ret);
}

StatusOr<std::unique_ptr<VerifiedHloModule>>
LocalClientTestBase::ParseAndReturnVerifiedModule(absl::string_view hlo_text) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("hlo_text: \"" + std::string(hlo_text.data(), hlo_text.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_17(mht_17_v, 437, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "LocalClientTestBase::ParseAndReturnVerifiedModule");

  return ParseAndReturnVerifiedModule(hlo_text, HloModuleConfig());
}

StatusOr<std::unique_ptr<VerifiedHloModule>>
LocalClientTestBase::ParseAndReturnVerifiedModule(
    absl::string_view hlo_text, const HloModuleConfig& config) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("hlo_text: \"" + std::string(hlo_text.data(), hlo_text.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSlocal_client_test_baseDTcc mht_18(mht_18_v, 447, "", "./tensorflow/compiler/xla/tests/local_client_test_base.cc", "LocalClientTestBase::ParseAndReturnVerifiedModule");

  auto module = absl::make_unique<VerifiedHloModule>(
      TestName(), config, /*verifier_layout_sensitive=*/false,
      /*allow_mixed_precision_in_hlo_verifier=*/true,
      local_client_->backend().compiler()->ShapeSizeBytesFunction());
  TF_RETURN_IF_ERROR(module->ParseHloStringAndVerifyModule(hlo_text));
  return std::move(module);
}

}  // namespace xla
