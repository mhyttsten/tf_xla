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
class MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cache_testDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cache_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cache_testDTcc() {
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
#include "tensorflow/core/tfrt/eager/op_cache.h"

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/tfrt/eager/c_api_tfrt.h"
#include "tfrt/cpu/core_runtime/null_op_handler.h"  // from @tf_runtime

namespace tfrt {
namespace tf {
namespace {

constexpr char device_name[] = "/job:localhost/replica:0/task:0/device:CPU:0";
constexpr char op_name[] = "Add";
constexpr char dtype[] = "DT_INT8";

class OpCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSop_cache_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/tfrt/eager/op_cache_test.cc", "SetUp");

    // Set up context.
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    TFE_ContextOptionsSetTfrt(opts, /*use_tfrt=*/true);
    tensorflow::AbstractContext* ctx_raw = nullptr;
    ctx_raw =
        tensorflow::unwrap(TF_NewEagerExecutionContext(opts, status.get()));
    tensorflow::Status s = tensorflow::StatusFromTF_Status(status.get());
    ASSERT_TRUE(s.ok());
    TFE_DeleteContextOptions(opts);
    ctx_.reset(ctx_raw);

    // Set up operation.
    auto op_interface_ptr =
        tensorflow::down_cast<::tfrt::tf::OperationInterface*>(
            ctx_->CreateOperation());
    op_interface_.reset(op_interface_ptr);
    ASSERT_TRUE(op_interface_->Reset(op_name, device_name).ok());
    ASSERT_TRUE(op_interface_->SetAttrType("T", tensorflow::DT_INT8).ok());
  }

  tensorflow::AbstractContextPtr ctx_;
  std::unique_ptr<::tfrt::tf::OperationInterface> op_interface_;
  ::tfrt::tf::OpCache cache_;
};

TEST_F(OpCacheTest, TestOpCacheInitiallyEmpty) {
  // Cache is empty initially.
  EXPECT_EQ(cache_.Size(), 0);
  EXPECT_FALSE(
      cache_.Contains(op_name, /*op_handler=*/nullptr, device_name, {dtype}));
}

TEST_F(OpCacheTest, TestOpCacheCacheHit) {
  auto expected_op =
      cache_.GetOrAddOp(op_name, /*op_handler=*/nullptr, device_name, {dtype},
                        op_interface_.get());
  // Inserts a new cache entry.
  EXPECT_TRUE((bool)expected_op) << StrCat(expected_op.takeError());

  // There's one entry in the cache.
  EXPECT_EQ(cache_.Size(), 1);
  EXPECT_TRUE(
      cache_.Contains(op_name, /*op_handler=*/nullptr, device_name, {dtype}));

  // This lookup is a cache hit.
  expected_op = cache_.GetOrAddOp(op_name, /*op_handler=*/nullptr, device_name,
                                  {dtype}, op_interface_.get());
  EXPECT_TRUE((bool)expected_op) << StrCat(expected_op.takeError());

  // Cache hit doesn't create new entry in the cache.
  EXPECT_EQ(cache_.Size(), 1);
}

TEST_F(OpCacheTest, TestOpCacheCacheDeviceNameNotSpecifiedAndCacheMiss) {
  // Inserts a new cache entry.
  auto expected_op =
      cache_.GetOrAddOp(op_name, /*op_handler=*/nullptr, device_name, {dtype},
                        op_interface_.get());
  EXPECT_TRUE((bool)expected_op) << StrCat(expected_op.takeError());

  // Inserts a op with empty device name. This incurs a cache miss.
  expected_op =
      cache_.GetOrAddOp(op_name, /*op_handler=*/nullptr, /*device_name=*/"",
                        {dtype}, op_interface_.get());
  EXPECT_TRUE((bool)expected_op) << StrCat(expected_op.takeError());

  // The eager placer (OpHandlerSelector) picks a device for the op.
  EXPECT_STREQ(expected_op.get()->DeviceName().str().c_str(), device_name);

  // This is a cache miss and will insert a new entry to the cache.
  EXPECT_EQ(cache_.Size(), 2);

  // Inserts a op with another dtype. This incurs a cache miss.
  expected_op =
      cache_.GetOrAddOp(op_name, /*op_handler=*/nullptr, /*device_name=*/"",
                        {"F64"}, op_interface_.get());
  EXPECT_TRUE((bool)expected_op) << StrCat(expected_op.takeError());

  // This is a cache miss and will insert a new entry to the cache.
  EXPECT_EQ(cache_.Size(), 3);
}

TEST_F(OpCacheTest, TestOpCacheAlreadyPlaced) {
  auto* op_handler =
      tensorflow::down_cast<::tfrt::tf::ContextInterface*>(ctx_.get())
          ->GetCoreRuntime()
          ->GetOpHandler(device_name);
  EXPECT_TRUE(op_handler != nullptr);
  // Inserts a new cache entry.
  auto expected_op = cache_.GetOrAddOp(op_name, op_handler, device_name,
                                       {dtype}, op_interface_.get());
  EXPECT_TRUE((bool)expected_op) << StrCat(expected_op.takeError());

  // The lookup is a cache hit.
  expected_op = cache_.GetOrAddOp(op_name, op_handler, device_name, {dtype},
                                  op_interface_.get());
  EXPECT_TRUE((bool)expected_op) << StrCat(expected_op.takeError());

  EXPECT_EQ(cache_.Size(), 1);
}

}  // namespace
}  // namespace tf
}  // namespace tfrt
