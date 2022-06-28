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
class MHTracer_DTPStensorflowPScorePSframeworkPSresource_op_kernel_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_op_kernel_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSresource_op_kernel_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/resource_op_kernel.h"

#include <memory>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

// Stub DeviceBase subclass which only returns allocators.
class StubDevice : public DeviceBase {
 public:
  StubDevice() : DeviceBase(nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_op_kernel_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/framework/resource_op_kernel_test.cc", "StubDevice");
}

  Allocator* GetAllocator(AllocatorAttributes) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_op_kernel_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/framework/resource_op_kernel_test.cc", "GetAllocator");

    return cpu_allocator();
  }
};

// Stub resource for testing resource op kernel.
class StubResource : public ResourceBase {
 public:
  string DebugString() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_op_kernel_testDTcc mht_2(mht_2_v, 224, "", "./tensorflow/core/framework/resource_op_kernel_test.cc", "DebugString");
 return ""; }
  int code;
};

class StubResourceOpKernel : public ResourceOpKernel<StubResource> {
 public:
  using ResourceOpKernel::ResourceOpKernel;

  StubResource* resource() TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    return resource_;
  }

 private:
  Status CreateResource(StubResource** resource) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_op_kernel_testDTcc mht_3(mht_3_v, 241, "", "./tensorflow/core/framework/resource_op_kernel_test.cc", "CreateResource");

    *resource = CHECK_NOTNULL(new StubResource);
    return GetNodeAttr(def(), "code", &(*resource)->code);
  }

  Status VerifyResource(StubResource* resource) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_op_kernel_testDTcc mht_4(mht_4_v, 249, "", "./tensorflow/core/framework/resource_op_kernel_test.cc", "VerifyResource");

    int code;
    TF_RETURN_IF_ERROR(GetNodeAttr(def(), "code", &code));
    if (code != resource->code) {
      return errors::InvalidArgument("stub has code ", resource->code,
                                     " but requested code ", code);
    }
    return Status::OK();
  }
};

REGISTER_OP("StubResourceOp")
    .Attr("code: int")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("output: Ref(string)");

REGISTER_KERNEL_BUILDER(Name("StubResourceOp").Device(DEVICE_CPU),
                        StubResourceOpKernel);

class ResourceOpKernelTest : public ::testing::Test {
 protected:
  std::unique_ptr<StubResourceOpKernel> CreateOp(int code,
                                                 const string& shared_name) {
    static std::atomic<int64_t> count(0);
    NodeDef node_def;
    TF_CHECK_OK(NodeDefBuilder(strings::StrCat("test-node", count.fetch_add(1)),
                               "StubResourceOp")
                    .Attr("code", code)
                    .Attr("shared_name", shared_name)
                    .Finalize(&node_def));
    Status status;
    std::unique_ptr<OpKernel> op(CreateOpKernel(
        DEVICE_CPU, &device_, device_.GetAllocator(AllocatorAttributes()),
        node_def, TF_GRAPH_DEF_VERSION, &status));
    TF_EXPECT_OK(status) << status;
    EXPECT_TRUE(op != nullptr);

    // Downcast to StubResourceOpKernel to call resource() later.
    std::unique_ptr<StubResourceOpKernel> resource_op(
        dynamic_cast<StubResourceOpKernel*>(op.get()));
    EXPECT_TRUE(resource_op != nullptr);
    if (resource_op != nullptr) {
      op.release();
    }
    return resource_op;
  }

  Status RunOpKernel(OpKernel* op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_op_kernel_testDTcc mht_5(mht_5_v, 300, "", "./tensorflow/core/framework/resource_op_kernel_test.cc", "RunOpKernel");

    OpKernelContext::Params params;

    params.device = &device_;
    params.resource_manager = &mgr_;
    params.op_kernel = op;

    OpKernelContext context(&params);
    op->Compute(&context);
    return context.status();
  }

  StubDevice device_;
  ResourceMgr mgr_;
};

TEST_F(ResourceOpKernelTest, PrivateResource) {
  // Empty shared_name means private resource.
  const int code = -100;
  auto op = CreateOp(code, "");
  ASSERT_TRUE(op != nullptr);
  TF_EXPECT_OK(RunOpKernel(op.get()));

  // Default non-shared name provided from ContainerInfo.
  // TODO(gonnet): This test is brittle since it assumes that the
  // ResourceManager is untouched and thus the private resource name starts
  // with "_0_".
  const string key = "_0_" + op->name();

  StubResource* resource;
  TF_ASSERT_OK(
      mgr_.Lookup<StubResource>(mgr_.default_container(), key, &resource));
  EXPECT_EQ(op->resource(), resource);  // Check resource identity.
  EXPECT_EQ(code, resource->code);      // Check resource stored information.
  resource->Unref();

  // Destroy the op kernel. Expect the resource to be released.
  op = nullptr;
  Status s =
      mgr_.Lookup<StubResource>(mgr_.default_container(), key, &resource);

  EXPECT_FALSE(s.ok());
}

TEST_F(ResourceOpKernelTest, SharedResource) {
  const string shared_name = "shared_stub";
  const int code = -201;
  auto op = CreateOp(code, shared_name);
  ASSERT_TRUE(op != nullptr);
  TF_EXPECT_OK(RunOpKernel(op.get()));

  StubResource* resource;
  TF_ASSERT_OK(mgr_.Lookup<StubResource>(mgr_.default_container(), shared_name,
                                         &resource));
  EXPECT_EQ(op->resource(), resource);  // Check resource identity.
  EXPECT_EQ(code, resource->code);      // Check resource stored information.
  resource->Unref();

  // Destroy the op kernel. Expect the resource not to be released.
  op = nullptr;
  TF_ASSERT_OK(mgr_.Lookup<StubResource>(mgr_.default_container(), shared_name,
                                         &resource));
  resource->Unref();
}

TEST_F(ResourceOpKernelTest, LookupShared) {
  auto op1 = CreateOp(-333, "shared_stub");
  auto op2 = CreateOp(-333, "shared_stub");
  ASSERT_TRUE(op1 != nullptr);
  ASSERT_TRUE(op2 != nullptr);

  TF_EXPECT_OK(RunOpKernel(op1.get()));
  TF_EXPECT_OK(RunOpKernel(op2.get()));
  EXPECT_EQ(op1->resource(), op2->resource());
}

TEST_F(ResourceOpKernelTest, VerifyResource) {
  auto op1 = CreateOp(-444, "shared_stub");
  auto op2 = CreateOp(0, "shared_stub");  // Different resource code.
  ASSERT_TRUE(op1 != nullptr);
  ASSERT_TRUE(op2 != nullptr);

  TF_EXPECT_OK(RunOpKernel(op1.get()));
  EXPECT_FALSE(RunOpKernel(op2.get()).ok());
  EXPECT_TRUE(op1->resource() != nullptr);
  EXPECT_TRUE(op2->resource() == nullptr);
}

}  // namespace
}  // namespace tensorflow
