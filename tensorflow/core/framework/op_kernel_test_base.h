/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_OP_KERNEL_TEST_BASE_H_
#define TENSORFLOW_CORE_FRAMEWORK_OP_KERNEL_TEST_BASE_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSop_kernel_test_baseDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernel_test_baseDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSop_kernel_test_baseDTh() {
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


#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

static std::vector<DeviceType> DeviceTypes() {
  return {DeviceType(DEVICE_GPU), DeviceType(DEVICE_CPU)};
}

class OpKernelBuilderTest : public ::testing::Test {
 protected:
  // Each attr is described by a "name|type|value".
  NodeDef CreateNodeDef(const string& op_type,
                        const std::vector<string>& attrs) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("op_type: \"" + op_type + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernel_test_baseDTh mht_0(mht_0_v, 227, "", "./tensorflow/core/framework/op_kernel_test_base.h", "CreateNodeDef");

    NodeDef node_def;
    node_def.set_name(op_type + "-op");
    node_def.set_op(op_type);
    for (const string& attr_desc : attrs) {
      std::vector<string> parts = str_util::Split(attr_desc, '|');
      CHECK_EQ(parts.size(), 3);
      AttrValue attr_value;
      CHECK(ParseAttrValue(parts[1], parts[2], &attr_value)) << attr_desc;
      node_def.mutable_attr()->insert(
          AttrValueMap::value_type(parts[0], attr_value));
    }
    return node_def;
  }

  std::unique_ptr<OpKernel> ExpectSuccess(const string& op_type,
                                          const DeviceType& device_type,
                                          const std::vector<string>& attrs,
                                          DataTypeSlice input_types = {}) {
    Status status;
    NodeDef def = CreateNodeDef(op_type, attrs);
    for (size_t i = 0; i < input_types.size(); ++i) {
      def.add_input("a:0");
    }

    Env* env = Env::Default();
    DeviceBase device(env);

    // Test CreateOpKernel()
    std::unique_ptr<OpKernel> op(CreateOpKernel(device_type, &device,
                                                cpu_allocator(), def,
                                                TF_GRAPH_DEF_VERSION, &status));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(op != nullptr);
    if (op != nullptr) {
      EXPECT_EQ(input_types.size(), op->num_inputs());
      EXPECT_EQ(0, op->num_outputs());
    }

    // Test SupportedDeviceTypesForNode()
    PrioritizedDeviceTypeVector devices;
    TF_EXPECT_OK(SupportedDeviceTypesForNode(DeviceTypes(), def, &devices));
    bool found = false;
    for (const auto& dt : devices) {
      if (dt.first == device_type) {
        found = true;
      }
    }
    EXPECT_TRUE(found) << "Missing " << device_type << " from "
                       << devices.size() << " devices.";

    // In case the caller wants to use the OpKernel
    return op;
  }

  void ExpectFailure(const string& op_type, const DeviceType& device_type,
                     const std::vector<string>& attrs, error::Code code) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("op_type: \"" + op_type + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernel_test_baseDTh mht_1(mht_1_v, 287, "", "./tensorflow/core/framework/op_kernel_test_base.h", "ExpectFailure");

    Status status;
    const NodeDef def = CreateNodeDef(op_type, attrs);
    Env* env = Env::Default();
    DeviceBase device(env);

    // Test CreateOpKernel().
    std::unique_ptr<OpKernel> op(CreateOpKernel(device_type, &device,
                                                cpu_allocator(), def,
                                                TF_GRAPH_DEF_VERSION, &status));
    EXPECT_TRUE(op == nullptr);
    EXPECT_FALSE(status.ok());
    if (!status.ok()) {
      LOG(INFO) << "Status message: " << status.error_message();
      EXPECT_EQ(code, status.code());

      // Test SupportedDeviceTypesForNode().
      PrioritizedDeviceTypeVector devices;
      if (errors::IsNotFound(status)) {
        TF_EXPECT_OK(SupportedDeviceTypesForNode(DeviceTypes(), def, &devices));
        for (const auto& dt : devices) {
          EXPECT_NE(dt.first, device_type);
        }
      } else {
        Status status2 =
            SupportedDeviceTypesForNode(DeviceTypes(), def, &devices);
        EXPECT_EQ(status.code(), status2.code());
      }
    }
  }

  string GetKernelClassName(const string& op_type,
                            const DeviceType& device_type,
                            const std::vector<string>& attrs,
                            DataTypeSlice input_types = {}) {
    NodeDef def = CreateNodeDef(op_type, attrs);
    for (size_t i = 0; i < input_types.size(); ++i) {
      def.add_input("a:0");
    }

    const KernelDef* kernel_def = nullptr;
    string kernel_class_name;
    const Status status =
        FindKernelDef(device_type, def, &kernel_def, &kernel_class_name);
    if (status.ok()) {
      return kernel_class_name;
    } else if (errors::IsNotFound(status)) {
      return "not found";
    } else {
      return status.ToString();
    }
  }
};

class BaseKernel : public ::tensorflow::OpKernel {
 public:
  explicit BaseKernel(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernel_test_baseDTh mht_2(mht_2_v, 346, "", "./tensorflow/core/framework/op_kernel_test_base.h", "BaseKernel");
}
  void Compute(::tensorflow::OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernel_test_baseDTh mht_3(mht_3_v, 350, "", "./tensorflow/core/framework/op_kernel_test_base.h", "Compute");
}
  virtual int Which() const = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_OP_KERNEL_TEST_BASE_H_
