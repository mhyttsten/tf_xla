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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSfused_batchnorm_reserve_space_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfused_batchnorm_reserve_space_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSfused_batchnorm_reserve_space_testDTcc() {
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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {
Status GetTestDevice(Session* session, string* test_device) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfused_batchnorm_reserve_space_testDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/tf2xla/fused_batchnorm_reserve_space_test.cc", "GetTestDevice");

  std::vector<DeviceAttributes> devices;
  TF_RETURN_IF_ERROR(session->ListDevices(&devices));

  bool found_cpu = absl::c_any_of(devices, [&](const DeviceAttributes& device) {
    return device.device_type() == "CPU";
  });

  bool found_gpu = absl::c_any_of(devices, [&](const DeviceAttributes& device) {
    return device.device_type() == "GPU";
  });

  if (!found_gpu && !found_cpu) {
    return errors::Internal("Expected at least one CPU or GPU!");
  }

  *test_device = found_gpu ? "GPU" : "CPU";
  VLOG(2) << "Using test device " << *test_device;
  return Status::OK();
}

void FillZeros(Tensor* tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfused_batchnorm_reserve_space_testDTcc mht_1(mht_1_v, 238, "", "./tensorflow/compiler/tf2xla/fused_batchnorm_reserve_space_test.cc", "FillZeros");

  auto flat = tensor->flat<float>();
  for (int i = 0; i < flat.size(); i++) {
    flat.data()[i] = 0.0f;
  }
}

// This tests check that the implementation outputs from FusedBatchnorm
// training, reserve_space_{1|2}, are what we assume them to be in the TF/XLA
// lowering.
//
// If this test starts failing then it doesn't indicate that TF/cudnn have
// violated their contract, but it indicates that we need to update the TF/XLA
// lowering for FusedBatchnorm training to match the new implementation defined
// behavior.
TEST(FusedBatchnormReserveSpaceTest, Test) {
  using ::tensorflow::ops::Const;
  using ::tensorflow::ops::FusedBatchNorm;

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions{}));

  string test_device;
  TF_ASSERT_OK(GetTestDevice(session.get(), &test_device));

  Scope root = tensorflow::Scope::NewRootScope();
  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);

  Tensor scale_data(DT_FLOAT, TensorShape({10}));
  FillZeros(&scale_data);
  Output scale =
      Const(root.WithOpName("scale"), Input::Initializer(scale_data));

  Tensor offset_data(DT_FLOAT, TensorShape({10}));
  FillZeros(&offset_data);
  Output offset =
      Const(root.WithOpName("offset"), Input::Initializer(offset_data));

  Tensor mean_data(DT_FLOAT, TensorShape({0}));
  Output mean = Const(root.WithOpName("offset"), Input::Initializer(mean_data));

  Tensor variance_data(DT_FLOAT, TensorShape({0}));
  Output variance =
      Const(root.WithOpName("variance"), Input::Initializer(variance_data));

  string tf_device = absl::StrCat("/device:", test_device, ":0");
  string xla_device = absl::StrCat("/device:XLA_", test_device, ":0");

  FusedBatchNorm fused_batch_norm_tf(
      root.WithOpName("fused_batch_norm_tf").WithDevice(tf_device), input,
      scale, offset, mean, variance, FusedBatchNorm::Attrs{}.IsTraining(true));
  FusedBatchNorm fused_batch_norm_xla(
      root.WithOpName("fused_batch_norm_xla").WithDevice(xla_device), input,
      scale, offset, mean, variance, FusedBatchNorm::Attrs{}.IsTraining(true));

  tensorflow::GraphDef graph;
  TF_ASSERT_OK(root.ToGraphDef(&graph));

  TF_ASSERT_OK(session->Create(graph));

  Tensor input_data(DT_FLOAT, TensorShape({10, 10, 10, 10}));
  auto flat_input = input_data.flat<float>();
  for (int i = 0; i < flat_input.size(); i++) {
    flat_input.data()[i] = (i - 5) / 1000.0f;
  }

  std::vector<Tensor> results;
  TF_ASSERT_OK(session->Run({{"input", input_data}},
                            {fused_batch_norm_tf.reserve_space_1.name(),
                             fused_batch_norm_xla.reserve_space_1.name(),
                             fused_batch_norm_tf.reserve_space_2.name(),
                             fused_batch_norm_xla.reserve_space_2.name()},
                            {}, &results));

  test::ExpectClose(results[0], results[1], /*atol=*/1e-4);
  test::ExpectClose(results[2], results[3], /*atol=*/1e-4);
}

static bool Initialized = [] {
  tensorflow::GetXlaDeviceFlags()->tf_xla_enable_xla_devices = true;
  return true;
}();

}  // namespace
}  // namespace tensorflow
