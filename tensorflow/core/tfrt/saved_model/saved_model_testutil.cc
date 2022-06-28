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
class MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_model_testutilDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_model_testutilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_model_testutilDTcc() {
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
#include "tensorflow/core/tfrt/saved_model/saved_model_testutil.h"

#include <functional>
#include <string>

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

ABSL_FLAG(bool, enable_optimizer, true,
          "enable optimizations in CoreRT dialect (e.g., constant-folding)");
ABSL_FLAG(std::string, force_data_format, "",
          "force data format for all layout sensitive operations. Currently "
          "the supported formats are 'NHWC' and 'NCHW'");

ABSL_FLAG(bool, enable_native_ops, true,
          "If true, native ops will be used if they are implemented in TFRT. "
          "If false, all ops are using fallback.");

ABSL_FLAG(
    bool, enable_grappler, false,
    "If true, run grappler passes before importing the SavedModel into MLIR.");

namespace tensorflow {
namespace tfrt_stub {

std::unique_ptr<tensorflow::tfrt_stub::Runtime> DefaultTfrtRuntime(
    int num_threads) {
  return tensorflow::tfrt_stub::Runtime::Create(
      tensorflow::tfrt_stub::WrapDefaultWorkQueue(
          tfrt::CreateMultiThreadedWorkQueue(num_threads, num_threads)));
}

SavedModel::Options DefaultSavedModelOptions(
    tensorflow::tfrt_stub::Runtime* runtime) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_model_testutilDTcc mht_0(mht_0_v, 218, "", "./tensorflow/core/tfrt/saved_model/saved_model_testutil.cc", "DefaultSavedModelOptions");

  SavedModel::Options options(runtime);
  auto& compile_options = options.graph_execution_options.compile_options;
  compile_options.enable_optimizer = absl::GetFlag(FLAGS_enable_optimizer);
  compile_options.enable_native_ops = absl::GetFlag(FLAGS_enable_native_ops);
  compile_options.enable_grappler = absl::GetFlag(FLAGS_enable_grappler);
  compile_options.force_data_format = absl::GetFlag(FLAGS_force_data_format);
  return options;
}

TFRTSavedModelTest::TFRTSavedModelTest(const std::string& saved_model_dir)
    : TFRTSavedModelTest(saved_model_dir,
                         DefaultTfrtRuntime(/*num_threads=*/1)) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("saved_model_dir: \"" + saved_model_dir + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_model_testutilDTcc mht_1(mht_1_v, 234, "", "./tensorflow/core/tfrt/saved_model/saved_model_testutil.cc", "TFRTSavedModelTest::TFRTSavedModelTest");
}

TFRTSavedModelTest::TFRTSavedModelTest(
    const std::string& saved_model_dir,
    std::unique_ptr<tensorflow::tfrt_stub::Runtime> runtime)
    : runtime_(std::move(runtime)) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("saved_model_dir: \"" + saved_model_dir + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_model_testutilDTcc mht_2(mht_2_v, 243, "", "./tensorflow/core/tfrt/saved_model/saved_model_testutil.cc", "TFRTSavedModelTest::TFRTSavedModelTest");

  CHECK(runtime_);
  auto options = DefaultSavedModelOptions(runtime_.get());

  tensorflow::Status status;
  saved_model_ = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                /*tags=*/{"serve"}, &status);
  TF_DCHECK_OK(status);
}

// Compute the results using TF1 session loaded from the saved model. In
// addition to returning the result tensors, it also fills `bundle` with the
// loaded savedmodel. This is useful as sometimes the result tensors may only be
// valid when the bundle is alive.
void ComputeCurrentTFResult(const std::string& saved_model_dir,
                            const std::string& signature_name,
                            const std::vector<std::string>& input_names,
                            const std::vector<tensorflow::Tensor>& inputs,
                            const std::vector<std::string>& output_names,
                            std::vector<tensorflow::Tensor>* outputs,
                            tensorflow::SavedModelBundle* bundle,
                            bool enable_mlir_bridge, bool disable_grappler) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("saved_model_dir: \"" + saved_model_dir + "\"");
   mht_3_v.push_back("signature_name: \"" + signature_name + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_model_testutilDTcc mht_3(mht_3_v, 269, "", "./tensorflow/core/tfrt/saved_model/saved_model_testutil.cc", "ComputeCurrentTFResult");

  DCHECK(bundle);
  tensorflow::SessionOptions session_options;
  session_options.config.mutable_experimental()->set_enable_mlir_bridge(
      enable_mlir_bridge);
  // Disable grappler optimization for numerical analysis.
  if (disable_grappler) {
    session_options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_disable_meta_optimizer(true);
  }
  TF_CHECK_OK(tensorflow::LoadSavedModel(session_options, /*run_options=*/{},
                                         saved_model_dir,
                                         /* tags = */ {"serve"}, bundle));

  const auto& signature_def =
      bundle->meta_graph_def.signature_def().at(signature_name);

  std::vector<std::pair<std::string, tensorflow::Tensor>> session_inputs;
  session_inputs.reserve(inputs.size());
  for (const auto& iter : llvm::zip(input_names, inputs)) {
    const auto& node_name = signature_def.inputs().at(std::get<0>(iter)).name();
    session_inputs.emplace_back(node_name, std::get<1>(iter));
  }

  std::vector<std::string> session_output_names;
  session_output_names.reserve(output_names.size());
  for (const auto& output_name : output_names) {
    const auto& node_name = signature_def.outputs().at(output_name).name();
    session_output_names.push_back(node_name);
  }

  TF_CHECK_OK(bundle->GetSession()->Run(session_inputs, session_output_names,
                                        {}, outputs));
}

void ComputeCurrentTFResult(const std::string& saved_model_dir,
                            const std::string& signature_name,
                            const std::vector<std::string>& input_names,
                            const std::vector<tensorflow::Tensor>& inputs,
                            const std::vector<std::string>& output_names,
                            std::vector<tensorflow::Tensor>* outputs,
                            bool enable_mlir_bridge, bool disable_grappler) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("saved_model_dir: \"" + saved_model_dir + "\"");
   mht_4_v.push_back("signature_name: \"" + signature_name + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_model_testutilDTcc mht_4(mht_4_v, 316, "", "./tensorflow/core/tfrt/saved_model/saved_model_testutil.cc", "ComputeCurrentTFResult");

  tensorflow::SavedModelBundle bundle;
  ComputeCurrentTFResult(saved_model_dir, signature_name, input_names, inputs,
                         output_names, outputs, &bundle, enable_mlir_bridge,
                         disable_grappler);
}

void ExpectTensorEqual(const tensorflow::Tensor& x, const tensorflow::Tensor& y,
                       absl::optional<double> error) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_model_testutilDTcc mht_5(mht_5_v, 327, "", "./tensorflow/core/tfrt/saved_model/saved_model_testutil.cc", "ExpectTensorEqual");

  DCHECK_EQ(x.dtype(), y.dtype());
  VLOG(1) << "TFRT result: " << x.DebugString();
  VLOG(1) << "TF result  : " << y.DebugString();
  switch (y.dtype()) {
    case tensorflow::DT_STRING:
      tensorflow::test::ExpectTensorEqual<tensorflow::tstring>(x, y);
      break;
    case tensorflow::DT_FLOAT:
    case tensorflow::DT_DOUBLE:
      if (error) {
        tensorflow::test::ExpectClose(x, y, *error, /*rtol=*/0.0);
      } else {
        tensorflow::test::ExpectEqual(x, y);
      }
      break;
    default:
      tensorflow::test::ExpectEqual(x, y);
      break;
  }
}

}  // namespace tfrt_stub
}  // namespace tensorflow
