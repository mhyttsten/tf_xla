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
#ifndef TENSORFLOW_TFRT_SAVED_MODEL_SAVED_MODEL_TESTUTIL_H_
#define TENSORFLOW_TFRT_SAVED_MODEL_SAVED_MODEL_TESTUTIL_H_
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
class MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_model_testutilDTh {
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
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_model_testutilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_model_testutilDTh() {
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


#include <stdlib.h>

#include <limits>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/tensor/btf_util.h"  // from @tf_runtime

ABSL_DECLARE_FLAG(bool, enable_optimizer);
ABSL_DECLARE_FLAG(std::string, force_data_format);

namespace tensorflow {
namespace tfrt_stub {

std::unique_ptr<tensorflow::tfrt_stub::Runtime> DefaultTfrtRuntime(
    int num_threads);

SavedModel::Options DefaultSavedModelOptions(
    tensorflow::tfrt_stub::Runtime* runtime);

class TFRTSavedModelTest {
 public:
  explicit TFRTSavedModelTest(const std::string& saved_model_dir);
  TFRTSavedModelTest(const std::string& saved_model_dir,
                     std::unique_ptr<tensorflow::tfrt_stub::Runtime> runtime);

  SavedModel* GetSavedModel() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_model_testutilDTh mht_0(mht_0_v, 215, "", "./tensorflow/core/tfrt/saved_model/saved_model_testutil.h", "GetSavedModel");
 return saved_model_.get(); }

  tfrt::HostContext* GetHostContext() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_model_testutilDTh mht_1(mht_1_v, 220, "", "./tensorflow/core/tfrt/saved_model/saved_model_testutil.h", "GetHostContext");

    return saved_model_->GetHostContext();
  }

 private:
  std::unique_ptr<tensorflow::tfrt_stub::Runtime> runtime_;
  std::unique_ptr<SavedModel> saved_model_;
};

template <typename T, typename U = T>
tensorflow::Tensor CreateTfTensor(absl::Span<const int64_t> shape,
                                  absl::Span<const U> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_model_testutilDTh mht_2(mht_2_v, 234, "", "./tensorflow/core/tfrt/saved_model/saved_model_testutil.h", "CreateTfTensor");

  tensorflow::Tensor tensor(tensorflow::DataTypeToEnum<T>::value,
                            tensorflow::TensorShape(shape));
  auto flat = tensor.flat<T>();
  for (int i = 0; i < data.size(); ++i) {
    flat(i) = data[i];
  }
  return tensor;
}

template <typename T>
std::vector<T> GetTfTensorData(const tensorflow::Tensor& tensor) {
  return std::vector<T>(tensor.flat<T>().data(),
                        tensor.flat<T>().data() + tensor.NumElements());
}

inline tensorflow::Tensor CreateTfStringTensor(
    absl::Span<const int64_t> shape, absl::Span<const std::string> data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSsaved_modelPSsaved_model_testutilDTh mht_3(mht_3_v, 254, "", "./tensorflow/core/tfrt/saved_model/saved_model_testutil.h", "CreateTfStringTensor");

  return CreateTfTensor<tensorflow::tstring>(shape, data);
}

void ComputeCurrentTFResult(const std::string& saved_model_dir,
                            const std::string& signature_name,
                            const std::vector<std::string>& input_names,
                            const std::vector<tensorflow::Tensor>& inputs,
                            const std::vector<std::string>& output_names,
                            std::vector<tensorflow::Tensor>* outputs,
                            bool enable_mlir_bridge = false,
                            bool disable_grappler = false);

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
                            bool enable_mlir_bridge = false,
                            bool disable_grappler = false);

void ExpectTensorEqual(const tensorflow::Tensor& x, const tensorflow::Tensor& y,
                       absl::optional<double> error = absl::nullopt);

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_TFRT_SAVED_MODEL_SAVED_MODEL_TESTUTIL_H_
