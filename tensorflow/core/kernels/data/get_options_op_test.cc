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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSget_options_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSget_options_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSget_options_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/get_options_op.h"

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/options_dataset_op.h"
#include "tensorflow/core/kernels/data/range_dataset_op.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kOptions[] = R"proto(
  deterministic: true
  slack: true
  optimization_options { apply_default_optimizations: true autotune: true }
  distribute_options {}
)proto";

class GetOptionsParams : public DatasetParams {
 public:
  template <typename T>
  GetOptionsParams(T input_dataset_params, DataTypeVector output_dtypes,
                   std::vector<PartialTensorShape> output_shapes,
                   string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSget_options_op_testDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/kernels/data/get_options_op_test.cc", "GetOptionsParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
  }

  std::vector<Tensor> GetInputTensors() const override { return {}; }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSget_options_op_testDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/kernels/data/get_options_op_test.cc", "GetInputNames");

    input_names->emplace_back(OptionsDatasetOp::kInputDataset);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSget_options_op_testDTcc mht_2(mht_2_v, 226, "", "./tensorflow/core/kernels/data/get_options_op_test.cc", "GetAttributes");

    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSget_options_op_testDTcc mht_3(mht_3_v, 233, "", "./tensorflow/core/kernels/data/get_options_op_test.cc", "dataset_type");
 return "GetOptions"; }

  string op_name() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSget_options_op_testDTcc mht_4(mht_4_v, 238, "", "./tensorflow/core/kernels/data/get_options_op_test.cc", "op_name");
 return dataset_type(); }

 private:
  string serialized_options_;
};

class GetOptionsOpTest : public DatasetOpsTestBase {};

OptionsDatasetParams OptionsDatasetParams0() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSget_options_op_testDTcc mht_5(mht_5_v, 249, "", "./tensorflow/core/kernels/data/get_options_op_test.cc", "OptionsDatasetParams0");

  Options options;
  protobuf::TextFormat::ParseFromString(kOptions, &options);
  return OptionsDatasetParams(RangeDatasetParams(0, 10, 3),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_0");
}

GetOptionsParams GetOptionsParams0() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSget_options_op_testDTcc mht_6(mht_6_v, 262, "", "./tensorflow/core/kernels/data/get_options_op_test.cc", "GetOptionsParams0");

  return GetOptionsParams(OptionsDatasetParams0(),
                          /*output_dtypes=*/{DT_INT64},
                          /*output_shapes=*/{PartialTensorShape({})},
                          /*node_name=*/"get_options_0");
}

TEST_F(GetOptionsOpTest, Compute) {
  auto test_case_params = GetOptionsParams0();
  TF_ASSERT_OK(InitializeRuntime(test_case_params));
  std::vector<Tensor> output;
  TF_ASSERT_OK(RunDatasetOp(test_case_params, &output));
  EXPECT_EQ(1, output.size());
  Options options;
  protobuf::TextFormat::ParseFromString(kOptions, &options);
  Tensor expected_tensor =
      CreateTensor<tstring>(TensorShape({}), {options.SerializeAsString()});
  Tensor result_tensor = output[0];
  string serialized_options = result_tensor.scalar<tstring>()();
  Options result_options;
  result_options.ParseFromString(serialized_options);
  TF_EXPECT_OK(ExpectEqual(expected_tensor, result_tensor));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
