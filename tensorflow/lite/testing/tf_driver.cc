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
class MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc {
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
   MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc() {
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
#include "tensorflow/lite/testing/tf_driver.h"

#include <fstream>
#include <iostream>

#include "absl/strings/escaping.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/join.h"
#include "tensorflow/lite/testing/split.h"

namespace tflite {
namespace testing {

namespace {

tensorflow::Tensor CreateTensor(const tensorflow::DataType type,
                                const std::vector<int64_t>& dim) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/testing/tf_driver.cc", "CreateTensor");

  tensorflow::TensorShape shape{tensorflow::gtl::ArraySlice<int64_t>{
      reinterpret_cast<const int64_t*>(dim.data()), dim.size()}};
  return {type, shape};
}

template <typename T>
int FillTensorWithData(tensorflow::Tensor* tensor,
                       const string& values_as_string) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("values_as_string: \"" + values_as_string + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc mht_1(mht_1_v, 213, "", "./tensorflow/lite/testing/tf_driver.cc", "FillTensorWithData");

  const auto& values = testing::Split<T>(values_as_string, ",");

  if (values.size() == tensor->NumElements()) {
    auto data = tensor->flat<T>();
    for (int i = 0; i < values.size(); i++) {
      data(i) = values[i];
    }
  }

  return values.size();
}

// Assumes 'values_as_string' is a hex string that gets converted into a
// TF Lite DynamicBuffer. Strings are then extracted and copied into the
// TensorFlow tensor.
int FillTensorWithTfLiteHexString(tensorflow::Tensor* tensor,
                                  const string& values_as_string) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("values_as_string: \"" + values_as_string + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc mht_2(mht_2_v, 234, "", "./tensorflow/lite/testing/tf_driver.cc", "FillTensorWithTfLiteHexString");

  string s = absl::HexStringToBytes(values_as_string);

  int num_strings = values_as_string.empty() ? 0 : GetStringCount(s.data());

  if (num_strings == tensor->NumElements()) {
    auto data = tensor->flat<tensorflow::tstring>();
    for (size_t i = 0; i < num_strings; ++i) {
      auto ref = GetString(s.data(), i);
      data(i).assign(ref.str, ref.len);
    }
  }

  return num_strings;
}

template <typename T>
void FillTensorWithZeros(tensorflow::Tensor* tensor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc mht_3(mht_3_v, 254, "", "./tensorflow/lite/testing/tf_driver.cc", "FillTensorWithZeros");

  auto data = tensor->flat<T>();
  for (int i = 0; i < tensor->NumElements(); i++) {
    data(i) = 0;
  }
}

template <typename T>
string TensorDataToCsvString(const tensorflow::Tensor& tensor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc mht_4(mht_4_v, 265, "", "./tensorflow/lite/testing/tf_driver.cc", "TensorDataToCsvString");

  const auto& data = tensor.flat<T>();
  return Join(data.data(), data.size(), ",");
}

string TensorDataToTfLiteHexString(const tensorflow::Tensor& tensor) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc mht_5(mht_5_v, 273, "", "./tensorflow/lite/testing/tf_driver.cc", "TensorDataToTfLiteHexString");

  DynamicBuffer dynamic_buffer;

  auto data = tensor.flat<tensorflow::tstring>();
  for (int i = 0; i < tensor.NumElements(); ++i) {
    dynamic_buffer.AddString(data(i).data(), data(i).size());
  }

  char* char_buffer = nullptr;
  size_t size = dynamic_buffer.WriteToBuffer(&char_buffer);
  string s = absl::BytesToHexString({char_buffer, size});
  free(char_buffer);

  return s;
}

}  // namespace

TfDriver::TfDriver(const std::vector<string>& input_layer,
                   const std::vector<string>& input_layer_type,
                   const std::vector<string>& input_layer_shape,
                   const std::vector<string>& output_layer)
    : input_names_(input_layer), output_names_(output_layer) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc mht_6(mht_6_v, 298, "", "./tensorflow/lite/testing/tf_driver.cc", "TfDriver::TfDriver");

  CHECK_EQ(input_layer.size(), input_layer_type.size());
  CHECK_EQ(input_layer.size(), input_layer_shape.size());

  input_ids_.resize(input_layer.size());
  input_tensors_.reserve(input_layer.size());
  input_types_.resize(input_layer.size());
  input_shapes_.resize(input_layer.size());
  for (int i = 0; i < input_layer.size(); i++) {
    input_ids_[i] = i;
    input_tensors_[input_layer[i]] = {};
    CHECK(DataTypeFromString(input_layer_type[i], &input_types_[i]));
    input_shapes_[i] = Split<int64_t>(input_layer_shape[i], ",");
    input_name_to_id_[input_layer[i]] = i;
  }

  output_ids_.resize(output_layer.size());
  output_tensors_.reserve(output_layer.size());
  for (int i = 0; i < output_layer.size(); i++) {
    output_ids_[i] = i;
    output_name_to_id_[output_layer[i]] = i;
  }
}

void TfDriver::LoadModel(const string& bin_file_path) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("bin_file_path: \"" + bin_file_path + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc mht_7(mht_7_v, 326, "", "./tensorflow/lite/testing/tf_driver.cc", "TfDriver::LoadModel");

  if (!IsValid()) return;
  std::ifstream model(bin_file_path);
  if (model.fail()) {
    Invalidate("Failed to find the model " + bin_file_path);
    return;
  }

  tensorflow::GraphDef graphdef;
  if (!graphdef.ParseFromIstream(&model)) {
    Invalidate("Failed to parse tensorflow graphdef");
    return;
  }

  tensorflow::SessionOptions options;
  session_.reset(tensorflow::NewSession(options));
  auto status = session_->Create(graphdef);
  if (!status.ok()) {
    Invalidate("Failed to create session. " + status.error_message());
  }
}

void TfDriver::ReshapeTensor(const string& name, const string& csv_values) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + name + "\"");
   mht_8_v.push_back("csv_values: \"" + csv_values + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc mht_8(mht_8_v, 353, "", "./tensorflow/lite/testing/tf_driver.cc", "TfDriver::ReshapeTensor");

  if (!IsValid()) return;
  int id = input_name_to_id_[name];
  input_shapes_[id] = Split<int64_t>(csv_values, ",");
  input_tensors_[input_names_[id]] =
      CreateTensor(input_types_[id], input_shapes_[id]);
  ResetTensor(name);
}

void TfDriver::ResetTensor(const std::string& name) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc mht_9(mht_9_v, 366, "", "./tensorflow/lite/testing/tf_driver.cc", "TfDriver::ResetTensor");

  if (!IsValid()) return;
  int id = input_name_to_id_[name];
  auto tensor = input_tensors_[input_names_[id]];
  switch (input_types_[id]) {
    case tensorflow::DT_FLOAT: {
      FillTensorWithZeros<float>(&tensor);
      break;
    }
    case tensorflow::DT_INT32: {
      FillTensorWithZeros<int32_t>(&tensor);
      break;
    }
    default:
      Invalidate(absl::StrCat("Unsupported tensor type ", input_types_[id],
                              tensorflow::DataType_Name(input_types_[id]),
                              " in ResetInput"));
      return;
  }
}
string TfDriver::ReadOutput(const string& name) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc mht_10(mht_10_v, 390, "", "./tensorflow/lite/testing/tf_driver.cc", "TfDriver::ReadOutput");

  if (!IsValid()) return "";
  return ReadOutput(output_tensors_[output_name_to_id_[name]]);
}
void TfDriver::Invoke(const std::vector<std::pair<string, string>>& inputs) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc mht_11(mht_11_v, 397, "", "./tensorflow/lite/testing/tf_driver.cc", "TfDriver::Invoke");

  if (!IsValid()) return;
  for (const auto& input : inputs) {
    auto id = input_name_to_id_[input.first];
    auto tensor = CreateTensor(input_types_[id], input_shapes_[id]);
    SetInput(input.second, &tensor);
    input_tensors_[input_names_[id]] = tensor;
  }
  auto status = session_->Run({input_tensors_.begin(), input_tensors_.end()},
                              output_names_, {}, &output_tensors_);
  if (!status.ok()) {
    Invalidate(absl::StrCat("TensorFlow failed to run graph:",
                            status.error_message()));
  }
}

void TfDriver::SetInput(const string& values_as_string,
                        tensorflow::Tensor* tensor) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("values_as_string: \"" + values_as_string + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc mht_12(mht_12_v, 418, "", "./tensorflow/lite/testing/tf_driver.cc", "TfDriver::SetInput");

  int num_values_available = 0;
  switch (tensor->dtype()) {
    case tensorflow::DT_FLOAT:
      num_values_available =
          FillTensorWithData<float>(tensor, values_as_string);
      break;
    case tensorflow::DT_INT32:
      num_values_available =
          FillTensorWithData<int32_t>(tensor, values_as_string);
      break;
    case tensorflow::DT_UINT32:
      num_values_available =
          FillTensorWithData<uint32_t>(tensor, values_as_string);
      break;
    case tensorflow::DT_UINT8:
      num_values_available =
          FillTensorWithData<uint8_t>(tensor, values_as_string);
      break;
    case tensorflow::DT_STRING:
      num_values_available =
          FillTensorWithTfLiteHexString(tensor, values_as_string);
      break;
    default:
      Invalidate(absl::StrCat("Unsupported tensor type ",
                              tensorflow::DataType_Name(tensor->dtype()),
                              " in SetInput"));
      return;
  }

  if (tensor->NumElements() != num_values_available) {
    Invalidate(absl::StrCat("Needed ", tensor->NumElements(),
                            " values for input tensor, but was given ",
                            num_values_available, " instead."));
  }
}

string TfDriver::ReadOutput(const tensorflow::Tensor& tensor) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePStestingPStf_driverDTcc mht_13(mht_13_v, 458, "", "./tensorflow/lite/testing/tf_driver.cc", "TfDriver::ReadOutput");

  switch (tensor.dtype()) {
    case tensorflow::DT_FLOAT:
      return TensorDataToCsvString<float>(tensor);
    case tensorflow::DT_INT32:
      return TensorDataToCsvString<int32_t>(tensor);
    case tensorflow::DT_UINT32:
      return TensorDataToCsvString<uint32_t>(tensor);
    case tensorflow::DT_INT64:
      return TensorDataToCsvString<int64_t>(tensor);
    case tensorflow::DT_UINT8:
      return TensorDataToCsvString<uint8_t>(tensor);
    case tensorflow::DT_STRING:
      return TensorDataToTfLiteHexString(tensor);
    case tensorflow::DT_BOOL:
      return TensorDataToCsvString<bool>(tensor);
    default:
      Invalidate(absl::StrCat("Unsupported tensor type ",
                              tensorflow::DataType_Name(tensor.dtype()),
                              " in ReadOutput"));
      return "";
  }
}

}  // namespace testing
}  // namespace tflite
