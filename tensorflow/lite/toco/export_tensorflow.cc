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
class MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc() {
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
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "google/protobuf/map.h"
#include "google/protobuf/text_format.h"
#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/tensorflow_util.h"
#include "tensorflow/lite/toco/tooling_util.h"

using tensorflow::DT_BOOL;
using tensorflow::DT_COMPLEX64;
using tensorflow::DT_FLOAT;
using tensorflow::DT_INT16;
using tensorflow::DT_INT32;
using tensorflow::DT_INT64;
using tensorflow::DT_UINT32;
using tensorflow::DT_UINT8;
using tensorflow::GraphDef;
using tensorflow::TensorProto;

namespace toco {
namespace {

tensorflow::DataType GetTensorFlowDataType(ArrayDataType data_type,
                                           const std::string& error_location) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("error_location: \"" + error_location + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_0(mht_0_v, 223, "", "./tensorflow/lite/toco/export_tensorflow.cc", "GetTensorFlowDataType");

  switch (data_type) {
    case ArrayDataType::kBool:
      return tensorflow::DT_BOOL;
    case ArrayDataType::kFloat:
      return tensorflow::DT_FLOAT;
    case ArrayDataType::kUint8:
      return tensorflow::DT_UINT8;
    case ArrayDataType::kInt16:
      return tensorflow::DT_INT16;
    case ArrayDataType::kUint16:
      return tensorflow::DT_UINT16;
    case ArrayDataType::kInt32:
      return tensorflow::DT_INT32;
    case ArrayDataType::kUint32:
      return tensorflow::DT_UINT32;
    case ArrayDataType::kInt64:
      return tensorflow::DT_INT64;
    case ArrayDataType::kString:
      return tensorflow::DT_STRING;
    case ArrayDataType::kComplex64:
      return tensorflow::DT_COMPLEX64;
    default:
    case ArrayDataType::kNone:
      LOG(FATAL) << "Unsupported data type '" << ArrayDataTypeName(data_type)
                 << "' in " << error_location;
      return tensorflow::DT_INVALID;
  }
}

tensorflow::DataType GetTensorFlowDataTypeForOp(ArrayDataType data_type,
                                                const std::string& op_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_1(mht_1_v, 258, "", "./tensorflow/lite/toco/export_tensorflow.cc", "GetTensorFlowDataTypeForOp");

  return GetTensorFlowDataType(data_type, "op '" + op_name + "'");
}

tensorflow::DataType GetTensorFlowDataType(const Model& model,
                                           const std::string& array_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("array_name: \"" + array_name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_2(mht_2_v, 267, "", "./tensorflow/lite/toco/export_tensorflow.cc", "GetTensorFlowDataType");

  return GetTensorFlowDataType(model.GetArray(array_name).data_type,
                               "array '" + array_name + "'");
}

// TensorFlow sometimes forbids what it calls "legacy scalars",
// which are 1-D shapes where the unique shape size is 1.
// See OpKernel::IsLegacyScalar and OpKernel::allow_legacy_scalars.
// For that reason, we generally avoid creating legacy scalars,
// by detecting the case where a 1-D shape would be of size 1 and
// replacing that by a 0-D shape.
// However, there is a special circumstance where we must not do that
// and must unconditionally create a 1-D shape even if it is going to
// be of size 1: that is the case of bias vectors, with BiasAdd nodes.
// Indeed, TensorFlow requires bias vectors to be 1-D; in the case of
// a depth of 1, that would be a legacy scalar, so in that case we
// must go ahead and keep the shape 1-D, letting it be a legacy scalar.
enum class LegacyScalarPolicy { kAvoidLegacyScalars, kDoCreateLegacyScalars };

void ExportFloatArray(const Shape& input_shape, const float* input_data,
                      TensorProto* output_tensor,
                      LegacyScalarPolicy legacy_scalar_policy) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_3(mht_3_v, 291, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ExportFloatArray");

  output_tensor->set_dtype(DT_FLOAT);
  const int input_flat_size = RequiredBufferSizeForShape(input_shape);
  auto* shape = output_tensor->mutable_tensor_shape();

  const int kDims = input_shape.dimensions_count();
  if (legacy_scalar_policy == LegacyScalarPolicy::kDoCreateLegacyScalars ||
      kDims > 1 || (kDims == 1 && input_shape.dims(0) > 1)) {
    for (int i = 0; i < kDims; ++i) {
      shape->add_dim()->set_size(input_shape.dims(i));
    }
  }
  output_tensor->set_tensor_content(
      std::string(reinterpret_cast<const char*>(input_data),
                  sizeof(*input_data) * input_flat_size));
}

void ExportFloatArray(AxesOrder input_axes_order, const Shape& input_shape,
                      const float* input_data, AxesOrder output_axes_order,
                      TensorProto* output_tensor,
                      LegacyScalarPolicy legacy_scalar_policy) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_4(mht_4_v, 314, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ExportFloatArray");

  CHECK_EQ(AxesCount(output_axes_order), AxesCount(input_axes_order));
  output_tensor->set_dtype(DT_FLOAT);
  CHECK_EQ(input_shape.dimensions_count(), AxesCount(input_axes_order));
  const int input_flat_size = RequiredBufferSizeForShape(input_shape);

  Shape shuffled_shape;
  ShuffleDims(input_shape, input_axes_order, output_axes_order,
              &shuffled_shape);
  std::vector<float> shuffled_data(input_flat_size);
  ShuffleArray(input_shape, input_axes_order, output_axes_order, shuffled_shape,
               input_data, shuffled_data.data());

  ExportFloatArray(shuffled_shape, shuffled_data.data(), output_tensor,
                   legacy_scalar_policy);
}

bool HasAlreadyExportedConst(const std::string& name,
                             const GraphDef& tensorflow_graph) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_5(mht_5_v, 336, "", "./tensorflow/lite/toco/export_tensorflow.cc", "HasAlreadyExportedConst");

  for (const auto& node : tensorflow_graph.node()) {
    if (node.op() == "Const" && node.name() == name) {
      return true;
    }
  }
  return false;
}

void ConvertFloatTensorConst(const std::string& name, const Shape& input_shape,
                             const float* input_data,
                             AxesOrder input_axes_order,
                             AxesOrder output_axes_order,
                             GraphDef* tensorflow_graph,
                             LegacyScalarPolicy legacy_scalar_policy) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_6(mht_6_v, 354, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertFloatTensorConst");

  if (HasAlreadyExportedConst(name, *tensorflow_graph)) {
    return;
  }
  tensorflow::NodeDef* const_op = tensorflow_graph->add_node();
  const_op->set_op("Const");
  const_op->set_name(name);
  (*const_op->mutable_attr())["dtype"].set_type(DT_FLOAT);
  auto* tensor = (*const_op->mutable_attr())["value"].mutable_tensor();
  ExportFloatArray(input_axes_order, input_shape, input_data, output_axes_order,
                   tensor, legacy_scalar_policy);
}

void ConvertFloatTensorConst(const std::string& name, const Shape& input_shape,
                             const float* input_data,
                             AxesOrder input_axes_order,
                             AxesOrder output_axes_order,
                             GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_7(mht_7_v, 375, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertFloatTensorConst");

  ConvertFloatTensorConst(name, input_shape, input_data, input_axes_order,
                          output_axes_order, tensorflow_graph,
                          LegacyScalarPolicy::kAvoidLegacyScalars);
}

void ConvertFloatTensorConst(const Model& model, const std::string& name,
                             AxesOrder input_axes_order,
                             AxesOrder output_axes_order,
                             GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_8(mht_8_v, 388, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertFloatTensorConst");

  if (HasAlreadyExportedConst(name, *tensorflow_graph)) {
    return;
  }
  CHECK(model.HasArray(name));
  const auto& input_array = model.GetArray(name);
  const auto& input_shape = input_array.shape();
  CHECK(input_array.buffer);
  CHECK(input_array.buffer->type == ArrayDataType::kFloat);
  const float* input_data =
      input_array.GetBuffer<ArrayDataType::kFloat>().data.data();
  ConvertFloatTensorConst(name, input_shape, input_data, input_axes_order,
                          output_axes_order, tensorflow_graph);
}

void ConvertFloatTensorConst(const Model& model, const std::string& name,
                             GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_9(mht_9_v, 408, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertFloatTensorConst");

  if (HasAlreadyExportedConst(name, *tensorflow_graph)) {
    return;
  }
  tensorflow::NodeDef* const_op = tensorflow_graph->add_node();
  const_op->set_op("Const");
  const_op->set_name(name);
  (*const_op->mutable_attr())["dtype"].set_type(DT_FLOAT);
  auto* tensor = (*const_op->mutable_attr())["value"].mutable_tensor();
  CHECK(model.HasArray(name));
  const auto& input_array = model.GetArray(name);
  const auto& input_shape = input_array.shape();
  CHECK(input_array.buffer);
  CHECK(input_array.buffer->type == ArrayDataType::kFloat);
  const float* input_data =
      input_array.GetBuffer<ArrayDataType::kFloat>().data.data();
  ExportFloatArray(input_shape, input_data, tensor,
                   LegacyScalarPolicy::kAvoidLegacyScalars);
}

void ConvertBoolTensorConst(const Model& model, const std::string& name,
                            GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_10(mht_10_v, 433, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertBoolTensorConst");

  if (HasAlreadyExportedConst(name, *tensorflow_graph)) {
    return;
  }
  CHECK(model.HasArray(name));
  const auto& array = model.GetArray(name);
  tensorflow::NodeDef* const_op = tensorflow_graph->add_node();
  const_op->set_op("Const");
  const_op->set_name(name);
  (*const_op->mutable_attr())["dtype"].set_type(DT_BOOL);
  auto* tensor = (*const_op->mutable_attr())["value"].mutable_tensor();
  tensor->set_dtype(DT_BOOL);
  const auto& data = array.GetBuffer<ArrayDataType::kBool>().data;
  for (auto index : data) {
    tensor->add_bool_val(index);
  }
  const auto& array_shape = array.shape();
  auto* shape = tensor->mutable_tensor_shape();
  for (int i = 0; i < array_shape.dimensions_count(); i++) {
    shape->add_dim()->set_size(array_shape.dims(i));
  }
}

void ConvertIntTensorConst(const Model& model, const std::string& name,
                           GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_11(mht_11_v, 461, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertIntTensorConst");

  if (HasAlreadyExportedConst(name, *tensorflow_graph)) {
    return;
  }
  CHECK(model.HasArray(name));
  const auto& array = model.GetArray(name);
  tensorflow::NodeDef* const_op = tensorflow_graph->add_node();
  const_op->set_op("Const");
  const_op->set_name(name);
  (*const_op->mutable_attr())["dtype"].set_type(DT_INT32);
  auto* tensor = (*const_op->mutable_attr())["value"].mutable_tensor();
  tensor->set_dtype(DT_INT32);
  const auto& data = array.GetBuffer<ArrayDataType::kInt32>().data;
  for (auto index : data) {
    tensor->add_int_val(index);
  }
  const auto& array_shape = array.shape();
  auto* shape = tensor->mutable_tensor_shape();
  for (int i = 0; i < array_shape.dimensions_count(); i++) {
    shape->add_dim()->set_size(array_shape.dims(i));
  }
}

void CreateIntTensorConst(const std::string& name,
                          const std::vector<int32>& data,
                          const std::vector<int32>& shape,
                          GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_12(mht_12_v, 491, "", "./tensorflow/lite/toco/export_tensorflow.cc", "CreateIntTensorConst");

  if (HasAlreadyExportedConst(name, *tensorflow_graph)) {
    return;
  }
  tensorflow::NodeDef* const_op = tensorflow_graph->add_node();
  const_op->set_op("Const");
  const_op->set_name(name);
  (*const_op->mutable_attr())["dtype"].set_type(DT_INT32);
  auto* tensor = (*const_op->mutable_attr())["value"].mutable_tensor();
  tensor->set_dtype(DT_INT32);
  for (auto index : data) {
    tensor->add_int_val(index);
  }
  auto* tensor_shape = tensor->mutable_tensor_shape();
  int num_elements = 1;
  for (int size : shape) {
    tensor_shape->add_dim()->set_size(size);
    num_elements *= size;
  }
  CHECK_EQ(num_elements, data.size());
}

void ConvertComplex64TensorConst(const Model& model, const std::string& name,
                                 GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_13(mht_13_v, 518, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertComplex64TensorConst");

  if (HasAlreadyExportedConst(name, *tensorflow_graph)) {
    return;
  }
  CHECK(model.HasArray(name));
  const auto& array = model.GetArray(name);
  tensorflow::NodeDef* const_op = tensorflow_graph->add_node();
  const_op->set_op("Const");
  const_op->set_name(name);
  (*const_op->mutable_attr())["dtype"].set_type(DT_COMPLEX64);
  auto* tensor = (*const_op->mutable_attr())["value"].mutable_tensor();
  tensor->set_dtype(DT_COMPLEX64);
  const auto& data = array.GetBuffer<ArrayDataType::kComplex64>().data;
  for (auto index : data) {
    tensor->add_scomplex_val(std::real(index));
    tensor->add_scomplex_val(std::imag(index));
  }
  const auto& array_shape = array.shape();
  auto* shape = tensor->mutable_tensor_shape();
  for (int i = 0; i < array_shape.dimensions_count(); i++) {
    shape->add_dim()->set_size(array_shape.dims(i));
  }
}

void CreateMatrixShapeTensorConst(const std::string& name, int rows, int cols,
                                  GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_14(mht_14_v, 547, "", "./tensorflow/lite/toco/export_tensorflow.cc", "CreateMatrixShapeTensorConst");

  if (HasAlreadyExportedConst(name, *tensorflow_graph)) {
    return;
  }
  tensorflow::NodeDef* const_op = tensorflow_graph->add_node();
  const_op->set_op("Const");
  const_op->set_name(name);
  (*const_op->mutable_attr())["dtype"].set_type(DT_INT32);
  auto* tensor = (*const_op->mutable_attr())["value"].mutable_tensor();
  tensor->set_dtype(DT_INT32);
  const int32 data[2] = {cols, rows};
  tensor->set_tensor_content(
      std::string(reinterpret_cast<const char*>(data), sizeof(data)));
  auto* shape = tensor->mutable_tensor_shape();
  shape->add_dim()->set_size(2);
}

void CreateDummyConcatDimTensorConst(const std::string& name, int dim,
                                     GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_15(mht_15_v, 569, "", "./tensorflow/lite/toco/export_tensorflow.cc", "CreateDummyConcatDimTensorConst");

  if (HasAlreadyExportedConst(name, *tensorflow_graph)) {
    return;
  }
  tensorflow::NodeDef* const_op = tensorflow_graph->add_node();
  const_op->set_op("Const");
  const_op->set_name(name);
  (*const_op->mutable_attr())["dtype"].set_type(DT_INT32);
  auto* tensor = (*const_op->mutable_attr())["value"].mutable_tensor();
  tensor->set_dtype(DT_INT32);
  tensor->add_int_val(dim);
}

void CreateReshapeShapeTensorConst(const std::string& name,
                                   const std::vector<int32>& shape,
                                   GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_16(mht_16_v, 588, "", "./tensorflow/lite/toco/export_tensorflow.cc", "CreateReshapeShapeTensorConst");

  if (HasAlreadyExportedConst(name, *tensorflow_graph)) {
    return;
  }
  tensorflow::NodeDef* const_op = tensorflow_graph->add_node();
  const_op->set_op("Const");
  const_op->set_name(name);
  (*const_op->mutable_attr())["dtype"].set_type(DT_INT32);
  auto* tensor = (*const_op->mutable_attr())["value"].mutable_tensor();
  tensor->set_dtype(DT_INT32);
  for (auto s : shape) {
    tensor->add_int_val(s);
  }
  // TensorFlow sometimes forbids what it calls "legacy scalars",
  // which are shapes of size 1 where the unique shape size is 1.
  // See OpKernel::IsLegacyScalar and OpKernel::allow_legacy_scalars.
  if (shape.size() > 1) {
    auto* tensor_shape = tensor->mutable_tensor_shape();
    tensor_shape->add_dim()->set_size(shape.size());
  }
}

std::string WalkUpToConstantArray(const Model& model, const std::string& name) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_17(mht_17_v, 614, "", "./tensorflow/lite/toco/export_tensorflow.cc", "WalkUpToConstantArray");

  const Array& original_array = model.GetArray(name);
  if (original_array.buffer) {
    return name;
  }
  const auto* op = GetOpWithOutput(model, name);
  CHECK(op);
  CHECK(op->type == OperatorType::kFakeQuant);
  const std::string& input_of_fakequant_name = op->inputs[0];
  const Array& input_of_fakequant = model.GetArray(input_of_fakequant_name);
  CHECK(input_of_fakequant.buffer);
  return input_of_fakequant_name;
}

void ConvertConvOperator(const Model& model, const ConvOperator& src_op,
                         GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_18(mht_18_v, 632, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertConvOperator");

  const bool has_bias = src_op.inputs.size() >= 3;
  std::string conv_output = src_op.outputs[0];
  if (has_bias) {
    conv_output += "/conv";
  }

  tensorflow::NodeDef* conv2d_op = tensorflow_graph->add_node();
  conv2d_op->set_op("Conv2D");
  conv2d_op->set_name(conv_output);
  *conv2d_op->add_input() = src_op.inputs[0];
  *conv2d_op->add_input() = src_op.inputs[1];
  (*conv2d_op->mutable_attr())["T"].set_type(DT_FLOAT);
  const std::string& weights_array_name =
      WalkUpToConstantArray(model, src_op.inputs[1]);
  const auto& weights_array = model.GetArray(weights_array_name);
  CHECK(weights_array.buffer->type == ArrayDataType::kFloat);
  ConvertFloatTensorConst(model, weights_array_name, AxesOrder::kOHWI,
                          AxesOrder::kHWIO, tensorflow_graph);
  auto& strides = (*conv2d_op->mutable_attr())["strides"];
  strides.mutable_list()->add_i(1);
  strides.mutable_list()->add_i(src_op.stride_height);
  strides.mutable_list()->add_i(src_op.stride_width);
  strides.mutable_list()->add_i(1);
  if ((src_op.dilation_width_factor != 1) ||
      (src_op.dilation_height_factor != 1)) {
    auto& dilations = (*conv2d_op->mutable_attr())["dilations"];
    dilations.mutable_list()->add_i(1);
    dilations.mutable_list()->add_i(src_op.dilation_height_factor);
    dilations.mutable_list()->add_i(src_op.dilation_width_factor);
    dilations.mutable_list()->add_i(1);
  }
  std::string padding;
  if (src_op.padding.type == PaddingType::kSame) {
    padding = "SAME";
  } else if (src_op.padding.type == PaddingType::kValid) {
    padding = "VALID";
  } else {
    LOG(FATAL) << "Bad padding (only SAME and VALID are supported)";
  }
  (*conv2d_op->mutable_attr())["padding"].set_s(padding);

  if (has_bias) {
    tensorflow::NodeDef* biasadd_op = tensorflow_graph->add_node();
    biasadd_op->set_op("BiasAdd");
    biasadd_op->set_name(src_op.outputs[0]);
    biasadd_op->add_input(conv_output);
    biasadd_op->add_input(src_op.inputs[2]);
    (*biasadd_op->mutable_attr())["T"].set_type(DT_FLOAT);
    CHECK(model.HasArray(src_op.inputs[2]));
    const std::string& bias_array_name =
        WalkUpToConstantArray(model, src_op.inputs[2]);
    const auto& bias_array = model.GetArray(bias_array_name);
    // TODO(b/62904716) Bias arrays should be 1-D, and used directly.
    Shape bias_shape_1d = bias_array.shape();
    UnextendShape(&bias_shape_1d, 1);
    CHECK(bias_array.buffer->type == ArrayDataType::kFloat);
    const float* bias_data =
        bias_array.GetBuffer<ArrayDataType::kFloat>().data.data();
    ConvertFloatTensorConst(bias_array_name, bias_shape_1d, bias_data,
                            AxesOrder::kOneAxis, AxesOrder::kOneAxis,
                            tensorflow_graph,
                            LegacyScalarPolicy::kDoCreateLegacyScalars);
  }
}

void ConvertDepthwiseConvOperator(const Model& model,
                                  const DepthwiseConvOperator& src_op,
                                  GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_19(mht_19_v, 703, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertDepthwiseConvOperator");

  const bool has_bias = src_op.inputs.size() >= 3;
  std::string conv_output = src_op.outputs[0];
  if (has_bias) {
    conv_output += "/conv";
  }

  tensorflow::NodeDef* dc2d_op = tensorflow_graph->add_node();
  dc2d_op->set_op("DepthwiseConv2dNative");
  dc2d_op->set_name(conv_output);
  *dc2d_op->add_input() = src_op.inputs[0];
  *dc2d_op->add_input() = src_op.inputs[1];
  (*dc2d_op->mutable_attr())["T"].set_type(DT_FLOAT);

  // Our internal DepthwiseConv weights are 1 x H x W x OutputDepth.
  // We need to convert that to H x W x InputDepth x Multiplier.
  // That's only a matter of constructing a Dims object; the actual
  // array layout is the same.
  CHECK(model.HasArray(src_op.inputs[1]));
  const std::string& src_weights_name =
      WalkUpToConstantArray(model, src_op.inputs[1]);
  const auto& src_weights_array = model.GetArray(src_weights_name);
  const auto& src_weights_shape = src_weights_array.shape();
  CHECK_EQ(src_weights_shape.dimensions_count(), 4);
  const Shape dst_weights_shape =
      Shape({src_weights_shape.dims(1), src_weights_shape.dims(2),
             src_weights_shape.dims(3) / src_op.depth_multiplier,
             src_op.depth_multiplier});
  CHECK_EQ(src_weights_shape.dims(3) % src_op.depth_multiplier, 0);
  CHECK(dst_weights_shape.dims(2) * dst_weights_shape.dims(3) ==
        src_weights_shape.dims(3));
  CHECK_EQ(src_weights_shape.dims(0), 1);

  CHECK(src_weights_array.buffer->type == ArrayDataType::kFloat);
  const float* src_weights_data =
      src_weights_array.GetBuffer<ArrayDataType::kFloat>().data.data();
  ConvertFloatTensorConst(src_weights_name, dst_weights_shape, src_weights_data,
                          AxesOrder::kHWIM, AxesOrder::kHWIM, tensorflow_graph);

  auto& strides = (*dc2d_op->mutable_attr())["strides"];
  strides.mutable_list()->add_i(1);
  strides.mutable_list()->add_i(src_op.stride_height);
  strides.mutable_list()->add_i(src_op.stride_width);
  strides.mutable_list()->add_i(1);
  // TODO(b/116063589): To return a working TF GraphDef, we should be returning
  // the correct SpaceToBatchNd and BatchToSpaceND operation before and after
  // the conv since TF doesn't support dilations.
  if ((src_op.dilation_width_factor != 1) ||
      (src_op.dilation_height_factor != 1)) {
    auto& dilations = (*dc2d_op->mutable_attr())["dilations"];
    dilations.mutable_list()->add_i(1);
    dilations.mutable_list()->add_i(src_op.dilation_height_factor);
    dilations.mutable_list()->add_i(src_op.dilation_width_factor);
    dilations.mutable_list()->add_i(1);
  }
  std::string padding;
  if (src_op.padding.type == PaddingType::kSame) {
    padding = "SAME";
  } else if (src_op.padding.type == PaddingType::kValid) {
    padding = "VALID";
  } else {
    LOG(FATAL) << "Bad padding (only SAME and VALID are supported)";
  }
  (*dc2d_op->mutable_attr())["padding"].set_s(padding);

  if (has_bias) {
    tensorflow::NodeDef* biasadd_op = tensorflow_graph->add_node();
    biasadd_op->set_op("BiasAdd");
    biasadd_op->set_name(src_op.outputs[0]);
    biasadd_op->add_input(conv_output);
    biasadd_op->add_input(src_op.inputs[2]);
    (*biasadd_op->mutable_attr())["T"].set_type(DT_FLOAT);
    CHECK(model.HasArray(src_op.inputs[2]));
    const std::string& bias_name =
        WalkUpToConstantArray(model, src_op.inputs[2]);
    const auto& bias_array = model.GetArray(bias_name);
    // TODO(b/62904716) Bias arrays should be 1-D, and used directly.
    Shape bias_shape_1d = bias_array.shape();
    UnextendShape(&bias_shape_1d, 1);
    CHECK(bias_array.buffer->type == ArrayDataType::kFloat);
    const float* bias_data =
        bias_array.GetBuffer<ArrayDataType::kFloat>().data.data();
    ConvertFloatTensorConst(bias_name, bias_shape_1d, bias_data,
                            AxesOrder::kOneAxis, AxesOrder::kOneAxis,
                            tensorflow_graph,
                            LegacyScalarPolicy::kDoCreateLegacyScalars);
  }
}

void ConvertTransposeConvOperator(const Model& model,
                                  const TransposeConvOperator& src_op,
                                  GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_20(mht_20_v, 797, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertTransposeConvOperator");

  tensorflow::NodeDef* conv2d_op = tensorflow_graph->add_node();
  conv2d_op->set_op("Conv2DBackpropInput");
  conv2d_op->set_name(src_op.outputs[0]);
  *conv2d_op->add_input() = src_op.inputs[0];
  *conv2d_op->add_input() = src_op.inputs[1];
  *conv2d_op->add_input() = src_op.inputs[2];
  (*conv2d_op->mutable_attr())["T"].set_type(DT_FLOAT);
  const std::string& weights_array_name = WalkUpToConstantArray(
      model, src_op.inputs[TransposeConvOperator::WEIGHTS]);
  const auto& weights_array = model.GetArray(weights_array_name);
  CHECK(weights_array.buffer->type == ArrayDataType::kFloat);
  ConvertFloatTensorConst(model, weights_array_name, AxesOrder::kOHWI,
                          AxesOrder::kHWOI, tensorflow_graph);
  auto& strides = (*conv2d_op->mutable_attr())["strides"];
  strides.mutable_list()->add_i(1);
  strides.mutable_list()->add_i(src_op.stride_height);
  strides.mutable_list()->add_i(src_op.stride_width);
  strides.mutable_list()->add_i(1);
  std::string padding;
  if (src_op.padding.type == PaddingType::kSame) {
    padding = "SAME";
  } else if (src_op.padding.type == PaddingType::kValid) {
    padding = "VALID";
  } else {
    LOG(FATAL) << "Bad padding (only SAME and VALID are supported)";
  }
  (*conv2d_op->mutable_attr())["padding"].set_s(padding);
}

void ConvertDepthToSpaceOperator(const Model& model,
                                 const DepthToSpaceOperator& src_op,
                                 GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_21(mht_21_v, 832, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertDepthToSpaceOperator");

  tensorflow::NodeDef* op = tensorflow_graph->add_node();
  op->set_op("DepthToSpace");
  op->set_name(src_op.outputs[0]);
  *op->add_input() = src_op.inputs[0];
  (*op->mutable_attr())["T"].set_type(DT_FLOAT);
  (*op->mutable_attr())["block_size"].set_i(src_op.block_size);
}

void ConvertSpaceToDepthOperator(const Model& model,
                                 const SpaceToDepthOperator& src_op,
                                 GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_22(mht_22_v, 846, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertSpaceToDepthOperator");

  tensorflow::NodeDef* op = tensorflow_graph->add_node();
  op->set_op("SpaceToDepth");
  op->set_name(src_op.outputs[0]);
  *op->add_input() = src_op.inputs[0];
  (*op->mutable_attr())["T"].set_type(DT_FLOAT);
  (*op->mutable_attr())["block_size"].set_i(src_op.block_size);
}

void ConvertFullyConnectedOperator(const Model& model,
                                   const FullyConnectedOperator& src_op,
                                   GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_23(mht_23_v, 860, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertFullyConnectedOperator");

  // Reshape input activations to have the shape expected by the MatMul.
  const std::string reshape_output =
      AvailableArrayName(model, src_op.outputs[0] + "/reshape");
  const std::string reshape_shape =
      AvailableArrayName(model, reshape_output + "/shape");
  const auto& fc_weights_array = model.GetArray(src_op.inputs[1]);
  const auto& fc_weights_shape = fc_weights_array.shape();
  CHECK_EQ(fc_weights_shape.dimensions_count(), 2);
  CreateMatrixShapeTensorConst(reshape_shape, fc_weights_shape.dims(1), -1,
                               tensorflow_graph);
  tensorflow::NodeDef* reshape_op = tensorflow_graph->add_node();
  reshape_op->set_op("Reshape");
  reshape_op->set_name(reshape_output);
  reshape_op->add_input(src_op.inputs[0]);
  reshape_op->add_input(reshape_shape);
  (*reshape_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[0]));

  const bool has_bias = src_op.inputs.size() >= 3;
  std::string matmul_output = src_op.outputs[0];
  if (has_bias) {
    matmul_output += "/matmul";
  }

  // Transpose the RHS input from column-major to row-major to match TensorFlow
  // expectations. This is the inverse of the transpose we do during
  // ResolveTensorFlowMatMul.
  const std::string transpose_output =
      AvailableArrayName(model, matmul_output + "/transpose_weights");
  const std::string transpose_perm =
      AvailableArrayName(model, transpose_output + "/perm");
  CreateIntTensorConst(transpose_perm, {1, 0}, {2}, tensorflow_graph);
  tensorflow::NodeDef* transpose_op = tensorflow_graph->add_node();
  transpose_op->set_op("Transpose");
  transpose_op->set_name(transpose_output);
  *transpose_op->add_input() = src_op.inputs[1];
  *transpose_op->add_input() = transpose_perm;
  (*transpose_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[1]));
  (*transpose_op->mutable_attr())["Tperm"].set_type(DT_INT32);

  tensorflow::NodeDef* matmul_op = tensorflow_graph->add_node();
  matmul_op->set_op("MatMul");
  matmul_op->set_name(matmul_output);
  *matmul_op->add_input() = reshape_output;
  *matmul_op->add_input() = transpose_op->name();
  (*matmul_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[0]));
  (*matmul_op->mutable_attr())["transpose_a"].set_b(false);
  (*matmul_op->mutable_attr())["transpose_b"].set_b(false);
  CHECK(model.HasArray(src_op.inputs[1]));

  // Add the bias, if it exists.
  if (has_bias) {
    tensorflow::NodeDef* biasadd_op = tensorflow_graph->add_node();
    biasadd_op->set_op("BiasAdd");
    biasadd_op->set_name(src_op.outputs[0]);
    biasadd_op->add_input(matmul_output);
    biasadd_op->add_input(src_op.inputs[2]);
    (*biasadd_op->mutable_attr())["T"].set_type(
        GetTensorFlowDataType(model, src_op.inputs[0]));
    CHECK(model.HasArray(src_op.inputs[2]));
    const auto& bias_array = model.GetArray(src_op.inputs[2]);
    // TODO(b/62904716) Bias arrays should be 1-D, and used directly.
    Shape bias_shape_1d = bias_array.shape();
    UnextendShape(&bias_shape_1d, 1);
    CHECK(bias_array.buffer);
    CHECK(bias_array.buffer->type == ArrayDataType::kFloat);
    const float* bias_data =
        bias_array.GetBuffer<ArrayDataType::kFloat>().data.data();
    ConvertFloatTensorConst(WalkUpToConstantArray(model, src_op.inputs[2]),
                            bias_shape_1d, bias_data, AxesOrder::kOneAxis,
                            AxesOrder::kOneAxis, tensorflow_graph,
                            LegacyScalarPolicy::kDoCreateLegacyScalars);
  }
}

void ConvertAddOperator(const Model& model, const AddOperator& src_op,
                        GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_24(mht_24_v, 942, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertAddOperator");

  tensorflow::NodeDef* add_op = tensorflow_graph->add_node();
  add_op->set_op("Add");
  add_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *add_op->add_input() = src_op.inputs[0];
  *add_op->add_input() = src_op.inputs[1];
  (*add_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.outputs[0]));
}

void ConvertAddNOperator(const Model& model, const AddNOperator& src_op,
                         GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_25(mht_25_v, 957, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertAddNOperator");

  tensorflow::NodeDef* add_op = tensorflow_graph->add_node();
  add_op->set_op("AddN");
  add_op->set_name(src_op.outputs[0]);
  for (const auto& input : src_op.inputs) {
    *add_op->add_input() = input;
  }
  (*add_op->mutable_attr())["N"].set_i(src_op.inputs.size());
  (*add_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.outputs[0]));
}

void ConvertMulOperator(const Model& model, const MulOperator& src_op,
                        GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_26(mht_26_v, 973, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertMulOperator");

  tensorflow::NodeDef* mul_op = tensorflow_graph->add_node();
  mul_op->set_op("Mul");
  mul_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *mul_op->add_input() = src_op.inputs[0];
  *mul_op->add_input() = src_op.inputs[1];
  (*mul_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.outputs[0]));
}

void ConvertDivOperator(const Model& model, const DivOperator& src_op,
                        GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_27(mht_27_v, 988, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertDivOperator");

  tensorflow::NodeDef* div_op = tensorflow_graph->add_node();
  div_op->set_op("Div");
  div_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *div_op->add_input() = src_op.inputs[0];
  *div_op->add_input() = src_op.inputs[1];
  (*div_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.outputs[0]));
}

void ConvertReluOperator(const Model& model, const ReluOperator& src_op,
                         GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_28(mht_28_v, 1003, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertReluOperator");

  tensorflow::NodeDef* relu_op = tensorflow_graph->add_node();
  relu_op->set_op("Relu");
  relu_op->set_name(src_op.outputs[0]);
  *relu_op->add_input() = src_op.inputs[0];
  (*relu_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.outputs[0]));
}

void ConvertRelu1Operator(const Relu1Operator& src_op,
                          GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_29(mht_29_v, 1016, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertRelu1Operator");

  const std::string max_bounds = src_op.outputs[0] + "/max_bounds";
  const std::string min_bounds = src_op.outputs[0] + "/min_bounds";
  const std::string max_output = src_op.outputs[0] + "/max_output";

  tensorflow::NodeDef* max_bounds_const_op = tensorflow_graph->add_node();
  max_bounds_const_op->set_op("Const");
  max_bounds_const_op->set_name(max_bounds);
  (*max_bounds_const_op->mutable_attr())["dtype"].set_type(DT_FLOAT);
  auto* max_bounds_const_op_tensor =
      (*max_bounds_const_op->mutable_attr())["value"].mutable_tensor();
  max_bounds_const_op_tensor->set_dtype(DT_FLOAT);
  max_bounds_const_op_tensor->add_float_val(-1.0f);

  tensorflow::NodeDef* min_bounds_const_op = tensorflow_graph->add_node();
  min_bounds_const_op->set_op("Const");
  min_bounds_const_op->set_name(min_bounds);
  (*min_bounds_const_op->mutable_attr())["dtype"].set_type(DT_FLOAT);
  auto* min_bounds_const_op_tensor =
      (*min_bounds_const_op->mutable_attr())["value"].mutable_tensor();
  min_bounds_const_op_tensor->set_dtype(DT_FLOAT);
  min_bounds_const_op_tensor->add_float_val(1.0f);

  tensorflow::NodeDef* max_op = tensorflow_graph->add_node();
  max_op->set_op("Maximum");
  max_op->set_name(max_output);
  *max_op->add_input() = src_op.inputs[0];
  *max_op->add_input() = max_bounds;
  (*max_op->mutable_attr())["T"].set_type(DT_FLOAT);

  tensorflow::NodeDef* min_op = tensorflow_graph->add_node();
  min_op->set_op("Minimum");
  min_op->set_name(src_op.outputs[0]);
  *min_op->add_input() = max_output;
  *min_op->add_input() = min_bounds;
  (*min_op->mutable_attr())["T"].set_type(DT_FLOAT);
}

void ConvertRelu6Operator(const Relu6Operator& src_op,
                          GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_30(mht_30_v, 1058, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertRelu6Operator");

  tensorflow::NodeDef* relu_op = tensorflow_graph->add_node();
  relu_op->set_op("Relu6");
  relu_op->set_name(src_op.outputs[0]);
  *relu_op->add_input() = src_op.inputs[0];
  (*relu_op->mutable_attr())["T"].set_type(DT_FLOAT);
}

void ConvertLogOperator(const LogOperator& src_op, GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_31(mht_31_v, 1069, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertLogOperator");

  tensorflow::NodeDef* op = tensorflow_graph->add_node();
  op->set_op("Log");
  op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 1);
  *op->add_input() = src_op.inputs[0];
  (*op->mutable_attr())["T"].set_type(DT_FLOAT);
}

void ConvertLogisticOperator(const LogisticOperator& src_op,
                             GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_32(mht_32_v, 1082, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertLogisticOperator");

  tensorflow::NodeDef* relu_op = tensorflow_graph->add_node();
  relu_op->set_op("Sigmoid");
  relu_op->set_name(src_op.outputs[0]);
  *relu_op->add_input() = src_op.inputs[0];
  (*relu_op->mutable_attr())["T"].set_type(DT_FLOAT);
}

void ConvertTanhOperator(const TanhOperator& src_op,
                         GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_33(mht_33_v, 1094, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertTanhOperator");

  tensorflow::NodeDef* tanh_op = tensorflow_graph->add_node();
  tanh_op->set_op("Tanh");
  tanh_op->set_name(src_op.outputs[0]);
  *tanh_op->add_input() = src_op.inputs[0];
  (*tanh_op->mutable_attr())["T"].set_type(DT_FLOAT);
}

void ConvertSoftmaxOperator(const Model& model, const SoftmaxOperator& src_op,
                            GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_34(mht_34_v, 1106, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertSoftmaxOperator");

  std::string softmax_input;
  Operator* providing_op = GetOpWithOutput(model, src_op.inputs[0]);
  if (providing_op != nullptr && providing_op->type == OperatorType::kReshape) {
    softmax_input = src_op.inputs[0];
  } else {
    // Insert a reshape operator that reduces the dimensions down to the 2 that
    // are required for TensorFlow Logits.
    const std::string reshape_output =
        src_op.outputs[0] + "/softmax_insert_reshape";
    const std::string softmax_size = src_op.outputs[0] + "/softmax_insert_size";
    softmax_input = reshape_output;

    tensorflow::NodeDef* reshape_op = tensorflow_graph->add_node();
    reshape_op->set_op("Reshape");
    reshape_op->set_name(reshape_output);
    *reshape_op->add_input() = src_op.inputs[0];
    *reshape_op->add_input() = softmax_size;
    (*reshape_op->mutable_attr())["T"].set_type(DT_FLOAT);

    const auto& input_shape = model.GetArray(src_op.inputs[0]).shape();
    int32_t flattened_size = 1;
    for (int i = 0; i < input_shape.dimensions_count() - 1; ++i) {
      flattened_size *= input_shape.dims(i);
    }
    const std::vector<int32> shape_data = {
        flattened_size, input_shape.dims(input_shape.dimensions_count() - 1)};
    CreateReshapeShapeTensorConst(softmax_size, shape_data, tensorflow_graph);
  }

  tensorflow::NodeDef* softmax_op = tensorflow_graph->add_node();
  softmax_op->set_op("Softmax");
  softmax_op->set_name(src_op.outputs[0]);
  *softmax_op->add_input() = softmax_input;
  // TensorFlow's Softmax doesn't seem to admit a 'beta' parameter
  CHECK_EQ(src_op.beta, 1.f);
  (*softmax_op->mutable_attr())["T"].set_type(DT_FLOAT);
}

void ConvertLogSoftmaxOperator(const Model& model,
                               const LogSoftmaxOperator& src_op,
                               GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_35(mht_35_v, 1150, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertLogSoftmaxOperator");

  std::string softmax_input;
  Operator* providing_op = GetOpWithOutput(model, src_op.inputs[0]);
  if (providing_op != nullptr && providing_op->type == OperatorType::kReshape) {
    softmax_input = src_op.inputs[0];
  } else {
    // Insert a reshape operator that reduces the dimensions down to the 2 that
    // are required for TensorFlow Logits.
    const std::string reshape_output =
        src_op.outputs[0] + "/log_softmax_insert_reshape";
    const std::string softmax_size =
        src_op.outputs[0] + "/log_softmax_insert_size";
    softmax_input = reshape_output;

    tensorflow::NodeDef* reshape_op = tensorflow_graph->add_node();
    reshape_op->set_op("Reshape");
    reshape_op->set_name(reshape_output);
    *reshape_op->add_input() = src_op.inputs[0];
    *reshape_op->add_input() = softmax_size;
    (*reshape_op->mutable_attr())["T"].set_type(DT_FLOAT);

    const auto& input_shape = model.GetArray(src_op.inputs[0]).shape();
    int32_t flattened_size = 1;
    for (int i = 0; i < input_shape.dimensions_count() - 1; ++i) {
      flattened_size *= input_shape.dims(i);
    }
    const std::vector<int32> shape_data = {
        flattened_size, input_shape.dims(input_shape.dimensions_count() - 1)};
    CreateReshapeShapeTensorConst(softmax_size, shape_data, tensorflow_graph);
  }

  tensorflow::NodeDef* log_softmax_op = tensorflow_graph->add_node();
  log_softmax_op->set_op("LogSoftmax");
  log_softmax_op->set_name(src_op.outputs[0]);
  *log_softmax_op->add_input() = softmax_input;
  (*log_softmax_op->mutable_attr())["T"].set_type(DT_FLOAT);
}

void ConvertL2NormalizationOperator(const L2NormalizationOperator& src_op,
                                    GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_36(mht_36_v, 1192, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertL2NormalizationOperator");

  const std::string square_output = src_op.outputs[0] + "/square";
  const std::string sum_reduction_indices =
      src_op.outputs[0] + "/reduction_indices";
  const std::string sum_output = src_op.outputs[0] + "/sum";
  const std::string rsqrt_output = src_op.outputs[0] + "/rsqrt";
  const std::string rsqrt_tiled_output = src_op.outputs[0] + "/rsqrt_tiled";

  tensorflow::NodeDef* sum_reduction_indices_op = tensorflow_graph->add_node();
  sum_reduction_indices_op->set_op("Const");
  sum_reduction_indices_op->set_name(sum_reduction_indices);
  (*sum_reduction_indices_op->mutable_attr())["dtype"].set_type(DT_INT32);
  auto* sum_reduction_indices_tensor =
      (*sum_reduction_indices_op->mutable_attr())["value"].mutable_tensor();
  sum_reduction_indices_tensor->set_dtype(DT_INT32);
  auto* sum_reduction_indices_shape =
      sum_reduction_indices_tensor->mutable_tensor_shape();
  auto* sum_reduction_indices_dim = sum_reduction_indices_shape->add_dim();
  sum_reduction_indices_dim->set_size(2);
  sum_reduction_indices_tensor->add_int_val(0);
  sum_reduction_indices_tensor->add_int_val(1);

  tensorflow::NodeDef* square_op = tensorflow_graph->add_node();
  square_op->set_op("Square");
  square_op->set_name(square_output);
  *square_op->add_input() = src_op.inputs[0];
  (*square_op->mutable_attr())["T"].set_type(DT_FLOAT);

  tensorflow::NodeDef* sum_op = tensorflow_graph->add_node();
  sum_op->set_op("Sum");
  sum_op->set_name(sum_output);
  *sum_op->add_input() = square_output;
  *sum_op->add_input() = sum_reduction_indices;
  (*sum_op->mutable_attr())["T"].set_type(DT_FLOAT);

  tensorflow::NodeDef* rsqrt_op = tensorflow_graph->add_node();
  rsqrt_op->set_op("Rsqrt");
  rsqrt_op->set_name(rsqrt_output);
  *rsqrt_op->add_input() = sum_output;
  (*rsqrt_op->mutable_attr())["T"].set_type(DT_FLOAT);

  tensorflow::NodeDef* mul_op = tensorflow_graph->add_node();
  mul_op->set_op("Mul");
  mul_op->set_name(src_op.outputs[0]);
  *mul_op->add_input() = src_op.inputs[0];
  *mul_op->add_input() = rsqrt_output;
  (*mul_op->mutable_attr())["T"].set_type(DT_FLOAT);
}

void ConvertLocalResponseNormalizationOperator(
    const LocalResponseNormalizationOperator& src_op,
    GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_37(mht_37_v, 1246, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertLocalResponseNormalizationOperator");

  tensorflow::NodeDef* lrn_op = tensorflow_graph->add_node();
  lrn_op->set_op("LRN");
  lrn_op->set_name(src_op.outputs[0]);
  *lrn_op->add_input() = src_op.inputs[0];
  (*lrn_op->mutable_attr())["depth_radius"].set_i(src_op.range);
  (*lrn_op->mutable_attr())["bias"].set_f(src_op.bias);
  (*lrn_op->mutable_attr())["alpha"].set_f(src_op.alpha);
  (*lrn_op->mutable_attr())["beta"].set_f(src_op.beta);
}

void ConvertFakeQuantOperator(const FakeQuantOperator& src_op,
                              GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_38(mht_38_v, 1261, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertFakeQuantOperator");

  tensorflow::NodeDef* fakequant_op = tensorflow_graph->add_node();
  fakequant_op->set_op("FakeQuantWithMinMaxArgs");
  fakequant_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 1);
  *fakequant_op->add_input() = src_op.inputs[0];
  CHECK(src_op.minmax);
  (*fakequant_op->mutable_attr())["min"].set_f(src_op.minmax->min);
  (*fakequant_op->mutable_attr())["max"].set_f(src_op.minmax->max);
  if (src_op.num_bits) {
    (*fakequant_op->mutable_attr())["num_bits"].set_i(src_op.num_bits);
  }
  if (src_op.narrow_range) {
    (*fakequant_op->mutable_attr())["narrow_range"].set_b(src_op.narrow_range);
  }
}

void ConvertMaxPoolOperator(const MaxPoolOperator& src_op,
                            GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_39(mht_39_v, 1282, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertMaxPoolOperator");

  tensorflow::NodeDef* maxpool_op = tensorflow_graph->add_node();
  maxpool_op->set_op("MaxPool");
  maxpool_op->set_name(src_op.outputs[0]);
  *maxpool_op->add_input() = src_op.inputs[0];
  auto& strides = (*maxpool_op->mutable_attr())["strides"];
  strides.mutable_list()->add_i(1);
  strides.mutable_list()->add_i(src_op.stride_height);
  strides.mutable_list()->add_i(src_op.stride_width);
  strides.mutable_list()->add_i(1);
  std::string padding;
  if (src_op.padding.type == PaddingType::kSame) {
    padding = "SAME";
  } else if (src_op.padding.type == PaddingType::kValid) {
    padding = "VALID";
  } else {
    LOG(FATAL) << "Bad padding (only SAME and VALID are supported)";
  }
  (*maxpool_op->mutable_attr())["padding"].set_s(padding);
  (*maxpool_op->mutable_attr())["T"].set_type(DT_FLOAT);
  auto& ksize = (*maxpool_op->mutable_attr())["ksize"];
  ksize.mutable_list()->add_i(1);
  ksize.mutable_list()->add_i(src_op.kheight);
  ksize.mutable_list()->add_i(src_op.kwidth);
  ksize.mutable_list()->add_i(1);
}

void ConvertAveragePoolOperator(const AveragePoolOperator& src_op,
                                GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_40(mht_40_v, 1313, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertAveragePoolOperator");

  tensorflow::NodeDef* avgpool_op = tensorflow_graph->add_node();
  avgpool_op->set_op("AvgPool");
  avgpool_op->set_name(src_op.outputs[0]);
  *avgpool_op->add_input() = src_op.inputs[0];
  auto& strides = (*avgpool_op->mutable_attr())["strides"];
  strides.mutable_list()->add_i(1);
  strides.mutable_list()->add_i(src_op.stride_height);
  strides.mutable_list()->add_i(src_op.stride_width);
  strides.mutable_list()->add_i(1);
  std::string padding;
  if (src_op.padding.type == PaddingType::kSame) {
    padding = "SAME";
  } else if (src_op.padding.type == PaddingType::kValid) {
    padding = "VALID";
  } else {
    LOG(FATAL) << "Bad padding (only SAME and VALID are supported)";
  }
  (*avgpool_op->mutable_attr())["padding"].set_s(padding);
  (*avgpool_op->mutable_attr())["T"].set_type(DT_FLOAT);
  auto& ksize = (*avgpool_op->mutable_attr())["ksize"];
  ksize.mutable_list()->add_i(1);
  ksize.mutable_list()->add_i(src_op.kheight);
  ksize.mutable_list()->add_i(src_op.kwidth);
  ksize.mutable_list()->add_i(1);
}

void ConvertConcatenationOperator(const Model& model,
                                  const ConcatenationOperator& src_op,
                                  GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_41(mht_41_v, 1345, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertConcatenationOperator");

  tensorflow::NodeDef* dc_op = tensorflow_graph->add_node();
  dc_op->set_op("ConcatV2");
  dc_op->set_name(src_op.outputs[0]);
  const std::string dummy_axis = src_op.outputs[0] + "/axis";
  CreateDummyConcatDimTensorConst(dummy_axis, src_op.axis, tensorflow_graph);
  for (const auto& input : src_op.inputs) {
    *dc_op->add_input() = input;
  }
  *dc_op->add_input() = dummy_axis;
  (*dc_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[0]));
  (*dc_op->mutable_attr())["Tidx"].set_type(DT_INT32);
  (*dc_op->mutable_attr())["N"].set_i(src_op.inputs.size());
}

void ConvertTensorFlowReshapeOperator(const Model& model,
                                      const TensorFlowReshapeOperator& src_op,
                                      GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_42(mht_42_v, 1366, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertTensorFlowReshapeOperator");

  tensorflow::NodeDef* reshape_op = tensorflow_graph->add_node();
  reshape_op->set_op("Reshape");
  reshape_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *reshape_op->add_input() = src_op.inputs[0];
  *reshape_op->add_input() = src_op.inputs[1];
  (*reshape_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.outputs[0]));
  const auto& shape_array = model.GetArray(src_op.inputs[1]);
  QCHECK(shape_array.data_type == ArrayDataType::kInt32)
      << "Only int32 shape is supported.";
  QCHECK(shape_array.buffer != nullptr)
      << "Shape inferred at runtime is not supported.";
  const auto& shape_data = shape_array.GetBuffer<ArrayDataType::kInt32>().data;
  CreateReshapeShapeTensorConst(src_op.inputs[1], shape_data, tensorflow_graph);
}

void ConvertL2PoolOperator(const L2PoolOperator& src_op,
                           GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_43(mht_43_v, 1388, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertL2PoolOperator");

  const std::string square_output = src_op.outputs[0] + "/square";
  const std::string avgpool_output = src_op.outputs[0] + "/avgpool";

  tensorflow::NodeDef* square_op = tensorflow_graph->add_node();
  square_op->set_op("Square");
  square_op->set_name(square_output);
  *square_op->add_input() = src_op.inputs[0];
  (*square_op->mutable_attr())["T"].set_type(DT_FLOAT);

  std::string padding;
  if (src_op.padding.type == PaddingType::kSame) {
    padding = "SAME";
  } else if (src_op.padding.type == PaddingType::kValid) {
    padding = "VALID";
  } else {
    LOG(FATAL) << "Bad padding (only SAME and VALID are supported)";
  }

  tensorflow::NodeDef* avgpool_op = tensorflow_graph->add_node();
  avgpool_op->set_op("AvgPool");
  avgpool_op->set_name(avgpool_output);
  *avgpool_op->add_input() = square_output;
  auto& strides = (*avgpool_op->mutable_attr())["strides"];
  strides.mutable_list()->add_i(1);
  strides.mutable_list()->add_i(src_op.stride_height);
  strides.mutable_list()->add_i(src_op.stride_width);
  strides.mutable_list()->add_i(1);

  (*avgpool_op->mutable_attr())["padding"].set_s(padding);
  (*avgpool_op->mutable_attr())["T"].set_type(DT_FLOAT);
  auto& ksize = (*avgpool_op->mutable_attr())["ksize"];
  ksize.mutable_list()->add_i(1);
  ksize.mutable_list()->add_i(src_op.kheight);
  ksize.mutable_list()->add_i(src_op.kwidth);
  ksize.mutable_list()->add_i(1);

  tensorflow::NodeDef* sqrt_op = tensorflow_graph->add_node();
  sqrt_op->set_op("Sqrt");
  sqrt_op->set_name(src_op.outputs[0]);
  *sqrt_op->add_input() = avgpool_output;
  (*sqrt_op->mutable_attr())["T"].set_type(DT_FLOAT);
}

void ConvertSquareOperator(const TensorFlowSquareOperator& src_op,
                           GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_44(mht_44_v, 1436, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertSquareOperator");

  tensorflow::NodeDef* square_op = tensorflow_graph->add_node();
  square_op->set_op("Square");
  square_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 1);
  *square_op->add_input() = src_op.inputs[0];
  (*square_op->mutable_attr())["T"].set_type(DT_FLOAT);
}

void ConvertSqrtOperator(const TensorFlowSqrtOperator& src_op,
                         GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_45(mht_45_v, 1449, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertSqrtOperator");

  tensorflow::NodeDef* sqrt_op = tensorflow_graph->add_node();
  sqrt_op->set_op("Sqrt");
  sqrt_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 1);
  *sqrt_op->add_input() = src_op.inputs[0];
  (*sqrt_op->mutable_attr())["T"].set_type(DT_FLOAT);
}

void ConvertRsqrtOperator(const Model& model,
                          const TensorFlowRsqrtOperator& src_op,
                          GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_46(mht_46_v, 1463, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertRsqrtOperator");

  tensorflow::NodeDef* rsqrt_op = tensorflow_graph->add_node();
  rsqrt_op->set_op("Rsqrt");
  rsqrt_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 1);
  *rsqrt_op->add_input() = src_op.inputs[0];
  const tensorflow::DataType data_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*rsqrt_op->mutable_attr())["T"].set_type(data_type);
}

void ConvertSplitOperator(const Model& model,
                          const TensorFlowSplitOperator& src_op,
                          GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_47(mht_47_v, 1479, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertSplitOperator");

  tensorflow::NodeDef* split_op = tensorflow_graph->add_node();
  split_op->set_op("Split");
  split_op->set_name(src_op.outputs[0]);
  for (const auto& input : src_op.inputs) {
    *split_op->add_input() = input;
  }
  (*split_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.outputs[0]));
  (*split_op->mutable_attr())["num_split"].set_i(src_op.num_split);
  const auto& split_dim_array = model.GetArray(src_op.inputs[0]);
  CHECK(split_dim_array.buffer);
  CHECK(split_dim_array.data_type == ArrayDataType::kInt32);
  const auto& split_dim_data =
      split_dim_array.GetBuffer<ArrayDataType::kInt32>().data;
  CHECK_EQ(split_dim_data.size(), 1);
  const int split_dim = split_dim_data[0];
  CreateDummyConcatDimTensorConst(src_op.inputs[0], split_dim,
                                  tensorflow_graph);
}

void ConvertSplitVOperator(const Model& model,
                           const TensorFlowSplitVOperator& src_op,
                           GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_48(mht_48_v, 1505, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertSplitVOperator");

  tensorflow::NodeDef* split_v_op = tensorflow_graph->add_node();
  split_v_op->set_op("SplitV");
  split_v_op->set_name(src_op.outputs[0]);
  for (const auto& input : src_op.inputs) {
    *split_v_op->add_input() = input;
  }
  (*split_v_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.outputs[0]));
  (*split_v_op->mutable_attr())["Tlen"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[1]));
  (*split_v_op->mutable_attr())["num_split"].set_i(src_op.num_split);
  ConvertIntTensorConst(model, src_op.inputs[1], tensorflow_graph);
}

void ConvertCastOperator(const Model& model, const CastOperator& src_op,
                         GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_49(mht_49_v, 1524, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertCastOperator");

  tensorflow::NodeDef* cast_op = tensorflow_graph->add_node();
  cast_op->set_op("Cast");
  cast_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 1);
  *cast_op->add_input() = src_op.inputs[0];

  (*cast_op->mutable_attr())["DstT"].set_type(
      GetTensorFlowDataType(model, src_op.outputs[0]));
  (*cast_op->mutable_attr())["SrcT"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[0]));
}

void ConvertFloorOperator(const Model& model, const FloorOperator& src_op,
                          GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_50(mht_50_v, 1541, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertFloorOperator");

  tensorflow::NodeDef* floor_op = tensorflow_graph->add_node();
  floor_op->set_op("Floor");
  floor_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 1);
  *floor_op->add_input() = src_op.inputs[0];
  (*floor_op->mutable_attr())["T"].set_type(DT_FLOAT);
}

void ConvertCeilOperator(const Model& model, const CeilOperator& src_op,
                         GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_51(mht_51_v, 1554, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertCeilOperator");

  tensorflow::NodeDef* ceil_op = tensorflow_graph->add_node();
  ceil_op->set_op("Ceil");
  ceil_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 1);
  *ceil_op->add_input() = src_op.inputs[0];
  (*ceil_op->mutable_attr())["T"].set_type(DT_FLOAT);
}

void ConvertRoundOperator(const Model& model, const RoundOperator& src_op,
                          GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_52(mht_52_v, 1567, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertRoundOperator");

  tensorflow::NodeDef* round_op = tensorflow_graph->add_node();
  round_op->set_op("Round");
  round_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 1);
  *round_op->add_input() = src_op.inputs[0];
  (*round_op->mutable_attr())["T"].set_type(DT_FLOAT);
}

void ConvertGatherOperator(const Model& model, const GatherOperator& src_op,
                           GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_53(mht_53_v, 1580, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertGatherOperator");

  tensorflow::NodeDef* gather_op = tensorflow_graph->add_node();
  gather_op->set_op("GatherV2");
  gather_op->set_name(src_op.outputs[0]);
  *gather_op->add_input() = src_op.inputs[0];
  *gather_op->add_input() = src_op.inputs[1];

  if (!src_op.axis) {
    // Dynamic axis.
    CHECK_EQ(src_op.inputs.size(), 3);
    *gather_op->add_input() = src_op.inputs[2];
  } else {
    // Constant axis.
    CHECK_EQ(src_op.inputs.size(), 2);
    const std::string gather_axis =
        AvailableArrayName(model, gather_op->name() + "/axis");
    CreateIntTensorConst(gather_axis, {src_op.axis.value()}, {},
                         tensorflow_graph);
    *gather_op->add_input() = gather_axis;
  }

  (*gather_op->mutable_attr())["Tindices"].set_type(DT_INT32);
  (*gather_op->mutable_attr())["Taxis"].set_type(DT_INT32);
  const tensorflow::DataType params_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*gather_op->mutable_attr())["Tparams"].set_type(params_type);
}

void ConvertArgMaxOperator(const Model& model, const ArgMaxOperator& src_op,
                           GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_54(mht_54_v, 1612, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertArgMaxOperator");

  tensorflow::NodeDef* argmax_op = tensorflow_graph->add_node();
  argmax_op->set_op("ArgMax");
  argmax_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *argmax_op->add_input() = src_op.inputs[0];
  *argmax_op->add_input() = src_op.inputs[1];
  (*argmax_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[0]));
  (*argmax_op->mutable_attr())["Tidx"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[1]));
  (*argmax_op->mutable_attr())["output_type"].set_type(
      GetTensorFlowDataType(model, src_op.outputs[0]));
}

void ConvertArgMinOperator(const Model& model, const ArgMinOperator& src_op,
                           GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_55(mht_55_v, 1631, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertArgMinOperator");

  tensorflow::NodeDef* argmin_op = tensorflow_graph->add_node();
  argmin_op->set_op("ArgMin");
  argmin_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *argmin_op->add_input() = src_op.inputs[0];
  *argmin_op->add_input() = src_op.inputs[1];
  (*argmin_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[0]));
  (*argmin_op->mutable_attr())["Tidx"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[1]));
  (*argmin_op->mutable_attr())["output_type"].set_type(
      GetTensorFlowDataType(model, src_op.outputs[0]));
}

void ConvertTransposeOperator(const Model& model,
                              const TransposeOperator& src_op,
                              GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_56(mht_56_v, 1651, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertTransposeOperator");

  tensorflow::NodeDef* transpose_op = tensorflow_graph->add_node();
  transpose_op->set_op("Transpose");
  transpose_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *transpose_op->add_input() = src_op.inputs[0];
  *transpose_op->add_input() = src_op.inputs[1];
  (*transpose_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[0]));
  (*transpose_op->mutable_attr())["Tperm"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[1]));
}

void ConvertTensorFlowShapeOperator(const Model& model,
                                    const TensorFlowShapeOperator& src_op,
                                    GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_57(mht_57_v, 1669, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertTensorFlowShapeOperator");

  tensorflow::NodeDef* shape_op = tensorflow_graph->add_node();
  shape_op->set_op("Shape");
  shape_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 1);
  *shape_op->add_input() = src_op.inputs[0];
  (*shape_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[0]));
  (*shape_op->mutable_attr())["out_type"].set_type(
      GetTensorFlowDataType(model, src_op.outputs[0]));
}

void ConvertRankOperator(const Model& model,
                         const TensorFlowRankOperator& src_op,
                         GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_58(mht_58_v, 1686, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertRankOperator");

  tensorflow::NodeDef* rank_op = tensorflow_graph->add_node();
  rank_op->set_op("Rank");
  rank_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 1);
  *rank_op->add_input() = src_op.inputs[0];
  (*rank_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[0]));
}

void ConvertRangeOperator(const Model& model, const RangeOperator& src_op,
                          GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_59(mht_59_v, 1700, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertRangeOperator");

  tensorflow::NodeDef* range_op = tensorflow_graph->add_node();
  range_op->set_op("Range");
  range_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 3);
  *range_op->add_input() = src_op.inputs[0];
  *range_op->add_input() = src_op.inputs[1];
  *range_op->add_input() = src_op.inputs[2];
  (*range_op->mutable_attr())["Tidx"].set_type(
      GetTensorFlowDataTypeForOp(src_op.dtype, /*op_name=*/src_op.outputs[0]));
}

void ConvertPackOperator(const Model& model, const PackOperator& src_op,
                         GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_60(mht_60_v, 1716, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertPackOperator");

  tensorflow::NodeDef* pack_op = tensorflow_graph->add_node();
  pack_op->set_op("Pack");
  pack_op->set_name(src_op.outputs[0]);
  for (const auto& input : src_op.inputs) {
    *pack_op->add_input() = input;
  }
  (*pack_op->mutable_attr())["axis"].set_i(src_op.axis);
  (*pack_op->mutable_attr())["N"].set_i(src_op.inputs.size());
  (*pack_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataTypeForOp(src_op.dtype, src_op.outputs[0]));
}

void ConvertFillOperator(const Model& model, const FillOperator& src_op,
                         GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_61(mht_61_v, 1733, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertFillOperator");

  tensorflow::NodeDef* fill_op = tensorflow_graph->add_node();
  fill_op->set_op("Fill");
  fill_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *fill_op->add_input() = src_op.inputs[0];
  *fill_op->add_input() = src_op.inputs[1];
  (*fill_op->mutable_attr())["index_type"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[0]));
  (*fill_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[1]));
}

void ConvertFloorDivOperator(const Model& model, const FloorDivOperator& src_op,
                             GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_62(mht_62_v, 1750, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertFloorDivOperator");

  tensorflow::NodeDef* floor_div_op = tensorflow_graph->add_node();
  floor_div_op->set_op("FloorDiv");
  floor_div_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *floor_div_op->add_input() = src_op.inputs[0];
  *floor_div_op->add_input() = src_op.inputs[1];
  (*floor_div_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[0]));
}

void ConvertFloorModOperator(const Model& model, const FloorModOperator& src_op,
                             GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_63(mht_63_v, 1765, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertFloorModOperator");

  tensorflow::NodeDef* floor_mod_op = tensorflow_graph->add_node();
  floor_mod_op->set_op("FloorMod");
  floor_mod_op->set_name(src_op.outputs[0]);
  DCHECK_EQ(src_op.inputs.size(), 2);
  *floor_mod_op->add_input() = src_op.inputs[0];
  *floor_mod_op->add_input() = src_op.inputs[1];
  (*floor_mod_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[0]));
}

void ConvertExpandDimsOperator(const Model& model,
                               const ExpandDimsOperator& src_op,
                               GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_64(mht_64_v, 1781, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertExpandDimsOperator");

  tensorflow::NodeDef* expand_dims_op = tensorflow_graph->add_node();
  expand_dims_op->set_op("ExpandDims");
  expand_dims_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *expand_dims_op->add_input() = src_op.inputs[0];
  *expand_dims_op->add_input() = src_op.inputs[1];
  (*expand_dims_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[0]));
  (*expand_dims_op->mutable_attr())["Tdim"].set_type(
      GetTensorFlowDataType(model, src_op.inputs[1]));
}

void ConvertResizeBilinearOperator(const Model& model,
                                   const ResizeBilinearOperator& src_op,
                                   GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_65(mht_65_v, 1799, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertResizeBilinearOperator");

  tensorflow::NodeDef* resize_op = tensorflow_graph->add_node();
  resize_op->set_op("ResizeBilinear");
  resize_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *resize_op->add_input() = src_op.inputs[0];
  *resize_op->add_input() = src_op.inputs[1];
  (*resize_op->mutable_attr())["T"].set_type(DT_FLOAT);
  (*resize_op->mutable_attr())["align_corners"].set_b(src_op.align_corners);
  (*resize_op->mutable_attr())["half_pixel_centers"].set_b(
      src_op.half_pixel_centers);
}

void ConvertResizeNearestNeighborOperator(
    const Model& model, const ResizeNearestNeighborOperator& src_op,
    GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_66(mht_66_v, 1817, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertResizeNearestNeighborOperator");

  tensorflow::NodeDef* resize_op = tensorflow_graph->add_node();
  resize_op->set_op("ResizeNearestNeighbor");
  resize_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *resize_op->add_input() = src_op.inputs[0];
  *resize_op->add_input() = src_op.inputs[1];
  (*resize_op->mutable_attr())["T"].set_type(DT_FLOAT);
  (*resize_op->mutable_attr())["align_corners"].set_b(src_op.align_corners);
  (*resize_op->mutable_attr())["half_pixel_centers"].set_b(
      src_op.half_pixel_centers);
}

void ConvertOneHotOperator(const Model& model, const OneHotOperator& src_op,
                           GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_67(mht_67_v, 1834, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertOneHotOperator");

  tensorflow::NodeDef* onehot_op = tensorflow_graph->add_node();
  onehot_op->set_op("OneHot");
  onehot_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 4);
  for (const auto& input : src_op.inputs) {
    *onehot_op->add_input() = input;
  }
  (*onehot_op->mutable_attr())["T"].set_type(
      GetTensorFlowDataType(model, src_op.outputs[0]));
  (*onehot_op->mutable_attr())["axis"].set_i(src_op.axis);
}

namespace {
// TODO(aselle): Remove when available in absl
absl::string_view FindLongestCommonPrefix(absl::string_view a,
                                          absl::string_view b) {
   std::vector<std::string> mht_68_v;
   mht_68_v.push_back("a: \"" + std::string(a.data(), a.size()) + "\"");
   mht_68_v.push_back("b: \"" + std::string(b.data(), b.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_68(mht_68_v, 1855, "", "./tensorflow/lite/toco/export_tensorflow.cc", "FindLongestCommonPrefix");

  if (a.empty() || b.empty()) return absl::string_view();

  const char* pa = a.data();
  const char* pb = b.data();
  std::string::difference_type count = 0;
  const std::string::difference_type limit = std::min(a.size(), b.size());
  while (count < limit && *pa == *pb) {
    ++pa;
    ++pb;
    ++count;
  }

  return absl::string_view(a.data(), count);
}
}  // namespace

void ConvertLstmCellOperator(const Model& model, const LstmCellOperator& src_op,
                             GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_69(mht_69_v, 1876, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertLstmCellOperator");

  // Find the base name
  const std::string base(
      FindLongestCommonPrefix(src_op.outputs[LstmCellOperator::STATE_OUTPUT],
                              src_op.outputs[LstmCellOperator::ACTIV_OUTPUT]));

  // Concatenate inputs
  const std::string concat_output = base + "basic_lstm_cell/concat";
  // Op names have been chosen to match the tf.slim LSTM naming
  // as closely as possible.
  const int axis =
      model.GetArray(src_op.inputs[LstmCellOperator::PREV_ACTIV_INPUT])
          .shape()
          .dimensions_count() -
      1;
  // Note that DATA_INPUT may have extra size 1 dimensions, but TF concat
  // works the same since the tensor has the same underlying data layout.
  const std::string axis_output = concat_output + "/axis";
  CreateDummyConcatDimTensorConst(axis_output, axis, tensorflow_graph);
  tensorflow::NodeDef* concat_op = tensorflow_graph->add_node();
  concat_op->set_op("ConcatV2");
  concat_op->set_name(concat_output);
  *concat_op->add_input() = src_op.inputs[LstmCellOperator::DATA_INPUT];
  *concat_op->add_input() = src_op.inputs[LstmCellOperator::PREV_ACTIV_INPUT];
  *concat_op->add_input() = axis_output;
  (*concat_op->mutable_attr())["T"].set_type(DT_FLOAT);
  (*concat_op->mutable_attr())["Tidx"].set_type(DT_INT32);
  (*concat_op->mutable_attr())["N"].set_i(2);  // Number of inputs

  // Write weights
  const std::string weights_output = base + "weights";
  CHECK(model.HasArray(src_op.inputs[LstmCellOperator::WEIGHTS_INPUT]));
  const std::string weights_name = WalkUpToConstantArray(
      model, src_op.inputs[LstmCellOperator::WEIGHTS_INPUT]);
  const auto& weights_array = model.GetArray(weights_name);
  // Convert 4D FullyConnected weights into 2D matrix
  const auto& weights_shape = weights_array.shape();
  CHECK_EQ(weights_shape.dimensions_count(), 2);
  CHECK(weights_array.buffer);
  CHECK(weights_array.buffer->type == ArrayDataType::kFloat);
  const float* weights_data =
      weights_array.GetBuffer<ArrayDataType::kFloat>().data.data();
  ConvertFloatTensorConst(weights_output, weights_shape, weights_data,
                          AxesOrder::kCR, AxesOrder::kRC, tensorflow_graph);

  // Fully connected matrix multiply
  const std::string matmul_output = base + "MatMul";
  tensorflow::NodeDef* matmul_op = tensorflow_graph->add_node();
  matmul_op->set_op("MatMul");
  matmul_op->set_name(matmul_output);
  *matmul_op->add_input() = concat_output;
  *matmul_op->add_input() = weights_output;
  (*matmul_op->mutable_attr())["transpose_a"].set_b(false);
  (*matmul_op->mutable_attr())["transpose_b"].set_b(false);
  (*matmul_op->mutable_attr())["T"].set_type(DT_FLOAT);

  // Write biases
  const std::string biases_output = base + "biases";
  CHECK(model.HasArray(src_op.inputs[LstmCellOperator::BIASES_INPUT]));
  const std::string bias_name = WalkUpToConstantArray(
      model, src_op.inputs[LstmCellOperator::BIASES_INPUT]);
  const auto& bias_array = model.GetArray(bias_name);
  // TODO(b/62904716) Bias arrays should be 1-D, and used directly.
  Shape bias_shape_1d = bias_array.shape();
  UnextendShape(&bias_shape_1d, 1);
  CHECK(bias_array.buffer);
  CHECK(bias_array.buffer->type == ArrayDataType::kFloat);
  const float* bias_data =
      bias_array.GetBuffer<ArrayDataType::kFloat>().data.data();
  ConvertFloatTensorConst(biases_output, bias_shape_1d, bias_data,
                          AxesOrder::kOneAxis, AxesOrder::kOneAxis,
                          tensorflow_graph,
                          LegacyScalarPolicy::kDoCreateLegacyScalars);

  // Add biases
  std::string biasadd_output = base + "BiasAdd";
  tensorflow::NodeDef* biasadd_op = tensorflow_graph->add_node();
  biasadd_op->set_op("BiasAdd");
  biasadd_op->set_name(biasadd_output);
  biasadd_op->add_input(matmul_output);
  biasadd_op->add_input(biases_output);
  (*biasadd_op->mutable_attr())["data_format"].set_s("NHWC");
  (*biasadd_op->mutable_attr())["T"].set_type(DT_FLOAT);

  // Split
  std::string split_dim_output = base + "split/split_dim";
  // The dimension is the same as the concatenation dimension
  CreateDummyConcatDimTensorConst(split_dim_output, axis, tensorflow_graph);
  std::string split_output = base + "split";
  tensorflow::NodeDef* split_op = tensorflow_graph->add_node();
  split_op->set_op("Split");
  split_op->set_name(split_output);
  *split_op->add_input() = split_dim_output;
  *split_op->add_input() = biasadd_output;
  (*split_op->mutable_attr())["T"].set_type(DT_FLOAT);
  (*split_op->mutable_attr())["num_split"].set_i(4);  // Split into four outputs

  // Activation functions and memory computations
  const std::string tanh_0_output = base + "Tanh";
  tensorflow::NodeDef* tanh_0_op = tensorflow_graph->add_node();
  tanh_0_op->set_op("Tanh");
  tanh_0_op->set_name(tanh_0_output);
  *tanh_0_op->add_input() = split_output + ":1";
  (*tanh_0_op->mutable_attr())["T"].set_type(DT_FLOAT);

  const std::string sigmoid_1_output = base + "Sigmoid_1";
  tensorflow::NodeDef* logistic_1_op = tensorflow_graph->add_node();
  logistic_1_op->set_op("Sigmoid");
  logistic_1_op->set_name(sigmoid_1_output);
  *logistic_1_op->add_input() = split_output;
  (*logistic_1_op->mutable_attr())["T"].set_type(DT_FLOAT);

  const std::string mul_1_output = base + "mul_1";
  tensorflow::NodeDef* mul_1_op = tensorflow_graph->add_node();
  mul_1_op->set_op("Mul");
  mul_1_op->set_name(mul_1_output);
  *mul_1_op->add_input() = sigmoid_1_output;
  *mul_1_op->add_input() = tanh_0_output;
  (*mul_1_op->mutable_attr())["T"].set_type(DT_FLOAT);

  const std::string sigmoid_0_output = base + "Sigmoid";
  tensorflow::NodeDef* logistic_2_op = tensorflow_graph->add_node();
  logistic_2_op->set_op("Sigmoid");
  logistic_2_op->set_name(sigmoid_0_output);
  *logistic_2_op->add_input() = split_output + ":2";
  (*logistic_2_op->mutable_attr())["T"].set_type(DT_FLOAT);

  const std::string sigmoid_2_output = base + "Sigmoid_2";
  tensorflow::NodeDef* logistic_3_op = tensorflow_graph->add_node();
  logistic_3_op->set_op("Sigmoid");
  logistic_3_op->set_name(sigmoid_2_output);
  *logistic_3_op->add_input() = split_output + ":3";
  (*logistic_3_op->mutable_attr())["T"].set_type(DT_FLOAT);

  const std::string mul_0_output = base + "mul";
  tensorflow::NodeDef* mul_0_op = tensorflow_graph->add_node();
  mul_0_op->set_op("Mul");
  mul_0_op->set_name(mul_0_output);
  *mul_0_op->add_input() = src_op.inputs[LstmCellOperator::PREV_STATE_INPUT];
  *mul_0_op->add_input() = sigmoid_0_output;
  (*mul_0_op->mutable_attr())["T"].set_type(DT_FLOAT);

  const std::string add_1_output =
      src_op.outputs[LstmCellOperator::STATE_OUTPUT];
  tensorflow::NodeDef* add_1_op = tensorflow_graph->add_node();
  add_1_op->set_op("Add");
  add_1_op->set_name(add_1_output);
  *add_1_op->add_input() = mul_0_output;
  *add_1_op->add_input() = mul_1_output;
  (*add_1_op->mutable_attr())["T"].set_type(DT_FLOAT);

  const std::string tanh_1_output = base + "Tanh_1";
  tensorflow::NodeDef* tanh_1_op = tensorflow_graph->add_node();
  tanh_1_op->set_op("Tanh");
  tanh_1_op->set_name(tanh_1_output);
  *tanh_1_op->add_input() = add_1_output;
  (*tanh_1_op->mutable_attr())["T"].set_type(DT_FLOAT);

  const std::string mul_2_output =
      src_op.outputs[LstmCellOperator::ACTIV_OUTPUT];
  tensorflow::NodeDef* mul_2_op = tensorflow_graph->add_node();
  mul_2_op->set_op("Mul");
  mul_2_op->set_name(mul_2_output);
  *mul_2_op->add_input() = tanh_1_output;
  *mul_2_op->add_input() = sigmoid_2_output;
  (*mul_2_op->mutable_attr())["T"].set_type(DT_FLOAT);
}

void ConvertSpaceToBatchNDOperator(const Model& model,
                                   const SpaceToBatchNDOperator& src_op,
                                   GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_70(mht_70_v, 2049, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertSpaceToBatchNDOperator");

  tensorflow::NodeDef* new_op = tensorflow_graph->add_node();
  new_op->set_op("SpaceToBatchND");
  new_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 3);
  *new_op->add_input() = src_op.inputs[0];
  *new_op->add_input() = src_op.inputs[1];
  *new_op->add_input() = src_op.inputs[2];
  const tensorflow::DataType params_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*new_op->mutable_attr())["T"].set_type(params_type);
  (*new_op->mutable_attr())["Tblock_shape"].set_type(DT_INT32);
  (*new_op->mutable_attr())["Tpaddings"].set_type(DT_INT32);
}

void ConvertBatchToSpaceNDOperator(const Model& model,
                                   const BatchToSpaceNDOperator& src_op,
                                   GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_71(mht_71_v, 2069, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertBatchToSpaceNDOperator");

  tensorflow::NodeDef* new_op = tensorflow_graph->add_node();
  new_op->set_op("BatchToSpaceND");
  new_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 3);
  *new_op->add_input() = src_op.inputs[0];
  *new_op->add_input() = src_op.inputs[1];
  *new_op->add_input() = src_op.inputs[2];
  const tensorflow::DataType params_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*new_op->mutable_attr())["T"].set_type(params_type);
  (*new_op->mutable_attr())["Tblock_shape"].set_type(DT_INT32);
  (*new_op->mutable_attr())["Tcrops"].set_type(DT_INT32);
}

void ConvertPadOperator(const Model& model, const PadOperator& src_op,
                        GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_72(mht_72_v, 2088, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertPadOperator");

  tensorflow::NodeDef* new_op = tensorflow_graph->add_node();
  new_op->set_op("Pad");
  new_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *new_op->add_input() = src_op.inputs[0];
  *new_op->add_input() = src_op.inputs[1];

  const tensorflow::DataType params_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*new_op->mutable_attr())["T"].set_type(params_type);

  // Create the params tensor.
  tensorflow::NodeDef* params_op = tensorflow_graph->add_node();
  params_op->set_op("Const");
  params_op->set_name(src_op.inputs[1]);
  (*params_op->mutable_attr())["dtype"].set_type(DT_INT32);
  auto* tensor = (*params_op->mutable_attr())["value"].mutable_tensor();
  tensor->set_dtype(DT_INT32);

  CHECK_EQ(src_op.left_padding.size(), src_op.right_padding.size());
  for (int i = 0; i < src_op.left_padding.size(); ++i) {
    tensor->add_int_val(src_op.left_padding[i]);
    tensor->add_int_val(src_op.right_padding[i]);
  }
  auto* shape = tensor->mutable_tensor_shape();
  shape->add_dim()->set_size(src_op.left_padding.size());
  shape->add_dim()->set_size(2);
}

void ConvertPadV2Operator(const Model& model, const PadV2Operator& src_op,
                          GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_73(mht_73_v, 2122, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertPadV2Operator");

  tensorflow::NodeDef* new_op = tensorflow_graph->add_node();
  new_op->set_op("PadV2");
  new_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *new_op->add_input() = src_op.inputs[0];
  *new_op->add_input() = src_op.inputs[1];
  *new_op->add_input() = src_op.inputs[2];

  const tensorflow::DataType params_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*new_op->mutable_attr())["T"].set_type(params_type);

  // Create the params tensor.
  tensorflow::NodeDef* params_op = tensorflow_graph->add_node();
  params_op->set_op("Const");
  params_op->set_name(src_op.inputs[1]);
  (*params_op->mutable_attr())["dtype"].set_type(DT_INT32);
  auto* tensor = (*params_op->mutable_attr())["value"].mutable_tensor();
  tensor->set_dtype(DT_INT32);

  CHECK_EQ(src_op.left_padding.size(), src_op.right_padding.size());
  for (int i = 0; i < src_op.left_padding.size(); ++i) {
    tensor->add_int_val(src_op.left_padding[i]);
    tensor->add_int_val(src_op.right_padding[i]);
  }
  auto* shape = tensor->mutable_tensor_shape();
  shape->add_dim()->set_size(src_op.left_padding.size());
  shape->add_dim()->set_size(2);
}

void CreateSliceInput(const std::string& input_name,
                      const std::vector<int>& values,
                      GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_74_v;
   mht_74_v.push_back("input_name: \"" + input_name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_74(mht_74_v, 2159, "", "./tensorflow/lite/toco/export_tensorflow.cc", "CreateSliceInput");

  tensorflow::NodeDef* params_op = tensorflow_graph->add_node();
  params_op->set_op("Const");
  params_op->set_name(input_name);
  (*params_op->mutable_attr())["dtype"].set_type(DT_INT32);
  auto* tensor = (*params_op->mutable_attr())["value"].mutable_tensor();
  tensor->set_dtype(DT_INT32);

  for (int i = 0; i < values.size(); ++i) {
    tensor->add_int_val(values[i]);
  }
  auto* shape = tensor->mutable_tensor_shape();
  shape->add_dim()->set_size(values.size());
}

void ConvertStridedSliceOperator(const Model& model,
                                 const StridedSliceOperator& src_op,
                                 GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_75(mht_75_v, 2179, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertStridedSliceOperator");

  tensorflow::NodeDef* new_op = tensorflow_graph->add_node();
  new_op->set_op("StridedSlice");
  new_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 4);
  *new_op->add_input() = src_op.inputs[0];
  *new_op->add_input() = src_op.inputs[1];
  *new_op->add_input() = src_op.inputs[2];
  *new_op->add_input() = src_op.inputs[3];

  const tensorflow::DataType params_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*new_op->mutable_attr())["T"].set_type(params_type);

  (*new_op->mutable_attr())["Index"].set_type(DT_INT32);
  (*new_op->mutable_attr())["begin_mask"].set_i(src_op.begin_mask);
  (*new_op->mutable_attr())["ellipsis_mask"].set_i(src_op.ellipsis_mask);
  (*new_op->mutable_attr())["end_mask"].set_i(src_op.end_mask);
  (*new_op->mutable_attr())["new_axis_mask"].set_i(src_op.new_axis_mask);
  (*new_op->mutable_attr())["shrink_axis_mask"].set_i(src_op.shrink_axis_mask);

  // Create tensors for start/stop indices and strides.
  CreateSliceInput(src_op.inputs[1], src_op.start_indices, tensorflow_graph);
  CreateSliceInput(src_op.inputs[2], src_op.stop_indices, tensorflow_graph);
  CreateSliceInput(src_op.inputs[3], src_op.strides, tensorflow_graph);
}

void ConvertSliceOperator(const Model& model, const SliceOperator& src_op,
                          GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_76(mht_76_v, 2210, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertSliceOperator");

  tensorflow::NodeDef* new_op = tensorflow_graph->add_node();
  new_op->set_op("Slice");
  new_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 3);
  *new_op->add_input() = src_op.inputs[0];
  *new_op->add_input() = src_op.inputs[1];
  *new_op->add_input() = src_op.inputs[2];

  const tensorflow::DataType params_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*new_op->mutable_attr())["T"].set_type(params_type);
  (*new_op->mutable_attr())["Index"].set_type(DT_INT32);

  // Create tensors for begin and size inputs.
  CreateSliceInput(src_op.inputs[1], src_op.begin, tensorflow_graph);
  CreateSliceInput(src_op.inputs[2], src_op.size, tensorflow_graph);
}

template <typename T>
void ConvertReduceOperator(const Model& model, const T& src_op,
                           GraphDef* tensorflow_graph,
                           const std::string& op_name) {
   std::vector<std::string> mht_77_v;
   mht_77_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_77(mht_77_v, 2236, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertReduceOperator");

  tensorflow::NodeDef* new_op = tensorflow_graph->add_node();
  new_op->set_op(op_name);
  new_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *new_op->add_input() = src_op.inputs[0];
  *new_op->add_input() = src_op.inputs[1];

  if (src_op.type != OperatorType::kAny) {
    const tensorflow::DataType params_type =
        GetTensorFlowDataType(model, src_op.inputs[0]);
    (*new_op->mutable_attr())["T"].set_type(params_type);
  }
  const tensorflow::DataType indices_type =
      GetTensorFlowDataType(model, src_op.inputs[1]);
  (*new_op->mutable_attr())["Tidx"].set_type(indices_type);

  if (src_op.keep_dims) {
    (*new_op->mutable_attr())["keep_dims"].set_b(true);
  }

  // Create the params tensor.
  tensorflow::NodeDef* params_op = tensorflow_graph->add_node();
  params_op->set_op("Const");
  params_op->set_name(src_op.inputs[1]);
  (*params_op->mutable_attr())["dtype"].set_type(DT_INT32);
  auto* tensor = (*params_op->mutable_attr())["value"].mutable_tensor();
  tensor->set_dtype(DT_INT32);

  for (int i = 0; i < src_op.axis.size(); ++i) {
    tensor->add_int_val(src_op.axis[i]);
  }
  auto* shape = tensor->mutable_tensor_shape();
  shape->add_dim()->set_size(src_op.axis.size());
}

void ConvertSqueezeOperator(const Model& model, const SqueezeOperator& src_op,
                            GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_78(mht_78_v, 2276, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertSqueezeOperator");

  tensorflow::NodeDef* new_op = tensorflow_graph->add_node();
  new_op->set_op("Squeeze");
  new_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 1);
  *new_op->add_input() = src_op.inputs[0];

  const tensorflow::DataType params_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*new_op->mutable_attr())["T"].set_type(params_type);

  if (!src_op.squeeze_dims.empty()) {
    auto& squeeze_dims = (*new_op->mutable_attr())["squeeze_dims"];
    for (int i : src_op.squeeze_dims) {
      squeeze_dims.mutable_list()->add_i(i);
    }
  }
}

void ConvertSubOperator(const Model& model, const SubOperator& src_op,
                        GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_79(mht_79_v, 2299, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertSubOperator");

  tensorflow::NodeDef* sub_op = tensorflow_graph->add_node();
  sub_op->set_op("Sub");
  sub_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *sub_op->add_input() = src_op.inputs[0];
  *sub_op->add_input() = src_op.inputs[1];
  const tensorflow::DataType data_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*sub_op->mutable_attr())["T"].set_type(data_type);
}

void ConvertTensorFlowMinimumOperator(const Model& model,
                                      const TensorFlowMinimumOperator& src_op,
                                      GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_80(mht_80_v, 2316, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertTensorFlowMinimumOperator");

  tensorflow::NodeDef* min_op = tensorflow_graph->add_node();
  min_op->set_op("Minimum");
  min_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *min_op->add_input() = src_op.inputs[0];
  *min_op->add_input() = src_op.inputs[1];
  const tensorflow::DataType data_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*min_op->mutable_attr())["T"].set_type(data_type);
}

void ConvertTensorFlowMaximumOperator(const Model& model,
                                      const TensorFlowMaximumOperator& src_op,
                                      GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_81(mht_81_v, 2333, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertTensorFlowMaximumOperator");

  tensorflow::NodeDef* max_op = tensorflow_graph->add_node();
  max_op->set_op("Maximum");
  max_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *max_op->add_input() = src_op.inputs[0];
  *max_op->add_input() = src_op.inputs[1];
  const tensorflow::DataType data_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*max_op->mutable_attr())["T"].set_type(data_type);
}

void ConvertSelectOperator(const Model& model, const SelectOperator& src_op,
                           GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_82(mht_82_v, 2349, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertSelectOperator");

  tensorflow::NodeDef* select_op = tensorflow_graph->add_node();
  select_op->set_op("Select");
  select_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 3);
  *select_op->add_input() = src_op.inputs[0];
  *select_op->add_input() = src_op.inputs[1];
  *select_op->add_input() = src_op.inputs[2];
  const tensorflow::DataType data_type =
      GetTensorFlowDataType(model, src_op.inputs[1]);
  (*select_op->mutable_attr())["T"].set_type(data_type);
}

void ConvertTileOperator(const Model& model,
                         const TensorFlowTileOperator& src_op,
                         GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_83(mht_83_v, 2367, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertTileOperator");

  tensorflow::NodeDef* tile_op = tensorflow_graph->add_node();
  tile_op->set_op("Tile");
  tile_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *tile_op->add_input() = src_op.inputs[0];
  *tile_op->add_input() = src_op.inputs[1];
  const tensorflow::DataType data_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*tile_op->mutable_attr())["T"].set_type(data_type);
  const tensorflow::DataType multiples_data_type =
      GetTensorFlowDataType(model, src_op.inputs[1]);
  (*tile_op->mutable_attr())["Tmultiples"].set_type(multiples_data_type);
}

void ConvertTopKV2Operator(const Model& model, const TopKV2Operator& src_op,
                           GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_84(mht_84_v, 2386, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertTopKV2Operator");

  tensorflow::NodeDef* topk_op = tensorflow_graph->add_node();
  topk_op->set_op("TopKV2");
  topk_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *topk_op->add_input() = src_op.inputs[0];
  *topk_op->add_input() = src_op.inputs[1];
  const tensorflow::DataType data_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*topk_op->mutable_attr())["T"].set_type(data_type);
  (*topk_op->mutable_attr())["sorted"].set_b(true);
}

void ConvertRandomUniformOperator(const Model& model,
                                  const RandomUniformOperator& src_op,
                                  GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_85(mht_85_v, 2404, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertRandomUniformOperator");

  CHECK(tensorflow_graph != nullptr);
  tensorflow::NodeDef* new_op = tensorflow_graph->add_node();
  new_op->set_op("RandomUniform");
  CHECK_EQ(src_op.inputs.size(), 1);
  new_op->set_name(src_op.outputs[0]);
  *new_op->add_input() = src_op.inputs[0];
  const tensorflow::DataType shape_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*new_op->mutable_attr())["T"].set_type(shape_type);
  (*new_op->mutable_attr())["dtype"].set_type(
      GetTensorFlowDataTypeForOp(src_op.dtype, src_op.outputs[0]));
  (*new_op->mutable_attr())["seed"].set_i(src_op.seed);
  (*new_op->mutable_attr())["seed2"].set_i(src_op.seed2);
}

void ConvertComparisonOperator(const Model& model, const Operator& src_op,
                               const char* op_name,
                               GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_86_v;
   mht_86_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_86(mht_86_v, 2426, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertComparisonOperator");

  tensorflow::NodeDef* comparison_op = tensorflow_graph->add_node();
  comparison_op->set_op(op_name);
  comparison_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *comparison_op->add_input() = src_op.inputs[0];
  *comparison_op->add_input() = src_op.inputs[1];
  const tensorflow::DataType data_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*comparison_op->mutable_attr())["T"].set_type(data_type);
}

void ConvertSparseToDenseOperator(const Model& model,
                                  const SparseToDenseOperator& src_op,
                                  const char* op_name,
                                  GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_87_v;
   mht_87_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_87(mht_87_v, 2445, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertSparseToDenseOperator");

  tensorflow::NodeDef* sparse_to_dense_op = tensorflow_graph->add_node();
  sparse_to_dense_op->set_op(op_name);
  sparse_to_dense_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 4);
  for (int i = 0; i < 4; ++i) {
    *sparse_to_dense_op->add_input() = src_op.inputs[i];
  }
  const tensorflow::DataType data_type =
      GetTensorFlowDataType(model, src_op.inputs[3]);
  (*sparse_to_dense_op->mutable_attr())["T"].set_type(data_type);
  const tensorflow::DataType index_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*sparse_to_dense_op->mutable_attr())["Tindices"].set_type(index_type);
  (*sparse_to_dense_op->mutable_attr())["Tindices"].set_b(
      src_op.validate_indices);
}

void ConvertPowOperator(const Model& model, const PowOperator& src_op,
                        const char* op_name, GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_88_v;
   mht_88_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_88(mht_88_v, 2468, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertPowOperator");

  tensorflow::NodeDef* pow_op = tensorflow_graph->add_node();
  pow_op->set_op(op_name);
  pow_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  for (int i = 0; i < 2; ++i) {
    *pow_op->add_input() = src_op.inputs[i];
  }
  const tensorflow::DataType data_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*pow_op->mutable_attr())["T"].set_type(data_type);
}

void ConvertLogicalAndOperator(const Model& model,
                               const LogicalAndOperator& src_op,
                               GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_89(mht_89_v, 2486, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertLogicalAndOperator");

  tensorflow::NodeDef* logical_op = tensorflow_graph->add_node();
  logical_op->set_op("LogicalAnd");
  logical_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  for (int i = 0; i < 2; ++i) {
    *logical_op->add_input() = src_op.inputs[i];
  }
}

void ConvertLogicalNotOperator(const Model& model,
                               const LogicalNotOperator& src_op,
                               GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_90(mht_90_v, 2501, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertLogicalNotOperator");

  tensorflow::NodeDef* logical_op = tensorflow_graph->add_node();
  logical_op->set_op("LogicalNot");
  logical_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 1);
  *logical_op->add_input() = src_op.inputs[0];
}

void ConvertLogicalOrOperator(const Model& model,
                              const LogicalOrOperator& src_op,
                              const char* op_name, GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_91_v;
   mht_91_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_91(mht_91_v, 2515, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertLogicalOrOperator");

  tensorflow::NodeDef* logical_or_op = tensorflow_graph->add_node();
  logical_or_op->set_op(op_name);
  logical_or_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  for (int i = 0; i < 2; ++i) {
    *logical_or_op->add_input() = src_op.inputs[i];
  }
  const tensorflow::DataType data_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*logical_or_op->mutable_attr())["T"].set_type(data_type);
}

void ConvertCTCBeamSearchDecoderOperator(
    const Model& model, const CTCBeamSearchDecoderOperator& src_op,
    const char* op_name, GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_92_v;
   mht_92_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_92(mht_92_v, 2534, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertCTCBeamSearchDecoderOperator");

  auto* op = tensorflow_graph->add_node();
  op->set_op(op_name);
  op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  for (int i = 0; i < 2; ++i) {
    *op->add_input() = src_op.inputs[i];
  }
  (*op->mutable_attr())["beam_width"].set_i(src_op.beam_width);
  (*op->mutable_attr())["top_paths"].set_i(src_op.top_paths);
  (*op->mutable_attr())["merge_repeated"].set_b(src_op.merge_repeated);
}

void ConvertUnpackOperator(const Model& model, const UnpackOperator& src_op,
                           const char* op_name, GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_93_v;
   mht_93_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_93(mht_93_v, 2552, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertUnpackOperator");

  tensorflow::NodeDef* unpack_op = tensorflow_graph->add_node();
  unpack_op->set_op(op_name);
  unpack_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *unpack_op->add_input() = src_op.inputs[0];
  const tensorflow::DataType data_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*unpack_op->mutable_attr())["T"].set_type(data_type);
  (*unpack_op->mutable_attr())["num"].set_i(src_op.num);
  (*unpack_op->mutable_attr())["axis"].set_i(src_op.axis);
}

void ConvertZerosLikeOperator(const Model& model,
                              const TensorFlowZerosLikeOperator& src_op,
                              const char* op_name, GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_94_v;
   mht_94_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_94(mht_94_v, 2571, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertZerosLikeOperator");

  tensorflow::NodeDef* zeros_like_op = tensorflow_graph->add_node();
  zeros_like_op->set_op(op_name);
  zeros_like_op->set_name(src_op.outputs[0]);
  DCHECK_EQ(src_op.inputs.size(), 1);
  *zeros_like_op->add_input() = src_op.inputs[0];
  const tensorflow::DataType data_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*zeros_like_op->mutable_attr())["T"].set_type(data_type);
}

void ConvertReverseV2Operator(const Model& model,
                              const ReverseV2Operator& src_op,
                              const char* op_name, GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_95_v;
   mht_95_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_95(mht_95_v, 2588, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertReverseV2Operator");

  tensorflow::NodeDef* reverse_v2_op = tensorflow_graph->add_node();
  reverse_v2_op->set_op(op_name);
  reverse_v2_op->set_name(src_op.outputs[0]);
  DCHECK_EQ(src_op.inputs.size(), 2);
  *reverse_v2_op->add_input() = src_op.inputs[0];
  *reverse_v2_op->add_input() = src_op.inputs[1];
  const tensorflow::DataType data_type =
      GetTensorFlowDataType(model, src_op.inputs[0]);
  (*reverse_v2_op->mutable_attr())["T"].set_type(data_type);
}

void ConvertReverseSequenceOperator(const Model& model,
                                    const ReverseSequenceOperator& src_op,
                                    GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_96(mht_96_v, 2605, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertReverseSequenceOperator");

  tensorflow::NodeDef* reverse_seq_op = tensorflow_graph->add_node();
  reverse_seq_op->set_op("ReverseSequence");
  reverse_seq_op->set_name(src_op.outputs[0]);
  CHECK_EQ(src_op.inputs.size(), 2);
  *reverse_seq_op->add_input() = src_op.inputs[0];
  *reverse_seq_op->add_input() = src_op.inputs[1];
  (*reverse_seq_op->mutable_attr())["seq_dim"].set_i(src_op.seq_dim);
  (*reverse_seq_op->mutable_attr())["batch_dim"].set_i(src_op.batch_dim);
}

void ConvertOperator(const Model& model, const Operator& src_op,
                     GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_97(mht_97_v, 2620, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ConvertOperator");

  if (src_op.fused_activation_function != FusedActivationFunctionType::kNone) {
    LOG(FATAL)
        << "Unsupported: the input model has a fused activation function";
  }

  if (src_op.type == OperatorType::kConv) {
    ConvertConvOperator(model, static_cast<const ConvOperator&>(src_op),
                        tensorflow_graph);
  } else if (src_op.type == OperatorType::kDepthwiseConv) {
    ConvertDepthwiseConvOperator(
        model, static_cast<const DepthwiseConvOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kDepthToSpace) {
    ConvertDepthToSpaceOperator(
        model, static_cast<const DepthToSpaceOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kSpaceToDepth) {
    ConvertSpaceToDepthOperator(
        model, static_cast<const SpaceToDepthOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kFullyConnected) {
    ConvertFullyConnectedOperator(
        model, static_cast<const FullyConnectedOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kAdd) {
    ConvertAddOperator(model, static_cast<const AddOperator&>(src_op),
                       tensorflow_graph);
  } else if (src_op.type == OperatorType::kAddN) {
    ConvertAddNOperator(model, static_cast<const AddNOperator&>(src_op),
                        tensorflow_graph);
  } else if (src_op.type == OperatorType::kMul) {
    ConvertMulOperator(model, static_cast<const MulOperator&>(src_op),
                       tensorflow_graph);
  } else if (src_op.type == OperatorType::kDiv) {
    ConvertDivOperator(model, static_cast<const DivOperator&>(src_op),
                       tensorflow_graph);
  } else if (src_op.type == OperatorType::kRelu) {
    ConvertReluOperator(model, static_cast<const ReluOperator&>(src_op),
                        tensorflow_graph);
  } else if (src_op.type == OperatorType::kRelu1) {
    ConvertRelu1Operator(static_cast<const Relu1Operator&>(src_op),
                         tensorflow_graph);
  } else if (src_op.type == OperatorType::kRelu6) {
    ConvertRelu6Operator(static_cast<const Relu6Operator&>(src_op),
                         tensorflow_graph);
  } else if (src_op.type == OperatorType::kLog) {
    ConvertLogOperator(static_cast<const LogOperator&>(src_op),
                       tensorflow_graph);
  } else if (src_op.type == OperatorType::kLogistic) {
    ConvertLogisticOperator(static_cast<const LogisticOperator&>(src_op),
                            tensorflow_graph);
  } else if (src_op.type == OperatorType::kTanh) {
    ConvertTanhOperator(static_cast<const TanhOperator&>(src_op),
                        tensorflow_graph);
  } else if (src_op.type == OperatorType::kL2Normalization) {
    ConvertL2NormalizationOperator(
        static_cast<const L2NormalizationOperator&>(src_op), tensorflow_graph);
  } else if (src_op.type == OperatorType::kSoftmax) {
    ConvertSoftmaxOperator(model, static_cast<const SoftmaxOperator&>(src_op),
                           tensorflow_graph);
  } else if (src_op.type == OperatorType::kLogSoftmax) {
    ConvertLogSoftmaxOperator(model,
                              static_cast<const LogSoftmaxOperator&>(src_op),
                              tensorflow_graph);
  } else if (src_op.type == OperatorType::kLocalResponseNormalization) {
    ConvertLocalResponseNormalizationOperator(
        static_cast<const LocalResponseNormalizationOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kLstmCell) {
    ConvertLstmCellOperator(model, static_cast<const LstmCellOperator&>(src_op),
                            tensorflow_graph);
  } else if (src_op.type == OperatorType::kMaxPool) {
    ConvertMaxPoolOperator(static_cast<const MaxPoolOperator&>(src_op),
                           tensorflow_graph);
  } else if (src_op.type == OperatorType::kAveragePool) {
    ConvertAveragePoolOperator(static_cast<const AveragePoolOperator&>(src_op),
                               tensorflow_graph);
  } else if (src_op.type == OperatorType::kConcatenation) {
    ConvertConcatenationOperator(
        model, static_cast<const ConcatenationOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kReshape) {
    ConvertTensorFlowReshapeOperator(
        model, static_cast<const TensorFlowReshapeOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kL2Pool) {
    ConvertL2PoolOperator(static_cast<const L2PoolOperator&>(src_op),
                          tensorflow_graph);
  } else if (src_op.type == OperatorType::kSquare) {
    ConvertSquareOperator(static_cast<const TensorFlowSquareOperator&>(src_op),
                          tensorflow_graph);
  } else if (src_op.type == OperatorType::kSqrt) {
    ConvertSqrtOperator(static_cast<const TensorFlowSqrtOperator&>(src_op),
                        tensorflow_graph);
  } else if (src_op.type == OperatorType::kRsqrt) {
    ConvertRsqrtOperator(model,
                         static_cast<const TensorFlowRsqrtOperator&>(src_op),
                         tensorflow_graph);
  } else if (src_op.type == OperatorType::kSplit) {
    ConvertSplitOperator(model,
                         static_cast<const TensorFlowSplitOperator&>(src_op),
                         tensorflow_graph);
  } else if (src_op.type == OperatorType::kSplitV) {
    ConvertSplitVOperator(model,
                          static_cast<const TensorFlowSplitVOperator&>(src_op),
                          tensorflow_graph);
  } else if (src_op.type == OperatorType::kFakeQuant) {
    ConvertFakeQuantOperator(static_cast<const FakeQuantOperator&>(src_op),
                             tensorflow_graph);
  } else if (src_op.type == OperatorType::kCast) {
    ConvertCastOperator(model, static_cast<const CastOperator&>(src_op),
                        tensorflow_graph);
  } else if (src_op.type == OperatorType::kFloor) {
    ConvertFloorOperator(model, static_cast<const FloorOperator&>(src_op),
                         tensorflow_graph);
  } else if (src_op.type == OperatorType::kCeil) {
    ConvertCeilOperator(model, static_cast<const CeilOperator&>(src_op),
                        tensorflow_graph);
  } else if (src_op.type == OperatorType::kRound) {
    ConvertRoundOperator(model, static_cast<const RoundOperator&>(src_op),
                         tensorflow_graph);
  } else if (src_op.type == OperatorType::kGather) {
    ConvertGatherOperator(model, static_cast<const GatherOperator&>(src_op),
                          tensorflow_graph);
  } else if (src_op.type == OperatorType::kResizeBilinear) {
    ConvertResizeBilinearOperator(
        model, static_cast<const ResizeBilinearOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kResizeNearestNeighbor) {
    ConvertResizeNearestNeighborOperator(
        model, static_cast<const ResizeNearestNeighborOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kSpaceToBatchND) {
    ConvertSpaceToBatchNDOperator(
        model, static_cast<const SpaceToBatchNDOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kBatchToSpaceND) {
    ConvertBatchToSpaceNDOperator(
        model, static_cast<const BatchToSpaceNDOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kPad) {
    ConvertPadOperator(model, static_cast<const PadOperator&>(src_op),
                       tensorflow_graph);
  } else if (src_op.type == OperatorType::kPadV2) {
    ConvertPadV2Operator(model, static_cast<const PadV2Operator&>(src_op),
                         tensorflow_graph);
  } else if (src_op.type == OperatorType::kStridedSlice) {
    ConvertStridedSliceOperator(
        model, static_cast<const StridedSliceOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kMean) {
    ConvertReduceOperator(model, static_cast<const MeanOperator&>(src_op),
                          tensorflow_graph, "Mean");
  } else if (src_op.type == OperatorType::kSum) {
    ConvertReduceOperator(model,
                          static_cast<const TensorFlowSumOperator&>(src_op),
                          tensorflow_graph, "Sum");
  } else if (src_op.type == OperatorType::kReduceProd) {
    ConvertReduceOperator(model,
                          static_cast<const TensorFlowProdOperator&>(src_op),
                          tensorflow_graph, "Prod");
  } else if (src_op.type == OperatorType::kReduceMin) {
    ConvertReduceOperator(model,
                          static_cast<const TensorFlowMinOperator&>(src_op),
                          tensorflow_graph, "Min");
  } else if (src_op.type == OperatorType::kReduceMax) {
    ConvertReduceOperator(model,
                          static_cast<const TensorFlowMaxOperator&>(src_op),
                          tensorflow_graph, "Max");
  } else if (src_op.type == OperatorType::kSub) {
    ConvertSubOperator(model, static_cast<const SubOperator&>(src_op),
                       tensorflow_graph);
  } else if (src_op.type == OperatorType::kMinimum) {
    ConvertTensorFlowMinimumOperator(
        model, static_cast<const TensorFlowMinimumOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kMaximum) {
    ConvertTensorFlowMaximumOperator(
        model, static_cast<const TensorFlowMaximumOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kSqueeze) {
    ConvertSqueezeOperator(model, static_cast<const SqueezeOperator&>(src_op),
                           tensorflow_graph);
  } else if (src_op.type == OperatorType::kSlice) {
    ConvertSliceOperator(model, static_cast<const SliceOperator&>(src_op),
                         tensorflow_graph);
  } else if (src_op.type == OperatorType::kArgMax) {
    ConvertArgMaxOperator(model, static_cast<const ArgMaxOperator&>(src_op),
                          tensorflow_graph);
  } else if (src_op.type == OperatorType::kArgMin) {
    ConvertArgMinOperator(model, static_cast<const ArgMinOperator&>(src_op),
                          tensorflow_graph);
  } else if (src_op.type == OperatorType::kTopK_V2) {
    ConvertTopKV2Operator(model, static_cast<const TopKV2Operator&>(src_op),
                          tensorflow_graph);
  } else if (src_op.type == OperatorType::kTranspose) {
    ConvertTransposeOperator(
        model, static_cast<const TransposeOperator&>(src_op), tensorflow_graph);
  } else if (src_op.type == OperatorType::kShape) {
    ConvertTensorFlowShapeOperator(
        model, static_cast<const TensorFlowShapeOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kRank) {
    ConvertRankOperator(model,
                        static_cast<const TensorFlowRankOperator&>(src_op),
                        tensorflow_graph);
  } else if (src_op.type == OperatorType::kRange) {
    ConvertRangeOperator(model, static_cast<const RangeOperator&>(src_op),
                         tensorflow_graph);
  } else if (src_op.type == OperatorType::kPack) {
    ConvertPackOperator(model, static_cast<const PackOperator&>(src_op),
                        tensorflow_graph);
  } else if (src_op.type == OperatorType::kFill) {
    ConvertFillOperator(model, static_cast<const FillOperator&>(src_op),
                        tensorflow_graph);
  } else if (src_op.type == OperatorType::kFloorDiv) {
    ConvertFloorDivOperator(model, static_cast<const FloorDivOperator&>(src_op),
                            tensorflow_graph);
  } else if (src_op.type == OperatorType::kFloorMod) {
    ConvertFloorModOperator(model, static_cast<const FloorModOperator&>(src_op),
                            tensorflow_graph);
  } else if (src_op.type == OperatorType::kExpandDims) {
    ConvertExpandDimsOperator(model,
                              static_cast<const ExpandDimsOperator&>(src_op),
                              tensorflow_graph);
  } else if (src_op.type == OperatorType::kTransposeConv) {
    ConvertTransposeConvOperator(
        model, static_cast<const TransposeConvOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kRandomUniform) {
    ConvertRandomUniformOperator(
        model, static_cast<const RandomUniformOperator&>(src_op),
        tensorflow_graph);
  } else if (src_op.type == OperatorType::kEqual) {
    ConvertComparisonOperator(model, src_op, "Equal", tensorflow_graph);
  } else if (src_op.type == OperatorType::kNotEqual) {
    ConvertComparisonOperator(model, src_op, "NotEqual", tensorflow_graph);
  } else if (src_op.type == OperatorType::kGreater) {
    ConvertComparisonOperator(model, src_op, "Greater", tensorflow_graph);
  } else if (src_op.type == OperatorType::kGreaterEqual) {
    ConvertComparisonOperator(model, src_op, "GreaterEqual", tensorflow_graph);
  } else if (src_op.type == OperatorType::kLess) {
    ConvertComparisonOperator(model, src_op, "Less", tensorflow_graph);
  } else if (src_op.type == OperatorType::kLessEqual) {
    ConvertComparisonOperator(model, src_op, "LessEqual", tensorflow_graph);
  } else if (src_op.type == OperatorType::kSelect) {
    ConvertSelectOperator(model, static_cast<const SelectOperator&>(src_op),
                          tensorflow_graph);
  } else if (src_op.type == OperatorType::kTile) {
    ConvertTileOperator(model,
                        static_cast<const TensorFlowTileOperator&>(src_op),
                        tensorflow_graph);
  } else if (src_op.type == OperatorType::kPow) {
    ConvertPowOperator(model, static_cast<const PowOperator&>(src_op), "Pow",
                       tensorflow_graph);
  } else if (src_op.type == OperatorType::kAny) {
    ConvertReduceOperator(model,
                          static_cast<const TensorFlowAnyOperator&>(src_op),
                          tensorflow_graph, "Any");
  } else if (src_op.type == OperatorType::kLogicalAnd) {
    ConvertLogicalAndOperator(model,
                              static_cast<const LogicalAndOperator&>(src_op),
                              tensorflow_graph);
  } else if (src_op.type == OperatorType::kLogicalNot) {
    ConvertLogicalNotOperator(model,
                              static_cast<const LogicalNotOperator&>(src_op),
                              tensorflow_graph);
  } else if (src_op.type == OperatorType::kOneHot) {
    ConvertOneHotOperator(model, static_cast<const OneHotOperator&>(src_op),
                          tensorflow_graph);
  } else if (src_op.type == OperatorType::kLogicalOr) {
    ConvertLogicalOrOperator(model,
                             static_cast<const LogicalOrOperator&>(src_op),
                             "LogicalOr", tensorflow_graph);
  } else if (src_op.type == OperatorType::kCTCBeamSearchDecoder) {
    ConvertCTCBeamSearchDecoderOperator(
        model, static_cast<const CTCBeamSearchDecoderOperator&>(src_op),
        "CTCBeamSearchDecoder", tensorflow_graph);
  } else if (src_op.type == OperatorType::kUnpack) {
    ConvertUnpackOperator(model, static_cast<const UnpackOperator&>(src_op),
                          "Unpack", tensorflow_graph);
  } else if (src_op.type == OperatorType::kZerosLike) {
    ConvertZerosLikeOperator(
        model, static_cast<const TensorFlowZerosLikeOperator&>(src_op),
        "ZerosLike", tensorflow_graph);
  } else if (src_op.type == OperatorType::kReverseV2) {
    ConvertReverseV2Operator(model,
                             static_cast<const ReverseV2Operator&>(src_op),
                             "Reverse_V2", tensorflow_graph);
  } else if (src_op.type == OperatorType::kReverseSequence) {
    ConvertReverseSequenceOperator(
        model, static_cast<const ReverseSequenceOperator&>(src_op),
        tensorflow_graph);
  } else {
    LOG(FATAL) << "Unhandled operator type " << OperatorTypeName(src_op.type);
  }
}

void AddPlaceholder(const std::string& name, ArrayDataType type,
                    GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_98_v;
   mht_98_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_98(mht_98_v, 2924, "", "./tensorflow/lite/toco/export_tensorflow.cc", "AddPlaceholder");

  tensorflow::NodeDef* placeholder = tensorflow_graph->add_node();
  placeholder->set_op("Placeholder");
  switch (type) {
    case ArrayDataType::kBool:
      (*placeholder->mutable_attr())["dtype"].set_type(DT_BOOL);
      break;
    case ArrayDataType::kFloat:
      (*placeholder->mutable_attr())["dtype"].set_type(DT_FLOAT);
      break;
    case ArrayDataType::kUint8:
      (*placeholder->mutable_attr())["dtype"].set_type(DT_UINT8);
      break;
    case ArrayDataType::kInt32:
      (*placeholder->mutable_attr())["dtype"].set_type(DT_INT32);
      break;
    case ArrayDataType::kUint32:
      (*placeholder->mutable_attr())["dtype"].set_type(DT_UINT32);
      break;
    case ArrayDataType::kInt64:
      (*placeholder->mutable_attr())["dtype"].set_type(DT_INT64);
      break;
    case ArrayDataType::kInt16:
      (*placeholder->mutable_attr())["dtype"].set_type(DT_INT16);
      break;
    case ArrayDataType::kComplex64:
      (*placeholder->mutable_attr())["dtype"].set_type(DT_COMPLEX64);
      break;
    default:
      LOG(FATAL) << "Unexpected data type in array \"" << name << "\"";
  }
  placeholder->set_name(name);
}

void AddPlaceholderForRNNState(const Model& model, const std::string& name,
                               int size, GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_99_v;
   mht_99_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_99(mht_99_v, 2963, "", "./tensorflow/lite/toco/export_tensorflow.cc", "AddPlaceholderForRNNState");

  tensorflow::NodeDef* placeholder = tensorflow_graph->add_node();
  placeholder->set_op("Placeholder");
  placeholder->set_name(name);
  (*placeholder->mutable_attr())["dtype"].set_type(DT_FLOAT);

  auto* shape = (*placeholder->mutable_attr())["shape"].mutable_shape();
  const auto& state_array = model.GetArray(name);
  if (state_array.has_shape()) {
    const auto& state_shape = state_array.shape();
    const int kDims = state_shape.dimensions_count();
    for (int i = 0; i < kDims; ++i) {
      shape->add_dim()->set_size(state_shape.dims(i));
    }
  } else {
    shape->add_dim()->set_size(1);
    shape->add_dim()->set_size(size);
  }
}

void ExportTensorFlowGraphDefImplementation(const Model& model,
                                            GraphDef* tensorflow_graph) {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_100(mht_100_v, 2987, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ExportTensorFlowGraphDefImplementation");

  for (const auto& input_array : model.flags.input_arrays()) {
    AddPlaceholder(input_array.name(),
                   model.GetArray(input_array.name()).data_type,
                   tensorflow_graph);
  }
  for (const auto& rnn_state : model.flags.rnn_states()) {
    AddPlaceholderForRNNState(model, rnn_state.state_array(), rnn_state.size(),
                              tensorflow_graph);
  }
  for (const auto& op : model.operators) {
    ConvertOperator(model, *op, tensorflow_graph);
  }
  // Generically export arrays that haven't been exported already
  // by the above operators export. It's important that this comes
  // after, as some operators need to export arrays that they reference
  // in a specific way, rather than in the generic way done below.
  for (const auto& array_pair : model.GetArrayMap()) {
    const std::string& array_name = array_pair.first;
    const auto& array = *array_pair.second;
    if (array.buffer) {
      switch (array.data_type) {
        case ArrayDataType::kBool:
          ConvertBoolTensorConst(model, array_name, tensorflow_graph);
          break;
        case ArrayDataType::kFloat:
          ConvertFloatTensorConst(model, array_name, tensorflow_graph);
          break;
        case ArrayDataType::kInt32:
          ConvertIntTensorConst(model, array_name, tensorflow_graph);
          break;
        case ArrayDataType::kComplex64:
          ConvertComplex64TensorConst(model, array_name, tensorflow_graph);
          break;
        default:
          break;
      }
    }
  }
}
}  // namespace

void EncodeConstantArraysMinMaxByWrappingThemInFakeQuantNodes(Model* model) {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_101(mht_101_v, 3032, "", "./tensorflow/lite/toco/export_tensorflow.cc", "EncodeConstantArraysMinMaxByWrappingThemInFakeQuantNodes");

  for (const auto& array_kv : model->GetArrayMap()) {
    const std::string& array_name = array_kv.first;
    Array& array = *array_kv.second;
    if (!array.buffer || !array.minmax) {
      continue;
    }
    const std::string& wrapped_array_name =
        AvailableArrayName(*model, array_name + "/data");
    Array& wrapped_array = model->GetOrCreateArray(wrapped_array_name);
    wrapped_array.data_type = array.data_type;
    wrapped_array.copy_shape(array.shape());
    wrapped_array.buffer = std::move(array.buffer);
    FakeQuantOperator* fakequant_op = new FakeQuantOperator;
    fakequant_op->inputs = {wrapped_array_name};
    fakequant_op->outputs = {array_name};
    fakequant_op->minmax.reset(new MinMax);
    *fakequant_op->minmax = *array.minmax;
    const auto& it = FindOpWithInput(*model, array_name);
    model->operators.emplace(it, fakequant_op);
  }
  CheckInvariants(*model);
}

void ExportTensorFlowGraphDef(const Model& model,
                              std::string* output_file_contents) {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPSlitePStocoPSexport_tensorflowDTcc mht_102(mht_102_v, 3060, "", "./tensorflow/lite/toco/export_tensorflow.cc", "ExportTensorFlowGraphDef");

  CHECK(output_file_contents->empty());
  GraphDef tensorflow_graph;
  ExportTensorFlowGraphDefImplementation(model, &tensorflow_graph);
  LogDumpGraphDef(kLogLevelModelChanged, "AT EXPORT", tensorflow_graph);
  CHECK(tensorflow_graph.SerializeToString(output_file_contents));
}
}  // namespace toco
