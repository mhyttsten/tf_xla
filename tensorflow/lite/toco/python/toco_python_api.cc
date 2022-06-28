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
class MHTracer_DTPStensorflowPSlitePStocoPSpythonPStoco_python_apiDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSpythonPStoco_python_apiDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSpythonPStoco_python_apiDTcc() {
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
#include "tensorflow/lite/toco/python/toco_python_api.h"

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "google/protobuf/text_format.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/compiler/mlir/lite/metrics/error_collector.h"
#include "tensorflow/compiler/mlir/lite/python/flatbuffer_to_mlir.h"
#include "tensorflow/compiler/mlir/lite/python/graphdef_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/lite/python/jax_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/lite/python/saved_model_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/quantize_model.h"
#include "tensorflow/compiler/mlir/lite/sparsity/sparsify_model.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_error_reporter.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/toco/import_tensorflow.h"
#include "tensorflow/lite/toco/logging/conversion_log_util.h"
#include "tensorflow/lite/toco/logging/toco_conversion_log.pb.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_convert.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"
#include "tensorflow/lite/toco/toco_graphviz_dump_options.h"
#include "tensorflow/lite/toco/toco_port.h"
#include "tensorflow/lite/toco/toco_tooling.h"
#include "tensorflow/lite/toco/toco_types.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/lite/toco/types.pb.h"

namespace toco {
using mlir::lite::StringSet;

void PopulateConversionLogHelper(const toco::ModelFlags& model_flags,
                                 toco::TocoFlags* toco_flags,
                                 const std::string& input_contents_txt,
                                 const std::string& output_file_contents_txt,
                                 const std::string& error_message,
                                 GraphVizDumpOptions* dump_options) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("input_contents_txt: \"" + input_contents_txt + "\"");
   mht_0_v.push_back("output_file_contents_txt: \"" + output_file_contents_txt + "\"");
   mht_0_v.push_back("error_message: \"" + error_message + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSpythonPStoco_python_apiDTcc mht_0(mht_0_v, 232, "", "./tensorflow/lite/toco/python/toco_python_api.cc", "PopulateConversionLogHelper");

  // Make sure the graphviz file will be dumped under the same folder.
  dump_options->dump_graphviz = toco_flags->conversion_summary_dir();
  // Here we construct the `toco::Model` class based on the input graph def,
  // it will then be used to populate the conversion log.
  // TODO(haoliang): Don't depend on `toco::Model`.
  std::unique_ptr<toco::Model> imported_model =
      toco::Import(*toco_flags, model_flags, input_contents_txt);
  // Dump pre-conversion toco logs.
  TocoConversionLog toco_log_before;
  PopulateConversionLog(*imported_model, &toco_log_before);
  std::ofstream osstream_before(toco_flags->conversion_summary_dir() +
                                "/toco_log_before.pb");
  toco_log_before.SerializeToOstream(&osstream_before);
  osstream_before.close();
  toco::LogDump(toco::kLogLevelModelChanged, "tf_graph", *imported_model);

  // Populate the post-conversion log, for convenient initiate the
  // `toco::Model` class from the generated flatbuffer.
  toco_flags->set_input_format(toco::FileFormat::TFLITE);
  std::unique_ptr<toco::Model> flatbuffer_model =
      toco::Import(*toco_flags, model_flags, output_file_contents_txt);
  // Dump post-conversion toco logs.
  TocoConversionLog toco_log_after;
  PopulateConversionLog(*flatbuffer_model, &toco_log_after);
  // Make sure we sanitize the error message.
  toco_log_after.set_toco_err_logs(SanitizeErrorMessage(error_message));
  std::ofstream ostream_after(toco_flags->conversion_summary_dir() +
                              "/toco_log_after.pb");
  toco_log_after.SerializeToOstream(&ostream_after);
  ostream_after.close();
  toco::LogDump(toco::kLogLevelModelChanged, "tflite_graph", *flatbuffer_model);
}

// NOTE(aselle): We are using raw PyObject's here because we want to make
// sure we input and output bytes rather than unicode strings for Python3.
PyObject* TocoConvert(PyObject* model_flags_proto_txt_raw,
                      PyObject* toco_flags_proto_txt_raw,
                      PyObject* input_contents_txt_raw, bool extended_return,
                      PyObject* debug_info_txt_raw,
                      bool enable_mlir_converter) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSpythonPStoco_python_apiDTcc mht_1(mht_1_v, 275, "", "./tensorflow/lite/toco/python/toco_python_api.cc", "TocoConvert");

  // Use Python C API to validate and convert arguments. In py3 (bytes),
  // in py2 (str).
  auto ConvertArg = [&](PyObject* obj, bool* error) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSpythonPStoco_python_apiDTcc mht_2(mht_2_v, 281, "", "./tensorflow/lite/toco/python/toco_python_api.cc", "lambda");

    char* buf;
    Py_ssize_t len;
    if (::tflite::python_utils::ConvertFromPyString(obj, &buf, &len) == -1) {
      *error = true;
      return std::string();
    } else {
      *error = false;
      return std::string(buf, len);
    }
  };

  bool error;
  std::string model_flags_proto_txt =
      ConvertArg(model_flags_proto_txt_raw, &error);
  if (error) {
    PyErr_SetString(PyExc_ValueError, "Model flags are invalid.");
    return nullptr;
  }
  std::string toco_flags_proto_txt =
      ConvertArg(toco_flags_proto_txt_raw, &error);
  if (error) {
    PyErr_SetString(PyExc_ValueError, "Toco flags are invalid.");
    return nullptr;
  }

  // Use TOCO to produce new outputs.
  toco::ModelFlags model_flags;
  if (!model_flags.ParseFromString(model_flags_proto_txt)) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert Model to Python String.");
    return nullptr;
  }
  toco::TocoFlags toco_flags;
  if (!toco_flags.ParseFromString(toco_flags_proto_txt)) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert Toco to Python String.");
    return nullptr;
  }

  tensorflow::GraphDebugInfo debug_info;
  if (debug_info_txt_raw && debug_info_txt_raw != Py_None) {
    std::string debug_info_txt = ConvertArg(debug_info_txt_raw, &error);
    if (error) {
      PyErr_SetString(PyExc_ValueError, "Input DebugInfo is invalid.");
      return nullptr;
    }
    if (!debug_info.ParseFromString(debug_info_txt)) {
      PyErr_SetString(PyExc_ValueError,
                      "Failed to convert DebugInfo to Python String.");
      return nullptr;
    }
  }

  tensorflow::GraphDef graph_def;
  std::string input_contents_txt;
  if (model_flags.saved_model_dir().empty()) {
    input_contents_txt = ConvertArg(input_contents_txt_raw, &error);
    if (error) {
      PyErr_SetString(PyExc_ValueError, "Input GraphDef is invalid.");
      return nullptr;
    }
    if (!model_flags.use_hlo_import() &&
        !graph_def.ParseFromString(input_contents_txt)) {
      PyErr_SetString(PyExc_ValueError,
                      "Failed to convert GraphDef to Python String.");
      return nullptr;
    }
  }

  auto& dump_options = *GraphVizDumpOptions::singleton();
  if (toco_flags.has_dump_graphviz_dir()) {
    dump_options.dump_graphviz = toco_flags.dump_graphviz_dir();
  }
  if (toco_flags.has_dump_graphviz_include_video()) {
    dump_options.dump_graphviz_video = toco_flags.dump_graphviz_include_video();
  }

  std::string output_file_contents_txt;
  tensorflow::Status status;
  int64_t arithmetic_ops_count;

  // Convert model.
  if (enable_mlir_converter) {
    if (model_flags.use_hlo_import() && model_flags.has_saved_model_dir()) {
      PyErr_SetString(PyExc_ValueError,
                      "Cannot specify both saved_model and hlo import.");
      return nullptr;
    }

    if (model_flags.use_hlo_import()) {
      status = tensorflow::ConvertJaxToTFLiteFlatBuffer(
          input_contents_txt, model_flags, toco_flags,
          &output_file_contents_txt);
    } else if (!model_flags.saved_model_dir().empty()) {
      status = tensorflow::ConvertSavedModelToTFLiteFlatBuffer(
          model_flags, toco_flags, &output_file_contents_txt);
    } else {
      tensorflow::GraphDef graph_def;
      if (!graph_def.ParseFromString(input_contents_txt)) {
        PyErr_SetString(PyExc_ValueError,
                        "Failed to convert GraphDef to Python String.");
        return nullptr;
      }

      status = tensorflow::ConvertGraphDefToTFLiteFlatBuffer(
          model_flags, toco_flags, debug_info, graph_def,
          &output_file_contents_txt);
      if (!toco_flags.conversion_summary_dir().empty()) {
        PopulateConversionLogHelper(
            model_flags, &toco_flags, input_contents_txt,
            output_file_contents_txt, status.error_message(), &dump_options);
      }
    }
  } else {
    status = Convert(input_contents_txt, toco_flags, model_flags,
                     &output_file_contents_txt, &arithmetic_ops_count);
  }

  if (!status.ok()) {
    PyErr_SetString(PyExc_Exception, status.error_message().c_str());
    return nullptr;
  }
  if (extended_return && !enable_mlir_converter) {
    PyObject* dict = PyDict_New();
    PyDict_SetItemString(
        dict, "flatbuffer",
        ::tflite::python_utils::ConvertToPyString(
            output_file_contents_txt.data(), output_file_contents_txt.size()));
    PyDict_SetItemString(dict, "arithmetic_ops",
                         PyLong_FromLong(arithmetic_ops_count));
    return dict;
  }
  // Convert arguments back to byte (py3) or str (py2)
  return ::tflite::python_utils::ConvertToPyString(
      output_file_contents_txt.data(), output_file_contents_txt.size());
}

tflite::TensorType FromTocoDataTypeToTflitToTensorType(int inference_type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPSpythonPStoco_python_apiDTcc mht_3(mht_3_v, 422, "", "./tensorflow/lite/toco/python/toco_python_api.cc", "FromTocoDataTypeToTflitToTensorType");

  switch (inference_type) {
    case toco::IODataType::QUANTIZED_INT16:
      return tflite::TensorType_INT16;
    case toco::IODataType::QUANTIZED_UINT8:
      return tflite::TensorType_UINT8;
    case toco::IODataType::UINT8:
      return tflite::TensorType_UINT8;
    case toco::IODataType::QUANTIZED_INT8:
      return tflite::TensorType_INT8;
    case toco::IODataType::INT8:
      return tflite::TensorType_INT8;
    default:
      return tflite::TensorType_FLOAT32;
  }
}

int ToStringSet(PyObject* py_denylist, StringSet* string_set) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPSpythonPStoco_python_apiDTcc mht_4(mht_4_v, 442, "", "./tensorflow/lite/toco/python/toco_python_api.cc", "ToStringSet");

  using tflite::python_utils::ConvertFromPyString;
  // Ensure op_denylist is non null
  if (!py_denylist) {
    return 0;
  }
  if (PyList_Check(py_denylist)) {
    for (int i = 0; i < PyList_GET_SIZE(py_denylist); ++i) {
      PyObject* value = PyList_GetItem(py_denylist, i);
      char* str_buf;
      Py_ssize_t length;
      if (ConvertFromPyString(value, &str_buf, &length) == -1) {
        return -1;
      }
      string_set->emplace(str_buf, length);
    }
  }
  if (PySet_Check(py_denylist)) {
    auto* tmp = PySet_New(py_denylist);
    while (PySet_GET_SIZE(tmp)) {
      PyObject* value = PySet_Pop(tmp);
      char* str_buf;
      Py_ssize_t length;
      if (ConvertFromPyString(value, &str_buf, &length) == -1) {
        return -1;
      }
      string_set->emplace(str_buf, length);
    }
  }
  return 0;
}

PyObject* MlirQuantizeModel(PyObject* data, bool disable_per_channel,
                            bool fully_quantize, int inference_type,
                            int input_data_type, int output_data_type,
                            bool enable_numeric_verify,
                            bool enable_whole_model_verify,
                            PyObject* op_denylist, PyObject* node_denylist) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStocoPSpythonPStoco_python_apiDTcc mht_5(mht_5_v, 482, "", "./tensorflow/lite/toco/python/toco_python_api.cc", "MlirQuantizeModel");

  using tflite::interpreter_wrapper::PythonErrorReporter;
  char* buf = nullptr;
  Py_ssize_t length;
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);

  if (tflite::python_utils::ConvertFromPyString(data, &buf, &length) == -1) {
    PyErr_Format(PyExc_ValueError, "Failed to convert input PyObject");
    return nullptr;
  }

  StringSet denylisted_ops;
  StringSet denylisted_nodes;
  if (ToStringSet(op_denylist, &denylisted_ops) == -1) {
    PyErr_Format(PyExc_ValueError, "Failed to convert op denylist PyObject");
    return nullptr;
  }
  if (ToStringSet(node_denylist, &denylisted_nodes) == -1) {
    PyErr_Format(PyExc_ValueError, "Failed to convert node denylist PyObject");
    return nullptr;
  }

  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(buf, length,
                                               error_reporter.get());
  if (!model) {
    PyErr_Format(PyExc_ValueError, "Invalid model");
    return nullptr;
  }
  auto tflite_model = absl::make_unique<tflite::ModelT>();
  model->GetModel()->UnPackTo(tflite_model.get(), nullptr);

  tflite::TensorType inference_tensor_type =
      FromTocoDataTypeToTflitToTensorType(inference_type);
  tflite::TensorType input_type =
      FromTocoDataTypeToTflitToTensorType(input_data_type);
  tflite::TensorType output_type =
      FromTocoDataTypeToTflitToTensorType(output_data_type);

  flatbuffers::FlatBufferBuilder builder;
  auto status = mlir::lite::QuantizeModel(
      *tflite_model, input_type, output_type, inference_tensor_type, {},
      disable_per_channel, fully_quantize, &builder, error_reporter.get(),
      enable_numeric_verify, enable_whole_model_verify,
      /*legacy_float_scale=*/true, denylisted_ops, denylisted_nodes);

  if (status != kTfLiteOk) {
    error_reporter->exception();
    return nullptr;
  }
  return tflite::python_utils::ConvertToPyString(
      reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
      builder.GetSize());
}

PyObject* MlirSparsifyModel(PyObject* data) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStocoPSpythonPStoco_python_apiDTcc mht_6(mht_6_v, 540, "", "./tensorflow/lite/toco/python/toco_python_api.cc", "MlirSparsifyModel");

  using tflite::interpreter_wrapper::PythonErrorReporter;
  char* buf = nullptr;
  Py_ssize_t length;
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);

  if (tflite::python_utils::ConvertFromPyString(data, &buf, &length) == -1) {
    PyErr_Format(PyExc_ValueError, "Failed to convert input PyObject");
    return nullptr;
  }
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(buf, length,
                                               error_reporter.get());
  if (!model) {
    PyErr_Format(PyExc_ValueError, "Invalid model");
    return nullptr;
  }
  auto tflite_model = absl::make_unique<tflite::ModelT>();
  model->GetModel()->UnPackTo(tflite_model.get(), nullptr);

  flatbuffers::FlatBufferBuilder builder;
  auto status =
      mlir::lite::SparsifyModel(*tflite_model, &builder, error_reporter.get());

  if (status != kTfLiteOk) {
    error_reporter->exception();
    return nullptr;
  }
  return tflite::python_utils::ConvertToPyString(
      reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
      builder.GetSize());
}

PyObject* RegisterCustomOpdefs(PyObject* list) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStocoPSpythonPStoco_python_apiDTcc mht_7(mht_7_v, 576, "", "./tensorflow/lite/toco/python/toco_python_api.cc", "RegisterCustomOpdefs");

  if (!PyList_Check(list)) {
    PyErr_SetString(PyExc_TypeError, "Expected list in argument");
    return nullptr;
  }

  int64_t size = PyList_Size(list);
  for (int i = 0; i < size; ++i) {
    // Get character array from Python object.
    char* tf_opdefs;
    Py_ssize_t len;
    if (tflite::python_utils::ConvertFromPyString(PyList_GetItem(list, i),
                                                  &tf_opdefs, &len) == -1) {
      PyErr_Format(PyExc_ValueError,
                   "Failed to convert Python string at index %d of custom op "
                   "defs argument",
                   i);
      return nullptr;
    }

    // Parse op def from character array.
    tensorflow::OpDef opdef;
    if (!tensorflow::protobuf::TextFormat::ParseFromString(tf_opdefs, &opdef)) {
      PyErr_Format(
          PyExc_ValueError,
          "Failed to parse opdefs at index %d of custom op defs argument: %s",
          i, tf_opdefs);
      return nullptr;
    }

    // Register extra opdefs to TensorFlow global op registry.
    tensorflow::OpRegistry::Global()->Register(
        [opdef](
            tensorflow::OpRegistrationData* op_reg_data) -> tensorflow::Status {
          *op_reg_data = tensorflow::OpRegistrationData(opdef);
          return tensorflow::Status::OK();
        });

    // Register the corresponding fake op kernel.
    const char* node_name = opdef.name().c_str();
    const char* op_name = opdef.name().c_str();
    const char* device_name = "CPU";
    static auto fake_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStocoPSpythonPStoco_python_apiDTcc mht_8(mht_8_v, 621, "", "./tensorflow/lite/toco/python/toco_python_api.cc", "lambda");

    };

    TF_KernelBuilder* builder =
        TF_NewKernelBuilder(op_name, device_name, /*create_func=*/nullptr,
                            fake_compute_func, /*delete_func=*/nullptr);

    TF_Status* status = TF_NewStatus();
    TF_RegisterKernelBuilder(node_name, builder, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_DeleteStatus(status);
      PyErr_Format(PyExc_ValueError,
                   "Failed to register fake op kernel at index %d of custom op "
                   "defs argument",
                   i);
      return nullptr;
    }
    TF_DeleteStatus(status);
  }

  Py_RETURN_TRUE;
}

const std::vector<std::string> RetrieveCollectedErrors() {
  mlir::TFL::ErrorCollector* collector =
      mlir::TFL::ErrorCollector::GetErrorCollector();
  std::vector<std::string> collected_errors;
  for (const auto& error_data : collector->CollectedErrors()) {
    collected_errors.push_back(error_data.SerializeAsString());
  }
  collector->Clear();
  return collected_errors;
}

std::string FlatBufferFileToMlir(const std::string& model,
                                 bool input_is_filepath) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("model: \"" + model + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSpythonPStoco_python_apiDTcc mht_9(mht_9_v, 660, "", "./tensorflow/lite/toco/python/toco_python_api.cc", "FlatBufferFileToMlir");

  return ::tensorflow::FlatBufferFileToMlir(model, input_is_filepath);
}

}  // namespace toco
