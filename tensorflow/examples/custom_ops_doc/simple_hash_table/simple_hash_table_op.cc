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
class MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_opDTcc {
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
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_opDTcc() {
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

#include <string>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

// Please use the appropriate namespace for your project
namespace tensorflow {
namespace custom_op_examples {

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeAndType;
using ::tensorflow::shape_inference::ShapeHandle;

Status ScalarOutput(InferenceContext* c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_opDTcc mht_0(mht_0_v, 199, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_op.cc", "ScalarOutput");

  c->set_output(0, c->Scalar());
  return Status::OK();
}

Status TwoScalarInputs(InferenceContext* c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_opDTcc mht_1(mht_1_v, 207, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_op.cc", "TwoScalarInputs");

  ShapeHandle handle;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &handle));
  return Status::OK();
}

Status TwoScalarInputsScalarOutput(InferenceContext* c) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_opDTcc mht_2(mht_2_v, 217, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_op.cc", "TwoScalarInputsScalarOutput");

  ShapeHandle handle;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &handle));
  return ScalarOutput(c);
}

Status ThreeScalarInputs(InferenceContext* c) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_opDTcc mht_3(mht_3_v, 227, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_op.cc", "ThreeScalarInputs");

  ShapeHandle handle;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &handle));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &handle));
  return Status::OK();
}

Status ThreeScalarInputsScalarOutput(InferenceContext* c) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_opDTcc mht_4(mht_4_v, 238, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_op.cc", "ThreeScalarInputsScalarOutput");

  ShapeHandle handle;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &handle));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &handle));
  return ScalarOutput(c);
}

Status ValidateTableType(InferenceContext* c,
                         const ShapeAndType& key_shape_and_type,
                         const string& key_dtype_attr,
                         const ShapeAndType& value_shape_and_type,
                         const string& value_dtype_attr) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("key_dtype_attr: \"" + key_dtype_attr + "\"");
   mht_5_v.push_back("value_dtype_attr: \"" + value_dtype_attr + "\"");
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_opDTcc mht_5(mht_5_v, 255, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_op.cc", "ValidateTableType");

  DataType key_dtype;
  TF_RETURN_IF_ERROR(c->GetAttr(key_dtype_attr, &key_dtype));
  if (key_shape_and_type.dtype != key_dtype) {
    return errors::InvalidArgument(
        "Trying to read value with wrong dtype. "
        "Expected ",
        DataTypeString(key_shape_and_type.dtype), " got ",
        DataTypeString(key_dtype));
  }
  DataType value_dtype;
  TF_RETURN_IF_ERROR(c->GetAttr(value_dtype_attr, &value_dtype));
  if (value_shape_and_type.dtype != value_dtype) {
    return errors::InvalidArgument(
        "Trying to read value with wrong dtype. "
        "Expected ",
        DataTypeString(value_shape_and_type.dtype), " got ",
        DataTypeString(value_dtype));
  }
  return Status::OK();
}

Status ExportShapeFunction(InferenceContext* c) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_opDTcc mht_6(mht_6_v, 280, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_op.cc", "ExportShapeFunction");

  ShapeHandle handle;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data != nullptr && handle_data->size() == 2) {
    const ShapeAndType& key_shape_and_type = (*handle_data)[0];
    const ShapeAndType& value_shape_and_type = (*handle_data)[1];
    TF_RETURN_IF_ERROR(ValidateTableType(c, key_shape_and_type,
                                         /*key_dtype_attr*/ "key_dtype",
                                         value_shape_and_type,
                                         /*value_dtype_attr*/ "value_dtype"));
  }
  // Different lookup tables have different output shapes.
  c->set_output(0, c->UnknownShape());
  c->set_output(1, c->UnknownShape());
  return Status::OK();
}

Status ImportShapeFunction(InferenceContext* c) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsimple_hash_tablePSsimple_hash_table_opDTcc mht_7(mht_7_v, 301, "", "./tensorflow/examples/custom_ops_doc/simple_hash_table/simple_hash_table_op.cc", "ImportShapeFunction");

  ShapeHandle handle;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

  ShapeHandle keys;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &keys));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(
      c->Merge(c->Dim(keys, 0), c->Dim(c->input(2), 0), &unused));
  return Status::OK();
}

// Note that if an op has any Input or Output of type "resource", it
// is automatically marked as stateful so there is no need to explicitly
// use "SetIsStateful()".
// (See FinalizeInputOrOutput in core/framework/op_def_builder.cc.)

REGISTER_OP("Examples>SimpleHashTableCreate")
    .Output("output: resource")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ScalarOutput);

REGISTER_OP("Examples>SimpleHashTableFind")
    .Input("resource_handle: resource")
    .Input("key: key_dtype")
    .Input("default_value: value_dtype")
    .Output("value: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ThreeScalarInputsScalarOutput);

REGISTER_OP("Examples>SimpleHashTableInsert")
    .Input("resource_handle: resource")
    .Input("key: key_dtype")
    .Input("value: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ThreeScalarInputs);

REGISTER_OP("Examples>SimpleHashTableRemove")
    .Input("resource_handle: resource")
    .Input("key: key_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(TwoScalarInputs);

REGISTER_OP("Examples>SimpleHashTableExport")
    .Input("table_handle: resource")
    .Output("keys: key_dtype")
    .Output("values: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ExportShapeFunction);

REGISTER_OP("Examples>SimpleHashTableImport")
    .Input("table_handle: resource")
    .Input("keys: key_dtype")
    .Input("values: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ImportShapeFunction);

}  // namespace custom_op_examples
}  // namespace tensorflow
