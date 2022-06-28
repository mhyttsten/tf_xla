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
class MHTracer_DTPStensorflowPScorePSopsPSlookup_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSlookup_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSlookup_opsDTcc() {
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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset_stateful_op_allowlist.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeAndType;
using shape_inference::ShapeHandle;

// --------------------------------------------------------------------------

namespace {
Status TwoElementVectorInputsAndScalarOutputs(InferenceContext* c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSlookup_opsDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/ops/lookup_ops.cc", "TwoElementVectorInputsAndScalarOutputs");

  ShapeHandle handle;
  DimensionHandle unused_handle;
  for (int i = 0; i < c->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &handle));
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_handle));
  }
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->Scalar());
  }
  return Status::OK();
}

Status ScalarAndTwoElementVectorInputsAndScalarOutputs(InferenceContext* c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSlookup_opsDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/ops/lookup_ops.cc", "ScalarAndTwoElementVectorInputsAndScalarOutputs");

  ShapeHandle handle;
  DimensionHandle unused_handle;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
  for (int i = 1; i < c->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &handle));
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_handle));
  }
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->Scalar());
  }
  return Status::OK();
}

Status TwoElementOutput(InferenceContext* c) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPSlookup_opsDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/ops/lookup_ops.cc", "TwoElementOutput");

  c->set_output(0, c->Vector(2));
  return Status::OK();
}

Status ScalarOutput(InferenceContext* c) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSopsPSlookup_opsDTcc mht_3(mht_3_v, 242, "", "./tensorflow/core/ops/lookup_ops.cc", "ScalarOutput");

  c->set_output(0, c->Scalar());
  return Status::OK();
}
}  // namespace

REGISTER_OP("LookupTableFind")
    .Input("table_handle: Ref(string)")
    .Input("keys: Tin")
    .Input("default_value: Tout")
    .Output("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      // Default value must be scalar or vector.
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(2), 1, &unused));
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    });

Status ValidateTableType(InferenceContext* c,
                         const ShapeAndType& key_shape_and_type,
                         const string& key_dtype_attr,
                         const ShapeAndType& value_shape_and_type,
                         const string& value_dtype_attr) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("key_dtype_attr: \"" + key_dtype_attr + "\"");
   mht_4_v.push_back("value_dtype_attr: \"" + value_dtype_attr + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSlookup_opsDTcc mht_4(mht_4_v, 277, "", "./tensorflow/core/ops/lookup_ops.cc", "ValidateTableType");

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

Status ValidateTableResourceHandle(InferenceContext* c, ShapeHandle keys,
                                   const string& key_dtype_attr,
                                   const string& value_dtype_attr,
                                   ShapeAndType* output_shape_and_type) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("key_dtype_attr: \"" + key_dtype_attr + "\"");
   mht_5_v.push_back("value_dtype_attr: \"" + value_dtype_attr + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSlookup_opsDTcc mht_5(mht_5_v, 307, "", "./tensorflow/core/ops/lookup_ops.cc", "ValidateTableResourceHandle");

  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data == nullptr || handle_data->size() != 2) {
    output_shape_and_type->shape = c->UnknownShape();
    output_shape_and_type->dtype = DT_INVALID;
  } else {
    const ShapeAndType& key_shape_and_type = (*handle_data)[0];
    const ShapeAndType& value_shape_and_type = (*handle_data)[1];
    TF_RETURN_IF_ERROR(ValidateTableType(c, key_shape_and_type, key_dtype_attr,
                                         value_shape_and_type,
                                         value_dtype_attr));
    output_shape_and_type->dtype = value_shape_and_type.dtype;
    if (c->RankKnown(key_shape_and_type.shape) && c->RankKnown(keys)) {
      int keys_rank = c->Rank(keys);
      int key_suffix_rank = c->Rank(key_shape_and_type.shape);
      if (keys_rank < key_suffix_rank) {
        return errors::InvalidArgument(
            "Expected keys to have suffix ",
            c->DebugString(key_shape_and_type.shape),
            " but saw shape: ", c->DebugString(keys));
      }
      for (int d = 0; d < key_suffix_rank; d++) {
        // Ensure the suffix of keys match what's in the Table.
        DimensionHandle dim = c->Dim(key_shape_and_type.shape, d);
        TF_RETURN_IF_ERROR(
            c->ReplaceDim(keys, keys_rank - key_suffix_rank + d, dim, &keys));
      }
      std::vector<DimensionHandle> keys_prefix_vec;
      keys_prefix_vec.reserve(keys_rank - key_suffix_rank);
      for (int d = 0; d < keys_rank - key_suffix_rank; ++d) {
        keys_prefix_vec.push_back(c->Dim(keys, d));
      }
      ShapeHandle keys_prefix = c->MakeShape(keys_prefix_vec);
      TF_RETURN_IF_ERROR(c->Concatenate(keys_prefix, value_shape_and_type.shape,
                                        &output_shape_and_type->shape));
    } else {
      output_shape_and_type->shape = c->UnknownShape();
    }
  }
  return Status::OK();
}

REGISTER_OP("LookupTableFindV2")
    .Input("table_handle: resource")
    .Input("keys: Tin")
    .Input("default_value: Tout")
    .Output("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      ShapeAndType value_shape_and_type;
      TF_RETURN_IF_ERROR(ValidateTableResourceHandle(
          c,
          /*keys=*/c->input(1),
          /*key_dtype_attr=*/"Tin",
          /*value_dtype_attr=*/"Tout", &value_shape_and_type));
      c->set_output(0, value_shape_and_type.shape);

      return Status::OK();
    });
ALLOW_STATEFUL_OP_FOR_DATASET_FUNCTIONS("LookupTableFindV2");
// TODO(b/72710477): Update this.

REGISTER_OP("LookupTableInsert")
    .Input("table_handle: Ref(string)")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      // TODO(ebrevdo): Validate keys and values shape.
      return Status::OK();
    });

REGISTER_OP("LookupTableInsertV2")
    .Input("table_handle: resource")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      // TODO: Validate keys and values shape.
      return Status::OK();
    });

REGISTER_OP("LookupTableRemoveV2")
    .Input("table_handle: resource")
    .Input("keys: Tin")
    .Attr("Tin: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &handle));

      // TODO(turboale): Validate keys shape.
      return Status::OK();
    });

REGISTER_OP("LookupTableSize")
    .Input("table_handle: Ref(string)")
    .Output("size: int64")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs);
ALLOW_STATEFUL_OP_FOR_DATASET_FUNCTIONS("LookupTableSize");

REGISTER_OP("LookupTableSizeV2")
    .Input("table_handle: resource")
    .Output("size: int64")
    .SetShapeFn(ScalarAndTwoElementVectorInputsAndScalarOutputs);
ALLOW_STATEFUL_OP_FOR_DATASET_FUNCTIONS("LookupTableSizeV2");

REGISTER_OP("LookupTableExport")
    .Input("table_handle: Ref(string)")
    .Output("keys: Tkeys")
    .Output("values: Tvalues")
    .Attr("Tkeys: type")
    .Attr("Tvalues: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      ShapeHandle values = c->UnknownShape();
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(values, 1, &values));
      ShapeHandle keys = c->Vector(c->Dim(values, 0));
      c->set_output(0, keys);
      c->set_output(1, values);
      return Status::OK();
    });

REGISTER_OP("LookupTableExportV2")
    .Input("table_handle: resource")
    .Output("keys: Tkeys")
    .Output("values: Tvalues")
    .Attr("Tkeys: type")
    .Attr("Tvalues: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr && handle_data->size() == 2) {
        const ShapeAndType& key_shape_and_type = (*handle_data)[0];
        const ShapeAndType& value_shape_and_type = (*handle_data)[1];
        TF_RETURN_IF_ERROR(ValidateTableType(c, key_shape_and_type,
                                             /*key_dtype_attr*/ "Tkeys",
                                             value_shape_and_type,
                                             /*value_dtype_attr*/ "Tvalues"));
      }
      // Different lookup tables have different output shapes.
      c->set_output(0, c->UnknownShape());
      c->set_output(1, c->UnknownShape());
      return Status::OK();
    });

REGISTER_OP("LookupTableImport")
    .Input("table_handle: Ref(string)")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      // TODO(ebrevdo): Validate keys and values shape.
      return Status::OK();
    });

REGISTER_OP("LookupTableImportV2")
    .Input("table_handle: resource")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      ShapeHandle keys;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &keys));
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(keys, 0), c->Dim(c->input(2), 0), &unused));
      return Status::OK();
    });

Status MutableHashTableShape(InferenceContext* c, const ShapeHandle& key,
                             const ShapeHandle& value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSopsPSlookup_opsDTcc mht_6(mht_6_v, 510, "", "./tensorflow/core/ops/lookup_ops.cc", "MutableHashTableShape");

  c->set_output(0, c->Scalar());

  ShapeHandle key_s;
  TF_RETURN_IF_ERROR(c->WithRankAtMost(key, 1, &key_s));

  DataType key_t;
  TF_RETURN_IF_ERROR(c->GetAttr("key_dtype", &key_t));

  DataType value_t;
  TF_RETURN_IF_ERROR(c->GetAttr("value_dtype", &value_t));

  // ShapeAndType vector for {key, value}.
  c->set_output_handle_shapes_and_types(
      0, std::vector<ShapeAndType>{{key_s, key_t}, {value, value_t}});

  return Status::OK();
}

Status MutableHashTableShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSopsPSlookup_opsDTcc mht_7(mht_7_v, 532, "", "./tensorflow/core/ops/lookup_ops.cc", "MutableHashTableShapeFn");

  return MutableHashTableShape(c, /*key=*/c->Scalar(),
                               /*value=*/c->Scalar());
}

Status MutableHashTableOfTensorsShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSopsPSlookup_opsDTcc mht_8(mht_8_v, 540, "", "./tensorflow/core/ops/lookup_ops.cc", "MutableHashTableOfTensorsShapeFn");

  PartialTensorShape value_p;
  TF_RETURN_IF_ERROR(c->GetAttr("value_shape", &value_p));
  ShapeHandle value_s;
  TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(value_p, &value_s));
  return MutableHashTableShape(c, /*key=*/c->Scalar(), /*value=*/value_s);
}

Status MutableDenseHashTableShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSopsPSlookup_opsDTcc mht_9(mht_9_v, 551, "", "./tensorflow/core/ops/lookup_ops.cc", "MutableDenseHashTableShapeFn");

  PartialTensorShape value_p;
  TF_RETURN_IF_ERROR(c->GetAttr("value_shape", &value_p));
  ShapeHandle value_s;
  TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(value_p, &value_s));
  return MutableHashTableShape(c, /*key=*/c->input(0), /*value=*/value_s);
}

REGISTER_OP("HashTable")
    .Output("table_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("HashTableV2")
    .Output("table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(ScalarOutput);

REGISTER_OP("AnonymousHashTable")
    .Output("table_handle: resource")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(ScalarOutput);

REGISTER_OP("MutableHashTable")
    .Output("table_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("MutableHashTableV2")
    .Output("table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(MutableHashTableShapeFn);

REGISTER_OP("AnonymousMutableHashTable")
    .Output("table_handle: resource")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(MutableHashTableShapeFn);

REGISTER_OP("MutableHashTableOfTensors")
    .Output("table_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("value_shape: shape = {}")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("MutableHashTableOfTensorsV2")
    .Output("table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("value_shape: shape = {}")
    .SetIsStateful()
    .SetShapeFn(MutableHashTableOfTensorsShapeFn);

REGISTER_OP("AnonymousMutableHashTableOfTensors")
    .Output("table_handle: resource")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("value_shape: shape = {}")
    .SetIsStateful()
    .SetShapeFn(MutableHashTableOfTensorsShapeFn);

REGISTER_OP("MutableDenseHashTable")
    .Input("empty_key: key_dtype")
    .Output("table_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("value_shape: shape = {}")
    .Attr("initial_num_buckets: int = 131072")  // 2^17
    .Attr("max_load_factor: float = 0.8")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("MutableDenseHashTableV2")
    .Input("empty_key: key_dtype")
    .Input("deleted_key: key_dtype")
    .Output("table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("value_shape: shape = {}")
    .Attr("initial_num_buckets: int = 131072")  // 2^17
    .Attr("max_load_factor: float = 0.8")
    .SetIsStateful()
    .SetShapeFn(MutableDenseHashTableShapeFn);

REGISTER_OP("AnonymousMutableDenseHashTable")
    .Input("empty_key: key_dtype")
    .Input("deleted_key: key_dtype")
    .Output("table_handle: resource")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("value_shape: shape = {}")
    .Attr("initial_num_buckets: int = 131072")  // 2^17
    .Attr("max_load_factor: float = 0.8")
    .SetIsStateful()
    .SetShapeFn(MutableDenseHashTableShapeFn);

REGISTER_OP("InitializeTable")
    .Input("table_handle: Ref(string)")
    .Input("keys: Tkey")
    .Input("values: Tval")
    .Attr("Tkey: type")
    .Attr("Tval: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      ShapeHandle keys;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &keys));
      TF_RETURN_IF_ERROR(c->Merge(keys, c->input(2), &keys));
      return Status::OK();
    });

REGISTER_OP("InitializeTableV2")
    .Input("table_handle: resource")
    .Input("keys: Tkey")
    .Input("values: Tval")
    .Attr("Tkey: type")
    .Attr("Tval: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      ShapeHandle keys;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &keys));
      TF_RETURN_IF_ERROR(c->Merge(keys, c->input(2), &keys));
      return Status::OK();
    });

REGISTER_OP("InitializeTableFromTextFile")
    .Input("table_handle: Ref(string)")
    .Input("filename: string")
    .Attr("key_index: int >= -2")
    .Attr("value_index: int >= -2")
    .Attr("vocab_size: int >= -1 = -1")
    .Attr("delimiter: string = '\t'")
    .Attr("offset: int = 0")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &handle));
      return Status::OK();
    });

REGISTER_OP("InitializeTableFromTextFileV2")
    .Input("table_handle: resource")
    .Input("filename: string")
    .Attr("key_index: int >= -2")
    .Attr("value_index: int >= -2")
    .Attr("vocab_size: int >= -1 = -1")
    .Attr("delimiter: string = '\t'")
    .Attr("offset: int = 0")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &handle));
      return Status::OK();
    });

}  // namespace tensorflow
