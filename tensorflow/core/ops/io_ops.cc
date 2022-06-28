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
class MHTracer_DTPStensorflowPScorePSopsPSio_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSio_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSio_opsDTcc() {
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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

Status ScalarInputsAndOutputs(InferenceContext* c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSio_opsDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/ops/io_ops.cc", "ScalarInputsAndOutputs");

  ShapeHandle unused;
  for (int i = 0; i < c->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &unused));
  }
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->Scalar());
  }
  return Status::OK();
}

Status TwoElementVectorAndScalarOutputs(InferenceContext* c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSio_opsDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/ops/io_ops.cc", "TwoElementVectorAndScalarOutputs");

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

Status TwoElementOutput(InferenceContext* c) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPSio_opsDTcc mht_2(mht_2_v, 228, "", "./tensorflow/core/ops/io_ops.cc", "TwoElementOutput");

  c->set_output(0, c->Vector(2));
  return Status::OK();
}

}  // namespace

REGISTER_OP("SaveV2")
    .Input("prefix: string")
    .Input("tensor_names: string")
    .Input("shape_and_slices: string")
    .Input("tensors: dtypes")
    .Attr("dtypes: list(type)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      ShapeHandle s;
      DimensionHandle unused_dim;

      // Validate prefix.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

      // Validate tensor_names and shapes_and_slices.
      for (int i = 1; i <= 2; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &s));
        TF_RETURN_IF_ERROR(
            c->WithValue(c->Dim(s, 0), c->num_inputs() - 3, &unused_dim));
      }
      // TODO(mrry): Attempt to parse the shapes_and_slices values and use
      // them to constrain the shape of the remaining inputs.
      return Status::OK();
    });

REGISTER_OP("RestoreV2")
    .Input("prefix: string")
    .Input("tensor_names: string")
    .Input("shape_and_slices: string")
    .Output("tensors: dtypes")
    .Attr("dtypes: list(type)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle shape0, shape1, shape2;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &shape0));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &shape1));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &shape2));
      TF_RETURN_IF_ERROR(c->Merge(shape1, shape2, &shape0));

      // Attempt to infer output shapes from its shape_and_slice input.
      const Tensor* shape_and_slices_tensor = c->input_tensor(2);
      if (shape_and_slices_tensor) {
        if (shape_and_slices_tensor->dtype() != DT_STRING) {
          return errors::InvalidArgument(
              "Expected an input tensor of type string.");
        }

        const auto& shape_and_slices_flat =
            shape_and_slices_tensor->flat<tstring>();
        if (shape_and_slices_flat.size() != c->num_outputs()) {
          return errors::InvalidArgument(
              "The number of shape_and_slice doesn't match tensor outputs.");
        }
        for (int i = 0; i < shape_and_slices_flat.size(); ++i) {
          const string& shape_and_slice = shape_and_slices_flat(i);
          if (shape_and_slice.empty()) {
            c->set_output(i, c->UnknownShape());
            continue;
          }
          TensorShape parsed_full_shape;
          TensorSlice parsed_slice;
          TensorShape parsed_slice_shape;
          TF_RETURN_IF_ERROR(checkpoint::ParseShapeAndSlice(
              shape_and_slice, &parsed_full_shape, &parsed_slice,
              &parsed_slice_shape));
          ShapeHandle shape_handle;
          TF_RETURN_IF_ERROR(
              c->MakeShapeFromTensorShape(parsed_slice_shape, &shape_handle));
          c->set_output(i, shape_handle);
        }
        return Status::OK();
      } else {
        return UnknownShape(c);
      }
    });

REGISTER_OP("MergeV2Checkpoints")
    .Input("checkpoint_prefixes: string")
    .Input("destination_prefix: string")
    .Attr("delete_old_dirs: bool = true")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    });

REGISTER_OP("Save")
    .Input("filename: string")
    .Input("tensor_names: string")
    .Input("data: T")
    .Attr("T: list(type)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      ShapeHandle s;
      DimensionHandle unused_dim;

      // Validate filename.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

      // Validate tensor_names.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &s));
      TF_RETURN_IF_ERROR(
          c->WithValue(c->Dim(s, 0), c->num_inputs() - 2, &unused_dim));

      return Status::OK();
    });

REGISTER_OP("SaveSlices")
    .Input("filename: string")
    .Input("tensor_names: string")
    .Input("shapes_and_slices: string")
    .Input("data: T")
    .Attr("T: list(type)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      ShapeHandle s;
      DimensionHandle unused_dim;

      // Validate filename.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

      // Validate tensor_names and unused_shapes_and_slices.
      for (int i = 1; i <= 2; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &s));
        TF_RETURN_IF_ERROR(
            c->WithValue(c->Dim(s, 0), c->num_inputs() - 3, &unused_dim));
      }
      // TODO(mrry): Attempt to parse the shapes_and_slices values and use
      // them to constrain the shape of the remaining inputs.
      return Status::OK();
    });

REGISTER_OP("Restore")
    .Input("file_pattern: string")
    .Input("tensor_name: string")
    .Output("tensor: dt")
    .Attr("dt: type")
    .Attr("preferred_shard: int = -1")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    });

REGISTER_OP("RestoreSlice")
    .Input("file_pattern: string")
    .Input("tensor_name: string")
    .Input("shape_and_slice: string")
    .Output("tensor: dt")
    .Attr("dt: type")
    .Attr("preferred_shard: int = -1")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));

      // Attempt to infer output shapes from its shape_and_slice input.
      const Tensor* shape_and_slices_tensor = c->input_tensor(2);
      if (shape_and_slices_tensor) {
        const auto& shape_and_slice =
            shape_and_slices_tensor->flat<tstring>()(0);
        if (shape_and_slice.empty()) {
          c->set_output(0, c->UnknownShape());
        } else {
          TensorShape parsed_full_shape;
          TensorSlice parsed_slice;
          TensorShape parsed_slice_shape;
          TF_RETURN_IF_ERROR(checkpoint::ParseShapeAndSlice(
              shape_and_slice, &parsed_full_shape, &parsed_slice,
              &parsed_slice_shape));
          ShapeHandle shape_handle;
          TF_RETURN_IF_ERROR(
              c->MakeShapeFromTensorShape(parsed_slice_shape, &shape_handle));
          c->set_output(0, shape_handle);
        }
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return Status::OK();
    });

REGISTER_OP("ShardedFilename")
    .Input("basename: string")
    .Input("shard: int32")
    .Input("num_shards: int32")
    .Output("filename: string")
    .SetShapeFn(ScalarInputsAndOutputs);

REGISTER_OP("ShardedFilespec")
    .Input("basename: string")
    .Input("num_shards: int32")
    .Output("filename: string")
    .SetShapeFn(ScalarInputsAndOutputs);

// Reader source ops ----------------------------------------------------------

REGISTER_OP("WholeFileReader")
    .Output("reader_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("WholeFileReaderV2")
    .Output("reader_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("TextLineReader")
    .Output("reader_handle: Ref(string)")
    .Attr("skip_header_lines: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Deprecated(26, "Use TextLineReaderV2");

REGISTER_OP("TextLineReaderV2")
    .Output("reader_handle: resource")
    .Attr("skip_header_lines: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("FixedLengthRecordReader")
    .Output("reader_handle: Ref(string)")
    .Attr("header_bytes: int = 0")
    .Attr("record_bytes: int")
    .Attr("footer_bytes: int = 0")
    .Attr("hop_bytes: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Deprecated(26, "Use FixedLengthRecordReaderV2");

REGISTER_OP("FixedLengthRecordReaderV2")
    .Output("reader_handle: resource")
    .Attr("header_bytes: int = 0")
    .Attr("record_bytes: int")
    .Attr("footer_bytes: int = 0")
    .Attr("hop_bytes: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("encoding: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("TFRecordReader")
    .Output("reader_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("compression_type: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Deprecated(26, "Use TFRecordReaderV2");

REGISTER_OP("TFRecordReaderV2")
    .Output("reader_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("compression_type: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("LMDBReader")
    .Output("reader_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("IdentityReader")
    .Output("reader_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Deprecated(26, "Use IdentityReaderV2");

REGISTER_OP("IdentityReaderV2")
    .Output("reader_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

// Ops that operate on Readers ------------------------------------------------

REGISTER_OP("ReaderRead")
    .Input("reader_handle: Ref(string)")
    .Input("queue_handle: Ref(string)")
    .Output("key: string")
    .Output("value: string")
    .SetShapeFn(TwoElementVectorAndScalarOutputs);

REGISTER_OP("ReaderReadV2")
    .Input("reader_handle: resource")
    .Input("queue_handle: resource")
    .Output("key: string")
    .Output("value: string")
    .SetShapeFn(ScalarInputsAndOutputs);

REGISTER_OP("ReaderReadUpTo")
    .Input("reader_handle: Ref(string)")
    .Input("queue_handle: Ref(string)")
    .Input("num_records: int64")
    .Output("keys: string")
    .Output("values: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      ShapeHandle out = c->Vector(InferenceContext::kUnknownDim);
      c->set_output(0, out);
      c->set_output(1, out);
      return Status::OK();
    });

REGISTER_OP("ReaderReadUpToV2")
    .Input("reader_handle: resource")
    .Input("queue_handle: resource")
    .Input("num_records: int64")
    .Output("keys: string")
    .Output("values: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      ShapeHandle out = c->Vector(InferenceContext::kUnknownDim);
      c->set_output(0, out);
      c->set_output(1, out);
      return Status::OK();
    });

REGISTER_OP("ReaderNumRecordsProduced")
    .Input("reader_handle: Ref(string)")
    .Output("records_produced: int64")
    .SetShapeFn(TwoElementVectorAndScalarOutputs);

REGISTER_OP("ReaderNumRecordsProducedV2")
    .Input("reader_handle: resource")
    .Output("records_produced: int64")
    .SetShapeFn(ScalarInputsAndOutputs);

REGISTER_OP("ReaderNumWorkUnitsCompleted")
    .Input("reader_handle: Ref(string)")
    .Output("units_completed: int64")
    .SetShapeFn(TwoElementVectorAndScalarOutputs);

REGISTER_OP("ReaderNumWorkUnitsCompletedV2")
    .Input("reader_handle: resource")
    .Output("units_completed: int64")
    .SetShapeFn(ScalarInputsAndOutputs);

REGISTER_OP("ReaderSerializeState")
    .Input("reader_handle: Ref(string)")
    .Output("state: string")
    .SetShapeFn(TwoElementVectorAndScalarOutputs);

REGISTER_OP("ReaderSerializeStateV2")
    .Input("reader_handle: resource")
    .Output("state: string")
    .SetShapeFn(ScalarInputsAndOutputs);

REGISTER_OP("ReaderRestoreState")
    .Input("reader_handle: Ref(string)")
    .Input("state: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      DimensionHandle unused_handle;
      TF_RETURN_IF_ERROR(
          c->WithValue(c->Dim(c->input(0), 0), 2, &unused_handle));

      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    });

REGISTER_OP("ReaderRestoreStateV2")
    .Input("reader_handle: resource")
    .Input("state: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    });

REGISTER_OP("ReaderReset")
    .Input("reader_handle: Ref(string)")
    .SetShapeFn(TwoElementVectorAndScalarOutputs);

REGISTER_OP("ReaderResetV2")
    .Input("reader_handle: resource")
    .SetShapeFn(ScalarInputsAndOutputs);

// Other input Ops ----------------------------------------------------------

REGISTER_OP("ReadFile")
    .Input("filename: string")
    .Output("contents: string")
    .SetShapeFn(ScalarInputsAndOutputs);

REGISTER_OP("WriteFile")
    .Input("filename: string")
    .Input("contents: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    });

REGISTER_OP("MatchingFiles")
    .Input("pattern: string")
    .Output("filenames: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &unused));
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    });

}  // namespace tensorflow
