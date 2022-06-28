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
class MHTracer_DTPStensorflowPScorePSopsPSparsing_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSparsing_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSparsing_ops_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(ParsingOpsTest, DecodeRaw_ShapeFn) {
  ShapeInferenceTestOp op("DecodeRaw");

  // Output is input + an unknown dim.
  INFER_OK(op, "?", "?");
  INFER_OK(op, "[?,?,?]", "[d0_0,d0_1,d0_2,?]");
}

TEST(ParsingOpsTest, DecodeCSV_ShapeFn) {
  ShapeInferenceTestOp op("DecodeCSV");
  auto set_n_outputs = [&op](int n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSparsing_ops_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/ops/parsing_ops_test.cc", "lambda");

    std::vector<NodeDefBuilder::NodeOut> src_list;
    std::vector<DataType> out_types;
    for (int i = 0; i < n; ++i) {
      src_list.emplace_back("b", 0, DT_FLOAT);
      out_types.push_back(DT_FLOAT);
    }
    TF_ASSERT_OK(NodeDefBuilder("test", "DecodeCSV")
                     .Input("a", 0, DT_STRING)
                     .Input(src_list)
                     .Attr("OUT_TYPE", out_types)
                     .Finalize(&op.node_def));
  };

  // Output is always n copies of input 0.
  set_n_outputs(2);
  INFER_OK(op, "?;?;?", "in0;in0");
  INFER_OK(op, "[1,2,?,4];?;?", "in0;in0");
  INFER_OK(op, "[1,2,?,4];[?];[?]", "in0;in0");

  // Scalar defaults are ok
  INFER_OK(op, "?;?;[]", "in0;in0");

  // Check errors in the record_defaults inputs.
  INFER_ERROR("must be at most rank 1 but is rank 2", op, "?;?;[1,2]");
  INFER_ERROR("must be at most rank 1 but is rank 2", op, "?;[3,4];?");
  INFER_ERROR("Shape of a default must be", op, "?;?;[2]");
  INFER_ERROR("Shape of a default must be", op, "?;[2];?");
}

static std::vector<PartialTensorShape> MakeDenseShapes(int size,
                                                       bool add_extra_shape,
                                                       int unknown_outer_dims) {
  std::vector<PartialTensorShape> shapes(size);
  for (int i = 0; i < size; ++i) {
    // Make shapes be the sequence [?,1]; [?,1,2], [?,1,2,3]...
    // where the number of prefixed ? depends on unknown_outer_dims.
    if (i == 0) {
      shapes[i].Clear();
      for (int d = 0; d < unknown_outer_dims; ++d) {
        shapes[i].AddDim(-1);
      }
    } else {
      shapes[i] = shapes[i - 1];
    }
    shapes[i].AddDim(i + 1);
  }
  if (add_extra_shape) shapes.push_back(PartialTensorShape({}));
  return shapes;
}

TEST(ParsingOpsTest, ParseExample_ShapeFn) {
  ShapeInferenceTestOp op("ParseExample");
  auto set_outputs = [&op](int num_sparse, int num_dense,
                           bool add_extra_shape = false,
                           int unknown_outer_dims = 0) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSparsing_ops_testDTcc mht_1(mht_1_v, 262, "", "./tensorflow/core/ops/parsing_ops_test.cc", "lambda");

    using NodeOutList = std::vector<NodeDefBuilder::NodeOut>;
    using DataTypeList = std::vector<DataType>;
    NodeDefBuilder::NodeOut string_in{"a", 0, DT_STRING};

    TF_ASSERT_OK(
        NodeDefBuilder("test", "ParseExample")
            .Input("serialized", 0, DT_STRING)
            .Input("names", 0, DT_STRING)
            .Input(NodeOutList(num_sparse, string_in))
            .Input(NodeOutList(num_dense, string_in))
            .Input(NodeOutList(num_dense, string_in))
            .Attr("sparse_types", DataTypeList(num_sparse, DT_FLOAT))
            // Tdense is inferred from  dense_defaults.
            .Attr("dense_shapes", MakeDenseShapes(num_dense, add_extra_shape,
                                                  unknown_outer_dims))
            .Finalize(&op.node_def));
  };

  // Verify inputs 'serialized' and 'names'.
  set_outputs(0 /* num_sparse */, 0 /* num_dense */);
  INFER_OK(op, "?;?", "");
  INFER_OK(op, "[10];[20]", "");
  INFER_ERROR("must be rank 1", op, "[1,2];?");
  INFER_ERROR("must be rank 1", op, "?;[2,3]");

  // Verify the sparse and dense outputs.
  set_outputs(2 /* num_sparse */, 3 /* num_dense */);
  INFER_OK(op, "?;?;?;?;?;?;?;?;?;?",
           ("[?,2];[?,2];[?];[?];[2];[2];"  // sparse outputs
            "[?,1];[?,1,2];[?,1,2,3]"));    // dense outputs
  INFER_OK(op, "[10];?;?;?;?;?;?;?;?;?",
           ("[?,2];[?,2];[?];[?];[2];[2];"         // sparse outputs
            "[d0_0,1];[d0_0,1,2];[d0_0,1,2,3]"));  // dense outputs

  // Confirm an error from ParseExampleAttrs.Init().
  set_outputs(2, 3, true /* add_extra_shape */);
  INFER_ERROR("len(dense_keys) != len(dense_shapes)", op,
              "?;?;?;?;?;?;?;?;?;?");

  // Allow variable strides
  set_outputs(2, 3, false /* add_extra_shape */, 1 /* unknown_outer_dims */);
  INFER_OK(op, "?;?;?;?;?;?;?;?;?;?",
           ("[?,2];[?,2];[?];[?];[2];[2];"      // sparse outputs
            "[?,?,1];[?,?,1,2];[?,?,1,2,3]"));  // dense outputs
  INFER_OK(op, "[?];?;?;?;?;?;?;?;?;?",
           ("[?,2];[?,2];[?];[?];[2];[2];"               // sparse outputs
            "[d0_0,?,1];[d0_0,?,1,2];[d0_0,?,1,2,3]"));  // dense outputs
  INFER_OK(op, "[10];?;?;?;?;?;?;?;?;?",
           ("[?,2];[?,2];[?];[?];[2];[2];"               // sparse outputs
            "[d0_0,?,1];[d0_0,?,1,2];[d0_0,?,1,2,3]"));  // dense outputs

  set_outputs(2, 3, true /* add_extra_shape */, 1 /* unknown_outer_dims */);
  INFER_ERROR("len(dense_keys) != len(dense_shapes)", op,
              "?;?;?;?;?;?;?;?;?;?");

  // Variable inner dimensions are not supported
  set_outputs(2, 3, false /* add_extra_shape */, 2 /* unknown_outer_dims */);
  INFER_ERROR("shapes[0] has unknown rank or unknown inner dimensions", op,
              "?;?;?;?;?;?;?;?;?;?");
}

TEST(ParsingOpsTest, ParseSequenceExample_ShapeFn) {
  ShapeInferenceTestOp op("ParseSequenceExample");
  auto set_outputs = [&op](int num_context_sparse, int num_context_dense,
                           int num_feature_list_sparse,
                           int num_feature_list_dense,
                           bool add_extra_shape = false) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPSparsing_ops_testDTcc mht_2(mht_2_v, 332, "", "./tensorflow/core/ops/parsing_ops_test.cc", "lambda");

    using NodeOutList = std::vector<NodeDefBuilder::NodeOut>;
    using DataTypeList = std::vector<DataType>;
    string string_in("test");
    NodeDefBuilder::NodeOut node_in{"a", 0, DT_STRING};
    TF_ASSERT_OK(
        NodeDefBuilder("test", "ParseSequenceExample")
            .Input("serialized", 0, DT_STRING)
            .Input("debug_name", 0, DT_STRING)
            .Input(NodeOutList(num_context_dense, node_in))
            .Attr("Ncontext_sparse", num_context_sparse)
            .Attr("Ncontext_dense", num_context_dense)
            .Attr("Nfeature_list_sparse", num_feature_list_sparse)
            .Attr("Nfeature_list_dense", num_feature_list_dense)
            .Attr("feature_list_dense_missing_assumed_empty",
                  std::vector<string>(num_feature_list_dense, string_in))
            .Attr("context_sparse_keys",
                  std::vector<string>(num_context_sparse, string_in))
            .Attr("context_dense_keys",
                  std::vector<string>(num_context_dense, string_in))
            .Attr("feature_list_sparse_keys",
                  std::vector<string>(num_feature_list_sparse, string_in))
            .Attr("feature_list_dense_keys",
                  std::vector<string>(num_feature_list_dense, string_in))
            .Attr("context_sparse_types",
                  DataTypeList(num_context_sparse, DT_FLOAT))
            .Attr("context_dense_types",
                  DataTypeList(num_context_dense, DT_FLOAT))
            .Attr("context_dense_shapes",
                  MakeDenseShapes(num_context_dense, add_extra_shape, 0))
            .Attr("feature_list_sparse_types",
                  DataTypeList(num_feature_list_sparse, DT_FLOAT))
            .Attr("feature_list_dense_types",
                  DataTypeList(num_feature_list_dense, DT_FLOAT))
            .Attr("feature_list_dense_shapes",
                  MakeDenseShapes(num_feature_list_dense, add_extra_shape, 0))
            .Finalize(&op.node_def));
  };

  // Verify inputs 'serialized' and 'debug_name'.
  set_outputs(0, 0, 0, 0);
  INFER_OK(op, "[?];[?]", "");
  INFER_OK(op, "[8];[8]", "");
  INFER_ERROR("must be rank 1", op, "[];[?]");
  INFER_ERROR("must be rank 1", op, "[?];[]");

  // context inputs with no feature_list inputs.
  set_outputs(2 /* num_context_sparse */, 3 /* num_context_dense */, 0, 0);
  INFER_OK(op, "[?];[?];?;?;?",
           ("[?,2];[?,2];[?];[?];[2];[2];"         // context sparse
            "[d0_0,1];[d0_0,1,2];[d0_0,1,2,3]"));  // context dense

  // feature_list inputs with no context inputs.
  set_outputs(0, 0, 2 /* num_feature_list_sparse */,
              3 /* num_feature_list_dense */);
  INFER_OK(op, "[?];[?]",
           ("[?,3];[?,3];[?];[?];[3];[3];"             // feature_list sparse
            "[d0_0,?,1];[d0_0,?,1,2];[d0_0,?,1,2,3];"  // feature_list dense
            "in0;in0;in0"));                           // feature_list length

  // Combine previous two test cases.
  set_outputs(2, 3, 2, 3);
  INFER_OK(op, "[7];[7];?;?;?",
           ("[?,2];[?,2];[?];[?];[2];[2];"             // context sparse
            "[d0_0,1];[d0_0,1,2];[d0_0,1,2,3];"        // context dense
            "[?,3];[?,3];[?];[?];[3];[3];"             // feature_list sparse
            "[d0_0,?,1];[d0_0,?,1,2];[d0_0,?,1,2,3];"  // feature_list dense
            "in0;in0;in0"));                           // feature_list length

  // Confirm an error from ParseSequenceExampleAttrs.Init().
  set_outputs(1, 1, 1, 1, true /* add_extra_shape */);
  INFER_ERROR(
      "num_context_dense (1) must match the size of "
      "context_dense_types (1) and context_dense_shapes (2)",
      op, "[?];[?];?");
}

TEST(ParsingOpsTest, ParseSingleSequenceExample_ShapeFn) {
  ShapeInferenceTestOp op("ParseSingleSequenceExample");
  auto set_outputs = [&op](int num_context_sparse, int num_context_dense,
                           int num_feature_list_sparse,
                           int num_feature_list_dense,
                           bool add_extra_shape = false) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSopsPSparsing_ops_testDTcc mht_3(mht_3_v, 417, "", "./tensorflow/core/ops/parsing_ops_test.cc", "lambda");

    using NodeOutList = std::vector<NodeDefBuilder::NodeOut>;
    using DataTypeList = std::vector<DataType>;
    NodeDefBuilder::NodeOut string_in{"a", 0, DT_STRING};

    TF_ASSERT_OK(
        NodeDefBuilder("test", "ParseSingleSequenceExample")
            .Input("serialized", 0, DT_STRING)
            .Input("feature_list_dense_missing_assumed_empty", 0, DT_STRING)
            .Input(NodeOutList(num_context_sparse, string_in))
            .Input(NodeOutList(num_context_dense, string_in))
            .Input(NodeOutList(num_feature_list_sparse, string_in))
            .Input(NodeOutList(num_feature_list_dense, string_in))
            .Input(NodeOutList(num_context_dense, string_in))
            .Input("debug_name", 0, DT_STRING)
            .Attr("context_sparse_types",
                  DataTypeList(num_context_sparse, DT_FLOAT))
            .Attr("context_dense_types",
                  DataTypeList(num_context_dense, DT_FLOAT))
            .Attr("context_dense_shapes",
                  MakeDenseShapes(num_context_dense, add_extra_shape, 0))
            .Attr("feature_list_sparse_types",
                  DataTypeList(num_feature_list_sparse, DT_FLOAT))
            .Attr("feature_list_dense_types",
                  DataTypeList(num_feature_list_dense, DT_FLOAT))
            .Attr("feature_list_dense_shapes",
                  MakeDenseShapes(num_feature_list_dense, add_extra_shape, 0))
            .Finalize(&op.node_def));
  };

  // Verify inputs 'serialized' and 'feature_list_dense_missing_assumed_empty'.
  set_outputs(0, 0, 0, 0);
  INFER_OK(op, "?;?;?", "");
  INFER_OK(op, "[];[20];?", "");
  INFER_ERROR("must be rank 0", op, "[1];?;?");
  INFER_ERROR("must be rank 1", op, "?;[2,3];?");

  // context inputs with no feature_list inputs.
  set_outputs(2 /* num_context_sparse */, 3 /* num_context_dense */, 0, 0);
  INFER_OK(op, "?;?;?;?;?;?;?;?;?;?;?",
           ("[?,1];[?,1];[?];[?];[1];[1];"  // context sparse outputs
            "[1];[1,2];[1,2,3]"));          // context dense outputs

  // feature_list inputs with no context inputs.
  set_outputs(0, 0, 2 /* num_feature_list_sparse */,
              3 /* num_feature_list_dense */);
  INFER_OK(op, "?;?;?;?;?;?;?;?",
           ("[?,2];[?,2];[?];[?];[2];[2];"  // feature_list sparse outputs
            "[?,1];[?,1,2];[?,1,2,3]"));    // feature_list dense outputs

  // Combine previous two test cases.
  set_outputs(2, 3, 2, 3);
  INFER_OK(op, "?;?;?;?;?;?;?;?;?;?;?;?;?;?;?;?",
           ("[?,1];[?,1];[?];[?];[1];[1];"  // context sparse outputs
            "[1];[1,2];[1,2,3];"            // context dense outputs
            "[?,2];[?,2];[?];[?];[2];[2];"  // feature_list sparse outputs
            "[?,1];[?,1,2];[?,1,2,3]"));    // feature_list dense outputs

  // Confirm an error from ParseSingleSequenceExampleAttrs.Init().
  set_outputs(1, 1, 1, 1, true /* add_extra_shape */);
  INFER_ERROR("len(context_dense_keys) != len(context_dense_shapes)", op,
              "?;?;?;?;?;?;?;?");
}

TEST(ParsingOpsTest, ParseExampleV2_ShapeFn) {
  ShapeInferenceTestOp op("ParseExampleV2");
  auto set_outputs = [&op](int num_sparse, int num_dense, int num_ragged,
                           bool add_extra_shape = false,
                           int unknown_outer_dims = 0) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSopsPSparsing_ops_testDTcc mht_4(mht_4_v, 488, "", "./tensorflow/core/ops/parsing_ops_test.cc", "lambda");

    using NodeOutList = std::vector<NodeDefBuilder::NodeOut>;
    using DataTypeList = std::vector<DataType>;
    NodeDefBuilder::NodeOut string_in{"a", 0, DT_STRING};

    TF_ASSERT_OK(
        NodeDefBuilder("test", "ParseExampleV2")
            .Input("serialized", 0, DT_STRING)
            .Input("names", 0, DT_STRING)
            .Input("sparse_keys", 0, DT_STRING)
            .Input("dense_keys", 0, DT_STRING)
            .Input("ragged_keys", 0, DT_STRING)
            .Input(NodeOutList(num_dense, string_in))  // dense_defaults
            .Attr("num_sparse", num_sparse)
            .Attr("sparse_types", DataTypeList(num_sparse, DT_FLOAT))
            .Attr("ragged_value_types", DataTypeList(num_ragged, DT_FLOAT))
            .Attr("ragged_split_types", DataTypeList(num_ragged, DT_INT32))
            .Attr("dense_shapes", MakeDenseShapes(num_dense, add_extra_shape,
                                                  unknown_outer_dims))
            .Finalize(&op.node_def));
  };

  // Verify inputs 'serialized' and 'names'.
  set_outputs(0 /* num_sparse */, 0 /* num_dense */, 0 /* num_ragged */);
  INFER_OK(op, "?;?;[0];[0];[0]", "");
  INFER_OK(op, "[10];[10];[0];[0];[0]", "");
  INFER_OK(op, "[];[];[0];[0];[0]", "");
  INFER_ERROR("must be at most rank 1", op, "[1,2];?;?;?;?");
  INFER_ERROR("must be at most rank 1", op, "?;[2,3];?;?;?");

  // Verify the sparse, dense, and ragged outputs.
  set_outputs(2 /* num_sparse */, 3 /* num_dense */, 4 /* num_ragged */);
  INFER_OK(op, "[?];?;?;?;?;?;?;?",              // Vector input, unknown size
           ("[?,2];[?,2];"                       // sparse indices
            "[?];[?];"                           // sparse values
            "[2];[2];"                           // sparse dense_shapes
            "[d0_0,1];[d0_0,1,2];[d0_0,1,2,3];"  // dense outputs
            "[?];[?];[?];[?];"                   // ragged values
            "[?];[?];[?];[?]"));                 // ragged row_splits
  INFER_OK(op, "[10];?;?;?;?;?;?;?",             // Vector input, known size
           ("[?,2];[?,2];"                       // sparse indices
            "[?];[?];"                           // sparse values
            "[2];[2];"                           // sparse dense_shapes
            "[d0_0,1];[d0_0,1,2];[d0_0,1,2,3];"  // dense outputs
            "[?];[?];[?];[?];"                   // ragged values
            "[11];[11];[11];[11]"));             // ragged row_splits
  INFER_OK(op, "[];?;?;?;?;?;?;?",               // Scalar input
           ("[?,1];[?,1];"                       // sparse indices
            "[?];[?];"                           // sparse values
            "[1];[1];"                           // sparse dense_shapes
            "[1];[1,2];[1,2,3];"                 // dense outputs
            "[?];[?];[?];[?];"                   // ragged values
            "[?];[?];[?];[?]"));                 // ragged row_splits
  INFER_OK(op, "?;?;?;?;?;?;?;?",                // Input with unknown rank
           ("[?,?];[?,?];"                       // sparse indices
            "[?];[?];"                           // sparse values
            "[?];[?];"                           // sparse dense_shapes
            "?;?;?;"                             // dense outputs
            "[?];[?];[?];[?];"                   // ragged values
            "[?];[?];[?];[?]"));                 // ragged row_splits

  // Confirm an error from ParseExampleAttrs.Init().
  set_outputs(2, 3, 0, true /* add_extra_shape */);
  INFER_ERROR("len(dense_keys) != len(dense_shapes)", op, "?;?;?;?;?;?;?;?");
  set_outputs(2, 3, 0, true /* add_extra_shape */, 1 /* unknown_outer_dims */);
  INFER_ERROR("len(dense_keys) != len(dense_shapes)", op, "?;?;?;?;?;?;?;?");

  // Allow variable strides
  set_outputs(2, 3, 0, false /* add_extra_shape */, 1 /* unknown_outer_dims */);
  INFER_OK(op, "[?];?;?;?;?;?;?;?",
           ("[?,2];[?,2];[?];[?];[2];[2];"               // sparse outputs
            "[d0_0,?,1];[d0_0,?,1,2];[d0_0,?,1,2,3]"));  // dense outputs
  INFER_OK(op, "[10];?;?;?;?;?;?;?",
           ("[?,2];[?,2];[?];[?];[2];[2];"               // sparse outputs
            "[d0_0,?,1];[d0_0,?,1,2];[d0_0,?,1,2,3]"));  // dense outputs

  // Variable inner dimensions are not supported
  set_outputs(2, 3, 0, false /* add_extra_shape */, 2 /* unknown_outer_dims */);
  INFER_ERROR("shapes[0] has unknown rank or unknown inner dimensions", op,
              "?;?;?;?;?;?;?;?");
}

TEST(ParsingOpsTest, ParseSequenceExampleV2_ShapeFn) {
  ShapeInferenceTestOp op("ParseSequenceExampleV2");
  auto set_outputs = [&op](int num_context_sparse, int num_context_dense,
                           int num_context_ragged, int num_feature_list_sparse,
                           int num_feature_list_dense,
                           int num_feature_list_ragged,
                           bool add_extra_shape = false) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSopsPSparsing_ops_testDTcc mht_5(mht_5_v, 579, "", "./tensorflow/core/ops/parsing_ops_test.cc", "lambda");

    using NodeOutList = std::vector<NodeDefBuilder::NodeOut>;
    using DataTypeList = std::vector<DataType>;
    string string_in("test");
    NodeDefBuilder::NodeOut node_in{"a", 0, DT_STRING};
    TF_ASSERT_OK(
        NodeDefBuilder("test", "ParseSequenceExampleV2")
            .Input("serialized", 0, DT_STRING)
            .Input("debug_name", 0, DT_STRING)
            .Input("context_sparse_keys", 0, DT_STRING)
            .Input("context_dense_keys", 0, DT_STRING)
            .Input("context_ragged_keys", 0, DT_STRING)
            .Input("feature_list_sparse_keys", 0, DT_STRING)
            .Input("feature_list_dense_keys", 0, DT_STRING)
            .Input("feature_list_ragged_keys", 0, DT_STRING)
            .Input("feature_list_dense_missing_assumed_empty", 0, DT_BOOL)
            .Input(NodeOutList(num_context_dense, node_in))
            .Attr("Ncontext_sparse", num_context_sparse)
            .Attr("Nfeature_list_sparse", num_feature_list_sparse)
            .Attr("Nfeature_list_dense", num_feature_list_dense)
            .Attr("context_sparse_types",
                  DataTypeList(num_context_sparse, DT_FLOAT))
            .Attr("context_dense_types",
                  DataTypeList(num_context_dense, DT_FLOAT))
            .Attr("context_dense_shapes",
                  MakeDenseShapes(num_context_dense, add_extra_shape, 0))
            .Attr("feature_list_sparse_types",
                  DataTypeList(num_feature_list_sparse, DT_FLOAT))
            .Attr("feature_list_dense_types",
                  DataTypeList(num_feature_list_dense, DT_FLOAT))
            .Attr("feature_list_dense_shapes",
                  MakeDenseShapes(num_feature_list_dense, add_extra_shape, 0))
            .Attr("context_ragged_value_types",
                  DataTypeList(num_context_ragged, DT_FLOAT))
            .Attr("context_ragged_split_types",
                  DataTypeList(num_context_ragged, DT_INT32))
            .Attr("feature_list_ragged_value_types",
                  DataTypeList(num_feature_list_ragged, DT_FLOAT))
            .Attr("feature_list_ragged_split_types",
                  DataTypeList(num_feature_list_ragged, DT_INT32))
            .Finalize(&op.node_def));
  };

  // Verify inputs 'serialized' and 'debug_name'.
  set_outputs(0, 0, 0, 0, 0, 0);  // no features
  INFER_OK(op, "?;[?];?;?;?;?;?;?;?", "");
  INFER_OK(op, "[?];[?];?;?;?;?;?;?;?", "");
  INFER_OK(op, "[8];[8];?;?;?;?;?;?;?", "");
  INFER_OK(op, "[];[];?;?;?;?;?;?;?", "");
  INFER_ERROR("must be at most rank 1", op, "[1,2];?;?;?;?;?;?;?;?");
  INFER_ERROR("must be at most rank 1", op, "?;[2,3];?;?;?;?;?;?;?");

  // context inputs with no feature_list inputs.
  set_outputs(2 /* num_context_sparse */, 3 /* num_context_dense */,
              4 /* num_ragged */, 0, 0, 0);
  INFER_OK(op, "[?];[?];?;?;?;?;?;?;?;?;?;?",    // Vector input, unknown size
           ("[?,2];[?,2];"                       //  context sparse indices
            "[?];[?];"                           //  context sparse values
            "[2];[2];"                           //  context sparse dense_shapes
            "[d0_0,1];[d0_0,1,2];[d0_0,1,2,3];"  //  context dense outputs
            "[?];[?];[?];[?];"                   //  context ragged values
            "[?];[?];[?];[?]"));                 //  context ragged row_splits
  INFER_OK(op, "[5];[?];?;?;?;?;?;?;?;?;?;?",    // Vector input, known size
           ("[?,2];[?,2];"                       //  context sparse indices
            "[?];[?];"                           //  context sparse values
            "[2];[2];"                           //  context sparse dense_shapes
            "[d0_0,1];[d0_0,1,2];[d0_0,1,2,3];"  //  context dense outputs
            "[?];[?];[?];[?];"                   //  context ragged values
            "[6];[6];[6];[6]"));                 //  context ragged row_splits
  INFER_OK(op, "[];[?];?;?;?;?;?;?;?;?;?;?",     // Scalar input
           ("[?,1];[?,1];"                       //  context sparse indices
            "[?];[?];"                           //  context sparse values
            "[1];[1];"                           //  context sparse dense_shapes
            "[1];[1,2];[1,2,3];"                 //  context dense outputs
            "[?];[?];[?];[?];"                   //  context ragged values
            "[?];[?];[?];[?]"));                 //  context ragged row_splits
  INFER_OK(op, "?;[?];?;?;?;?;?;?;?;?;?;?",      // Unknown rank
           ("[?,?];[?,?];"                       //  context sparse indices
            "[?];[?];"                           //  context sparse values
            "[?];[?];"                           //  context sparse dense_shapes
            "?;?;?;"                             //  context dense outputs
            "[?];[?];[?];[?];"                   //  context ragged values
            "[?];[?];[?];[?]"));                 //  context ragged row_splits

  // feature_list inputs with no context inputs.
  set_outputs(0, 0, 0, 2 /* num_context_sparse */, 3 /* num_context_dense */,
              4 /* num_ragged */);
  INFER_OK(op, "[?];[?];?;?;?;?;?;?;?",  // Vector input, unknown size
           ("[?,3];[?,3];"               //  f_list sparse indices
            "[?];[?];"                   //  f_list sparse values
            "[3];[3];"                   //  f_list sparse dense_shapes
            "[d0_0,?,1];[d0_0,?,1,2];"   //  f_list dense outputs
            "[d0_0,?,1,2,3];"            //     (continued)
            "in0;in0;in0;"               //  f_list dense lengths
            "[?];[?];[?];[?];"           //  f_list ragged values
            "[?];[?];[?];[?];"           //  f_list ragged outer_splits
            "[?];[?];[?];[?]"));         //  f_list ragged inner_splits
  INFER_OK(op, "[5];[?];?;?;?;?;?;?;?",  // Vector input, known size
           ("[?,3];[?,3];"               //  f_list sparse indices
            "[?];[?];"                   //  f_list sparse values
            "[3];[3];"                   //  f_list sparse dense_shapes
            "[d0_0,?,1];[d0_0,?,1,2];"   //  f_list dense outputs
            "[d0_0,?,1,2,3];"            //    (continued)
            "in0;in0;in0;"               //  f_list dense lengths
            "[?];[?];[?];[?];"           //  f_list ragged values
            "[6];[6];[6];[6];"           //  f_list ragged outer_splits
            "[?];[?];[?];[?]"));         //  f_list ragged inner_splits
  INFER_OK(op, "[];[?];?;?;?;?;?;?;?",   // Scalar input
           ("[?,2];[?,2];"               //  f_list sparse indices
            "[?];[?];"                   //  f_list sparse values
            "[2];[2];"                   //  f_list sparse dense_shapes
            "[?,1];[?,1,2];[?,1,2,3];"   //  f_list dense outputs
            "in0;in0;in0;"               //  f_list dense lengths
            "[?];[?];[?];[?];"           //  f_list ragged values
            "[?];[?];[?];[?];"           //  f_list ragged outer_splits
            "[?];[?];[?];[?]"));         //  f_list ragged inner_splits
  INFER_OK(op, "?;[?];?;?;?;?;?;?;?",    // Unknown rank
           ("[?,?];[?,?];"               //  f_list sparse indices
            "[?];[?];"                   //  f_list sparse values
            "[?];[?];"                   //  f_list sparse dense_shapes
            "?;?;?;"                     //  f_list dense outputs
            "in0;in0;in0;"               //  f_list dense lengths
            "[?];[?];[?];[?];"           //  f_list ragged values
            "[?];[?];[?];[?];"           //  f_list ragged outer_splits
            "[?];[?];[?];[?]"));         //  f_list ragged inner_splits

  // Combine previous two test cases.
  set_outputs(2 /* num_context_sparse */, 3 /* num_context_dense */,
              4 /* num_ragged */, 2 /* num_context_sparse */,
              3 /* num_context_dense */, 4 /* num_ragged */);
  INFER_OK(op, "[?];[?];?;?;?;?;?;?;?;?;?;?",    // Vector input, unknown size
           ("[?,2];[?,2];"                       //  context sparse indices
            "[?];[?];"                           //  context sparse values
            "[2];[2];"                           //  context sparse dense_shapes
            "[d0_0,1];[d0_0,1,2];[d0_0,1,2,3];"  //  context dense outputs
            "[?];[?];[?];[?];"                   //  context ragged values
            "[?];[?];[?];[?];"                   //  context ragged row_splits
            "[?,3];[?,3];"                       //  f_list sparse indices
            "[?];[?];"                           //  f_list sparse values
            "[3];[3];"                           //  f_list sparse dense_shapes
            "[d0_0,?,1];[d0_0,?,1,2];"           //  f_list dense outputs
            "[d0_0,?,1,2,3];"                    //     (continued)
            "in0;in0;in0;"                       //  f_list dense lengths
            "[?];[?];[?];[?];"                   //  f_list ragged values
            "[?];[?];[?];[?];"                   //  f_list ragged outer_splits
            "[?];[?];[?];[?]"));                 //  f_list ragged inner_splits
  INFER_OK(op, "[5];[?];?;?;?;?;?;?;?;?;?;?",    // Vector input, known size
           ("[?,2];[?,2];"                       //  context sparse indices
            "[?];[?];"                           //  context sparse values
            "[2];[2];"                           //  context sparse dense_shapes
            "[d0_0,1];[d0_0,1,2];[d0_0,1,2,3];"  //  context dense outputs
            "[?];[?];[?];[?];"                   //  context ragged values
            "[6];[6];[6];[6];"                   //  context ragged row_splits
            "[?,3];[?,3];"                       //  f_list sparse indices
            "[?];[?];"                           //  f_list sparse values
            "[3];[3];"                           //  f_list sparse dense_shapes
            "[d0_0,?,1];[d0_0,?,1,2];"           //  f_list dense outputs
            "[d0_0,?,1,2,3];"                    //    (continued)
            "in0;in0;in0;"                       //  f_list dense lengths
            "[?];[?];[?];[?];"                   //  f_list ragged values
            "[6];[6];[6];[6];"                   //  f_list ragged outer_splits
            "[?];[?];[?];[?]"));                 //  f_list ragged inner_splits
  INFER_OK(op, "[];[?];?;?;?;?;?;?;?;?;?;?",     // Scalar input
           ("[?,1];[?,1];"                       //  context sparse indices
            "[?];[?];"                           //  context sparse values
            "[1];[1];"                           //  context sparse dense_shapes
            "[1];[1,2];[1,2,3];"                 //  context dense outputs
            "[?];[?];[?];[?];"                   //  context ragged values
            "[?];[?];[?];[?];"                   //  context ragged row_splits
            "[?,2];[?,2];"                       //  f_list sparse indices
            "[?];[?];"                           //  f_list sparse values
            "[2];[2];"                           //  f_list sparse dense_shapes
            "[?,1];[?,1,2];[?,1,2,3];"           //  f_list dense outputs
            "in0;in0;in0;"                       //  f_list dense lengths
            "[?];[?];[?];[?];"                   //  f_list ragged values
            "[?];[?];[?];[?];"                   //  f_list ragged outer_splits
            "[?];[?];[?];[?]"));                 //  f_list ragged inner_splits
  INFER_OK(op, "?;[?];?;?;?;?;?;?;?;?;?;?",      // Unknown rank
           ("[?,?];[?,?];"                       //  context sparse indices
            "[?];[?];"                           //  context sparse values
            "[?];[?];"                           //  context sparse dense_shapes
            "?;?;?;"                             //  context dense outputs
            "[?];[?];[?];[?];"                   //  context ragged values
            "[?];[?];[?];[?];"                   //  context ragged row_splits
            "[?,?];[?,?];"                       //  f_list sparse indices
            "[?];[?];"                           //  f_list sparse values
            "[?];[?];"                           //  f_list sparse dense_shapes
            "?;?;?;"                             //  f_list dense outputs
            "in0;in0;in0;"                       //  f_list dense lengths
            "[?];[?];[?];[?];"                   //  f_list ragged values
            "[?];[?];[?];[?];"                   //  f_list ragged outer_splits
            "[?];[?];[?];[?]"));                 //  f_list ragged inner_splits

  // Confirm an error from ParseSequenceExampleAttrs.Init().
  set_outputs(1, 1, 1, 1, 1, 1, true /* add_extra_shape */);
  INFER_ERROR(
      "num_context_dense (1) must match the size of "
      "context_dense_types (1) and context_dense_shapes (2)",
      op, "[?];[?];?;?;?;?;?;?;?;?");
}

}  // end namespace tensorflow
