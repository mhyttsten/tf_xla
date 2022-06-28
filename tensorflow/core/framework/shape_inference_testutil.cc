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
class MHTracer_DTPStensorflowPScorePSframeworkPSshape_inference_testutilDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inference_testutilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSshape_inference_testutilDTcc() {
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
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/framework/shape_inference_testutil.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace shape_inference {

using errors::Unknown;

Status ShapeInferenceTestutil::InferShapes(ShapeInferenceTestOp op,
                                           const string& ins,
                                           const string& expected_outs) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("ins: \"" + ins + "\"");
   mht_0_v.push_back("expected_outs: \"" + expected_outs + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inference_testutilDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/framework/shape_inference_testutil.cc", "ShapeInferenceTestutil::InferShapes");

  const OpRegistrationData* op_reg_data;
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUp(op.name, &op_reg_data));

  std::vector<string> ins_v = str_util::Split(ins, ';');

  InferenceContext::ShapeManager manager;
  std::vector<ShapeHandle> in_shapes;
  for (const string& spec : ins_v) {
    ShapeHandle shape;
    TF_RETURN_IF_ERROR(MakeShapeFromString(&manager, spec, &shape));
    in_shapes.push_back(shape);
  }

  std::vector<std::unique_ptr<std::vector<shape_inference::ShapeAndType>>>
      input_resource_handle_shapes_and_types;
  for (const auto p : op.input_resource_handle_shapes_and_types) {
    if (p == nullptr) {
      input_resource_handle_shapes_and_types.push_back(nullptr);
    } else {
      std::unique_ptr<std::vector<ShapeAndType>> v(
          new std::vector<ShapeAndType>());
      for (const auto& shape_and_type : *p) {
        ShapeHandle shape;
        TF_RETURN_IF_ERROR(
            MakeShapeFromString(&manager, shape_and_type.first, &shape));
        v->emplace_back(shape, shape_and_type.second);
      }
      input_resource_handle_shapes_and_types.emplace_back(v.release());
    }
  }
  shape_inference::InferenceContext c(
      op.graph_def_version, op.node_def, op_reg_data->op_def, in_shapes,
      op.input_tensors, {}, std::move(input_resource_handle_shapes_and_types));
  TF_RETURN_IF_ERROR(c.construction_status());
  if (op_reg_data->shape_inference_fn == nullptr) {
    return errors::InvalidArgument(
        "No shape inference function exists for op '", op.name,
        "', did you forget to define it?");
  }

  TF_RETURN_IF_ERROR(c.Run(op_reg_data->shape_inference_fn));

  const int num_outputs = c.num_outputs();

  if (expected_outs == "e") {
    return Unknown("Shape inference should have returned error");
  }

  // Verify the output shape.
  std::vector<string> expected_outs_v = str_util::Split(expected_outs, ';');
  if (num_outputs != expected_outs_v.size()) {
    return Unknown("The expected output string lists the wrong number of ",
                   "outputs. It lists ", expected_outs_v.size(),
                   " but should list ", num_outputs);
  }
  for (int i = 0; i < num_outputs; ++i) {
    StringPiece expected(expected_outs_v[i]);
    shape_inference::ShapeHandle out = c.output(i);

    string err_prefix = strings::StrCat("Output ", i);
    string err_suffix =
        strings::StrCat(". Output shape was ", c.DebugString(out));

    int in_index = -1;
    for (int i = 0; i < c.num_inputs(); ++i) {
      if (c.input(i).SameHandle(out)) {
        in_index = i;
      }
    }

    if (absl::StartsWith(expected, "in")) {
      if (in_index == -1) {
        return Unknown(err_prefix,
                       " should have matched an input shape by "
                       "handle, but matched no input shape. This means the ",
                       "shape function was expected to pass an input "
                       "ShapeHandle through for this output, but did not",
                       err_suffix);
      }
      auto v = str_util::Split(expected, '|');
      if (std::find(v.begin(), v.end(), strings::StrCat("in", in_index)) ==
          v.end()) {
        return Unknown(
            err_prefix, " matched input ", in_index,
            " by handle, but should have matched one of (", expected,
            ") instead. This means the shape function passed the ShapeHandle ",
            "for input ", in_index,
            " to the output, but should have passed a different input ",
            "ShapeHandle through", err_suffix);
      }
      continue;
    }
    if (in_index != -1) {
      return Unknown(err_prefix, " matched input ", in_index,
                     " by ShapeHandle, but was expected to not match an input ",
                     "shape by handle", err_suffix);
    }
    if (expected == "?") {
      if (c.RankKnown(out)) {
        return Unknown(err_prefix, " expected to be unknown", err_suffix);
      }
      continue;
    }

    // Verify the dimensions.
    CHECK(absl::StartsWith(expected, "[") && str_util::EndsWith(expected, "]"))
        << expected;
    expected.remove_prefix(1);
    expected.remove_suffix(1);

    // Split expected as a dimension.
    auto expected_dims = str_util::Split(expected, ',');
    if (!c.RankKnown(out)) {
      return Unknown(err_prefix, " expected rank ", expected_dims.size(),
                     " but was ?", err_suffix);
    }
    if (c.Rank(out) != expected_dims.size()) {
      return Unknown(err_prefix, " expected rank ", expected_dims.size(),
                     " but was ", c.Rank(out), err_suffix);
    }
    for (int j = 0; j < expected_dims.size(); ++j) {
      err_prefix = strings::StrCat("Output dim ", i, ",", j);
      StringPiece expected_dim(expected_dims[j]);
      DimensionHandle out_dim = c.Dim(out, j);

      std::pair<int, int> in_dim_idx(-1, -1);
      for (int i = 0; i < c.num_inputs(); ++i) {
        auto in = c.input(i);
        for (int j = 0; j < c.Rank(in); ++j) {
          if (c.Dim(in, j).SameHandle(out_dim)) {
            in_dim_idx = std::make_pair(i, j);
          }
        }
      }

      if (expected_dim == "?") {
        if (in_dim_idx.first != -1) {
          return Unknown(err_prefix,
                         " expected to be an unknown but matched input d",
                         in_dim_idx.first, "_", in_dim_idx.second,
                         ". The shape function passed through ",
                         "a DimensionHandle from an input instead of making ",
                         "a new unknown dimension", err_suffix);
        } else if (c.ValueKnown(out_dim)) {
          return Unknown(err_prefix, " expected to be unknown but was ",
                         c.Value(out_dim), err_suffix);
        }
      } else if (absl::StartsWith(expected_dim, "d")) {
        // Compare the dimension values.
        auto v = str_util::Split(expected_dim, '|');
        if (in_dim_idx.first == -1) {
          return Unknown(
              err_prefix, " was expected to match the dimension of an input, ",
              "but did not match any input dimension. The shape ",
              "function was expected to pass through a ",
              "DimensionHandle for an input, but did not", err_suffix);
        }
        if (std::find(v.begin(), v.end(),
                      strings::StrCat("d", in_dim_idx.first, "_",
                                      in_dim_idx.second)) == v.end()) {
          return Unknown(err_prefix, " matched input d", in_dim_idx.first, "_",
                         in_dim_idx.second,
                         ", but should have matched one of (", expected_dim,
                         "). The shape function passed through "
                         "the DimensionHandle for an input, but ",
                         "was expected to pass a different one", err_suffix);
        }
      } else {
        // Parse it as a value.
        int64_t value = -1;
        if (!strings::safe_strto64(expected_dim, &value)) {
          return Unknown(err_prefix, ": the expected dimension value '",
                         expected_dim, "' failed to parse as int64",
                         err_suffix);
        }
        if (in_dim_idx.first != -1) {
          return Unknown(  //
              err_prefix, " expected to be ", value, " but matched input d",
              in_dim_idx.first, "_", in_dim_idx.second,
              ". The shape function was not expected to pass a DimensionHandle "
              "from the input to the output, but did. Note that even if the "
              "passed through output has the same dimension value as the "
              "expected value, this is considered a failure for the test; "
              "switch to using d#_# syntax if passing through the "
              "DimensionHandle should be the expected behavior",
              err_suffix);
        } else if (value != c.Value(out_dim)) {
          return Unknown(err_prefix, " expected to be ", value, " but was ",
                         c.DebugString(out_dim), err_suffix);
        }
      }
    }
  }
  return Status::OK();
}

// static
Status ShapeInferenceTestutil::MakeShapeFromString(
    InferenceContext::ShapeManager* manager, const string& spec,
    ShapeHandle* output) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("spec: \"" + spec + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inference_testutilDTcc mht_1(mht_1_v, 406, "", "./tensorflow/core/framework/shape_inference_testutil.cc", "ShapeInferenceTestutil::MakeShapeFromString");

  if (spec == "?") {
    *output = manager->UnknownShape();
    return Status::OK();
  }

  std::vector<DimensionHandle> dims;
  strings::Scanner scanner(spec);
  scanner.OneLiteral("[");
  while (scanner.Peek() != ']') {
    if (scanner.Peek() == '?') {
      scanner.OneLiteral("?");
      dims.push_back(manager->MakeDim(InferenceContext::kUnknownDim));
    } else {
      scanner.RestartCapture().Many(strings::Scanner::DIGIT);
      StringPiece match;
      int64_t dim_size = 0;

      if (!scanner.GetResult(nullptr, &match) ||
          !strings::safe_strto64(match, &dim_size)) {
        return errors::InvalidArgument("Could not parse number in ", spec);
      }

      dims.push_back(manager->MakeDim(dim_size));
    }

    if (scanner.Peek() == ',') {
      scanner.OneLiteral(",");
    } else if (scanner.Peek() != ']') {
      return errors::InvalidArgument(
          "Invalid input spec (] not found in dim shape): ", spec);
    }
  }
  if (!scanner.OneLiteral("]").Eos().GetResult()) {
    return errors::InvalidArgument("Malformed shape spec: did not end in ']'.");
  }
  *output = manager->MakeShape(dims);

  return Status::OK();
}

}  // namespace shape_inference
}  // namespace tensorflow
