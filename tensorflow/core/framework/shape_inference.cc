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
class MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc() {
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
#include "tensorflow/core/framework/shape_inference.h"

#include <cstdint>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/full_type_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/overflow.h"

namespace tensorflow {
namespace shape_inference {

constexpr int32_t InferenceContext::kUnknownRank;
constexpr int64_t InferenceContext::kUnknownDim;

// Same as above, but with PartialTensorShape instead of TensorShapeProto
InferenceContext::InferenceContext(
    int graph_def_version, const AttrSlice& attrs, const OpDef& op_def,
    const std::vector<PartialTensorShape>& input_shapes,
    const std::vector<const Tensor*>& input_tensors,
    const std::vector<PartialTensorShape>& input_tensors_as_shapes,
    const std::vector<
        std::unique_ptr<std::vector<std::pair<PartialTensorShape, DataType>>>>&
        input_handle_shapes_and_types)
    : graph_def_version_(graph_def_version), attrs_(attrs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::InferenceContext");

  std::vector<ShapeHandle> input_tensors_as_shape_handles;
  input_tensors_as_shape_handles.reserve(input_tensors_as_shapes.size());
  for (const PartialTensorShape& p : input_tensors_as_shapes) {
    ShapeHandle shape;
    construction_status_.Update(MakeShapeFromPartialTensorShape(p, &shape));
    if (!construction_status_.ok()) {
      return;
    }
    input_tensors_as_shape_handles.push_back(shape);
  }
  PreInputInit(op_def, input_tensors, input_tensors_as_shape_handles);
  if (!construction_status_.ok()) return;
  inputs_.reserve(input_shapes.size());
  for (const PartialTensorShape& p : input_shapes) {
    ShapeHandle shape;
    construction_status_.Update(MakeShapeFromPartialTensorShape(p, &shape));
    if (!construction_status_.ok()) {
      return;
    }
    inputs_.push_back(shape);
  }
  std::vector<std::unique_ptr<std::vector<ShapeAndType>>> handle_data(
      input_shapes.size());
  for (int i = 0, end = input_handle_shapes_and_types.size(); i < end; ++i) {
    const auto& v = input_handle_shapes_and_types[i];
    if (v == nullptr) {
      continue;
    }
    handle_data[i].reset(new std::vector<ShapeAndType>(v->size()));
    auto& new_v = *handle_data[i];
    for (int j = 0, end = v->size(); j < end; ++j) {
      const auto& p = (*v)[j];
      construction_status_.Update(
          MakeShapeFromPartialTensorShape(p.first, &new_v[j].shape));
      if (!construction_status_.ok()) {
        return;
      }
      new_v[j].dtype = p.second;
    }
  }
  PostInputInit(std::move(handle_data));
}

InferenceContext::InferenceContext(
    int graph_def_version, const AttrSlice& attrs, const OpDef& op_def,
    const std::vector<ShapeHandle>& input_shapes,
    const std::vector<const Tensor*>& input_tensors,
    const std::vector<ShapeHandle>& input_tensors_as_shapes,
    std::vector<std::unique_ptr<std::vector<ShapeAndType>>>
        input_handle_shapes_and_types)
    : graph_def_version_(graph_def_version), attrs_(attrs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_1(mht_1_v, 269, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::InferenceContext");

  PreInputInit(op_def, input_tensors, input_tensors_as_shapes);
  if (!construction_status_.ok()) return;
  inputs_ = input_shapes;

  PostInputInit(std::move(input_handle_shapes_and_types));
}

InferenceContext::~InferenceContext() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_2(mht_2_v, 280, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::~InferenceContext");
}

Status InferenceContext::Run(
    const std::function<Status(shape_inference::InferenceContext* c)>& fn) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_3(mht_3_v, 286, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Run");

  ForgetMerges();
  Status s = fn(this);
  if (!s.ok()) {
    ForgetMerges();
    return AttachContext(s);
  }
#ifndef NDEBUG
  for (int i = 0; i < num_outputs(); ++i) {
    DCHECK(output(i).IsSet()) << i << " for " << attrs_.SummarizeNode();
  }
#endif  // NDEBUG
  return s;
}

Status InferenceContext::set_output(StringPiece output_name,
                                    const std::vector<ShapeHandle>& shapes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_4(mht_4_v, 305, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::set_output");

  auto result = output_name_map_.find(output_name);
  if (result == output_name_map_.end()) {
    return errors::InvalidArgument("Unknown output name: ", output_name);
  } else {
    const int start = result->second.first;
    const int size = result->second.second - start;
    const int shapes_size = shapes.size();
    if (size != shapes_size) {
      return errors::InvalidArgument("Must have exactly ", shapes.size(),
                                     " shapes.");
    }
    for (int i = 0; i < shapes_size; ++i) {
      outputs_[i + start] = shapes[i];
    }
  }
  return Status::OK();
}

Status InferenceContext::input(StringPiece input_name,
                               std::vector<ShapeHandle>* output) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_5(mht_5_v, 328, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::input");

  const auto result = input_name_map_.find(input_name);
  if (result == input_name_map_.end()) {
    return errors::InvalidArgument("Unknown input name: ", input_name);
  } else {
    output->clear();
    for (int i = result->second.first; i < result->second.second; ++i) {
      output->push_back(inputs_[i]);
    }
  }
  return Status::OK();
}

Status InferenceContext::output(StringPiece output_name,
                                std::vector<ShapeHandle>* output) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_6(mht_6_v, 345, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::output");

  const auto result = output_name_map_.find(output_name);
  if (result == output_name_map_.end()) {
    return errors::InvalidArgument("Unknown output name: ", output_name);
  } else {
    output->clear();
    for (int i = result->second.first; i < result->second.second; ++i) {
      output->push_back(outputs_[i]);
    }
  }
  return Status::OK();
}

void InferenceContext::PreInputInit(
    const OpDef& op_def, const std::vector<const Tensor*>& input_tensors,
    const std::vector<ShapeHandle>& input_tensors_as_shapes) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_7(mht_7_v, 363, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::PreInputInit");

  // TODO(mdan): This is also done at graph construction. Run only here instead?
  Status s = full_type::SpecializeType(attrs_, op_def, ret_types_);
  if (!s.ok()) {
    construction_status_ = s;
    return;
  }

  input_tensors_ = input_tensors;
  input_tensors_as_shapes_ = input_tensors_as_shapes;

  construction_status_ =
      NameRangesForNode(attrs_, op_def, &input_name_map_, &output_name_map_);
  if (!construction_status_.ok()) return;

  int num_outputs = 0;
  for (const auto& e : output_name_map_) {
    num_outputs = std::max(num_outputs, e.second.second);
  }
  outputs_.assign(num_outputs, nullptr);
  output_handle_shapes_and_types_.resize(num_outputs);
}

Status InferenceContext::ExpandOutputs(int new_output_size) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_8(mht_8_v, 389, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::ExpandOutputs");

  const int outputs_size = outputs_.size();
  if (new_output_size < outputs_size) {
    return errors::InvalidArgument("Trying to reduce number of outputs of op.");
  }
  outputs_.resize(new_output_size, nullptr);
  output_handle_shapes_and_types_.resize(new_output_size);
  return Status::OK();
}

void InferenceContext::PostInputInit(
    std::vector<std::unique_ptr<std::vector<ShapeAndType>>> input_handle_data) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_9(mht_9_v, 403, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::PostInputInit");

  int num_inputs_from_node_def = 0;
  for (const auto& e : input_name_map_) {
    num_inputs_from_node_def =
        std::max(num_inputs_from_node_def, e.second.second);
  }

  // Allow passing empty shapes/dtypes to avoid changing every single test.
  if (input_handle_data.empty()) {
    input_handle_shapes_and_types_.resize(inputs_.size());
  } else {
    if (input_handle_data.size() != inputs_.size()) {
      construction_status_ = errors::InvalidArgument(
          "Wrong number of handle shapes passed; expected ", inputs_.size(),
          " got ", input_handle_data.size());
      return;
    }
    input_handle_shapes_and_types_ = std::move(input_handle_data);
  }
  const int inputs_size = inputs_.size();
  if (inputs_size != num_inputs_from_node_def) {
    construction_status_ = errors::InvalidArgument(
        "Wrong number of inputs passed: ", inputs_.size(), " while ",
        num_inputs_from_node_def, " expected based on NodeDef");
    return;
  }

  CHECK_LE(input_tensors_.size(), inputs_.size());
  input_tensors_.resize(inputs_.size());
  requested_input_tensor_.resize(inputs_.size());
  requested_input_tensor_as_partial_shape_.resize(inputs_.size());
}

void InferenceContext::ShapeHandleToProto(ShapeHandle handle,
                                          TensorShapeProto* proto) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_10(mht_10_v, 440, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::ShapeHandleToProto");

  if (!RankKnown(handle)) {
    proto->set_unknown_rank(true);
    return;
  }

  for (int32_t i = 0; i < Rank(handle); ++i) {
    DimensionHandle dim = Dim(handle, i);
    auto* dim_shape = proto->add_dim();
    if (ValueKnown(dim)) {
      dim_shape->set_size(Value(dim));
    } else {
      dim_shape->set_size(-1);
    }
  }
}

bool InferenceContext::FullyDefined(ShapeHandle s) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_11(mht_11_v, 460, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::FullyDefined");

  if (!RankKnown(s)) return false;
  for (int i = 0; i < Rank(s); ++i) {
    if (!ValueKnown(Dim(s, i))) return false;
  }
  return true;
}

DimensionHandle InferenceContext::NumElements(ShapeHandle s) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_12(mht_12_v, 471, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::NumElements");

  const auto rank = Rank(s);
  if (rank == kUnknownRank) return UnknownDim();
  bool found_unknown = false;
  int64_t size = 1;
  for (int i = 0; i < rank; ++i) {
    int64_t dim_val = Value(Dim(s, i));
    if (dim_val == kUnknownDim) {
      found_unknown = true;
    } else if (dim_val == 0) {
      return MakeDim(0);
    } else {
      size *= dim_val;
    }
  }
  if (found_unknown) {
    return UnknownDim();
  } else {
    return MakeDim(size);
  }
}

string InferenceContext::DebugString(ShapeHandle s) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_13(mht_13_v, 496, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::DebugString");

  if (RankKnown(s)) {
    std::vector<string> vals;
    for (auto d : s->dims_) vals.push_back(DebugString(d));
    return strings::StrCat("[", absl::StrJoin(vals, ","), "]");
  } else {
    return "?";
  }
}

string InferenceContext::DebugString(DimensionHandle d) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_14(mht_14_v, 509, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::DebugString");

  return ValueKnown(d) ? strings::StrCat(Value(d)) : "?";
}

string InferenceContext::DebugString() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_15(mht_15_v, 516, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::DebugString");

  return strings::StrCat("InferenceContext for node: ", attrs_.SummarizeNode());
}

string InferenceContext::DebugString(const ShapeAndType& shape_and_type) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_16(mht_16_v, 523, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::DebugString");

  return strings::StrCat(DebugString(shape_and_type.shape), ":",
                         DataTypeString(shape_and_type.dtype));
}

string InferenceContext::DebugString(
    gtl::ArraySlice<ShapeAndType> shape_and_types) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_17(mht_17_v, 532, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::DebugString");

  std::vector<string> pieces;
  for (const ShapeAndType& s : shape_and_types) {
    pieces.push_back(DebugString(s));
  }
  return strings::StrCat("[", absl::StrJoin(pieces, ","), "]");
}

Status InferenceContext::WithRank(ShapeHandle shape, int64_t rank,
                                  ShapeHandle* out) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_18(mht_18_v, 544, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::WithRank");

  if (rank > kint32max) {
    return errors::InvalidArgument("Rank cannot exceed kint32max");
  }
  const int32_t existing = Rank(shape);
  if (existing == rank) {
    *out = shape;
    return Status::OK();
  }
  if (existing == kUnknownRank) {
    std::vector<DimensionHandle> dims;
    dims.reserve(rank);
    for (int i = 0; i < rank; ++i) {
      dims.push_back(UnknownDim());
    }
    ShapeHandle shp = shape_manager_.MakeShape(dims);
    return Merge(shape, shp, out);
  }
  *out = nullptr;

  return errors::InvalidArgument("Shape must be rank ", rank, " but is rank ",
                                 existing);
}

Status InferenceContext::WithRankAtLeast(ShapeHandle shape, int64_t rank,
                                         ShapeHandle* out) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_19(mht_19_v, 572, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::WithRankAtLeast");

  if (rank > kint32max) {
    return errors::InvalidArgument("Rank cannot exceed kint32max");
  }
  const int32_t existing = Rank(shape);
  if (existing >= rank || existing == kUnknownRank) {
    *out = shape;
    return Status::OK();
  }
  *out = nullptr;
  return errors::InvalidArgument("Shape must be at least rank ", rank,
                                 " but is rank ", existing);
}

Status InferenceContext::WithRankAtMost(ShapeHandle shape, int64_t rank,
                                        ShapeHandle* out) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_20(mht_20_v, 590, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::WithRankAtMost");

  if (rank > kint32max) {
    return errors::InvalidArgument("Rank cannot exceed kint32max");
  }
  const int32_t existing = Rank(shape);
  if (existing <= rank || existing == kUnknownRank) {
    *out = shape;
    return Status::OK();
  }
  *out = nullptr;
  return errors::InvalidArgument("Shape must be at most rank ", rank,
                                 " but is rank ", existing);
}

Status InferenceContext::WithValue(DimensionHandle dim, int64_t value,
                                   DimensionHandle* out) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_21(mht_21_v, 608, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::WithValue");

  const int64_t existing = Value(dim);
  if (existing == value) {
    *out = dim;
    return Status::OK();
  }
  if (existing == kUnknownDim) {
    DimensionHandle d = MakeDim(value);
    return Merge(dim, d, out);
  }
  *out = nullptr;
  return errors::InvalidArgument("Dimension must be ", value, " but is ",
                                 existing);
}

void InferenceContext::Relax(DimensionHandle d_old, DimensionHandle d_new,
                             DimensionHandle* out) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_22(mht_22_v, 627, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Relax");

  if (d_old.SameHandle(d_new)) {
    *out = d_old;
  } else if (!ValueKnown(d_old) && !ValueKnown(d_new)) {
    // The node will be fed by the dimension d_new instead of d_old: any
    // equality assertion between d_old and other input dimension on this node
    // may not be true anymore, so forget them all.
    ForgetMerges();
    // Return the new shape handle to force the relaxation to propagate to the
    // fanout of the context.
    *out = d_new;
  } else if (!ValueKnown(d_new)) {
    ForgetMerges();
    *out = d_new;
  } else if (Value(d_old) == Value(d_new)) {
    // Return the old shape handle. This will stop the relaxation in the fanout
    // of the context.
    *out = d_old;
  } else {
    // Return a new handle that encodes a different unknown dim.
    ForgetMerges();
    *out = UnknownDim();
  }
}

Status InferenceContext::Merge(DimensionHandle d0, DimensionHandle d1,
                               DimensionHandle* out) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_23(mht_23_v, 656, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Merge");

  if (d0.SameHandle(d1)) {
    *out = d0;
    return Status::OK();
  } else if (!ValueKnown(d1)) {
    *out = d0;
    merged_dims_.emplace_back(d0, d1);
    return Status::OK();
  } else if (!ValueKnown(d0)) {
    *out = d1;
    merged_dims_.emplace_back(d0, d1);
    return Status::OK();
  } else if (Value(d0) == Value(d1)) {
    *out = d0;
    return Status::OK();
  } else {
    *out = nullptr;
    return errors::InvalidArgument("Dimensions must be equal, but are ",
                                   Value(d0), " and ", Value(d1));
  }
}

Status InferenceContext::MergePrefix(ShapeHandle s, ShapeHandle prefix,
                                     ShapeHandle* s_out,
                                     ShapeHandle* prefix_out) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_24(mht_24_v, 683, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::MergePrefix");

  *s_out = *prefix_out = nullptr;
  if (!RankKnown(prefix) || !RankKnown(s)) {
    *s_out = s;
    *prefix_out = prefix;
    return Status::OK();
  }
  const int32_t rank = Rank(prefix);
  TF_RETURN_IF_ERROR(WithRankAtLeast(s, rank, &s));

  // Merge the prefix dims and create the new output shapes.
  const int32_t rank_s = Rank(s);
  std::vector<DimensionHandle> dims;
  dims.reserve(std::max(rank, rank_s));
  dims.resize(rank);
  for (int i = 0; i < rank; ++i) {
    TF_RETURN_IF_ERROR(Merge(Dim(s, i), Dim(prefix, i), &dims[i]));
  }
  *prefix_out = MakeShape(dims);
  for (int i = rank; i < rank_s; ++i) dims.push_back(Dim(s, i));
  *s_out = MakeShape(dims);
  return Status::OK();
}

void InferenceContext::Relax(ShapeHandle s_old, ShapeHandle s_new,
                             ShapeHandle* out) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_25(mht_25_v, 711, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Relax");

  if (s_old.SameHandle(s_new)) {
    *out = s_old;
    return;
  } else if (!RankKnown(s_new) || !s_old.IsSet()) {
    ForgetMerges();
    *out = s_new;
    return;
  }

  const int32_t rank = Rank(s_old);
  if (rank != Rank(s_new)) {
    ForgetMerges();
    *out = UnknownShape();
    return;
  }

  bool return_s_old = true;
  for (int i = 0; i < rank; ++i) {
    auto d0 = Dim(s_old, i);
    auto d1 = Dim(s_new, i);
    if (d0.SameHandle(d1)) continue;

    auto v0 = Value(d0);
    auto v1 = Value(d1);
    if (v0 == kUnknownDim || v1 == kUnknownDim || v0 != v1) {
      return_s_old = false;
      break;
    }
  }
  if (return_s_old) {
    *out = s_old;
    return;
  }

  // Relax dims.
  std::vector<DimensionHandle> dims(rank);
  for (int i = 0; i < rank; ++i) {
    Relax(Dim(s_old, i), Dim(s_new, i), &dims[i]);
  }
  ForgetMerges();
  *out = MakeShape(dims);
}

Status InferenceContext::Merge(ShapeHandle s0, ShapeHandle s1,
                               ShapeHandle* out) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_26(mht_26_v, 759, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Merge");

  if (s0.SameHandle(s1)) {
    *out = s0;
    return Status::OK();
  } else if (!RankKnown(s1)) {
    *out = s0;
    merged_shapes_.emplace_back(s0, s1);
    return Status::OK();
  } else if (!RankKnown(s0)) {
    *out = s1;
    merged_shapes_.emplace_back(s0, s1);
    return Status::OK();
  }

  const int32_t rank = Rank(s0);
  if (rank != Rank(s1)) {
    *out = nullptr;
    return errors::InvalidArgument("Shapes must be equal rank, but are ", rank,
                                   " and ", Rank(s1));
  }

  bool return_s0 = true;
  bool return_s1 = true;
  for (int i = 0; i < rank; ++i) {
    auto d0 = Dim(s0, i);
    auto d1 = Dim(s1, i);
    if (d0.SameHandle(d1)) continue;

    auto v0 = Value(d0);
    auto v1 = Value(d1);
    if (v0 == kUnknownDim) {
      if (v1 != kUnknownDim) {
        return_s0 = false;
      }
    } else if (v1 == kUnknownDim) {
      return_s1 = false;
    } else if (v0 != v1) {
      *out = nullptr;
      return errors::InvalidArgument(
          "Dimension ", i, " in both shapes must be equal, but are ", Value(d0),
          " and ", Value(d1), ". Shapes are ", DebugString(s0), " and ",
          DebugString(s1), ".");
    }
  }

  merged_shapes_.emplace_back(s0, s1);

  if (return_s0 || return_s1) {
    *out = return_s0 ? s0 : s1;
    return Status::OK();
  }

  // Merge dims.
  std::vector<DimensionHandle> dims(rank, nullptr);
  for (int i = 0; i < rank; ++i) {
    // Invariant for merge was checked earlier, so CHECK is ok.
    TF_CHECK_OK(Merge(Dim(s0, i), Dim(s1, i), &dims[i]));
  }

  Status s = ReturnCreatedShape(dims, out);
  if (s.ok()) {
    // Merge the new shape with s0. Since s0 and s1 are merged, this implies
    // that s1 and out are also merged.
    merged_shapes_.emplace_back(s0, *out);
  }
  return s;
}

Status InferenceContext::Subshape(ShapeHandle s, int64_t start,
                                  ShapeHandle* out) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_27(mht_27_v, 831, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Subshape");

  return Subshape(s, start, std::numeric_limits<int64_t>::max() /* end */, out);
}

Status InferenceContext::Subshape(ShapeHandle s, int64_t start, int64_t end,
                                  ShapeHandle* out) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_28(mht_28_v, 839, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Subshape");

  return Subshape(s, start, end, 1 /* stride */, out);
}

Status InferenceContext::Subshape(ShapeHandle s, int64_t start, int64_t end,
                                  int64_t stride, ShapeHandle* out) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_29(mht_29_v, 847, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Subshape");

  int64_t start_in = start;
  int64_t end_in = end;

  const int32_t rank = Rank(s);
  if (start == 0 && stride == 1 &&
      ((RankKnown(s) && end >= rank) ||
       end == std::numeric_limits<int64_t>::max())) {
    *out = s;
    return Status::OK();
  }
  if (!RankKnown(s)) {
    return ReturnUnknownShape(out);
  }

  if (start > rank) start = rank;
  if (end > rank) end = rank;

  if (stride < 0 && start == rank) --start;

  if (start < 0) {
    start = rank + start;
    if (start < 0) {
      *out = nullptr;
      return errors::InvalidArgument("Subshape start out of bounds: ", start_in,
                                     ", for shape with rank ", rank);
    }
  }

  if (end < 0) {
    end = rank + end;
    if (end < 0) {
      *out = nullptr;
      return errors::InvalidArgument("Subshape end out of bounds: ", end_in,
                                     ", for shape with rank ", rank);
    }
  }
  if (stride > 0 && start > end) {
    *out = nullptr;
    return errors::InvalidArgument(
        "Subshape must have computed start <= end, but is ", start, " and ",
        end, " (computed from start ", start_in, " and end ", end_in,
        " over shape with rank ", rank, ")");
  } else if (stride < 0 && start < end) {
    *out = nullptr;
    return errors::InvalidArgument(
        "Subshape must have computed start >= end since stride is negative, "
        "but is ",
        start, " and ", end, " (computed from start ", start_in, " and end ",
        end_in, " over shape with rank ", rank, " and stride", stride, ")");
  }

  std::vector<DimensionHandle> dims;
  for (int i = start; stride > 0 ? i < end : i > end; i += stride) {
    dims.push_back(Dim(s, i));
  }
  return ReturnCreatedShape(dims, out);
}

Status InferenceContext::Concatenate(ShapeHandle s1, ShapeHandle s2,
                                     ShapeHandle* out) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_30(mht_30_v, 910, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Concatenate");

  if (!RankKnown(s1) || !RankKnown(s2)) {
    return ReturnUnknownShape(out);
  }
  const int32_t s1_rank = Rank(s1);
  const int32_t s2_rank = Rank(s2);
  const int32_t rank = s1_rank + s2_rank;
  std::vector<DimensionHandle> dims;
  dims.reserve(rank);
  for (int i = 0; i < s1_rank; ++i) dims.push_back(Dim(s1, i));
  for (int i = 0; i < s2_rank; ++i) dims.push_back(Dim(s2, i));
  return ReturnCreatedShape(dims, out);
}

Status InferenceContext::ReplaceDim(ShapeHandle s, int64_t dim_index_in,
                                    DimensionHandle new_dim, ShapeHandle* out) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_31(mht_31_v, 928, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::ReplaceDim");

  if (!RankKnown(s)) {
    return ReturnUnknownShape(out);
  }
  int64_t dim_index = dim_index_in;
  if (dim_index < 0) {
    dim_index = s->dims_.size() + dim_index;
  }
  if (!FastBoundsCheck(dim_index, s->dims_.size())) {
    *out = nullptr;
    return errors::InvalidArgument("Out of range dim_index ", dim_index_in,
                                   " for shape with ", s->dims_.size(),
                                   " dimensions");
  }
  std::vector<DimensionHandle> dims(s->dims_);
  dims[dim_index] = new_dim;
  return ReturnCreatedShape(dims, out);
}

ShapeHandle InferenceContext::MakeShape(
    const std::vector<DimensionHandle>& dims) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_32(mht_32_v, 951, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::MakeShape");

  return shape_manager_.MakeShape(dims);
}

ShapeHandle InferenceContext::MakeShape(
    std::initializer_list<DimensionOrConstant> dims) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_33(mht_33_v, 959, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::MakeShape");

  std::vector<DimensionHandle> dims_actual;
  dims_actual.reserve(dims.size());
  for (const DimensionOrConstant& d : dims) {
    dims_actual.push_back(MakeDim(d));
  }

  return shape_manager_.MakeShape(dims_actual);
}

ShapeHandle InferenceContext::UnknownShape() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_34(mht_34_v, 972, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::UnknownShape");

  return shape_manager_.UnknownShape();
}

ShapeHandle InferenceContext::UnknownShapeOfRank(int64_t rank) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_35(mht_35_v, 979, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::UnknownShapeOfRank");

  CHECK_LE(rank, kint32max) << "rank must be less than kint32max";
  if (rank == kUnknownRank) {
    return UnknownShape();
  }
  CHECK_GE(rank, 0) << "rank must not be negative";
  std::vector<DimensionHandle> dims(rank);
  for (int32_t i = 0; i < rank; ++i) {
    dims[i] = UnknownDim();
  }
  return MakeShape(dims);
}

ShapeHandle InferenceContext::Scalar() {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_36(mht_36_v, 995, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Scalar");
 return MakeShape({}); }

ShapeHandle InferenceContext::Vector(DimensionOrConstant dim) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_37(mht_37_v, 1000, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Vector");

  return MakeShape({dim});
}

ShapeHandle InferenceContext::Matrix(DimensionOrConstant dim1,
                                     DimensionOrConstant dim2) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_38(mht_38_v, 1008, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Matrix");

  return MakeShape({dim1, dim2});
}

Status InferenceContext::MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
    int input_idx, ShapeHandle* out) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_39(mht_39_v, 1016, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::MakeShapeFromShapeTensorTreatScalarAsUnknownShape");

  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(WithRankAtMost(input(input_idx), 1, &input_shape));

  request_input_tensor_as_partial_shape(input_idx);
  const int input_tensors_as_shapes_size = input_tensors_as_shapes_.size();
  if (input_idx < input_tensors_as_shapes_size &&
      input_tensors_as_shapes_[input_idx].IsSet() &&
      RankKnown(input_tensors_as_shapes_[input_idx])) {
    *out = input_tensors_as_shapes_[input_idx];
    return Status::OK();
  }

  return InternalMakeShapeFromTensor(
      true /* treat_unknown_scalar_tensor_as_unknown_shape */,
      input_tensor(input_idx), input_shape, out);
}

Status InferenceContext::MakeShapeFromShapeTensor(int input_idx,
                                                  ShapeHandle* out) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_40(mht_40_v, 1038, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::MakeShapeFromShapeTensor");

  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(WithRank(input(input_idx), 1, &input_shape));

  request_input_tensor_as_partial_shape(input_idx);
  const int input_tensors_as_shapes_size = input_tensors_as_shapes_.size();
  if (input_idx < input_tensors_as_shapes_size &&
      input_tensors_as_shapes_[input_idx].IsSet() &&
      RankKnown(input_tensors_as_shapes_[input_idx])) {
    *out = input_tensors_as_shapes_[input_idx];
    return Status::OK();
  }

  return InternalMakeShapeFromTensor(
      false /* treat_unknown_scalar_tensor_as_unknown_shape */,
      input_tensor(input_idx), input_shape, out);
}

Status InferenceContext::MakeShapeFromTensor(const Tensor* t,
                                             ShapeHandle tensor_shape,
                                             ShapeHandle* out) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_41(mht_41_v, 1061, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::MakeShapeFromTensor");

  return InternalMakeShapeFromTensor(
      false /* treat_unknown_scalar_tensor_as_unknown_shape */, t, tensor_shape,
      out);
}

Status InferenceContext::InternalMakeShapeFromTensor(
    bool treat_unknown_scalar_tensor_as_unknown_shape, const Tensor* t,
    ShapeHandle tensor_shape, ShapeHandle* out) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_42(mht_42_v, 1072, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::InternalMakeShapeFromTensor");

  // Only callers who have set
  if (!treat_unknown_scalar_tensor_as_unknown_shape) {
    TF_RETURN_IF_ERROR(WithRank(tensor_shape, 1, &tensor_shape));
  }
  if (t == nullptr) {
    // This is guarded by the check above.
    if (Rank(tensor_shape) == 0) {
      return ReturnUnknownShape(out);
    }
    // Shape tensor is not known, but if the shape of the shape tensor is then
    // the right number of unknown dims can be created.
    DimensionHandle shape_dim = Dim(tensor_shape, 0);
    if (!ValueKnown(shape_dim)) {
      return ReturnUnknownShape(out);
    }
    const auto num_dims = Value(shape_dim);
    // TODO(mihaimaruseac): Should be `TensorShape::MaxDimensions()` as we are
    // not able to materialize shapes with more than this number of dimensions
    // but then shape inference would fail for operations such as
    // `tf.range`/`tf.ones`, etc. where the shape is not really materialized,
    // only used during the inference. Hence, just prevent doing a `reserve`
    // with a very large argument.
    const int64_t max_dimensions = 1 << 25;
    if (num_dims >= max_dimensions) {
      return errors::Internal(
          "Cannot create a tensor with ", num_dims,
          " dimensions, as these would be more than maximum of ",
          max_dimensions);
    }
    std::vector<DimensionHandle> dims;
    dims.reserve(num_dims);
    for (int i = 0; i < num_dims; i++) dims.push_back(UnknownDim());
    return ReturnCreatedShape(dims, out);
  }

  if (t->shape().dims() == 0) {
    if (t->dtype() == DataType::DT_INT32) {
      auto flat_t = t->scalar<int32>();
      if (flat_t() != -1) {
        *out = nullptr;
        return errors::InvalidArgument(
            "Input tensor must be rank 1, or if its rank 0 it must have value "
            "-1 "
            "(representing an unknown shape).  Saw value: ",
            flat_t());
      }
      return ReturnUnknownShape(out);
    } else if (t->dtype() == DataType::DT_INT64) {
      auto flat_t = t->scalar<int64_t>();
      if (flat_t() != -1) {
        *out = nullptr;
        return errors::InvalidArgument(
            "Input tensor must be rank 1, or if its rank 0 it must have value "
            "-1 "
            "(representing an unknown shape).  Saw value: ",
            flat_t());
      }
      return ReturnUnknownShape(out);
    } else {
      *out = nullptr;
      return errors::InvalidArgument(
          "Input tensor must be int32 or int64, but was ",
          DataTypeString(t->dtype()));
    }
  }

  if (t->shape().dims() != 1) {
    *out = nullptr;
    return errors::InvalidArgument(
        "Input tensor must be rank 1, but was rank ", t->shape().dims(), ".",
        ((t->shape().dims() == 0)
             ? "If it is rank 0 rank 0 it must have statically known value -1 "
               "(representing an unknown shape). "
             : " "),
        "Saw tensor shape ", t->shape().DebugString());
  }
  std::vector<DimensionHandle> dims;
  if (t->dtype() == DataType::DT_INT32) {
    auto flat_t = t->flat<int32>();
    for (int i = 0; i < flat_t.size(); ++i) {
      const int32_t val = flat_t(i);
      if (val < -1) {
        return errors::InvalidArgument(
            "Invalid value in tensor used for shape: ", val);
      }
      // -1 will become an unknown dim.
      dims.push_back(MakeDim(val));
    }
  } else if (t->dtype() == DataType::DT_INT64) {
    auto flat_t = t->flat<int64_t>();
    for (int i = 0; i < flat_t.size(); ++i) {
      const int64_t val = flat_t(i);
      if (val < -1) {
        return errors::InvalidArgument(
            "Invalid value in tensor used for shape: ", val);
      }
      // -1 will become an unknown dim.
      dims.push_back(MakeDim(val));
    }
  } else {
    *out = nullptr;
    return errors::InvalidArgument(
        "Input tensor must be int32 or int64, but was ",
        DataTypeString(t->dtype()));
  }

  return ReturnCreatedShape(dims, out);
}

Status InferenceContext::MakeShapeFromPartialTensorShape(
    const PartialTensorShape& partial_shape, ShapeHandle* out) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_43(mht_43_v, 1186, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::MakeShapeFromPartialTensorShape");

  *out = nullptr;
  if (partial_shape.dims() == -1) {
    return ReturnUnknownShape(out);
  }
  const int num_dims = partial_shape.dims();
  std::vector<DimensionHandle> dims(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    // -1 is unknown in PartialTensorShape and in InferenceContext, so this size
    // can be passed directly to MakeDim.
    dims[i] = MakeDim(partial_shape.dim_size(i));
  }
  return ReturnCreatedShape(dims, out);
}

Status InferenceContext::MakeShapeFromTensorShape(const TensorShape& shape,
                                                  ShapeHandle* out) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_44(mht_44_v, 1205, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::MakeShapeFromTensorShape");

  return MakeShapeFromPartialTensorShape(PartialTensorShape(shape.dim_sizes()),
                                         out);
}

Status InferenceContext::MakeShapeFromShapeProto(const TensorShapeProto& proto,
                                                 ShapeHandle* out) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_45(mht_45_v, 1214, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::MakeShapeFromShapeProto");

  *out = nullptr;
  TF_RETURN_IF_ERROR(PartialTensorShape::IsValidShape(proto));
  PartialTensorShape partial_shape(proto);
  return MakeShapeFromPartialTensorShape(partial_shape, out);
}

Status InferenceContext::GetScalarFromTensor(const Tensor* t, int64_t* val) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_46(mht_46_v, 1224, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::GetScalarFromTensor");

  // Caller must ensure that <t> is not NULL.
  const int rank = t->dims();
  if (rank != 0) {
    return errors::InvalidArgument("Input must be scalar but has rank ", rank);
  }

  if (t->dtype() == DataType::DT_INT32) {
    *val = t->scalar<int32>()();
    return Status::OK();
  } else if (t->dtype() == DataType::DT_INT64) {
    *val = t->scalar<int64_t>()();
    return Status::OK();
  } else {
    return errors::InvalidArgument("Scalar input must be int32 or int64.");
  }
}

Status InferenceContext::GetScalarFromTensor(const Tensor* t, int64_t idx,
                                             int64_t* val) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_47(mht_47_v, 1246, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::GetScalarFromTensor");

  // Caller must ensure that <t> is not NULL.
  const int rank = t->dims();
  if (rank != 1) {
    return errors::InvalidArgument("Input must be 1D but has rank ", rank);
  }

  if (t->dtype() == DataType::DT_INT32) {
    auto flat_t = t->flat<int32>();
    if (idx < 0 || idx >= flat_t.size()) {
      return errors::InvalidArgument("Invalid index ", idx,
                                     " for Tensor of size ", flat_t.size());
    }
    *val = flat_t(idx);
    return Status::OK();
  } else if (t->dtype() == DataType::DT_INT64) {
    auto flat_t = t->flat<int64_t>();
    if (idx < 0 || idx >= flat_t.size()) {
      return errors::InvalidArgument("Invalid index ", idx,
                                     " for Tensor of size ", flat_t.size());
    }
    *val = flat_t(idx);
    return Status::OK();
  } else {
    return errors::InvalidArgument("Tensor input must be int32 or int64.");
  }
}

// Returns a new dimension whose value is given by a scalar input tensor.
Status InferenceContext::MakeDimForScalarInput(int idx, DimensionHandle* out) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_48(mht_48_v, 1278, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::MakeDimForScalarInput");

  int64_t val;
  const Tensor* t = input_tensor(idx);
  if (t == nullptr) {
    *out = UnknownDim();
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(GetScalarFromTensor(t, &val));
  if (val < 0) {
    return errors::InvalidArgument("Dimension size, given by scalar input ",
                                   idx, ", must be non-negative but is ", val);
  }
  *out = MakeDim(val);
  return Status::OK();
}

Status InferenceContext::MakeDimForScalarInputWithNegativeIndexing(
    int idx, int input_rank, DimensionHandle* out) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_49(mht_49_v, 1298, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::MakeDimForScalarInputWithNegativeIndexing");

  int64_t val;
  const Tensor* t = input_tensor(idx);
  if (t == nullptr) {
    *out = UnknownDim();
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(GetScalarFromTensor(t, &val));
  if (val < 0) {
    if (input_rank < 0) {
      *out = UnknownDim();
      return Status::OK();
    } else if (val + input_rank < 0) {
      return errors::InvalidArgument("Dimension size, given by scalar input ",
                                     val, " must be in range [-", input_rank,
                                     ", ", input_rank, ")");
    } else {
      val += input_rank;
    }
  } else if (input_rank >= 0 && val >= input_rank) {
    return errors::InvalidArgument("Dimension size, given by scalar input ",
                                   val, " must be in range [-", input_rank,
                                   ", ", input_rank, ")");
  }
  *out = MakeDim(val);
  return Status::OK();
}

Status InferenceContext::Divide(DimensionHandle dividend,
                                DimensionOrConstant divisor,
                                bool evenly_divisible, DimensionHandle* out) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_50(mht_50_v, 1331, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Divide");

  const int64_t divisor_value = Value(divisor);
  if (divisor_value == 1) {
    *out = dividend;
  } else if (!ValueKnown(dividend) ||
             (divisor.dim.IsSet() && !ValueKnown(divisor.dim))) {
    *out = UnknownDim();
  } else {
    const int64_t v = Value(dividend);
    if (divisor_value <= 0) {
      return errors::InvalidArgument("Divisor must be positive but is ",
                                     divisor_value);
    }
    if (evenly_divisible && (v % divisor_value) != 0) {
      return errors::InvalidArgument(
          "Dimension size must be evenly divisible by ", divisor_value,
          " but is ", v);
    }
    *out = MakeDim(v / divisor_value);
  }
  return Status::OK();
}

Status InferenceContext::Add(DimensionHandle first, DimensionOrConstant second,
                             DimensionHandle* out) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_51(mht_51_v, 1358, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Add");

  const int64_t first_value = Value(first);
  const int64_t second_value = Value(second);
  // Special cases.
  if (first_value == 0) {
    *out = MakeDim(second);
  } else if (second_value == 0) {
    *out = first;
  } else if (first_value == kUnknownDim || second_value == kUnknownDim) {
    *out = UnknownDim();
  } else {
    // Invariant: Both values are known and positive. Still in run-time we can
    // get pair of values which cannot be store in output. Check below will
    // report error. We still need to avoid undefined behavior of signed
    // overflow and use unsigned addition.
    const int64_t sum = static_cast<uint64>(first_value) + second_value;
    if (sum < 0) {
      return errors::InvalidArgument("Dimension size overflow from adding ",
                                     first_value, " and ", second_value);
    }
    *out = MakeDim(sum);
  }
  return Status::OK();
}

Status InferenceContext::Subtract(DimensionHandle first,
                                  DimensionOrConstant second,
                                  DimensionHandle* out) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_52(mht_52_v, 1388, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Subtract");

  const int64_t first_value = Value(first);
  const int64_t second_value = Value(second);
  // Special cases.
  if (second_value == 0) {
    *out = first;
  } else if (first_value == kUnknownDim || second_value == kUnknownDim) {
    *out = UnknownDim();
  } else {
    // Invariant: Both values are known, first_value is non-negative, and
    // second_value is positive.
    if (first_value < second_value) {
      return errors::InvalidArgument(
          "Negative dimension size caused by subtracting ", second_value,
          " from ", first_value);
    }
    *out = MakeDim(first_value - second_value);
  }
  return Status::OK();
}

Status InferenceContext::Multiply(DimensionHandle first,
                                  DimensionOrConstant second,
                                  DimensionHandle* out) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_53(mht_53_v, 1414, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Multiply");

  const int64_t first_value = Value(first);
  const int64_t second_value = Value(second);
  // Special cases.
  if (first_value == 0) {
    *out = first;
  } else if (second_value == 0) {
    *out = MakeDim(second);
  } else if (first_value == 1) {
    *out = MakeDim(second);
  } else if (second_value == 1) {
    *out = first;
  } else if (first_value == kUnknownDim || second_value == kUnknownDim) {
    *out = UnknownDim();
  } else {
    // Invariant: Both values are known and greater than 1.
    const int64_t product = MultiplyWithoutOverflow(first_value, second_value);
    if (product < 0) {
      return errors::InvalidArgument(
          "Negative dimension size caused by overflow when multiplying ",
          first_value, " and ", second_value);
    }
    *out = MakeDim(product);
  }
  return Status::OK();
}

Status InferenceContext::Min(DimensionHandle first, DimensionOrConstant second,
                             DimensionHandle* out) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_54(mht_54_v, 1445, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Min");

  const int64_t first_value = Value(first);
  const int64_t second_value = Value(second);
  if (first_value == 0) {
    *out = first;
  } else if (second_value == 0) {
    *out = MakeDim(second);
  } else if (first_value == kUnknownDim || second_value == kUnknownDim) {
    *out = UnknownDim();
  } else {
    if (first_value <= second_value) {
      *out = first;
    } else {
      *out = MakeDim(second);
    }
  }
  return Status::OK();
}

Status InferenceContext::Max(DimensionHandle first, DimensionOrConstant second,
                             DimensionHandle* out) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_55(mht_55_v, 1468, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::Max");

  const int64_t first_value = Value(first);
  const int64_t second_value = Value(second);
  if (first_value == kUnknownDim || second_value == kUnknownDim) {
    *out = UnknownDim();
  } else {
    if (first_value >= second_value) {
      *out = first;
    } else {
      *out = MakeDim(second);
    }
  }
  return Status::OK();
}

Status InferenceContext::AttachContext(const Status& status) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_56(mht_56_v, 1486, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::AttachContext");

  std::vector<string> input_shapes;
  input_shapes.reserve(inputs_.size());
  for (const ShapeHandle& input_shape : inputs_) {
    input_shapes.emplace_back(DebugString(input_shape));
  }

  // Add information about the input tensors and partial tensor shapes used.
  std::vector<string> input_from_tensors_str;
  std::vector<string> input_from_tensors_as_shape_str;
  input_from_tensors_as_shape_str.reserve(inputs_.size());
  for (int i = 0, end = inputs_.size(); i < end; ++i) {
    const int input_tensors_as_shapes_size = input_tensors_as_shapes_.size();
    const int input_tensors_size = input_tensors_.size();
    if (requested_input_tensor_as_partial_shape_[i] &&
        i < input_tensors_as_shapes_size &&
        input_tensors_as_shapes_[i].IsSet() &&
        RankKnown(input_tensors_as_shapes_[i])) {
      input_from_tensors_as_shape_str.push_back(strings::StrCat(
          "input[", i, "] = ", DebugString(input_tensors_as_shapes_[i])));
    } else if (requested_input_tensor_[i] && i < input_tensors_size &&
               input_tensors_[i] != nullptr) {
      input_from_tensors_str.push_back(strings::StrCat(
          "input[", i, "] = <",
          input_tensors_[i]->SummarizeValue(256 /* max_values */), ">"));
    }
  }

  string error_context = strings::StrCat(
      " for '", attrs_.SummarizeNode(),
      "' with input shapes: ", absl::StrJoin(input_shapes, ", "));
  if (!input_from_tensors_str.empty()) {
    strings::StrAppend(&error_context, " and with computed input tensors: ",
                       absl::StrJoin(input_from_tensors_str, ", "));
  }
  if (!input_from_tensors_as_shape_str.empty()) {
    strings::StrAppend(&error_context,
                       " and with input tensors computed as partial shapes: ",
                       absl::StrJoin(input_from_tensors_as_shape_str, ","));
  }

  strings::StrAppend(&error_context, ".");
  return errors::CreateWithUpdatedMessage(
      status, strings::StrCat(status.error_message(), error_context));
}

bool InferenceContext::MergeHandleShapesAndTypes(
    const std::vector<ShapeAndType>& shapes_and_types,
    std::vector<ShapeAndType>* to_update) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_57(mht_57_v, 1537, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::MergeHandleShapesAndTypes");

  if (shapes_and_types.size() != to_update->size()) {
    return false;
  }
  std::vector<ShapeAndType> new_values(shapes_and_types.size());
  bool refined = false;
  for (int i = 0, end = shapes_and_types.size(); i < end; ++i) {
    const ShapeAndType& existing = (*to_update)[i];
    if (shapes_and_types[i].dtype == existing.dtype) {
      new_values[i].dtype = existing.dtype;
    } else {
      if (existing.dtype != DT_INVALID) {
        return false;
      } else {
        new_values[i].dtype = shapes_and_types[i].dtype;
        refined = true;
      }
    }
    if (!Merge(existing.shape, shapes_and_types[i].shape, &new_values[i].shape)
             .ok()) {
      // merge failed, ignore the new value.
      new_values[i].shape = existing.shape;
    }
    if (!existing.shape.SameHandle(new_values[i].shape)) {
      refined = true;
    }
  }
  if (!refined) {
    return false;
  }
  for (int i = 0, end = new_values.size(); i < end; ++i) {
    (*to_update)[i] = new_values[i];
  }
  return true;
}

bool InferenceContext::MergeOutputHandleShapesAndTypes(
    int idx, const std::vector<ShapeAndType>& shapes_and_types) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_58(mht_58_v, 1577, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::MergeOutputHandleShapesAndTypes");

  if (output_handle_shapes_and_types_[idx] == nullptr) {
    output_handle_shapes_and_types_[idx].reset(
        new std::vector<ShapeAndType>(shapes_and_types));
    return true;
  }
  return MergeHandleShapesAndTypes(shapes_and_types,
                                   output_handle_shapes_and_types_[idx].get());
}

bool InferenceContext::MergeInputHandleShapesAndTypes(
    int idx, const std::vector<ShapeAndType>& shapes_and_types) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_59(mht_59_v, 1591, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::MergeInputHandleShapesAndTypes");

  if (input_handle_shapes_and_types_[idx] == nullptr) {
    input_handle_shapes_and_types_[idx].reset(
        new std::vector<ShapeAndType>(shapes_and_types));
    return true;
  }
  return MergeHandleShapesAndTypes(shapes_and_types,
                                   input_handle_shapes_and_types_[idx].get());
}

bool InferenceContext::RelaxHandleShapesAndMergeTypes(
    const std::vector<ShapeAndType>& shapes_and_types,
    std::vector<ShapeAndType>* to_update) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_60(mht_60_v, 1606, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::RelaxHandleShapesAndMergeTypes");

  if (shapes_and_types.size() != to_update->size()) {
    return false;
  }
  std::vector<ShapeAndType> new_values(shapes_and_types.size());
  for (int i = 0, end = shapes_and_types.size(); i < end; ++i) {
    const ShapeAndType& existing = (*to_update)[i];
    if (shapes_and_types[i].dtype == existing.dtype) {
      new_values[i].dtype = existing.dtype;
    } else {
      if (existing.dtype != DT_INVALID) {
        return false;
      } else {
        new_values[i].dtype = shapes_and_types[i].dtype;
      }
    }
    Relax(existing.shape, shapes_and_types[i].shape, &new_values[i].shape);
  }
  to_update->swap(new_values);
  return true;
}

bool InferenceContext::RelaxOutputHandleShapesAndMergeTypes(
    int idx, const std::vector<ShapeAndType>& shapes_and_types) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_61(mht_61_v, 1632, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::RelaxOutputHandleShapesAndMergeTypes");

  if (output_handle_shapes_and_types_[idx] == nullptr) {
    output_handle_shapes_and_types_[idx].reset(
        new std::vector<ShapeAndType>(shapes_and_types));
    return true;
  }
  return RelaxHandleShapesAndMergeTypes(
      shapes_and_types, output_handle_shapes_and_types_[idx].get());
}

bool InferenceContext::RelaxInputHandleShapesAndMergeTypes(
    int idx, const std::vector<ShapeAndType>& shapes_and_types) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_62(mht_62_v, 1646, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::RelaxInputHandleShapesAndMergeTypes");

  if (input_handle_shapes_and_types_[idx] == nullptr) {
    input_handle_shapes_and_types_[idx].reset(
        new std::vector<ShapeAndType>(shapes_and_types));
    return true;
  }
  return RelaxHandleShapesAndMergeTypes(
      shapes_and_types, input_handle_shapes_and_types_[idx].get());
}

// -----------------------------------------------------------------------------
// ShapeManager
// -----------------------------------------------------------------------------
InferenceContext::ShapeManager::ShapeManager() {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_63(mht_63_v, 1662, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::ShapeManager::ShapeManager");
}
InferenceContext::ShapeManager::~ShapeManager() {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_64(mht_64_v, 1666, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::ShapeManager::~ShapeManager");

  for (auto* s : all_shapes_) delete s;
  for (auto* d : all_dims_) delete d;
}

ShapeHandle InferenceContext::ShapeManager::MakeShape(
    const std::vector<DimensionHandle>& dims) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_65(mht_65_v, 1675, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::ShapeManager::MakeShape");

  all_shapes_.push_back(new Shape(dims));
  return all_shapes_.back();
}

ShapeHandle InferenceContext::ShapeManager::UnknownShape() {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSshape_inferenceDTcc mht_66(mht_66_v, 1683, "", "./tensorflow/core/framework/shape_inference.cc", "InferenceContext::ShapeManager::UnknownShape");

  all_shapes_.push_back(new Shape());
  return all_shapes_.back();
}

}  // namespace shape_inference
}  // namespace tensorflow
