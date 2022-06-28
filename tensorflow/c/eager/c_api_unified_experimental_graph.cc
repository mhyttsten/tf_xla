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
class MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc {
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
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/graph_function.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::dyn_cast;
using tensorflow::string;
using tensorflow::gtl::ArraySlice;

namespace tensorflow {
namespace tracing {
namespace graph {

class GraphContext;
class GraphOperation;
class GraphTensor;

auto& kUnknownDim = shape_inference::InferenceContext::kUnknownDim;
auto& kUnknownRank = shape_inference::InferenceContext::kUnknownRank;

// GraphTensor wraps a `TF_Output`, i.e. a pointer to TF_Operation and the index
// into the list of outputs for the operation.
class GraphTensor : public TracingTensorHandle {
 public:
  explicit GraphTensor(TF_Output output, TF_Graph* graph)
      : TracingTensorHandle(kGraph), output_(output), graph_(graph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_0(mht_0_v, 226, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "GraphTensor");
}

  tensorflow::DataType DataType() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_1(mht_1_v, 231, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "DataType");

    return static_cast<tensorflow::DataType>(TF_OperationOutputType(output_));
  }

  tensorflow::Status Shape(
      tensorflow::PartialTensorShape* shape) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_2(mht_2_v, 239, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "Shape");

    DCHECK(shape != nullptr);
    TF_Status status;
    int num_dims = TF_GraphGetTensorNumDims(graph_, output_, &status);
    DCHECK_GE(num_dims, -1);
    TF_RETURN_IF_ERROR(StatusFromTF_Status(&status));
    if (num_dims == kUnknownRank) {
      return Status::OK();
    }

    std::vector<int64_t> dims(num_dims, kUnknownDim);
    TF_GraphGetTensorShape(graph_, output_,
                           reinterpret_cast<int64_t*>(dims.data()), num_dims,
                           &status);
    TF_RETURN_IF_ERROR(StatusFromTF_Status(&status));
    TF_RETURN_IF_ERROR(tensorflow::TensorShapeUtils::MakeShape(dims, shape));

    return Status::OK();
  }

  TF_Output output_;

  // For LLVM style RTTI.
  static bool classof(const AbstractTensorHandle* ptr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_3(mht_3_v, 265, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "classof");

    return ptr->getKind() == kGraph;
  }

 private:
  TF_Graph* graph_;  // For shape inference.
};

// GraphOperation wraps and populates a TF_OperationDescription.
class GraphOperation : public TracingOperation {
 public:
  explicit GraphOperation(TF_Graph* g) : TracingOperation(kGraph), g_(g) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_4(mht_4_v, 279, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "GraphOperation");
}
  void Release() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_5(mht_5_v, 283, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "Release");
 delete this; }
  Status Reset(const char* op, const char* raw_device_name) override {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   mht_6_v.push_back("raw_device_name: \"" + (raw_device_name == nullptr ? std::string("nullptr") : std::string((char*)raw_device_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_6(mht_6_v, 289, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "Reset");

    if (op_) {
      return errors::FailedPrecondition("Reset called on already built op.");
    }
    if (raw_device_name) {
      device_name_ = raw_device_name;
    }
    op_type_ = op;
    return Status::OK();
  }
  Status SetOpName(const char* const op_name) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_7(mht_7_v, 302, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetOpName");

    if (op_) {
      return errors::FailedPrecondition(
          "SetOpName called on already built op.");
    }
    if (op_type_.empty()) {
      return errors::FailedPrecondition(
          "GraphOperation::Reset must be called before calling SetOpName.");
    }
    // TODO(b/145674566): We use Graph::NewName to get a unique name here but
    // this may not be consistent with python's naming policy.
    mutex_lock l(g_->mu);
    op_.reset(new TF_OperationDescription(g_, op_type_.c_str(),
                                          g_->graph.NewName(op_name).c_str()));
    return Status::OK();
  }
  const string& Name() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_8(mht_8_v, 321, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "Name");
 return op_type_; }
  const string& DeviceName() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_9(mht_9_v, 325, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "DeviceName");
 return device_name_; }

  Status SetDeviceName(const char* name) override {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_10(mht_10_v, 331, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetDeviceName");

    // TODO(srbs): Implement this.
    device_name_ = name;
    return Status::OK();
  }

  Status AddInput(AbstractTensorHandle* input) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_11(mht_11_v, 340, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "AddInput");

    GraphTensor* t = dyn_cast<GraphTensor>(input);
    if (!t) {
      return tensorflow::errors::InvalidArgument(
          "Unable to cast input to GraphTensor");
    }
    TF_AddInput(op_.get(), t->output_);
    return Status::OK();
  }
  Status AddInputList(absl::Span<AbstractTensorHandle* const> inputs) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_12(mht_12_v, 352, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "AddInputList");

    std::vector<TF_Output> tf_outputs(inputs.size());
    for (int i = 0; i < inputs.size(); i++) {
      GraphTensor* t = dyn_cast<GraphTensor>(inputs[i]);
      if (!t) {
        return tensorflow::errors::InvalidArgument(
            "Unable to cast input to GraphTensor");
      }
      tf_outputs[i] = t->output_;
    }
    TF_AddInputList(op_.get(), tf_outputs.data(), tf_outputs.size());
    return Status::OK();
  }
  Status Execute(absl::Span<AbstractTensorHandle*> retvals,
                 int* num_retvals) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_13(mht_13_v, 369, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "Execute");

    auto* tf_opdesc = op_.release();
    if (tf_opdesc == nullptr) {
      return errors::InvalidArgument("AbstractOp is incomplete.");
    }
    TF_Status* s = TF_NewStatus();
    auto* operation = TF_FinishOperation(tf_opdesc, s);
    TF_RETURN_IF_ERROR(StatusFromTF_Status(s));
    TF_DeleteStatus(s);
    *num_retvals = TF_OperationNumOutputs(operation);
    for (int i = 0; i < *num_retvals; ++i) {
      retvals[i] = new GraphTensor({operation, i}, g_);
    }
    return Status::OK();
  }

  Status SetAttrString(const char* attr_name, const char* data,
                       size_t length) override {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_14_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_14(mht_14_v, 391, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetAttrString");

    tensorflow::StringPiece s(data, length);
    op_->node_builder.Attr(attr_name, s);
    return Status::OK();
  }
  Status SetAttrInt(const char* attr_name, int64_t value) override {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_15(mht_15_v, 400, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetAttrInt");

    op_->node_builder.Attr(attr_name, static_cast<int64_t>(value));
    return Status::OK();
  }
  Status SetAttrFloat(const char* attr_name, float value) override {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_16(mht_16_v, 408, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetAttrFloat");

    op_->node_builder.Attr(attr_name, value);
    return Status::OK();
  }
  Status SetAttrBool(const char* attr_name, bool value) override {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_17(mht_17_v, 416, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetAttrBool");

    op_->node_builder.Attr(attr_name, value);
    return Status::OK();
  }
  Status SetAttrType(const char* const attr_name, DataType value) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_18(mht_18_v, 423, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetAttrType");

    if (!op_) {
      return Status(
          error::Code::FAILED_PRECONDITION,
          "op_type and op_name must be specified before specifying attrs.");
    }
    op_->node_builder.Attr(attr_name, value);
    return Status::OK();
  }
  Status SetAttrShape(const char* attr_name, const int64_t* dims,
                      const int num_dims) override {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_19(mht_19_v, 437, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetAttrShape");

    PartialTensorShape shape;
    if (num_dims >= 0) {
      shape = PartialTensorShape(ArraySlice<int64_t>(
          reinterpret_cast<const int64_t*>(dims), num_dims));
    }
    op_->node_builder.Attr(attr_name, shape);
    return Status::OK();
  }
  Status SetAttrFunction(const char* attr_name,
                         const AbstractOperation* value) override {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_20(mht_20_v, 451, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetAttrFunction");

    return tensorflow::errors::Unimplemented(
        "SetAttrFunction has not been implemented yet.");
  }
  Status SetAttrFunctionName(const char* attr_name, const char* value,
                             size_t length) override {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_21_v.push_back("value: \"" + (value == nullptr ? std::string("nullptr") : std::string((char*)value)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_21(mht_21_v, 461, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetAttrFunctionName");

    tensorflow::NameAttrList func_name;
    func_name.set_name(string(value, value + length));
    op_->node_builder.Attr(attr_name, func_name);
    return Status::OK();
  }
  Status SetAttrTensor(const char* attr_name,
                       AbstractTensorInterface* tensor) override {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_22(mht_22_v, 472, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetAttrTensor");

    return tensorflow::errors::Unimplemented(
        "SetAttrTensor has not been implemented yet.");
  }
  Status SetAttrStringList(const char* attr_name, const void* const* values,
                           const size_t* lengths, int num_values) override {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_23(mht_23_v, 481, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetAttrStringList");

    if (strcmp(attr_name, tensorflow::kColocationAttrName) == 0) {
      op_->colocation_constraints.clear();
      for (int i = 0; i < num_values; ++i) {
        op_->colocation_constraints.emplace(static_cast<const char*>(values[i]),
                                            lengths[i]);
      }
    } else {
      std::vector<tensorflow::StringPiece> v;
      v.reserve(num_values);
      for (int i = 0; i < num_values; ++i) {
        v.emplace_back(static_cast<const char*>(values[i]), lengths[i]);
      }
      op_->node_builder.Attr(attr_name, v);
    }
    return Status::OK();
  }
  Status SetAttrFloatList(const char* attr_name, const float* values,
                          int num_values) override {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_24(mht_24_v, 503, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetAttrFloatList");

    op_->node_builder.Attr(attr_name,
                           ArraySlice<const float>(values, num_values));
    return Status::OK();
  }
  Status SetAttrIntList(const char* attr_name, const int64_t* values,
                        int num_values) override {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_25(mht_25_v, 513, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetAttrIntList");

    op_->node_builder.Attr(
        attr_name, ArraySlice<const int64_t>(
                       reinterpret_cast<const int64_t*>(values), num_values));
    return Status::OK();
  }
  Status SetAttrTypeList(const char* attr_name, const DataType* values,
                         int num_values) override {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_26(mht_26_v, 524, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetAttrTypeList");

    op_->node_builder.Attr(attr_name,
                           ArraySlice<const DataType>(values, num_values));
    return Status::OK();
  }
  Status SetAttrBoolList(const char* attr_name, const unsigned char* values,
                         int num_values) override {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_27_v.push_back("values: \"" + (values == nullptr ? std::string("nullptr") : std::string((char*)values)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_27(mht_27_v, 535, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetAttrBoolList");

    std::unique_ptr<bool[]> b(new bool[num_values]);
    for (int i = 0; i < num_values; ++i) {
      b[i] = values[i];
    }
    op_->node_builder.Attr(attr_name,
                           ArraySlice<const bool>(b.get(), num_values));

    return Status::OK();
  }
  Status SetAttrShapeList(const char* attr_name, const int64_t** dims,
                          const int* num_dims, int num_values) override {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_28(mht_28_v, 550, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetAttrShapeList");

    std::vector<PartialTensorShape> shapes;
    shapes.reserve(num_values);
    for (int i = 0; i < num_values; ++i) {
      if (num_dims[i] < 0) {
        shapes.emplace_back();
      } else {
        shapes.emplace_back(ArraySlice<int64_t>(
            reinterpret_cast<const int64_t*>(dims[i]), num_dims[i]));
      }
    }
    op_->node_builder.Attr(attr_name, shapes);
    return Status::OK();
  }
  Status SetAttrFunctionList(
      const char* attr_name,
      absl::Span<const AbstractOperation*> values) override {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_29(mht_29_v, 570, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "SetAttrFunctionList");

    return tensorflow::errors::Unimplemented(
        "SetAttrFunctionList has not been implemented yet.");
  }
  // For LLVM style RTTI.
  static bool classof(const AbstractOperation* ptr) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_30(mht_30_v, 578, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "classof");

    return ptr->getKind() == kGraph;
  }
  ~GraphOperation() override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_31(mht_31_v, 584, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "~GraphOperation");
}

 private:
  friend class GraphContext;  // For access to op_.
  TF_Graph* g_;
  std::unique_ptr<TF_OperationDescription> op_;
  // Hold `op_type` and `op_name` till both are available since we need both
  // to build a graph operation.
  string op_type_;
  const char* op_name_ = nullptr;
  // TODO(srbs): Use this.
  string device_name_;
};

// GraphContext wraps a TF_Graph modeling a single function and manages the
// "execution" of operation, i.e. adding them to the function.
class GraphContext : public TracingContext {
 public:
  explicit GraphContext(const char* name)
      : TracingContext(kGraph),
        graph_(new TF_Graph(), TF_DeleteGraph),
        name_(name) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_32(mht_32_v, 609, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "GraphContext");
}

  void Release() override {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_33(mht_33_v, 614, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "Release");
 delete this; }

  TracingOperation* CreateOperation() override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_34(mht_34_v, 619, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "CreateOperation");

    return new GraphOperation(graph_.get());
  }

  Status AddParameter(DataType dtype, const PartialTensorShape& shape,
                      TracingTensorHandle** output) override {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_35(mht_35_v, 627, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "AddParameter");

    TracingOperationPtr operation(CreateOperation());
    TF_RETURN_IF_ERROR(operation->Reset("Placeholder", nullptr));
    TF_RETURN_IF_ERROR(
        operation->SetOpName(absl::StrCat("_input_", inputs_.size()).c_str()));
    TF_RETURN_IF_ERROR(operation->SetAttrType("dtype", dtype));
    if (!shape.unknown_rank()) {
      TF_RETURN_IF_ERROR(operation->SetAttrShape(
          "shape", reinterpret_cast<int64_t*>(shape.dim_sizes().data()),
          shape.dims()));
    }
    int num_outputs = 1;
    std::vector<AbstractTensorHandle*> outputs(num_outputs);
    TF_RETURN_IF_ERROR(operation->Execute(
        absl::Span<AbstractTensorHandle*>(outputs), &num_outputs));

    if (num_outputs != 1) {
      return errors::Internal("Expected 1 output but found ", num_outputs);
    }
    auto* t = dyn_cast<GraphTensor>(outputs[0]);
    if (!t) {
      return tensorflow::errors::InvalidArgument(
          "Unable to cast input to GraphTensor");
    }
    inputs_.push_back(t->output_);
    *output = tensorflow::down_cast<TracingTensorHandle*>(outputs[0]);
    return Status::OK();
  }

  Status Finalize(OutputList* outputs, AbstractFunction** f) override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_36(mht_36_v, 659, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "Finalize");

    std::vector<TF_Output> graph_outputs;
    graph_outputs.reserve(outputs->outputs.size());
    for (auto* abstract_output : outputs->outputs) {
      GraphTensor* output = dyn_cast<GraphTensor>(abstract_output);
      if (!output) {
        return errors::Unimplemented(
            "Returning a non-graph tensor from a function has not "
            "been implemented yet.");
      }
      graph_outputs.push_back(output->output_);
    }

    auto s = TF_NewStatus();
    auto func = TF_GraphToFunction(graph_.get(), name_.data(), 0, -1, nullptr,
                                   inputs_.size(), inputs_.data(),
                                   graph_outputs.size(), graph_outputs.data(),
                                   nullptr, nullptr, name_.data(), s);
    *f = new GraphFunction(std::move(func->fdef));
    TF_DeleteFunction(func);
    TF_RETURN_IF_ERROR(StatusFromTF_Status(s));
    TF_DeleteStatus(s);
    return Status::OK();
  }

  Status RegisterFunction(AbstractFunction* func) override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_37(mht_37_v, 687, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "RegisterFunction");

    return errors::Unimplemented(
        "Registering graph functions has not been implemented yet.");
  }

  Status RemoveFunction(const string& func) override {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_38(mht_38_v, 696, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "RemoveFunction");

    return errors::Unimplemented(
        "GraphContext::RemoveFunction has not been implemented yet.");
  }
  // For LLVM style RTTI.
  static bool classof(const AbstractContext* ptr) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_39(mht_39_v, 704, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "classof");

    return ptr->getKind() == kGraph;
  }

 private:
  std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)> graph_;
  std::vector<TF_Output> inputs_;
  string name_;
};

static TracingContext* GraphTracingFactory(const char* name, TF_Status* s) {
   std::vector<std::string> mht_40_v;
   mht_40_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_graphDTcc mht_40(mht_40_v, 718, "", "./tensorflow/c/eager/c_api_unified_experimental_graph.cc", "GraphTracingFactory");

  return new GraphContext(name);
}

// Register the tracing implemented in this file as the default tracing engine.
static bool register_tracing = [] {
  RegisterTracingEngineFactory("graphdef", GraphTracingFactory);
  SetDefaultTracingEngine("graphdef").IgnoreError();
  return true;
}();

}  // namespace graph
}  // namespace tracing
}  // namespace tensorflow
