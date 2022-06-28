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
class MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc() {
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

#include "tensorflow/core/framework/fake_input.h"

#include <vector>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace {

class FakeInputImpl {
 public:
  FakeInputImpl(const OpDef* op_def, int in_index, const NodeDef* node_def,
                NodeDefBuilder* builder);
  void SetN(int n);
  void SetDataType(DataType dt);
  void SetTypeList(DataTypeSlice dts);
  Status AddInputToBuilder();

 private:
  static string FakeNodeName(int in_index);
  Status GetN(int* n) const;
  Status GetDataType(DataType* dt) const;
  void NSources(int n, DataType dt) const;
  void SourceList(DataTypeSlice dts) const;

  const OpDef* const op_def_;
  const OpDef::ArgDef* const arg_;
  const string in_node_;
  const NodeDef* const node_def_;
  NodeDefBuilder* const builder_;

  bool n_specified_;
  int n_;
  bool dt_specified_;
  DataType dt_;
  bool dts_specified_;
  DataTypeSlice dts_;
};

FakeInputImpl::FakeInputImpl(const OpDef* op_def, int in_index,
                             const NodeDef* node_def, NodeDefBuilder* builder)
    : op_def_(op_def),
      arg_(&op_def->input_arg(in_index)),
      in_node_(FakeNodeName(in_index)),
      node_def_(node_def),
      builder_(builder),
      n_specified_(false),
      dt_specified_(false),
      dts_specified_(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_0(mht_0_v, 237, "", "./tensorflow/core/framework/fake_input.cc", "FakeInputImpl::FakeInputImpl");
}

void FakeInputImpl::SetN(int n) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_1(mht_1_v, 242, "", "./tensorflow/core/framework/fake_input.cc", "FakeInputImpl::SetN");

  n_specified_ = true;
  n_ = n;
}

void FakeInputImpl::SetDataType(DataType dt) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_2(mht_2_v, 250, "", "./tensorflow/core/framework/fake_input.cc", "FakeInputImpl::SetDataType");

  dt_specified_ = true;
  dt_ = dt;
}

void FakeInputImpl::SetTypeList(DataTypeSlice dts) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_3(mht_3_v, 258, "", "./tensorflow/core/framework/fake_input.cc", "FakeInputImpl::SetTypeList");

  dts_specified_ = true;
  dts_ = dts;
}

Status FakeInputImpl::AddInputToBuilder() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_4(mht_4_v, 266, "", "./tensorflow/core/framework/fake_input.cc", "FakeInputImpl::AddInputToBuilder");

  if (dts_specified_) {
    SourceList(dts_);

  } else if (n_specified_ || !arg_->number_attr().empty()) {
    int n;
    TF_RETURN_IF_ERROR(GetN(&n));

    DataType dt;
    if (n > 0) {
      TF_RETURN_IF_ERROR(GetDataType(&dt));
    } else {
      dt = DT_FLOAT;
    }

    NSources(n, dt);
  } else {
    if (!dt_specified_ && !arg_->type_list_attr().empty()) {
      DataTypeVector dts;
      Status status = GetNodeAttr(*node_def_, arg_->type_list_attr(), &dts);
      if (!status.ok()) {
        return errors::InvalidArgument(
            "Could not infer list of types for input '", arg_->name(),
            "': ", status.error_message());
      }
      SourceList(dts);
      return Status::OK();
    }

    DataType dt;
    TF_RETURN_IF_ERROR(GetDataType(&dt));
    builder_->Input(in_node_, 0, dt);
  }
  return Status::OK();
}

// static
string FakeInputImpl::FakeNodeName(int in_index) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_5(mht_5_v, 306, "", "./tensorflow/core/framework/fake_input.cc", "FakeInputImpl::FakeNodeName");

  char c = 'a' + (in_index % 26);
  return string(&c, 1);
}

Status FakeInputImpl::GetN(int* n) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_6(mht_6_v, 314, "", "./tensorflow/core/framework/fake_input.cc", "FakeInputImpl::GetN");

  if (n_specified_) {
    *n = n_;
  } else {
    Status status = GetNodeAttr(*node_def_, arg_->number_attr(), n);
    if (!status.ok()) {
      return errors::InvalidArgument("Could not infer length of input '",
                                     arg_->name(),
                                     "': ", status.error_message());
    }
  }
  return Status::OK();
}

Status FakeInputImpl::GetDataType(DataType* dt) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_7(mht_7_v, 331, "", "./tensorflow/core/framework/fake_input.cc", "FakeInputImpl::GetDataType");

  if (dt_specified_) {
    *dt = dt_;
    return Status::OK();  // Ignore is_ref field of arg_.
  } else if (arg_->type() != DT_INVALID) {
    *dt = arg_->type();
  } else if (!arg_->type_attr().empty()) {
    Status status = GetNodeAttr(*node_def_, arg_->type_attr(), dt);
    if (!status.ok()) {
      // Check if the type attr has a default
      const OpDef::AttrDef* attr = FindAttr(arg_->type_attr(), *op_def_);
      if (attr && attr->has_default_value()) {
        *dt = attr->default_value().type();
      } else {
        return errors::InvalidArgument("Could not infer type for input '",
                                       arg_->name(),
                                       "': ", status.error_message());
      }
    }
  } else {
    return errors::InvalidArgument("No type or type_attr field in arg '",
                                   arg_->name(), "'");
  }
  if (arg_->is_ref()) {
    *dt = MakeRefType(*dt);
  }
  return Status::OK();
}

void FakeInputImpl::NSources(int n, DataType dt) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_8(mht_8_v, 363, "", "./tensorflow/core/framework/fake_input.cc", "FakeInputImpl::NSources");

  std::vector<NodeDefBuilder::NodeOut> srcs;
  srcs.reserve(n);
  for (int i = 0; i < n; ++i) {
    srcs.emplace_back(in_node_, i, dt);
  }
  builder_->Input(gtl::ArraySlice<NodeDefBuilder::NodeOut>(srcs));
}

void FakeInputImpl::SourceList(DataTypeSlice dts) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_9(mht_9_v, 375, "", "./tensorflow/core/framework/fake_input.cc", "FakeInputImpl::SourceList");

  std::vector<NodeDefBuilder::NodeOut> srcs;
  srcs.reserve(dts.size());
  for (size_t i = 0; i < dts.size(); ++i) {
    srcs.emplace_back(in_node_, i, dts[i]);
  }
  builder_->Input(gtl::ArraySlice<NodeDefBuilder::NodeOut>(srcs));
}

}  // namespace

// Public interface ------------------------------------------------------------

FakeInputFunctor FakeInput() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_10(mht_10_v, 391, "", "./tensorflow/core/framework/fake_input.cc", "FakeInput");

  return [](const OpDef& op_def, int in_index, const NodeDef& node_def,
            NodeDefBuilder* builder) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_11(mht_11_v, 396, "", "./tensorflow/core/framework/fake_input.cc", "lambda");

    FakeInputImpl impl(&op_def, in_index, &node_def, builder);
    return impl.AddInputToBuilder();
  };
}

FakeInputFunctor FakeInput(DataType dt) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_12(mht_12_v, 405, "", "./tensorflow/core/framework/fake_input.cc", "FakeInput");

  return [dt](const OpDef& op_def, int in_index, const NodeDef& node_def,
              NodeDefBuilder* builder) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_13(mht_13_v, 410, "", "./tensorflow/core/framework/fake_input.cc", "lambda");

    FakeInputImpl impl(&op_def, in_index, &node_def, builder);
    impl.SetDataType(dt);
    return impl.AddInputToBuilder();
  };
}

FakeInputFunctor FakeInput(int n) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_14(mht_14_v, 420, "", "./tensorflow/core/framework/fake_input.cc", "FakeInput");

  return [n](const OpDef& op_def, int in_index, const NodeDef& node_def,
             NodeDefBuilder* builder) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_15(mht_15_v, 425, "", "./tensorflow/core/framework/fake_input.cc", "lambda");

    FakeInputImpl impl(&op_def, in_index, &node_def, builder);
    impl.SetN(n);
    return impl.AddInputToBuilder();
  };
}

FakeInputFunctor FakeInput(int n, DataType dt) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_16(mht_16_v, 435, "", "./tensorflow/core/framework/fake_input.cc", "FakeInput");

  return [n, dt](const OpDef& op_def, int in_index, const NodeDef& node_def,
                 NodeDefBuilder* builder) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_17(mht_17_v, 440, "", "./tensorflow/core/framework/fake_input.cc", "lambda");

    FakeInputImpl impl(&op_def, in_index, &node_def, builder);
    impl.SetN(n);
    impl.SetDataType(dt);
    return impl.AddInputToBuilder();
  };
}

FakeInputFunctor FakeInput(DataTypeSlice dts) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_18(mht_18_v, 451, "", "./tensorflow/core/framework/fake_input.cc", "FakeInput");

  // Make a copy to ensure the data will still be around when the lambda is
  // called.
  DataTypeVector dtv(dts.begin(), dts.end());
  return [dtv](const OpDef& op_def, int in_index, const NodeDef& node_def,
               NodeDefBuilder* builder) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfake_inputDTcc mht_19(mht_19_v, 459, "", "./tensorflow/core/framework/fake_input.cc", "lambda");

    FakeInputImpl impl(&op_def, in_index, &node_def, builder);
    impl.SetTypeList(dtv);
    return impl.AddInputToBuilder();
  };
}

}  // namespace tensorflow
