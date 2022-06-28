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
class MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc {
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
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/js/ops/ts_op_gen.h"
#include <unordered_map>

#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

static bool IsListAttr(const OpDef_ArgDef& arg) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc mht_0(mht_0_v, 197, "", "./tensorflow/js/ops/ts_op_gen.cc", "IsListAttr");

  return !arg.type_list_attr().empty() || !arg.number_attr().empty();
}

// Struct to hold a combo OpDef and ArgDef for a given Op argument:
struct ArgDefs {
  ArgDefs(const OpDef::ArgDef& op_def_arg, const ApiDef::Arg& api_def_arg)
      : op_def_arg(op_def_arg), api_def_arg(api_def_arg) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc mht_1(mht_1_v, 207, "", "./tensorflow/js/ops/ts_op_gen.cc", "ArgDefs");
}

  const OpDef::ArgDef& op_def_arg;
  const ApiDef::Arg& api_def_arg;
};

// Struct to hold a combo OpDef::AttrDef and ApiDef::Attr for an Op.
struct OpAttrs {
  OpAttrs(const OpDef::AttrDef& op_def_attr, const ApiDef::Attr& api_def_attr)
      : op_def_attr(op_def_attr), api_def_attr(api_def_attr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc mht_2(mht_2_v, 219, "", "./tensorflow/js/ops/ts_op_gen.cc", "OpAttrs");
}

  const OpDef::AttrDef& op_def_attr;
  const ApiDef::Attr& api_def_attr;
};

// Helper class to generate TypeScript code for a given OpDef:
class GenTypeScriptOp {
 public:
  GenTypeScriptOp(const OpDef& op_def, const ApiDef& api_def);
  ~GenTypeScriptOp();

  // Returns the generated code as a string:
  string Code();

 private:
  void ProcessArgs();
  void ProcessAttrs();
  void AddAttrForArg(const string& attr, int arg_index);
  string InputForAttr(const OpDef::AttrDef& op_def_attr);

  void AddMethodSignature();
  void AddOpAttrs();
  void AddMethodReturnAndClose();

  const OpDef& op_def_;
  const ApiDef& api_def_;

  // Placeholder string for all generated code:
  string result_;

  // Holds in-order vector of Op inputs:
  std::vector<ArgDefs> input_op_args_;

  // Holds in-order vector of Op attributes:
  std::vector<OpAttrs> op_attrs_;

  // Stores attributes-to-arguments by name:
  typedef std::unordered_map<string, std::vector<int>> AttrArgIdxMap;
  AttrArgIdxMap attr_arg_idx_map_;

  // Holds number of outputs:
  int num_outputs_;
};

GenTypeScriptOp::GenTypeScriptOp(const OpDef& op_def, const ApiDef& api_def)
    : op_def_(op_def), api_def_(api_def), num_outputs_(0) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc mht_3(mht_3_v, 268, "", "./tensorflow/js/ops/ts_op_gen.cc", "GenTypeScriptOp::GenTypeScriptOp");
}

GenTypeScriptOp::~GenTypeScriptOp() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc mht_4(mht_4_v, 273, "", "./tensorflow/js/ops/ts_op_gen.cc", "GenTypeScriptOp::~GenTypeScriptOp");
}

string GenTypeScriptOp::Code() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc mht_5(mht_5_v, 278, "", "./tensorflow/js/ops/ts_op_gen.cc", "GenTypeScriptOp::Code");

  ProcessArgs();
  ProcessAttrs();

  // Generate exported function for Op:
  AddMethodSignature();
  AddOpAttrs();
  AddMethodReturnAndClose();

  strings::StrAppend(&result_, "\n");
  return result_;
}

void GenTypeScriptOp::ProcessArgs() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc mht_6(mht_6_v, 294, "", "./tensorflow/js/ops/ts_op_gen.cc", "GenTypeScriptOp::ProcessArgs");

  for (int i = 0; i < api_def_.arg_order_size(); i++) {
    auto op_def_arg = FindInputArg(api_def_.arg_order(i), op_def_);
    if (op_def_arg == nullptr) {
      LOG(WARNING) << "Could not find OpDef::ArgDef for "
                   << api_def_.arg_order(i);
      continue;
    }
    auto api_def_arg = FindInputArg(api_def_.arg_order(i), api_def_);
    if (api_def_arg == nullptr) {
      LOG(WARNING) << "Could not find ApiDef::Arg for "
                   << api_def_.arg_order(i);
      continue;
    }

    // Map attr names to arg indexes:
    if (!op_def_arg->type_attr().empty()) {
      AddAttrForArg(op_def_arg->type_attr(), i);
    } else if (!op_def_arg->type_list_attr().empty()) {
      AddAttrForArg(op_def_arg->type_list_attr(), i);
    }
    if (!op_def_arg->number_attr().empty()) {
      AddAttrForArg(op_def_arg->number_attr(), i);
    }

    input_op_args_.push_back(ArgDefs(*op_def_arg, *api_def_arg));
  }

  num_outputs_ = api_def_.out_arg_size();
}

void GenTypeScriptOp::ProcessAttrs() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc mht_7(mht_7_v, 328, "", "./tensorflow/js/ops/ts_op_gen.cc", "GenTypeScriptOp::ProcessAttrs");

  for (int i = 0; i < op_def_.attr_size(); i++) {
    op_attrs_.push_back(OpAttrs(op_def_.attr(i), api_def_.attr(i)));
  }
}

void GenTypeScriptOp::AddAttrForArg(const string& attr, int arg_index) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("attr: \"" + attr + "\"");
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc mht_8(mht_8_v, 338, "", "./tensorflow/js/ops/ts_op_gen.cc", "GenTypeScriptOp::AddAttrForArg");

  // Keep track of attributes-to-arguments by name. These will be used for
  // construction Op attributes that require information about the inputs.
  auto iter = attr_arg_idx_map_.find(attr);
  if (iter == attr_arg_idx_map_.end()) {
    attr_arg_idx_map_.insert(AttrArgIdxMap::value_type(attr, {arg_index}));
  } else {
    iter->second.push_back(arg_index);
  }
}

string GenTypeScriptOp::InputForAttr(const OpDef::AttrDef& op_def_attr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc mht_9(mht_9_v, 352, "", "./tensorflow/js/ops/ts_op_gen.cc", "GenTypeScriptOp::InputForAttr");

  string inputs;
  auto arg_list = attr_arg_idx_map_.find(op_def_attr.name());
  if (arg_list != attr_arg_idx_map_.end()) {
    for (auto iter = arg_list->second.begin(); iter != arg_list->second.end();
         ++iter) {
      strings::StrAppend(&inputs, input_op_args_[*iter].op_def_arg.name());
    }
  }
  return inputs;
}

void GenTypeScriptOp::AddMethodSignature() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc mht_10(mht_10_v, 367, "", "./tensorflow/js/ops/ts_op_gen.cc", "GenTypeScriptOp::AddMethodSignature");

  strings::StrAppend(&result_, "export function ", api_def_.endpoint(0).name(),
                     "(");

  bool is_first = true;
  for (auto& in_arg : input_op_args_) {
    if (is_first) {
      is_first = false;
    } else {
      strings::StrAppend(&result_, ", ");
    }

    auto op_def_arg = in_arg.op_def_arg;

    strings::StrAppend(&result_, op_def_arg.name(), ": ");
    if (IsListAttr(op_def_arg)) {
      strings::StrAppend(&result_, "tfc.Tensor[]");
    } else {
      strings::StrAppend(&result_, "tfc.Tensor");
    }
  }

  if (num_outputs_ == 1) {
    strings::StrAppend(&result_, "): tfc.Tensor {\n");
  } else {
    strings::StrAppend(&result_, "): tfc.Tensor[] {\n");
  }
}

void GenTypeScriptOp::AddOpAttrs() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc mht_11(mht_11_v, 399, "", "./tensorflow/js/ops/ts_op_gen.cc", "GenTypeScriptOp::AddOpAttrs");

  strings::StrAppend(&result_, "  const opAttrs = [\n");

  bool is_first = true;
  for (auto& attr : op_attrs_) {
    if (is_first) {
      is_first = false;
    } else {
      strings::StrAppend(&result_, ",\n");
    }

    // Append 4 spaces to start:
    strings::StrAppend(&result_, "    ");

    if (attr.op_def_attr.type() == "type") {
      // Type OpAttributes can be generated from a helper function:
      strings::StrAppend(&result_, "createTensorsTypeOpAttr('",
                         attr.op_def_attr.name(), "', ",
                         InputForAttr(attr.op_def_attr), ")");
    } else if (attr.op_def_attr.type() == "int") {
      strings::StrAppend(&result_, "{name: '", attr.op_def_attr.name(), "', ");
      strings::StrAppend(&result_, "type: nodeBackend().binding.TF_ATTR_INT, ");
      strings::StrAppend(&result_, "value: ", InputForAttr(attr.op_def_attr),
                         ".length}");
    }
  }
  strings::StrAppend(&result_, "\n  ];\n");
}

void GenTypeScriptOp::AddMethodReturnAndClose() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc mht_12(mht_12_v, 431, "", "./tensorflow/js/ops/ts_op_gen.cc", "GenTypeScriptOp::AddMethodReturnAndClose");

  strings::StrAppend(&result_, "  return null;\n}\n");
}

void WriteTSOp(const OpDef& op_def, const ApiDef& api_def, WritableFile* ts) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc mht_13(mht_13_v, 438, "", "./tensorflow/js/ops/ts_op_gen.cc", "WriteTSOp");

  GenTypeScriptOp ts_op(op_def, api_def);
  TF_CHECK_OK(ts->Append(GenTypeScriptOp(op_def, api_def).Code()));
}

void StartFile(WritableFile* ts_file) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc mht_14(mht_14_v, 446, "", "./tensorflow/js/ops/ts_op_gen.cc", "StartFile");

  const string header =
      R"header(/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// This file is MACHINE GENERATED! Do not edit

import * as tfc from '@tensorflow/tfjs-core';
import {createTensorsTypeOpAttr, nodeBackend} from './op_utils';

)header";

  TF_CHECK_OK(ts_file->Append(header));
}

}  // namespace

void WriteTSOps(const OpList& ops, const ApiDefMap& api_def_map,
                const string& ts_filename) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("ts_filename: \"" + ts_filename + "\"");
   MHTracer_DTPStensorflowPSjsPSopsPSts_op_genDTcc mht_15(mht_15_v, 482, "", "./tensorflow/js/ops/ts_op_gen.cc", "WriteTSOps");

  Env* env = Env::Default();

  std::unique_ptr<WritableFile> ts_file = nullptr;
  TF_CHECK_OK(env->NewWritableFile(ts_filename, &ts_file));

  StartFile(ts_file.get());

  for (const auto& op_def : ops.op()) {
    // Skip deprecated ops
    if (op_def.has_deprecation() &&
        op_def.deprecation().version() <= TF_GRAPH_DEF_VERSION) {
      continue;
    }

    const auto* api_def = api_def_map.GetApiDef(op_def.name());
    if (api_def->visibility() == ApiDef::VISIBLE) {
      WriteTSOp(op_def, *api_def, ts_file.get());
    }
  }

  TF_CHECK_OK(ts_file->Close());
}

}  // namespace tensorflow
