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
class MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_utilDTcc() {
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

#include "tensorflow/core/framework/full_type_inference_util.h"

#include <functional>
#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/full_type_util.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

namespace full_type {

// Note about error handling:
// For inputs which depend on the correctness of the op definition
// (i.e. if the op has three inputs, don't set an `i` that exceeds that),
// use DCHECK - an incorrect op def is considered a bug.
// Whereas for inputs that depend on the correctness of the graph (i.e. user
// used the correct ops), use Status - an incorrect graph is considered a user
// error.

ForwardTypeInferenceFn ReplicateInput(int i, int n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_utilDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/framework/full_type_inference_util.cc", "ReplicateInput");

  return [i, n](const TypeRefVector& input_types, const TypeRefMap& type_vars) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_utilDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/framework/full_type_inference_util.cc", "lambda");

    const FullTypeDef& in_type = input_types.at(i).get();
    FullTypeDef ret_type;
    if (in_type.type_id() != TFT_UNSET) {
      ret_type.set_type_id(TFT_PRODUCT);
      for (int k = 0; k < n; k++) {
        *(ret_type.add_args()) = in_type;
      }
    }
    return ret_type;
  };
}

ForwardTypeInferenceFn Merge() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_utilDTcc mht_2(mht_2_v, 230, "", "./tensorflow/core/framework/full_type_inference_util.cc", "Merge");

  return [](const TypeRefVector& input_types,
            const TypeRefMap& type_vars) -> StatusOr<FullTypeDef> {
    DCHECK(!input_types.empty());

    FullTypeDef merged;
    for (int i = 0; i < input_types.size(); i++) {
      const auto& t = input_types[i].get();

      if (t.type_id() == TFT_UNSET) {
        continue;
      }

      if (IsSubtype(t, merged)) {
        merged = t;
        continue;
      }
      if (IsSubtype(merged, t)) {
        continue;
      }

      return Status(error::INVALID_ARGUMENT,
                    absl::StrCat("expected compatible input types, but input ",
                                 i, ":\n", t.DebugString(),
                                 " is neither a subtype nor a supertype of the "
                                 "combined inputs preceding it:\n",
                                 merged.DebugString()));
    }

    FullTypeDef ret_type;
    if (merged.type_id() != TFT_UNSET) {
      ret_type.set_type_id(TFT_PRODUCT);
      *(ret_type.add_args()) = merged;
    }
    return ret_type;
  };
}

ForwardTypeInferenceFn UnaryContainerCreate(FullTypeId t, int element_idx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_utilDTcc mht_3(mht_3_v, 271, "", "./tensorflow/core/framework/full_type_inference_util.cc", "UnaryContainerCreate");

  return
      [t, element_idx](const TypeRefVector& input_types,
                       const TypeRefMap& type_vars) -> StatusOr<FullTypeDef> {
        DCHECK(input_types.size() >= element_idx);

        FullTypeDef ret_type;
        ret_type.set_type_id(TFT_PRODUCT);
        FullTypeDef* arg_t = ret_type.add_args();
        arg_t->set_type_id(t);
        *(arg_t->add_args()) = input_types[element_idx].get();

        return ret_type;
      };
}

ForwardTypeInferenceFn UnaryContainerAdd(FullTypeId t, int container_idx,
                                         int element_idx, bool homogeneous) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_utilDTcc mht_4(mht_4_v, 291, "", "./tensorflow/core/framework/full_type_inference_util.cc", "UnaryContainerAdd");

  return [t, container_idx, element_idx, homogeneous](
             const TypeRefVector& input_types,
             const TypeRefMap& type_vars) -> StatusOr<FullTypeDef> {
    DCHECK(input_types.size() >= container_idx);
    DCHECK(input_types.size() >= element_idx);

    FullTypeDef ret_type;
    ret_type.set_type_id(TFT_PRODUCT);
    FullTypeDef* cont_t = ret_type.add_args();
    cont_t->set_type_id(t);

    const FullTypeDef& in_cont_t = input_types[container_idx].get();
    const FullTypeDef& in_el_t = input_types[element_idx].get();

    if (in_cont_t.type_id() != TFT_UNSET) {
      if (in_cont_t.type_id() != t) {
        return Status(
            error::INVALID_ARGUMENT,
            absl::StrCat("expected container type ", t, " for input ",
                         container_idx, ", got ", in_cont_t.DebugString()));
      }
      *cont_t = in_cont_t;
    }

    VLOG(1) << "ContainerAddUnary: " << cont_t->DebugString() << ", "
            << in_el_t.DebugString() << ", " << container_idx << "; "
            << element_idx;
    for (const auto& tmp : input_types) {
      VLOG(1) << "  input: " << tmp.get().DebugString();
    }

    if (in_el_t.type_id() == TFT_UNSET) {
      return ret_type;
    }

    const FullTypeDef& el_t = GetArgDefaultUnset(*cont_t, 0);

    if (el_t.type_id() == TFT_UNSET) {
      cont_t->clear_args();
      *(cont_t->add_args()) = in_el_t;
      return ret_type;
    }

    if (IsSubtype(in_el_t, el_t)) {
      // Nothing to do, will not refine the container type based on a single
      // addition.
      return ret_type;
    }

    if (homogeneous) {
      return Status(error::INVALID_ARGUMENT,
                    absl::StrCat("expected a subtype of ", el_t.DebugString(),
                                 " for input ", element_idx,
                                 " of a homogeneous container ", t, ", got ",
                                 in_el_t.DebugString()));
    } else {
      // TODO(mdan): Implement if needed.
      return Status(
          error::UNIMPLEMENTED,
          absl::StrCat("need union types for heterogeneous containers.\n"
                       "A homogeneous container would expect a subtype of ",
                       el_t.DebugString(), " for input ", element_idx,
                       ", but got ", in_el_t.DebugString()));
    }
  };
}

ForwardTypeInferenceFn MultiaryUnstack(
    FullTypeId t, std::function<FullTypeDef(const FullTypeDef&)> unstack) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_utilDTcc mht_5(mht_5_v, 363, "", "./tensorflow/core/framework/full_type_inference_util.cc", "MultiaryUnstack");

  return [t, unstack](const TypeRefVector& input_types,
                      const TypeRefMap& type_vars) -> StatusOr<FullTypeDef> {
    FullTypeDef ret_type;
    ret_type.set_type_id(TFT_PRODUCT);
    FullTypeDef* cont_t = ret_type.add_args();
    cont_t->set_type_id(t);
    FullTypeDef* el_t = cont_t->add_args();
    el_t->set_type_id(TFT_PRODUCT);
    for (int element_idx = 0; element_idx < input_types.size(); ++element_idx) {
      *(el_t->add_args()) = unstack(input_types[element_idx].get());
    }
    return ret_type;
  };
}

FullTypeDef UnstackTensor(const FullTypeDef& t) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_utilDTcc mht_6(mht_6_v, 382, "", "./tensorflow/core/framework/full_type_inference_util.cc", "UnstackTensor");

  // For now, only TFT_TENSOR and TFT_RAGGED are supported and
  // only if they have a single argument (i.e. they don't specify a shape).
  // If these have a shape in the future, this function needs to changed
  // so that the output shape is computed based on the input shape and the
  // effect of the unstack operation (e.g. a dimension is removed).
  // TFT_UNSET is also allowed to support weak type inference where
  // not having a fulltype is allowed.
  DCHECK((t.type_id() == TFT_TENSOR) || (t.type_id() == TFT_RAGGED) ||
         (t.type_id() == TFT_UNSET));
  DCHECK_LE(t.args_size(), 1);
  return t;
}

ForwardTypeInferenceFn ContainerMap(
    FullTypeId t, int input_idx,
    std::function<FullTypeDef(const FullTypeDef&)> map) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_utilDTcc mht_7(mht_7_v, 401, "", "./tensorflow/core/framework/full_type_inference_util.cc", "ContainerMap");

  return [t, input_idx, map](
             const TypeRefVector& input_types,
             const TypeRefMap& type_vars) -> StatusOr<FullTypeDef> {
    DCHECK_GE(input_types.size(), input_idx);
    const FullTypeDef& in_cont_t = input_types.at(input_idx).get();
    FullTypeDef ret_type;
    if (in_cont_t.type_id() == TFT_UNSET) {
      return ret_type;
    }
    if (in_cont_t.type_id() != t) {
      return Status(error::INVALID_ARGUMENT,
                    absl::StrCat("expected type ", t, " for input ", input_idx,
                                 ", got ", in_cont_t.DebugString()));
    }
    ret_type.set_type_id(TFT_PRODUCT);
    FullTypeDef* out_cont_t = ret_type.add_args();
    out_cont_t->set_type_id(t);
    const FullTypeDef& in_el_t = GetArgDefaultUnset(in_cont_t, 0);
    if (in_el_t.type_id() == TFT_UNSET) {
      return ret_type;
    }
    if (in_el_t.type_id() != TFT_PRODUCT) {
      return Status(error::INVALID_ARGUMENT,
                    absl::StrCat("expected PRODUCT element type for input ",
                                 input_idx, ", got ", in_el_t.DebugString()));
    }
    FullTypeDef* out_el_t = out_cont_t->add_args();
    out_el_t->set_type_id(TFT_PRODUCT);
    for (int k = 0; k < in_el_t.args_size(); k++) {
      *(out_el_t->add_args()) = map(in_el_t.args(k));
    }
    return ret_type;
  };
}

FullTypeDef BatchTensor(const FullTypeDef& t) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_utilDTcc mht_8(mht_8_v, 440, "", "./tensorflow/core/framework/full_type_inference_util.cc", "BatchTensor");

  // For now, just return the input type.
  // If the input type has a shape in the future, this function needs to be
  // changed so that the output shape is computed based on the input shape and
  // the effect of the op that changes the batch size (and this function would
  // require more information to do this computation).
  return t;
}

FullTypeDef ShardTensor(const FullTypeDef& t) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_utilDTcc mht_9(mht_9_v, 452, "", "./tensorflow/core/framework/full_type_inference_util.cc", "ShardTensor");

  // For now, just return the input type.
  // If the input type has a shape in the future, this function needs to be
  // changed so that the output shape is computed based on the input shape and
  // the effect of the op that shards the input into multiple tensors (and this
  // function would require more information to do this computation).
  return t;
}

}  // namespace full_type

}  // namespace tensorflow
