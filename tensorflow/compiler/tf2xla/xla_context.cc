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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTcc() {
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

#include "tensorflow/compiler/tf2xla/xla_context.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

const char XlaContext::kXlaContextResourceName[] = "_xla_context";

// Looks up the context associated with the current step. It is stored
// in a resource container managed by the device.
/* static */ XlaContext& XlaContext::Get(const OpKernelContext* ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/tf2xla/xla_context.cc", "XlaContext::Get");

  // When an Op kernel wants to use an XLA JIT context, the
  // per-step context is looked up in the resource manager. The
  // JIT will prepopulate the JITContext.
  XlaContext* context;
  TF_CHECK_OK(ctx->step_container()->Lookup(ctx->resource_manager(),
                                            kXlaContextResourceName, &context));
  // The resource manager handed us a fresh reference to 'context', but retains
  // a reference itself so the context won't be freed. The resource manager will
  // outlive the JIT compilation.
  context->Unref();
  return *context;
}

void XlaContext::set_args(std::vector<XlaExpression> args) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTcc mht_1(mht_1_v, 228, "", "./tensorflow/compiler/tf2xla/xla_context.cc", "XlaContext::set_args");

  args_ = std::move(args);
}

XlaContext::XlaContext(XlaCompiler* compiler, xla::XlaBuilder* builder,
                       const Graph* graph)
    : compiler_(compiler), builder_(builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTcc mht_2(mht_2_v, 237, "", "./tensorflow/compiler/tf2xla/xla_context.cc", "XlaContext::XlaContext");

  if (graph) {
    for (const Node* node : graph->nodes()) {
      stack_traces_[node->name()] = node->GetStackTrace();
    }
  }
}

string XlaContext::DebugString() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTcc mht_3(mht_3_v, 248, "", "./tensorflow/compiler/tf2xla/xla_context.cc", "XlaContext::DebugString");
 return "XLA JIT context"; }

void XlaContext::SetRetval(int index, const XlaExpression& expression) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTcc mht_4(mht_4_v, 253, "", "./tensorflow/compiler/tf2xla/xla_context.cc", "XlaContext::SetRetval");

  const int64_t retvals_size = retvals_.size();
  if (retvals_size <= index) {
    retvals_.resize(index + 1);
  }
  retvals_[index] = expression;
}

XlaResource* XlaContext::AddResource(std::unique_ptr<XlaResource> resource) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTcc mht_5(mht_5_v, 264, "", "./tensorflow/compiler/tf2xla/xla_context.cc", "XlaContext::AddResource");

  resources_.push_back(std::move(resource));
  return resources_.back().get();
}

const xla::XlaComputation* XlaContext::GetOrCreateMax(const DataType type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTcc mht_6(mht_6_v, 272, "", "./tensorflow/compiler/tf2xla/xla_context.cc", "XlaContext::GetOrCreateMax");

  return LookupOrCreate(type, &max_func_, [type] {
    const string type_string = DataTypeString(type);
    VLOG(1) << "Building Max() for " << type_string;
    xla::XlaBuilder b("max<" + type_string + ">");
    xla::PrimitiveType xla_type;
    TF_CHECK_OK(DataTypeToPrimitiveType(type, &xla_type));
    auto x =
        xla::Parameter(&b, 0, xla::ShapeUtil::MakeShape(xla_type, {}), "x");
    auto y =
        xla::Parameter(&b, 1, xla::ShapeUtil::MakeShape(xla_type, {}), "y");
    xla::Max(x, y);
    return b.Build().ConsumeValueOrDie();
  });
}

const xla::XlaComputation* XlaContext::GetOrCreateMin(const DataType type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTcc mht_7(mht_7_v, 291, "", "./tensorflow/compiler/tf2xla/xla_context.cc", "XlaContext::GetOrCreateMin");

  return LookupOrCreate(type, &min_func_, [type] {
    const string type_string = DataTypeString(type);
    VLOG(1) << "Building Min() for " << type_string;
    xla::XlaBuilder b("min<" + type_string + ">");
    xla::PrimitiveType xla_type;
    TF_CHECK_OK(DataTypeToPrimitiveType(type, &xla_type));
    auto x =
        xla::Parameter(&b, 0, xla::ShapeUtil::MakeShape(xla_type, {}), "x");
    auto y =
        xla::Parameter(&b, 1, xla::ShapeUtil::MakeShape(xla_type, {}), "y");
    xla::Min(x, y);
    return b.Build().ConsumeValueOrDie();
  });
}

const xla::XlaComputation* XlaContext::GetOrCreateAdd(const DataType type) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTcc mht_8(mht_8_v, 310, "", "./tensorflow/compiler/tf2xla/xla_context.cc", "XlaContext::GetOrCreateAdd");

  return LookupOrCreate(type, &add_func_, [type] {
    const string type_string = DataTypeString(type);
    VLOG(1) << "Building Add() for " << type_string;
    xla::XlaBuilder b("add<" + type_string + ">");
    xla::PrimitiveType xla_type;
    TF_CHECK_OK(DataTypeToPrimitiveType(type, &xla_type));
    auto x =
        xla::Parameter(&b, 0, xla::ShapeUtil::MakeShape(xla_type, {}), "x");
    auto y =
        xla::Parameter(&b, 1, xla::ShapeUtil::MakeShape(xla_type, {}), "y");
    xla::Add(x, y);
    return b.Build().ConsumeValueOrDie();
  });
}

const xla::XlaComputation* XlaContext::GetOrCreateMul(const DataType type) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTcc mht_9(mht_9_v, 329, "", "./tensorflow/compiler/tf2xla/xla_context.cc", "XlaContext::GetOrCreateMul");

  return LookupOrCreate(type, &mul_func_, [type] {
    const string type_string = DataTypeString(type);
    VLOG(1) << "Building Mul() for " << type_string;
    xla::XlaBuilder b("mul<" + type_string + ">");
    xla::PrimitiveType xla_type;
    TF_CHECK_OK(DataTypeToPrimitiveType(type, &xla_type));
    auto x =
        xla::Parameter(&b, 0, xla::ShapeUtil::MakeShape(xla_type, {}), "x");
    auto y =
        xla::Parameter(&b, 1, xla::ShapeUtil::MakeShape(xla_type, {}), "y");
    xla::Mul(x, y);
    return b.Build().ConsumeValueOrDie();
  });
}

const xla::XlaComputation* XlaContext::LookupOrCreate(
    DataType type, ComputationMap* out,
    const std::function<xla::XlaComputation()>& create) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTcc mht_10(mht_10_v, 350, "", "./tensorflow/compiler/tf2xla/xla_context.cc", "XlaContext::LookupOrCreate");

  {
    const auto& entry = (*out)[type];
    if (!entry.IsNull()) {
      return &entry;
    }
  }
  auto new_entry = create();
  {
    // Somebody else might have made one concurrently.
    auto& entry = (*out)[type];
    if (entry.IsNull()) {
      entry = std::move(new_entry);
    }
    return &entry;
  }
}

Status XlaContext::RecordCollectiveInfoFromNestedCompilationResult(
    const XlaCompilationResult& result) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTcc mht_11(mht_11_v, 372, "", "./tensorflow/compiler/tf2xla/xla_context.cc", "XlaContext::RecordCollectiveInfoFromNestedCompilationResult");

  if (result.collective_info) {
    return RecordCollectiveInfo(result.collective_info->group_key,
                                result.collective_info->group_size)
        .status();
  }
  return Status::OK();
}

StatusOr<int64_t> XlaContext::RecordCollectiveInfo(int group_key,
                                                   int group_size) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_contextDTcc mht_12(mht_12_v, 385, "", "./tensorflow/compiler/tf2xla/xla_context.cc", "XlaContext::RecordCollectiveInfo");

  if (!collective_info_) {
    collective_info_ = {group_key, group_size, 0};
  } else if (collective_info_->group_key != group_key ||
             collective_info_->group_size != group_size) {
    return errors::InvalidArgument(
        "Only single configuration of CollectiveReduceV2Op is ",
        "supported in a given cluster. Recorded group_key=",
        collective_info_->group_key,
        " attempting to insert group_key=", group_key);
  }

  // Create the channel_id to be used for the collective. Avoid having the
  // same channel_id to be used for 2 or more collectives since XLA attempts
  // to "gang schedule" all collectives with the same channel_id.
  return (static_cast<int64_t>(group_key) << 32) | collective_info_->next_id++;
}

}  // namespace tensorflow
