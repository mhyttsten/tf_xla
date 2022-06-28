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
class MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc {
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
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc() {
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

#include <algorithm>
#include <vector>

#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

Scope::Scope(Impl* impl) : impl_(impl) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_0(mht_0_v, 196, "", "./tensorflow/cc/framework/scope.cc", "Scope::Scope");
}

Scope::Scope(const Scope& other) : impl_(new Impl(*other.impl())) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_1(mht_1_v, 201, "", "./tensorflow/cc/framework/scope.cc", "Scope::Scope");
}

Scope::~Scope() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_2(mht_2_v, 206, "", "./tensorflow/cc/framework/scope.cc", "Scope::~Scope");
}

Scope& Scope::operator=(const Scope& other) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_3(mht_3_v, 211, "", "./tensorflow/cc/framework/scope.cc", "=");

  // We can't copy Impls because of the const members, use copy ctor instead
  impl_.reset(new Impl(*other.impl_));
  return *this;
}

namespace {
const char kScopeSeparator[] = "/";
const char kSuffixSeparator[] = "_";
}  // namespace

Scope::Impl::Impl(Graph* graph, Status* status, NameMap* name_map,
                  ShapeRefiner* refiner, bool disable_shape_inference)
    : graph_(graph),
      status_(status),
      name_map_(name_map),
      refiner_(refiner),
      scope_used_(nullptr),
      colocation_constraints_(),
      disable_shape_inference_(disable_shape_inference) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_4(mht_4_v, 233, "", "./tensorflow/cc/framework/scope.cc", "Scope::Impl::Impl");
}

Scope::Impl::Impl(const std::shared_ptr<Graph>& graph,
                  const std::shared_ptr<Status>& status,
                  const std::shared_ptr<NameMap>& name_map,
                  const std::shared_ptr<ShapeRefiner>& refiner)
    : graph_(graph),
      status_(status),
      name_map_(name_map),
      refiner_(refiner),
      scope_used_(nullptr),
      colocation_constraints_(),
      disable_shape_inference_(refiner_ == nullptr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_5(mht_5_v, 248, "", "./tensorflow/cc/framework/scope.cc", "Scope::Impl::Impl");
}

Scope Scope::NewRootScope() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_6(mht_6_v, 253, "", "./tensorflow/cc/framework/scope.cc", "Scope::NewRootScope");

  Graph* graph = new Graph(OpRegistry::Global());
  ShapeRefiner* refiner =
      new ShapeRefiner(graph->versions(), graph->op_registry());
  return Scope(new Impl(graph, new Status, new Impl::NameMap, refiner,
                        /* disable_shape_inference */ false));
}

Scope Scope::DisabledShapeInferenceScope() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_7(mht_7_v, 264, "", "./tensorflow/cc/framework/scope.cc", "Scope::DisabledShapeInferenceScope");

  Graph* graph = new Graph(OpRegistry::Global());
  ShapeRefiner* refiner =
      new ShapeRefiner(graph->versions(), graph->op_registry());
  return Scope(new Impl(graph, new Status, new Impl::NameMap, refiner,
                        /* disable_shape_inference */ true));
}

Scope::Impl::Impl(const Scope& other, Tags::ScopeName, const string& name,
                  bool copy_names)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(copy_names ? other.impl()->name_map_
                           : std::shared_ptr<NameMap>(new NameMap)),
      refiner_(other.impl()->refiner_),
      scope_used_(nullptr),
      control_deps_(other.impl()->control_deps_),
      name_(name),
      op_name_(""),
      exit_on_error_(other.impl()->exit_on_error_),
      kernel_label_(other.impl()->kernel_label_),
      device_(other.impl()->device_),
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_8(mht_8_v, 293, "", "./tensorflow/cc/framework/scope.cc", "Scope::Impl::Impl");
}

Scope::Impl::Impl(const Scope& other, Tags::OpName, const string& name,
                  const string& op_name)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(other.impl()->name_map_),
      refiner_(other.impl()->refiner_),
      scope_used_(other.impl()->scope_used_),
      control_deps_(other.impl()->control_deps_),
      name_(name),
      op_name_(op_name),
      exit_on_error_(other.impl()->exit_on_error_),
      kernel_label_(other.impl()->kernel_label_),
      device_(other.impl()->device_),
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + name + "\"");
   mht_9_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_9(mht_9_v, 316, "", "./tensorflow/cc/framework/scope.cc", "Scope::Impl::Impl");
}

Scope::Impl::Impl(const Scope& other, Tags::ControlDeps,
                  std::vector<Operation> control_deps, bool clear_control_deps)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(other.impl()->name_map_),
      refiner_(other.impl()->refiner_),
      scope_used_(other.impl()->scope_used_),
      control_deps_(
          clear_control_deps
              ? std::vector<Operation>()
              : (control_deps.insert(control_deps.begin(),
                                     other.impl()->control_deps_.begin(),
                                     other.impl()->control_deps_.end()),
                 control_deps)),
      name_(other.impl()->name_),
      op_name_(other.impl()->op_name_),
      exit_on_error_(other.impl()->exit_on_error_),
      kernel_label_(other.impl()->kernel_label_),
      device_(other.impl()->device_),
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_10(mht_10_v, 343, "", "./tensorflow/cc/framework/scope.cc", "Scope::Impl::Impl");
}

Scope::Impl::Impl(const Scope& other, Tags::Device, const string& device)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(other.impl()->name_map_),
      refiner_(other.impl()->refiner_),
      scope_used_(other.impl()->scope_used_),
      control_deps_(other.impl()->control_deps_),
      name_(other.impl()->name_),
      op_name_(other.impl()->op_name_),
      exit_on_error_(other.impl()->exit_on_error_),
      kernel_label_(other.impl()->kernel_label_),
      device_(device),
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_11(mht_11_v, 364, "", "./tensorflow/cc/framework/scope.cc", "Scope::Impl::Impl");
}

Scope::Impl::Impl(const Scope& other, Tags::SingleUseScope,
                  const string& op_name)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(other.impl()->name_map_),
      refiner_(other.impl()->refiner_),
      scope_used_(new bool(false)),
      control_deps_(other.impl()->control_deps_),
      name_(other.impl()->name_),
      op_name_(op_name),
      exit_on_error_(other.impl()->exit_on_error_),
      kernel_label_(other.impl()->kernel_label_),
      device_(other.impl()->device_),
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_12(mht_12_v, 386, "", "./tensorflow/cc/framework/scope.cc", "Scope::Impl::Impl");
}

Scope::Impl::Impl(const Scope& other, Tags::ExitOnError)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(other.impl()->name_map_),
      refiner_(other.impl()->refiner_),
      scope_used_(other.impl()->scope_used_),
      control_deps_(other.impl()->control_deps_),
      name_(other.impl()->name_),
      op_name_(other.impl()->op_name_),
      exit_on_error_(true),
      kernel_label_(other.impl()->kernel_label_),
      device_(other.impl()->device_),
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_13(mht_13_v, 406, "", "./tensorflow/cc/framework/scope.cc", "Scope::Impl::Impl");
}

Scope::Impl::Impl(const Scope& other, Tags::KernelLabel,
                  const string& kernel_label)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(other.impl()->name_map_),
      refiner_(other.impl()->refiner_),
      scope_used_(other.impl()->scope_used_),
      control_deps_(other.impl()->control_deps_),
      name_(other.impl()->name_),
      op_name_(other.impl()->op_name_),
      exit_on_error_(other.impl()->exit_on_error_),
      kernel_label_(kernel_label),
      device_(other.impl()->device_),
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("kernel_label: \"" + kernel_label + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_14(mht_14_v, 428, "", "./tensorflow/cc/framework/scope.cc", "Scope::Impl::Impl");
}

Scope::Impl::Impl(const Scope& other, Tags::Colocate,
                  const Operation& colocate_with_op, bool clear_colocations)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(other.impl()->name_map_),
      refiner_(other.impl()->refiner_),
      scope_used_(other.impl()->scope_used_),
      control_deps_(other.impl()->control_deps_),
      name_(other.impl()->name_),
      op_name_(other.impl()->op_name_),
      exit_on_error_(other.impl()->exit_on_error_),
      kernel_label_(other.impl()->kernel_label_),
      device_(other.impl()->device_),
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(
          clear_colocations
              ? std::unordered_set<string>()
              : other.impl()->GetColocationConstraints(colocate_with_op)),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_15(mht_15_v, 452, "", "./tensorflow/cc/framework/scope.cc", "Scope::Impl::Impl");
}

Scope::Impl::Impl(const Scope& other, Tags::AssignedDevice,
                  const string& assigned_device)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(other.impl()->name_map_),
      refiner_(other.impl()->refiner_),
      scope_used_(other.impl()->scope_used_),
      control_deps_(other.impl()->control_deps_),
      name_(other.impl()->name_),
      op_name_(other.impl()->op_name_),
      exit_on_error_(other.impl()->exit_on_error_),
      kernel_label_(other.impl()->kernel_label_),
      device_(other.impl()->device_),
      assigned_device_(assigned_device),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("assigned_device: \"" + assigned_device + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_16(mht_16_v, 474, "", "./tensorflow/cc/framework/scope.cc", "Scope::Impl::Impl");
}

Scope::Impl::Impl(const Scope& other, Tags::XlaCluster,
                  const string& xla_cluster)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(other.impl()->name_map_),
      refiner_(other.impl()->refiner_),
      scope_used_(other.impl()->scope_used_),
      control_deps_(other.impl()->control_deps_),
      name_(other.impl()->name_),
      op_name_(other.impl()->op_name_),
      exit_on_error_(other.impl()->exit_on_error_),
      kernel_label_(other.impl()->kernel_label_),
      device_(other.impl()->device_),
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(xla_cluster),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("xla_cluster: \"" + xla_cluster + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_17(mht_17_v, 496, "", "./tensorflow/cc/framework/scope.cc", "Scope::Impl::Impl");
}

std::unordered_set<string> Scope::Impl::GetColocationConstraints(
    const Operation& colocate_with_op) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_18(mht_18_v, 502, "", "./tensorflow/cc/framework/scope.cc", "Scope::Impl::GetColocationConstraints");

  std::unordered_set<string> current_constraints(colocation_constraints_);
  const AttrSlice attrs = colocate_with_op.node()->attrs();
  std::vector<string> node_constraints;
  if (TryGetNodeAttr(attrs, kColocationAttrName, &node_constraints)) {
    for (const string& entry : node_constraints) {
      StringPiece s(entry);
      if (absl::ConsumePrefix(&s, kColocationGroupPrefix)) {
        current_constraints.emplace(s);
      }
    }
  } else {
    current_constraints.insert(colocate_with_op.node()->name());
  }
  return current_constraints;
}

bool Scope::ok() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_19(mht_19_v, 522, "", "./tensorflow/cc/framework/scope.cc", "Scope::ok");
 return impl()->status_->ok(); }

Graph* Scope::graph() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_20(mht_20_v, 527, "", "./tensorflow/cc/framework/scope.cc", "Scope::graph");
 return impl()->graph_.get(); }

std::shared_ptr<Graph> Scope::graph_as_shared_ptr() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_21(mht_21_v, 532, "", "./tensorflow/cc/framework/scope.cc", "Scope::graph_as_shared_ptr");

  return impl()->graph_;
}

Status Scope::status() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_22(mht_22_v, 539, "", "./tensorflow/cc/framework/scope.cc", "Scope::status");
 return *impl()->status_; }

const std::vector<Operation>& Scope::control_deps() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_23(mht_23_v, 544, "", "./tensorflow/cc/framework/scope.cc", "Scope::control_deps");

  return impl()->control_deps_;
}

void Scope::UpdateStatus(const Status& s) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_24(mht_24_v, 551, "", "./tensorflow/cc/framework/scope.cc", "Scope::UpdateStatus");

  impl()->status_->Update(s);
  if (impl()->exit_on_error_ && !ok()) {
    LOG(FATAL) << *impl()->status_;
  }
}

Status Scope::ToGraphDef(GraphDef* gdef) const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_25(mht_25_v, 561, "", "./tensorflow/cc/framework/scope.cc", "Scope::ToGraphDef");

  if (!ok()) {
    return *impl()->status_;
  }
  graph()->ToGraphDef(gdef);
  return Status::OK();
}

Status Scope::ToGraph(Graph* g, GraphConstructorOptions opts) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_26(mht_26_v, 572, "", "./tensorflow/cc/framework/scope.cc", "Scope::ToGraph");

  if (ok()) {
    GraphDef graph_def;
    graph()->ToGraphDef(&graph_def);
    UpdateStatus(ConvertGraphDefToGraph(opts, std::move(graph_def), g));
  }
  return *impl()->status_;
}

void Scope::UpdateBuilder(NodeBuilder* builder) const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_27(mht_27_v, 584, "", "./tensorflow/cc/framework/scope.cc", "Scope::UpdateBuilder");

  std::vector<Node*> control_inputs;
  for (const auto& op : impl()->control_deps_) {
    control_inputs.push_back(op.node());
  }
  builder->ControlInputs(control_inputs);

  if (!impl()->kernel_label_.empty()) {
    builder->Attr("_kernel", impl()->kernel_label_);
  }

  if (!impl()->colocation_constraints_.empty()) {
    std::vector<string> constraints(impl()->colocation_constraints_.begin(),
                                    impl()->colocation_constraints_.end());
    // Sort the set.
    std::sort(constraints.begin(), constraints.end());
    // Add loc:@ prefix
    std::transform(constraints.begin(), constraints.end(), constraints.begin(),
                   [](const string& s) {
                     return strings::StrCat(kColocationGroupPrefix, s);
                   });
    builder->Attr(kColocationAttrName, constraints);
  }
  if (!impl()->device_.empty()) {
    builder->Device(impl()->device_);
  }
  if (!impl()->assigned_device_.empty()) {
    builder->AssignedDevice(impl()->assigned_device_);
  }
  if (!impl()->xla_cluster_.empty()) {
    builder->XlaCluster(impl()->xla_cluster_);
  }
}

string Scope::Impl::GetUniqueName(const string& prefix,
                                  bool check_single_use) const {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_28(mht_28_v, 623, "", "./tensorflow/cc/framework/scope.cc", "Scope::Impl::GetUniqueName");

  if (check_single_use && single_use_scope()) {
    if (*scope_used_) {
      *status_ =
          errors::AlreadyExists(prefix, " already exists in the current scope");
      return "";
    }
    *scope_used_ = true;
    return prefix;
  }
  auto entry = name_map_->find(prefix);
  if (entry == name_map_->end()) {
    name_map_->insert({prefix, 0});
    return prefix;
  }
  string unique_name;
  do {
    unique_name = strings::StrCat(prefix, kSuffixSeparator, ++entry->second);
  } while (name_map_->find(unique_name) != name_map_->end());
  name_map_->insert({unique_name, 0});
  return unique_name;
}

string Scope::Impl::GetNameForOp(const string& default_name) const {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("default_name: \"" + default_name + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_29(mht_29_v, 650, "", "./tensorflow/cc/framework/scope.cc", "Scope::Impl::GetNameForOp");

  const string unique_name =
      GetUniqueName(default_name, true /* check_single_use */);
  const string sep =
      name_.empty() || unique_name.empty() ? "" : kScopeSeparator;
  return strings::StrCat(name_, sep, unique_name);
}

string Scope::GetUniqueNameForOp(const string& default_name) const {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("default_name: \"" + default_name + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_30(mht_30_v, 662, "", "./tensorflow/cc/framework/scope.cc", "Scope::GetUniqueNameForOp");

  if (impl()->single_use_scope()) {
    if (impl()->op_name_.empty() || *impl()->scope_used_) {
      *impl()->status_ =
          errors::InvalidArgument("Cannot get a unique name in this scope");
      return "";
    }
    *impl()->scope_used_ = true;
    return impl()->op_name_;
  }
  return impl()->op_name_.empty() ? impl()->GetNameForOp(default_name)
                                  : impl()->GetNameForOp(impl()->op_name_);
}

Scope Scope::NewSubScope(const string& child_scope_name) const {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("child_scope_name: \"" + child_scope_name + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_31(mht_31_v, 680, "", "./tensorflow/cc/framework/scope.cc", "Scope::NewSubScope");

  if (child_scope_name.empty()) {
    return Scope(new Impl(*this, Impl::Tags::ScopeName(), impl()->name_,
                          true /* copy_names */));
  }
  const string unique_name =
      impl()->GetUniqueName(child_scope_name, false /* check_single_use */);
  const string sep =
      impl()->name_.empty() || unique_name.empty() ? "" : kScopeSeparator;
  return Scope(new Impl(*this, Impl::Tags::ScopeName(),
                        strings::StrCat(impl()->name_, sep, unique_name),
                        false /* copy_names */));
}

Scope Scope::WithOpNameImpl(const string& op_name) const {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_32(mht_32_v, 698, "", "./tensorflow/cc/framework/scope.cc", "Scope::WithOpNameImpl");

  if (impl()->single_use_scope()) {
    UpdateStatus(errors::InvalidArgument("Cannot set op name ", op_name,
                                         " on this scope"));
    return *this;
  }
  return Scope(new Impl(*this, Impl::Tags::OpName(), impl()->name_, op_name));
}

Scope Scope::WithControlDependencies(
    const gtl::ArraySlice<Operation>& control_deps) const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_33(mht_33_v, 711, "", "./tensorflow/cc/framework/scope.cc", "Scope::WithControlDependencies");

  return Scope(
      new Impl(*this, Impl::Tags::ControlDeps(),
               std::vector<Operation>(control_deps.begin(), control_deps.end()),
               /* clear_control_deps */ false));
}

Scope Scope::WithControlDependencies(const Output& control_dep) const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_34(mht_34_v, 721, "", "./tensorflow/cc/framework/scope.cc", "Scope::WithControlDependencies");

  return Scope(new Impl(*this, Impl::Tags::ControlDeps(),
                        std::vector<Operation>(1, control_dep.op()),
                        /* clear_control_deps */ false));
}

Scope Scope::WithNoControlDependencies() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_35(mht_35_v, 730, "", "./tensorflow/cc/framework/scope.cc", "Scope::WithNoControlDependencies");

  return Scope(new Impl(*this, Impl::Tags::ControlDeps(),
                        std::vector<Operation>(),
                        /* clear_control_deps */ true));
}

Scope Scope::WithDevice(const string& device) const {
   std::vector<std::string> mht_36_v;
   mht_36_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_36(mht_36_v, 740, "", "./tensorflow/cc/framework/scope.cc", "Scope::WithDevice");

  return Scope(new Impl(*this, Impl::Tags::Device(), device));
}

Scope Scope::WithAssignedDevice(const string& assigned_device) const {
   std::vector<std::string> mht_37_v;
   mht_37_v.push_back("assigned_device: \"" + assigned_device + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_37(mht_37_v, 748, "", "./tensorflow/cc/framework/scope.cc", "Scope::WithAssignedDevice");

  return Scope(new Impl(*this, Impl::Tags::AssignedDevice(), assigned_device));
}

Scope Scope::WithXlaCluster(const string& xla_cluster) const {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("xla_cluster: \"" + xla_cluster + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_38(mht_38_v, 756, "", "./tensorflow/cc/framework/scope.cc", "Scope::WithXlaCluster");

  return Scope(new Impl(*this, Impl::Tags::XlaCluster(), xla_cluster));
}

Scope Scope::ColocateWith(const Operation& op) const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_39(mht_39_v, 763, "", "./tensorflow/cc/framework/scope.cc", "Scope::ColocateWith");

  return Scope(new Impl(*this, Impl::Tags::Colocate(), op,
                        /* clear_colocations */ false));
}

Scope Scope::ClearColocation() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_40(mht_40_v, 771, "", "./tensorflow/cc/framework/scope.cc", "Scope::ClearColocation");

  return Scope(new Impl(*this, Impl::Tags::Colocate(), Operation(),
                        /* clear_colocations */ true));
}

Scope Scope::ExitOnError() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_41(mht_41_v, 779, "", "./tensorflow/cc/framework/scope.cc", "Scope::ExitOnError");

  return Scope(new Impl(*this, Impl::Tags::ExitOnError()));
}

Scope Scope::WithKernelLabel(const string& kernel_label) const {
   std::vector<std::string> mht_42_v;
   mht_42_v.push_back("kernel_label: \"" + kernel_label + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_42(mht_42_v, 787, "", "./tensorflow/cc/framework/scope.cc", "Scope::WithKernelLabel");

  return Scope(new Impl(*this, Impl::Tags::KernelLabel(), kernel_label));
}

CompositeOpScopes Scope::GetCompositeOpScopes(
    const string& composite_op_name) const {
   std::vector<std::string> mht_43_v;
   mht_43_v.push_back("composite_op_name: \"" + composite_op_name + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_43(mht_43_v, 796, "", "./tensorflow/cc/framework/scope.cc", "Scope::GetCompositeOpScopes");

  if (impl()->op_name_.empty() && composite_op_name.empty()) {
    UpdateStatus(errors::InvalidArgument(
        "Cannot create composite op scopes with empty name"));
    return {*this, *this};
  }
  if (!impl()->single_use_scope()) {
    Scope child = NewSubScope(impl()->op_name_.empty() ? composite_op_name
                                                       : impl()->op_name_);
    const string child_op_sep = impl()->name_.empty() ? "" : kSuffixSeparator;
    const string child_name =
        strings::StrCat(impl()->name_, child_op_sep, child.impl()->name_);
    return {child,
            Scope(new Impl(child, Impl::Tags::SingleUseScope(), child_name))};
  } else {
    return {Scope(new Impl(*this, Impl::Tags::ScopeName(), impl()->op_name_,
                           true /* copy_names */)),
            *this};
  }
}

Status Scope::DoShapeInference(Node* node) const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_44(mht_44_v, 820, "", "./tensorflow/cc/framework/scope.cc", "Scope::DoShapeInference");

  if (impl_->disable_shape_inference_) return Status::OK();
  return impl_->refiner_->AddNode(node);
}

class InternalScope {
 public:
  // NewScope doesn't take ownership of the inputs.
  static Scope NewScope(Graph* graph, Status* status, ShapeRefiner* refiner) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_45(mht_45_v, 831, "", "./tensorflow/cc/framework/scope.cc", "NewScope");

    Scope::Impl::NameMap* name_map = new Scope::Impl::NameMap;
    for (const Node* node : graph->nodes()) {
      const string& name = node->name();
      (*name_map)[name] = 0;
      // Add all name prefixes ('/' separated).
      size_t idx = -1;
      while ((idx = name.find(kScopeSeparator, idx + 1)) != string::npos) {
        (*name_map)[name.substr(0, idx)] = 0;
      }
    }
    // We provide null destructors for these shared ptrs (except for name_map)
    // since the caller owns them and doesn't want the scope to destroy them.
    return Scope(new Scope::Impl(
        std::shared_ptr<Graph>(graph, [](Graph*) {}),
        std::shared_ptr<Status>(status, [](Status*) {}),
        std::shared_ptr<Scope::Impl::NameMap>(name_map),
        std::shared_ptr<ShapeRefiner>(refiner, [](ShapeRefiner*) {})));
  }
};

Scope NewInternalScope(Graph* graph, Status* status, ShapeRefiner* refiner) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_46(mht_46_v, 855, "", "./tensorflow/cc/framework/scope.cc", "NewInternalScope");

  return InternalScope::NewScope(graph, status, refiner);
}

Status CreateOutputWithScope(string op_name,
                             absl::Span<const ::tensorflow::Input> inputs,
                             const Scope& scope, Output* output) {
   std::vector<std::string> mht_47_v;
   mht_47_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSscopeDTcc mht_47(mht_47_v, 865, "", "./tensorflow/cc/framework/scope.cc", "CreateOutputWithScope");

  TF_RETURN_IF_ERROR(scope.status());
  const auto unique_name = scope.GetUniqueNameForOp(op_name);
  auto builder = ::tensorflow::NodeBuilder(unique_name, op_name);
  for (const auto& input : inputs) {
    TF_RETURN_IF_ERROR(scope.status());
    builder = builder.Input(input.node());
  }
  ::tensorflow::Node* ret;
  scope.UpdateBuilder(&builder);
  TF_RETURN_IF_ERROR(scope.status());
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  TF_RETURN_IF_ERROR(scope.status());
  *output = Output(ret, 0);
  return Status::OK();
}

}  // namespace tensorflow
