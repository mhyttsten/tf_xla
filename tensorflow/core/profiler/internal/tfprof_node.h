/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_NODE_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_NODE_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh() {
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


#include <map>
#include <set>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/profiler/tfprof_log.pb.h"
#include "tensorflow/core/profiler/tfprof_options.h"

namespace tensorflow {
namespace tfprof {
std::vector<int64_t> ShapeProtoToVec(const TensorShapeProto& shape_pb);

TensorShapeProto VecToShapeProto(const std::vector<int64_t>& shape_vec);

class TFGraphNode;

class CallStack {
 public:
  class Trace {
   public:
    Trace(const CodeDef::Trace* trace,
          const std::map<int64_t, string>* id_to_string)
        : trace_(trace), id_to_string_(id_to_string) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_0(mht_0_v, 219, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "Trace");
}

    const int32 lineno() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_1(mht_1_v, 224, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "lineno");
 return trace_->lineno(); }
    string file() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_2(mht_2_v, 228, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "file");

      // Backward compatible with old proto files.
      if (!trace_->file().empty()) return trace_->file();
      return id_to_string_->at(trace_->file_id());
    }
    string function() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_3(mht_3_v, 236, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "function");

      // Backward compatible with old proto files.
      if (!trace_->function().empty()) return trace_->function();
      return id_to_string_->at(trace_->function_id());
    }
    int32 func_start_line() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_4(mht_4_v, 244, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "func_start_line");
 return trace_->func_start_line(); }

   private:
    const CodeDef::Trace* trace_;
    const std::map<int64_t, string>* id_to_string_;
  };

  CallStack(const CodeDef& def, const std::map<int64_t, string>* id_to_string)
      : def_(def) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_5(mht_5_v, 255, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "CallStack");

    traces_.reserve(def.traces_size());
    for (const auto& t : def_.traces()) {
      traces_.emplace_back(&t, id_to_string);
    }
  }

  const CodeDef& code_def() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_6(mht_6_v, 265, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "code_def");
 return def_; }
  const std::vector<Trace>& traces() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_7(mht_7_v, 269, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "traces");
 return traces_; }

 private:
  std::vector<Trace> traces_;
  CodeDef def_;
};

class ExecStep {
 public:
  ExecStep() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_8(mht_8_v, 281, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "ExecStep");
}

  void AddTimeStats(const string& dev, const NodeExecStats& step_stat);

  void AddMemoryStats(const string& dev, const NodeExecStats& step_stat);

  int64_t run_count() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_9(mht_9_v, 290, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "run_count");
 return exec_.run_count(); }
  // The execution time of an op. If it runs on accelerator, then it's
  // accelerator_exec_micros(). Otherwise, it's CPU time.
  int64_t exec_micros() const;
  // The accelerator execution time of an op. 0 if not run on accelerator.
  int64_t accelerator_exec_micros() const;
  // The cpu execution time of an op.
  int64_t cpu_exec_micros() const;

  const std::map<string, std::vector<std::pair<int64_t, int64_t>>>& op_execs()
      const {
    return op_execs_;
  }
  const std::map<string, std::vector<std::pair<int64_t, int64_t>>>& cpu_execs()
      const {
    return cpu_execs_;
  }
  int64_t all_start_micros() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_10(mht_10_v, 310, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "all_start_micros");
 return exec_.all_start_micros(); }
  int64_t latest_end_micros() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_11(mht_11_v, 314, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "latest_end_micros");
 return exec_.latest_end_micros(); }
  int64_t lastest_schedule_end_micros() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_12(mht_12_v, 318, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "lastest_schedule_end_micros");

    int64_t ret = 0;
    for (const auto& exec : cpu_execs_) {
      for (const auto& pair : exec.second) {
        ret = std::max(ret, pair.first + pair.second);
      }
    }
    return ret;
  }
  int64_t requested_bytes() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_13(mht_13_v, 330, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "requested_bytes");

    int64_t requested_bytes = 0;
    for (const ExecMemory& exec : memory_execs_) {
      requested_bytes += exec.requested_bytes();
    }
    return requested_bytes;
  }
  int64_t peak_bytes() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_14(mht_14_v, 340, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "peak_bytes");

    int64_t peak_bytes = 0;
    for (const ExecMemory& exec : memory_execs_) {
      peak_bytes += exec.peak_bytes();
    }
    return peak_bytes;
  }
  int64_t residual_bytes() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_15(mht_15_v, 350, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "residual_bytes");

    int64_t residual_bytes = 0;
    for (const ExecMemory& exec : memory_execs_) {
      residual_bytes += exec.residual_bytes();
    }
    return residual_bytes;
  }
  int64_t output_bytes() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_16(mht_16_v, 360, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "output_bytes");

    int64_t output_bytes = 0;
    for (const ExecMemory& exec : memory_execs_) {
      output_bytes += exec.output_bytes();
    }
    return output_bytes;
  }
  int64_t accelerator_temp_bytes() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_17(mht_17_v, 370, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "accelerator_temp_bytes");

    int64_t accelerator_temp_bytes = 0;
    for (const ExecMemory& exec : memory_execs_) {
      accelerator_temp_bytes += exec.accelerator_temp_bytes();
    }
    return accelerator_temp_bytes;
  }
  int64_t host_temp_bytes() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_18(mht_18_v, 380, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "host_temp_bytes");

    int64_t host_temp_bytes = 0;
    for (const ExecMemory& exec : memory_execs_) {
      host_temp_bytes += exec.host_temp_bytes();
    }
    return host_temp_bytes;
  }
  int64_t accelerator_persistent_bytes() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_19(mht_19_v, 390, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "accelerator_persistent_bytes");

    int64_t accelerator_persistent_bytes = 0;
    for (const ExecMemory& exec : memory_execs_) {
      accelerator_persistent_bytes += exec.accelerator_persistent_bytes();
    }
    return accelerator_persistent_bytes;
  }
  int64_t host_persistent_bytes() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_20(mht_20_v, 400, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "host_persistent_bytes");

    int64_t host_persistent_bytes = 0;
    for (const ExecMemory& exec : memory_execs_) {
      host_persistent_bytes += exec.host_persistent_bytes();
    }
    return host_persistent_bytes;
  }
  std::map<int64_t, int64_t> allocator_bytes_in_use() const {
    std::map<int64_t, int64_t> bytes_in_use;
    for (const ExecMemory& exec : memory_execs_) {
      bytes_in_use[exec.memory_micros()] = exec.allocator_bytes_in_use();
    }
    return bytes_in_use;
  }

  const std::vector<AllocationRecord>& allocations() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_21(mht_21_v, 418, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "allocations");

    return allocations_;
  }

  const ExecProfile& ToProto() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_22(mht_22_v, 425, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "ToProto");

    exec_.mutable_accelerator_execs()->clear();
    for (const auto& e : accelerator_execs_) {
      auto& exec_time = (*exec_.mutable_accelerator_execs())[e.first];
      for (const auto& p : e.second) {
        auto* t = exec_time.mutable_times()->Add();
        t->add_int64_values(p.first);
        t->add_int64_values(p.second);
      }
    }

    exec_.mutable_cpu_execs()->clear();
    for (const auto& e : cpu_execs_) {
      auto& exec_time = (*exec_.mutable_cpu_execs())[e.first];
      for (const auto& p : e.second) {
        auto* t = exec_time.mutable_times()->Add();
        t->add_int64_values(p.first);
        t->add_int64_values(p.second);
      }
    }

    exec_.mutable_devices()->Clear();
    exec_.mutable_devices()->Reserve(devices_.size());
    for (const string& d : devices_) {
      exec_.add_devices(d);
    }
    exec_.mutable_allocations()->Clear();
    for (const auto& r : allocations_) {
      exec_.add_allocations()->MergeFrom(r);
    }

    exec_.mutable_memory_execs()->Clear();
    for (const auto& m : memory_execs_) {
      exec_.add_memory_execs()->MergeFrom(m);
    }
    return exec_;
  }

  void FromProto(const ExecProfile& exec) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_23(mht_23_v, 466, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "FromProto");

    exec_.Clear();
    exec_.MergeFrom(exec);

    devices_.clear();
    devices_.insert(exec.devices().begin(), exec.devices().end());

    accelerator_execs_.clear();
    cpu_execs_.clear();
    op_execs_.clear();

    allocations_.clear();
    memory_execs_.clear();

    for (const auto& exec_time : exec_.accelerator_execs()) {
      auto& exec = accelerator_execs_[exec_time.first];
      auto& op_exec = op_execs_[exec_time.first];
      for (const auto& p : exec_time.second.times()) {
        exec.push_back(std::make_pair(p.int64_values(0), p.int64_values(1)));
        op_exec.push_back(std::make_pair(p.int64_values(0), p.int64_values(1)));
      }
    }
    for (const auto& exec_time : exec_.cpu_execs()) {
      auto& exec = cpu_execs_[exec_time.first];
      auto& op_exec = op_execs_[exec_time.first];
      for (const auto& p : exec_time.second.times()) {
        exec.push_back(std::make_pair(p.int64_values(0), p.int64_values(1)));
        op_exec.push_back(std::make_pair(p.int64_values(0), p.int64_values(1)));
      }
    }
    for (const auto& r : exec_.allocations()) {
      allocations_.push_back(r);
    }
    for (const auto& m : exec_.memory_execs()) {
      memory_execs_.push_back(m);
    }
  }

 private:
  ExecProfile exec_;
  // device -> vector of {op_start_micros, op_exec_micros} pairs.
  // accelerator_execs: gpu:id/stream:all -> {op_start_micros, op_exec_micros}
  // For accelerator, vector size can be larger than 1, multiple kernel fires
  // or in tf.while_loop.
  std::map<string, std::vector<std::pair<int64_t, int64_t>>> accelerator_execs_;
  // cpu_execs: cpu/gpu:id -> {op_start_micros, op_exec_micros}
  // For cpu, vector size can be larger than 1 if in tf.while_loop.
  std::map<string, std::vector<std::pair<int64_t, int64_t>>> cpu_execs_;
  // combines accelerator_execs_ and cpu_execs_.
  std::map<string, std::vector<std::pair<int64_t, int64_t>>> op_execs_;
  // Each ExecMemory corresponds to one scheduling of the op. Normally,
  // there are multiple schedulings in while_loop.
  std::vector<ExecMemory> memory_execs_;
  // All devices the op is associated with (e.g. gpu:0 (scheduling),
  // gpu:0:stream:xx (kernel exec), cpu:0 host)
  std::set<string> devices_;

  // The history of accelerator allocations and deallocations of this step.
  std::vector<AllocationRecord> allocations_;
};

#define GRAPH_NODE_BYTES(type)             \
  do {                                     \
    if (execs_.empty()) {                  \
      return 0;                            \
    }                                      \
    if (step >= 0) {                       \
      auto exec = execs_.find(step);       \
      if (exec == execs_.end()) return 0;  \
      return exec->second.type##_bytes();  \
    }                                      \
                                           \
    int64_t bytes = 0;                     \
    for (const auto& exec : execs_) {      \
      bytes += exec.second.type##_bytes(); \
    }                                      \
    return bytes / execs_.size();          \
  } while (0)

class TFGraphNode {
 public:
  TFGraphNode(const ProfileNode& node, const ProfileProto& profile,
              const std::map<int64_t, string>* id_to_string,
              const std::map<string, std::unique_ptr<TFGraphNode>>* nodes_map) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_24(mht_24_v, 552, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "TFGraphNode");

    nodes_map_ = nodes_map;
    FromProto(node, profile, id_to_string);
  }

  TFGraphNode(const NodeDef* node, int64_t id,
              const std::map<string, std::unique_ptr<TFGraphNode>>* nodes_map) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_25(mht_25_v, 561, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "TFGraphNode");

    nodes_map_ = nodes_map;
    node_.set_id(id);
    node_.set_name(node->name());
    node_.set_op(node->op());
    node_.set_float_ops(0);

    for (const auto& attr : node->attr()) {
      (*node_.mutable_attrs())[attr.first].MergeFrom(attr.second);
      if (attr.first == "shape" && attr.second.has_shape()) {
        if (!shape_.empty()) {
          absl::FPrintF(stderr, "Found duplicated shapes!\n");
          continue;
        }
        shape_ = ShapeProtoToVec(attr.second.shape());
      } else if (attr.first == "_output_shapes" && attr.second.has_list()) {
        if (!output_shapes_.empty()) {
          absl::FPrintF(stderr, "Found duplicated output shapes!\n");
          continue;
        }
        for (int i = 0; i < attr.second.list().shape_size(); ++i) {
          output_shapes_[i] = ShapeProtoToVec(attr.second.list().shape(i));
        }
      }
    }
    op_types_.insert(node->op());
  }

  void AddInput(const string& input, int64_t output_index, int input_idx) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("input: \"" + input + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_26(mht_26_v, 593, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "AddInput");

    inputs_[input_idx] = input;
    src_output_idx_[input] = output_index;
  }

  void AddOpType(const string& op_type) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("op_type: \"" + op_type + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_27(mht_27_v, 602, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "AddOpType");
 op_types_.insert(op_type); }

  void AddStepStat(int64_t step, const string& device,
                   const NodeExecStats& step_stat);

  void AddFloatOps(int64_t float_ops) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_28(mht_28_v, 610, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "AddFloatOps");
 node_.set_float_ops(float_ops); }

  // TODO(xpan): This could take a lot of memory.
  void AddCode(const CodeDef& code,
               const std::map<int64_t, string>* id_to_string) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_29(mht_29_v, 617, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "AddCode");

    if (!call_stack_) {
      call_stack_.reset(new CallStack(code, id_to_string));
    }
  }

  const string& name() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_30(mht_30_v, 626, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "name");
 return node_.name(); }
  int64_t id() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_31(mht_31_v, 630, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "id");
 return node_.id(); }
  const string& op() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_32(mht_32_v, 634, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "op");
 return node_.op(); }
  const ProfileNode& node() {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_33(mht_33_v, 638, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "node");
 return node_; }

  bool trackable(int64_t step) const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_34(mht_34_v, 643, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "trackable");

    auto exec = execs_.find(step);
    if (exec == execs_.end()) return false;

    if (exec->second.all_start_micros() == 0) return false;
    if (node_.canonical_device().empty() || node_.host_device().empty()) {
      return false;
    }
    return true;
  }

  const ProfileNode& ToProto(
      const std::map<string, std::unique_ptr<TFGraphNode>>& nodes_map) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_35(mht_35_v, 658, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "ToProto");

    node_.clear_shape();
    node_.mutable_shape()->Reserve(shape().size());
    for (int64_t s : shape()) {
      node_.add_shape(s);
    }

    node_.clear_op_types();
    node_.mutable_op_types()->Reserve(op_types().size());
    for (const string& t : op_types()) {
      node_.add_op_types(t);
    }

    node_.clear_execs();
    for (auto& exec : execs_) {
      auto& exec_pb = (*node_.mutable_execs())[exec.first];
      exec_pb.MergeFrom(exec.second.ToProto());
    }

    node_.clear_inputs();
    for (const auto& inp : inputs_) {
      (*node_.mutable_inputs())[inp.first] = nodes_map.at(inp.second)->id();
    }

    node_.clear_input_shapes();
    for (const auto& s : input_shapes_) {
      auto& shape = (*node_.mutable_input_shapes())[s.first];
      for (int64_t d : s.second) {
        shape.add_int64_values(d);
      }
    }

    node_.clear_output_shapes();
    for (const auto& s : output_shapes_) {
      auto& shape = (*node_.mutable_output_shapes())[s.first];
      for (int64_t d : s.second) {
        shape.add_int64_values(d);
      }
    }

    node_.clear_src_output_index();
    for (const auto& s : src_output_idx_) {
      int64_t id = nodes_map.at(s.first)->id();
      (*node_.mutable_src_output_index())[id] = s.second;
    }

    if (call_stack_) {
      node_.clear_trace();
      node_.mutable_trace()->MergeFrom(call_stack_->code_def());
    }
    return node_;
  }

  void FromProto(const ProfileNode& node, const ProfileProto& profile,
                 const std::map<int64_t, string>* id_to_string) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_36(mht_36_v, 715, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "FromProto");

    node_.Clear();
    node_.MergeFrom(node);

    call_stack_.reset(new CallStack(node.trace(), id_to_string));

    op_types_.clear();
    op_types_.insert(node_.op_types().begin(), node_.op_types().end());

    shape_.clear();
    for (int64_t s : node_.shape()) {
      shape_.push_back(s);
    }

    execs_.clear();
    for (const auto& exec_pb : node.execs()) {
      auto& exec = execs_[exec_pb.first];
      exec.FromProto(exec_pb.second);
    }

    inputs_.clear();
    for (const auto& inp : node.inputs()) {
      inputs_[inp.first] = profile.nodes().at(inp.second).name();
    }

    input_shapes_.clear();
    for (const auto& s : node.input_shapes()) {
      auto& shape = input_shapes_[s.first];
      for (const int64_t d : s.second.int64_values()) {
        shape.push_back(d);
      }
    }

    output_shapes_.clear();
    for (const auto& s : node.output_shapes()) {
      auto& shape = output_shapes_[s.first];
      for (const int64_t d : s.second.int64_values()) {
        shape.push_back(d);
      }
    }

    src_output_idx_.clear();
    for (const auto& s : node.src_output_index()) {
      src_output_idx_[profile.nodes().at(s.first).name()] = s.second;
    }
  }

  const std::map<int32, string>& inputs() const { return inputs_; }

  // Number of times the graph node is executed. When step < 0, the
  // average number of times executed across all steps.
  int64_t run_count(int64_t step) const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_37(mht_37_v, 769, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "run_count");

    if (execs_.empty()) {
      return 0;
    }
    if (step >= 0) {
      auto exec = execs_.find(step);
      if (exec == execs_.end()) {
        return 0;
      }
      return exec->second.run_count();
    }
    int64_t total_run_count = 0;
    for (const auto& exec : execs_) {
      total_run_count += exec.second.run_count();
    }
    return total_run_count / execs_.size();
  }
  // This is overall computation time, including both cpu and accelerator.
  // Note, cpu and accelerator might or might not run in parallel.
  int64_t exec_micros(int64_t step) const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_38(mht_38_v, 791, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "exec_micros");

    // Empty when no RunMetadata is provided.
    if (execs_.empty()) {
      return 0;
    }
    if (step >= 0) {
      auto exec = execs_.find(step);
      if (exec == execs_.end()) {
        return 0;
      }
      return exec->second.exec_micros();
    }

    int64_t total_micros = 0;
    for (const auto& exec : execs_) {
      total_micros += exec.second.exec_micros();
    }
    return total_micros / execs_.size();
  }

  // This is accelerator computation time of a step, or average of
  // multiple step, when step < 0.
  int64_t accelerator_exec_micros(int64_t step) const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_39(mht_39_v, 816, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "accelerator_exec_micros");

    // Empty when no RunMetadata is provided.
    if (execs_.empty()) {
      return 0;
    }
    if (step >= 0) {
      auto exec = execs_.find(step);
      if (exec == execs_.end()) {
        return 0;
      }
      return exec->second.accelerator_exec_micros();
    }

    int64_t total_micros = 0;
    for (const auto& exec : execs_) {
      total_micros += exec.second.accelerator_exec_micros();
    }
    return total_micros / execs_.size();
  }

  // This is cpu computation time of a step, or average of
  // multiple step, when step < 0.
  int64_t cpu_exec_micros(int64_t step) const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_40(mht_40_v, 841, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "cpu_exec_micros");

    // Empty when no RunMetadata is provided.
    if (execs_.empty()) {
      return 0;
    }
    if (step >= 0) {
      auto exec = execs_.find(step);
      if (exec == execs_.end()) {
        return 0;
      }
      return exec->second.cpu_exec_micros();
    }

    int64_t total_micros = 0;
    for (const auto& exec : execs_) {
      total_micros += exec.second.cpu_exec_micros();
    }
    return total_micros / execs_.size();
  }

  int64_t requested_bytes(int64_t step) const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_41(mht_41_v, 864, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "requested_bytes");
 GRAPH_NODE_BYTES(requested); }
  int64_t peak_bytes(int64_t step) const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_42(mht_42_v, 868, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "peak_bytes");
 GRAPH_NODE_BYTES(peak); }
  int64_t residual_bytes(int64_t step) const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_43(mht_43_v, 872, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "residual_bytes");
 GRAPH_NODE_BYTES(residual); }
  int64_t output_bytes(int64_t step) const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_44(mht_44_v, 876, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "output_bytes");
 GRAPH_NODE_BYTES(output); }

  int64_t all_start_micros(int64_t step) const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_45(mht_45_v, 881, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "all_start_micros");

    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return 0;
    }
    return exec->second.all_start_micros();
  }

  int64_t latest_end_micros(int64_t step) const {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_46(mht_46_v, 892, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "latest_end_micros");

    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return 0;
    }
    return exec->second.latest_end_micros();
  }

  int64_t lastest_schedule_end_micros(int64_t step) const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_47(mht_47_v, 903, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "lastest_schedule_end_micros");

    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return 0;
    }
    return exec->second.lastest_schedule_end_micros();
  }

  const std::map<string, std::vector<std::pair<int64_t, int64_t>>>& op_execs(
      int64_t step) const {
    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return empty_execs_;
    }
    return exec->second.op_execs();
  }
  const std::map<string, std::vector<std::pair<int64_t, int64_t>>>& cpu_execs(
      int64_t step) const {
    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return empty_execs_;
    }
    return exec->second.cpu_execs();
  }

  const std::map<int64_t, ExecStep>& all_op_execs() const { return execs_; }

  int64_t accelerator_temp_bytes(int64_t step) const {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_48(mht_48_v, 933, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "accelerator_temp_bytes");

    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return 0;
    }
    return exec->second.accelerator_temp_bytes();
  }
  int64_t host_temp_bytes(int64_t step) const {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_49(mht_49_v, 943, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "host_temp_bytes");

    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return 0;
    }
    return exec->second.host_temp_bytes();
  }
  int64_t accelerator_persistent_bytes() const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_50(mht_50_v, 953, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "accelerator_persistent_bytes");

    int64_t persistent_bytes = 0;
    for (const auto& exec : execs_) {
      persistent_bytes = std::max(persistent_bytes,
                                  exec.second.accelerator_persistent_bytes());
    }
    return persistent_bytes;
  }
  const std::map<int64_t, int64_t> allocator_bytes_in_use(int64_t step) const {
    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return empty_bytes_in_use_;
    }
    return exec->second.allocator_bytes_in_use();
  }

  const std::vector<AllocationRecord>& allocations(int64_t step) const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_51(mht_51_v, 972, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "allocations");

    auto exec = execs_.find(step);
    if (exec == execs_.end()) {
      return empty_allocations_;
    }
    return exec->second.allocations();
  }

  int64_t parameters() const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_52(mht_52_v, 983, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "parameters");

    if (!shape().empty()) {
      int64_t params = 1;
      bool complete_shape = true;
      for (int64_t d : shape()) {
        // Sometimes parameters could be <0 when a dim is unknown.
        if (d < 0) {
          complete_shape = false;
          break;
        }
        params *= d;
      }
      if (complete_shape) {
        return params;
      } else {
        absl::FPrintF(stderr, "Incomplete shape.\n");
      }
    }
    return 0;
  }

  int64_t float_ops(int64_t step) const {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_53(mht_53_v, 1007, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "float_ops");

    // If not run, return static analysis.
    if (execs_.empty()) {
      return node_.float_ops();
    }
    // Otherwise, return dynamic float_ops.
    return node_.float_ops() * run_count(step);
  }
  const CallStack* call_stack() {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_54(mht_54_v, 1018, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "call_stack");
 return call_stack_.get(); }
  string canonical_device() const {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_55(mht_55_v, 1022, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "canonical_device");
 return node_.canonical_device(); }
  string host_device() const {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_56(mht_56_v, 1026, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "host_device");
 return node_.host_device(); }
  const std::set<string>& op_types() const {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_57(mht_57_v, 1030, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "op_types");
 return op_types_; }

  const AttrValue* op_attrs(const string& name) const {
   std::vector<std::string> mht_58_v;
   mht_58_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_58(mht_58_v, 1036, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "op_attrs");

    const auto it = node_.attrs().find(name);
    if (it == node_.attrs().end()) {
      return nullptr;
    }
    return &it->second;
  }

  const std::vector<int64_t>& shape() const {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_59(mht_59_v, 1047, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "shape");
 return shape_; }

  const std::map<int, std::vector<int64_t>>& output_shapes() const {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_60(mht_60_v, 1052, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "output_shapes");

    return output_shapes_;
  }

  const std::map<int, std::vector<int64_t>> input_shapes() const {
    std::map<int, std::vector<int64_t>> input_shapes;
    for (const auto& inp : inputs_) {
      // Always create an empty vec even if the shape info might be missing.
      std::vector<int64_t>& shape_vec = input_shapes[inp.first];
      if (!nodes_map_) continue;
      auto input_it = nodes_map_->find(inp.second);
      if (input_it == nodes_map_->end()) continue;
      auto output_it = src_output_idx_.find(inp.second);
      if (output_it == src_output_idx_.end()) continue;

      const TFGraphNode* input_node = input_it->second.get();
      if (!input_node) continue;
      const auto& output_shapes = input_node->output_shapes();
      const auto& output_shape = output_shapes.find(output_it->second);
      if (output_shape == output_shapes.end()) continue;

      if (output_shape != input_node->output_shapes().end()) {
        shape_vec.assign(output_shape->second.begin(),
                         output_shape->second.end());
      }
    }
    return input_shapes;
  }

 private:
  // maps graph node name to TFGraphNode. Not owned.
  const std::map<string, std::unique_ptr<TFGraphNode>>* nodes_map_;
  // inputs to the node. input index -> input node name.
  std::map<int, string> inputs_;
  // The output index of the source node.
  std::map<string, int32> src_output_idx_;
  // proto for serialize/deserialized representation of the node.
  ProfileNode node_;
  // Python call stack that creates the name.
  std::unique_ptr<CallStack> call_stack_;
  // Shape of the node (e.g. Variable) if available.
  std::vector<int64_t> shape_;
  // Won't missing input_idx. But some shapes might be empty (unknown).
  std::map<int, std::vector<int64_t>> input_shapes_;
  // Could miss output_idx if no _output_shapes attr. some shapes can also
  // be empty.
  std::map<int, std::vector<int64_t>> output_shapes_;

  std::set<string> op_types_;

  std::map<int64_t, ExecStep> execs_;

  // Placeholder for empty cases.
  std::map<int64_t, int64_t> empty_bytes_in_use_;
  std::map<string, std::vector<std::pair<int64_t, int64_t>>> empty_execs_;
  std::vector<AllocationRecord> empty_allocations_;
};

class TFMultiGraphNode {
 public:
  TFMultiGraphNode(const string& name)
      : name_(name),
        step_(-1),
        run_count_(0),
        exec_micros_(0),
        accelerator_exec_micros_(0),
        cpu_exec_micros_(0),
        requested_bytes_(0),
        peak_bytes_(0),
        residual_bytes_(0),
        output_bytes_(0),
        float_ops_(0),
        parameters_(0) {
   std::vector<std::string> mht_61_v;
   mht_61_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_61(mht_61_v, 1128, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "TFMultiGraphNode");
}

  bool SnapshotNodes(int64_t step, const std::vector<string>& type_regexes) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_62(mht_62_v, 1133, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "SnapshotNodes");

    run_count_ = 0;
    exec_micros_ = 0;
    accelerator_exec_micros_ = 0;
    cpu_exec_micros_ = 0;

    requested_bytes_ = 0;
    peak_bytes_ = 0;
    residual_bytes_ = 0;
    output_bytes_ = 0;

    float_ops_ = 0;
    parameters_ = 0;
    op_types_.clear();
    shapes_.clear();
    devices_.clear();
    snapshot_nodes_.clear();

    step_ = step;
    std::vector<const TFGraphNode*> nodes = pick_nodes(type_regexes);

    if (nodes.empty()) {
      return (type_regexes.size() == 1 && type_regexes[0] == ".*");
    }

    for (const TFGraphNode* node : nodes) {
      op_types_.insert(node->op_types().begin(), node->op_types().end());

      run_count_ += node->run_count(step);
      exec_micros_ += node->exec_micros(step);
      accelerator_exec_micros_ += node->accelerator_exec_micros(step);
      cpu_exec_micros_ += node->cpu_exec_micros(step);

      requested_bytes_ += node->requested_bytes(step);
      peak_bytes_ += node->peak_bytes(step);
      residual_bytes_ += node->residual_bytes(step);
      output_bytes_ += node->output_bytes(step);

      float_ops_ += node->float_ops(step);
      parameters_ += node->parameters();
      if (node->shape().size() > 0) {
        shapes_.push_back(node->shape());
      }
      devices_.insert(node->canonical_device());
      snapshot_nodes_[node->name()] = node;
    }
    return true;
  }

  int64_t step() const {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_63(mht_63_v, 1185, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "step");
 return step_; }

  void AddGraphNode(const TFGraphNode* node) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_64(mht_64_v, 1190, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "AddGraphNode");

    if (nodes_.find(node->name()) != nodes_.end()) {
      return;
    }
    nodes_[node->name()] = node;
  }

  const std::map<string, const TFGraphNode*>& graph_nodes() const {
    return snapshot_nodes_;
  }

  const string& name() const {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_65(mht_65_v, 1204, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "name");
 return name_; }

  int64_t run_count() const {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_66(mht_66_v, 1209, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "run_count");
 return run_count_; }
  int64_t exec_micros() const {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_67(mht_67_v, 1213, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "exec_micros");
 return exec_micros_; }
  int64_t accelerator_exec_micros() const {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_68(mht_68_v, 1217, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "accelerator_exec_micros");
 return accelerator_exec_micros_; }
  int64_t cpu_exec_micros() const {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_69(mht_69_v, 1221, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "cpu_exec_micros");
 return cpu_exec_micros_; }

  int64_t requested_bytes() const {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_70(mht_70_v, 1226, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "requested_bytes");
 return requested_bytes_; }
  int64_t peak_bytes() const {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_71(mht_71_v, 1230, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "peak_bytes");
 return peak_bytes_; }
  int64_t residual_bytes() const {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_72(mht_72_v, 1234, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "residual_bytes");
 return residual_bytes_; }
  int64_t output_bytes() const {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_73(mht_73_v, 1238, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "output_bytes");
 return output_bytes_; }

  int64_t float_ops() const {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_74(mht_74_v, 1243, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "float_ops");
 return float_ops_; }

  int64_t parameters() const {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_75(mht_75_v, 1248, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "parameters");
 return parameters_; }

  const std::set<string>& devices() const {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_76(mht_76_v, 1253, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "devices");
 return devices_; }

  const std::set<string>& op_types() const {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_77(mht_77_v, 1258, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "op_types");
 return op_types_; }

  const std::vector<std::vector<int64_t>>& shapes() const {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_nodeDTh mht_78(mht_78_v, 1263, "", "./tensorflow/core/profiler/internal/tfprof_node.h", "shapes");
 return shapes_; }

 private:
  std::vector<const TFGraphNode*> pick_nodes(
      const std::vector<string>& type_regexes) {
    if (type_regexes.empty()) {
      return {};
    }
    std::vector<const TFGraphNode*> ret;
    if (type_regexes.size() == 1 && type_regexes[0] == ".*") {
      for (const auto& n : nodes_) {
        ret.push_back(n.second);
      }
      return ret;
    }

    for (const string& regex : type_regexes) {
      for (const auto& n : nodes_) {
        for (const string& type : n.second->op_types()) {
          if (RE2::FullMatch(type, regex)) {
            ret.push_back(n.second);
            break;
          }
        }
      }
    }
    return ret;
  }

  const string name_;
  int64_t step_;
  // Snapshot based on type_regexes
  std::set<string> op_types_;
  int64_t run_count_;
  int64_t exec_micros_;
  int64_t accelerator_exec_micros_;
  int64_t cpu_exec_micros_;

  int64_t requested_bytes_;
  int64_t peak_bytes_;
  int64_t residual_bytes_;
  int64_t output_bytes_;
  int64_t float_ops_;
  int64_t parameters_;
  std::set<string> devices_;
  std::vector<std::vector<int64_t>> shapes_;
  std::map<string, const TFGraphNode*> snapshot_nodes_;

  // Overall data held by the TFMultiGraphNode.
  std::map<string, const TFGraphNode*> nodes_;
};

bool IsPlacedOnCPU(const string& device);
bool IsPlacedOnAccelerator(const string& device);
bool CountAsAcceleratorTime(const string& device);
bool CountAsCPUTime(const string& device);
bool IsCanonicalDevice(const string& device);

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_NODE_H_
