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

// Parent class and utilities for tfprof_code.

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_SHOW_MULTI_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_SHOW_MULTI_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_multiDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_multiDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_multiDTh() {
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


#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/profiler/internal/tfprof_constants.h"
#include "tensorflow/core/profiler/internal/tfprof_node.h"
#include "tensorflow/core/profiler/internal/tfprof_node_show.h"
#include "tensorflow/core/profiler/internal/tfprof_show.h"
#include "tensorflow/core/profiler/internal/tfprof_tensor.h"
#include "tensorflow/core/profiler/internal/tfprof_timeline.h"
#include "tensorflow/core/profiler/internal/tfprof_utils.h"
#include "tensorflow/core/profiler/tfprof_options.h"
#include "tensorflow/core/profiler/tfprof_output.pb.h"

namespace tensorflow {
namespace tfprof {

class TFMultiShow {
 public:
  explicit TFMultiShow() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_multiDTh mht_0(mht_0_v, 212, "", "./tensorflow/core/profiler/internal/tfprof_show_multi.h", "TFMultiShow");
}
  virtual ~TFMultiShow() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_multiDTh mht_1(mht_1_v, 216, "", "./tensorflow/core/profiler/internal/tfprof_show_multi.h", "~TFMultiShow");
}
  virtual void AddNode(TFGraphNode* node) = 0;
  virtual void Build() = 0;
  const MultiGraphNodeProto& Show(const string& prefix, const Options& opts);

 protected:
  virtual const ShowMultiNode* ShowInternal(const Options& opts,
                                            Timeline* timeline) = 0;

  bool LookUpCheckPoint(const string& name,
                        std::unique_ptr<TFProfTensor>* tensor);

  // Overridden by subclass if extra requirements need to be met.
  virtual bool ShouldShowIfExtra(const ShowMultiNode* node, const Options& opts,
                                 int depth) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_multiDTh mht_2(mht_2_v, 233, "", "./tensorflow/core/profiler/internal/tfprof_show_multi.h", "ShouldShowIfExtra");

    return true;
  }

  bool ShouldShow(const ShowMultiNode* node, const Options& opts,
                  int depth) const;

  bool ShouldTrim(const ShowMultiNode* node,
                  const std::vector<string>& regexes) const;

  bool ReAccount(ShowMultiNode* node, const Options& opts);

  string FormatLegend(const Options& opts) const;
  string FormatInputShapes(const MultiGraphNodeProto& proto) const;
  std::vector<string> FormatTimes(const ShowMultiNode* node,
                                  const Options& opts) const;

  template <typename T>
  std::vector<T*> SortNodes(const std::vector<T*>& nodes, const Options& opts) {
    if (opts.order_by.empty() || nodes.empty()) {
      return nodes;
    }
    std::vector<T*> sorted_nodes = nodes;
    std::stable_sort(sorted_nodes.begin(), sorted_nodes.end(),
                     [&opts](const T* n1, const T* n2) {
                       if (n1->name() == kTFProfRoot) return true;
                       if (n2->name() == kTFProfRoot) return false;
                       bool name_cmp = n1->name() < n2->name();
                       if (opts.order_by == kOrderBy[0]) {
                         return name_cmp;
                       } else if (opts.order_by == kOrderBy[1]) {
                         return n1->proto().total_requested_bytes() >
                                n2->proto().total_requested_bytes();
                       } else if (opts.order_by == kOrderBy[2]) {
                         return n1->proto().total_peak_bytes() >
                                n2->proto().total_peak_bytes();
                       } else if (opts.order_by == kOrderBy[3]) {
                         return n1->proto().total_residual_bytes() >
                                n2->proto().total_residual_bytes();
                       } else if (opts.order_by == kOrderBy[4]) {
                         return n1->proto().total_output_bytes() >
                                n2->proto().total_output_bytes();
                       } else if (opts.order_by == kOrderBy[5]) {
                         return n1->proto().total_exec_micros() >
                                n2->proto().total_exec_micros();
                       } else if (opts.order_by == kOrderBy[6]) {
                         return n1->proto().total_accelerator_exec_micros() >
                                n2->proto().total_accelerator_exec_micros();
                       } else if (opts.order_by == kOrderBy[7]) {
                         return n1->proto().total_cpu_exec_micros() >
                                n2->proto().total_cpu_exec_micros();
                       } else if (opts.order_by == kOrderBy[8]) {
                         return n1->proto().total_parameters() >
                                n2->proto().total_parameters();
                       } else if (opts.order_by == kOrderBy[9]) {
                         return n1->proto().total_float_ops() >
                                n2->proto().total_float_ops();
                       } else if (opts.order_by == kOrderBy[10]) {
                         return n1->node->graph_nodes().size() >
                                n2->node->graph_nodes().size();
                       }
                       return name_cmp;
                     });
    return sorted_nodes;
  }
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_SHOW_MULTI_H_
