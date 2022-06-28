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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSprint_model_analysisDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSprint_model_analysisDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSprint_model_analysisDTcc() {
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

#include "tensorflow/core/profiler/internal/print_model_analysis.h"

#include <stdio.h>

#include <memory>
#include <utility>

#include "absl/strings/str_format.h"
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/profiler/internal/advisor/tfprof_advisor.h"
#include "tensorflow/core/profiler/internal/tfprof_stats.h"
#include "tensorflow/core/profiler/tfprof_log.pb.h"
#include "tensorflow/core/profiler/tfprof_options.h"
#include "tensorflow/core/profiler/tfprof_options.pb.h"
#include "tensorflow/core/profiler/tfprof_output.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace tfprof {
namespace {
TFStats* tf_stat = nullptr;

string RunProfile(const string& command, const string& options,
                  TFStats* tf_stats) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("command: \"" + command + "\"");
   mht_0_v.push_back("options: \"" + options + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSprint_model_analysisDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/profiler/internal/print_model_analysis.cc", "RunProfile");

  if (command == kCmds[4]) {
    AdvisorOptionsProto option_pb;
    if (!option_pb.ParseFromString(options)) {
      absl::FPrintF(stderr, "Cannot parse AdvisorOptionsProto\n");
      return "";
    }
    tf_stats->BuildAllViews();
    return Advisor(tf_stats).Advise(option_pb).SerializeAsString();
  } else {
    tf_stats->BuildView(command);
  }

  Options opts;
  tensorflow::Status s = Options::FromProtoStr(options, &opts);
  if (!s.ok()) {
    absl::FPrintF(stderr, "%s\n", s.ToString());
    return "";
  }

  if (opts.output_type == kOutput[1]) {
    absl::PrintF(
        "\n=========================Options=============================\n");
    absl::PrintF("%s", opts.ToString());
    absl::PrintF(
        "\n==================Model Analysis Report======================\n");
    string ret = "";
    if (command == kCmds[2] || command == kCmds[3]) {
      ret = tf_stats->ShowMultiGraphNode(command, opts).SerializeAsString();
    } else if (command == kCmds[0] || command == kCmds[1]) {
      ret = tf_stats->ShowGraphNode(command, opts).SerializeAsString();
    } else {
      absl::FPrintF(stderr, "Unknown command: %s\n", command);
    }
    absl::PrintF(
        "\n======================End of Report==========================\n");
    fflush(stdout);
    return ret;
  }
  if (command == kCmds[2] || command == kCmds[3]) {
    return tf_stats->ShowMultiGraphNode(command, opts).SerializeAsString();
  } else if (command == kCmds[0] || command == kCmds[1]) {
    return tf_stats->ShowGraphNode(command, opts).SerializeAsString();
  } else {
    absl::FPrintF(stderr, "Unknown command: %s\n", command);
    return "";
  }
}
}  // namespace

bool NewProfiler(const string* graph, const string* op_log) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSprint_model_analysisDTcc mht_1(mht_1_v, 265, "", "./tensorflow/core/profiler/internal/print_model_analysis.cc", "NewProfiler");

  std::unique_ptr<GraphDef> graph_ptr(new GraphDef());
  if (graph && !graph->empty()) {
    if (!graph_ptr->ParseFromString(*graph)) {
      if (!protobuf::TextFormat::ParseFromString(*graph, graph_ptr.get())) {
        absl::FPrintF(stderr, "Failed to parse graph\n");
        return false;
      }
    }
  }

  std::unique_ptr<OpLogProto> op_log_ptr;
  if (op_log && !op_log->empty()) {
    op_log_ptr.reset(new OpLogProto());
    if (!op_log_ptr->ParseFromString(*op_log)) {
      absl::FPrintF(stderr, "Failed to parse OpLogProto.\n");
      return false;
    }
  }
  tf_stat = new TFStats(std::move(graph_ptr), nullptr, std::move(op_log_ptr),
                        nullptr);
  return true;
}

void ProfilerFromFile(const string* filename) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSprint_model_analysisDTcc mht_2(mht_2_v, 292, "", "./tensorflow/core/profiler/internal/print_model_analysis.cc", "ProfilerFromFile");

  CHECK(!tf_stat) << "Currently only 1 living tfprof profiler is allowed";
  CHECK(filename) << "Missing profile filename to init profiler from file";
  tf_stat = new TFStats(*filename, nullptr);
}

void DeleteProfiler() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSprint_model_analysisDTcc mht_3(mht_3_v, 301, "", "./tensorflow/core/profiler/internal/print_model_analysis.cc", "DeleteProfiler");

  if (tf_stat) {
    delete tf_stat;
    tf_stat = nullptr;
  }
}

double AddStep(int64_t step, const string* graph, const string* run_meta,
               const string* op_log) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSprint_model_analysisDTcc mht_4(mht_4_v, 312, "", "./tensorflow/core/profiler/internal/print_model_analysis.cc", "AddStep");

  CHECK(tf_stat);

  if (graph && !graph->empty()) {
    std::unique_ptr<GraphDef> graph_ptr(new GraphDef());
    if (!graph_ptr->ParseFromString(*graph)) {
      if (!protobuf::TextFormat::ParseFromString(*graph, graph_ptr.get())) {
        absl::FPrintF(stderr, "Failed to parse graph\n");
      }
    }
    tf_stat->AddGraph(std::move(graph_ptr));
  }

  CHECK(run_meta && !run_meta->empty());
  // TODO(xpan): Better error handling.
  std::unique_ptr<RunMetadata> run_meta_ptr(new RunMetadata());
  run_meta_ptr->ParseFromString(*run_meta);
  tf_stat->AddRunMeta(step, std::move(run_meta_ptr));

  if (op_log && !op_log->empty()) {
    std::unique_ptr<OpLogProto> op_log_ptr;
    op_log_ptr.reset(new OpLogProto());
    op_log_ptr->ParseFromString(*op_log);
    tf_stat->AddOpLogProto(std::move(op_log_ptr));
  }
  return tf_stat->run_coverage();
}

string Profile(const string* command, const string* options) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSprint_model_analysisDTcc mht_5(mht_5_v, 343, "", "./tensorflow/core/profiler/internal/print_model_analysis.cc", "Profile");

  CHECK(tf_stat);
  CHECK(command) << "command mustn't be null";
  CHECK(options) << "options mustn't be null";
  return RunProfile(*command, *options, tf_stat);
}

string SerializeToString() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSprint_model_analysisDTcc mht_6(mht_6_v, 353, "", "./tensorflow/core/profiler/internal/print_model_analysis.cc", "SerializeToString");

  CHECK(tf_stat);
  string content;
  tf_stat->SerializeToString(&content);
  return content;
}

void WriteProfile(const string* filename) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSprint_model_analysisDTcc mht_7(mht_7_v, 363, "", "./tensorflow/core/profiler/internal/print_model_analysis.cc", "WriteProfile");

  CHECK(tf_stat);
  CHECK(filename) << "empty file name when asking to write profile.";
  tf_stat->WriteProfile(*filename);
}

string PrintModelAnalysis(const string* graph, const string* run_meta,
                          const string* op_log, const string* command,
                          const string* options) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSprint_model_analysisDTcc mht_8(mht_8_v, 374, "", "./tensorflow/core/profiler/internal/print_model_analysis.cc", "PrintModelAnalysis");

  CHECK(command) << "command mustn't be null";
  CHECK(options) << "options mustn't be null";
  std::unique_ptr<GraphDef> graph_ptr(new GraphDef());
  if (graph && !graph->empty()) {
    graph_ptr->ParseFromString(*graph);
  }

  std::unique_ptr<RunMetadata> run_meta_ptr;
  if (run_meta && !run_meta->empty()) {
    run_meta_ptr.reset(new RunMetadata());
    run_meta_ptr->ParseFromString(*run_meta);
  }

  std::unique_ptr<OpLogProto> op_log_ptr;
  if (op_log && !op_log->empty()) {
    op_log_ptr.reset(new OpLogProto());
    op_log_ptr->ParseFromString(*op_log);
  }

  // TODO(xpan): Maybe need to init the checkpoint reader?
  std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader;

  TFStats tf_stats(std::move(graph_ptr), std::move(run_meta_ptr),
                   std::move(op_log_ptr), std::move(ckpt_reader));

  return RunProfile(*command, *options, &tf_stats);
}

}  // namespace tfprof
}  // namespace tensorflow
