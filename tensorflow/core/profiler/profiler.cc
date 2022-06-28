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
class MHTracer_DTPStensorflowPScorePSprofilerPSprofilerDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSprofilerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSprofilerDTcc() {
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

#include <stdio.h>
#include <stdlib.h>

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "linenoise.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/profiler/internal/advisor/tfprof_advisor.h"
#include "tensorflow/core/profiler/internal/tfprof_stats.h"
#include "tensorflow/core/profiler/internal/tfprof_utils.h"
#include "tensorflow/core/profiler/tfprof_log.pb.h"
#include "tensorflow/core/profiler/tfprof_options.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace tfprof {
void completion(const char* buf, linenoiseCompletions* lc) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buf: \"" + (buf == nullptr ? std::string("nullptr") : std::string((char*)buf)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSprofilerDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/profiler/profiler.cc", "completion");

  string buf_str = buf;
  if (buf_str.find(' ') == buf_str.npos) {
    for (const char* opt : kCmds) {
      if (string(opt).find(buf_str) == 0) {
        linenoiseAddCompletion(lc, opt);
      }
    }
    return;
  }

  string prefix;
  int last_dash = buf_str.find_last_of(' ');
  if (last_dash != string::npos) {
    prefix = buf_str.substr(0, last_dash + 1);
    buf_str = buf_str.substr(last_dash + 1, kint32max);
  }
  for (const char* opt : kOptions) {
    if (string(opt).find(buf_str) == 0) {
      linenoiseAddCompletion(lc, (prefix + opt).c_str());
    }
  }
}

int Run(int argc, char** argv) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSprofilerDTcc mht_1(mht_1_v, 243, "", "./tensorflow/core/profiler/profiler.cc", "Run");

  string FLAGS_profile_path = "";
  string FLAGS_graph_path = "";
  string FLAGS_run_meta_path = "";
  string FLAGS_op_log_path = "";
  string FLAGS_checkpoint_path = "";
  int32_t FLAGS_max_depth = 10;
  int64_t FLAGS_min_bytes = 0;
  int64_t FLAGS_min_peak_bytes = 0;
  int64_t FLAGS_min_residual_bytes = 0;
  int64_t FLAGS_min_output_bytes = 0;
  int64_t FLAGS_min_micros = 0;
  int64_t FLAGS_min_accelerator_micros = 0;
  int64_t FLAGS_min_cpu_micros = 0;
  int64_t FLAGS_min_params = 0;
  int64_t FLAGS_min_float_ops = 0;
  int64_t FLAGS_min_occurrence = 0;
  int64_t FLAGS_step = -1;
  string FLAGS_order_by = "name";
  string FLAGS_account_type_regexes = ".*";
  string FLAGS_start_name_regexes = ".*";
  string FLAGS_trim_name_regexes = "";
  string FLAGS_show_name_regexes = ".*";
  string FLAGS_hide_name_regexes;
  bool FLAGS_account_displayed_op_only = false;
  string FLAGS_select = "micros";
  string FLAGS_output = "";
  for (int i = 0; i < argc; i++) {
    absl::FPrintF(stderr, "%s\n", argv[i]);
  }

  std::vector<Flag> flag_list = {
      Flag("profile_path", &FLAGS_profile_path, "Profile binary file name."),
      Flag("graph_path", &FLAGS_graph_path, "GraphDef proto text file name"),
      Flag("run_meta_path", &FLAGS_run_meta_path,
           "Comma-separated list of RunMetadata proto binary "
           "files. Each file is given step number 0,1,2,etc"),
      Flag("op_log_path", &FLAGS_op_log_path,
           "tensorflow::tfprof::OpLogProto proto binary file name"),
      Flag("checkpoint_path", &FLAGS_checkpoint_path,
           "TensorFlow Checkpoint file name"),
      Flag("max_depth", &FLAGS_max_depth, "max depth"),
      Flag("min_bytes", &FLAGS_min_bytes, "min_bytes"),
      Flag("min_peak_bytes", &FLAGS_min_peak_bytes, "min_peak_bytes"),
      Flag("min_residual_bytes", &FLAGS_min_residual_bytes,
           "min_residual_bytes"),
      Flag("min_output_bytes", &FLAGS_min_output_bytes, "min_output_bytes"),
      Flag("min_micros", &FLAGS_min_micros, "min micros"),
      Flag("min_accelerator_micros", &FLAGS_min_accelerator_micros,
           "min accelerator_micros"),
      Flag("min_cpu_micros", &FLAGS_min_cpu_micros, "min_cpu_micros"),
      Flag("min_params", &FLAGS_min_params, "min params"),
      Flag("min_float_ops", &FLAGS_min_float_ops, "min float ops"),
      Flag("min_occurrence", &FLAGS_min_occurrence, "min occurrence"),
      Flag("step", &FLAGS_step,
           "The stats of which step to use. By default average"),
      Flag("order_by", &FLAGS_order_by, "order by"),
      Flag("account_type_regexes", &FLAGS_start_name_regexes,
           "start name regexes"),
      Flag("trim_name_regexes", &FLAGS_trim_name_regexes, "trim name regexes"),
      Flag("show_name_regexes", &FLAGS_show_name_regexes, "show name regexes"),
      Flag("hide_name_regexes", &FLAGS_hide_name_regexes, "hide name regexes"),
      Flag("account_displayed_op_only", &FLAGS_account_displayed_op_only,
           "account displayed op only"),
      Flag("select", &FLAGS_select, "select"),
      Flag("output", &FLAGS_output, "output"),
  };
  string usage = Flags::Usage(argv[0], flag_list);
  bool parse_ok = Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    absl::PrintF("%s", usage);
    return (2);
  }
  port::InitMain(argv[0], &argc, &argv);

  if (!FLAGS_profile_path.empty() &&
      (!FLAGS_graph_path.empty() || !FLAGS_run_meta_path.empty())) {
    absl::FPrintF(stderr,
                  "--profile_path is set, do not set --graph_path or "
                  "--run_meta_path\n");
    return 1;
  }

  std::vector<string> account_type_regexes =
      absl::StrSplit(FLAGS_account_type_regexes, ',', absl::SkipEmpty());
  std::vector<string> start_name_regexes =
      absl::StrSplit(FLAGS_start_name_regexes, ',', absl::SkipEmpty());
  std::vector<string> trim_name_regexes =
      absl::StrSplit(FLAGS_trim_name_regexes, ',', absl::SkipEmpty());
  std::vector<string> show_name_regexes =
      absl::StrSplit(FLAGS_show_name_regexes, ',', absl::SkipEmpty());
  std::vector<string> hide_name_regexes =
      absl::StrSplit(FLAGS_hide_name_regexes, ',', absl::SkipEmpty());
  std::vector<string> select =
      absl::StrSplit(FLAGS_select, ',', absl::SkipEmpty());

  string output_type;
  std::map<string, string> output_options;
  Status s = ParseOutput(FLAGS_output, &output_type, &output_options);
  CHECK(s.ok()) << s.ToString();

  string cmd = "";
  if (argc == 1 && FLAGS_graph_path.empty() && FLAGS_profile_path.empty() &&
      FLAGS_run_meta_path.empty()) {
    PrintHelp();
    return 0;
  } else if (argc > 1) {
    if (string(argv[1]) == kCmds[6]) {
      PrintHelp();
      return 0;
    }
    if (string(argv[1]) == kCmds[0] || string(argv[1]) == kCmds[1] ||
        string(argv[1]) == kCmds[2] || string(argv[1]) == kCmds[3] ||
        string(argv[1]) == kCmds[4]) {
      cmd = argv[1];
    }
  }

  absl::PrintF("Reading Files...\n");
  std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader;
  TF_Status* status = TF_NewStatus();
  if (!FLAGS_checkpoint_path.empty()) {
    ckpt_reader.reset(
        new checkpoint::CheckpointReader(FLAGS_checkpoint_path, status));
    if (TF_GetCode(status) != TF_OK) {
      absl::FPrintF(stderr, "%s\n", TF_Message(status));
      TF_DeleteStatus(status);
      return 1;
    }
    TF_DeleteStatus(status);
  }

  std::unique_ptr<TFStats> tf_stat;
  if (!FLAGS_profile_path.empty()) {
    tf_stat.reset(new TFStats(FLAGS_profile_path, std::move(ckpt_reader)));
  } else {
    absl::PrintF(
        "Try to use a single --profile_path instead of "
        "graph_path,op_log_path,run_meta_path\n");
    std::unique_ptr<GraphDef> graph(new GraphDef());
    if (!FLAGS_graph_path.empty()) {
      s = ReadProtoFile(Env::Default(), FLAGS_graph_path, graph.get(), false);
      if (!s.ok()) {
        absl::FPrintF(stderr, "Failed to read graph_path: %s\n", s.ToString());
        return 1;
      }
    }

    std::unique_ptr<OpLogProto> op_log(new OpLogProto());
    if (!FLAGS_op_log_path.empty()) {
      string op_log_str;
      s = ReadFileToString(Env::Default(), FLAGS_op_log_path, &op_log_str);
      if (!s.ok()) {
        absl::FPrintF(stderr, "Failed to read op_log_path: %s\n", s.ToString());
        return 1;
      }
      if (!ParseProtoUnlimited(op_log.get(), op_log_str)) {
        absl::FPrintF(stderr, "Failed to parse op_log_path\n");
        return 1;
      }
    }
    tf_stat.reset(new TFStats(std::move(graph), nullptr, std::move(op_log),
                              std::move(ckpt_reader)));

    std::vector<string> run_meta_files =
        absl::StrSplit(FLAGS_run_meta_path, ',', absl::SkipEmpty());
    for (int i = 0; i < run_meta_files.size(); ++i) {
      std::unique_ptr<RunMetadata> run_meta(new RunMetadata());
      s = ReadProtoFile(Env::Default(), run_meta_files[i], run_meta.get(),
                        true);
      if (!s.ok()) {
        absl::FPrintF(stderr, "Failed to read run_meta_path %s. Status: %s\n",
                      run_meta_files[i], s.ToString());
        return 1;
      }
      tf_stat->AddRunMeta(i, std::move(run_meta));
      absl::FPrintF(stdout, "run graph coverage: %.2f\n",
                    tf_stat->run_coverage());
    }
  }

  if (cmd == kCmds[4]) {
    tf_stat->BuildAllViews();
    Advisor(tf_stat.get()).Advise(Advisor::DefaultOptions());
    return 0;
  }

  Options opts(
      FLAGS_max_depth, FLAGS_min_bytes, FLAGS_min_peak_bytes,
      FLAGS_min_residual_bytes, FLAGS_min_output_bytes, FLAGS_min_micros,
      FLAGS_min_accelerator_micros, FLAGS_min_cpu_micros, FLAGS_min_params,
      FLAGS_min_float_ops, FLAGS_min_occurrence, FLAGS_step, FLAGS_order_by,
      account_type_regexes, start_name_regexes, trim_name_regexes,
      show_name_regexes, hide_name_regexes, FLAGS_account_displayed_op_only,
      select, output_type, output_options);

  if (cmd == kCmds[2] || cmd == kCmds[3]) {
    tf_stat->BuildView(cmd);
    tf_stat->ShowMultiGraphNode(cmd, opts);
    return 0;
  } else if (cmd == kCmds[0] || cmd == kCmds[1]) {
    tf_stat->BuildView(cmd);
    tf_stat->ShowGraphNode(cmd, opts);
    return 0;
  }

  linenoiseSetCompletionCallback(completion);
  linenoiseHistoryLoad(".tfprof_history.txt");

  bool looped = false;
  while (true) {
    char* line = linenoise("tfprof> ");
    if (line == nullptr) {
      if (!looped) {
        absl::FPrintF(stderr,
                      "Cannot start interactive shell, "
                      "use 'bazel-bin' instead of 'bazel run'.\n");
      }
      break;
    }
    looped = true;
    string line_s = line;
    free(line);

    if (line_s.empty()) {
      absl::PrintF("%s", opts.ToString());
      continue;
    }
    linenoiseHistoryAdd(line_s.c_str());
    linenoiseHistorySave(".tfprof_history.txt");

    Options new_opts = opts;
    Status s = ParseCmdLine(line_s, &cmd, &new_opts);
    if (!s.ok()) {
      absl::FPrintF(stderr, "E: %s\n", s.ToString());
      continue;
    }
    if (cmd == kCmds[5]) {
      opts = new_opts;
    } else if (cmd == kCmds[6]) {
      PrintHelp();
    } else if (cmd == kCmds[2] || cmd == kCmds[3]) {
      tf_stat->BuildView(cmd);
      tf_stat->ShowMultiGraphNode(cmd, new_opts);
    } else if (cmd == kCmds[0] || cmd == kCmds[1]) {
      tf_stat->BuildView(cmd);
      tf_stat->ShowGraphNode(cmd, new_opts);
    } else if (cmd == kCmds[4]) {
      tf_stat->BuildAllViews();
      Advisor(tf_stat.get()).Advise(Advisor::DefaultOptions());
    }
  }
  return 0;
}
}  // namespace tfprof
}  // namespace tensorflow

int main(int argc, char** argv) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSprofilerDTcc mht_2(mht_2_v, 503, "", "./tensorflow/core/profiler/profiler.cc", "main");
 return tensorflow::tfprof::Run(argc, argv); }
