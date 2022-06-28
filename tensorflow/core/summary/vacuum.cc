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
class MHTracer_DTPStensorflowPScorePSsummaryPSvacuumDTcc {
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
   MHTracer_DTPStensorflowPScorePSsummaryPSvacuumDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSsummaryPSvacuumDTcc() {
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
#include <iostream>

#include "tensorflow/core/lib/db/sqlite.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace {

void Vacuum(const char* path) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("path: \"" + (path == nullptr ? std::string("nullptr") : std::string((char*)path)) + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSvacuumDTcc mht_0(mht_0_v, 194, "", "./tensorflow/core/summary/vacuum.cc", "Vacuum");

  LOG(INFO) << "Opening SQLite DB: " << path;
  Sqlite* db;
  TF_CHECK_OK(Sqlite::Open(path, SQLITE_OPEN_READWRITE, &db));
  core::ScopedUnref db_unref(db);

  // TODO(jart): Maybe defragment rowids on Tensors.
  // TODO(jart): Maybe LIMIT deletes and incremental VACUUM.

  // clang-format off

  LOG(INFO) << "Deleting orphaned Experiments";
  db->PrepareOrDie(R"sql(
    DELETE FROM
      Experiments
    WHERE
      user_id IS NOT NULL
      AND user_id NOT IN (SELECT user_id FROM Users)
  )sql").StepAndResetOrDie();

  LOG(INFO) << "Deleting orphaned Runs";
  db->PrepareOrDie(R"sql(
    DELETE FROM
      Runs
    WHERE
      experiment_id IS NOT NULL
      AND experiment_id NOT IN (SELECT experiment_id FROM Experiments)
  )sql").StepAndResetOrDie();

  LOG(INFO) << "Deleting orphaned Tags";
  db->PrepareOrDie(R"sql(
    DELETE FROM
      Tags
    WHERE
      run_id IS NOT NULL
      AND run_id NOT IN (SELECT run_id FROM Runs)
  )sql").StepAndResetOrDie();

  // TODO(jart): What should we do if plugins define non-tag tensor series?
  LOG(INFO) << "Deleting orphaned Tensors";
  db->PrepareOrDie(R"sql(
    DELETE FROM
      Tensors
    WHERE
      series IS NOT NULL
      AND series NOT IN (SELECT tag_id FROM Tags)
  )sql").StepAndResetOrDie();

  LOG(INFO) << "Deleting orphaned TensorStrings";
  db->PrepareOrDie(R"sql(
    DELETE FROM
      TensorStrings
    WHERE
      tensor_rowid NOT IN (SELECT rowid FROM Tensors)
  )sql").StepAndResetOrDie();

  LOG(INFO) << "Deleting orphaned Graphs";
  db->PrepareOrDie(R"sql(
    DELETE FROM
      Graphs
    WHERE
      run_id IS NOT NULL
      AND run_id NOT IN (SELECT run_id FROM Runs)
  )sql").StepAndResetOrDie();

  LOG(INFO) << "Deleting orphaned Nodes";
  db->PrepareOrDie(R"sql(
    DELETE FROM
      Nodes
    WHERE
      graph_id NOT IN (SELECT graph_id FROM Graphs)
  )sql").StepAndResetOrDie();

  LOG(INFO) << "Deleting orphaned NodeInputs";
  db->PrepareOrDie(R"sql(
    DELETE FROM
      NodeInputs
    WHERE
      graph_id NOT IN (SELECT graph_id FROM Graphs)
  )sql").StepAndResetOrDie();

  LOG(INFO) << "Running VACUUM";
  db->PrepareOrDie("VACUUM").StepAndResetOrDie();

  // clang-format on
}

int main(int argc, char* argv[]) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSvacuumDTcc mht_1(mht_1_v, 284, "", "./tensorflow/core/summary/vacuum.cc", "main");

  string usage = Flags::Usage(argv[0], {});
  bool parse_result = Flags::Parse(&argc, argv, {});
  if (!parse_result) {
    std::cerr << "The vacuum tool rebuilds SQLite database files created by\n"
              << "SummaryDbWriter, which makes them smaller.\n\n"
              << "This means deleting orphaned rows and rebuilding b-tree\n"
              << "pages so empty space from deleted rows is cleared. Any\n"
              << "superfluous padding of Tensor BLOBs is also removed.\n\n"
              << usage;
    return -1;
  }
  port::InitMain(argv[0], &argc, &argv);
  if (argc < 2 || argv[1][0] == '-') {
    std::cerr << "Need at least one SQLite DB path.\n";
    return -1;
  }
  for (int i = 1; i < argc; ++i) {
    Vacuum(argv[i]);
  }
  return 0;
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSvacuumDTcc mht_2(mht_2_v, 313, "", "./tensorflow/core/summary/vacuum.cc", "main");
 return tensorflow::main(argc, argv); }
