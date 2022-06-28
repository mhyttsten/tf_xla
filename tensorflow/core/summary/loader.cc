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
class MHTracer_DTPStensorflowPScorePSsummaryPSloaderDTcc {
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
   MHTracer_DTPStensorflowPScorePSsummaryPSloaderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSsummaryPSloaderDTcc() {
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
#include <vector>

#include "tensorflow/core/summary/schema.h"
#include "tensorflow/core/summary/summary_db_writer.h"
#include "tensorflow/core/lib/db/sqlite.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {
namespace {

template <typename T>
string AddCommas(T n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSloaderDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/summary/loader.cc", "AddCommas");

  static_assert(std::is_integral<T>::value, "is_integral");
  string s = strings::StrCat(n);
  if (s.size() > 3) {
    int extra = s.size() / 3 - (s.size() % 3 == 0 ? 1 : 0);
    s.append(extra, 'X');
    int c = 0;
    for (int i = s.size() - 1; i > 0; --i) {
      s[i] = s[i - extra];
      if (++c % 3 == 0) {
        s[--i] = ',';
        --extra;
      }
    }
  }
  return s;
}

int main(int argc, char* argv[]) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSloaderDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/summary/loader.cc", "main");

  string path;
  string events;
  string experiment_name;
  string run_name;
  string user_name;
  std::vector<Flag> flag_list = {
      Flag("db", &path, "Path of SQLite DB file"),
      Flag("events", &events, "TensorFlow record proto event log file"),
      Flag("experiment_name", &experiment_name, "The DB experiment_name value"),
      Flag("run_name", &run_name, "The DB run_name value"),
      Flag("user_name", &user_name, "The DB user_name value"),
  };
  string usage = Flags::Usage(argv[0], flag_list);
  bool parse_result = Flags::Parse(&argc, argv, flag_list);
  if (!parse_result || path.empty()) {
    std::cerr << "The loader tool imports tf.Event record files, created by\n"
              << "SummaryFileWriter, into the sorts of SQLite database files\n"
              << "created by SummaryDbWriter.\n\n"
              << "In addition to the flags below, the environment variables\n"
              << "defined by core/lib/db/sqlite.cc can also be set.\n\n"
              << usage;
    return -1;
  }
  port::InitMain(argv[0], &argc, &argv);
  Env* env = Env::Default();

  LOG(INFO) << "Opening SQLite file: " << path;
  Sqlite* db;
  TF_CHECK_OK(Sqlite::Open(
      path, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX,
      &db));
  core::ScopedUnref unref_db(db);

  LOG(INFO) << "Initializing TensorBoard schema";
  TF_CHECK_OK(SetupTensorboardSqliteDb(db));

  LOG(INFO) << "Creating SummaryDbWriter";
  SummaryWriterInterface* db_writer;
  TF_CHECK_OK(CreateSummaryDbWriter(db, experiment_name, run_name, user_name,
                                    env, &db_writer));
  core::ScopedUnref unref(db_writer);

  LOG(INFO) << "Loading TF event log: " << events;
  std::unique_ptr<RandomAccessFile> file;
  TF_CHECK_OK(env->NewRandomAccessFile(events, &file));
  io::RecordReader reader(file.get());

  uint64 start = env->NowMicros();
  uint64 records = 0;
  uint64 offset = 0;
  tstring record;
  while (true) {
    std::unique_ptr<Event> event = std::unique_ptr<Event>(new Event);
    Status s = reader.ReadRecord(&offset, &record);
    if (s.code() == error::OUT_OF_RANGE) break;
    TF_CHECK_OK(s);
    if (!ParseProtoUnlimited(event.get(), record)) {
      LOG(FATAL) << "Corrupt tf.Event record"
                 << " offset=" << (offset - record.size())
                 << " size=" << static_cast<int>(record.size());
    }
    TF_CHECK_OK(db_writer->WriteEvent(std::move(event)));
    ++records;
  }
  uint64 elapsed = env->NowMicros() - start;
  uint64 bps = (elapsed == 0 ? offset : static_cast<uint64>(
                                            offset / (elapsed / 1000000.0)));
  LOG(INFO) << "Loaded " << AddCommas(offset) << " bytes with "
            << AddCommas(records) << " records at " << AddCommas(bps) << " bps";
  return 0;
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSloaderDTcc mht_2(mht_2_v, 299, "", "./tensorflow/core/summary/loader.cc", "main");
 return tensorflow::main(argc, argv); }
