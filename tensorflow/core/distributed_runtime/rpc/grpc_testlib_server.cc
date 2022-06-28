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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_testlib_serverDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_testlib_serverDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_testlib_serverDTcc() {
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

#include <vector>

#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server_builder.h"

#include "tensorflow/core/distributed_runtime/server_lib.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/command_line_flags.h"

// This binary starts a TensorFlow server (master and worker) for test purposes.
namespace tensorflow {
namespace {

Status FillServerDef(const string& job_spec, const string& job_name,
                     int num_cpus, int num_gpus, int task_index,
                     ServerDef* options) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("job_spec: \"" + job_spec + "\"");
   mht_0_v.push_back("job_name: \"" + job_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_testlib_serverDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/distributed_runtime/rpc/grpc_testlib_server.cc", "FillServerDef");

  options->set_protocol("grpc");
  options->set_job_name(job_name);
  options->set_task_index(task_index);

  uint32 my_tasks_per_replica = 0;
  for (const string& job_str : str_util::Split(job_spec, ',')) {
    JobDef* job_def = options->mutable_cluster()->add_job();
    // Split each entry in the flag into 2 pieces, separated by "|".
    const std::vector<string> job_pieces = str_util::Split(job_str, '|');
    CHECK_EQ(2, job_pieces.size()) << job_str;
    job_def->set_name(job_pieces[0]);
    // Does a bit more validation of the tasks_per_replica.
    const StringPiece spec = job_pieces[1];
    // job_str is of form <job_name>|<host_ports>.
    const std::vector<string> host_ports = str_util::Split(spec, ';');
    uint32 tasks_per_replica = host_ports.size();
    for (size_t i = 0; i < host_ports.size(); ++i) {
      (*job_def->mutable_tasks())[i] = host_ports[i];
    }
    if (job_def->name() == options->job_name()) {
      my_tasks_per_replica = tasks_per_replica;
    }
    LOG(INFO) << "Peer " << job_def->name() << " " << tasks_per_replica << " {"
              << absl::StrJoin(host_ports, ", ") << "}";
  }
  if (my_tasks_per_replica == 0) {
    return errors::InvalidArgument("Invalid job specification");
  }
  ConfigProto* config = options->mutable_default_session_config();
  (*config->mutable_device_count())["CPU"] = num_cpus;
  (*config->mutable_device_count())["GPU"] = num_gpus;
  return Status::OK();
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_testlib_serverDTcc mht_1(mht_1_v, 252, "", "./tensorflow/core/distributed_runtime/rpc/grpc_testlib_server.cc", "main");

  tensorflow::port::InitMain(argv[0], &argc, &argv);
  tensorflow::string job_spec;
  tensorflow::string job_name;
  int num_cpus = 1;
  int num_gpus = 0;
  int task_index = 0;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("tf_jobs", &job_spec, "job specification"),
      tensorflow::Flag("tf_job", &job_name, "job name"),
      tensorflow::Flag("tf_task", &task_index, "task index"),
      tensorflow::Flag("num_cpus", &num_cpus, "number of CPUs"),
      tensorflow::Flag("num_gpus", &num_gpus, "number of GPUs"),
  };
  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result || argc != 1) {
    LOG(ERROR) << usage;
    return -1;
  }

  tensorflow::ServerDef def;
  tensorflow::Status s = tensorflow::FillServerDef(job_spec, job_name, num_cpus,
                                                   num_gpus, task_index, &def);
  if (!s.ok()) {
    LOG(ERROR) << "Could not parse job spec: " << s.error_message() << "\n"
               << usage;
    return -1;
  }

  std::unique_ptr<tensorflow::ServerInterface> svr;
  s = tensorflow::NewServer(def, &svr);

  if (!s.ok()) {
    LOG(ERROR) << "Could not create server: " << s.error_message();
    return -1;
  }
  TF_QCHECK_OK(svr->Start());
  TF_QCHECK_OK(svr->Join());

  // NOTE(mrry): Unreachable code.
  return 0;
}
