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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_tensorflow_serverDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_tensorflow_serverDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_tensorflow_serverDTcc() {
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

#include <iostream>
#include <vector>

#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server_builder.h"

#include "tensorflow/core/distributed_runtime/server_lib.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/command_line_flags.h"

// This binary starts a TensorFlow server (master and worker).
//
// TODO(mrry): Replace with a py_binary that uses `tf.GrpcServer()`.
namespace tensorflow {
namespace {

Status FillServerDef(const string& cluster_spec, const string& job_name,
                     int task_index, ServerDef* options) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("cluster_spec: \"" + cluster_spec + "\"");
   mht_0_v.push_back("job_name: \"" + job_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_tensorflow_serverDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server.cc", "FillServerDef");

  options->set_protocol("grpc");
  options->set_job_name(job_name);
  options->set_task_index(task_index);

  size_t my_num_tasks = 0;

  ClusterDef* const cluster = options->mutable_cluster();

  for (const string& job_str : str_util::Split(cluster_spec, ',')) {
    JobDef* const job_def = cluster->add_job();
    // Split each entry in the flag into 2 pieces, separated by "|".
    const std::vector<string> job_pieces = str_util::Split(job_str, '|');
    CHECK_EQ(2, job_pieces.size()) << job_str;
    const string& job_name = job_pieces[0];
    job_def->set_name(job_name);
    // Does a bit more validation of the tasks_per_replica.
    const StringPiece spec = job_pieces[1];
    // job_str is of form <job_name>|<host_ports>.
    const std::vector<string> host_ports = str_util::Split(spec, ';');
    for (size_t i = 0; i < host_ports.size(); ++i) {
      (*job_def->mutable_tasks())[i] = host_ports[i];
    }
    size_t num_tasks = host_ports.size();
    if (job_name == options->job_name()) {
      my_num_tasks = host_ports.size();
    }
    LOG(INFO) << "Peer " << job_name << " " << num_tasks << " {"
              << absl::StrJoin(host_ports, ", ") << "}";
  }
  if (my_num_tasks == 0) {
    return errors::InvalidArgument("Job name \"", options->job_name(),
                                   "\" does not appear in the cluster spec");
  }
  if (options->task_index() >= my_num_tasks) {
    return errors::InvalidArgument("Task index ", options->task_index(),
                                   " is invalid (job \"", options->job_name(),
                                   "\" contains ", my_num_tasks, " tasks");
  }
  return Status::OK();
}

}  // namespace
}  // namespace tensorflow

void Usage(char* const argv_0) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_tensorflow_serverDTcc mht_1(mht_1_v, 261, "", "./tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server.cc", "Usage");

  std::cerr << "Usage: " << argv_0
            << " --cluster_spec=SPEC --job_name=NAME --task_id=ID" << std::endl;
  std::cerr << "Where:" << std::endl;
  std::cerr << "    SPEC is <JOB>(,<JOB>)*" << std::endl;
  std::cerr << "    JOB  is <NAME>|<HOST:PORT>(;<HOST:PORT>)*" << std::endl;
  std::cerr << "    NAME is a valid job name ([a-z][0-9a-z]*)" << std::endl;
  std::cerr << "    HOST is a hostname or IP address" << std::endl;
  std::cerr << "    PORT is a port number" << std::endl;
}

int main(int argc, char* argv[]) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_tensorflow_serverDTcc mht_2(mht_2_v, 275, "", "./tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server.cc", "main");

  tensorflow::string cluster_spec;
  tensorflow::string job_name;
  int task_index = 0;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("cluster_spec", &cluster_spec, "cluster spec"),
      tensorflow::Flag("job_name", &job_name, "job name"),
      tensorflow::Flag("task_id", &task_index, "task id"),
  };
  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (!parse_result || argc != 1) {
    std::cerr << usage << std::endl;
    Usage(argv[0]);
    return -1;
  }
  tensorflow::ServerDef server_def;
  tensorflow::Status s = tensorflow::FillServerDef(cluster_spec, job_name,
                                                   task_index, &server_def);
  if (!s.ok()) {
    std::cerr << "ERROR: " << s.error_message() << std::endl;
    Usage(argv[0]);
    return -1;
  }
  std::unique_ptr<tensorflow::ServerInterface> server;
  TF_QCHECK_OK(tensorflow::NewServer(server_def, &server));
  TF_QCHECK_OK(server->Start());
  TF_QCHECK_OK(server->Join());
}
