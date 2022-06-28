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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc() {
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

#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"

#include <cstdlib>
#include <limits>
#include <map>
#include <unordered_map>

#include "grpcpp/create_channel.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

namespace {

string MakeAddress(const string& job, int task) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("job: \"" + job + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "MakeAddress");

  return strings::StrCat("/job:", job, "/replica:0/task:", task);
}

// Allows the host to be a raw IP (either v4 or v6).
Status ValidateHostPortPair(const string& host_port) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("host_port: \"" + host_port + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "ValidateHostPortPair");

  string bns_prefix = "/bns/";
  if (host_port.substr(0, bns_prefix.length()) == bns_prefix) {
    return Status::OK();
  }
  uint32 port;
  auto colon_index = host_port.find_last_of(':');
  if (!strings::safe_strtou32(host_port.substr(colon_index + 1), &port) ||
      host_port.substr(0, colon_index).find('/') != string::npos) {
    return errors::InvalidArgument("Could not interpret \"", host_port,
                                   "\" as a host-port pair.");
  }
  return Status::OK();
}

::grpc::ChannelArguments* CreateDefaultChannelArguments() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_2(mht_2_v, 241, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "CreateDefaultChannelArguments");

  ::grpc::ChannelArguments* args = new ::grpc::ChannelArguments();
  const char* env = std::getenv("TF_GRPC_DEFAULT_OPTIONS");
  if (env != nullptr) {
    for (auto& grpc_option : absl::StrSplit(env, ',')) {
      std::vector<string> name_value = absl::StrSplit(grpc_option, '=');
      if (name_value.size() != 2) {
        LOG(ERROR) << "Invalid GRPC options format: " << grpc_option;
        continue;
      }
      VLOG(3) << "Setting GRPC default for '" << name_value[0] << "' to '"
              << name_value[1] << "'";
      if (name_value[1].size() >= 2 && name_value[1][0] == '"') {
        string ue_value = name_value[1].substr(1, name_value[1].size() - 2);
        string value;
        string error;
        if (!absl::CUnescape(ue_value, &value, &error)) {
          LOG(ERROR) << "Failed to parse escaped string for " << grpc_option
                     << ": " << error;
        } else {
          args->SetString(name_value[0], value);
        }
      } else {
        int64_t value;
        if (strings::safe_strto64(name_value[1], &value)) {
          args->SetInt(name_value[0], value);
        } else {
          LOG(ERROR) << "Invalid integer value: " << grpc_option;
        }
      }
    }
  }
  return args;
}

const ::grpc::ChannelArguments* GetDefaultChannelArguments() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_3(mht_3_v, 279, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "GetDefaultChannelArguments");

  static const ::grpc::ChannelArguments* args = CreateDefaultChannelArguments();
  return args;
}

}  // namespace

::grpc::ChannelArguments GetChannelArguments(const RPCOptions* rpc_options) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_4(mht_4_v, 289, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "GetChannelArguments");

  // TODO(mrry): Implement secure channels.
  ::grpc::ChannelArguments args = *GetDefaultChannelArguments();
  args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH, std::numeric_limits<int32>::max());
  // NOTE(mrry): Some versions of gRPC use a 20-second minimum backoff
  // on connection failure, which makes our tests time out.
  args.SetInt(GRPC_ARG_MAX_RECONNECT_BACKOFF_MS, 1000);
  if (rpc_options != nullptr) {
    if (rpc_options->compression_algorithm() == "deflate") {
      args.SetCompressionAlgorithm(GRPC_COMPRESS_DEFLATE);
      args.SetInt(GRPC_COMPRESSION_CHANNEL_DEFAULT_LEVEL,
                  rpc_options->compression_level());
      VLOG(5) << "Setting GRPC compression : algo='"
              << rpc_options->compression_algorithm()
              << "' level=" << rpc_options->compression_level();
    } else if (rpc_options->compression_algorithm() == "gzip") {
      args.SetCompressionAlgorithm(GRPC_COMPRESS_GZIP);
      args.SetInt(GRPC_COMPRESSION_CHANNEL_DEFAULT_LEVEL,
                  rpc_options->compression_level());
      VLOG(5) << "Setting GRPC compression : algo='"
              << rpc_options->compression_algorithm()
              << "' level=" << rpc_options->compression_level();
    } else if (!rpc_options->compression_algorithm().empty()) {
      LOG(ERROR) << "Invalid compression algorithm: "
                 << rpc_options->compression_algorithm();
    }
    if (rpc_options->disable_session_connection_sharing()) {
      VLOG(5) << "Disabling TCP connection sharing";
      args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, true);
    }
  }
  return args;
}

Status NewHostPortGrpcChannel(const string& target,
                              const RPCOptions* rpc_options,
                              SharedGrpcChannelPtr* channel_pointer) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_5(mht_5_v, 329, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "NewHostPortGrpcChannel");

  // Minimally ensure that the target is valid
  TF_RETURN_IF_ERROR(ValidateHostPortPair(target));

  ::grpc::ChannelArguments args = GetChannelArguments(rpc_options);
  *channel_pointer = ::grpc::CreateCustomChannel(
      "dns:///" + target, ::grpc::InsecureChannelCredentials(), args);
  return Status::OK();
}

ChannelCreationFunction ConvertToChannelCreationFunction(
    const std::function<Status(string, const RPCOptions*,
                               SharedGrpcChannelPtr*)>& new_channel_func_ptr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_6(mht_6_v, 344, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "ConvertToChannelCreationFunction");

  return [new_channel_func_ptr](const string& target) -> SharedGrpcChannelPtr {
    SharedGrpcChannelPtr channel_ptr;
    if (new_channel_func_ptr(target, /*rpc_options=*/nullptr, &channel_ptr)
            .ok()) {
      return channel_ptr;
    } else {
      return nullptr;
    }
  };
}

Status GrpcChannelSpec::AddHostPortsJob(const string& job_id,
                                        const std::vector<string>& host_ports) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("job_id: \"" + job_id + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_7(mht_7_v, 361, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "GrpcChannelSpec::AddHostPortsJob");

  std::map<int, string> host_ports_map;
  for (size_t i = 0; i < host_ports.size(); ++i) {
    host_ports_map[i] = host_ports[i];
  }
  return AddHostPortsJob(job_id, host_ports_map);
}

Status GrpcChannelSpec::AddHostPortsJob(
    const string& job_id, const std::map<int, string>& host_ports) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("job_id: \"" + job_id + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_8(mht_8_v, 374, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "GrpcChannelSpec::AddHostPortsJob");

  if (!job_ids_.insert(job_id).second) {
    return errors::InvalidArgument(
        "Duplicate job ID in cluster specification: ", job_id);
  }
  for (const auto& id_host_port : host_ports) {
    TF_RETURN_IF_ERROR(ValidateHostPortPair(id_host_port.second));
  }
  host_ports_jobs_.emplace_back(job_id, host_ports);
  return Status::OK();
}

namespace {

// GrpcChannelCache that caches results to FindWorkerChannel() calls.
using CachingGrpcChannelCache = GenericCachingChannelCache<GrpcChannelCache>;

// A ChannelCache that is the union of multiple ChannelCaches.
// Takes ownership of the caches passed to the constructor.
class MultiGrpcChannelCache : public CachingGrpcChannelCache {
 public:
  explicit MultiGrpcChannelCache(const std::vector<GrpcChannelCache*>& caches,
                                 int num_channels_per_target)
      : CachingGrpcChannelCache(num_channels_per_target), caches_(caches) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_9(mht_9_v, 400, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "MultiGrpcChannelCache");
}

  ~MultiGrpcChannelCache() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_10(mht_10_v, 405, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "~MultiGrpcChannelCache");

    for (GrpcChannelCache* cache : caches_) {
      delete cache;
    }
  }

  void ListWorkers(std::vector<string>* workers) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_11(mht_11_v, 414, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "ListWorkers");

    for (GrpcChannelCache* cache : caches_) {
      cache->ListWorkers(workers);
    }
  }

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) override {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("job_name: \"" + job_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_12(mht_12_v, 425, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "ListWorkersInJob");

    for (GrpcChannelCache* cache : caches_) {
      cache->ListWorkersInJob(job_name, workers);
    }
  }

  string TranslateTask(const string& target) override {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_13(mht_13_v, 435, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "TranslateTask");

    mutex_lock l(mu_);  // could use reader lock
    GrpcChannelCache* cache = gtl::FindPtrOrNull(target_caches_, target);
    if (cache == nullptr) {
      for (GrpcChannelCache* c : caches_) {
        string r = c->TranslateTask(target);
        if (!r.empty()) {
          target_caches_.insert({target, c});
          cache = c;
          break;
        }
      }
    }
    CHECK(cache) << "Could not find GrpcChannelCache holding channel for "
                 << target;
    return cache->TranslateTask(target);
  }

 protected:
  SharedGrpcChannelPtr FindChannelOnce(const string& target) override {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_14(mht_14_v, 458, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "FindChannelOnce");

    for (GrpcChannelCache* cache : caches_) {
      SharedGrpcChannelPtr ch(cache->FindWorkerChannel(target));
      if (ch) {
        mutex_lock l(mu_);
        target_caches_.insert({target, cache});
        return ch;
      }
    }
    return nullptr;
  }

 private:
  // List of channels used by this MultiGrpcChannelCache.
  const std::vector<GrpcChannelCache*> caches_;

  mutex mu_;
  // Cache of channels keyed by the target they are handling.
  // The same GrpcChannelCache can appear multiple times in the cache.
  std::unordered_map<string, GrpcChannelCache*> target_caches_
      TF_GUARDED_BY(mu_);
};

class SparseGrpcChannelCache : public CachingGrpcChannelCache {
 public:
  SparseGrpcChannelCache(const string& job_id,
                         const std::map<int, string>& host_ports,
                         ChannelCreationFunction channel_func,
                         int num_channels_per_target)
      : CachingGrpcChannelCache(num_channels_per_target),
        job_id_(job_id),
        host_ports_(host_ports),
        channel_func_(std::move(channel_func)) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("job_id: \"" + job_id + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_15(mht_15_v, 494, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "SparseGrpcChannelCache");

    LOG(INFO) << "Initialize GrpcChannelCache for job " << ToString();
  }
  ~SparseGrpcChannelCache() override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_16(mht_16_v, 500, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "~SparseGrpcChannelCache");
}

  void ListWorkers(std::vector<string>* workers) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_17(mht_17_v, 505, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "ListWorkers");

    workers->reserve(workers->size() + host_ports_.size());
    for (const auto& id_host_port : host_ports_) {
      workers->emplace_back(MakeAddress(job_id_, id_host_port.first));
    }
  }

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) override {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("job_name: \"" + job_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_18(mht_18_v, 517, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "ListWorkersInJob");

    if (job_name == job_id_) {
      ListWorkers(workers);
    }
  }

  string TranslateTask(const string& target) override {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_19(mht_19_v, 527, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "TranslateTask");

    DeviceNameUtils::ParsedName parsed;
    if (!DeviceNameUtils::ParseFullName(target, &parsed)) {
      LOG(WARNING) << "Invalid target: " << target;
      return "";
    }

    if (!parsed.has_job || parsed.job != job_id_) {
      return "";
    }
    if (!parsed.has_replica || parsed.replica != 0) {
      LOG(WARNING) << "Replica ID must be 0 in target: " << target;
      return "";
    }
    int32_t task = parsed.has_task ? parsed.task : -1;
    auto iter = host_ports_.find(task);
    if (iter == host_ports_.end()) {
      LOG(WARNING) << "Task " << task << " was not defined in sparse job "
                   << job_id_ << ": " << target;
      return "";
    }
    return iter->second;
  }

 protected:
  SharedGrpcChannelPtr FindChannelOnce(const string& target) override {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_20(mht_20_v, 556, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "FindChannelOnce");

    const string host_port = TranslateTask(target);
    if (host_port.empty()) {
      return nullptr;
    }
    auto chan_ptr = channel_func_(host_port);
    VLOG(5) << "Channel created for: job: " << job_id_
            << " host_port: " << host_port << " target : " << target
            << " Ptr: " << chan_ptr.get();
    return chan_ptr;
  }

 private:
  string ToString() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_21(mht_21_v, 572, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "ToString");

    std::vector<string> task_strings;
    task_strings.reserve(host_ports_.size());
    for (const auto& id_host_port : host_ports_) {
      task_strings.emplace_back(
          strings::StrCat(id_host_port.first, " -> ", id_host_port.second));
    }
    return strings::StrCat(job_id_, " -> {", absl::StrJoin(task_strings, ", "),
                           "}");
  }

  const string job_id_;
  const std::map<int, string> host_ports_;
  const ChannelCreationFunction channel_func_;
  TF_DISALLOW_COPY_AND_ASSIGN(SparseGrpcChannelCache);
};

}  // namespace

GrpcChannelCache* NewGrpcChannelCache(const GrpcChannelSpec& spec,
                                      ChannelCreationFunction channel_func,
                                      const RPCOptions& options) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channelDTcc mht_22(mht_22_v, 596, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel.cc", "NewGrpcChannelCache");

  const int num_jobs = spec.host_ports_jobs().size();
  if (!num_jobs) {
    LOG(ERROR) << "Empty channel spec.";
    return nullptr;
  }
  std::vector<GrpcChannelCache*> caches;
  caches.reserve(num_jobs);
  for (auto& job : spec.host_ports_jobs()) {
    VLOG(2) << "Creating Grpc Channel Cache for: " << job.job_id;
    caches.push_back(
        new SparseGrpcChannelCache(job.job_id, job.host_ports, channel_func,
                                   options.num_channels_per_target()));
  }
  return caches.size() == 1 ? caches[0]
                            : new MultiGrpcChannelCache(
                                  caches, options.num_channels_per_target());
}

}  // end namespace tensorflow
