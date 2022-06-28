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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CHANNEL_COMMON_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CHANNEL_COMMON_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channel_commonDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channel_commonDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channel_commonDTh() {
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


#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// GenericCachingChannelCache that caches results to FindWorkerChannel() calls.
// To use instantiate with the type of channel cache needed.
// GenericCachingChannelCache allows using multiple channels to communiate with
// same target to provide throughput gains. When multiple channels exist for
// the same target they are chosen in a simple round robin fashion on each call
// to FindWorkerChannel.
template <typename ChannelCacheT>
class GenericCachingChannelCache : public ChannelCacheT {
 public:
  explicit GenericCachingChannelCache(int num_channels_per_target)
      : num_channels_per_target_(
            num_channels_per_target > 0 ? num_channels_per_target : 1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channel_commonDTh mht_0(mht_0_v, 209, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel_common.h", "GenericCachingChannelCache");
}

  ~GenericCachingChannelCache() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channel_commonDTh mht_1(mht_1_v, 214, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel_common.h", "~GenericCachingChannelCache");
}

  SharedGrpcChannelPtr FindWorkerChannel(const string& target) override {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channel_commonDTh mht_2(mht_2_v, 220, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel_common.h", "FindWorkerChannel");

    {
      mutex_lock l(mu_);
      auto iter = channels_.find(target);
      if (iter != channels_.end()) {
        return GetNextChannelPtrAndUpdateState(iter->second);
      }
    }
    ChannelState new_chan_state;
    for (int indx = 0; indx < num_channels_per_target_; indx++) {
      auto ch = FindChannelOnce(target);
      if (!ch) return nullptr;
      new_chan_state.channels.push_back(ch);
    }
    new_chan_state.last_used = num_channels_per_target_ - 1;

    {
      mutex_lock l(mu_);
      typename absl::flat_hash_map<string, ChannelState>::iterator iter;
      bool was_inserted;
      std::tie(iter, was_inserted) = channels_.insert({target, new_chan_state});
      VLOG(2) << "Channel cache for target: " << target
              << " Size: " << new_chan_state.channels.size()
              << " insertion: " << was_inserted;
      return GetNextChannelPtrAndUpdateState(iter->second);
    }
  }

 protected:
  // Find the ClientChannel for "target".  Only called when no channel was
  // found in the channels_ cache for "target".  A non nullptr result will be
  // cached in channels_.
  virtual SharedGrpcChannelPtr FindChannelOnce(const string& target) = 0;

 private:
  struct ChannelState {
    std::vector<SharedGrpcChannelPtr> channels;
    int last_used;
  };

  // Should be called with mu_ held.
  SharedGrpcChannelPtr GetNextChannelPtrAndUpdateState(
      ChannelState& chan_state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channel_commonDTh mht_3(mht_3_v, 265, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel_common.h", "GetNextChannelPtrAndUpdateState");

    // Following statement is marked as Crash OK as this is an invariant of
    // code flow in this class.
    CHECK_EQ(chan_state.channels.size(), num_channels_per_target_);  // Crash OK
    chan_state.last_used =
        (chan_state.last_used + 1) % num_channels_per_target_;
    return chan_state.channels[chan_state.last_used];
  }

  const int num_channels_per_target_;
  // TODO(zhifengc): Eviction when the map becomes too big.
  mutex mu_;
  absl::flat_hash_map<string, ChannelState> channels_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CHANNEL_COMMON_H_
