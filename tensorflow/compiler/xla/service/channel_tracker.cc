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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSchannel_trackerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSchannel_trackerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSchannel_trackerDTcc() {
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

#include "tensorflow/compiler/xla/service/channel_tracker.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

ChannelTracker::ChannelTracker() : next_channel_(1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSchannel_trackerDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/service/channel_tracker.cc", "ChannelTracker::ChannelTracker");
}

StatusOr<ChannelHandle> ChannelTracker::NewChannel(
    ChannelHandle::ChannelType type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSchannel_trackerDTcc mht_1(mht_1_v, 205, "", "./tensorflow/compiler/xla/service/channel_tracker.cc", "ChannelTracker::NewChannel");

  if (type != ChannelHandle::DEVICE_TO_DEVICE &&
      type != ChannelHandle::HOST_TO_DEVICE &&
      type != ChannelHandle::DEVICE_TO_HOST) {
    return InvalidArgument("Invalid channel type: %d", type);
  }
  absl::MutexLock lock(&channel_mutex_);

  // Create a new channel handle with a unique value.
  ChannelHandle new_handle = AllocateHandle(type);

  // Register a channel object associated with the handle.
  Channel channel;
  channel.has_sender = false;
  channel.receiver_count = 0;
  channel.type = type;
  opaque_to_channel_[new_handle.handle()] = channel;

  return new_handle;
}

Status ChannelTracker::RegisterSend(const ChannelHandle& handle) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSchannel_trackerDTcc mht_2(mht_2_v, 229, "", "./tensorflow/compiler/xla/service/channel_tracker.cc", "ChannelTracker::RegisterSend");

  absl::MutexLock lock(&channel_mutex_);
  return RegisterSendInternal(handle);
}

Status ChannelTracker::RegisterRecv(const ChannelHandle& handle) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSchannel_trackerDTcc mht_3(mht_3_v, 237, "", "./tensorflow/compiler/xla/service/channel_tracker.cc", "ChannelTracker::RegisterRecv");

  absl::MutexLock lock(&channel_mutex_);
  return RegisterRecvInternal(handle);
}

ChannelHandle ChannelTracker::AllocateHandle(ChannelHandle::ChannelType type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSchannel_trackerDTcc mht_4(mht_4_v, 245, "", "./tensorflow/compiler/xla/service/channel_tracker.cc", "ChannelTracker::AllocateHandle");

  int64_t handle_value = next_channel_++;
  ChannelHandle result;
  result.set_handle(handle_value);
  result.set_type(type);
  return result;
}

Status ChannelTracker::RegisterSendInternal(const ChannelHandle& handle) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSchannel_trackerDTcc mht_5(mht_5_v, 256, "", "./tensorflow/compiler/xla/service/channel_tracker.cc", "ChannelTracker::RegisterSendInternal");

  if (!opaque_to_channel_.contains(handle.handle())) {
    return NotFound("channel handle not found: %d", handle.handle());
  }
  Channel& channel = opaque_to_channel_[handle.handle()];
  if (channel.type == ChannelHandle::HOST_TO_DEVICE) {
    return FailedPrecondition(
        "host-to-device channels cannot be used with a Send operation; "
        "channel handle: %d",
        handle.handle());
  }

  if (channel.has_sender) {
    return FailedPrecondition(
        "when registering send, passed a channel handle that is already used "
        "by a sender: %d",
        handle.handle());
  }
  channel.has_sender = true;
  return Status::OK();
}

Status ChannelTracker::RegisterRecvInternal(const ChannelHandle& handle) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSchannel_trackerDTcc mht_6(mht_6_v, 281, "", "./tensorflow/compiler/xla/service/channel_tracker.cc", "ChannelTracker::RegisterRecvInternal");

  if (!opaque_to_channel_.contains(handle.handle())) {
    return NotFound("channel handle not found: %d", handle.handle());
  }
  Channel& channel = opaque_to_channel_[handle.handle()];
  if (channel.type == ChannelHandle::DEVICE_TO_HOST) {
    return FailedPrecondition(
        "device-to-host channels cannot be used with a Recv operation; "
        "channel handle: %d",
        handle.handle());
  }

  // TODO(b/33942691): Allow more than 1 receivers for broadcast.
  if (channel.receiver_count >= 1) {
    return FailedPrecondition(
        "when registering recv, passed a channel handle that is already used "
        "by a receiver: %d",
        handle.handle());
  }
  channel.receiver_count += 1;
  return Status::OK();
}

}  // namespace xla
