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
class MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/rendezvous.h"

#include <deque>
#include <functional>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/local_rendezvous.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

Rendezvous::ParsedKey& Rendezvous::ParsedKey::operator=(const ParsedKey& b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/framework/rendezvous.cc", "=");

  const char* b_base = b.buf_.data();
  buf_ = b.buf_;
  src_device = StringPiece(buf_.data() + (b.src_device.data() - b_base),
                           b.src_device.size());
  src = b.src;
  src_incarnation = b.src_incarnation;
  dst_device = StringPiece(buf_.data() + (b.dst_device.data() - b_base),
                           b.dst_device.size());
  dst = b.dst;
  edge_name = StringPiece(buf_.data() + (b.edge_name.data() - b_base),
                          b.edge_name.size());
  return *this;
}

/*  static */
string Rendezvous::CreateKey(const string& src_device, uint64 src_incarnation,
                             const string& dst_device, const string& name,
                             const FrameAndIter& frame_iter) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("src_device: \"" + src_device + "\"");
   mht_1_v.push_back("dst_device: \"" + dst_device + "\"");
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTcc mht_1(mht_1_v, 231, "", "./tensorflow/core/framework/rendezvous.cc", "Rendezvous::CreateKey");

  // NOTE: ';' is not used in the device name's job name.
  //
  // We include both sender and receiver in the key to facilitate
  // debugging. For correctness, we only need to encode the receiver.
  //
  // "src_incarnation" is used to distinguish a worker when it
  // restarts.
  char buf[strings::kFastToBufferSize];
  return strings::StrCat(
      src_device, ";", strings::Uint64ToHexString(src_incarnation, buf), ";",
      dst_device, ";", name, ";", frame_iter.frame_id, ":", frame_iter.iter_id);
}

// Return the prefix of "*s" up to the next occurrence of "delim", or
// the whole remaining string if "delim" is not found.  "*s" is advanced
// past the string returned plus the delimiter (if found).
static StringPiece ConsumeNextPart(StringPiece* s, char delim) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("delim: '" + std::string(1, delim) + "'");
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTcc mht_2(mht_2_v, 252, "", "./tensorflow/core/framework/rendezvous.cc", "ConsumeNextPart");

  for (size_t offset = 0; offset < s->size(); offset++) {
    if ((*s)[offset] == delim) {
      StringPiece result(s->data(), offset);
      s->remove_prefix(offset + 1);  // +1: remove delim, as well
      return result;
    }
  }
  // No delimiter found: return rest of string
  StringPiece result(s->data(), s->size());
  s->remove_prefix(s->size());
  return result;
}

/* static */
Status Rendezvous::ParseKey(StringPiece key, ParsedKey* out) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTcc mht_3(mht_3_v, 270, "", "./tensorflow/core/framework/rendezvous.cc", "Rendezvous::ParseKey");

  if (key.data() == out->buf_.data()) {
    // Caller used our buf_ string directly, so we don't need to copy.  (The
    // SendOp and RecvOp implementations do this, for example).
    DCHECK_EQ(key.size(), out->buf_.size());
  } else {
    // Make a copy that our StringPieces can point at a copy that will persist
    // for the lifetime of the ParsedKey object.
    out->buf_.assign(key.data(), key.size());
  }
  StringPiece s(out->buf_);
  StringPiece parts[5];
  for (int i = 0; i < 5; i++) {
    parts[i] = ConsumeNextPart(&s, ';');
  }
  if (s.empty() &&          // Consumed the whole string
      !parts[4].empty() &&  // Exactly five parts
      DeviceNameUtils::ParseFullName(parts[0], &out->src) &&
      strings::HexStringToUint64(parts[1], &out->src_incarnation) &&
      DeviceNameUtils::ParseFullName(parts[2], &out->dst) &&
      !parts[3].empty()) {
    out->src_device = StringPiece(parts[0].data(), parts[0].size());
    out->dst_device = StringPiece(parts[2].data(), parts[2].size());
    out->edge_name = StringPiece(parts[3].data(), parts[3].size());
    return Status::OK();
  }
  return errors::InvalidArgument("Invalid  rendezvous key: ", key);
}

RendezvousInterface::~RendezvousInterface() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTcc mht_4(mht_4_v, 302, "", "./tensorflow/core/framework/rendezvous.cc", "RendezvousInterface::~RendezvousInterface");
}

Status RendezvousInterface::Recv(const ParsedKey& key, const Args& recv_args,
                                 Tensor* val, bool* is_dead,
                                 int64_t timeout_ms) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTcc mht_5(mht_5_v, 309, "", "./tensorflow/core/framework/rendezvous.cc", "RendezvousInterface::Recv");

  Status ret;
  Notification n;
  RecvAsync(key, recv_args,
            [&ret, &n, val, is_dead](const Status& s, const Args& send_args,
                                     const Args& recv_args, const Tensor& v,
                                     const bool dead) {
              ret = s;
              *val = v;
              *is_dead = dead;
              n.Notify();
            });
  if (timeout_ms > 0) {
    int64_t timeout_us = timeout_ms * 1000;
    bool notified = WaitForNotificationWithTimeout(&n, timeout_us);
    if (!notified) {
      return Status(error::DEADLINE_EXCEEDED,
                    "Timed out waiting for notification");
    }
  } else {
    n.WaitForNotification();
  }
  return ret;
}

Status RendezvousInterface::Recv(const ParsedKey& key, const Args& args,
                                 Tensor* val, bool* is_dead) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTcc mht_6(mht_6_v, 338, "", "./tensorflow/core/framework/rendezvous.cc", "RendezvousInterface::Recv");

  const int64_t no_timeout = 0;
  return Recv(key, args, val, is_dead, no_timeout);
}

namespace {
class LocalRendezvousWrapper : public Rendezvous {
 public:
  LocalRendezvousWrapper() : impl_(this) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTcc mht_7(mht_7_v, 349, "", "./tensorflow/core/framework/rendezvous.cc", "LocalRendezvousWrapper");
}

  Status Send(const ParsedKey& key, const Args& send_args, const Tensor& val,
              const bool is_dead) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTcc mht_8(mht_8_v, 355, "", "./tensorflow/core/framework/rendezvous.cc", "Send");

    return impl_.Send(key, send_args, val, is_dead);
  }

  void RecvAsync(const ParsedKey& key, const Args& recv_args,
                 DoneCallback done) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTcc mht_9(mht_9_v, 363, "", "./tensorflow/core/framework/rendezvous.cc", "RecvAsync");

    impl_.RecvAsync(key, recv_args, std::move(done));
  }

  void StartAbort(const Status& status) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTcc mht_10(mht_10_v, 370, "", "./tensorflow/core/framework/rendezvous.cc", "StartAbort");
 impl_.StartAbort(status); }

 private:
  LocalRendezvous impl_;

  TF_DISALLOW_COPY_AND_ASSIGN(LocalRendezvousWrapper);
};
}  // namespace

Rendezvous* NewLocalRendezvous() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTcc mht_11(mht_11_v, 382, "", "./tensorflow/core/framework/rendezvous.cc", "NewLocalRendezvous");
 return new LocalRendezvousWrapper; }

}  // end namespace tensorflow
