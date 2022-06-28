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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/buf_rendezvous.h"

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"

namespace tensorflow {
namespace {
void DeregisterCancellation(BufRendezvous::Hook* h) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/common_runtime/buf_rendezvous.cc", "DeregisterCancellation");

  if (h->cancellation_manager != nullptr) {
    h->cancellation_manager->DeregisterCallback(h->cancellation_token);
    h->cancellation_manager = nullptr;
    h->cancellation_token = CancellationManager::kInvalidToken;
  }
}
}  // namespace

BufRendezvous::~BufRendezvous() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/common_runtime/buf_rendezvous.cc", "BufRendezvous::~BufRendezvous");

  mutex_lock l(mu_);
  if (!hook_table_.empty()) {
    PurgeTable(errors::Internal("Delete called on non-empty BufRendezvous"),
               &hook_table_);
  }
}

void BufRendezvous::StartAbort(const Status& s) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTcc mht_2(mht_2_v, 221, "", "./tensorflow/core/common_runtime/buf_rendezvous.cc", "BufRendezvous::StartAbort");

  CHECK(!s.ok());
  HookTable dummy_table;
  {
    mutex_lock l(mu_);
    // Use a "derived" status as the status for the rendezvous. Derived
    // status messages are ignored when aggregating errors across devices: this
    // allows us to prefer our original status message over any cancellation
    // related errors.
    status_.Update(StatusGroup::MakeDerived(s));
    hook_table_.swap(dummy_table);
  }
  PurgeTable(s, &dummy_table);
}

void BufRendezvous::PurgeTable(const Status& s, HookTable* table) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/common_runtime/buf_rendezvous.cc", "BufRendezvous::PurgeTable");

  for (auto& it : *table) {
    Hook* h = it.second;
    if (h->cancellation_manager != nullptr) {
      h->cancellation_manager->TryDeregisterCallback(h->cancellation_token);
    }
    if (h->cons_cb != nullptr) {
      h->cons_cb(s, nullptr);
    }
    if (h->prod_cb != nullptr) {
      h->prod_cb(s);
    }
    delete h;
  }
  table->clear();
}

string BufRendezvous::Hook::DebugString() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTcc mht_4(mht_4_v, 259, "", "./tensorflow/core/common_runtime/buf_rendezvous.cc", "BufRendezvous::Hook::DebugString");

  return absl::StrCat("[dev:", (prod_dev ? prod_dev->name() : "none"),
                      ", ctx:", reinterpret_cast<uint64>(prod_ctx),
                      ", val:", reinterpret_cast<uint64>(prod_value),
                      ", pcb:", reinterpret_cast<uint64>(&prod_cb),
                      ", ccb:", reinterpret_cast<uint64>(&cons_cb), "]");
}

void BufRendezvous::ProvideBuf(const string& key, Device* dev,
                               DeviceContext* dev_ctx, const Tensor* v,
                               const AllocatorAttributes& attr,
                               const ProducerCallback& done,
                               CancellationManager* cancellation_manager) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTcc mht_5(mht_5_v, 275, "", "./tensorflow/core/common_runtime/buf_rendezvous.cc", "BufRendezvous::ProvideBuf");

  Hook* h = nullptr;
  Status providebuf_status;
  do {
    mutex_lock l(mu_);
    if (!status_.ok()) {
      providebuf_status = status_;
      break;
    } else {
      CancellationToken cancellation_token = CancellationManager::kInvalidToken;
      auto it = hook_table_.find(key);
      if (it == hook_table_.end()) {
        if (cancellation_manager != nullptr) {
          cancellation_token = cancellation_manager->get_cancellation_token();
        }
        h = new Hook(cancellation_manager, cancellation_token);
        it = hook_table_.insert(std::make_pair(key, h)).first;
      } else {
        if (it->second->prod_cb != nullptr) {
          providebuf_status = errors::Internal(
              "BufRendezvous::ProvideBuf already called for key ", key);
          break;
        }
        h = it->second;
      }
      // Populate Hook with all of the prod values.
      h->prod_dev = dev;
      h->prod_ctx = dev_ctx;
      h->prod_value = v;
      h->prod_attr = attr;
      h->prod_cb = done;
      if (h->cons_cb != nullptr) {
        // If consumer is waiting, kick off right away, removing Hook from
        // table.
        hook_table_.erase(it);
      } else {
        if (cancellation_manager != nullptr &&
            !cancellation_manager->RegisterCallback(
                cancellation_token, [this, key]() { CancelHook(key); })) {
          // Register cancellation callback with CancellationManager.  If it is
          // already cancelled, call done immediately with cancelled status.
          providebuf_status = errors::Cancelled(
              "Operation was cancelled for BufRendezvous key ", key);
          hook_table_.erase(it);
          delete h;
        }
        h = nullptr;
      }
    }
  } while (false);
  if (h) {
    DeregisterCancellation(h);
    h->cons_cb(Status::OK(), h);
  }
  if (!providebuf_status.ok()) {
    done(providebuf_status);
  }
}

void BufRendezvous::ConsumeBuf(const string& key, const string& device_name,
                               const uint64 device_incarnation,
                               const ConsumerCallback& done,
                               CancellationManager* cancellation_manager) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("key: \"" + key + "\"");
   mht_6_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTcc mht_6(mht_6_v, 342, "", "./tensorflow/core/common_runtime/buf_rendezvous.cc", "BufRendezvous::ConsumeBuf");

  // Check the incarnation in the request matches the current device
  // incarnation of the producer.
  Device* device;
  Status consumebuf_status = dev_mgr_->LookupDevice(device_name, &device);
  if (consumebuf_status.ok() &&
      device->attributes().incarnation() != device_incarnation) {
    consumebuf_status = errors::FailedPrecondition(
        "RecvBuf expects a different device incarnation: ", device_incarnation,
        " vs. ", device->attributes().incarnation(),
        ". Your worker job that contains the device (\"", device_name,
        "\") was probably restarted. Check your "
        "worker job for the reason why it was restarted.");
  }
  if (!consumebuf_status.ok()) {
    done(consumebuf_status, nullptr);
    return;
  }

  Hook* existing_hook = nullptr;
  do {
    mutex_lock l(mu_);
    if (!status_.ok()) {
      consumebuf_status = status_;
      break;
    }
    auto it = hook_table_.find(key);
    if (it != hook_table_.end()) {
      // Prepare to consume immediately.
      if (it->second->cons_cb) {
        consumebuf_status =
            errors::Internal("Second consumer arrived for key ", key);
        break;
      }
      existing_hook = it->second;
      hook_table_.erase(it);
      existing_hook->cons_cb = done;
    } else {
      // Hang consumer callback on the Hook.
      CancellationToken cancellation_token = CancellationManager::kInvalidToken;
      bool already_cancelled = false;
      if (cancellation_manager != nullptr) {
        cancellation_token = cancellation_manager->get_cancellation_token();
        already_cancelled = !cancellation_manager->RegisterCallback(
            cancellation_token, [this, key]() { CancelHook(key); });
      }
      if (already_cancelled) {
        consumebuf_status = errors::Cancelled(
            "Operation was cancelled for BufRendezvous key ", key);
      } else {
        Hook* h = new Hook(cancellation_manager, cancellation_token);
        h->cons_cb = done;
        it = hook_table_.insert(std::make_pair(key, h)).first;
        return;
      }
    }
  } while (false);
  if (existing_hook) {
    DeregisterCancellation(existing_hook);
    existing_hook->cons_cb(Status::OK(), existing_hook);
    return;
  }
  if (!consumebuf_status.ok()) {
    done(consumebuf_status, nullptr);
    return;
  }
}

void BufRendezvous::CancelHook(const string& key) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTcc mht_7(mht_7_v, 414, "", "./tensorflow/core/common_runtime/buf_rendezvous.cc", "BufRendezvous::CancelHook");

  Hook* h = nullptr;
  {
    mutex_lock l(mu_);
    auto it = hook_table_.find(key);
    if (it == hook_table_.end()) return;
    h = it->second;
    hook_table_.erase(it);
  }
  if (h != nullptr) {
    auto s = errors::Cancelled("Operation was cancelled for BufRendezvous key ",
                               key);
    if (h->prod_cb != nullptr) {
      h->prod_cb(s);
    }
    if (h->cons_cb != nullptr) {
      h->cons_cb(s, /*Hook=*/nullptr);
    }
    delete h;
  }
}

/*static*/
void BufRendezvous::DoneWithHook(Hook* h) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTcc mht_8(mht_8_v, 440, "", "./tensorflow/core/common_runtime/buf_rendezvous.cc", "BufRendezvous::DoneWithHook");

  h->prod_cb(Status::OK());
  delete h;
}

void BufRendezvous::LogContents() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTcc mht_9(mht_9_v, 448, "", "./tensorflow/core/common_runtime/buf_rendezvous.cc", "BufRendezvous::LogContents");

  mutex_lock l(mu_);
  LOG(INFO) << strings::StrCat("BufRendezvous ",
                               strings::Hex(reinterpret_cast<uint64>(this)),
                               " step_id=", step_id_, " current contents:");
  for (const auto& it : hook_table_) {
    LOG(INFO) << it.first << ":" << it.second->DebugString();
  }
}

}  // namespace tensorflow
