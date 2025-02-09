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
class MHTracer_DTPStensorflowPScorePSframeworkPSlocal_rendezvousDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSlocal_rendezvousDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSlocal_rendezvousDTcc() {
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

#include "tensorflow/core/framework/local_rendezvous.h"

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Represents a blocked Send() or Recv() call in the rendezvous.
struct LocalRendezvous::Item {
  enum Type { kSend = 0, kRecv = 1 };

  Item(Rendezvous::Args send_args, const Tensor& value, bool is_dead)
      : Item(send_args, kSend) {
    send_state.value.Init(value);
    send_state.is_dead = is_dead;
  }

  Item(Rendezvous::Args recv_args, Rendezvous::DoneCallback waiter,
       CancellationToken cancellation_token)
      : Item(recv_args, kRecv) {
    recv_state.waiter.Init(std::move(waiter));
    recv_state.cancellation_token = cancellation_token;
  }

  ~Item() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSlocal_rendezvousDTcc mht_0(mht_0_v, 219, "", "./tensorflow/core/framework/local_rendezvous.cc", "~Item");

    if (args.device_context) {
      args.device_context->Unref();
    }
    if (type == kSend) {
      send_state.value.Destroy();
    } else {
      recv_state.waiter.Destroy();
    }
  }

  const Rendezvous::Args args;
  const Type type;

  // Link to next item in an ItemQueue.
  Item* next = nullptr;

  // The validity of `send_state` or `recv_state` is determined by `type ==
  // kSend` or `type == kRecv` respectively.
  union {
    struct {
      ManualConstructor<Tensor> value;
      bool is_dead;
    } send_state;
    struct {
      ManualConstructor<Rendezvous::DoneCallback> waiter;
      CancellationToken cancellation_token;
    } recv_state;
  };

 private:
  Item(Rendezvous::Args args, Type type) : args(args), type(type) {
    if (args.device_context) {
      args.device_context->Ref();
    }
  }
};

void LocalRendezvous::ItemQueue::push_back(Item* item) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSlocal_rendezvousDTcc mht_1(mht_1_v, 260, "", "./tensorflow/core/framework/local_rendezvous.cc", "LocalRendezvous::ItemQueue::push_back");

  if (TF_PREDICT_TRUE(head == nullptr)) {
    // The queue is empty.
    head = item;
    tail = item;
  } else {
    DCHECK_EQ(tail->type, item->type);
    tail->next = item;
    tail = item;
  }
}

LocalRendezvous::~LocalRendezvous() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSlocal_rendezvousDTcc mht_2(mht_2_v, 275, "", "./tensorflow/core/framework/local_rendezvous.cc", "LocalRendezvous::~LocalRendezvous");

  // Before destroying this rendezvous instance, make sure all the done-callback
  // calls have finished and the tensors have been released from the queue.
  {
    mutex_lock l(mu_);
    while (pending_callback_counter_ != 0) {
      pending_callback_cond_var_.wait_for(l, std::chrono::milliseconds(50));
    }
  }

  if (!table_.empty()) {
    StartAbort(errors::Cancelled("LocalRendezvous deleted"));
  }
}

namespace {
uint64 KeyHash(const StringPiece& k) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSlocal_rendezvousDTcc mht_3(mht_3_v, 294, "", "./tensorflow/core/framework/local_rendezvous.cc", "KeyHash");
 return Hash64(k.data(), k.size()); }
}  // namespace

Status LocalRendezvous::Send(const Rendezvous::ParsedKey& key,
                             const Rendezvous::Args& send_args,
                             const Tensor& val, const bool is_dead) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSlocal_rendezvousDTcc mht_4(mht_4_v, 302, "", "./tensorflow/core/framework/local_rendezvous.cc", "LocalRendezvous::Send");

  uint64 key_hash = KeyHash(key.FullKey());
  DVLOG(2) << "Send " << this << " " << key_hash << " " << key.FullKey();

  if (is_dead) {
    static auto* rendezvous_dead_values_sent = monitoring::Counter<2>::New(
        "/tensorflow/core/rendezvous_dead_values_sent",
        "The number of dead values sent between a pair of devices.",
        "send_device", "recv_device");
    rendezvous_dead_values_sent
        ->GetCell(string(key.src_device), string(key.dst_device))
        ->IncrementBy(1);
  }

  mu_.lock();
  if (!status_.ok()) {
    // Rendezvous has been aborted.
    Status s = status_;
    mu_.unlock();
    return s;
  }

  ItemQueue* queue = &table_[key_hash];
  if (queue->head == nullptr || queue->head->type == Item::kSend) {
    // There is no waiter for this message. Append the message
    // into the queue. The waiter will pick it up when arrives.
    // Only send-related fields need to be filled.
    // TODO(b/143786186): Investigate moving the allocation of `Item` outside
    // the lock.
    DVLOG(2) << "Enqueue Send Item (key:" << key.FullKey() << "). ";
    queue->push_back(new Item(send_args, val, is_dead));
    mu_.unlock();
    return Status::OK();
  }

  DVLOG(2) << "Consume Recv Item (key:" << key.FullKey() << "). ";
  // There is an earliest waiter to consume this message.
  Item* item = queue->head;

  // Delete the queue when the last element has been consumed.
  if (item->next == nullptr) {
    DVLOG(2) << "Clean up Send/Recv queue (key:" << key.FullKey() << "). ";
    table_.erase(key_hash);
  } else {
    queue->head = item->next;
  }

  // Make sure the ref-count of the rendezvous won't reach 0 while the
  // done_callback is running, which would otherwise become deadlock:
  // the done_callback waits for the Unref() to return, while the destructor
  // wiats for the pending_callback_counter to reach 0.
  core::RefCountPtr<const Rendezvous> rc_owner_ref;
  if (rc_owner_) {
    rc_owner_ref.reset(rc_owner_);
    rc_owner_->Ref();
  }
  pending_callback_counter_++;
  // Invoke the done-callback, without holding the lock.
  mu_.unlock();
  DCHECK_EQ(item->type, Item::kRecv);
  (*item->recv_state.waiter)(Status::OK(), send_args, item->args, val, is_dead);
  delete item;
  {
    mutex_lock l(mu_);
    pending_callback_counter_--;
    if (pending_callback_counter_ == 0) {
      pending_callback_cond_var_.notify_all();
    }
  }
  return Status::OK();
}

void LocalRendezvous::RecvAsync(const Rendezvous::ParsedKey& key,
                                const Rendezvous::Args& recv_args,
                                Rendezvous::DoneCallback done) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSlocal_rendezvousDTcc mht_5(mht_5_v, 379, "", "./tensorflow/core/framework/local_rendezvous.cc", "LocalRendezvous::RecvAsync");

  uint64 key_hash = KeyHash(key.FullKey());
  DVLOG(2) << "Recv " << this << " " << key_hash << " " << key.FullKey();

  mu_.lock();
  if (!status_.ok()) {
    // Rendezvous has been aborted.
    Status s = status_;
    mu_.unlock();
    done(s, Rendezvous::Args(), recv_args, Tensor(), false);
    return;
  }

  ItemQueue* queue = &table_[key_hash];
  if (queue->head == nullptr || queue->head->type == Item::kRecv) {
    // There is no message to pick up.
    // Only recv-related fields need to be filled.
    CancellationManager* cm = recv_args.cancellation_manager;
    CancellationToken token = CancellationManager::kInvalidToken;
    bool already_cancelled = false;
    if (cm != nullptr) {
      // Increment the refcount when cancellation manager is present, to make
      // sure the rendezvous outlives the recv and its cancel callbacks.
      // This refcount is dropped in exactly one of the following cases:
      // (1) Recv registers cancellation callback to cm, and then cm is
      //     cancelled, unref in the cancellation callback;
      // (2) Recv registers cancellation callback to cm, but cm is already
      //     cancelled, unref in the already_cancelled check;
      // (3) Recv is successful, and item done callback finishes deregistering
      //     the cancellation callback, unref in the item done callback;
      // (4) Recv is successful, but the item done callback fails to deregister
      //     the cancellation callback because cm already StartCancel, in this
      //     case the cancellation callback will be invoked by the cm anyway,
      //     unref in the cancellation callback.
      if (rc_owner_) rc_owner_->Ref();
      token = cm->get_cancellation_token();
      already_cancelled = !cm->RegisterCallback(token, [this, token, key_hash] {
        Item* item = nullptr;
        {
          mutex_lock l(mu_);
          ItemQueue* queue = &table_[key_hash];
          // Find an item in the queue with a cancellation token that matches
          // `token`, and remove it.
          if (queue->head != nullptr && queue->head->type == Item::kRecv) {
            for (Item *prev = nullptr, *curr = queue->head; curr != nullptr;
                 prev = curr, curr = curr->next) {
              if (curr->recv_state.cancellation_token == token) {
                item = curr;
                if (queue->head->next == nullptr) {
                  // We have a single-element queue, so we can erase it from
                  // the table.
                  table_.erase(key_hash);
                } else {
                  // Remove the current item from the queue.
                  if (curr == queue->head) {
                    DCHECK_EQ(prev, nullptr);
                    queue->head = curr->next;
                  } else {
                    DCHECK_NE(prev, nullptr);
                    prev->next = curr->next;
                  }
                  if (queue->tail == curr) {
                    queue->tail = prev;
                  }
                }
                break;
              }
            }
          }
        }

        if (item != nullptr) {
          (*item->recv_state.waiter)(
              StatusGroup::MakeDerived(
                  errors::Cancelled("RecvAsync is cancelled.")),
              Rendezvous::Args(), item->args, Tensor(), /*is_dead=*/false);
          delete item;
        }
        // Unref case (1) and (4)
        if (rc_owner_) rc_owner_->Unref();
      });
    }
    if (already_cancelled) {
      mu_.unlock();
      // Unref case (2)
      if (rc_owner_) rc_owner_->Unref();
      done(StatusGroup::MakeDerived(
               errors::Cancelled("RecvAsync is cancelled.")),
           Rendezvous::Args(), recv_args, Tensor(), /*is_dead=*/false);
      return;
    }

    DVLOG(2) << "Enqueue Recv Item (key:" << key.FullKey() << "). ";

    // TODO(b/143786186): Investigate moving the allocation of `Item` outside
    // the lock.
    if (cm != nullptr) {
      // NOTE(mrry): We must wrap `done` with code that deregisters the
      // cancellation callback before calling the `done` callback, because the
      // cancellation manager may no longer be live after `done` is called.
      queue->push_back(new Item(
          recv_args,
          [this, cm, token, done = std::move(done)](
              const Status& s, const Rendezvous::Args& send_args,
              const Rendezvous::Args& recv_args, const Tensor& v, bool dead) {
            // TryDeregisterCallback returns true when the cancellation callback
            // is successfully deregistered. If it fails because the CM already
            // StartAbort, Unref will happen inside the cancellation callback
            // when called by the CM.
            if (cm->TryDeregisterCallback(token)) {
              // Unref case (3)
              if (this->rc_owner_) this->rc_owner_->Unref();
            }
            done(s, send_args, recv_args, v, dead);
          },
          token));
    } else {
      queue->push_back(new Item(recv_args, std::move(done), token));
    }

    mu_.unlock();
    return;
  }

  DVLOG(2) << "Consume Send Item (key:" << key.FullKey() << "). ";
  // A message has already arrived and is queued in the table under
  // this key.  Consumes the message and invokes the done closure.
  Item* item = queue->head;

  // Delete the queue when the last element has been consumed.
  if (item->next == nullptr) {
    DVLOG(2) << "Clean up Send/Recv queue (key:" << key.FullKey() << "). ";
    table_.erase(key_hash);
  } else {
    queue->head = item->next;
  }

  // Make sure the ref-count of the rendezvous won't reach 0 while the
  // done_callback is running, which would otherwise become deadlock:
  // the done_callback waits for the Unref() to return, while the destructor
  // wiats for the pending_callback_counter to reach 0.
  core::RefCountPtr<const Rendezvous> rc_owner_ref;
  if (rc_owner_) {
    rc_owner_ref.reset(rc_owner_);
    rc_owner_->Ref();
  }
  pending_callback_counter_++;
  // Invoke the done-callback, without holding the lock.
  mu_.unlock();
  DCHECK_EQ(item->type, Item::kSend);
  done(Status::OK(), item->args, recv_args, *item->send_state.value,
       item->send_state.is_dead);
  delete item;
  {
    mutex_lock l(mu_);
    pending_callback_counter_--;
    if (pending_callback_counter_ == 0) {
      pending_callback_cond_var_.notify_all();
    }
  }
}

void LocalRendezvous::StartAbort(const Status& status) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSlocal_rendezvousDTcc mht_6(mht_6_v, 544, "", "./tensorflow/core/framework/local_rendezvous.cc", "LocalRendezvous::StartAbort");

  CHECK(!status.ok());
  Table table;
  {
    mutex_lock l(mu_);
    status_.Update(status);
    table_.swap(table);
  }
  for (auto& p : table) {
    Item* item = p.second.head;
    while (item != nullptr) {
      if (item->type == Item::kRecv) {
        (*item->recv_state.waiter)(status, Rendezvous::Args(),
                                   Rendezvous::Args(), Tensor(), false);
      }
      Item* to_delete = item;
      item = item->next;
      delete to_delete;
    }
  }
}

Status LocalRendezvous::status() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSlocal_rendezvousDTcc mht_7(mht_7_v, 569, "", "./tensorflow/core/framework/local_rendezvous.cc", "LocalRendezvous::status");

  mu_.lock();
  Status s = status_;
  mu_.unlock();
  return s;
}

}  // namespace tensorflow
