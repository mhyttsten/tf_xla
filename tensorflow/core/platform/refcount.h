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

#ifndef TENSORFLOW_CORE_PLATFORM_REFCOUNT_H_
#define TENSORFLOW_CORE_PLATFORM_REFCOUNT_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh() {
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


#include <atomic>
#include <map>
#include <memory>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace core {

class RefCounted {
 public:
  // Initial reference count is one.
  RefCounted();

  // Increments reference count by one.
  void Ref() const;

  // Decrements reference count by one.  If the count remains
  // positive, returns false.  When the count reaches zero, returns
  // true and deletes this, in which case the caller must not access
  // the object afterward.
  bool Unref() const;

  // Gets the current reference count.
  int_fast32_t RefCount() const;

  // Return whether the reference count is one.
  // If the reference count is used in the conventional way, a
  // reference count of 1 implies that the current thread owns the
  // reference and no other thread shares it.
  // This call performs the test for a reference count of one, and
  // performs the memory barrier needed for the owning thread
  // to act on the object, knowing that it has exclusive access to the
  // object.
  bool RefCountIsOne() const;

 protected:
  // Make destructor protected so that RefCounted objects cannot
  // be instantiated directly. Only subclasses can be instantiated.
  virtual ~RefCounted();

  // Increments reference count by one if the object is not being destructed.
  // This function is used by WeakRefCounted for securely acquiring a
  // strong reference. It is only safe to call this as part of the weak
  // reference implementation.
  bool TryRef() const;

 private:
  mutable std::atomic_int_fast32_t ref_;

  RefCounted(const RefCounted&) = delete;
  void operator=(const RefCounted&) = delete;
};

// A deleter class to form a std::unique_ptr that unrefs objects.
struct RefCountDeleter {
  void operator()(const RefCounted* o) const { o->Unref(); }
};

// A unique_ptr that unrefs the owned object on destruction.
template <typename T>
using RefCountPtr = std::unique_ptr<T, RefCountDeleter>;

// Helper class to unref an object when out-of-scope.
class ScopedUnref {
 public:
  explicit ScopedUnref(const RefCounted* o) : obj_(o) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_0(mht_0_v, 256, "", "./tensorflow/core/platform/refcount.h", "ScopedUnref");
}
  ~ScopedUnref() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_1(mht_1_v, 260, "", "./tensorflow/core/platform/refcount.h", "~ScopedUnref");

    if (obj_) obj_->Unref();
  }

 private:
  const RefCounted* obj_;

  ScopedUnref(const ScopedUnref&) = delete;
  void operator=(const ScopedUnref&) = delete;
};

// Forward declaration for friend class of WeakRefCounted.
template <typename T>
class WeakPtr;

// A WeakNotifyFn is called when the weakly referred object is being destroyed.
// The object may already be destructed when the call occurs. A WeakNotifyFn
// can be passed into WeakPtr at construction.
using WeakNotifyFn = std::function<void()>;

// A base class for RefCounted objects that allow weak references by WeakPtr.
// WeakRefCounted and every WeakPtr to it, each holds a strong reference to a
// WeakRefData.
//
// If the WeakRefCounted is valid, WeakPtr::GetNewRef() returns a new strong
// reference to the WeakRefCounted.
// If the WeakRefCounted is being destructed, `WeakRefCounted::ref_ == 0`;
// if the WeakRefcounted is already destructed,`WeakRefData::ptr == nullptr`.
// In either case, WeakPtr::GetNewRef() returns a nullptr.
class WeakRefCounted : public RefCounted {
 public:
  int WeakRefCount() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_2(mht_2_v, 294, "", "./tensorflow/core/platform/refcount.h", "WeakRefCount");

    // Each weak ref owns one ref to data_, and *this owns the last one.
    return data_->RefCount() - 1;
  }

 protected:
  ~WeakRefCounted() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_3(mht_3_v, 303, "", "./tensorflow/core/platform/refcount.h", "~WeakRefCounted");
 data_->Notify(); }

 private:
  struct WeakRefData : public RefCounted {
    explicit WeakRefData(WeakRefCounted* ptr) : ptr(ptr), next_notifier_id(1) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_4(mht_4_v, 310, "", "./tensorflow/core/platform/refcount.h", "WeakRefData");
}

    mutable mutex mu;
    WeakRefCounted* ptr TF_GUARDED_BY(mu);
    std::map<int, WeakNotifyFn> notifiers;
    int next_notifier_id;

    // Notifies WeakPtr instansces that this object is being destructed.
    void Notify() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_5(mht_5_v, 321, "", "./tensorflow/core/platform/refcount.h", "Notify");

      mutex_lock ml(mu);

      while (!notifiers.empty()) {
        auto iter = notifiers.begin();
        WeakNotifyFn notify_fn = std::move(iter->second);
        notifiers.erase(iter);

        mu.unlock();
        notify_fn();
        mu.lock();
      }
      ptr = nullptr;
    }

    WeakRefCounted* GetNewRef() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_6(mht_6_v, 339, "", "./tensorflow/core/platform/refcount.h", "GetNewRef");

      mutex_lock ml(mu);
      if (ptr != nullptr && ptr->TryRef()) {
        return ptr;
      }
      return nullptr;
    }

    // Inserts notify_fn and returns a non-zero id.
    // Returns 0 if insertion fails due to the object is being destroyed.
    // 0 is also used by WeakPtr to represent "no notify_fn".
    int AddNotifier(WeakNotifyFn notify_fn) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_7(mht_7_v, 353, "", "./tensorflow/core/platform/refcount.h", "AddNotifier");

      mutex_lock ml(mu);
      if (ptr == nullptr) {
        return 0;
      }
      int notifier_id = next_notifier_id++;
      notifiers.emplace(notifier_id, std::move(notify_fn));
      return notifier_id;
    }

    void RemoveNotifier(int notifier_id) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_8(mht_8_v, 366, "", "./tensorflow/core/platform/refcount.h", "RemoveNotifier");

      mutex_lock ml(mu);
      notifiers.erase(notifier_id);
    }
  };

  RefCountPtr<WeakRefData> data_{new WeakRefData(this)};

  template <typename T>
  friend class WeakPtr;
  // MSVC14 workaround: access permission of a nested class member is not
  // treated as an ordinary member in MSVC14.
  friend struct WeakRefData;
};

// A weak reference to a WeakRefCounted object. Refer to WeakRefCounted.
template <typename T>
class WeakPtr {
 public:
  // Creates a weak reference.
  // When the object is being destroyed, notify_fn is called.
  explicit WeakPtr(WeakRefCounted* ptr, WeakNotifyFn notify_fn = nullptr)
      : data_(nullptr), notifier_id_(0) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_9(mht_9_v, 391, "", "./tensorflow/core/platform/refcount.h", "WeakPtr");

    if (ptr != nullptr) {
      ptr->data_->Ref();
      data_.reset(ptr->data_.get());
      if (notify_fn) {
        notifier_id_ = data_->AddNotifier(notify_fn);
      }
    }
  }

  ~WeakPtr() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_10(mht_10_v, 404, "", "./tensorflow/core/platform/refcount.h", "~WeakPtr");

    if (data_ != nullptr && notifier_id_ != 0) {
      data_->RemoveNotifier(notifier_id_);
    }
  }

  // NOTE(feyu): change data_ to a IntrusivePtr to make WeakPtr copyable.
  WeakPtr(const WeakPtr& other) = delete;
  WeakPtr& operator=(const WeakPtr& other) = delete;

  WeakPtr(WeakPtr&& other) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_11(mht_11_v, 417, "", "./tensorflow/core/platform/refcount.h", "WeakPtr");

    data_ = std::move(other.data_);
    notifier_id_ = other.notifier_id_;
    other.notifier_id_ = 0;
  }

  WeakPtr& operator=(WeakPtr&& other) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_12(mht_12_v, 426, "", "./tensorflow/core/platform/refcount.h", "=");

    if (this != &other) {
      if (data_ != nullptr && notifier_id_ != 0) {
        data_->RemoveNotifier(notifier_id_);
      }
      data_ = std::move(other.data_);
      notifier_id_ = other.notifier_id_;
      other.notifier_id_ = 0;
    }
    return *this;
  }

  // Returns a new strong reference to the referred object, or nullptr if the
  // object is in an invalid state (being destructed or already destructed).
  RefCountPtr<T> GetNewRef() const {
    RefCountPtr<T> ref;
    if (data_ != nullptr) {
      WeakRefCounted* ptr = data_->GetNewRef();
      ref.reset(static_cast<T*>(ptr));
    }
    return std::move(ref);
  }

 private:
  RefCountPtr<WeakRefCounted::WeakRefData> data_;
  int notifier_id_;
};

// Inlined routines, since these are performance critical
inline RefCounted::RefCounted() : ref_(1) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_13(mht_13_v, 458, "", "./tensorflow/core/platform/refcount.h", "RefCounted::RefCounted");
}

inline RefCounted::~RefCounted() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_14(mht_14_v, 463, "", "./tensorflow/core/platform/refcount.h", "RefCounted::~RefCounted");

  // A destructing object has ref_ == 0.
  // It is a bug if the object is resurrected (ref_ > 0) before delete is
  // called by Unref().
  DCHECK_EQ(ref_.load(), 0);
}

inline void RefCounted::Ref() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_15(mht_15_v, 473, "", "./tensorflow/core/platform/refcount.h", "RefCounted::Ref");

  // Ref() uses relaxed order because it is never called with old_ref == 0.
  // When old_ref >= 1, no actions depend on the new value of ref.
  int_fast32_t old_ref = ref_.fetch_add(1, std::memory_order_relaxed);
  DCHECK_GT(old_ref, 0);
}

inline bool RefCounted::TryRef() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_16(mht_16_v, 483, "", "./tensorflow/core/platform/refcount.h", "RefCounted::TryRef");

  // This is not on a hot path.
  // Be conservative and use seq_cst to prevent racing with Unref() when
  // old_ref == 0, as done in LLVM libstdc++.
  int_fast32_t old_ref = ref_.load();
  while (old_ref != 0) {
    if (ref_.compare_exchange_weak(old_ref, old_ref + 1)) {
      return true;
    }
  }
  // Already destructing, cannot increase ref.
  return false;
}

inline bool RefCounted::Unref() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_17(mht_17_v, 500, "", "./tensorflow/core/platform/refcount.h", "RefCounted::Unref");

  DCHECK_GT(ref_.load(), 0);
  // acq_rel is used to prevent reordering introduces object access after
  // destruction.

  // Using release alone is a bug on systems where acq_rel differs from release.
  // (e.g. arm), according to Herb Sutter's 2012 talk on "Atomic<> Weapons".
  if (ref_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    delete this;
    return true;
  }
  return false;
}

inline int_fast32_t RefCounted::RefCount() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_18(mht_18_v, 517, "", "./tensorflow/core/platform/refcount.h", "RefCounted::RefCount");

  return ref_.load(std::memory_order_acquire);
}

inline bool RefCounted::RefCountIsOne() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSplatformPSrefcountDTh mht_19(mht_19_v, 524, "", "./tensorflow/core/platform/refcount.h", "RefCounted::RefCountIsOne");

  return (ref_.load(std::memory_order_acquire) == 1);
}

}  // namespace core
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_REFCOUNT_H_
