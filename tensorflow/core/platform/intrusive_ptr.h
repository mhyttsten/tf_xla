/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PLATFORM_REFCOUNTED_SHARED_PTR_H_
#define TENSORFLOW_CORE_PLATFORM_REFCOUNTED_SHARED_PTR_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptrDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptrDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptrDTh() {
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


#include <algorithm>
namespace tensorflow {
namespace core {

// A utility for managing the lifetime of ref-counted objects.
//
// Generally used for objects that derive from `tensorflow::RefCounted`.
template <class T>
class IntrusivePtr {
 public:
  // add_ref=false indicates that IntrusivePtr owns the underlying pointer.
  //
  // In most cases, we expect this to be called with add_ref=false, except in
  // special circumstances where the lifetime of the underlying RefCounted
  // object needs to be externally managed.
  IntrusivePtr(T* h, bool add_ref) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptrDTh mht_0(mht_0_v, 202, "", "./tensorflow/core/platform/intrusive_ptr.h", "IntrusivePtr");
 reset(h, add_ref); }
  IntrusivePtr(const IntrusivePtr& o) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptrDTh mht_1(mht_1_v, 206, "", "./tensorflow/core/platform/intrusive_ptr.h", "IntrusivePtr");
 reset(o.handle_, /*add_ref=*/true); }
  IntrusivePtr(IntrusivePtr&& o) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptrDTh mht_2(mht_2_v, 210, "", "./tensorflow/core/platform/intrusive_ptr.h", "IntrusivePtr");
 *this = std::move(o); }
  IntrusivePtr() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptrDTh mht_3(mht_3_v, 214, "", "./tensorflow/core/platform/intrusive_ptr.h", "IntrusivePtr");
}
  void reset(T* h, bool add_ref) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptrDTh mht_4(mht_4_v, 218, "", "./tensorflow/core/platform/intrusive_ptr.h", "reset");

    if (h != handle_) {
      if (add_ref && h) h->Ref();
      if (handle_) handle_->Unref();
      handle_ = h;
    }
  }
  IntrusivePtr& operator=(const IntrusivePtr& o) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptrDTh mht_5(mht_5_v, 228, "", "./tensorflow/core/platform/intrusive_ptr.h", "=");

    reset(o.handle_, /*add_ref=*/true);
    return *this;
  }
  IntrusivePtr& operator=(IntrusivePtr&& o) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptrDTh mht_6(mht_6_v, 235, "", "./tensorflow/core/platform/intrusive_ptr.h", "=");

    if (handle_ != o.handle_) {
      // Must clear o.handle_ before calling reset to capture the case where
      // handle_->member == o. In this case, calling handle_->Unref first would
      // delete o.handle_ so we clear it out first.
      reset(o.detach(), /*add_ref=*/false);
    }
    return *this;
  }
  bool operator==(const IntrusivePtr& o) const { return handle_ == o.handle_; }
  T* operator->() const { return handle_; }
  T& operator*() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptrDTh mht_7(mht_7_v, 249, "", "./tensorflow/core/platform/intrusive_ptr.h", "*");
 return *handle_; }
  explicit operator bool() const noexcept { return get(); }
  T* get() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptrDTh mht_8(mht_8_v, 254, "", "./tensorflow/core/platform/intrusive_ptr.h", "get");
 return handle_; }
  // Releases ownership of the pointer without unreffing. Caller is responsible
  // for calling Unref on the returned pointer.
  T* detach() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptrDTh mht_9(mht_9_v, 260, "", "./tensorflow/core/platform/intrusive_ptr.h", "detach");

    T* handle = handle_;
    handle_ = nullptr;
    return handle;
  }

  ~IntrusivePtr() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptrDTh mht_10(mht_10_v, 269, "", "./tensorflow/core/platform/intrusive_ptr.h", "~IntrusivePtr");

    if (handle_) handle_->Unref();
  }

 private:
  T* handle_ = nullptr;
};

}  // namespace core
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_REFCOUNTED_SHARED_PTR_H_
