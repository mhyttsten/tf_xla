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
class MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptr_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptr_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptr_testDTcc() {
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

#include "tensorflow/core/platform/intrusive_ptr.h"

#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace core {
namespace {

TEST(IntrusivePtr, ConstructorAddRefFalse) {
  auto ptr = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  // This is needed so that the compiler does not optimize away dead code.
  ASSERT_TRUE(ptr->RefCountIsOne());
  // Test that there is no leak.
}

TEST(IntrusivePtr, ConstructorAddRefTrue) {
  auto raw = new RefCounted();
  auto ptr = IntrusivePtr<RefCounted>(raw, /*add_ref=*/true);
  ASSERT_FALSE(raw->RefCountIsOne());
  raw->Unref();
  ASSERT_TRUE(raw->RefCountIsOne());
}

TEST(IntrusivePtr, CopyConstructor) {
  auto ptr1 = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  auto ptr2 = IntrusivePtr<RefCounted>(ptr1);
  ASSERT_FALSE(ptr2->RefCountIsOne());
}

TEST(IntrusivePtr, CopyAssignment) {
  auto ptr1 = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  auto raw = new RefCounted();
  auto ptr2 = IntrusivePtr<RefCounted>(raw, /*add_ref=*/true);
  ptr2 = ptr1;
  ASSERT_EQ(ptr1.get(), ptr2.get());
  ASSERT_FALSE(ptr2->RefCountIsOne());
  ASSERT_TRUE(raw->RefCountIsOne());
  raw->Unref();
}

TEST(IntrusivePtr, CopyAssignmentIntoEmpty) {
  auto ptr1 = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  auto ptr2 = IntrusivePtr<RefCounted>();
  ptr2 = ptr1;
  ASSERT_FALSE(ptr2->RefCountIsOne());
}

TEST(IntrusivePtr, MoveConstructor) {
  auto ptr1 = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  auto ptr2 = IntrusivePtr<RefCounted>(std::move(ptr1));
  ASSERT_TRUE(ptr2->RefCountIsOne());
  ASSERT_EQ(ptr1.get(), nullptr);  // NOLINT(bugprone-use-after-move)
}

TEST(IntrusivePtr, MoveAssignment) {
  auto ptr1 = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  auto ptr2 = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  ptr2 = std::move(ptr1);
  ASSERT_TRUE(ptr2->RefCountIsOne());
  ASSERT_EQ(ptr1.get(), nullptr);  // NOLINT(bugprone-use-after-move)
}

TEST(IntrusivePtr, MoveAssignmentIntoEmpty) {
  auto ptr1 = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  auto ptr2 = IntrusivePtr<RefCounted>();
  ptr2 = std::move(ptr1);
  ASSERT_TRUE(ptr2->RefCountIsOne());
  ASSERT_EQ(ptr1.get(), nullptr);  // NOLINT(bugprone-use-after-move)
}

TEST(IntrusivePtr, MoveAssignmentAlias) {
  auto ptr = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  auto& ptr_alias = ptr;
  ptr = std::move(ptr_alias);
  ASSERT_TRUE(ptr->RefCountIsOne());
}

TEST(IntrusivePtr, Reset) {
  auto ptr = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  ptr.reset(new RefCounted(), /*add_ref=*/false);
  ASSERT_TRUE(ptr->RefCountIsOne());
  // Test no leak.
}

TEST(IntrusivePtr, ResetIntoEmpty) {
  auto ptr = IntrusivePtr<RefCounted>();
  ptr.reset(new RefCounted(), /*add_ref=*/false);
  ASSERT_TRUE(ptr->RefCountIsOne());
  // Test no leak.
}

TEST(IntrusivePtr, ResetAlias) {
  auto ptr = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  ASSERT_TRUE(ptr->RefCountIsOne());
  ptr.reset(ptr.get(), /*add_ref=*/false);  // No-op.
  ASSERT_TRUE(ptr->RefCountIsOne());
}

TEST(IntrusivePtr, ResetRefBeforeUnref) {
  class Foo : public RefCounted {
   public:
    explicit Foo(char label, Foo* ptr = nullptr)
        : label_(label), ptr_(ptr, false) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("label: '" + std::string(1, label) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptr_testDTcc mht_0(mht_0_v, 289, "", "./tensorflow/core/platform/intrusive_ptr_test.cc", "Foo");
}
    char label_;
    IntrusivePtr<Foo> ptr_;
  };
  IntrusivePtr<Foo> x(new Foo{'a', new Foo{'b', new Foo{'c'}}}, false);
  // This test ensures that reset calls Ref on the new handle before unreffing
  // the current handle to avoid subtle use-after-delete bugs.
  // Here if we were to call Unref first, we will Unref the "Foo" with the
  // label 'b', thereby destroying it.  This will in turn Unref 'c' and destroy
  // that. So reset would try to Ref a deleted object. Calling
  // x->ptr_->ptr_.Ref() before x->ptr_.Unref() avoids this.
  x->ptr_ = x->ptr_->ptr_;
}

TEST(IntrusivePtr, ResetStealPtrBeforeUnref) {
  class Foo : public RefCounted {
   public:
    explicit Foo(char label, Foo* ptr = nullptr)
        : label_(label), ptr_(ptr, false) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("label: '" + std::string(1, label) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPSintrusive_ptr_testDTcc mht_1(mht_1_v, 311, "", "./tensorflow/core/platform/intrusive_ptr_test.cc", "Foo");
}
    char label_;
    IntrusivePtr<Foo> ptr_;
  };
  IntrusivePtr<Foo> x(new Foo{'a', new Foo{'b', new Foo{'c'}}}, false);
  // This test ensures that move assignment clears the handle_ of the moved
  // object before Unreffing the current handle_.
  x->ptr_ = std::move(x->ptr_->ptr_);
}

TEST(IntrusivePtr, Detach) {
  auto ptr = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  ASSERT_TRUE(ptr->RefCountIsOne());
  auto raw = ptr.detach();
  ASSERT_TRUE(raw->RefCountIsOne());
  raw->Unref();
}
}  // namespace
}  // namespace core
}  // namespace tensorflow
