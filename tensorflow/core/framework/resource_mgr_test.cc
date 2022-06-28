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
class MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc() {
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

#include "tensorflow/core/framework/resource_mgr.h"

#include <memory>

#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

class Resource : public ResourceBase {
 public:
  explicit Resource(const string& label) : label_(label) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("label: \"" + label + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/framework/resource_mgr_test.cc", "Resource");
}
  ~Resource() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/framework/resource_mgr_test.cc", "~Resource");
}

  string DebugString() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_2(mht_2_v, 217, "", "./tensorflow/core/framework/resource_mgr_test.cc", "DebugString");
 return strings::StrCat("R/", label_); }

 private:
  string label_;
};

class Other : public ResourceBase {
 public:
  explicit Other(const string& label) : label_(label) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("label: \"" + label + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_3(mht_3_v, 229, "", "./tensorflow/core/framework/resource_mgr_test.cc", "Other");
}
  ~Other() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_4(mht_4_v, 233, "", "./tensorflow/core/framework/resource_mgr_test.cc", "~Other");
}

  string DebugString() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_5(mht_5_v, 238, "", "./tensorflow/core/framework/resource_mgr_test.cc", "DebugString");
 return strings::StrCat("O/", label_); }

 private:
  string label_;
};

template <typename T>
string Find(const ResourceMgr& rm, const string& container,
            const string& name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("container: \"" + container + "\"");
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_6(mht_6_v, 251, "", "./tensorflow/core/framework/resource_mgr_test.cc", "Find");

  T* r;
  TF_CHECK_OK(rm.Lookup(container, name, &r));
  const string ret = r->DebugString();
  r->Unref();
  return ret;
}

template <typename T>
string LookupOrCreate(ResourceMgr* rm, const string& container,
                      const string& name, const string& label) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("container: \"" + container + "\"");
   mht_7_v.push_back("name: \"" + name + "\"");
   mht_7_v.push_back("label: \"" + label + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_7(mht_7_v, 267, "", "./tensorflow/core/framework/resource_mgr_test.cc", "LookupOrCreate");

  T* r;
  TF_CHECK_OK(rm->LookupOrCreate<T>(container, name, &r, [&label](T** ret) {
    *ret = new T(label);
    return Status::OK();
  }));
  const string ret = r->DebugString();
  r->Unref();
  return ret;
}

static void HasError(const Status& s, const error::Code code,
                     const string& substr) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("substr: \"" + substr + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_8(mht_8_v, 283, "", "./tensorflow/core/framework/resource_mgr_test.cc", "HasError");

  EXPECT_EQ(s.code(), code);
  EXPECT_TRUE(absl::StrContains(s.error_message(), substr))
      << s << ", expected substring " << substr;
}

template <typename T>
Status FindErr(const ResourceMgr& rm, const string& container,
               const string& name) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("container: \"" + container + "\"");
   mht_9_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_9(mht_9_v, 296, "", "./tensorflow/core/framework/resource_mgr_test.cc", "FindErr");

  T* r;
  Status s = rm.Lookup(container, name, &r);
  CHECK(!s.ok());
  return s;
}

TEST(ResourceMgrTest, Basic) {
  ResourceMgr rm;
  TF_CHECK_OK(rm.Create("foo", "bar", new Resource("cat")));
  TF_CHECK_OK(rm.Create("foo", "baz", new Resource("dog")));
  TF_CHECK_OK(rm.Create("foo", "bar", new Other("tiger")));

  // Expected to fail.
  HasError(rm.Create("foo", "bar", new Resource("kitty")),
           error::ALREADY_EXISTS, "Resource foo/bar");

  // Expected to be found.
  EXPECT_EQ("R/cat", Find<Resource>(rm, "foo", "bar"));
  EXPECT_EQ("R/dog", Find<Resource>(rm, "foo", "baz"));
  EXPECT_EQ("O/tiger", Find<Other>(rm, "foo", "bar"));

  // Expected to be not found.
  HasError(FindErr<Resource>(rm, "bar", "foo"), error::NOT_FOUND,
           "Container bar");
  HasError(FindErr<Resource>(rm, "foo", "xxx"), error::NOT_FOUND,
           "Resource foo/xxx");
  HasError(FindErr<Other>(rm, "foo", "baz"), error::NOT_FOUND,
           "Resource foo/baz");

  // Delete foo/bar/Resource.
  TF_CHECK_OK(rm.Delete<Resource>("foo", "bar"));
  HasError(FindErr<Resource>(rm, "foo", "bar"), error::NOT_FOUND,
           "Resource foo/bar");
  // Deleting foo/bar/Resource a second time is not OK.
  HasError(rm.Delete<Resource>("foo", "bar"), error::NOT_FOUND,
           "Resource foo/bar");

  TF_CHECK_OK(rm.Create("foo", "bar", new Resource("kitty")));
  EXPECT_EQ("R/kitty", Find<Resource>(rm, "foo", "bar"));

  // Drop the whole container foo.
  TF_CHECK_OK(rm.Cleanup("foo"));
  HasError(FindErr<Resource>(rm, "foo", "bar"), error::NOT_FOUND,
           "Container foo");

  // Dropping it a second time is OK.
  TF_CHECK_OK(rm.Cleanup("foo"));
  HasError(FindErr<Resource>(rm, "foo", "bar"), error::NOT_FOUND,
           "Container foo");

  // Dropping a non-existent container is also ok.
  TF_CHECK_OK(rm.Cleanup("bar"));
}

TEST(ResourceMgrTest, CreateUnowned) {
  core::RefCountPtr<Resource> cat{new Resource("cat")};
  core::RefCountPtr<Resource> kitty{new Resource("kitty")};

  ASSERT_TRUE(cat->RefCountIsOne());
  ASSERT_TRUE(kitty->RefCountIsOne());

  ResourceMgr rm;

  TF_CHECK_OK(rm.CreateUnowned("foo", "bar", cat.get()));
  EXPECT_TRUE(cat->RefCountIsOne());

  // Expected to fail.
  HasError(rm.CreateUnowned("foo", "bar", kitty.get()), error::ALREADY_EXISTS,
           "Resource foo/bar");
  EXPECT_TRUE(kitty->RefCountIsOne());

  // Expected to be found.
  EXPECT_EQ("R/cat", Find<Resource>(rm, "foo", "bar"));

  // Expected to be not found.
  HasError(FindErr<Resource>(rm, "bar", "foo"), error::NOT_FOUND,
           "Container bar");
  HasError(FindErr<Resource>(rm, "foo", "xxx"), error::NOT_FOUND,
           "Resource foo/xxx");

  // Deleting foo/bar/Resource is not OK because it is not owned by the manager.
  HasError(rm.Delete<Resource>("foo", "bar"), error::INTERNAL,
           "Cannot delete an unowned Resource foo/bar");

  TF_CHECK_OK(rm.CreateUnowned("foo", "bar", kitty.get()));
  EXPECT_TRUE(kitty->RefCountIsOne());
  EXPECT_EQ("R/kitty", Find<Resource>(rm, "foo", "bar"));

  {
    core::RefCountPtr<Resource> dog{new Resource("dog")};
    TF_CHECK_OK(rm.CreateUnowned("foo", "bark", dog.get()));
    EXPECT_EQ("R/dog", Find<Resource>(rm, "foo", "bark"));
    EXPECT_EQ(1, dog->WeakRefCount());
    {
      ResourceMgr rm1;
      TF_CHECK_OK(rm1.CreateUnowned("foo", "bark", dog.get()));
      EXPECT_EQ("R/dog", Find<Resource>(rm1, "foo", "bark"));
      EXPECT_EQ(2, dog->WeakRefCount());
    }
    // If manager goes out of scope, the resource loses the weak ref.
    EXPECT_EQ(1, dog->WeakRefCount());
  }
  // If resource goes out of scope, the look up reports not found.
  HasError(FindErr<Resource>(rm, "foo", "bark"), error::NOT_FOUND,
           "Resource foo/bark");

  // Drop the whole container foo.
  TF_CHECK_OK(rm.Cleanup("foo"));
  HasError(FindErr<Resource>(rm, "foo", "bar"), error::NOT_FOUND,
           "Container foo");

  // Dropping it a second time is OK.
  TF_CHECK_OK(rm.Cleanup("foo"));
  HasError(FindErr<Resource>(rm, "foo", "bar"), error::NOT_FOUND,
           "Container foo");

  // Dropping a non-existent container is also ok.
  TF_CHECK_OK(rm.Cleanup("bar"));

  EXPECT_TRUE(cat->RefCountIsOne());
  EXPECT_TRUE(kitty->RefCountIsOne());
}

TEST(ResourceMgrTest, CreateOrLookup) {
  ResourceMgr rm;
  EXPECT_EQ("R/cat", LookupOrCreate<Resource>(&rm, "foo", "bar", "cat"));
  EXPECT_EQ("R/cat", LookupOrCreate<Resource>(&rm, "foo", "bar", "dog"));
  EXPECT_EQ("R/cat", Find<Resource>(rm, "foo", "bar"));

  EXPECT_EQ("O/tiger", LookupOrCreate<Other>(&rm, "foo", "bar", "tiger"));
  EXPECT_EQ("O/tiger", LookupOrCreate<Other>(&rm, "foo", "bar", "lion"));
  TF_CHECK_OK(rm.Delete<Other>("foo", "bar"));
  HasError(FindErr<Other>(rm, "foo", "bar"), error::NOT_FOUND,
           "Resource foo/bar");
}

TEST(ResourceMgrTest, CreateOrLookupRaceCondition) {
  ResourceMgr rm;
  std::atomic<int> atomic_int(0);
  {
    thread::ThreadPool threads(Env::Default(), "racing_creates", 2);
    for (int i = 0; i < 2; i++) {
      threads.Schedule([&rm, &atomic_int] {
        Resource* r;
        TF_CHECK_OK(rm.LookupOrCreate<Resource>(
            "container", "resource-name", &r, [&atomic_int](Resource** ret) {
              // Maximize chance of encountering race condition if one exists.
              Env::Default()->SleepForMicroseconds(1 * 1000 * 1000);
              atomic_int += 1;
              *ret = new Resource("label");
              return Status::OK();
            }));
        r->Unref();
      });
    }
  }
  // Resource creator function should always run exactly once.
  EXPECT_EQ(1, atomic_int);
}

Status ComputePolicy(const string& attr_container,
                     const string& attr_shared_name,
                     bool use_node_name_as_default, string* result) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("attr_container: \"" + attr_container + "\"");
   mht_10_v.push_back("attr_shared_name: \"" + attr_shared_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_10(mht_10_v, 464, "", "./tensorflow/core/framework/resource_mgr_test.cc", "ComputePolicy");

  ContainerInfo cinfo;
  ResourceMgr rmgr;
  NodeDef ndef;
  ndef.set_name("foo");
  if (attr_container != "none") {
    AddNodeAttr("container", attr_container, &ndef);
  }
  if (attr_shared_name != "none") {
    AddNodeAttr("shared_name", attr_shared_name, &ndef);
  }
  TF_RETURN_IF_ERROR(cinfo.Init(&rmgr, ndef, use_node_name_as_default));
  *result = cinfo.DebugString();
  return Status::OK();
}

string Policy(const string& attr_container, const string& attr_shared_name,
              bool use_node_name_as_default) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("attr_container: \"" + attr_container + "\"");
   mht_11_v.push_back("attr_shared_name: \"" + attr_shared_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_11(mht_11_v, 486, "", "./tensorflow/core/framework/resource_mgr_test.cc", "Policy");

  string ret;
  TF_CHECK_OK(ComputePolicy(attr_container, attr_shared_name,
                            use_node_name_as_default, &ret));
  return ret;
}

TEST(ContainerInfo, Basic) {
  // Correct cases.
  EXPECT_TRUE(RE2::FullMatch(Policy("", "", false),
                             "\\[localhost,_\\d+_foo,private\\]"));
  EXPECT_EQ(Policy("", "", true), "[localhost,foo,public]");
  EXPECT_EQ(Policy("", "bar", false), "[localhost,bar,public]");
  EXPECT_EQ(Policy("", "bar", true), "[localhost,bar,public]");
  EXPECT_TRUE(
      RE2::FullMatch(Policy("cat", "", false), "\\[cat,_\\d+_foo,private\\]"));
  EXPECT_EQ(Policy("cat", "", true), "[cat,foo,public]");
  EXPECT_EQ(Policy("cat", "bar", false), "[cat,bar,public]");
  EXPECT_EQ(Policy("cat", "bar", true), "[cat,bar,public]");
  EXPECT_EQ(Policy("cat.0-dog", "bar", true), "[cat.0-dog,bar,public]");
  EXPECT_EQ(Policy(".cat", "bar", true), "[.cat,bar,public]");
}

Status WrongPolicy(const string& attr_container, const string& attr_shared_name,
                   bool use_node_name_as_default) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("attr_container: \"" + attr_container + "\"");
   mht_12_v.push_back("attr_shared_name: \"" + attr_shared_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_12(mht_12_v, 515, "", "./tensorflow/core/framework/resource_mgr_test.cc", "WrongPolicy");

  string dbg;
  auto s = ComputePolicy(attr_container, attr_shared_name,
                         use_node_name_as_default, &dbg);
  CHECK(!s.ok());
  return s;
}

TEST(ContainerInfo, Error) {
  // Missing attribute.
  HasError(WrongPolicy("none", "", false), error::NOT_FOUND, "No attr");
  HasError(WrongPolicy("", "none", false), error::NOT_FOUND, "No attr");
  HasError(WrongPolicy("none", "none", false), error::NOT_FOUND, "No attr");

  // Invalid container.
  HasError(WrongPolicy("12$%", "", false), error::INVALID_ARGUMENT,
           "container contains invalid char");
  HasError(WrongPolicy("-cat", "", false), error::INVALID_ARGUMENT,
           "container contains invalid char");

  // Invalid shared name.
  HasError(WrongPolicy("", "_foo", false), error::INVALID_ARGUMENT,
           "shared_name cannot start with '_'");
}

// Stub DeviceBase subclass which only sets a device name, for testing resource
// handles.
class StubDevice : public DeviceBase {
 public:
  explicit StubDevice(const string& name) : DeviceBase(nullptr) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_13(mht_13_v, 548, "", "./tensorflow/core/framework/resource_mgr_test.cc", "StubDevice");

    attr_.set_name(name);
  }

  Allocator* GetAllocator(AllocatorAttributes) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_14(mht_14_v, 555, "", "./tensorflow/core/framework/resource_mgr_test.cc", "GetAllocator");

    return cpu_allocator();
  }

  const DeviceAttributes& attributes() const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_15(mht_15_v, 562, "", "./tensorflow/core/framework/resource_mgr_test.cc", "attributes");
 return attr_; }
  const string& name() const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_16(mht_16_v, 566, "", "./tensorflow/core/framework/resource_mgr_test.cc", "name");
 return attr_.name(); }

 private:
  DeviceAttributes attr_;
};

// Empty stub resource for testing resource handles.
class StubResource : public ResourceBase {
 public:
  string DebugString() const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_17(mht_17_v, 578, "", "./tensorflow/core/framework/resource_mgr_test.cc", "DebugString");
 return ""; }
  int value_{0};
};

TEST(ResourceHandleTest, CRUD) {
  ResourceMgr resource_mgr("");
  OpKernelContext::Params params;
  params.resource_manager = &resource_mgr;
  StubDevice device("device_name");
  params.device = &device;
  OpKernelContext ctx(&params, 0);

  ResourceHandle p =
      MakeResourceHandle<StubResource>(&ctx, "container", "name");

  {
    auto* r = new StubResource();
    r->value_ = 42;
    TF_EXPECT_OK(CreateResource(&ctx, p, r));
  }
  {
    core::RefCountPtr<StubResource> r;
    TF_ASSERT_OK(LookupResource(&ctx, p, &r));
    ASSERT_TRUE(r != nullptr);
    EXPECT_EQ(r->value_, 42);
  }
  {
    TF_EXPECT_OK(DeleteResource<StubResource>(&ctx, p));
    core::RefCountPtr<StubResource> unused;
    EXPECT_FALSE(LookupResource(&ctx, p, &unused).ok());
  }
}

TEST(ResourceHandleTest, LookupDeleteGenericResource) {
  ResourceMgr resource_mgr("");
  OpKernelContext::Params params;
  params.resource_manager = &resource_mgr;
  StubDevice device("device_name");
  params.device = &device;
  OpKernelContext ctx(&params, 0);

  ResourceHandle p =
      MakeResourceHandle<StubResource>(&ctx, "container", "name");

  {
    auto* r = new StubResource();
    r->value_ = 42;
    TF_EXPECT_OK(CreateResource(&ctx, p, r));
  }
  {
    ResourceBase* r;
    TF_ASSERT_OK(LookupResource(&ctx, p, &r));
    ASSERT_TRUE(r != nullptr);
    core::ScopedUnref unref(r);
    EXPECT_EQ(static_cast<StubResource*>(r)->value_, 42);
  }
  {
    TF_EXPECT_OK(DeleteResource(&ctx, p));
    ResourceBase* unused;
    EXPECT_FALSE(LookupResource(&ctx, p, &unused).ok());
  }
}

TEST(ResourceHandleTest, DifferentDevice) {
  ResourceMgr resource_mgr("");
  OpKernelContext::Params params;
  params.resource_manager = &resource_mgr;
  StubDevice device("device_name");
  params.device = &device;
  OpKernelContext ctx(&params, 0);

  ResourceHandle p =
      MakeResourceHandle<StubResource>(&ctx, "container", "name");

  ResourceMgr other_resource_mgr("");
  OpKernelContext::Params other_params;
  other_params.resource_manager = &other_resource_mgr;
  StubDevice other_device("other_device_name");
  other_params.device = &other_device;
  OpKernelContext other_ctx(&other_params, 0);

  auto* r = new StubResource();
  ASSERT_FALSE(CreateResource(&other_ctx, p, r).ok());
  r->Unref();
}

// Other stub resource to test type-checking of resource handles.
class OtherStubResource : public ResourceBase {
 public:
  string DebugString() const override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_mgr_testDTcc mht_18(mht_18_v, 670, "", "./tensorflow/core/framework/resource_mgr_test.cc", "DebugString");
 return ""; }
};

TEST(ResourceHandleTest, DifferentType) {
  ResourceMgr resource_mgr("");
  OpKernelContext::Params params;
  params.resource_manager = &resource_mgr;
  StubDevice device("device_name");
  params.device = &device;
  OpKernelContext ctx(&params, 0);

  ResourceHandle p =
      MakeResourceHandle<StubResource>(&ctx, "container", "name");

  auto* r = new OtherStubResource;
  ASSERT_FALSE(CreateResource(&ctx, p, r).ok());
  r->Unref();
}

TEST(ResourceHandleTest, DeleteUsingResourceHandle) {
  ResourceMgr resource_mgr("");
  OpKernelContext::Params params;
  params.resource_manager = &resource_mgr;
  StubDevice device("device_name");
  params.device = &device;
  OpKernelContext ctx(&params, 0);

  ResourceHandle p =
      MakeResourceHandle<StubResource>(&ctx, "container", "name");

  StubResource* r = new StubResource;
  TF_EXPECT_OK(CreateResource(&ctx, p, r));

  core::RefCountPtr<StubResource> lookup_r;
  TF_EXPECT_OK(LookupResource<StubResource>(&ctx, p, &lookup_r));
  EXPECT_EQ(lookup_r.get(), r);

  TF_EXPECT_OK(DeleteResource(&ctx, p));
  EXPECT_NE(LookupResource<StubResource>(&ctx, p, &lookup_r).ok(), true);
}

}  // end namespace tensorflow
