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
class MHTracer_DTPStensorflowPScorePSkernelsPSmerge_v2_checkpoints_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmerge_v2_checkpoints_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmerge_v2_checkpoints_op_testDTcc() {
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

#include <string>
#include <vector>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace {

void WriteCheckpoint(const string& prefix, gtl::ArraySlice<string> names,
                     gtl::ArraySlice<Tensor> tensors) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSmerge_v2_checkpoints_op_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/kernels/merge_v2_checkpoints_op_test.cc", "WriteCheckpoint");

  BundleWriter writer(Env::Default(), prefix);
  ASSERT_TRUE(names.size() == tensors.size());
  for (size_t i = 0; i < names.size(); ++i) {
    TF_ASSERT_OK(writer.Add(names[i], tensors[i]));
  }
  TF_ASSERT_OK(writer.Finish());
}

template <typename T>
Tensor Constant(T v, TensorShape shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmerge_v2_checkpoints_op_testDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/kernels/merge_v2_checkpoints_op_test.cc", "Constant");

  Tensor ret(DataTypeToEnum<T>::value, shape);
  ret.flat<T>().setConstant(v);
  return ret;
}

class MergeV2CheckpointsOpTest : public OpsTestBase {
 protected:
  void MakeOp(bool delete_old_dirs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmerge_v2_checkpoints_op_testDTcc mht_2(mht_2_v, 232, "", "./tensorflow/core/kernels/merge_v2_checkpoints_op_test.cc", "MakeOp");

    TF_ASSERT_OK(NodeDefBuilder("myop", "MergeV2Checkpoints")
                     .Input(FakeInput())  // checkpoint_prefixes
                     .Input(FakeInput())  // destination_prefix
                     .Attr("delete_old_dirs", delete_old_dirs)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  void RunMergeTest(bool delete_old_dirs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmerge_v2_checkpoints_op_testDTcc mht_3(mht_3_v, 244, "", "./tensorflow/core/kernels/merge_v2_checkpoints_op_test.cc", "RunMergeTest");

    // Writes two checkpoints.
    const std::vector<string> prefixes = {
        io::JoinPath(testing::TmpDir(), "worker0/ckpt0"),
        io::JoinPath(testing::TmpDir(), "worker1/ckpt1"),
        io::JoinPath(testing::TmpDir(), "merged/ckpt") /* merged prefix */};
    // In a different directory, to exercise "delete_old_dirs".
    const string& kMergedPrefix = prefixes[2];

    WriteCheckpoint(prefixes[0], {"tensor0"},
                    {Constant<float>(0, TensorShape({10}))});
    WriteCheckpoint(prefixes[1], {"tensor1", "tensor2"},
                    {Constant<int64_t>(1, TensorShape({1, 16, 18})),
                     Constant<bool>(true, TensorShape({}))});

    // Now merges.
    MakeOp(delete_old_dirs);
    // Add checkpoint_prefixes.
    AddInput<tstring>(TensorShape({2}),
                      [&prefixes](int i) -> tstring { return prefixes[i]; });
    // Add destination_prefix.
    AddInput<tstring>(TensorShape({}), [kMergedPrefix](int unused) -> tstring {
      return kMergedPrefix;
    });
    TF_ASSERT_OK(RunOpKernel());

    // Check that the merged checkpoint file is properly written.
    BundleReader reader(Env::Default(), kMergedPrefix);
    TF_EXPECT_OK(reader.status());

    // We expect to find all saved tensors.
    {
      Tensor val0;
      TF_EXPECT_OK(reader.Lookup("tensor0", &val0));
      test::ExpectTensorEqual<float>(Constant<float>(0, TensorShape({10})),
                                     val0);
    }
    {
      Tensor val1;
      TF_EXPECT_OK(reader.Lookup("tensor1", &val1));
      test::ExpectTensorEqual<int64_t>(
          Constant<int64_t>(1, TensorShape({1, 16, 18})), val1);
    }
    {
      Tensor val2;
      TF_EXPECT_OK(reader.Lookup("tensor2", &val2));
      test::ExpectTensorEqual<bool>(Constant<bool>(true, TensorShape({})),
                                    val2);
    }

    // Exercises "delete_old_dirs".
    for (int i = 0; i < 2; ++i) {
      int directory_found =
          Env::Default()->IsDirectory(string(io::Dirname(prefixes[i]))).code();
      if (delete_old_dirs) {
        EXPECT_EQ(error::NOT_FOUND, directory_found);
      } else {
        EXPECT_EQ(error::OK, directory_found);
      }
    }
  }
};

TEST_F(MergeV2CheckpointsOpTest, MergeNoDelete) {
  RunMergeTest(false /* don't delete old dirs */);
}

TEST_F(MergeV2CheckpointsOpTest, MergeAndDelete) {
  RunMergeTest(true /* delete old dirs */);
}

}  // namespace
}  // namespace tensorflow
