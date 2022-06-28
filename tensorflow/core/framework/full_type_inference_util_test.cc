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
class MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_util_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_util_testDTcc() {
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

#include <functional>
#include <string>

#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace full_type {

namespace {

TEST(ReplicateInput, Default) {
  FullTypeDef t;
  t.set_type_id(TFT_ARRAY);

  const auto ret = ReplicateInput()({t}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
}

TEST(ReplicateInput, Duplicate) {
  FullTypeDef t;
  t.set_type_id(TFT_ARRAY);

  const auto ret = ReplicateInput(0, 2)({t}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 2);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  EXPECT_EQ(rt.args(1).type_id(), TFT_ARRAY);
}

TEST(ReplicateInput, FirstOfMultipleArgs) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  FullTypeDef t2;
  t2.set_type_id(TFT_TENSOR);

  const auto ret = ReplicateInput(0, 2)({t1, t2}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 2);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  EXPECT_EQ(rt.args(1).type_id(), TFT_ARRAY);
}

TEST(ReplicateInput, SecondOfMultipleArgs) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  FullTypeDef t2;
  t2.set_type_id(TFT_TENSOR);

  const auto ret = ReplicateInput(1, 2)({t1, t2}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 2);
  EXPECT_EQ(rt.args(0).type_id(), TFT_TENSOR);
  EXPECT_EQ(rt.args(1).type_id(), TFT_TENSOR);
}

TEST(ReplicateInput, Unset) {
  FullTypeDef t;

  const auto ret = ReplicateInput()({t}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_UNSET);
}

TEST(Merge, Single) {
  FullTypeDef t;
  t.set_type_id(TFT_ARRAY);

  const auto ret = Merge()({t}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
}

TEST(Merge, Double) {
  FullTypeDef t;
  t.set_type_id(TFT_ARRAY);

  const auto ret = Merge()({t, t}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
}

TEST(Merge, Unset) {
  FullTypeDef t;
  t.set_type_id(TFT_UNSET);

  const auto ret = Merge()({t}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_UNSET);
}

TEST(Merge, UnsetComponents) {
  FullTypeDef t1;
  FullTypeDef t2;

  const auto ret = Merge()({t1, t2}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_UNSET);
}

void ExpectInferredArrayOfTensor(StatusOr<FullTypeDef> ret) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_inference_util_testDTcc mht_0(mht_0_v, 316, "", "./tensorflow/core/framework/full_type_inference_util_test.cc", "ExpectInferredArrayOfTensor");

  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  EXPECT_EQ(rt.args(0).args(0).type_id(), TFT_TENSOR);
}

TEST(Merge, RejectsMismatched) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  FullTypeDef t2;
  t2.set_type_id(TFT_TENSOR);

  const auto ret = Merge()({t1, t2}, {});
  EXPECT_THAT(ret.status().error_message(),
              ::testing::HasSubstr("expected compatible input types"));
}

TEST(Merge, UsesPartialInfo) {
  FullTypeDef t1;
  FullTypeDef t2;
  t2.set_type_id(TFT_ARRAY);
  t2.add_args()->set_type_id(TFT_TENSOR);

  ExpectInferredArrayOfTensor(Merge()({t1, t2}, {}));
  ExpectInferredArrayOfTensor(Merge()({t2, t1}, {}));
}

TEST(Merge, SelectsMostSpecificOfSubtypes) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  t1.add_args()->set_type_id(TFT_ANY);
  FullTypeDef t2;
  t2.set_type_id(TFT_ARRAY);
  t2.add_args()->set_type_id(TFT_TENSOR);

  ExpectInferredArrayOfTensor(Merge()({t1, t2}, {}));
  ExpectInferredArrayOfTensor(Merge()({t2, t1}, {}));
}

TEST(UnaryContainerCreate, Basic) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ANY);
  FullTypeDef t2;
  t2.set_type_id(TFT_TENSOR);
  FullTypeDef t3;
  t3.set_type_id(TFT_ANY);

  const auto ret = UnaryContainerCreate(TFT_ARRAY, 1)({t1, t2, t3}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  EXPECT_EQ(rt.args(0).args(0).type_id(), TFT_TENSOR);
}

TEST(UnaryContainerAdd, Basic) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ANY);
  FullTypeDef t2;
  t2.set_type_id(TFT_TENSOR);
  FullTypeDef t3;
  t3.set_type_id(TFT_ARRAY);

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/2, /*element_idx=*/1,
                        /*homogeneous=*/false)({t1, t2, t3}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  EXPECT_EQ(rt.args(0).args(0).type_id(), TFT_TENSOR);
}

TEST(UnaryContainerAdd, RejectsMismatchedContainerType) {
  FullTypeDef t1;
  t1.set_type_id(TFT_TENSOR);
  FullTypeDef t2;
  t2.set_type_id(TFT_DATASET);

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/1, /*element_idx=*/0,
                        /*homogeneous=*/false)({t1, t2}, {});
  EXPECT_THAT(ret.status().error_message(),
              ::testing::HasSubstr("expected container type"));
}

TEST(UnaryContainerAdd, IgnoresUnsetContainerType) {
  FullTypeDef t1;
  t1.set_type_id(TFT_TENSOR);
  FullTypeDef t2;

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/1, /*element_idx=*/0,
                        /*homogeneous=*/false)({t1, t2}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  EXPECT_EQ(rt.args(0).args(0).type_id(), TFT_TENSOR);
}

TEST(UnaryContainerAdd, UnsetElementTypeRemainsUnset) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ANY);
  FullTypeDef t2;
  FullTypeDef t3;
  t3.set_type_id(TFT_ARRAY);

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/2, /*element_idx=*/1,
                        /*homogeneous=*/false)({t1, t2, t3}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  ASSERT_EQ(rt.args(0).args_size(), 0);
}

TEST(UnaryContainerAdd, UnsetElementTypeKeepsOriginalElementType) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  t1.add_args()->set_type_id(TFT_TENSOR);
  FullTypeDef t2;

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/0, /*element_idx=*/1,
                        /*homogeneous=*/false)({t1, t2}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  EXPECT_EQ(rt.args(0).args(0).type_id(), TFT_TENSOR);
}

TEST(UnaryContainerAdd, KeepsContainerTypeIfElementIsSubtype) {
  // TODO(mdan): We may want to refine the type if homogeneous.
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  t1.add_args()->set_type_id(TFT_ANY);
  FullTypeDef t2;
  t2.set_type_id(TFT_TENSOR);

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/0, /*element_idx=*/1,
                        /*homogeneous=*/true)({t1, t2}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  EXPECT_EQ(rt.args(0).args(0).type_id(), TFT_ANY);
}

TEST(UnaryContainerAdd, RejectsMismatchedElementTypesHeterogenous) {
  // TODO(mdan): Implement if needed (see full_type_inference_util.cc).
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  t1.add_args()->set_type_id(TFT_TENSOR);
  FullTypeDef t2;
  t2.set_type_id(TFT_DATASET);

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/0, /*element_idx=*/1,
                        /*homogeneous=*/false)({t1, t2}, {});
  EXPECT_THAT(ret.status().error_message(),
              ::testing::HasSubstr("need union types"));
}

TEST(UnaryContainerAdd, RejectsMismatchedElementTypesHomogeneous) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  t1.add_args()->set_type_id(TFT_TENSOR);
  FullTypeDef t2;
  t2.set_type_id(TFT_DATASET);

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/0, /*element_idx=*/1,
                        /*homogeneous=*/true)({t1, t2}, {});
  EXPECT_THAT(ret.status().error_message(),
              ::testing::HasSubstr("expected a subtype"));
}

TEST(UnaryContainerAdd, RejectsSupertypeElementTypeHeterogeneous) {
  // TODO(mdan): Implement if needed (see full_type_inference_util.cc).
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  t1.add_args()->set_type_id(TFT_TENSOR);
  FullTypeDef t2;
  t2.set_type_id(TFT_ANY);

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/0, /*element_idx=*/1,
                        /*homogeneous=*/false)({t1, t2}, {});
  EXPECT_THAT(ret.status().error_message(),
              ::testing::HasSubstr("need union types"));
}

TEST(UnaryContainerAdd, RejectsSupertypeElementTypeHomogeneous) {
  // TODO(mdan): This might be acceptable.
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  t1.add_args()->set_type_id(TFT_TENSOR);
  FullTypeDef t2;
  t2.set_type_id(TFT_ANY);

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/0, /*element_idx=*/1,
                        /*homogeneous=*/true)({t1, t2}, {});
  EXPECT_THAT(ret.status().error_message(),
              ::testing::HasSubstr("expected a subtype"));
}

TEST(MultiaryUnstack, One) {
  FullTypeDef t1;
  t1.set_type_id(TFT_TENSOR);

  const auto ret = MultiaryUnstack(TFT_DATASET, UnstackTensor)({t1}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_DATASET);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args(0).args(0).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).args(0).type_id(), TFT_TENSOR);
}

TEST(MultiaryUnstack, Three) {
  FullTypeDef t1;
  t1.set_type_id(TFT_RAGGED);
  t1.add_args()->set_type_id(TFT_STRING);
  FullTypeDef t2;
  t2.set_type_id(TFT_TENSOR);
  FullTypeDef t3;
  t3.set_type_id(TFT_RAGGED);
  t3.add_args()->set_type_id(TFT_INT64);

  const auto ret =
      MultiaryUnstack(TFT_DATASET, UnstackTensor)({t1, t2, t3}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_DATASET);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args(0).args(0).args_size(), 3);
  ASSERT_EQ(rt.args(0).args(0).args(0).type_id(), TFT_RAGGED);
  ASSERT_EQ(rt.args(0).args(0).args(0).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).args(0).args(0).type_id(), TFT_STRING);
  ASSERT_EQ(rt.args(0).args(0).args(1).type_id(), TFT_TENSOR);
  ASSERT_EQ(rt.args(0).args(0).args(2).type_id(), TFT_RAGGED);
  ASSERT_EQ(rt.args(0).args(0).args(2).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).args(2).args(0).type_id(), TFT_INT64);
}

TEST(MapContainer, One) {
  FullTypeDef cont_t;
  cont_t.set_type_id(TFT_DATASET);
  FullTypeDef* el_t = cont_t.add_args();
  el_t->set_type_id(TFT_PRODUCT);
  (el_t->add_args())->set_type_id(TFT_TENSOR);

  const auto ret = ContainerMap(TFT_DATASET, 0, BatchTensor)({cont_t}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_DATASET);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args(0).args(0).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).args(0).type_id(), TFT_TENSOR);
}

TEST(MapContainer, Three) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ANY);
  FullTypeDef cont_t;
  cont_t.set_type_id(TFT_DATASET);
  FullTypeDef* el_t = cont_t.add_args();
  el_t->set_type_id(TFT_PRODUCT);
  FullTypeDef* e1 = el_t->add_args();
  e1->set_type_id(TFT_RAGGED);
  e1->add_args()->set_type_id(TFT_STRING);
  FullTypeDef* e2 = el_t->add_args();
  e2->set_type_id(TFT_TENSOR);
  FullTypeDef* e3 = el_t->add_args();
  e3->set_type_id(TFT_RAGGED);
  e3->add_args()->set_type_id(TFT_INT64);
  FullTypeDef t3;
  t3.set_type_id(TFT_ANY);

  const auto ret =
      ContainerMap(TFT_DATASET, 1, BatchTensor)({t1, cont_t, t3}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_DATASET);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args(0).args(0).args_size(), 3);
  ASSERT_EQ(rt.args(0).args(0).args(0).type_id(), TFT_RAGGED);
  ASSERT_EQ(rt.args(0).args(0).args(0).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).args(0).args(0).type_id(), TFT_STRING);
  ASSERT_EQ(rt.args(0).args(0).args(1).type_id(), TFT_TENSOR);
  ASSERT_EQ(rt.args(0).args(0).args(2).type_id(), TFT_RAGGED);
  ASSERT_EQ(rt.args(0).args(0).args(2).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).args(2).args(0).type_id(), TFT_INT64);
}

}  // namespace

}  // namespace full_type

}  // namespace tensorflow
