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
class MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writer_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writer_testDTcc() {
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
#include "tensorflow/core/summary/summary_db_writer.h"

#include "tensorflow/core/summary/schema.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/db/sqlite.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {
namespace {

Tensor MakeScalarInt64(int64_t x) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writer_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/summary/summary_db_writer_test.cc", "MakeScalarInt64");

  Tensor t(DT_INT64, TensorShape({}));
  t.scalar<int64_t>()() = x;
  return t;
}

class FakeClockEnv : public EnvWrapper {
 public:
  FakeClockEnv() : EnvWrapper(Env::Default()), current_millis_(0) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writer_testDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/summary/summary_db_writer_test.cc", "FakeClockEnv");
}
  void AdvanceByMillis(const uint64 millis) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writer_testDTcc mht_2(mht_2_v, 216, "", "./tensorflow/core/summary/summary_db_writer_test.cc", "AdvanceByMillis");
 current_millis_ += millis; }
  uint64 NowMicros() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writer_testDTcc mht_3(mht_3_v, 220, "", "./tensorflow/core/summary/summary_db_writer_test.cc", "NowMicros");
 return current_millis_ * 1000; }
  uint64 NowSeconds() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writer_testDTcc mht_4(mht_4_v, 224, "", "./tensorflow/core/summary/summary_db_writer_test.cc", "NowSeconds");
 return current_millis_ * 1000; }

 private:
  uint64 current_millis_;
};

class SummaryDbWriterTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writer_testDTcc mht_5(mht_5_v, 235, "", "./tensorflow/core/summary/summary_db_writer_test.cc", "SetUp");

    TF_ASSERT_OK(Sqlite::Open(":memory:", SQLITE_OPEN_READWRITE, &db_));
    TF_ASSERT_OK(SetupTensorboardSqliteDb(db_));
  }

  void TearDown() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writer_testDTcc mht_6(mht_6_v, 243, "", "./tensorflow/core/summary/summary_db_writer_test.cc", "TearDown");

    if (writer_ != nullptr) {
      writer_->Unref();
      writer_ = nullptr;
    }
    db_->Unref();
    db_ = nullptr;
  }

  int64_t QueryInt(const string& sql) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("sql: \"" + sql + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writer_testDTcc mht_7(mht_7_v, 256, "", "./tensorflow/core/summary/summary_db_writer_test.cc", "QueryInt");

    SqliteStatement stmt = db_->PrepareOrDie(sql);
    bool is_done;
    Status s = stmt.Step(&is_done);
    if (!s.ok() || is_done) {
      LOG(ERROR) << s << " due to " << sql;
      return -1;
    }
    return stmt.ColumnInt(0);
  }

  double QueryDouble(const string& sql) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("sql: \"" + sql + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writer_testDTcc mht_8(mht_8_v, 271, "", "./tensorflow/core/summary/summary_db_writer_test.cc", "QueryDouble");

    SqliteStatement stmt = db_->PrepareOrDie(sql);
    bool is_done;
    Status s = stmt.Step(&is_done);
    if (!s.ok() || is_done) {
      LOG(ERROR) << s << " due to " << sql;
      return -1;
    }
    return stmt.ColumnDouble(0);
  }

  string QueryString(const string& sql) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("sql: \"" + sql + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_db_writer_testDTcc mht_9(mht_9_v, 286, "", "./tensorflow/core/summary/summary_db_writer_test.cc", "QueryString");

    SqliteStatement stmt = db_->PrepareOrDie(sql);
    bool is_done;
    Status s = stmt.Step(&is_done);
    if (!s.ok() || is_done) {
      LOG(ERROR) << s << " due to " << sql;
      return "MISSINGNO";
    }
    return stmt.ColumnString(0);
  }

  FakeClockEnv env_;
  Sqlite* db_ = nullptr;
  SummaryWriterInterface* writer_ = nullptr;
};

TEST_F(SummaryDbWriterTest, WriteHistogram_VerifyTensorValues) {
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "histtest", "test1", "user1", &env_,
                                     &writer_));
  int step = 0;
  std::unique_ptr<Event> e{new Event};
  e->set_step(step);
  e->set_wall_time(123);
  Summary::Value* s = e->mutable_summary()->add_value();
  s->set_tag("normal/myhisto");

  double dummy_value = 10.123;
  HistogramProto* proto = s->mutable_histo();
  proto->Clear();
  proto->set_min(dummy_value);
  proto->set_max(dummy_value);
  proto->set_num(dummy_value);
  proto->set_sum(dummy_value);
  proto->set_sum_squares(dummy_value);

  int size = 3;
  double bucket_limits[] = {-30.5, -10.5, -5.5};
  double bucket[] = {-10, 10, 20};
  for (int i = 0; i < size; i++) {
    proto->add_bucket_limit(bucket_limits[i]);
    proto->add_bucket(bucket[i]);
  }
  TF_ASSERT_OK(writer_->WriteEvent(std::move(e)));
  TF_ASSERT_OK(writer_->Flush());
  writer_->Unref();
  writer_ = nullptr;

  // TODO(nickfelt): implement QueryTensor() to encapsulate this
  // Verify the data
  string result = QueryString("SELECT data FROM Tensors");
  const double* val = reinterpret_cast<const double*>(result.data());
  double histarray[] = {std::numeric_limits<double>::min(),
                        -30.5,
                        -10,
                        -30.5,
                        -10.5,
                        10,
                        -10.5,
                        -5.5,
                        20};
  int histarray_size = 9;
  for (int i = 0; i < histarray_size; i++) {
    EXPECT_EQ(histarray[i], val[i]);
  }
}

TEST_F(SummaryDbWriterTest, NothingWritten_NoRowsCreated) {
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "mad-science", "train", "jart", &env_,
                                     &writer_));
  TF_ASSERT_OK(writer_->Flush());
  writer_->Unref();
  writer_ = nullptr;
  EXPECT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Ids"));
  EXPECT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Users"));
  EXPECT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Experiments"));
  EXPECT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Runs"));
  EXPECT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Tags"));
  EXPECT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Tensors"));
}

TEST_F(SummaryDbWriterTest, TensorsWritten_RowsGetInitialized) {
  SummaryMetadata metadata;
  metadata.set_display_name("display_name");
  metadata.set_summary_description("description");
  metadata.mutable_plugin_data()->set_plugin_name("plugin_name");
  metadata.mutable_plugin_data()->set_content("plugin_data");
  SummaryMetadata metadata_nope;
  metadata_nope.set_display_name("nope");
  metadata_nope.set_summary_description("nope");
  metadata_nope.mutable_plugin_data()->set_plugin_name("nope");
  metadata_nope.mutable_plugin_data()->set_content("nope");
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "mad-science", "train", "jart", &env_,
                                     &writer_));
  env_.AdvanceByMillis(23);
  TF_ASSERT_OK(writer_->WriteTensor(1, MakeScalarInt64(123LL), "taggy",
                                    metadata.SerializeAsString()));
  env_.AdvanceByMillis(23);
  TF_ASSERT_OK(writer_->WriteTensor(2, MakeScalarInt64(314LL), "taggy",
                                    metadata_nope.SerializeAsString()));
  TF_ASSERT_OK(writer_->Flush());

  ASSERT_EQ(1LL, QueryInt("SELECT COUNT(*) FROM Users"));
  ASSERT_EQ(1LL, QueryInt("SELECT COUNT(*) FROM Experiments"));
  ASSERT_EQ(1LL, QueryInt("SELECT COUNT(*) FROM Runs"));
  ASSERT_EQ(1LL, QueryInt("SELECT COUNT(*) FROM Tags"));
  ASSERT_EQ(1000LL, QueryInt("SELECT COUNT(*) FROM Tensors"));

  int64_t user_id = QueryInt("SELECT user_id FROM Users");
  int64_t experiment_id = QueryInt("SELECT experiment_id FROM Experiments");
  int64_t run_id = QueryInt("SELECT run_id FROM Runs");
  int64_t tag_id = QueryInt("SELECT tag_id FROM Tags");
  EXPECT_LT(0LL, user_id);
  EXPECT_LT(0LL, experiment_id);
  EXPECT_LT(0LL, run_id);
  EXPECT_LT(0LL, tag_id);

  EXPECT_EQ("jart", QueryString("SELECT user_name FROM Users"));
  EXPECT_EQ(0.023, QueryDouble("SELECT inserted_time FROM Users"));

  EXPECT_EQ(user_id, QueryInt("SELECT user_id FROM Experiments"));
  EXPECT_EQ("mad-science",
            QueryString("SELECT experiment_name FROM Experiments"));
  EXPECT_EQ(0.023, QueryDouble("SELECT inserted_time FROM Experiments"));

  EXPECT_EQ(experiment_id, QueryInt("SELECT experiment_id FROM Runs"));
  EXPECT_EQ("train", QueryString("SELECT run_name FROM Runs"));
  EXPECT_EQ(0.023, QueryDouble("SELECT inserted_time FROM Runs"));

  EXPECT_EQ(run_id, QueryInt("SELECT run_id FROM Tags"));
  EXPECT_EQ("taggy", QueryString("SELECT tag_name FROM Tags"));
  EXPECT_EQ(0.023, QueryDouble("SELECT inserted_time FROM Tags"));

  EXPECT_EQ("display_name", QueryString("SELECT display_name FROM Tags"));
  EXPECT_EQ("plugin_name", QueryString("SELECT plugin_name FROM Tags"));
  EXPECT_EQ("plugin_data", QueryString("SELECT plugin_data FROM Tags"));
  EXPECT_EQ("description", QueryString("SELECT description FROM Descriptions"));

  EXPECT_EQ(tag_id, QueryInt("SELECT series FROM Tensors WHERE step = 1"));
  EXPECT_EQ(0.023,
            QueryDouble("SELECT computed_time FROM Tensors WHERE step = 1"));

  EXPECT_EQ(tag_id, QueryInt("SELECT series FROM Tensors WHERE step = 2"));
  EXPECT_EQ(0.046,
            QueryDouble("SELECT computed_time FROM Tensors WHERE step = 2"));
}

TEST_F(SummaryDbWriterTest, EmptyParentNames_NoParentsCreated) {
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "", "", "", &env_, &writer_));
  TF_ASSERT_OK(writer_->WriteTensor(1, MakeScalarInt64(123LL), "taggy", ""));
  TF_ASSERT_OK(writer_->Flush());
  ASSERT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Users"));
  ASSERT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Experiments"));
  ASSERT_EQ(0LL, QueryInt("SELECT COUNT(*) FROM Runs"));
  ASSERT_EQ(1LL, QueryInt("SELECT COUNT(*) FROM Tags"));
  ASSERT_EQ(1000LL, QueryInt("SELECT COUNT(*) FROM Tensors"));
}

TEST_F(SummaryDbWriterTest, WriteEvent_Scalar) {
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "", "", "", &env_, &writer_));
  std::unique_ptr<Event> e{new Event};
  e->set_step(7);
  e->set_wall_time(123.456);
  Summary::Value* s = e->mutable_summary()->add_value();
  s->set_tag("π");
  s->set_simple_value(3.14f);
  s = e->mutable_summary()->add_value();
  s->set_tag("φ");
  s->set_simple_value(1.61f);
  TF_ASSERT_OK(writer_->WriteEvent(std::move(e)));
  TF_ASSERT_OK(writer_->Flush());
  ASSERT_EQ(2LL, QueryInt("SELECT COUNT(*) FROM Tags"));
  ASSERT_EQ(2000LL, QueryInt("SELECT COUNT(*) FROM Tensors"));
  int64_t tag1_id = QueryInt("SELECT tag_id FROM Tags WHERE tag_name = 'π'");
  int64_t tag2_id = QueryInt("SELECT tag_id FROM Tags WHERE tag_name = 'φ'");
  EXPECT_GT(tag1_id, 0LL);
  EXPECT_GT(tag2_id, 0LL);
  EXPECT_EQ(123.456, QueryDouble(strings::StrCat(
                         "SELECT computed_time FROM Tensors WHERE series = ",
                         tag1_id, " AND step = 7")));
  EXPECT_EQ(123.456, QueryDouble(strings::StrCat(
                         "SELECT computed_time FROM Tensors WHERE series = ",
                         tag2_id, " AND step = 7")));
}

TEST_F(SummaryDbWriterTest, WriteGraph) {
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "", "R", "", &env_, &writer_));
  env_.AdvanceByMillis(23);
  GraphDef graph;
  graph.mutable_library()->add_gradient()->set_function_name("funk");
  NodeDef* node = graph.add_node();
  node->set_name("x");
  node->set_op("Placeholder");
  node = graph.add_node();
  node->set_name("y");
  node->set_op("Placeholder");
  node = graph.add_node();
  node->set_name("z");
  node->set_op("Love");
  node = graph.add_node();
  node->set_name("+");
  node->set_op("Add");
  node->add_input("x");
  node->add_input("y");
  node->add_input("^z");
  node->set_device("tpu/lol");
  std::unique_ptr<Event> e{new Event};
  graph.SerializeToString(e->mutable_graph_def());
  TF_ASSERT_OK(writer_->WriteEvent(std::move(e)));
  TF_ASSERT_OK(writer_->Flush());
  ASSERT_EQ(1LL, QueryInt("SELECT COUNT(*) FROM Runs"));
  ASSERT_EQ(1LL, QueryInt("SELECT COUNT(*) FROM Graphs"));
  ASSERT_EQ(4LL, QueryInt("SELECT COUNT(*) FROM Nodes"));
  ASSERT_EQ(3LL, QueryInt("SELECT COUNT(*) FROM NodeInputs"));

  ASSERT_EQ(QueryInt("SELECT run_id FROM Runs"),
            QueryInt("SELECT run_id FROM Graphs"));

  int64_t graph_id = QueryInt("SELECT graph_id FROM Graphs");
  EXPECT_GT(graph_id, 0LL);
  EXPECT_EQ(0.023, QueryDouble("SELECT inserted_time FROM Graphs"));

  GraphDef graph2;
  graph2.ParseFromString(QueryString("SELECT graph_def FROM Graphs"));
  EXPECT_EQ(0, graph2.node_size());
  EXPECT_EQ("funk", graph2.library().gradient(0).function_name());

  EXPECT_EQ("x", QueryString("SELECT node_name FROM Nodes WHERE node_id = 0"));
  EXPECT_EQ("y", QueryString("SELECT node_name FROM Nodes WHERE node_id = 1"));
  EXPECT_EQ("z", QueryString("SELECT node_name FROM Nodes WHERE node_id = 2"));
  EXPECT_EQ("+", QueryString("SELECT node_name FROM Nodes WHERE node_id = 3"));

  EXPECT_EQ("Placeholder",
            QueryString("SELECT op FROM Nodes WHERE node_id = 0"));
  EXPECT_EQ("Placeholder",
            QueryString("SELECT op FROM Nodes WHERE node_id = 1"));
  EXPECT_EQ("Love", QueryString("SELECT op FROM Nodes WHERE node_id = 2"));
  EXPECT_EQ("Add", QueryString("SELECT op FROM Nodes WHERE node_id = 3"));

  EXPECT_EQ("", QueryString("SELECT device FROM Nodes WHERE node_id = 0"));
  EXPECT_EQ("", QueryString("SELECT device FROM Nodes WHERE node_id = 1"));
  EXPECT_EQ("", QueryString("SELECT device FROM Nodes WHERE node_id = 2"));
  EXPECT_EQ("tpu/lol",
            QueryString("SELECT device FROM Nodes WHERE node_id = 3"));

  EXPECT_EQ(graph_id,
            QueryInt("SELECT graph_id FROM NodeInputs WHERE idx = 0"));
  EXPECT_EQ(graph_id,
            QueryInt("SELECT graph_id FROM NodeInputs WHERE idx = 1"));
  EXPECT_EQ(graph_id,
            QueryInt("SELECT graph_id FROM NodeInputs WHERE idx = 2"));

  EXPECT_EQ(3LL, QueryInt("SELECT node_id FROM NodeInputs WHERE idx = 0"));
  EXPECT_EQ(3LL, QueryInt("SELECT node_id FROM NodeInputs WHERE idx = 1"));
  EXPECT_EQ(3LL, QueryInt("SELECT node_id FROM NodeInputs WHERE idx = 2"));

  EXPECT_EQ(0LL,
            QueryInt("SELECT input_node_id FROM NodeInputs WHERE idx = 0"));
  EXPECT_EQ(1LL,
            QueryInt("SELECT input_node_id FROM NodeInputs WHERE idx = 1"));
  EXPECT_EQ(2LL,
            QueryInt("SELECT input_node_id FROM NodeInputs WHERE idx = 2"));

  EXPECT_EQ(0LL, QueryInt("SELECT is_control FROM NodeInputs WHERE idx = 0"));
  EXPECT_EQ(0LL, QueryInt("SELECT is_control FROM NodeInputs WHERE idx = 1"));
  EXPECT_EQ(1LL, QueryInt("SELECT is_control FROM NodeInputs WHERE idx = 2"));
}

TEST_F(SummaryDbWriterTest, UsesIdsTable) {
  SummaryMetadata metadata;
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "mad-science", "train", "jart", &env_,
                                     &writer_));
  env_.AdvanceByMillis(23);
  TF_ASSERT_OK(writer_->WriteTensor(1, MakeScalarInt64(123LL), "taggy",
                                    metadata.SerializeAsString()));
  TF_ASSERT_OK(writer_->Flush());
  ASSERT_EQ(4LL, QueryInt("SELECT COUNT(*) FROM Ids"));
  EXPECT_EQ(4LL, QueryInt(strings::StrCat(
                     "SELECT COUNT(*) FROM Ids WHERE id IN (",
                     QueryInt("SELECT user_id FROM Users"), ", ",
                     QueryInt("SELECT experiment_id FROM Experiments"), ", ",
                     QueryInt("SELECT run_id FROM Runs"), ", ",
                     QueryInt("SELECT tag_id FROM Tags"), ")")));
}

TEST_F(SummaryDbWriterTest, SetsRunFinishedTime) {
  SummaryMetadata metadata;
  TF_ASSERT_OK(CreateSummaryDbWriter(db_, "mad-science", "train", "jart", &env_,
                                     &writer_));
  env_.AdvanceByMillis(23);
  TF_ASSERT_OK(writer_->WriteTensor(1, MakeScalarInt64(123LL), "taggy",
                                    metadata.SerializeAsString()));
  TF_ASSERT_OK(writer_->Flush());
  ASSERT_EQ(0.023, QueryDouble("SELECT started_time FROM Runs"));
  ASSERT_EQ(0.0, QueryDouble("SELECT finished_time FROM Runs"));
  env_.AdvanceByMillis(23);
  writer_->Unref();
  writer_ = nullptr;
  ASSERT_EQ(0.023, QueryDouble("SELECT started_time FROM Runs"));
  ASSERT_EQ(0.046, QueryDouble("SELECT finished_time FROM Runs"));
}

}  // namespace
}  // namespace tensorflow
