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
class MHTracer_DTPStensorflowPSexamplesPSmultibox_detectorPSmainDTcc {
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
   MHTracer_DTPStensorflowPSexamplesPSmultibox_detectorPSmainDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSexamplesPSmultibox_detectorPSmainDTcc() {
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

#include <setjmp.h>
#include <stdio.h>
#include <string.h>

#include <cmath>
#include <fstream>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using tensorflow::uint8;

// Takes a file name, and loads a list of comma-separated box priors from it,
// one per line, and returns a vector of the values.
Status ReadLocationsFile(const string& file_name, std::vector<float>* result,
                         size_t* found_label_count) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPSexamplesPSmultibox_detectorPSmainDTcc mht_0(mht_0_v, 225, "", "./tensorflow/examples/multibox_detector/main.cc", "ReadLocationsFile");

  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    std::vector<string> string_tokens = tensorflow::str_util::Split(line, ',');
    result->reserve(string_tokens.size());
    for (const string& string_token : string_tokens) {
      float number;
      CHECK(tensorflow::strings::safe_strtof(string_token, &number));
      result->push_back(number);
    }
  }
  *found_label_count = result->size();
  return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPSexamplesPSmultibox_detectorPSmainDTcc mht_1(mht_1_v, 255, "", "./tensorflow/examples/multibox_detector/main.cc", "ReadTensorFromImageFile");

  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string original_name = "identity";
  string output_name = "normalized";
  auto file_reader =
      tensorflow::ops::ReadFile(root.WithOpName(input_name), file_name);
  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  if (tensorflow::str_util::EndsWith(file_name, ".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
    image_reader = DecodeGif(root.WithOpName("gif_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }

  // Also return identity so that we can know the original dimensions and
  // optionally save the image out with bounding boxes overlaid.
  auto original_image = Identity(root.WithOpName(original_name), image_reader);

  // Now cast the image data to float so we can do normal math on it.
  auto float_caster = Cast(root.WithOpName("float_caster"), original_image,
                           tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);

  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
      {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(
      session->Run({}, {output_name, original_name}, {}, out_tensors));
  return Status::OK();
}

Status SaveImage(const Tensor& tensor, const string& file_path) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("file_path: \"" + file_path + "\"");
   MHTracer_DTPStensorflowPSexamplesPSmultibox_detectorPSmainDTcc mht_2(mht_2_v, 316, "", "./tensorflow/examples/multibox_detector/main.cc", "SaveImage");

  LOG(INFO) << "Saving image to " << file_path;
  CHECK(tensorflow::str_util::EndsWith(file_path, ".png"))
      << "Only saving of png files is supported.";

  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string encoder_name = "encode";
  string output_name = "file_writer";

  tensorflow::Output image_encoder =
      EncodePng(root.WithOpName(encoder_name), tensor);
  tensorflow::ops::WriteFile file_saver = tensorflow::ops::WriteFile(
      root.WithOpName(output_name), file_path, image_encoder);

  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  std::vector<Tensor> outputs;
  TF_RETURN_IF_ERROR(session->Run({}, {}, {output_name}, &outputs));

  return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("graph_file_name: \"" + graph_file_name + "\"");
   MHTracer_DTPStensorflowPSexamplesPSmultibox_detectorPSmainDTcc mht_3(mht_3_v, 351, "", "./tensorflow/examples/multibox_detector/main.cc", "LoadGraph");

  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Analyzes the output of the MultiBox graph to retrieve the highest scores and
// their positions in the tensor, which correspond to individual box detections.
Status GetTopDetections(const std::vector<Tensor>& outputs, int how_many_labels,
                        Tensor* indices, Tensor* scores) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSexamplesPSmultibox_detectorPSmainDTcc mht_4(mht_4_v, 373, "", "./tensorflow/examples/multibox_detector/main.cc", "GetTopDetections");

  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "top_k";
  TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}

// Converts an encoded location to an actual box placement with the provided
// box priors.
void DecodeLocation(const float* encoded_location, const float* box_priors,
                    float* decoded_location) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSexamplesPSmultibox_detectorPSmainDTcc mht_5(mht_5_v, 403, "", "./tensorflow/examples/multibox_detector/main.cc", "DecodeLocation");

  bool non_zero = false;
  for (int i = 0; i < 4; ++i) {
    const float curr_encoding = encoded_location[i];
    non_zero = non_zero || curr_encoding != 0.0f;

    const float mean = box_priors[i * 2];
    const float std_dev = box_priors[i * 2 + 1];

    float currentLocation = curr_encoding * std_dev + mean;

    currentLocation = std::max(currentLocation, 0.0f);
    currentLocation = std::min(currentLocation, 1.0f);
    decoded_location[i] = currentLocation;
  }

  if (!non_zero) {
    LOG(WARNING) << "No non-zero encodings; check log for inference errors.";
  }
}

float DecodeScore(float encoded_score) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSexamplesPSmultibox_detectorPSmainDTcc mht_6(mht_6_v, 427, "", "./tensorflow/examples/multibox_detector/main.cc", "DecodeScore");

  return 1 / (1 + std::exp(-encoded_score));
}

void DrawBox(const int image_width, const int image_height, int left, int top,
             int right, int bottom, tensorflow::TTypes<uint8>::Flat* image) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSexamplesPSmultibox_detectorPSmainDTcc mht_7(mht_7_v, 435, "", "./tensorflow/examples/multibox_detector/main.cc", "DrawBox");

  tensorflow::TTypes<uint8>::Flat image_ref = *image;

  top = std::max(0, std::min(image_height - 1, top));
  bottom = std::max(0, std::min(image_height - 1, bottom));

  left = std::max(0, std::min(image_width - 1, left));
  right = std::max(0, std::min(image_width - 1, right));

  for (int i = 0; i < 3; ++i) {
    uint8 val = i == 2 ? 255 : 0;
    for (int x = left; x <= right; ++x) {
      image_ref((top * image_width + x) * 3 + i) = val;
      image_ref((bottom * image_width + x) * 3 + i) = val;
    }
    for (int y = top; y <= bottom; ++y) {
      image_ref((y * image_width + left) * 3 + i) = val;
      image_ref((y * image_width + right) * 3 + i) = val;
    }
  }
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopDetections(const std::vector<Tensor>& outputs,
                          const string& labels_file_name,
                          const int num_boxes,
                          const int num_detections,
                          const string& image_file_name,
                          Tensor* original_tensor) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("labels_file_name: \"" + labels_file_name + "\"");
   mht_8_v.push_back("image_file_name: \"" + image_file_name + "\"");
   MHTracer_DTPStensorflowPSexamplesPSmultibox_detectorPSmainDTcc mht_8(mht_8_v, 469, "", "./tensorflow/examples/multibox_detector/main.cc", "PrintTopDetections");

  std::vector<float> locations;
  size_t label_count;
  Status read_labels_status =
      ReadLocationsFile(labels_file_name, &locations, &label_count);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return read_labels_status;
  }
  CHECK_EQ(label_count, num_boxes * 8);

  const int how_many_labels =
      std::min(num_detections, static_cast<int>(label_count));
  Tensor indices;
  Tensor scores;
  TF_RETURN_IF_ERROR(
      GetTopDetections(outputs, how_many_labels, &indices, &scores));

  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();

  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();

  const Tensor& encoded_locations = outputs[1];
  auto locations_encoded = encoded_locations.flat<float>();

  LOG(INFO) << original_tensor->DebugString();
  const int image_width = original_tensor->shape().dim_size(1);
  const int image_height = original_tensor->shape().dim_size(0);

  tensorflow::TTypes<uint8>::Flat image_flat = original_tensor->flat<uint8>();

  LOG(INFO) << "===== Top " << how_many_labels << " Detections ======";
  for (int pos = 0; pos < how_many_labels; ++pos) {
    const int label_index = indices_flat(pos);
    const float score = scores_flat(pos);

    float decoded_location[4];
    DecodeLocation(&locations_encoded(label_index * 4),
                   &locations[label_index * 8], decoded_location);

    float left = decoded_location[0] * image_width;
    float top = decoded_location[1] * image_height;
    float right = decoded_location[2] * image_width;
    float bottom = decoded_location[3] * image_height;

    LOG(INFO) << "Detection " << pos << ": "
              << "L:" << left << " "
              << "T:" << top << " "
              << "R:" << right << " "
              << "B:" << bottom << " "
              << "(" << label_index << ") score: " << DecodeScore(score);

    DrawBox(image_width, image_height, left, top, right, bottom, &image_flat);
  }

  if (!image_file_name.empty()) {
    return SaveImage(*original_tensor, image_file_name);
  }
  return Status::OK();
}

int main(int argc, char* argv[]) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSexamplesPSmultibox_detectorPSmainDTcc mht_9(mht_9_v, 533, "", "./tensorflow/examples/multibox_detector/main.cc", "main");

  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than multibox_model you'll need to update these.
  string image =
      "tensorflow/examples/multibox_detector/data/surfers.jpg";
  string graph =
      "tensorflow/examples/multibox_detector/data/"
      "multibox_model.pb";
  string box_priors =
      "tensorflow/examples/multibox_detector/data/"
      "multibox_location_priors.txt";
  int32_t input_width = 224;
  int32_t input_height = 224;
  int32_t input_mean = 128;
  int32_t input_std = 128;
  int32_t num_detections = 5;
  int32_t num_boxes = 784;
  string input_layer = "ResizeBilinear";
  string output_location_layer = "output_locations/Reshape";
  string output_score_layer = "output_scores/Reshape";
  string root_dir = "";
  string image_out = "";

  std::vector<Flag> flag_list = {
      Flag("image", &image, "image to be processed"),
      Flag("image_out", &image_out,
           "location to save output image, if desired"),
      Flag("graph", &graph, "graph to be executed"),
      Flag("box_priors", &box_priors, "name of file containing box priors"),
      Flag("input_width", &input_width, "resize image to this width in pixels"),
      Flag("input_height", &input_height,
           "resize image to this height in pixels"),
      Flag("input_mean", &input_mean, "scale pixel values to this mean"),
      Flag("input_std", &input_std, "scale pixel values to this std deviation"),
      Flag("num_detections", &num_detections,
           "number of top detections to return"),
      Flag("num_boxes", &num_boxes,
           "number of boxes defined by the location file"),
      Flag("input_layer", &input_layer, "name of input layer"),
      Flag("output_location_layer", &output_location_layer,
           "name of location output layer"),
      Flag("output_score_layer", &output_score_layer,
           "name of score output layer"),
      Flag("root_dir", &root_dir,
           "interpret image and graph file names relative to this directory"),
  };

  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<Tensor> image_tensors;
  string image_path = tensorflow::io::JoinPath(root_dir, image);

  Status read_tensor_status =
      ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                              input_std, &image_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }
  const Tensor& resized_tensor = image_tensors[0];

  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  Status run_status =
      session->Run({{input_layer, resized_tensor}},
                   {output_score_layer, output_location_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  Status print_status = PrintTopDetections(outputs, box_priors, num_boxes,
                                           num_detections, image_out,
                                           &image_tensors[1]);

  if (!print_status.ok()) {
    LOG(ERROR) << "Running print failed: " << print_status;
    return -1;
  }
  return 0;
}
