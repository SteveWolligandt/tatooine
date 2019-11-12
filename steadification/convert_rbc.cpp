#include "datasets.h"

struct RBC_Converter : RBC {
  void calc() {
    std::array<double, dim[0] * dim[1]> data_x, data_z;
    std::array<double, 2 * dim[0] * dim[1]> merged_transposed_data;
    for (size_t ti = 0; ti < dim[2]; ++ti) {
      const auto t = this->sampler().dimension(2)[ti];
      const std::string x_file =
        abs_path + to_string(t) + "/" + x_comp_name;
      const std::string z_file =
        abs_path + to_string(t) + "/" + z_comp_name;

      read_hdf5_file(data_x, x_file, x_dataset_name);
      std::cout << "data_x = ["<<data_x[0]<<", "<<data_x[1]<<", "<<data_x[2]<<", ...]\n";
      read_hdf5_file(data_z, z_file, z_dataset_name);
      std::cout << "data_z = ["<<data_z[0]<<", "<<data_z[1]<<", "<<data_z[2]<<", ...]\n";
      size_t cnt = 0;
      for (size_t i = 0; i < dim[0]; ++i)
        for (size_t j = 0; j < dim[1]; ++j) {
          size_t idx = (i + j * dim[0]) * 2;
          merged_transposed_data[idx + 0] = data_x[cnt];
          merged_transposed_data[idx + 1] = data_z[cnt];
          ++cnt;
        }
      std::cout << "merged_transposed_data = [" <<
                merged_transposed_data[0] << ", " <<
                merged_transposed_data[1] << ", " <<
                merged_transposed_data[2] << ", " <<
                merged_transposed_data[3] << ", " <<
                merged_transposed_data[4] << ", " <<
                merged_transposed_data[5] << ", ...]\n";
      std::stringstream strstr;
      strstr << "rbc/rbc_" << t << ".bin";
      std::cout << "writing " << strstr.str();
      std::ofstream file{strstr.str(), std::ofstream::binary};
      if (file.is_open()) {
        file.write((char*)merged_transposed_data.data(),
                   sizeof(double) * dim[0] * dim[1] * 2);
        file.close();
      }
    }

  }
};

int main () {
  RBC_Converter conv; conv.calc();
}
