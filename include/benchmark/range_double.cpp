#include "../stx/btree.h"
#include "../alex/alex.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include "flags.h"
#include "utils.h"

#define KEY_TYPE double
#define PAYLOAD_TYPE double
#define UNIT 0.0000001f

int main(int argc, char *argv[])
{
      auto flags = parse_flags(argc, argv);
      std::string keys_file_path = get_required(flags, "keys_file");
      std::string keys_file_type = get_required(flags, "keys_file_type");
      auto init_num_keys = stoi(get_required(flags, "init_num_keys"));
      auto batch_size = stoi(get_required(flags, "batch_size"));
      std::string lookup_distribution =
          get_with_default(flags, "lookup_distribution", "uniform");

      // Read keys from file
      auto keys = new KEY_TYPE[init_num_keys];
      if (keys_file_type == "binary")
      {
            load_binary_data(keys, init_num_keys, keys_file_path);
      }
      else if (keys_file_type == "text")
      {
            load_text_data(keys, init_num_keys, keys_file_path);
      }
      else
      {
            std::cerr << "--keys_file_type must be either 'binary' or 'text'"
                      << std::endl;
            return 1;
      }

      std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> pairs(init_num_keys);
      auto values = new std::pair<KEY_TYPE, PAYLOAD_TYPE>[init_num_keys];
      std::mt19937_64 gen_payload(std::random_device{}());
      for (int i = 0; i < init_num_keys; i++)
      {
            pairs[i].first = keys[i];
            pairs[i].second = static_cast<PAYLOAD_TYPE>(gen_payload());
            values[i].first = keys[i];
            values[i].second = static_cast<PAYLOAD_TYPE>(gen_payload());
      }
      std::sort(pairs.begin(), pairs.end(), [](auto const &a, auto const &b) { return a.first < b.first; }); 
      std::sort(values, values + init_num_keys, [](auto const &a, auto const &b) { return a.first < b.first; }); 

      // Run workload
      std::cout << std::fixed; // scientific
      std::cout << std::setprecision(6);

      KEY_TYPE *lookup_keys = nullptr;
      if (lookup_distribution == "uniform")
      {
            lookup_keys = get_search_keys(keys, init_num_keys, batch_size);
      }
      else if (lookup_distribution == "zipf")
      {
            lookup_keys = get_search_keys_zipf(keys, init_num_keys, batch_size);
      }
      else
      {
            std::cerr << "--lookup_distribution must be either 'uniform' or 'zipf'"
                      << std::endl;
            return 1;
      }

      const std::vector<KEY_TYPE> num_search_keys = {
            UNIT*1, UNIT*2, UNIT*4, UNIT*8
            , UNIT*16, UNIT*32, UNIT*64, UNIT*128
            , UNIT*256,UNIT*512, UNIT*1024, UNIT*1024*2
            , UNIT*1024*4, UNIT*1024*8, UNIT*1024*16, UNIT*1024*32
            };

      // Create STX B+ Tree and bulk load
      stx::btree<KEY_TYPE, PAYLOAD_TYPE> index_bplus;
      index_bplus.bulk_load(pairs.begin(), pairs.end());   
      index_bplus.print_stat();
      index_bplus.print_root_depth();

      // Create ALEX and bulk load
      alex::Alex<KEY_TYPE, PAYLOAD_TYPE> index_alex;
      index_alex.bulk_load(values, init_num_keys);
      index_alex.print_stat();
      index_alex.print_all_data_nodes();

      for(int k=0; k<(int)num_search_keys.size(); k++)
      {
          std::cout << "==== range query time for "
                    << num_search_keys[k]
                    << " keys"
                    << std::endl;
          auto lookups_start_time =std::chrono::high_resolution_clock::now();
          std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> v_res;
          int sum_cardinality = 0;

          for (int i = 0; i < batch_size; i++)
          {
                KEY_TYPE start_key = lookup_keys[i];
                KEY_TYPE end_key = start_key + num_search_keys[k] - UNIT;
            //     std::cout << start_key << " - " << end_key << std::endl;
                v_res = index_bplus.query_range(start_key, end_key);
                sum_cardinality += v_res.size();
          }
          auto lookups_end_time = std::chrono::high_resolution_clock::now();
          double batch_lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(lookups_end_time - lookups_start_time)
                        .count();
          std::cout << "B+:   \t"
                    << (batch_lookup_time / batch_size)
                    << " ns/query with "
                    << (sum_cardinality / batch_size)
                    << " cardinality"
                    << std::endl;
	      
          lookups_start_time =std::chrono::high_resolution_clock::now();
          v_res.clear();
          sum_cardinality = 0;
          for (int j = 0; j < batch_size; j++)
          {
                KEY_TYPE start_key = lookup_keys[j];
                KEY_TYPE end_key = start_key + num_search_keys[k] - UNIT;
            //     std::cout << start_key << " - " << end_key << std::endl;
                v_res = index_alex.query_range(start_key, end_key);
                sum_cardinality += v_res.size();
          }
          lookups_end_time = std::chrono::high_resolution_clock::now();
          batch_lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(lookups_end_time - lookups_start_time)
                        .count();
          std::cout << "ALEX:\t"
                    << (batch_lookup_time / batch_size)
                    << " ns/query with "
                    << (sum_cardinality / batch_size)
                    << " cardinality"
                    << std::endl;
      }


      

      delete[] lookup_keys;
      delete[] keys;
      pairs.clear();

      return 0;
}
