#include "lm/builder/pipeline.hh"
#include "util/exception.hh"
#include "util/file.hh"
#include "util/file_piece.hh"
#include "util/usage.hh"

#include <iostream>

#include <boost/program_options.hpp>
#include <boost/version.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>

namespace {
class SizeNotify {
  public:
    SizeNotify(std::size_t &out) : behind_(out) {}

    void operator()(const std::string &from) {
      behind_ = util::ParseSize(from);
    }

  private:
    std::size_t &behind_;
};

boost::program_options::typed_value<std::string> *SizeOption(std::size_t &to, const char *default_value) {
  return boost::program_options::value<std::string>()->notifier(SizeNotify(to))->default_value(default_value);
}

} // namespace

std::vector<uint64_t> parse_pruning_parameter(std::string param, std::size_t order);

int main(int argc, char *argv[]) {
  try {
    namespace po = boost::program_options;
    po::options_description options("Language model building options");
    lm::builder::PipelineConfig pipeline;

    std::string text, arpa, pruning_param;

    options.add_options()
      ("order,o", po::value<std::size_t>(&pipeline.order)
#if BOOST_VERSION >= 104200
         ->required()
#endif
         , "Order of the model")
      ("interpolate_unigrams", po::bool_switch(&pipeline.initial_probs.interpolate_unigrams), "Interpolate the unigrams (default: emulate SRILM by not interpolating)")
      ("temp_prefix,T", po::value<std::string>(&pipeline.sort.temp_prefix)->default_value("/tmp/lm"), "Temporary file prefix")
      ("memory,S", SizeOption(pipeline.sort.total_memory, util::GuessPhysicalMemory() ? "80%" : "1G"), "Sorting memory")
      ("minimum_block", SizeOption(pipeline.minimum_block, "8K"), "Minimum block size to allow")
      ("sort_block", SizeOption(pipeline.sort.buffer_size, "64M"), "Size of IO operations for sort (determines arity)")
      ("vocab_estimate", po::value<lm::WordIndex>(&pipeline.vocab_estimate)->default_value(1000000), "Assume this vocabulary size for purposes of calculating memory in step 1 (corpus count) and pre-sizing the hash table")
      ("block_count", po::value<std::size_t>(&pipeline.block_count)->default_value(2), "Block count (per order)")
      ("vocab_file", po::value<std::string>(&pipeline.vocab_file)->default_value(""), "Location to write vocabulary file")
      ("verbose_header", po::bool_switch(&pipeline.verbose_header), "Add a verbose header to the ARPA file that includes information such as token count, smoothing type, etc.")
      ("text", po::value<std::string>(&text), "Read text from a file instead of stdin")
      ("arpa", po::value<std::string>(&arpa), "Write ARPA to a file instead of stdout")
      ("counts_threshold,P", po::value<std::string>(&pruning_param), "Prune n-grams of count equal to or lower than threshold. Specifie threshold for each n-gram order. For example: -P \"0 5 100\" or -P 0,5,100 for order 3 model. 0 means no pruning, default behavior.");
    if (argc == 1) {
      std::cerr << 
        "Builds unpruned language models with modified Kneser-Ney smoothing.\n\n"
        "Please cite:\n"
        "@inproceedings{Heafield-estimate,\n"
        "  author = {Kenneth Heafield and Ivan Pouzyrevsky and Jonathan H. Clark and Philipp Koehn},\n"
        "  title = {Scalable Modified {Kneser-Ney} Language Model Estimation},\n"
        "  year = {2013},\n"
        "  month = {8},\n"
        "  booktitle = {Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics},\n"
        "  address = {Sofia, Bulgaria},\n"
        "  url = {http://kheafield.com/professional/edinburgh/estimate\\_paper.pdf},\n"
        "}\n\n"
        "Provide the corpus on stdin.  The ARPA file will be written to stdout.  Order of\n"
        "the model (-o) is the only mandatory option.  As this is an on-disk program,\n"
        "setting the temporary file location (-T) and sorting memory (-S) is recommended.\n\n"
        "Memory sizes are specified like GNU sort: a number followed by a unit character.\n"
        "Valid units are \% for percentage of memory (supported platforms only) and (in\n"
        "increasing powers of 1024): b, K, M, G, T, P, E, Z, Y.  Default is K (*1024).\n\n";
      std::cerr << options << std::endl;
      return 1;
    }
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);

    // required() appeared in Boost 1.42.0.
#if BOOST_VERSION < 104200
    if (!vm.count("order")) {
      std::cerr << "the option '--order' is required but missing" << std::endl;
      return 1;
    }
#endif

    // parse pruning thresholds
    pipeline.counts_threshold = parse_pruning_parameter(pruning_param, pipeline.order);

    util::NormalizeTempPrefix(pipeline.sort.temp_prefix);

    lm::builder::InitialProbabilitiesConfig &initial = pipeline.initial_probs;
    // TODO: evaluate options for these.  
    initial.adder_in.total_memory = 32768;
    initial.adder_in.block_count = 2;
    initial.adder_out.total_memory = 32768;
    initial.adder_out.block_count = 2;
    pipeline.read_backoffs = initial.adder_out;

    util::scoped_fd in(0), out(1);
    if (vm.count("text")) {
      in.reset(util::OpenReadOrThrow(text.c_str()));
    }
    if (vm.count("arpa")) {
      out.reset(util::CreateOrThrow(arpa.c_str()));
    }

    // Read from stdin
    try {
      lm::builder::Pipeline(pipeline, in.release(), out.release());
    } catch (const util::MallocException &e) {
      std::cerr << e.what() << std::endl;
      std::cerr << "Try rerunning with a more conservative -S setting than " << vm["memory"].as<std::string>() << std::endl;
      return 1;
    }
    util::PrintUsage(std::cerr);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}

// parse and validate pruning thresholds and return vector of threshold counts for each n-grams order
// return vector of zeroes if param is empty
std::vector<uint64_t> parse_pruning_parameter(std::string param, std::size_t order) {
    // split threshold counts "0,1,2 3" -> [0,1,2,3]
    std::vector<std::string> pruning_thresholds;
    boost::split(pruning_thresholds, param, boost::is_any_of(", "), boost::token_compress_on);
    // convert to vector of integers
    std::vector<uint64_t> counts_threshold;
    counts_threshold.reserve(pruning_thresholds.size());
    for (std::vector<std::string>::iterator it = pruning_thresholds.begin(); it != pruning_thresholds.end(); ++it)
        try {
            uint64_t count = boost::lexical_cast<uint64_t>( it->c_str() );
            counts_threshold.push_back(count);
        } catch( boost::bad_lexical_cast const& ) {
            UTIL_THROW(util::Exception, "Pruning counts threshold has to be nonnegative numbers");
        }
    
    // validate pruning threshold if specified
    if (counts_threshold.size() > 0) {
        // throw if each n-gram order has not  threshold specified
        UTIL_THROW_IF(counts_threshold.size() != order, util::Exception, 
          "Incorrect use of pruning option. Set pruning threshold for each order n-grams.");
        // threshold for unigram can only be 0 (no pruning)
        UTIL_THROW_IF(counts_threshold[0] != 0, util::Exception, "You are not allowed to prune unigrams. Set counts_threshold to 0 for unigrams.");
        
        // check if threshold are not in decreasing order
        uint64_t lower_threshold = 0;

        for (std::vector<uint64_t>::iterator it = counts_threshold.begin(); it != counts_threshold.end(); ++it) {
            UTIL_THROW_IF(lower_threshold > *it, util::Exception, "Pruning counts threshold for lower n-grams are not allowed to be specified higher than for higher order n-grams");
            lower_threshold = *it;
        }
    } else {
       // pruning is not specified, so set threshold to 0 for all order n-grams
       counts_threshold.reserve(order);
       for (std::size_t i = 0; i < order; ++i)
           counts_threshold.push_back(0);
    }

    return counts_threshold;
}

