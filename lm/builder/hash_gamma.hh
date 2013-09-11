#ifndef LM_BUILDER_HASH_GAMMA__
#define LM_BUILDER_HASH_GAMMA__

#include <stdint.h>

namespace lm { namespace builder {

struct HashGamma {
    uint64_t hash_value;
    float gamma;
};

}} // namespaces
#endif // LM_BUILDER_HASH_GAMMA__
