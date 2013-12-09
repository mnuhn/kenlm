#ifndef LM_ENUMERATE_STATES__
#define LM_ENUMERATE_STATES__

namespace lm {

/* If you need all possible states of the lm, inherit from this class
 * and implement Add.  Then put a pointer in Config.enumerate_vocab; it does
 * not take ownership.  Add is called once per state.
 * This is only used by the Model constructor;
 * the pointer is not retained by the class.  
 */
class EnumerateStates {
  public:
    virtual ~EnumerateStates() {}

    virtual void Add(std::vector<WordIndex> ngram) = 0;

  protected:
    EnumerateStates() {}
};

} // namespace lm

#endif // LM_ENUMERATE_STATES__

