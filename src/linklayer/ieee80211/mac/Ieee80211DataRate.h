#ifndef __INET_IEEE80211DATARATE_H
#define __INET_IEEE80211DATARATE_H

#include "INETDefs.h"

#include "ModulationType.h"

namespace inet {

struct Ieee80211DescriptorData
{
    char mode;
    double bitrate;
    ModulationType modulationType;
};

class Ieee80211Descriptor
{
  private:
    static const int descriptorSize;
    static const Ieee80211DescriptorData data[];

  public:
    static int findIdx(char mode, double bitrate);
    static int getIdx(char mode, double bitrate);
    static int getMinIdx(char mode);
    static int getMaxIdx(char mode);
    static bool incIdx(int& idx);
    static bool decIdx(int& idx);
    static const Ieee80211DescriptorData& getDescriptor(int idx);
    static int size() { return descriptorSize; }
};

} // namespace inet

#endif // ifndef __INET_IEEE80211DATARATE_H
