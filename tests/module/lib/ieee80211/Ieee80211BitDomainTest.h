//
// Copyright (C) 2014 OpenSim Ltd.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, see <http://www.gnu.org/licenses/>.
//

#ifndef __INET_IEEE80211BITDOMAINTEST_H_
#define __INET_IEEE80211BITDOMAINTEST_H_

#include "ConvolutionalCoderModule.h"
#include "Ieee80211InterleaverModule.h"
#include "Ieee80211ScramblerModule.h"
#include "INETDefs.h"
#include "ModuleAccess.h"

using namespace inet::physicallayer;

namespace inet {

class INET_API Ieee80211BitDomainTest : public cSimpleModule
{
    protected:
        Ieee80211ScramblerModule *scrambler;
        Ieee80211InterleaverModule *interleaver;
        ConvolutionalCoderModule *convCoder;
        std::ifstream *fileStream;
        const char *testType;

    protected:
        virtual int numInitStages() const { return NUM_INIT_STAGES; }
        virtual void initialize(int stage);
        virtual void handleMessage(cMessage *msg) { throw cRuntimeError("This module doesn't handle self messages"); }
        void testConvolutionalCoder(unsigned int numberOfRandomErrors) const;
        void testScrambler() const;
        void testInterleaver() const;
        void testIeee80211BitDomain() const;

    public:
        ~Ieee80211BitDomainTest();
};

} /* namespace inet */

#endif /* __INET_IEEE80211BITDOMAINTEST_H_ */
