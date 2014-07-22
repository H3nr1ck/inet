//
// Copyright (C) 2013 OpenSim Ltd.
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

#ifndef __INET_RECEIVERBASE_H
#define __INET_RECEIVERBASE_H

#include "IReceiver.h"
#include "ITransmission.h"
#include "IReception.h"

namespace inet {

namespace physicallayer {

class INET_API ReceiverBase : public cModule, public virtual IReceiver
{
  protected:
    virtual bool computeIsSynchronizationPossible(const ITransmission *transmission) const;
    virtual bool computeIsSynchronizationPossible(const IListening *listening, const IReception *reception) const = 0;
    virtual bool computeIsSynchronizationAttempted(const IListening *listening, const IReception *reception, const std::vector<const IReception *> *interferingReceptions) const;
    virtual bool computeIsReceptionPossible(const ITransmission *transmission) const;
    virtual bool computeIsReceptionPossible(const IListening *listening, const IReception *reception) const = 0;
    virtual bool computeIsReceptionAttempted(const IListening *listening, const IReception *reception, const std::vector<const IReception *> *interferingReceptions) const;

  public:
    ReceiverBase() {}

    virtual W getMinInterferencePower() const { return W(qNaN); }
    virtual W getMinReceptionPower() const { return W(qNaN); }
};

} // namespace physicallayer

} // namespace inet

#endif // ifndef __INET_RECEIVERBASE_H

