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

#ifndef __INET_IMODULATION_H_
#define __INET_IMODULATION_H_

#include "inet/common/INETDefs.h"

namespace inet {
namespace physicallayer {

class INET_API IModulation
{
    public:
        virtual ~IModulation() {}
        virtual double calculateBER(double snir, double bandwidth, double bitrate) const = 0;
};

} /* namespace physicallayer */
} /* namespace inet */

#endif /* __INET_IMODULATION_H_ */
