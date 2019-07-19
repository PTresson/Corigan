def calc_IOU(xA, xB, yA, yB, wA, wB, hA, hB):
    xminA = xA - 0.5 * wA
    xmaxA = xA + 0.5 * wA
    xminB = xB - 0.5 * wB
    xmaxB = xB + 0.5 * wB

    yminA = yA - 0.5 * hA
    ymaxA = yA + 0.5 * hA
    yminB = yB - 0.5 * hB
    ymaxB = yB + 0.5 * hB

    areaA = wA * hA
    areaB = wB * hB

    areaTot = areaA + areaB

    xmin = min(xminA, xminB)
    ymin = min(yminA, yminB)
    xmax = max(xmaxA, xmaxB)
    ymax = max(ymaxA, ymaxB)

    '''
       _____      _____
      |___  |    | |   |
      |   | |    | |___|
      |___|_|    |_____|

    '''

    if ((xminA == xminB and yminA == yminB)):
        oppx = min(xmaxA, xmaxB)
        oppy = min(ymaxA, ymaxB)
        lengthx = oppx - xminA
        lengthy = oppy - yminA
        areaOver = lengthx * lengthy
        IOU = areaOver / (areaTot - areaOver)
        return IOU

    if (xmaxA == xmaxB and ymaxA == ymaxB):
        oppx = max(xminA, xminB)
        oppy = max(yminA, yminB)
        lengthx = oppx - xmaxA
        lengthy = oppy - ymaxA
        areaOver = lengthx * lengthy
        IOU = areaOver / (areaTot - areaOver)
        return IOU

    if ((xmaxA == xmaxB and yminA == yminB)):
        oppx = max(xminA, xminB)
        oppy = min(ymaxA, ymaxB)
        lengthx = xmaxA - oppx
        lengthy = oppy - yminA
        areaOver = lengthx * lengthy
        IOU = areaOver / (areaTot - areaOver)
        return IOU

    if (xminA == xminB and ymaxA == ymaxB):
        oppx = min(xmaxA, xmaxB)
        oppy = max(yminA, yminB)
        lengthx = oppx - xminA
        lengthy = ymaxA - oppy
        areaOver = lengthx * lengthy
        IOU = areaOver / (areaTot - areaOver)
        return IOU

    '''
         ___
       _|_  |
      | |_|_|
      |___|

    '''

    if (xmin == xminA and ymin == yminA):

        if ((xminB < xmaxA and yminB <= ymaxA) or (xminB <= xmaxA and yminB < ymaxA)):

            oppx = min(xmaxA, xmaxB)
            oppy = min(ymaxA, ymaxB)
            lengthx = oppx - xminB
            lengthy = oppy - yminB
            areaOver = lengthx * lengthy
            IOU = areaOver / (areaTot - areaOver)


        else:
            IOU = 0

        return IOU

    if (xmin == xminB and ymin == yminB):

        if ((xminA < xmaxB and yminA <= ymaxB) or (xminA <= xmaxB and yminA < ymaxB)):

            oppx = min(xmaxA, xmaxB)
            oppy = min(ymaxA, ymaxB)
            lengthx = oppx - xminA
            lengthy = oppy - yminA
            areaOver = lengthx * lengthy
            IOU = areaOver / (areaTot - areaOver)


        else:
            IOU = 0
        return IOU

    '''
         ___
        |  _|_
        |_|_| |
          |___|

    '''

    if (xmin == xminA and ymax == ymaxA):

        if ((xminB < xmaxA and ymaxB >= yminA) or (xminB <= xmaxA and ymaxB > yminA)):

            oppx = min(xmaxA, xmaxB)
            oppy = max(yminA, yminB)
            lengthx = oppx - xminB
            lengthy = ymaxB - oppy
            areaOver = lengthx * lengthy
            IOU = areaOver / (areaTot - areaOver)

        else:
            IOU = 0
        return IOU

    if (xmin == xminB and ymax == ymaxB):

        if ((xminA < xmaxB and ymaxA >= yminB) or (xminA <= xmaxB and ymaxA > yminB)):

            oppx = min(xmaxB, xmaxA)
            oppy = max(yminB, yminA)
            lengthx = oppx - xminA
            lengthy = ymaxA - oppy
            areaOver = lengthx * lengthy
            IOU = areaOver / (areaTot - areaOver)

        else:
            IOU = 0
        return IOU

    '''
         ___           ___       ___
        |  _|_       _|_  |    _|___|_
        | |_|_|     |_|_| |   |_|___|_|
        |___|         |___|     |___|

    '''

    if (ymin == yminA and ymax == ymaxA) :

        if(xmaxB > xminA and xminB < xmaxA) :
            lengthx = (min(xmaxA, xmaxB) - max(xminA, xminB))
            lengthy = ymaxB - yminB
            areaOver = lengthx * lengthy
            IOU = areaOver / (areaTot - areaOver)

        else :
            IOU = 0

        return IOU

    if (ymin == yminB and ymax == ymaxB):

        if(xmaxA > xminB and xminA < xmaxB) :
            lengthx = (min(xmaxA, xmaxB) - max(xminA, xminB))
            lengthy = ymaxA - yminA
            areaOver = lengthx * lengthy
            IOU = areaOver / (areaTot - areaOver)

        else :
            IOU = 0

        return IOU

###########################################################################################################################

def calc_Inter(xA, xB, yA, yB, wA, wB, hA, hB):
    xminA = xA - 0.5 * wA
    xmaxA = xA + 0.5 * wA
    xminB = xB - 0.5 * wB
    xmaxB = xB + 0.5 * wB

    yminA = yA - 0.5 * hA
    ymaxA = yA + 0.5 * hA
    yminB = yB - 0.5 * hB
    ymaxB = yB + 0.5 * hB

    areaA = wA * hA
    areaB = wB * hB



    xmin = min(xminA, xminB)
    ymin = min(yminA, yminB)
    xmax = max(xmaxA, xmaxB)
    ymax = max(ymaxA, ymaxB)

    '''
       _____      _____
      |___  |    | |   |
      |   | |    | |___|
      |___|_|    |_____|

    '''

    if ((xminA == xminB and yminA == yminB)):
        oppx = min(xmaxA, xmaxB)
        oppy = min(ymaxA, ymaxB)
        lengthx = oppx - xminA
        lengthy = oppy - yminA
        areaOver = lengthx * lengthy

        return areaOver

    if (xmaxA == xmaxB and ymaxA == ymaxB):
        oppx = max(xminA, xminB)
        oppy = max(yminA, yminB)
        lengthx = oppx - xmaxA
        lengthy = oppy - ymaxA
        areaOver = lengthx * lengthy

        return areaOver

    if ((xmaxA == xmaxB and yminA == yminB)):
        oppx = max(xminA, xminB)
        oppy = min(ymaxA, ymaxB)
        lengthx = xmaxA - oppx
        lengthy = oppy - yminA
        areaOver = lengthx * lengthy

        return areaOver

    if (xminA == xminB and ymaxA == ymaxB):
        oppx = min(xmaxA, xmaxB)
        oppy = max(yminA, yminB)
        lengthx = oppx - xminA
        lengthy = ymaxA - oppy
        areaOver = lengthx * lengthy

        return areaOver

    '''
         ___
       _|_  |
      | |_|_|
      |___|

    '''

    if (xmin == xminA and ymin == yminA):

        if ((xminB < xmaxA and yminB <= ymaxA) or (xminB <= xmaxA and yminB < ymaxA)):

            oppx = min(xmaxA, xmaxB)
            oppy = min(ymaxA, ymaxB)
            lengthx = oppx - xminB
            lengthy = oppy - yminB
            areaOver = lengthx * lengthy



        else:
            areaOver = 0

        return areaOver

    if (xmin == xminB and ymin == yminB):

        if ((xminA < xmaxB and yminA <= ymaxB) or (xminA <= xmaxB and yminA < ymaxB)):

            oppx = min(xmaxA, xmaxB)
            oppy = min(ymaxA, ymaxB)
            lengthx = oppx - xminA
            lengthy = oppy - yminA
            areaOver = lengthx * lengthy



        else:
            areaOver = 0
        return areaOver

    '''
         ___
        |  _|_
        |_|_| |
          |___|

    '''

    if (xmin == xminA and ymax == ymaxA):

        if ((xminB < xmaxA and ymaxB >= yminA) or (xminB <= xmaxA and ymaxB > yminA)):

            oppx = min(xmaxA, xmaxB)
            oppy = max(yminA, yminB)
            lengthx = oppx - xminB
            lengthy = ymaxB - oppy
            areaOver = lengthx * lengthy


        else:
            areaOver = 0
        return areaOver

    if (xmin == xminB and ymax == ymaxB):

        if ((xminA < xmaxB and ymaxA >= yminB) or (xminA <= xmaxB and ymaxA > yminB)):

            oppx = min(xmaxB, xmaxA)
            oppy = max(yminB, yminA)
            lengthx = oppx - xminA
            lengthy = ymaxA - oppy
            areaOver = lengthx * lengthy


        else:
            areaOver = 0
        return areaOver

    '''
         ___           ___       ___
        |  _|_       _|_  |    _|___|_
        | |_|_|     |_|_| |   |_|___|_|
        |___|         |___|     |___|

    '''

    if (ymin == yminA and ymax == ymaxA):

        if ((xmaxB > xminA and xminB < xmaxA)) :
            lengthx = (min(xmaxA, xmaxB) - max(xminA, xminB))
            lengthy = ymaxB - yminB
            areaOver = lengthx * lengthy

        else :
            areaOver = 0

        return areaOver

    if (ymin == yminB and ymax == ymaxB) :

        if (xmaxA > xminB and xminA < xmaxB) :
            lengthx = (min(xmaxA, xmaxB) - max(xminA, xminB))
            lengthy = ymaxA - yminA
            areaOver = lengthx * lengthy

        else :
            areaOver = 0

        return areaOver



