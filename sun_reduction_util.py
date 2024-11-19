#!/usr/bin/env python
#
# $Id: sun_reduction_util.py,v 1.1 2018/08/16 07:38:51 dpetry Exp $
#
# sun_reduction_util.py
#
#   2018/05/08 revised by M. Shimojo (bugfix "sol_getats_2")
#   2018/06/20 rebised by M. Shimojo (bugfix "sol_ampcal_2" with option exisTbl =True)
#   2022/06/16 revised by M. Shimojo (For Python version 3)
#   2024/04/15 revised by M. Shimojo (bugfix "sol_ampcal_2" with option exisTbl =True)
#
import os
import sys
import math
import csv
#####---------Copy from Analysis Utils--------------------------
import re
#import casadef
import pylab as pb
import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time as timeUtilities
import socket
import casalith

casaVersion = casalith.version_string()
casaAvailable = False
casaVersionWithMSMD = '4.1.0'
casaVersionWithMSMDFieldnames = '4.5'
casaVersionWithUndefinedFrame = '4.3.0'

#------------------------------------------------------------------------
def computeDurationOfScan(scanNumber, t=None, vis=None, returnSubscanTimes=False,
                          verbose=False, gapFactor=None, includeDate=False,
                          mymsmd=None, scienceSpwsOnly=False):
    """
    This function is used by timeOnSource() to empirically determine the number
    of subscans and the total duration of all the subscans of a particular scan.
    We implicitly assume that the same field is observed for the whole scan, which is
    not true of mosaics.
    Inputs:
    scanNumber: the scan number, simply for generating a file list of timestamps
    t = a sequence of integration timestamps (optional for casa >= 4.1.0)
    vis = the measurement set (not necessary if t is given)
    mymsmd = an msmd instance (as an alternative to vis)
    gapFactor: default=2.75 for integrations<1sec, 2.0 otherwise
    Returns:
    1) duration in seconds
    2) the number of subscans
    and if returnSubscanTimes==True
    3) the begin/end timestamps of each subscan
    4) the begin/end timestampStrings of each subscan
    -- Todd Hunter
    """
    if (t is None and vis is None and (mymsmd is None or mymsmd=='')):
        print("You must specify either vis, mymsmd or t.")
        return
    keepmymsmd = False
    if (t is None):
        if (casaVersion < casaVersionWithMSMD):
            print("For this version of casa, you must specify t rather than vis.")
            return
        if (mymsmd is None or mymsmd == ''):
            mymsmd = createCasaTool(msmdtool)
            mymsmd.open(vis)
        else:
            keepmymsmd = True
    elif (mymsmd is None or mymsmd==''):  # fix for CSV-2903 on 29-Apr-2016
        mymsmd = createCasaTool(msmdtool)
        mymsmd.open(vis)
    else:
        keepmymsmd = True
    if (t is None):
        t = pickTimesForScan(mymsmd, scanNumber, scienceSpwsOnly=scienceSpwsOnly)
    else:
        t = np.unique(t)
    if (len(t) <= 1):
        if (casaVersion < casaVersionWithMSMD):
            print("This version of CASA is too old for this function to handle single-dump integrations.")
            return(0,0)
        elif len(t) == 0:
            return(0,0)
        else:
            if (scanNumber==1):
                duration = np.min(pickTimesForScan(mymsmd, scanNumber+1, scienceSpwsOnly=scienceSpwsOnly)) - t[0]
            else:
                duration = t[0] - np.max(pickTimesForScan(mymsmd, scanNumber-1, scienceSpwsOnly=scienceSpwsOnly))
            if (not keepmymsmd): mymsmd.close()
            if (returnSubscanTimes):
                timestampsString = mjdsecToTimerange(t[0]-0.5*duration,t[0]+0.5*duration,
                                                     decimalDigits=1,includeDate=includeDate)
                return(duration,1,{0:t},{0:timestampsString})
            else:
                return(duration,1)
    else:
        if (mymsmd is not None and mymsmd != '' and not keepmymsmd):
            mymsmd.close()
        d = np.max(t) - np.min(t)
        # initial estimate for interval
        diffs = np.diff(t)
        avgInterval = np.median(diffs)
        startTime = previousTime = t[0]
        subscans = 1
        if (gapFactor is None):
            if (avgInterval > 1):
                gapFactor = 2
            else:
                gapFactor = 2.75 # it was 3 for a long time, but failed on 2013-01-24 dataset (2014-09-23)
        startTime = previousTime = t[0]
        duration = 0
        tdiffs = []
        s = ''
        timestamps = {}
        timestampsString = {}
        droppedTimeTotal = 0
        daygaps = 0
        for i in range(1,len(t)):
            s += "%.2f " % (t[i]-t[0])
            tdiff = t[i]-previousTime
            tdiffs.append(tdiff)
            if (tdiff > gapFactor*avgInterval):
                droppedTime = t[i]-previousTime+avgInterval
                droppedTimeTotal += droppedTime
                if droppedTime > 12*3600:
                    daygaps += droppedTime
                if (verbose):
                    print("    ***********************")
                    print("    i=%d, Dropped %.1f seconds" % (i,droppedTime))
                subscanLength = t[i-1] - startTime + avgInterval
                duration += subscanLength
                if (subscanLength > 1.5*avgInterval):
                    # Don't count single point subscans because they are probably not real.
                    timestamps[subscans] = [startTime,t[i-1]]
                    timestampsString[subscans] = mjdsecToTimerange(startTime,t[i-1],decimalDigits=1,includeDate=includeDate)
                    if (verbose):
                        print("Scan %d: Subscan %d: %s, duration=%f" % (scanNumber,subscans,s,subscanLength))
                    s = ''
                    subscans += 1
                elif (verbose):
                    print("Scan %d: dropped a dump of length %f because it was between subscans" % (scanNumber,subscanLength))
                startTime = t[i]
            previousTime = t[i]
        if (droppedTimeTotal > 0 and verbose):
            print("+++++ Scan %d: total dropped time = %.1f seconds = %.1f minutes" % (scanNumber,droppedTimeTotal, droppedTimeTotal/60.))
        if daygaps > 0:
            print("+++++ Scan %d: large time gaps: %.1f sec = %.1f hrs = %.1f days" % (scanNumber,daygaps,daygaps/3600.,daygaps/86400.))
        duration += t[len(t)-1] - startTime
        timestamps[subscans] = [startTime,t[len(t)-1]]
        timestampsString[subscans] = mjdsecToTimerange(startTime,t[len(t)-1],decimalDigits=1,includeDate=includeDate)
    if (returnSubscanTimes):
        return(duration, subscans, timestamps, timestampsString)
    else:
        return(duration, subscans)

#------------------------------------------------------------------------
def createCasaTool(mytool):
    """
    A wrapper to handle the changing ways in which casa tools are invoked.
    For CASA < 6, it relies on "from taskinit import *" in the preamble above.
    mytool: a tool name, like tbtool
    Todd Hunter
    """
    if 'casac' in locals():
        if (type(casac.Quantity) != type):  # casa 4.x and 5.x
            myt = mytool()
        else:  # casa 3.x
            myt = mytool.create()
    else:
        # this is CASA 6
        myt = mytool()
    return(myt)

#------------------------------------------------------------------------
def pickTimesForScan(mymsmd, scan, scienceSpwsOnly=True, useTimesForSpwsIfAvailable=False):
    """
    Chooses whether to run msmd.timesforscan() or the aU workaround for the
    bug in CASA 4.4 (CAS-7622).  Called only by computeDurationOfScan.
    -Todd Hunter
    """
    if (casaVersion >= '4.4' and casaVersion < '4.5'):
        t = getTimesForScan(mymsmd, scan)
    else:
        t = mymsmd.timesforscan(scan)
        if scienceSpwsOnly:
            scienceSpws = getScienceSpws(mymsmd.name(), mymsmd=mymsmd, returnString=False)
            if len(scienceSpws) == 0:
                print("There are no science spws, so considering all spws instead.")
                scienceSpwsOnly = False
        if scienceSpwsOnly:
            if useTimesForSpwsIfAvailable:
                try:
                    # available starting in 5.1.0-46
                    print("Calling msmd.timesforspws(%s)" % (str(scienceSpws[0])))
                    times = mymsmd.timesforspws(scienceSpws[0])
                    t = np.intersect1d(times, t)
                except:
                    mytb = createCasaTool(tbtool)
                    mytb.open(mymsmd.name())
                    print("Restricting times to first spw: %d" % (scienceSpws[0]))
                    myt = mytb.query('DATA_DESC_ID == %d && SCAN_NUMBER == %d' % (scienceSpws[0], scan))
                    times = myt.getcol('TIME')
                    myt.close()
                    mytb.close()
                    idx = np.where((t >= np.min(times)) & (t <= np.max(times)))
                    print("Dropped %d points" % (len(t)-len(idx)))
                    t = t[idx]
            else:
                    mytb = createCasaTool(tbtool)
                    mytb.open(mymsmd.name())
                    print("Restricting times to spw %d for scan %d" % (scienceSpws[0],scan))
                    myt = mytb.query('DATA_DESC_ID == %d && SCAN_NUMBER == %d' % (scienceSpws[0], scan))
                    times = myt.getcol('TIME')
                    myt.close()
                    mytb.close()
                    if False:
                        # This will include SQLD, and was the old version of the code
                        idx = np.where((t >= np.min(times)) & (t <= np.max(times)))
                        print("Dropped %d points" % (len(t)-len(idx)))
                        t = t[idx]
                    else:
                        # This will not include SQLD.
                        t = np.unique(times)
    return (t)

#------------------------------------------------------------------------
def getTimesForScan(mymsmd,scan):
    """
    This function replaces msmd.timesforscan() in CASA 4.4 release, which
    has a bug in that it does not find scans not in obs ID=0.
    It should not be necessary once CASA 4.5 is released.
    mymsmd: can be an msmd instance or a measurement set name
    scan: can be a list or a single integer
    Returns a list in MJD seconds.
    -Todd Hunter
    """
    if (casaVersion < '4.4' or casaVersion >= '4.5'):
        needToClose = False
        if type(mymsmd) == str:
            needToClose = True
            vis = mymsmd
            mymsmd = msmdtool()
            mymsmd.open(vis)
        if type(scan) == list:
            scans = scan
        t = []
        for scan in scans:
            if scan in mymsmd.scannumbers():
                t += list(mymsmd.timesforscan(scan))
            else:
                print("No scan %. Available scans: %s" % (scan, mymsmd.scannumbers()))
        if needToClose: mymsmd.close()
        return(t)
    allscans = mymsmd.scansforintent('*')
    obsids = mymsmd.nobservations()
    t = []
    for o in range(obsids):
        scans = mymsmd.scannumbers(o)
        if scan in scans:
            t = mymsmd.timesforscan(scan, obsid=o)
    return(t)

#------------------------------------------------------------------------
def mjdsecToTimerange(mjdsec1, mjdsec2=None, decimalDigits=2, includeDate=True,
                      use_metool=True, debug=False):
    """
    Converts two value of MJD seconds into a UT date/time string suitable for
    the timerange argument in plotms.  They can be entered as two separated values,
    or a single tuple.
    Example output: '2012/03/14/00:00:00.00~2012/03/14/00:10:00.00'
    input options:
       decimalDigits: how many digits to display after the decimal point
       use_metool: True=use casa tool to convert to UT, False: use formula in aU
    -Todd Hunter
    """
    if (type(mjdsec1) == list or type(mjdsec1)==np.ndarray):
        mjdsec2 = mjdsec1[1]
        mjdsec1 = mjdsec1[0]
    return(mjdsecToTimerangeComponent(mjdsec1, decimalDigits, includeDate, use_metool, debug) + '~' \
           + mjdsecToTimerangeComponent(mjdsec2, decimalDigits, includeDate, use_metool, debug))

#------------------------------------------------------------------------
def mjdsecToTimerangeComponent(mjdsec, decimalDigits=2, includeDate=True, use_metool=True, debug=False):
    """
    Converts a value of MJD seconds into a UT date/time string suitable for one
    member of the timerange argument in plotms.
    example: '2012/03/14/00:00:00.00'
    input options:
       decimalDigits: how many digits to display after the decimal point
       use_metool: True=use casa tool to convert to UT, False: use formula in aU
    Todd Hunter
    """
    if (not casaAvailable or use_metool==False):
        mjd = mjdsec / 86400.
        jd = mjdToJD(mjd)
        trialUnixTime = 1200000000
        diff  = ComputeJulianDayFromUnixTime(trialUnixTime) - jd
        if (debug): print("first difference = %f days" % (diff))
        trialUnixTime -= diff*86400
        diff  = ComputeJulianDayFromUnixTime(trialUnixTime) - jd
        if (debug): print("second difference = %f seconds" % (diff*86400))
        trialUnixTime -= diff*86400
        diff  = ComputeJulianDayFromUnixTime(trialUnixTime) - jd
        if (debug): print("third difference = %f seconds" % (diff*86400))
        # Convert unixtime to date string
        if (includeDate):
            utstring = timeUtilities.strftime('%Y/%m/%d/%H:%M:%S',
                                              timeUtilities.gmtime(trialUnixTime))
        else:
            utstring = timeUtilities.strftime('%H:%M:%S',
                                              timeUtilities.gmtime(trialUnixTime))
        utstring += '.%0*d'  % (decimalDigits, np.round(10**decimalDigits*(trialUnixTime % 1)))
    else:
        me = createCasaTool(metool)
        today = me.epoch('utc','today')
        mjd = np.array(mjdsec) / 86400.
        today['m0']['value'] =  mjd
        hhmmss = call_qa_time(today['m0'],prec=6+decimalDigits)
        myqa = createCasaTool(qatool)
        date = myqa.splitdate(today['m0'])
        myqa.done()
        if (includeDate):
            utstring = "%s/%02d/%02d/%s" % (date['year'],date['month'],date['monthday'],hhmmss)
        else:
            utstring = hhmmss
    return(utstring)

#------------------------------------------------------------------------
def ComputeJulianDayFromUnixTime(seconds):
    """
    Converts a time expressed in unix time (seconds since Jan 1, 1970)
    into Julian day number as a floating point value.
    - Todd Hunter
    """
    [tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst] = timeUtilities.gmtime(seconds)
    if (tm_mon < 3):
        tm_mon += 12
        tm_year -= 1
    UT = tm_hour + tm_min/60. + tm_sec/3600.
    a =  np.floor(tm_year / 100.)
    b = 2 - a + np.floor(a/4.)
    day = tm_mday + UT/24.
    jd  = np.floor(365.25*((tm_year)+4716)) + np.floor(30.6001*((tm_mon)+1))  + day + b - 1524.5
    return(jd)

#------------------------------------------------------------------------
def getAntennaNames(msFile) :
    """
    Returns the list of antenna names in the specified ms ANTENNA table using tbtool.
    Obsoleted by msmd.antennanames(range(msmd.nantennas())), but kept because it is faster.
    """
    if (msFile.find('*') >= 0):
        mylist = glob.glob(msFile)
        if (len(mylist) < 1):
            print("getAntennaNames: Could not find measurement set.")
            return
        msFile = mylist[0]
    mytb = createCasaTool(tbtool)
    mytb.open(msFile+'/ANTENNA')
    names = mytb.getcol('NAME')
    mytb.close()
    return names

#------------------------------------------------------------------------
def getLoadTemperatures(vis, antenna = None, doplot=False,
                        warnIfNoLoadTemperatures=True, mymsmd=None):
    """
    Gets the load temperatures and timestamps from the ASDM_CALDEVICE
    table (if present), otherwise get them from the CALDEVICE table.
    Returns a dictionary of all antennas
    keyed by antenna name and ID, then by scan number.
    doplot: if True, then draw a plot
    antenna: single antenna to plot (ID as integer or string or name)
             if None, then plot all antennas
    -Todd Hunter
    """
    if (os.path.exists(vis) == False):
        print("The measurement set is not found.")
        return
    if (os.path.exists(vis+'/table.dat') == False):
        print("No table.dat.  This does not appear to be an ms.")
        return
    if (casaVersion < casaVersionWithMSMD):
        print("This version of CASA is too old.  It needs msmd (4.1 or newer).")
        return
    if (os.path.exists(vis+'/ASDM_CALDEVICE')):
        table = 'ASDM_CALDEVICE'
        asdm = True
    else:
        table = 'CALDEVICE'
        asdm = False
    mytable = vis+'/'+table
    needToClose = False
    if mymsmd is None:
        mymsmd = createCasaTool(msmdtool)
        mymsmd.open(vis)
        needToClose = True
    scans = mymsmd.scannumbers()
    antennas = range(mymsmd.nantennas())
    antennaNames = mymsmd.antennanames(antennas)
    mydict = {}
    for ant in antennas:
        mydict[ant] = {}
        mydict[antennaNames[ant]] = {}
    if (os.path.exists(mytable) == False):
        if (warnIfNoLoadTemperatures):
            print("The ASDM_CALDEVICE table is not present. You need to importasdm(asis='CalDevice') to get")
            print("correct load temperatures.  Will proceed using default temperature for the hot load and ")
            print("ambient load, which is only a concern if you want to recalculate Tsys.")
        for ant in antennas:
            for scan in scans:
                mydict[ant][scan] = {'hot':DEFAULT_HOTLOAD_TEMP, 'amb':DEFAULT_AMBLOAD_TEMP}
                mydict[antennaNames[ant]][scan] = {'hot':DEFAULT_HOTLOAD_TEMP, 'amb':DEFAULT_AMBLOAD_TEMP}
        if needToClose:
            mymsmd.close()
        return(mydict)
    # need to use default temperatures if table present, but empty.
    mytb = createCasaTool(tbtool)
    mytb.open(mytable)
    if (asdm):
        temperatureLoad = mytb.getcol('temperatureLoad')
        timeInterval = mytb.getcol('timeInterval')
        antennaIds = mytb.getcol('antennaId')
        calLoadNames = mytb.getcol('calLoadNames')
        antennaId = np.array([int(x.split('_')[1]) for x in antennaIds])
    else:
        temperatureLoad = mytb.getcol('TEMPERATURE_LOAD')
        timeInterval = mytb.getcol('INTERVAL')
        antennaIds = mytb.getcol('ANTENNA_ID')
        calLoadNames = mytb.getcol('CAL_LOAD_NAMES')
        antennaId = antennaIds
    if len(temperatureLoad) < 1:
        if (warnIfNoLoadTemperatures):
            print("The ASDM_CALDEVICE table is not present. You need to importasdm(asis='CalDevice') to get")
            print("correct load temperatures.  Will proceed using default temperature for the hot load and ")
            print("ambient load, which is only a concern if you want to recalculate Tsys.")
        for ant in antennas:
            for scan in scans:
                mydict[ant][scan] = {'hot':DEFAULT_HOTLOAD_TEMP, 'amb':DEFAULT_AMBLOAD_TEMP}
                mydict[antennaNames[ant]][scan] = {'hot':DEFAULT_HOTLOAD_TEMP, 'amb':DEFAULT_AMBLOAD_TEMP}
    if (antenna is not None):
        antennaIDs = parseAntenna(vis, antenna)
        if (antennaIDs is None):
            print("Antenna %s not in this dataset (0..%d)" % (str(antenna),np.max(antennaId)))
            return
        singleAntennaID = antennaIDs[0]
    mytb.close()
    times = []
    hotLoadTemps = []
    ambLoadTemps = []
    antennaTempIDs = []
    for ant in antennas:
        for scan in scans:
            for row in range(len(antennaId)):
                if (ant==antennaId[row]):
                    meanscantime = np.mean(mymsmd.timesforscan(scan))
                    times.append(meanscantime)
                    amb = list(calLoadNames[:,row]).index('AMBIENT_LOAD')
                    hot = list(calLoadNames[:,row]).index('HOT_LOAD')
                    ambLoad = temperatureLoad[amb][row]
                    hotLoad = temperatureLoad[hot][row]
                    ambLoadTemps.append(ambLoad)
                    hotLoadTemps.append(hotLoad)
                    antennaTempIDs.append(ant)
                    mydict[antennaId[row]][scan] = {'hot':hotLoad,'amb':ambLoad}
                    mydict[antennaNames[antennaId[row]]][scan] = {'hot':hotLoad,'amb':ambLoad}
                    break
    if needToClose:
        mymsmd.close()
    times = np.array(times)
    antennaTempIDs = np.array(antennaTempIDs)
    hotLoadTemps = np.array(hotLoadTemps)
    ambLoadTemps = np.array(ambLoadTemps)
    if (antenna is not None):
        indices = np.where(antennaTempIDs == singleAntennaID)[0]
    else:
        indices = range(len(antennaTempIDs))
    if (doplot):
        pb.clf()
        timestamps = pb.date2num(mjdSecondsListToDateTime(times))
        pb.plot_date(timestamps[indices], ambLoadTemps[indices], 'k.')
#        pb.hold(True)# not needed
        pb.plot_date(timestamps[indices], hotLoadTemps[indices], 'r.')
        pb.xlabel('Time (UT)')
        pb.ylabel('Temperature (K)')
        if (antenna is not None):
            pb.title(os.path.basename(vis) + ' antenna %s: Black = ambient, red = heated'%str(antenna),size=12)
        else:
            pb.title(os.path.basename(vis) + ': Black = ambient,  red = heated',size=12)
        pb.draw()
    return(mydict)

#------------------------------------------------------------------------
def parseAntenna(vis, antenna, mymsmd=''):
    """
    Parse an antenna argument (integer or string or list) to emulate
    plotms selection, given a measurement set.
    Returns: an integer list of antenna IDs:
    antenna='' will return all antenna IDs; to exclude 2 antennas:
           antenna='!ea01,!ea08'
    Todd Hunter
    """
    msmdCreated = False
    if (casaVersion >= casaVersionWithMSMD):
        if (mymsmd == ''):
            mymsmd = createCasaTool(msmdtool)
            mymsmd.open(vis)
            msmdCreated = True
        if (getCasaSubversionRevision() >= '27137'):
            uniqueAntennaIds = mymsmd.antennaids()
        else:
            uniqueAntennaIds = range(mymsmd.nantennas())
    else:
        print("Running ValueMapping to translate antenna names")
        vm = ValueMapping(vis)
        uniqueAntennaIds = range(vm.numAntennas)
    if type(antenna) == str or type(antenna) == np.string_:
        if len(antenna) == 0:
            return(uniqueAntennaIds)
    result = parseAntennaArgument(antenna, uniqueAntennaIds, vis, mymsmd=mymsmd)
    if msmdCreated:
        mymsmd.close()
    return(result)

#------------------------------------------------------------------------
def mjdToJD(MJD=None):
    """
    Converts an MJD value to JD.  Default = now.
    """
    if (MJD==None): MJD = getMJD()
    JD = MJD + 2400000.5
    return(JD)
#------------------------------------------------------------------------
def getMJDSec():
    """
    Returns the current MJD in seconds.
    Todd Hunter
    """
    return(getCurrentMJDSec())
#------------------------------------------------------------------------
def getCurrentMJDSec():
    """
    Returns the current MJD in seconds.
    Todd Hunter
    """
    mjdsec = getMJD() * 86400
    return(mjdsec)
#------------------------------------------------------------------------
def getMJD():
    """
    Returns the current MJD.  See also getCurrentMJDSec().
    -Todd
    """
    myme = createCasaTool(metool)
    mjd = myme.epoch('utc','today')['m0']['value']
    myme.done()
    return(mjd)
#------------------------------------------------------------------------
def parseAntennaArgument(antenna, uniqueAntennaIds, msname='', verbose=False, mymsmd=''):
    """
    Parse an antenna argument (integer or string or list) to emulate
    plotms selection, given a list of unique antenna IDs present in
    the dataset.
    antenna: can be string, list of integers, or single integer
    uniqueAntennaIds: list of integers, or comma-delimited string
    Todd Hunter
    """
    myValidCharacterListWithBang = ['~', ',', ' ', '*', '!',] + [str(m) for m in range(10)]
    if (type(uniqueAntennaIds) == str):
        uniqueAntennaIds = uniqueAntennaIds.split(',')
    if (type(antenna) == str) or type(antenna) == np.string_:
        if (len(antenna) == sum([m in myValidCharacterListWithBang for m in antenna])):
            # a simple list of antenna numbers was given
            tokens = antenna.split(',')
            antlist = []
            removeAntenna = []
            for token in tokens:
                if (len(token) > 0):
                    if (token.find('*')==0 and len(token)==1):
                        antlist = uniqueAntennaIds
                        break
                    elif (token.find('!')==0):
                        antlist = uniqueAntennaIds
                        removeAntenna.append(int(token[1:]))
                    elif (token.find('~')>0):
                        (start,finish) = token.split('~')
                        antlist +=  range(int(start),int(finish)+1)
                    else:
                        antlist.append(int(token))
            antlist = np.array(antlist,dtype=int)
            removeAntenna = np.array(removeAntenna,dtype=int)
            for rm in removeAntenna:
                antlist = antlist[np.where(antlist != rm)[0]]
            antlist = list(antlist)
            if (len(antlist) < 1 and len(removeAntenna)>0):
                print("Too many negated antennas -- there are no antennas left.")
                return
        else:
            # The antenna name (or list of names) was specified
            if (verbose): print("name specified")
            tokens = antenna.split(',')
            if (msname != ''):
                if (casaVersion < casaVersionWithMSMD):
                    print("Running ValueMapping to translate antenna names")
                    vm = ValueMapping(msname)
                    uniqueAntennas = vm.uniqueAntennas
                else:
                    if mymsmd == '':
                        mymsmd = createCasaTool(msmdtool)
                        mymsmd.open(msname)
                        needToClose = True
                    else:
                        needToClose = False
                    if (getCasaSubversionRevision() >= '27137'):
                        uniqueAntennas = mymsmd.antennanames()
                    else:
                        uniqueAntennas = getAntennaNames(msname)
                antlist = []
                removeAntenna = []
                for token in tokens:
                    if (token in uniqueAntennas):
                        antlist = list(antlist)  # needed in case preceding antenna had ! modifier
                        if (casaVersion < casaVersionWithMSMD):
                            antlist.append(vm.getAntennaIdsForAntennaName(token))
                        else:
                            antlist.append(mymsmd.antennaids(token)[0])
                    elif (token[0] == '!'):
                        if (token[1:] in uniqueAntennas):
                            antlist = uniqueAntennaIds
                            if (casaVersion < casaVersionWithMSMD):
                                removeAntenna.append(vm.getAntennaIdsForAntennaName(token[1:]))
                            else:
                                removeAntenna.append(mymsmd.antennaids(token[1:]))
                        else:
                            print("Antenna %s is not in the ms. It contains: " % (token), uniqueAntennas)
                            return
                    else:
                        print("Antenna %s is not in the ms. It contains: " % (token), uniqueAntennas)
                        return
                antlist = np.array(antlist,dtype=int)
                removeAntenna = np.array(removeAntenna,dtype=int)
                for rm in removeAntenna:
                    antlist = antlist[np.where(antlist != rm)[0]]
                antlist = list(antlist)
                if (len(antlist) < 1 and len(removeAntenna)>0):
                    print("Too many negated antennas -- there are no antennas left.")
                    return
                if (casaVersion >= casaVersionWithMSMD and needToClose):
                    mymsmd.close()
            else:
                print("Antennas cannot be specified my name if the ms is not found.")
                return
    elif (type(antenna) == list or type(antenna) == np.ndarray):
        # it's a list or array of integers
        if (verbose): print("converting int to list")
        antlist = list(antenna)
    else:
        # It's a single, integer entry
        if (verbose): print("converting int to list")
        antlist = [antenna]
    return(antlist)

#------------------------------------------------------------------------
def getObservationStartDate(vis, obsid=0, delimiter='-', measuresToolFormat=False):
    """
    Uses the tb tool to read the start time of the observation and reports the date.
    See also getObservationStartDay.
    Returns: '2013-01-31 07:36:01 UT'
    measuresToolFormat: if True, then return '2013/01/31/07:36:01'
    -Todd Hunter
    """
    mjdsec = getObservationStart(vis, obsid)
    if (mjdsec is None):
        return
    obsdateString = mjdToUT(mjdsec/86400.)
    if (delimiter != '-'):
        obsdateString = obsdateString.replace('-', delimiter)
    if measuresToolFormat:
        return(obsdateString.replace(' UT','').replace(delimiter,'/').replace(' ','/'))
    else:
        return(obsdateString)

#------------------------------------------------------------------------
def getObservationStart(vis, obsid=-1, verbose=False):
    """
    Reads the start time of the observation from the OBSERVATION table (using tb tool)
    and reports it in MJD seconds.
    obsid: if -1, return the start time of the earliest obsID
    -Todd Hunter
    """
    if (os.path.exists(vis) == False):
        print("vis does not exist = %s" % (vis))
        return
    if (os.path.exists(vis+'/table.dat') == False):
        print("No table.dat.  This does not appear to be an ms.")
        print("Use au.getObservationStartDateFromASDM().")
        return
    mytb = createCasaTool(tbtool)
    try:
        mytb.open(vis+'/OBSERVATION')
    except:
        print("ERROR: failed to open OBSERVATION table on file "+vis)
        return(3)
    time_range = mytb.getcol('TIME_RANGE')
    mytb.close()
    if verbose:  print("time_range: ", str(time_range))
    # the first index is whether it is starttime(0) or stoptime(1)
    time_range = time_range[0]
    if verbose:  print("time_range[0]: ", str(time_range))
    if (obsid >= len(time_range)):
        print("Invalid obsid")
        return
    if obsid >= 0:
        time_range = time_range[obsid]
    elif (type(time_range) == np.ndarray):
        time_range = np.min(time_range)
    return(time_range)

#------------------------------------------------------------------------
def mjdToUT(mjd=None, use_metool=True, prec=6):
    """
    Converts an MJD value to a UT date and time string
    such as '2012-03-14 00:00:00 UT'
    use_metool: whether or not to use the CASA measures tool if running from CASA.
         This parameter is simply for testing the non-casa calculation.
    -Todd Hunter
    """
    if mjd is None:
        mjdsec = getCurrentMJDSec()
    else:
        mjdsec = mjd*86400
    utstring = mjdSecondsToMJDandUT(mjdsec, use_metool, prec=prec)[1]
    return(utstring)

#------------------------------------------------------------------------
def mjdSecondsToMJDandUT(mjdsec, use_metool=True, debug=False, prec=6, delimiter='-'):
    """
    Converts a value of MJD seconds into MJD, and into a UT date/time string.
    prec: 6 means HH:MM:SS,  7 means HH:MM:SS.S
    example: (56000.0, '2012-03-14 00:00:00 UT')
    Caveat: only works for a scalar input value
    Todd Hunter
    """
    if (not casaAvailable or use_metool==False):
        mjd = mjdsec / 86400.
        jd = mjdToJD(mjd)
        trialUnixTime = 1200000000
        diff  = ComputeJulianDayFromUnixTime(trialUnixTime) - jd
        if (debug): print("first difference = %f days" % (diff))
        trialUnixTime -= diff*86400
        diff  = ComputeJulianDayFromUnixTime(trialUnixTime) - jd
        if (debug): print("second difference = %f seconds" % (diff*86400))
        trialUnixTime -= diff*86400
        diff  = ComputeJulianDayFromUnixTime(trialUnixTime) - jd
        if (debug): print("third difference = %f seconds" % (diff*86400))
        # Convert unixtime to date string
        utstring = timeUtilities.strftime('%Y'+delimiter+'%m'+delimiter+'%d %H:%M:%S UT',
                                          timeUtilities.gmtime(trialUnixTime))
    else:
        me = createCasaTool(metool)
        today = me.epoch('utc','today')
        mjd = np.array(mjdsec) / 86400.
        today['m0']['value'] =  mjd
        hhmmss = call_qa_time(today['m0'], prec=prec)
        myqa = createCasaTool(qatool)
        date = myqa.splitdate(today['m0'])
        myqa.done()
        utstring = "%s%s%02d%s%02d %s UT" % (date['year'],delimiter,date['month'],delimiter,
                                             date['monthday'],hhmmss)
    return(mjd, utstring)

#------------------------------------------------------------------------
def rad2radec(ra=0,dec=0,imfitdict=None, prec=5, verbose=True, component=0,
              replaceDecDotsWithColons=True, hmsdms=False, delimiter=', ',
              prependEquinox=False, hmdm=False):
    """
    Convert a position in RA/Dec from radians to sexagesimal string which
    is comma-delimited, e.g. '20:10:49.01, +057:17:44.806'.
    The position can either be entered as scalars via the 'ra' and 'dec'
    parameters, as a tuple via the 'ra' parameter, as an array of shape (2,1)
    via the 'ra' parameter, or
    as an imfit dictionary can be passed via the 'imfitdict' argument, and the
    position of component 0 will be displayed in RA/Dec sexagesimal.
    replaceDecDotsWithColons: replace dots with colons as the Declination d/m/s delimiter
    hmsdms: produce output of format: '20h10m49.01s, +057d17m44.806s'
    hmdm: produce output of format: '20h10m49.01, +057d17m44.806' (for simobserve)
    delimiter: the character to use to delimit the RA and Dec strings output
    prependEquinox: if True, insert "J2000" before coordinates (i.e. for clean or simobserve)
    Todd Hunter
    """
#    print "rad2radec: type(ra) = ", type(ra)
    if (type(imfitdict) == dict):
        comp = 'component%d' % (component)
        ra  = imfitdict['results'][comp]['shape']['direction']['m0']['value']
        dec = imfitdict['results'][comp]['shape']['direction']['m1']['value']
    if (type(ra) == tuple or type(ra) == list or type(ra) == np.ndarray):
        if (len(ra) == 2):
            dec = ra[1] # must come first before ra is redefined
            ra = ra[0]
        else:
            ra = ra[0]
            dec = dec[0]
    if (np.shape(ra) == (2,1)):
        dec = ra[1][0]
        ra = ra[0][0]
    if (not casaAvailable):
        if (ra<0): ra += 2*np.pi
        rahr = ra*12/np.pi
        decdeg = dec*180/np.pi
        hr = int(rahr)
        min = int((rahr-hr)*60)
        sec = (rahr-hr-min/60.)*3600
        if (decdeg < 0):
            mysign = '-'
        else:
            mysign = '+'
        decdeg = abs(decdeg)
        d = int(decdeg)
        dm = int((decdeg-d)*60)
        ds = (decdeg-d-dm/60.)*3600
        mystring = '%02d:%02d:%08.5f, %c%02d:%02d:%08.5f' % (hr,min,sec,mysign,d,dm,ds)
    else:
        myqa = createCasaTool(qatool)
        myra = myqa.formxxx('%.12frad'%ra,format='hms',prec=prec+1)
        mydec = myqa.formxxx('%.12frad'%dec,format='dms',prec=prec-1)
        if replaceDecDotsWithColons:
            mydec = mydec.replace('.',':',2)
        if (len(mydec.split(':')[0]) > 3):
            mydec = mydec[0] + mydec[2:]
        mystring = '%s, %s' % (myra, mydec)
        myqa.done()
    if (hmsdms):
        mystring = convertColonDelimitersToHMSDMS(mystring)
        if (prependEquinox):
            mystring = "J2000 " + mystring
    elif (hmdm):
        mystring = convertColonDelimitersToHMSDMS(mystring, s=False)
        if (prependEquinox):
            mystring = "J2000 " + mystring
    if (delimiter != ', '):
        mystring = mystring.replace(', ', delimiter)
    if (verbose):
        print(mystring)
    return(mystring)

#------------------------------------------------------------------------
def convertColonDelimitersToHMSDMS(mystring, s=True, usePeriodsForDeclination=False):
    """
    Converts HH:MM:SS.SSS, +DD:MM:SS.SSS  to  HHhMMmSS.SSSs, +DDdMMmSS.SSSs
          or HH:MM:SS.SSS +DD:MM:SS.SSS   to  HHhMMmSS.SSSs +DDdMMmSS.SSSs
          or HH:MM:SS.SSS, +DD:MM:SS.SSS  to  HHhMMmSS.SSSs, +DD.MM.SS.SSS
          or HH:MM:SS.SSS +DD:MM:SS.SSS   to  HHhMMmSS.SSSs +DD.MM.SS.SSS
    s: whether or not to include the trailing 's' in both axes
    -Todd Hunter
    """
    colons = len(mystring.split(':'))
    if (colons < 5 and (mystring.strip().find(' ')>0 or mystring.find(',')>0)):
        print("Insufficient number of colons (%d) to proceed (need 4)" % (colons-1))
        return
    if (usePeriodsForDeclination):
        decdeg = '.'
        decmin = '.'
        decsec = ''
    else:
        decdeg = 'd'
        decmin = 'm'
        decsec = 's'
    if (s):
        outstring = mystring.strip(' ').replace(':','h',1).replace(':','m',1).replace(',','s,',1).replace(':',decdeg,1).replace(':',decmin,1) + decsec
        if (',' not in mystring):
            outstring = outstring.replace(' ', 's ',1)
    else:
        outstring = mystring.strip(' ').replace(':','h',1).replace(':','m',1).replace(':',decdeg,1).replace(':',decmin,1)
    return(outstring)

#------------------------------------------------------------------------
def radec2deg(radecstring):
    """
    Convert a position from a single RA/Dec sexagesimal string to RA and
    Dec in degrees. The string can be either comma or space delimited, or
    delimited by h,m,s/d,m,s.
    The Dec portion of the string can be either : or . delimited.
    See also deg2radec and radec2rad.
    -Todd Hunter
    """
    if (radecstring.find('h')>0 and radecstring.find('d')>0):
        radecstring = radecstring.replace('h',':').replace('m',':').replace('s',' ').replace('d',':')
    myrad = radec2rad(radecstring)
    return(list(np.array(myrad)*180/np.pi))

#------------------------------------------------------------------------
def radecOffsetToRadec(radec, rao, deco, replaceDecDotsWithColons=True,
                       mas=False, prec=5, useEulerAngles=True, verbose=True,
                       delimiter=', ',hms=False):
    """
    Converts an absolute J2000 position and offset into a new absolute position.
    radec: either a string ('hh:mm:ss.s dd:mm:ss') or a tuple of radians
    rao, deco: offset in arcseconds (floating point values)
    replaceDotsWithColons: change CASA's Dec string from xx.yy.zz to xx:yy:zz
    mas: if True, then treat rao and deco as milliarcsec
    prec: desired number of digits of precision after the seconds decimal point
    delimiter: string to use between RA and Dec
    useEulerAngles: if True, then use the expression of the control system;
      if False, then use the small-angle formula:   RAnew = RA + deltaRA/cos(Dec)
    hms: if True, then output HHhMMmSS.Ss +DDdMMmSS.Ss instead of using colons
    Returns: a sexagesimal string
    """
    if (type(radec) == str or type(radec) == np.string_):
        rarad, decrad = radec2rad(radec)
    elif (type(radec) == list):
        if radec.find('J2000')==0:
            radec = radec.replace('J2000','')
        rarad, decrad = radec
    else:
        print("type(radec) = ", type(radec))
        print("radec must be either a string ('hh:mm:ss.s dd:mm:ss') or a list in radians [ra,dec]")
        return
    if (mas):
        rao *= 0.001
        deco *= 0.001
    rao = np.radians(rao/3600.)
    deco = np.radians(deco/3600.)
    if useEulerAngles:
        Pl = [np.cos(rao)*np.cos(deco), np.sin(rao)*np.cos(deco), np.sin(deco)]
        Ps = rotationEuler(Pl, rarad, decrad)
        newrarad = np.arctan2(Ps[1], Ps[0])
        if (newrarad < 0): newrarad += 2*np.pi
        newdecrad = np.arcsin(Ps[2])
    else:
        newrarad = rarad + rao/np.cos(decrad)
        newdecrad = decrad + deco
    mystring = rad2radec(newrarad, newdecrad, replaceDecDotsWithColons=replaceDecDotsWithColons,
                         prec=prec, verbose=verbose, delimiter=delimiter, hmsdms=hms)
    return(mystring)

#------------------------------------------------------------------------
def radec2rad(radecstring, returnList=False):
    """
    Convert a position from a single RA/Dec sexagesimal string to RA and
    Dec in radians.
    radecstring: any leading 'J2000' string is removed before consideration
    The RA and Dec portions can be separated by a comma or a space.
    The RA portion of the string must be colon-delimited, space-delimited,
        or 'h/m/s' delimited.
    The Dec portion of the string can be either ":", "." or space-delimited.
    If it is "." delimited, then it must have degrees, minutes, *and* seconds.
    Returns: a tuple
    returnList: if True, then return a list of length 2
    See also rad2radec.
    -Todd Hunter
    """
    if radecstring.find('J2000')==0:
        radecstring = radecstring.replace('J2000','')
    if (radecstring.find('h')>0 and radecstring.find('d')>0):
        radecstring = radecstring.replace('h',':').replace('m',':').replace('d',':').replace('s','')
    radec1 = radecstring.replace(',',' ')
    tokens = radec1.split()
    if (len(tokens) == 2):
        (ra,dec) = radec1.split()
    elif (len(tokens) == 6):
        h,m,s,d,dm,ds = radec1.split()
        ra = '%s:%s:%s' % (h,m,s)
        dec = '%+f:%s:%s' % (float(d), dm, ds)
    else:
        print("Invalid format for RA/Dec string")
        return
    tokens = ra.strip().split(':')
    hours = 0
    for i,t in enumerate(tokens):
        hours += float(t)/(60.**i)
    if (dec.find(':') > 0):
        tokens = dec.lstrip().split(':')
    elif (dec.find('.') > 0):
        try:
            (d,m,s) = dec.lstrip().split('.')
        except:
            (d,m,s,sfraction) = dec.lstrip().split('.')
            s = s + '.' + sfraction
        tokens = [d,m,s]
    else:  # just an integer
        tokens = [dec]
    dec1 = 0
    for i,t in enumerate(tokens):
        dec1 += abs(float(t)/(60.**i))
    if (dec.lstrip().find('-') == 0):
        dec1 = -dec1
    decrad = dec1*np.pi/180.
    ra1 = hours*15
    rarad = ra1*np.pi/180.
    if returnList:
        return [rarad,decrad]
    else:
        return(rarad,decrad)

#------------------------------------------------------------------------
def rotationEuler(Pl, rLong, rLat):
    Ps = [np.cos(rLong)*np.cos(rLat)*Pl[0] - np.sin(rLong)*Pl[1] - np.cos(rLong)*np.sin(rLat)*Pl[2],
          np.sin(rLong)*np.cos(rLat)*Pl[0] + np.cos(rLong)*Pl[1] - np.sin(rLong)*np.sin(rLat)*Pl[2],
          np.sin(rLat)*Pl[0]                                     + np.cos(rLat)*Pl[2]]
    return(Ps)

#------------------------------------------------------------------------
def deg2radec(ra=0, dec=None, prec=5, verbose=True, hmsdms=False, delimiter=', ',
              filename=''):
    """
    Convert a position (or list of positions) in RA/Dec from degrees
    to sexagesimal string. See also hours2radec.
    -Todd Hunter
    Inputs:
    ra: RA in degrees (float or string), or a tuple of RA and Dec in degrees,
        or an array of RAs in degrees. If a string, 'deg' will be stripped.
        Can also be a space-delimited string of 'ra dec' if dec=None
    dec: Dec in degrees (float or string), or an array of Decs in degrees
    prec: number of digits after the secondsdecimal
    hmsdms: if True, then output format is HHhMMmSS.SSSs, +DDdMMmSS.SSSs
    filename: if specified, then treat ra and dec as column numbers (starting
       at zero) from which to read the ra and dec values.
    Output:
    default format is HH:MM:SS.SSSSS, +DD:MM:SS.SSSS
    """
    if type(ra) == str and dec is None:
        ra,dec = [float(i) for i in ra.split()]
    if (ra==0 and dec is None and filename==''):
        print("You must specify either ra and dec, or a filename")
        return
    elif (filename != ''):
        f = open(filename)
        lines = f.readlines()
        if (ra==0 and dec is None):
            dec = 1
        ralist = []
        declist = []
        for line in lines:
            token = line.split()
            ralist.append(np.double(token[ra]))
            declist.append(np.double(token[dec]))
        ra = np.array(ralist)
        dec = np.array(declist)
    if (type(ra) == np.ndarray and type(dec) == np.ndarray):
        radec = []
        for i in range(len(ra)):
            radec.append(rad2radec(np.double(ra[i])*np.pi/180., np.double(dec[i])*np.pi/180., None, prec,
                                 verbose, hmsdms=hmsdms, delimiter=delimiter))
    else:
        if (type(ra) == tuple or type(ra) == list):
            dec = ra[1]
            ra = ra[0]
        if type(ra) == str:
            ra = ra.strip('degree')
        if type(dec) == str:
            dec = dec.strip('degree')
        radec = rad2radec(np.double(ra)*np.pi/180., np.double(dec)*np.pi/180., None, prec, verbose, hmsdms=hmsdms, delimiter=delimiter)
    return(radec)

#------------------------------------------------------------------------
#============Utilities for solar data reduction=====================================
def sol_sqld_mean_2(myms, mymsmd, scan_id, spw_id, sub_scan_id, ant_names,avss=False):

    """
    Returns dictionary with mean amplitudes (using the 'real' complex value) in
    both correlations, for a set of subscans sub_scan_id or one Scan, by
    antenna.

    :param myms: an ms object created with aU.createCasaTool(mstool)
    :param mymsmd: an msmd objecte created with aU.createCasaTool(msmdtool)
    :param scan_id: the scan to be analysed. It has to be one integer.
    :param spw_id: a python list of 4 string representing spw ids.
    :param sub_scan_id: a python list of integers, one for each subscan.
    :param ant_names: a python list of strings, each string a valid antenna name
        (CMXX, PMXX, DAXX, DVXX)
    :param avss: average over all Subscans. Default False.
    :return:
    """

    if len(spw_id) != 4:
        print("Script requires 4 SPWs")
        return None

    if not (isinstance(scan_id, np.int32) or isinstance(scan_id, np.int64) or isinstance(scan_id, int)):
        print("Only one scan accepted as input, and it should be an integer")
        return None

    sub_inf = computeDurationOfScan(
        scan_id, mymsmd=mymsmd, returnSubscanTimes=True)

    if len(sub_scan_id) > sub_inf[1]:
        print("This scan doesn't have so many subscans...")
        return None

    if len(sub_scan_id) == 1:
        try:
            tag_per = sub_inf[3][sub_scan_id[0]]
        except KeyError:
            print("Subscan %s doesnt exist or doesn't have data." %
                  sub_scan_id[0])
    else:
        timeran = []
        avsub = []
        for key, value in sub_inf[3].items():
            if key in sub_scan_id:
                timeran.append(value)
                avsub.append(key)
        tag_per = ','.join(timeran)

        for s in sub_scan_id:
            if s not in avsub:
                print("Subscan %s doesn't exist or doesn't have data" % s)
                return None

    pow_dict = {}

    if avss:
        for i in range(0, len(ant_names)):
            print('Now proccess: % s.' % ant_names[i])
            try:
                s_xx = myms.statistics(
                    column='data', spw=','.join(spw_id), correlation='XX',
                    baseline=str(i) + '&&&', complex_value='real', time=tag_per,
                    reportingaxes='ddid')
                s_yy = myms.statistics(
                    column='data', spw=','.join(spw_id), correlation='YY',
                    baseline=str(i) + '&&&', complex_value='real', time=tag_per,
                    reportingaxes='ddid')
            except RuntimeError:
                print(
                    "\nSomething failed in the selection. Maybe a subscan in "
                    "the list doesn't have valid data? (Error: %s)")
                return None

            pow_dict[ant_names[i]] = {}
            xx = []
            yy = []
            for sp in spw_id:
                xx.append(s_xx['DATA_DESC_ID=%s' % sp]['mean'])
                yy.append(s_yy['DATA_DESC_ID=%s' % sp]['mean'])
                pow_dict[ant_names[i]]['XX'] = xx
                pow_dict[ant_names[i]]['YY'] = yy

        return pow_dict

    for i in range(0, len(ant_names)):
        print('Now proccess: % s.' % ant_names[i])
        try:
            s_xx = myms.statistics(
                column='data', spw=','.join(spw_id), correlation='XX',
                baseline=str(i) + '&&&', complex_value='real', time=tag_per,
                timeaverage=True, timebin='100s', reportingaxes='ddid')
            s_yy = myms.statistics(
                column='data', spw=','.join(spw_id), correlation='YY',
                baseline=str(i) + '&&&', complex_value='real', time=tag_per,
                timeaverage=True, timebin='100s', reportingaxes='ddid')
        except RuntimeError:
            print("\nSomething failed in the selection. Maybe a subscan in the "
                  "list doesn't have valid data? (Error: %s)")
            return None

        stateids = []
        for k in list(s_xx.keys()):
            s_xx[k.split(',TIM')[0]] = s_xx.pop(k)
            s_yy[k.split(',TIM')[0]] = s_yy.pop(k)
            if int(k.split(',')[2].replace('STATE_ID=', '')) not in stateids:
                stateids.append(int(k.split(',')[2].replace('STATE_ID=', '')))
        stateids.sort()

        for s, ss in enumerate(sub_scan_id):
            sids = stateids[s]
            if ss not in pow_dict.keys():
                pow_dict[ss] = {}
            pow_dict[ss][ant_names[i]] = {}
            xx = []
            yy = []
            for sp in spw_id:
                xx.append(s_xx['DATA_DESC_ID=%s,SCAN_NUMBER=%s,STATE_ID=%s' %
                               (sp, scan_id, sids)]['mean'])
                yy.append(s_yy['DATA_DESC_ID=%s,SCAN_NUMBER=%s,STATE_ID=%s' %
                               (sp, scan_id, sids)]['mean'])

            pow_dict[ss][ant_names[i]]['XX'] = xx
            pow_dict[ss][ant_names[i]]['YY'] = yy

    return pow_dict



#--------------------------
def sol_Tant(P_sky, P_amb, P_hot, P_off, P_sun, P_zer, TLoad, skyScan):
    Tant = {}
    antNames = list(P_sky.keys())
    pols = list(P_sky[antNames[0]].keys())

    for ant in antNames:
        for pol in pols:
            tantt = []
            for spw in range(0,4):

                P_skyt = P_sky[ant][pol][spw]
                P_ambt = P_amb[ant][pol][spw]
                P_hott = P_hot[ant][pol][spw]
                P_offt = P_off[ant][pol][spw]
                P_sunt = P_sun[ant][pol][spw]
                P_zert = P_zer[ant][pol][spw]
                T_ambt = TLoad[ant][skyScan]['amb']
                T_hott = TLoad[ant][skyScan]['hot']

                tantt.append((P_sunt - P_offt) * (P_skyt - P_zert) * (T_hott - T_ambt) / ((P_offt - P_zert) * (P_hott-P_ambt)))

                if pol == pols[0]:
                    Tant[ant] = {'XX': tantt,'YY': tantt}
                else:
                    Tant[ant]['YY'] = tantt


    return Tant

#--------------------------
def sol_gentats_2(vis, orgtbl, Tant, skyScan, sciScan, spw, tgtSubScan,mymsmd,atmScan):

    modtbl = orgtbl+'tant_'+format(sciScan,'03d')+'_'+format(tgtSubScan, '03d')
    os.system('cp -r ' +  orgtbl + ' ' + modtbl)

#    antNames = Tant.keys()
    antNames = getAntennaNames(modtbl)
    pols = list(Tant[antNames[0]].keys())

    tb.open(modtbl, nomodify = False)
    for i in range(0, len(antNames)):
        for j in range(0, len(spw)):

            #print 'Now proccess: % s.' % ant_names[i]

            sel = tb.query('ANTENNA1==%d && SCAN_NUMBER==%d && SPECTRAL_WINDOW_ID==%d' % (i, skyScan, spw[j]))
            try:
                tsyst = sel.getcol("FPARAM")
                tcomt = tsyst
                tsflag = sel.getcol("FLAG")

                for pol in range(0, tsyst.shape[0]):

                    Sun_Tant = Tant[antNames[i]][pols[abs(pol-1)]][j]

                    for ch in range(0, tsyst.shape[1]):
                        if tsflag[pol, ch, 0] == False:
                            tcomt[pol, ch, 0] = tsyst[pol, ch, 0] + Sun_Tant
                            sel.putcol("FPARAM", tcomt)
            finally:
                sel.close()

    tb.close()

    #Remove scan for P_zero and P_off
    tb.open(modtbl, nomodify = False)
    for i in range(0, len(atmScan)):
        if atmScan[i] == skyScan:
            print(atmScan[i])
        else:
            sel =  tb.query('SCAN_NUMBER==%d' % atmScan[i])
            try:
                zrows = sel.rownumbers()
                tb.removerows(zrows)
            finally:
                sel.close()
    tb.close()

    return modtbl
#--------------------------
def sol_ampcal_2(vis, orgtbl, anttbl, exisTbl, outCSV):

# execfile('SunRedUtil-AH.py')
#vis='uid___A002_Xbbc24a_X16c.ms'
#%time sol_ampcal(vis, vis + '.tsys', exisTbl=F, outCSV=T)
#%time sol_ampcal_2(vis, vis + '.tsys', exisTbl=F, outCSV=T)


    antNames = getAntennaNames(vis)
#    antNames=['CM01']


    if exisTbl == False:

        myms = createCasaTool(mstool)
        myms.open(vis)
        mymsmd = createCasaTool(msmdtool)
        mymsmd.open(vis)
#        t = time.time()
        asdm=vis.split('.ms')[0]
        os.system('rm -rf ' + vis + '.tsystant_*')
        zeroScan = 1 #HC of Scan for P_zero
#        print time.time() - t
        print('### Process of P_zero ###')
        #P_zer = sol_sqld_mean(vis, zeroScan, ['13','14','15','16'], [2]) #HC spw for P_zero

        P = sol_sqld_mean_2(myms, mymsmd, zeroScan, ['13', '14', '15','16'], [2], antNames)
        P_zer = P[2]
#        print time.time() - t
        print('### Process of T_load ###')
        TLoad = getLoadTemperatures(vis, doplot = False, warnIfNoLoadTemperatures = True)
#        print time.time() - t
        sciScan = mymsmd.scansforintent('*OBSERVE_TARGET*')
        atmScan = mymsmd.scansforintent('*CALIBRATE_ATMOSPHERE*')


        lastSkyScan=0
        for i in range(0, len(sciScan)):
#        for i in range(0, 1):
            print("processing Scan ", sciScan[i])
#            print time.time() - t
#            print('### aU.computeDurationOfScan  ###')
            subInf = computeDurationOfScan(sciScan[i], mymsmd=mymsmd, returnSubscanTimes=True)
            subNum = subInf[1]

            # AH: This assumes the previous scan was an ATM cal, which is NOT always the case
            skyScan = sciScan[i] - 1   ###HC skyScan#
            if skyScan in atmScan:
                print("This science scan "+ str(sciScan[i]) + " is preceeded by an ATM scan "+ str(skyScan))
                print("using it to compute P_sky, P_amb, P_hot ")
                lastSkyScan=skyScan
            else :
                print("This science scan "+ str(sciScan[i]) + "is NOT preceeded by an ATM scan ")
                print("Using last recorded ATM Scan instead "+ str(lastSkyScan))
                skyScan=lastSkyScan
               # when the previous scan was NOT a tscan, use the one before



            print("Processing P_sky, P_amb, P_hot using "+ str(skyScan))
            #print '### Process of P_sky of Scan:' + format(skyScan,'03d') +'###'
            #P_sky = sol_sqld_mean(vis, skyScan, ['0','1','2','3'], [1]) #HC spw subscanID
            #print '### Process of P_amb of Scan:' + format(skyScan,'03d') +'###'
            #P_amb = sol_sqld_mean(vis, skyScan, ['0','1','2','3'], [2]) #HC spw subscanID
            #print '### Process of P_hot of Scan:' + format(skyScan,'03d') +'###'
            #P_hot = sol_sqld_mean(vis, skyScan, ['0','1','2','3'], [3]) #HC spw subscanID
#            print time.time() - t
            # With new implementation get all 3 subscans
            Ps = sol_sqld_mean_2(myms, mymsmd, skyScan, ['0', '1', '2','3'], [1,2,3], antNames)
            P_sky = Ps[1]
            P_amb = Ps[2]
            P_hot = Ps[3]

            print('### Process of P_off of Scan:' + format(sciScan[i],'03d') +'###')
            #P_off = sol_sqld_mean(vis, sciScan[i], ['0','1','2','3'], [1, 2, subNum-1, subNum]) #HC spw subscanID
#            print time.time() - t
            P_off = sol_sqld_mean_2(myms, mymsmd, sciScan[i], ['0', '1', '2','3'], [1,subNum], antNames, avss=True)
#            print(P_off)


            # compute for all Solar Sub-scans in one line
            # print "compute for all Solar Sub-scans in one line "
            #print time.time() - t
            print('### Process of P_sun of Scan:' + format(sciScan[i],'03d') +'###')
            PSUN = sol_sqld_mean_2(myms, mymsmd, sciScan[i], ['0', '1', '2','3'], range(2,subNum), antNames)

            #print subNum

            for j in range(2, subNum):

                print('### Tsys+Tant Cal of Scan:' + format(sciScan[i],'03d') +' SubScan:'+format(j,'03d')+' ###')

                subs =[]
                subs.append(j)
#                print(subs)
                #P_sun = sol_sqld_mean(vis, sciScan[i], ['0','1','2','3'], subs) #HC spw
                P_sun = PSUN[j] #
#                print "sot_Tant "
#                print time.time() - t
                T_ant = sol_Tant(P_sky, P_amb, P_hot, P_off, P_sun, P_zer, TLoad, skyScan) ###HC spw
#                print "new sol_gentats "
#                print time.time() - t

                modtbl = sol_gentats_2(vis, orgtbl, T_ant, skyScan, sciScan[i], [5, 7, 9, 11], j,mymsmd,atmScan)

        ###HC spw
#                print "runing apply cal \n "
#                print time.time() - t
                apptr = qa.time(qa.quantity(subInf[2][j][0]-0.2,'s'), form='hms', prec=8)[0]+ '~' + qa.time(qa.quantity(subInf[2][j][1]+0.2,'s'), form='hms', prec=8)[0]
#                print apptr
                applycal(vis = vis, scan = str(sciScan[i]), spw = '5,7,9,11',
#                         gaintable = modtbl, timerange = subInf[3][j],
                         gaintable = [modtbl, anttbl], timerange = apptr,
                         interp = 'linear', calwt = True, flagbackup = False)

                if outCSV == True:
#                    print "outCSV = True \n"
#                    print time.time() - t
                    csv_file = asdm+'_Tsys+Tant_'+format(sciScan[i],'03d')+'_'+format(j, '03d')+'.csv'
                    os.system('rm -rf ' + csv_file)
                    csv_table =[]
                    csv_tcolumn = ['Ant',
                                   'BB1/Tant:XX','BB1/Tant:YY',
                                   'BB2/Tant:XX','BB2/Tant:YY',
                                   'BB3/Tant:XX','BB3/Tant:YY',
                                   'BB4/Tant:XX','BB4/Tant:YY',
                                   'BB1/Pzero:XX','BB1/Pzero:YY',
                                   'BB2/Pzero:XX','BB2/Pzero:YY',
                                   'BB3/Pzero:XX','BB3/Pzero:YY',
                                   'BB4/Pzero:XX','BB4/Pzero:YY',
                                   'BB1/Psun:XX','BB1/Psun:YY',
                                   'BB2/Psun:XX','BB2/Psun:YY',
                                   'BB3/Psun:XX','BB3/Psun:YY',
                                   'BB4/Psun:XX','BB4/Psun:YY',
                                   'BB1/Poff:XX','BB1/Poff:YY',
                                   'BB2/Poff:XX','BB2/Poff:YY',
                                   'BB3/Poff:XX','BB3/Poff:YY',
                                   'BB4/Poff:XX','BB4/Poff:YY',
                                   'BB1/Psky:XX','BB1/Psky:YY',
                                   'BB2/Psky:XX','BB2/Psky:YY',
                                   'BB3/Psky:XX','BB3/Psky:YY',
                                   'BB4/Psky:XX','BB4/Psky:YY',
                                   'BB1/Phot:XX','BB1/Phot:YY',
                                   'BB2/Phot:XX','BB2/Phot:YY',
                                   'BB3/Phot:XX','BB3/Phot:YY',
                                   'BB4/Phot:XX','BB4/Phot:YY',
                                   'BB1/Pamb:XX','BB1/Pamb:YY',
                                   'BB2/Pamb:XX','BB2/Pamb:YY',
                                   'BB3/Pamb:XX','BB3/Pamb:YY',
                                   'BB4/Pamb:XX','BB4/Pamb:YY']


                    csv_table.append(csv_tcolumn)

                    for ant in antNames:
                        csv_column = []
                        csv_column.append(ant)
                        csv_column.append(T_ant[ant]['XX'][0])
                        csv_column.append(T_ant[ant]['YY'][0])
                        csv_column.append(T_ant[ant]['XX'][1])
                        csv_column.append(T_ant[ant]['YY'][1])
                        csv_column.append(T_ant[ant]['XX'][2])
                        csv_column.append(T_ant[ant]['YY'][2])
                        csv_column.append(T_ant[ant]['XX'][3])
                        csv_column.append(T_ant[ant]['YY'][3])

                        csv_column.append(P_zer[ant]['XX'][0])
                        csv_column.append(P_zer[ant]['YY'][0])
                        csv_column.append(P_zer[ant]['XX'][1])
                        csv_column.append(P_zer[ant]['YY'][1])
                        csv_column.append(P_zer[ant]['XX'][2])
                        csv_column.append(P_zer[ant]['YY'][2])
                        csv_column.append(P_zer[ant]['XX'][3])
                        csv_column.append(P_zer[ant]['YY'][3])

                        csv_column.append(P_sun[ant]['XX'][0])
                        csv_column.append(P_sun[ant]['YY'][0])
                        csv_column.append(P_sun[ant]['XX'][1])
                        csv_column.append(P_sun[ant]['YY'][1])
                        csv_column.append(P_sun[ant]['XX'][2])
                        csv_column.append(P_sun[ant]['YY'][2])
                        csv_column.append(P_sun[ant]['XX'][3])
                        csv_column.append(P_sun[ant]['YY'][3])

                        csv_column.append(P_off[ant]['XX'][0])
                        csv_column.append(P_off[ant]['YY'][0])
                        csv_column.append(P_off[ant]['XX'][1])
                        csv_column.append(P_off[ant]['YY'][1])
                        csv_column.append(P_off[ant]['XX'][2])
                        csv_column.append(P_off[ant]['YY'][2])
                        csv_column.append(P_off[ant]['XX'][3])
                        csv_column.append(P_off[ant]['YY'][3])

                        csv_column.append(P_sky[ant]['XX'][0])
                        csv_column.append(P_sky[ant]['YY'][0])
                        csv_column.append(P_sky[ant]['XX'][1])
                        csv_column.append(P_sky[ant]['YY'][1])
                        csv_column.append(P_sky[ant]['XX'][2])
                        csv_column.append(P_sky[ant]['YY'][2])
                        csv_column.append(P_sky[ant]['XX'][3])
                        csv_column.append(P_sky[ant]['YY'][3])

                        csv_column.append(P_hot[ant]['XX'][0])
                        csv_column.append(P_hot[ant]['YY'][0])
                        csv_column.append(P_hot[ant]['XX'][1])
                        csv_column.append(P_hot[ant]['YY'][1])
                        csv_column.append(P_hot[ant]['XX'][2])
                        csv_column.append(P_hot[ant]['YY'][2])
                        csv_column.append(P_hot[ant]['XX'][3])
                        csv_column.append(P_hot[ant]['YY'][3])

                        csv_column.append(P_amb[ant]['XX'][0])
                        csv_column.append(P_amb[ant]['YY'][0])
                        csv_column.append(P_amb[ant]['XX'][1])
                        csv_column.append(P_amb[ant]['YY'][1])
                        csv_column.append(P_amb[ant]['XX'][2])
                        csv_column.append(P_amb[ant]['YY'][2])
                        csv_column.append(P_amb[ant]['XX'][3])
                        csv_column.append(P_amb[ant]['YY'][3])


                        csv_table.append(csv_column)
#                    print "write CSV \n"
                    with open (csv_file, 'w') as f:
                        tantsys = csv.writer(f)
                        tantsys.writerows(csv_table)

    if exisTbl == True:

        mmmsmd = createCasaTool(msmdtool)
        mmmsmd.open(vis)
        sciScan = mmmsmd.scansforintent('*OBSERVE_TARGET*')
        mmmsmd.done()

        for i in range(0, len(sciScan)):
            subInf = computeDurationOfScan(sciScan[i], vis=vis, returnSubscanTimes=True)
            subNum = subInf[1]
            skyScan = sciScan[i] - 1   ###HC skyScan#

            for j in range(2, subNum):

                print('### Tsys+Tant Cal of Scan:' + format(sciScan[i],'03d') +' SubScan:'+format(j,'03d')+' ###')
                modtbl = orgtbl+'tant_'+format(sciScan[i],'03d')+'_'+format(j, '03d')
                apptr = qa.time(qa.quantity(subInf[2][j][0]-0.2,'s'), form='hms', prec=8)[0]+ '~' + qa.time(qa.quantity(subInf[2][j][1]+0.2,'s'), form='hms', prec=8)[0]
                applycal(vis = vis, scan = str(sciScan[i]), spw = '5,7,9,11', 
                         gaintable = [modtbl, anttbl],
                         timerange = apptr,
                         interp = 'linear,linear', calwt = True, flagbackup = False)

#--------------------------
def sol_calflux(vis):

    # To get ID and Name of Amp Cal (for query the flux later)
    intentSources=es.getIntentsAndSourceNames(vis)
    ampCalId = intentSources['CALIBRATE_BANDPASS']['id']
#    ampCalId = intentSources['CALIBRATE_PHASE']['id']
    calFieldNames = intentSources['CALIBRATE_BANDPASS']['name']
#    calFieldNames = intentSources['CALIBRATE_PHASE']['name']
    amp_cal_name=calFieldNames[0]
    print('Bandpass/Flux Calibrator: '+amp_cal_name)

    # To get observing frequency (for query the flux later)
    spwInfo=es.getSpwInfo(vis)
    for i in sorted(spwInfo.keys()):
        print(i, spwInfo[i]['refFreq']/1e9)
    obs_freq="%fGHz"%(spwInfo[0]['refFreq']/1e9)
    print('Observing Frequencies: '+obs_freq)

    # To get observation date (for query the flux later)
    date=aU.getObservationStartDate(vis)
    date_obs=date.split()[0]
    spw1_flux=aU.getALMAFlux(sourcename=amp_cal_name, date=date_obs,frequency=obs_freq)
    spw1_flux['fluxDensity']
    print(spw1_flux['fluxDensity'])
    print(spw1_flux['spectralIndex'])

#--------------------------
def sol_tsunplot(uid):

    filnms = glob.glob(uid+'*.csv')
    outdir='./'+uid+'.ms.tant.plots'

    antnames =[]
    f = open(filnms[0], 'r')
    cdat = csv.reader(f)
    header = next(cdat)
    for row in cdat:
        antnames.append(row[0])
    f.close()

    ssnm = []
    tsun=np.zeros([len(antnames),8,len(filnms)])
    for i in range(0,len(filnms)):
        ssnm.append(filnms[i][len(uid)+11:len(uid)+18])
        f = open(filnms[i], 'r')
        cdat = csv.reader(f)
        header = next(cdat)
        rpnt = 0
        for row in cdat:
            tsun[rpnt,0,i] = row[1] #SPW1 XX
            tsun[rpnt,1,i] = row[2] #SPW1 YY
            tsun[rpnt,2,i] = row[3] #SPW2 XX
            tsun[rpnt,3,i] = row[4] #SPW2 YY
            tsun[rpnt,4,i] = row[5] #SPW3 XX
            tsun[rpnt,5,i] = row[6] #SPW3 YY
            tsun[rpnt,6,i] = row[7] #SPW4 XX
            tsun[rpnt,7,i] = row[8] #SPW4 YY
            rpnt = rpnt + 1
        f.close()

    os.system('rm -rf ' + outdir)
    os.system('mkdir ' + outdir)

    for i in range(0, len(antnames)):
        plt.subplot(111)
        plt.plot(tsun[i][0],color='blue', linestyle='-', label='SPW1 XX')
        plt.plot(tsun[i][1],color='blue', linestyle='--', label='SPW1 YY')
        plt.plot(tsun[i][2],color='Green', linestyle='-', label='SPW2 XX')
        plt.plot(tsun[i][3],color='Green', linestyle='--', label='SPW2 YY')
        plt.plot(tsun[i][4],color='Orange', linestyle='-', label='SPW3 XX')
        plt.plot(tsun[i][5],color='Orange', linestyle='--', label='SPW3 YY')
        plt.plot(tsun[i][6],color='Red', linestyle='-', label='SPW4 XX')
        plt.plot(tsun[i][7],color='Red', linestyle='--', label='SPW4 YY')
        plt.xticks(range(0,len(filnms)), ssnm)
        plt.legend(loc=0)
        plt.title(uid+' -- '+antnames[i]+' / T_sun')
        plt.xlabel('[Scan#_SubScan#]')
        plt.ylabel('Antenna Temperature [K]')
        plt.show()
        savefile = outdir+'/'+uid+'_'+antnames[i]+"_Tsun.png"
        plt.savefig(savefile)
        plt.close()
