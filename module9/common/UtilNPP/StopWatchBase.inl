/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


////////////////////////////////////////////////////////////////////////////////
//! Constructor, default
////////////////////////////////////////////////////////////////////////////////
template<class OSPolicy>
StopWatchBase<OSPolicy>::StopWatchBase() :
        OSPolicy() {}

////////////////////////////////////////////////////////////////////////////////
//! Destructor
////////////////////////////////////////////////////////////////////////////////
template<class OSPolicy>
StopWatchBase<OSPolicy>::~StopWatchBase() {}

///////////////////////////////////////////////////////////////////////////////
//! Start time measurement
////////////////////////////////////////////////////////////////////////////////
template<class OSPolicy>
inline void
StopWatchBase<OSPolicy>::start() {
    OSPolicy::start();
}

////////////////////////////////////////////////////////////////////////////////
//! Stop time measurement
////////////////////////////////////////////////////////////////////////////////
template<class OSPolicy>
inline void
StopWatchBase<OSPolicy>::stop() {
    OSPolicy::stop();
}


////////////////////////////////////////////////////////////////////////////////
//! Reset the timer to 0. Does not change the timer running state but does
//! recapture this point in time as the current start time if it is running.
////////////////////////////////////////////////////////////////////////////////
template<class OSPolicy>
inline void
StopWatchBase<OSPolicy>::reset() {
    OSPolicy::reset();
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. after start. If the stop watch is still running (i.e. there
//! was no call to stop()) then the elapsed time is returned, otherwise the
//! time between the last start() and stop call is returned
////////////////////////////////////////////////////////////////////////////////
template<class OSPolicy>
inline const double
StopWatchBase<OSPolicy>::elapsed() const {
    return OSPolicy::elapsed();
}
