/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <msckf_vio/msckf_vio_nodelet.h>

namespace msckf_vio {
void MsckfVioNodelet::onInit() {
  msckf_vio_ptr.reset(new MsckfVio(getPrivateNodeHandle()));
  if (!msckf_vio_ptr->initialize()) {
    ROS_ERROR("Cannot initialize MSCKF VIO...");
    return;
  }
  return;
}

PLUGINLIB_EXPORT_CLASS(msckf_vio::MsckfVioNodelet,
    nodelet::Nodelet);

} // end namespace msckf_vio

