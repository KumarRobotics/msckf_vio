/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_NODELET_H
#define MSCKF_VIO_NODELET_H

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <msckf_vio/msckf_vio.h>

namespace msckf_vio {
class MsckfVioNodelet : public nodelet::Nodelet {
public:
  MsckfVioNodelet() { return; }
  ~MsckfVioNodelet() { return; }

private:
  virtual void onInit();
  MsckfVioPtr msckf_vio_ptr;
};
} // end namespace msckf_vio

#endif

