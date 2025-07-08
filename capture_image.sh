gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 \
! 'video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12' \
! nvvidconv \
! 'video/x-raw, format=I420' \
! jpegenc \
! filesink location=capture.jpg