#!/bin/bash

rsync --archive --recursive --verbose -P --times --delete tct@tct-computer.physik.uzh.ch:/home/tct/measurements_data/* /home/alf/cernbox/projects/4D_sensors/TI-LGAD_FBK_RD50_1/measurements_data
