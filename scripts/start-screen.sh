#!/bin/bash

Xvfb :99 -screen 0 640x480x24 &
cd external && make && cd ..
/bin/bash
