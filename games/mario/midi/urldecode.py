#!/usr/bin/env python

import urllib
import sys
import os

safe_name = os.path.basename(sys.argv[1])

with open (safe_name, "r") as myfile:
    data=myfile.read()
    print "input data: " + data
    decoded = urllib.unquote(data)
    with open("%s.mid" % safe_name, "w") as text_file:
        text_file.write("%s" % (decoded))
