#!/usr/bin/env bash
set -u
mkdir -p /root/tmtools && cd /root/tmtools
echo "try USalign binary gz"
curl -fsSL -o USalign.gz "https://zhanggroup.org/US-align/bin/USalign.gz" && ls -lh USalign.gz && gzip -dk USalign.gz && chmod +x USalign && ./USalign 2>&1 | head -n 5 && echo USALIGN_OK && exit 0
echo "USalign.gz failed rc=$?"
curl -fsSL -o USalign.cpp "https://zhanggroup.org/US-align/bin/module/USalign.cpp" && ls -lh USalign.cpp && head -n 3 USalign.cpp && g++ -O3 -o USalign USalign.cpp 2>&1 | tail -5 && chmod +x USalign && ./USalign 2>&1 | head -n 5 && echo USALIGN_CPP_OK && exit 0
echo "USalign.cpp failed rc=$?"
curl -fsSL -o TMalign.cpp "https://zhanggroup.org/TM-align/TMalign.cpp" && ls -lh TMalign.cpp && head -n 3 TMalign.cpp && g++ -O3 -o TMalign TMalign.cpp 2>&1 | tail -5 && chmod +x TMalign && ./TMalign 2>&1 | head -n 5 && echo TMALIGN_OK && exit 0
echo "ALL_FAILED"
exit 1
