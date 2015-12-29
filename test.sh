#! /bin/bash
echo "do compile"
make lstm
echo "do run"
../../bin/singa-run.sh -exec examples/lstm/lstm.bin -conf examples/lstm/job.conf

