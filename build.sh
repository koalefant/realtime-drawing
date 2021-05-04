#!/bin/bash

set -e
BUILD_ROOT=$(dirname $(readlink -f $0))

cd ${BUILD_ROOT} 
cargo build --example demo 
cargo doc
