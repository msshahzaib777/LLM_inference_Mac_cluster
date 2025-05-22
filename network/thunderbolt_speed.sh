#!/bin/bash

# CONFIG
REMOTE_USER="stu"
REMOTE_HOST="mac2"
REMOTE_PATH="/tmp/testfile_remote"
LOCAL_TEST_FILE="testfile_local"
FILE_SIZE_MB=10000

# Create a dummy file
echo "🛠️ Creating ${FILE_SIZE_MB}MB test file..."
dd if=/dev/zero of=$LOCAL_TEST_FILE bs=1M count=$FILE_SIZE_MB status=none

# Measure transfer time with scp
echo "🚀 Testing transfer speed to $REMOTE_HOST..."

START=$(date +%s.%N)
scp $LOCAL_TEST_FILE ${REMOTE_HOST}:${REMOTE_PATH}
END=$(date +%s.%N)

# Calculate duration
DURATION=$(echo "$END - $START" | bc)
SPEED=$(echo "$FILE_SIZE_MB / $DURATION" | bc -l)

# Report
echo "✅ Transfer completed in ${DURATION} seconds"
printf "⚡ Effective Speed: %.2f MB/s\n" "$SPEED"

# Cleanup
ssh ${REMOTE_HOST} "rm -f ${REMOTE_PATH}"
rm -f $LOCAL_TEST_FILE
