#!/bin/sh
/redpanda-connect run /stockconnector.yaml &
(/initkafka && /echoserver) &
wait
