#!/usr/bin/env python3

import subprocess
import json


base_url = f"http://zaiko/v1"
# base_url = f"http://localhost:3030/v1"
# base_url = f"http://35.78.179.182:80/v1"

curl_commands = [
    {
        "command": f"curl -X DELETE {base_url}/stocks",
        "expected_output": ""
    },
    {
        "command": f"curl -d '{{\"name\": \"xxx\", \"amount\": 100}}' -H 'Content-Type: application/json' {base_url}/stocks",
        "expected_output": '{"name":"xxx","amount":100}'
    },
    {
        "command": f"curl -d '{{\"name\": \"xxx\", \"amount\": 4}}' -H 'Content-Type: application/json' {base_url}/sales",
        "expected_output": '{"name":"xxx","amount":4}'
    },
    {
        "command": f"curl {base_url}/stocks",
        "expected_output": '{"xxx":96}'
    },
    {
        "command": f"curl -d '{{\"name\": \"yyy\", \"amount\": 100}}' -H 'Content-Type: application/json' {base_url}/stocks",
        "expected_output": '{"name":"yyy","amount":100}'
    },
    {
        "command": f"curl -d '{{\"name\": \"YYY\", \"amount\": 100}}' -H 'Content-Type: application/json' {base_url}/stocks",
        "expected_output": '{"name":"YYY","amount":100}'
    },
    {
        "command": f"curl {base_url}/stocks",
        "expected_output": '{"YYY":100,"xxx":96,"yyy":100}'
    },
    {
        "command": f"curl -X DELETE {base_url}/stocks",
        "expected_output": ""
    },
    {
        "command": f"curl -d '{{\"name\": \"xxx\", \"amount\": 1.1}}' -H 'Content-Type: application/json' {base_url}/stocks",
        "expected_output": '{"message":"ERROR"}'
    },
    {
        "command": f"curl -X DELETE {base_url}/stocks",
        "expected_output": ""
    },
    {
        "command": f"curl -d '{{\"name\": \"aaa\", \"amount\": 10}}' -H 'Content-Type: application/json' {base_url}/stocks",
        "expected_output": '{"name":"aaa","amount":10}'
    },
    {
        "command": f"curl -d '{{\"name\": \"bbb\", \"amount\": 10}}' -H 'Content-Type: application/json' {base_url}/stocks",
        "expected_output": '{"name":"bbb","amount":10}'
    },
    {
        "command": f"curl -d '{{\"name\": \"aaa\", \"amount\": 4, \"price\": 100}}' -H 'Content-Type: application/json' {base_url}/sales",
        "expected_output": '{"name":"aaa","amount":4,"price":100}'
    },
    {
        "command": f"curl -d '{{\"name\": \"aaa\", \"price\": 80}}' -H 'Content-Type: application/json' {base_url}/sales",
        "expected_output": '{"name":"aaa","price":80}'
    },
    {
        "command": f"curl {base_url}/sales",
        "expected_output": '{"sales":480.0}'
    }
]


def run_curl_commands(commands):
    for item in commands:
        command = item["command"]
        expected_output = item["expected_output"]

        process = subprocess.run(
            command, shell=True, capture_output=True, text=True)

        # 実行結果の出力
        actual_output = process.stdout.strip()  # 改行や余分な空白を除去

        # 結果を比較
        print(f"Command: {command}")
        print(f"Expected: {expected_output}")
        print(f"Actual: {actual_output}")

        # 期待される出力と実際の出力を比較して結果を表示
        if compare_outputs(actual_output, expected_output):
            print("Result: ✅ Match")
        else:
            print("Result: ❌ Mismatch")

        if process.stderr:
            print(f"Error: {process.stderr}")
        print("=" * 50)


def compare_outputs(actual, expected):
    try:
        actual_json = json.loads(actual)
        expected_json = json.loads(expected)
        return actual_json == expected_json
    except json.JSONDecodeError:
        # JSONでない場合は、文字列として比較
        return actual == expected


run_curl_commands(curl_commands)
