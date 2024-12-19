#!/usr/bin/env python3

import subprocess
import json
import random

# base_url = f"http://localhost:3030/v1"
base_url = f"http://zaiko/v1"


class TestRunner:
    def __init__(self):
        self.stocks = {}  # key: name, value: amount (int)
        self.total_sales = 0.0  # total sales value (float)

    def reset(self):
        self.stocks = {}
        self.total_sales = 0.0

    def run(self, num_tests=100):
        for _ in range(num_tests):
            operation = self.choose_operation()
            if operation == 'DELETE_STOCKS':
                self.delete_stocks()
            elif operation == 'ADD_STOCK':
                self.add_stock()
            elif operation == 'MAKE_SALE':
                self.make_sale()
            elif operation == 'GET_STOCKS':
                self.get_stocks()
            elif operation == 'GET_SALES':
                self.get_sales()
            else:
                pass  # Unknown operation

    def choose_operation(self):
        # Weighted choice to ensure a mix of operations
        operations = ['DELETE_STOCKS', 'ADD_STOCK',
                      'MAKE_SALE', 'GET_STOCKS', 'GET_SALES']
        weights = [0.05, 0.4, 0.4, 0.1, 0.05]  # Adjust weights as needed
        return random.choices(operations, weights=weights, k=1)[0]

    def delete_stocks(self):
        command = f"curl -X DELETE {base_url}/stocks"
        expected_output = ''
        self.stocks = {}
        self.total_sales = 0.0
        self.run_command(command, expected_output)

    def add_stock(self):
        # Decide whether to generate a valid or invalid test case
        if random.random() < 0.9:
            # Valid test case
            name = self.generate_stock_name()
            amount = random.randint(1, 1000)
            valid = True
        else:
            # Invalid test case
            name = self.generate_stock_name()
            amount = random.choice([-10, 0, 1.1, "abc"])
            valid = False

        data = {"name": name, "amount": amount}
        command = f"curl -d '{json.dumps(data)}' -H 'Content-Type: application/json' {base_url}/stocks"

        if valid:
            expected_output = json.dumps(data)
            # Update internal state
            self.stocks[name] = self.stocks.get(name, 0) + amount
        else:
            expected_output = '{"message":"ERROR"}'

        self.run_command(command, expected_output)

    def make_sale(self):
        if not self.stocks:
            # No stocks to sell, skip
            return

        # Decide whether to generate a valid or invalid test case
        if random.random() < 0.9:
            # Valid test case
            name = random.choice(list(self.stocks.keys()))
            max_amount = self.stocks[name]
            if max_amount == 0:
                return
            amount = random.randint(1, max_amount)
            price = round(random.uniform(0.01, 10000.0), 6)
            valid = True
        else:
            # Invalid test case
            invalid_type = random.choice(
                ['nonexistent_stock', 'excess_amount', 'invalid_amount', 'invalid_price'])
            if invalid_type == 'nonexistent_stock':
                name = 'nonexistent_' + self.generate_stock_name()
                amount = random.randint(1, 10)
                price = round(random.uniform(1.0, 100.0), 2)
            elif invalid_type == 'excess_amount':
                name = random.choice(list(self.stocks.keys()))
                amount = self.stocks[name] + random.randint(1, 100)
                price = round(random.uniform(1.0, 100.0), 2)
            elif invalid_type == 'invalid_amount':
                name = random.choice(list(self.stocks.keys()))
                amount = random.choice([-5, 0, 1.5, "abc"])
                price = round(random.uniform(1.0, 100.0), 2)
            elif invalid_type == 'invalid_price':
                name = random.choice(list(self.stocks.keys()))
                max_amount = self.stocks[name]
                amount = random.randint(1, max_amount)
                price = random.choice([-10, 1.123456789, "abc"])
            valid = False

        data = {"name": name, "amount": amount, "price": price}  # type: ignore
        command = f"curl -d '{json.dumps(data)}' -H 'Content-Type: application/json' {base_url}/sales"

        # Determine expected output
        if valid:
            expected_output = json.dumps(data)
            # Update internal state
            self.stocks[name] -= amount  # type: ignore
            self.total_sales += amount * price  # type: ignore
            self.total_sales = round(self.total_sales, 1)
        else:
            expected_output = '{"message":"ERROR"}'

        self.run_command(command, expected_output)

    def get_stocks(self):
        command = f"curl {base_url}/stocks"
        expected_output = json.dumps(self.stocks)
        self.run_command(command, expected_output)

    def get_sales(self):
        command = f"curl {base_url}/sales"
        expected_output = json.dumps({"sales": self.total_sales})
        self.run_command(command, expected_output)

    def generate_stock_name(self):
        # Generate a random stock name, could be existing or new
        existing_names = list(self.stocks.keys())
        if random.random() < 0.5 and existing_names:
            return random.choice(existing_names)
        else:
            name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=3))
            # Randomly change case
            if random.random() < 0.5:
                name = name.upper()
            return name

    def run_command(self, command, expected_output):
        process = subprocess.run(
            command, shell=True, capture_output=True, text=True)
        actual_output = process.stdout.strip()

        # Print command and outputs
        print(f"Command: {command}")
        print(f"Expected: {expected_output}")
        print(f"Actual: {actual_output}")

        if self.compare_outputs(actual_output, expected_output):
            print("Result: ✅ Match")
        else:
            print("Result: ❌ Mismatch")

        if process.stderr:
            print(f"Error: {process.stderr}")
        print("=" * 50)

    def compare_outputs(self, actual, expected):
        try:
            actual_json = json.loads(actual)
            expected_json = json.loads(expected)
            return self.compare_json(actual_json, expected_json)
        except json.JSONDecodeError:
            # Not JSON, compare as strings
            return actual == expected

    def compare_json(self, actual_json, expected_json):
        if isinstance(actual_json, dict) and isinstance(expected_json, dict):
            if set(actual_json.keys()) != set(expected_json.keys()):
                return False
            for key in actual_json:
                if not self.compare_json(actual_json[key], expected_json[key]):
                    return False
            return True
        elif isinstance(actual_json, list) and isinstance(expected_json, list):
            if len(actual_json) != len(expected_json):
                return False
            for a, e in zip(actual_json, expected_json):
                if not self.compare_json(a, e):
                    return False
            return True
        elif isinstance(actual_json, float) or isinstance(expected_json, float):
            # Compare floating point numbers with a tolerance

            return abs(
                float(actual_json) - float(expected_json)  # type: ignore
            ) < 1e-6
        else:
            return actual_json == expected_json


if __name__ == "__main__":
    runner = TestRunner()
    runner.run(num_tests=200)  # Adjust the number of tests as needed
