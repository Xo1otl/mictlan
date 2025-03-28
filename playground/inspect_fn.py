import inspect

def sample_function():
    # これはコメントです
    print("Hello, World!")

source = inspect.getsource(sample_function)
print(source)

