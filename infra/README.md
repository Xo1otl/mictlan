# infra

infrastructure as code

基本pythonで書くこととする

## Memo

- サービスによっては複数のドメインで使われる場合がある
- その場合はドメイン固有の設定をドメインのフォルダに書いて、サービス自体のフォルダを別で用意してDIContainerのようにして集約する
- メインのdocker composeではその集約をincludeする
- pythonパッケージとして登録されているのでimportが使える

# Handling Circular Dependencies in the Infrastructure Codebase

This codebase is designed to manage various infrastructure components of our project, such as VPN, search engine, relational database, etc. Each component is represented as a module within the `infra/infra/` directory.

Due to the interconnected nature of these components, circular dependencies can easily occur, especially when components need to reference configurations from one another. Circular dependencies can lead to import errors and make the codebase difficult to maintain.

To resolve this issue, we've implemented a specific module structure and import strategy. This README explains the mechanism we've adopted to handle circular dependencies in an easy-to-understand way.

## The Problem: Circular Dependencies

In Python, a circular dependency occurs when two or more modules depend on each other directly or indirectly. For example:

- Module A imports Module B.
- Module B imports Module A.

This can cause Python's import system to get stuck in a loop, leading to errors and preventing the application from running correctly.

In our infrastructure code, components often need to reference configurations (like hostnames, ports, or environment variables) from other components, making circular dependencies a potential problem.

## Our Solution: Lazy Imports and Import Restrictions

To prevent circular dependencies, we've established the following rules and structures:

1. **Lazy Importing of `docker_compose.py` in `__init__.py`**

    - Each module's `__init__.py` contains a function `docker_compose()` that lazily imports and returns the `docker_compose` dictionary from its own `docker_compose.py`.
    - This means that the actual import of `docker_compose.py` happens only when `docker_compose()` is called, not when the module is first imported.
    - Example:
        ```python
        # __init__.py
        def docker_compose():
            from .docker_compose import docker_compose
            return docker_compose
        ```

2. **Direct Imports in `__init__.py` and `env.py`**

    - The `__init__.py` and `env.py` files can directly import other modules or their `env.py` files as needed.
    - However, they **must not** import or reference other modules' `docker_compose` functions or variables.
    - This allows sharing of configuration variables without creating circular dependencies.
    - Example:
        ```python
        # __init__.py
        from .env import *
        ```

3. **No Cross-Module `docker_compose()` Calls in `docker_compose.py`**

    - Within a module's `docker_compose.py`, you can import other modules and use their variables or functions.
    - **Do not** import or call `docker_compose()` functions from other modules in `docker_compose.py`.
    - This prevents the creation of circular dependencies when assembling the `docker_compose` configurations.
    - Example:
        ```python
        # docker_compose.py
        from infra import vpn  # Allowed
        # from infra.vpn import docker_compose  # Not allowed
        ```

4. **Free Imports in `docker_compose.py`**

    - You are free to import other modules and use their variables (like hostnames, ports) in `docker_compose.py`.
    - Just avoid importing their `docker_compose()` functions or referencing their `docker_compose` variables.
    - This provides flexibility in configuration while avoiding circular imports.

5. **Careful Imports in `__init__.py` and `env.py`**

    - In `__init__.py` and `env.py`, be cautious with imports to avoid circular dependencies.
    - Avoid importing modules or variables that depend on `docker_compose.py` from other modules.

## How It Works: An Example

Let's walk through an example to illustrate how this mechanism works.

### Module Structure

Assume we have two modules: `vpn` and `proxy`.

#### `vpn` Module

- **`vpn/__init__.py`**
    ```python
    from .env import *

    def docker_compose():
        from .docker_compose import docker_compose
        return docker_compose
    ```

- **`vpn/env.py`**
    ```python
    HOSTNAME = 'vpn.example.com'
    ```

- **`vpn/docker_compose.py`**
    ```python
    from . import HOSTNAME

    docker_compose = {
        'services': {
            'vpn-service': {
                'image': 'vpn-image',
                'hostname': HOSTNAME,
                # Other configurations...
            }
        }
    }
    ```

#### `proxy` Module

- **`proxy/__init__.py`**
    ```python
    from .env import *

    def docker_compose():
        from .docker_compose import docker_compose
        return docker_compose
    ```

- **`proxy/env.py`**
    ```python
    CERTBOT_EMAIL = 'admin@example.com'
    ```

- **`proxy/docker_compose.py`**
    ```python
    from infra import vpn  # Allowed
    # from infra.vpn import docker_compose  # Not allowed

    docker_compose = {
        'services': {
            'proxy-service': {
                'image': 'proxy-image',
                'environment': {
                    'VPN_HOSTNAME': vpn.HOSTNAME,
                    'CERTBOT_EMAIL': CERTBOT_EMAIL,
                    # Other configurations...
                }
            }
        }
    }
    ```

### Explanation

- **Lazy Importing in `__init__.py`**
    - Both modules' `__init__.py` define a `docker_compose()` function that lazily imports their own `docker_compose.py`.
    - This ensures that `docker_compose.py` is not imported during the initial module import, preventing circular dependencies at that point.

- **Avoiding Circular Dependencies in `env.py`**
    - The `env.py` files import necessary variables but do not import `docker_compose` from any module.
    - This allows sharing of configuration variables without creating circular dependencies.

- **Controlled Imports in `docker_compose.py`**
    - In `proxy/docker_compose.py`, we import `vpn` to access `vpn.HOSTNAME`.
    - We do not import or call `vpn.docker_compose()`.
    - This allows us to reference variables from other modules without causing circular imports.

- **No Cross-Module `docker_compose()` Calls**
    - Neither `docker_compose.py` imports or calls `docker_compose()` from other modules.
    - This is crucial to prevent circular dependencies when building the `docker_compose` configurations.

## Guidelines for Developers

To maintain this structure and prevent circular dependencies, follow these guidelines:

1. **When Writing `docker_compose.py`:**

    - **Allowed:**
        - Import other modules (e.g., `from infra import vpn`) to access shared variables.
        - Use variables and functions from other modules (excluding `docker_compose`).
    - **Not Allowed:**
        - Import or call `docker_compose()` functions from other modules.
        - Reference `docker_compose` variables from other modules.

2. **When Writing `__init__.py` and `env.py`:**

    - **Allowed:**
        - Import variables from other modules as needed.
    - **Not Allowed:**
        - Import or reference `docker_compose` functions or variables from other modules.
        - Create imports that could lead to circular dependencies.

3. **General Best Practices:**

    - Keep the import statements in `__init__.py` and `env.py` minimal and straightforward.
    - Use lazy imports (inside functions) when importing modules that could potentially cause circular dependencies.
    - Be mindful of the module hierarchy and dependencies when adding new imports.

## Benefits of This Approach

- **Avoids Circular Dependencies:**
    - By restricting how and where imports are made, we prevent circular dependencies that can cause runtime errors.

- **Modularity:**
    - Each component/module remains self-contained, making the codebase easier to maintain and understand.

- **Flexibility:**
    - Components can still share necessary configuration variables without importing entire configurations that might introduce circular dependencies.

- **Lazy Loading:**
    - Delaying the import of `docker_compose.py` until it's actually needed optimizes performance and prevents unnecessary loading of modules.

## Conclusion

By following these guidelines and understanding the import mechanism, we can build a complex infrastructure where components interact seamlessly without running into circular import issues. This approach ensures a clean, maintainable, and efficient codebase.

---

**Remember:** The key to avoiding circular dependencies in this codebase is to control when and how modules are imported, especially concerning the `docker_compose` configurations. Stick to the rules outlined above, and you'll help keep the codebase robust and error-free.