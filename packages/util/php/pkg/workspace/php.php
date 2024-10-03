<?php

namespace util\workspace;

class PHP implements Workspace
{
    public $workFile = 'workspace.php';
    public $root = "";
    public $use = [];

    function __construct()
    {
        $path = getcwd();
        while ($path !== '/' && $path !== '') {
            if (file_exists($path . '/' . $this->workFile)) {
                $this->root = $path;
                require $path . '/' . $this->workFile;
                $this->use = $use;
                return;
            }
            $path = dirname($path);
        }
        throw new \Exception("Workspace root (containing workspace.php) not found.");
    }

    function root(): string
    {
        return $this->root;
    }

    function packages(): array
    {
        return $this->use;
    }

    function locate(string $path, $allowNonExistent = false): string
    {
        $absolutePath = $this->isAbsolutePath($path) ? $path : getcwd() . '/' . $path;
        if (!$allowNonExistent) {
            $absolutePath = realpath($absolutePath);
            if ($absolutePath === false) {
                throw new \Exception("The specified path does not exist or is not accessible: $path");
            }
        } else {
            $absolutePath = $this->normalizePath($absolutePath);
        }

        $rootPath = realpath($this->root);
        if (strpos($absolutePath, $rootPath) !== 0) {
            throw new \Exception("The specified path is outside the workspace: $path");
        }

        return substr($absolutePath, strlen($rootPath) + 1);
    }

    private function isAbsolutePath($path)
    {
        return strpos($path, '/') === 0;
    }

    private function normalizePath($path)
    {
        $parts = [];
        $segments = explode('/', $path);
        foreach ($segments as $segment) {
            if ($segment === '' || $segment === '.') {
                continue;
            } elseif ($segment === '..') {
                array_pop($parts);
            } else {
                $parts[] = $segment;
            }
        }
        $normalizedPath = implode('/', $parts);
        return $this->isAbsolutePath($path) ? "/$normalizedPath" : $normalizedPath;
    }
}
