<?php

namespace workspace;

class Instance
{
    public $rootDir = "";
    function __construct()
    {
        $path = getcwd();
        while ($path !== '/' && $path !== '') {
            if (file_exists($path . DIRECTORY_SEPARATOR . 'workspace.php')) {
                $this->rootDir = $path;
                return;
            }
            $path = dirname($path);
        }
        throw new \Exception("Workspace root (containing workspace.php) not found.");
    }

    function locate($path, $allowNonExistent = false): string
    {
        // 絶対パスの場合
        if (strpos($path, '/') === 0 || (strlen($path) > 1 && $path[1] === ':')) {
            $absolutePath = $allowNonExistent ? $path : realpath($path);
            if ($absolutePath === false) {
                throw new \Exception("The specified absolute path does not exist or is not accessible: $path");
            }
            if (strpos($absolutePath, $this->rootDir) === 0) {
                return substr($absolutePath, strlen($this->rootDir) + 1);
            }
            throw new \Exception("The specified path is outside the workspace: $path");
        }

        // 相対パスの場合
        $absolutePath = $allowNonExistent ? $this->rootDir . DIRECTORY_SEPARATOR . $path : realpath($this->rootDir . DIRECTORY_SEPARATOR . $path);
        if ($absolutePath === false) {
            throw new \Exception("The specified relative path does not exist or is not accessible: $path");
        }
        if (strpos($absolutePath, $this->rootDir) === 0) {
            return substr($absolutePath, strlen($this->rootDir) + 1);
        }
        throw new \Exception("The specified path is outside the workspace: $path");
    }
}
